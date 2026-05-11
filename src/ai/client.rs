//! fal.ai + Hunyuan3D V3 provider.
//!
//! Implements `AiProvider` against fal.ai's serverless **queue API**
//! (the sync endpoint `https://fal.run/...` would 504 on most 3D-gen
//! jobs). Pipeline:
//!
//! 1. **Submit** — POST `https://queue.fal.run/fal-ai/hunyuan3d-v3/text-to-3d`
//!    with `Authorization: Key <api_key>` and `{"prompt": "..."}`,
//!    receive `request_id` + `status_url` + `response_url`.
//! 2. **Poll** — GET the status URL every 2 s until status is
//!    `COMPLETED` (translate `IN_QUEUE` / `IN_PROGRESS` to indeterminate
//!    progress events for the UI).
//! 3. **Fetch result** — GET the response URL, parse out
//!    `model_mesh.url`.
//! 4. **Download GLB** — GET that url, return bytes.
//!
//! The whole pipeline runs as a single async task on `App::ai_runtime`;
//! cancellation checkpoints sit between each stage so the worst-case
//! cancel latency is one poll interval (≈ 2 s).
//!
//! API keys come from the OS keychain at submit time (so a user
//! who clicks Save in the panel doesn't need to restart). The key
//! never appears in error messages or logs — only the failing HTTP
//! status / response body does.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use reqwest::Client;
use serde::Deserialize;
use tokio::time::sleep;

use super::job::JobEvent;
use super::keyring_store;
use super::provider::{AiProvider, AiRequest};
use super::voxelize::voxelize_glb;

/// fal.ai queue API base. The provider formats this with the model id
/// + endpoint at submit time so future providers (different models on
/// fal.ai) can share most of the polling code.
const TEXT_TO_3D_ENDPOINT: &str =
    "https://queue.fal.run/fal-ai/hunyuan3d-v3/text-to-3d";

/// How often to poll the status URL. fal.ai's queue updates ~ every
/// few seconds; faster polling just adds noise without speeding the
/// real bottleneck (the GPU job).
const POLL_INTERVAL: Duration = Duration::from_millis(2000);

/// Hard cap on poll attempts. 150 × 2 s ≈ 5 minutes. Hunyuan3D V3
/// usually finishes in 10–30 s; this only fires when fal.ai is
/// degraded or the queue is unusually long. Worker emits Failed with
/// "Timeout" rather than wedge forever.
const MAX_POLL_ATTEMPTS: u32 = 150;

/// Built-in fal.ai provider. Stateless except for a connection-pooled
/// reqwest client; loads the API key from the OS keychain on each
/// `submit`.
pub struct FalHunyuanProvider {
    http: Client,
}

impl FalHunyuanProvider {
    pub fn new() -> Self {
        // Generous total timeout (5 min) for the GLB download — the
        // file can be a few MB and fal.ai's CDN can be slow on
        // first-fetch. Connect timeout is short so we fail fast on
        // network outage instead of waiting for the full window.
        let http = Client::builder()
            .connect_timeout(Duration::from_secs(15))
            .timeout(Duration::from_secs(300))
            .user_agent(concat!("voxelith/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("Failed to build HTTP client");
        Self { http }
    }
}

impl Default for FalHunyuanProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl AiProvider for FalHunyuanProvider {
    fn name(&self) -> &str {
        "fal.ai · Hunyuan3D V3"
    }

    fn submit(
        &self,
        request: AiRequest,
        runtime: &tokio::runtime::Handle,
        events_tx: mpsc::Sender<JobEvent>,
        cancel: Arc<AtomicBool>,
    ) {
        let http = self.http.clone();
        runtime.spawn(async move {
            // Wrap the whole pipeline in `?`-able anyhow so any stage
            // failure fans out to a single Failed event with a clean
            // message. The cancel cooperative-flag still produces
            // Failed { "Cancelled" } via `bail!`.
            if let Err(e) = run_pipeline(&http, request, &cancel, &events_tx).await {
                let _ = events_tx.send(JobEvent::Failed {
                    message: e.to_string(),
                });
            }
        });
    }
}

async fn run_pipeline(
    http: &Client,
    request: AiRequest,
    cancel: &AtomicBool,
    events_tx: &mpsc::Sender<JobEvent>,
) -> Result<()> {
    // Phase 2 only handles text-to-3D. Image-to-3D will land in
    // Phase 4 with the upload UI; until then we explicitly bail so
    // the user gets a clear message instead of a confusing 422.
    if request.image.is_some() {
        bail!("Image input is Phase 4 — text-to-3D only for now");
    }
    if request.prompt.trim().is_empty() {
        bail!("Prompt is empty");
    }

    let api_key = keyring_store::load_api_key("fal_ai")
        .context("Loading API key from OS keychain")?;

    check_cancel(cancel)?;
    let queue = fal_submit(http, &api_key, &request.prompt).await?;
    let _ = events_tx.send(JobEvent::Submitted);

    fal_poll_until_done(http, &api_key, &queue.status_url, cancel, events_tx).await?;

    check_cancel(cancel)?;
    let result = fal_fetch_result(http, &api_key, &queue.response_url).await?;

    check_cancel(cancel)?;
    let glb_bytes = fal_download_glb(http, &result.model_mesh.url).await?;
    let byte_count = glb_bytes.len();
    let _ = events_tx.send(JobEvent::GlbReady { byte_count });

    // Voxelize on a blocking thread — it's CPU-bound (~hundreds of
    // ms at 64³, a few seconds at 128³) and would stall other tokio
    // tasks if we ran it directly on the worker thread.
    //
    // We don't thread cancellation into the voxelizer; it's short
    // enough that a Cancel click after this point will be observed
    // by the next stage's checkpoint (post-await below) and the
    // voxelize result will simply be discarded.
    let resolution = request.resolution;
    let patch = tokio::task::spawn_blocking(move || voxelize_glb(&glb_bytes, resolution))
        .await
        .context("Voxelize task panicked")??;

    check_cancel(cancel)?;

    let voxel_count = patch.len();
    let _ = events_tx.send(JobEvent::Done {
        summary: format!(
            "{} voxels from {} KB GLB ({})",
            voxel_count,
            byte_count.div_ceil(1024),
            request.prompt.chars().take(40).collect::<String>(),
        ),
        patch: Some(patch),
    });
    Ok(())
}

#[inline]
fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        bail!("Cancelled");
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct QueueSubmitResponse {
    #[allow(dead_code)] // Useful for diagnostics; kept for future logging.
    request_id: String,
    status_url: String,
    response_url: String,
}

#[derive(Debug, Deserialize)]
struct QueueStatusResponse {
    status: String, // IN_QUEUE | IN_PROGRESS | COMPLETED
    #[serde(default)]
    queue_position: Option<u32>,
}

/// Hunyuan3D V3 result envelope. The model returns a single GLB mesh;
/// we ignore the other fields (seed, intermediate previews) for now.
#[derive(Debug, Deserialize)]
struct HunyuanResult {
    model_mesh: ModelFile,
}

#[derive(Debug, Deserialize)]
struct ModelFile {
    url: String,
    #[allow(dead_code)]
    #[serde(default)]
    file_size: Option<u64>,
}

async fn fal_submit(
    http: &Client,
    api_key: &str,
    prompt: &str,
) -> Result<QueueSubmitResponse> {
    let body = serde_json::json!({ "prompt": prompt });
    let resp = http
        .post(TEXT_TO_3D_ENDPOINT)
        .header("Authorization", format!("Key {}", api_key))
        .json(&body)
        .send()
        .await
        .context("HTTP submit")?;
    let status = resp.status();
    if !status.is_success() {
        let body_text = resp.text().await.unwrap_or_default();
        bail!("Submit {}: {}", status.as_u16(), short(&body_text, 200));
    }
    resp.json::<QueueSubmitResponse>()
        .await
        .context("Parsing submit response")
}

async fn fal_poll_until_done(
    http: &Client,
    api_key: &str,
    status_url: &str,
    cancel: &AtomicBool,
    events_tx: &mpsc::Sender<JobEvent>,
) -> Result<()> {
    for attempt in 0..MAX_POLL_ATTEMPTS {
        check_cancel(cancel)?;
        sleep(POLL_INTERVAL).await;

        let resp = match http
            .get(status_url)
            .header("Authorization", format!("Key {}", api_key))
            .send()
            .await
        {
            Ok(r) => r,
            // Transient network errors during polling are common
            // (proxy hiccup, etc). Don't fail the whole job — wait
            // for the next poll. If cancel hits or we exceed the
            // attempt cap, the surrounding loop handles it.
            Err(_) => continue,
        };

        if !resp.status().is_success() {
            // Same logic for HTTP errors: usually a 502 from an
            // overloaded gateway. Skip and retry.
            continue;
        }

        let status = match resp.json::<QueueStatusResponse>().await {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Translate fal.ai status into a UI progress estimate. Without
        // real percent reporting we just give the user "queued" /
        // "running" / "almost done" steps.
        let progress = match status.status.as_str() {
            "IN_QUEUE" => 0.1,
            "IN_PROGRESS" => 0.5,
            "COMPLETED" => 0.9,
            _ => 0.3,
        };
        let _ = events_tx.send(JobEvent::Progress(progress));

        match status.status.as_str() {
            "COMPLETED" => return Ok(()),
            "FAILED" | "ERROR" => bail!(
                "Provider job failed (after {} polls, queue_position={:?})",
                attempt + 1,
                status.queue_position
            ),
            _ => {} // IN_QUEUE / IN_PROGRESS — keep polling
        }
    }
    Err(anyhow!("Provider didn't finish within {} polls", MAX_POLL_ATTEMPTS))
}

async fn fal_fetch_result(
    http: &Client,
    api_key: &str,
    response_url: &str,
) -> Result<HunyuanResult> {
    let resp = http
        .get(response_url)
        .header("Authorization", format!("Key {}", api_key))
        .send()
        .await
        .context("HTTP fetch result")?;
    let status = resp.status();
    if !status.is_success() {
        let body_text = resp.text().await.unwrap_or_default();
        bail!(
            "Fetch result {}: {}",
            status.as_u16(),
            short(&body_text, 200)
        );
    }
    resp.json::<HunyuanResult>()
        .await
        .context("Parsing result JSON")
}

async fn fal_download_glb(http: &Client, url: &str) -> Result<Vec<u8>> {
    // GLB downloads use the fal.ai CDN host (e.g. v3.fal.media). No
    // auth needed for these URLs; they're pre-signed and short-lived.
    let resp = http.get(url).send().await.context("HTTP download GLB")?;
    let status = resp.status();
    if !status.is_success() {
        bail!("Download {}", status.as_u16());
    }
    let bytes = resp.bytes().await.context("Reading GLB body")?;
    Ok(bytes.to_vec())
}

/// Truncate `s` to `max` chars, appending an ellipsis when truncated.
/// Used to keep error messages from exploding when fal.ai returns a
/// long HTML 5xx page.
fn short(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let head: String = s.chars().take(max).collect();
        format!("{}…", head)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_passes_through_when_under_limit() {
        assert_eq!(short("hello", 10), "hello");
    }

    #[test]
    fn short_truncates_with_ellipsis_at_limit() {
        assert_eq!(short("abcdefghij", 5), "abcde…");
    }

    #[test]
    fn short_handles_unicode_correctly() {
        // 5 char-codepoints, not 5 bytes — naive byte slicing would
        // cut a multi-byte char in half and panic.
        let s = "héllo wörld";
        let out = short(s, 5);
        assert!(out.starts_with("héllo"));
        assert!(out.ends_with('…'));
    }
}
