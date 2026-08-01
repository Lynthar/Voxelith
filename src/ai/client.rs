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
//! 3. **Fetch result** — GET the response URL, parse out the GLB URL
//!    (`model_glb.url`, falling back to `model_urls.glb.url`).
//! 4. **Download GLB** — GET that url, return bytes.
//!
//! The whole pipeline runs as a single async task on `App::ai_runtime`.
//! Cancellation is cooperative. During the poll phase — where the
//! remote GPU job actually runs and bills — a Cancel is observed at the
//! top of the next poll iteration (≈ 2 s typically, and at most one
//! status-GET timeout — `STATUS_POLL_TIMEOUT`, ~30 s — if a poll happens
//! to be in flight) and also fires a best-effort `PUT` to the queue
//! `cancel_url`, so fal.ai drops/stops the job instead of finishing it
//! for nothing. The other three network stages (submit, result fetch,
//! GLB download) run inside `cancellable`, which races them against the
//! cancel flag and abandons the request within
//! `CANCEL_POLL_INTERVAL`. Only the voxelize step is uninterruptible,
//! and by then there's no remote cost left to save.
//!
//! API keys come from the OS keychain at submit time (so a user
//! who clicks Save in the panel doesn't need to restart). The key
//! never appears in error messages or logs — only the failing HTTP
//! status / response body does.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

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

/// Wall-clock cap on the whole poll phase, measured with `Instant`
/// rather than an attempt count: `attempts × per-request-timeout` let a
/// hung or throttled gateway stretch the job far past its nominal window
/// (a black-holed GET could otherwise hang for the client's full global
/// timeout each iteration). Hunyuan3D V3 usually finishes in 10–30 s;
/// this only fires when fal.ai is degraded. Worker emits Failed rather
/// than wedge forever.
const POLL_TIMEOUT: Duration = Duration::from_secs(300);

/// Per-request timeout for a single status GET — far below the client's
/// generous global timeout (which is sized for the multi-MB GLB
/// download). Bounds each poll so `POLL_TIMEOUT` is actually honored.
const STATUS_POLL_TIMEOUT: Duration = Duration::from_secs(30);

/// Per-request timeout for the small JSON calls (submit, result fetch).
/// Both are a few kilobytes; without their own timeout they inherited
/// the client's 300 s window, which is sized for the GLB download — so
/// a hung gateway on submit could sit there for five minutes.
const JSON_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Ceiling on a downloaded GLB. The 300 s client timeout limits how
/// *long* a download runs but not how *large* it gets, and the body was
/// buffered whole — a fast link could push tens of gigabytes into
/// memory. Real Hunyuan3D output is ~60 MiB, so this is 4× headroom.
const MAX_GLB_BYTES: u64 = 256 * 1024 * 1024;

/// Ceiling on the queue API's JSON bodies. They're a few KB.
const MAX_JSON_BYTES: u64 = 4 * 1024 * 1024;

/// Cap on the initial `Vec` reservation when a response declares its
/// length. The declared value is remote input, so it seeds the capacity
/// hint only up to this — the read still grows as needed.
const ALLOC_HINT_CAP: u64 = 8 * 1024 * 1024;

/// How many consecutive unrecognized queue statuses to tolerate before
/// giving up. fal.ai documents IN_QUEUE / IN_PROGRESS / COMPLETED; an
/// unknown value used to just keep polling until the 300 s cap, and
/// then report a misleading timeout.
const MAX_UNKNOWN_STATUS_STREAK: u32 = 5;

/// How often the cancel flag is sampled while a request is in flight.
const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(150);

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
                    // `{:#}` walks anyhow's full context chain ("… : … :
                    // root cause") instead of only the outermost message,
                    // so a buried network/parse cause isn't lost.
                    message: format!("{:#}", e),
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
    // Race every network stage against the cancel flag. A one-shot
    // check before the await only helps if the request returns: submit
    // and the result fetch each have their own timeout, and without
    // this a Cancel during one of them wasn't observed until it
    // finished.
    let queue = cancellable(cancel, fal_submit(http, &api_key, &request.prompt)).await?;
    let _ = events_tx.send(JobEvent::Submitted);

    fal_poll_until_done(
        http,
        &api_key,
        &queue.status_url,
        queue.cancel_url.as_deref(),
        cancel,
        events_tx,
    )
    .await?;

    check_cancel(cancel)?;
    let glb =
        cancellable(cancel, fal_fetch_result(http, &api_key, &queue.response_url))
            .await?;

    check_cancel(cancel)?;
    let glb_bytes = cancellable(
        cancel,
        fal_download_glb(http, &glb.url, glb.file_size),
    )
    .await?;
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

/// Resolve as soon as `cancel` is set. Cancellation is a plain
/// `AtomicBool` (shared with the UI thread, which just flips it), so
/// there's nothing to await on directly — poll it instead.
async fn until_cancelled(cancel: &AtomicBool) {
    while !cancel.load(Ordering::Acquire) {
        sleep(CANCEL_POLL_INTERVAL).await;
    }
}

/// Run `work`, abandoning it the moment the user cancels.
///
/// Dropping an in-flight reqwest future aborts the request. The remote
/// side may still have received it — for submit that means fal.ai could
/// start a job we never learn the id of, and so can't cancel remotely.
/// That window is one request wide and the alternative is making the
/// user wait out a hung gateway, so we take it.
async fn cancellable<T>(
    cancel: &AtomicBool,
    work: impl std::future::Future<Output = Result<T>>,
) -> Result<T> {
    tokio::select! {
        biased;
        _ = until_cancelled(cancel) => bail!("Cancelled"),
        result = work => result,
    }
}

/// Read a response body with a hard ceiling.
///
/// A declared `Content-Length` over the cap fails before a single byte
/// is read, but it's remote input and may be absent (chunked) or a lie,
/// so the running total is what actually enforces the limit.
async fn read_capped(resp: reqwest::Response, cap: u64, what: &str) -> Result<Vec<u8>> {
    let declared = resp.content_length();
    if let Some(len) = declared {
        if len > cap {
            bail!("{} is {} bytes, over the {} byte limit", what, len, cap);
        }
    }
    let hint = declared.unwrap_or(0).min(ALLOC_HINT_CAP) as usize;
    let mut buf = Vec::with_capacity(hint);
    let mut resp = resp;
    while let Some(chunk) = resp
        .chunk()
        .await
        .with_context(|| format!("Reading {}", what))?
    {
        if buf.len() as u64 + chunk.len() as u64 > cap {
            bail!("{} exceeds the {} byte limit", what, cap);
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(buf)
}

/// Whether a non-2xx status while polling is worth retrying.
enum PollAction {
    Retry,
    /// Give up now, with this hint appended to the error.
    Fail(&'static str),
}

/// Retrying a 401 for five minutes and then reporting a timeout tells
/// the user nothing. Only statuses that can plausibly change on their
/// own are worth waiting on.
fn poll_disposition(code: u16) -> PollAction {
    match code {
        // The server is explicitly asking us to come back later, which
        // is what the loop already does.
        408 | 429 => PollAction::Retry,
        401 | 403 => PollAction::Fail("check your fal.ai API key in the AI panel"),
        404 => PollAction::Fail("the queued request is gone (expired or already collected)"),
        400..=499 => PollAction::Fail("the provider rejected the status request"),
        // 5xx and anything else: usually an overloaded gateway.
        _ => PollAction::Retry,
    }
}

/// Reject a GLB URL we shouldn't follow.
///
/// https is mandatory: over plain http an on-path attacker could swap
/// in any payload, and the GLB feeds a parser. The host is only warned
/// about — fal has already moved this CDN across subdomains
/// (`fal.media` → `v3.fal.media` → `v3b.fal.media`), and hard-failing
/// on the next move would break generation until we shipped a release.
/// The URL itself arrives inside an authenticated https response, so a
/// forged host means fal's API is already compromised.
fn validate_glb_url(url: &str) -> Result<()> {
    let parsed = reqwest::Url::parse(url).context("Parsing GLB URL")?;
    if parsed.scheme() != "https" {
        bail!("GLB URL is not https: {}", short(url, 120));
    }
    let host = parsed.host_str().unwrap_or_default();
    if host != "fal.media" && !host.ends_with(".fal.media") {
        log::warn!("GLB download from an unexpected host: {}", host);
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct QueueSubmitResponse {
    #[allow(dead_code)] // Useful for diagnostics; kept for future logging.
    request_id: String,
    status_url: String,
    response_url: String,
    /// Queue cancel endpoint — `PUT` here to drop a queued job or
    /// signal an in-progress runner to stop, freeing the user's quota.
    /// `Option` so a response without it still parses (remote cancel
    /// then degrades to a local-only cancel).
    #[serde(default)]
    cancel_url: Option<String>,
}

#[derive(Debug, Deserialize)]
struct QueueStatusResponse {
    status: String, // IN_QUEUE | IN_PROGRESS | COMPLETED
    #[serde(default)]
    queue_position: Option<u32>,
}

/// Hunyuan3D V3 result envelope. The textured mesh comes back in
/// `model_glb` (a File object); the same GLB is also reachable via the
/// per-format map `model_urls.glb`. Other fields (thumbnail, seed) are
/// ignored.
///
/// Both are `Option` so a minor provider-side schema change (one
/// present but not the other) still resolves a URL via `glb_url`
/// instead of hard-failing at deserialization. Verified against the
/// fal.ai docs example: top-level field is `model_glb`, NOT
/// `model_mesh` (an earlier mismatch silently broke the fetch stage).
#[derive(Debug, Deserialize)]
struct HunyuanResult {
    #[serde(default)]
    model_glb: Option<ModelFile>,
    #[serde(default)]
    model_urls: Option<ModelUrls>,
}

#[derive(Debug, Deserialize)]
struct ModelUrls {
    #[serde(default)]
    glb: Option<ModelFile>,
}

impl HunyuanResult {
    /// Resolve the GLB reference, preferring the top-level `model_glb`
    /// and falling back to `model_urls.glb`.
    fn glb_file(self) -> Option<ModelFile> {
        self.model_glb
            .or_else(|| self.model_urls.and_then(|u| u.glb))
    }
}

#[derive(Debug, Deserialize)]
struct ModelFile {
    url: String,
    /// The provider's claim about the file's size. Advisory only — it
    /// lets an oversized download be refused before it starts, but the
    /// download itself is what enforces the ceiling.
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
        .timeout(JSON_REQUEST_TIMEOUT)
        .json(&body)
        .send()
        .await
        .context("HTTP submit")?;
    let status = resp.status();
    if !status.is_success() {
        let code = status.as_u16();
        let body_text = resp.text().await.unwrap_or_default();
        // 401/403 is almost always a missing/invalid key — point the
        // user straight at the fix instead of a bare status code.
        let hint = if code == 401 || code == 403 {
            " (check your fal.ai API key in the AI panel)"
        } else {
            ""
        };
        bail!("Submit {}: {}{}", code, short(&body_text, 200), hint);
    }
    let body = read_capped(resp, MAX_JSON_BYTES, "submit response").await?;
    serde_json::from_slice(&body).context("Parsing submit response")
}

async fn fal_poll_until_done(
    http: &Client,
    api_key: &str,
    status_url: &str,
    cancel_url: Option<&str>,
    cancel: &AtomicBool,
    events_tx: &mpsc::Sender<JobEvent>,
) -> Result<()> {
    let start = Instant::now();
    let mut polls: u32 = 0;
    let mut unknown_streak: u32 = 0;
    loop {
        if cancel.load(Ordering::Acquire) {
            // User cancelled while the remote job is queued/running.
            // Tell fal.ai to stop it (best-effort) so it doesn't keep
            // burning the user's quota after we walk away.
            if let Some(url) = cancel_url {
                fal_cancel(http, api_key, url).await;
            }
            bail!("Cancelled");
        }
        // Wall-clock cap (not an attempt count): a slow/hung gateway
        // can't stretch the job past this. On timeout, best-effort tell
        // fal.ai to stop the abandoned job so it stops billing.
        if start.elapsed() >= POLL_TIMEOUT {
            if let Some(url) = cancel_url {
                fal_cancel(http, api_key, url).await;
            }
            bail!(
                "Provider didn't finish within {:?} ({} polls)",
                POLL_TIMEOUT,
                polls
            );
        }
        sleep(POLL_INTERVAL).await;
        polls += 1;

        let resp = match http
            .get(status_url)
            .header("Authorization", format!("Key {}", api_key))
            .timeout(STATUS_POLL_TIMEOUT)
            .send()
            .await
        {
            Ok(r) => r,
            // Transient network errors during polling are common (proxy
            // hiccup, or a GET that hit its short per-request timeout).
            // Don't fail the whole job — wait for the next poll; the
            // elapsed cap above bounds the total wait.
            Err(_) => continue,
        };

        let code = resp.status();
        if !code.is_success() {
            match poll_disposition(code.as_u16()) {
                PollAction::Retry => continue,
                // Retrying something that can't fix itself just burns
                // the 300 s budget and then blames a timeout.
                PollAction::Fail(hint) => {
                    bail!("Status poll {} — {}", code.as_u16(), hint)
                }
            }
        }

        let body = match read_capped(resp, MAX_JSON_BYTES, "status response").await {
            Ok(b) => b,
            Err(_) => continue,
        };
        let status: QueueStatusResponse = match serde_json::from_slice(&body) {
            Ok(s) => s,
            Err(_) => continue,
        };

        match status.status.as_str() {
            "COMPLETED" => {
                let _ = events_tx.send(JobEvent::Progress(0.9));
                return Ok(());
            }
            "FAILED" | "ERROR" => bail!(
                "Provider job failed (after {} polls, queue_position={:?})",
                polls,
                status.queue_position
            ),
            // Not in fal.ai's documented set, but a job cancelled from
            // their dashboard has to end somewhere other than our
            // timeout.
            "CANCELLED" | "CANCELED" => {
                bail!("Provider job was cancelled remotely (after {} polls)", polls)
            }
            // Translate the running states into a UI progress estimate.
            // Without real percent reporting we just give the user
            // "queued" / "running" steps.
            "IN_QUEUE" => {
                unknown_streak = 0;
                let _ = events_tx.send(JobEvent::Progress(0.1));
            }
            "IN_PROGRESS" => {
                unknown_streak = 0;
                let _ = events_tx.send(JobEvent::Progress(0.5));
            }
            other => {
                // Don't report progress for a state we don't
                // understand — a remotely-cancelled job used to flash
                // 30% on its way to a bogus timeout.
                unknown_streak += 1;
                log::warn!(
                    "fal.ai returned unrecognized status {:?} ({} in a row)",
                    short(other, 60),
                    unknown_streak
                );
                if unknown_streak >= MAX_UNKNOWN_STATUS_STREAK {
                    bail!(
                        "Provider returned unrecognized status {:?} {} times",
                        short(other, 60),
                        unknown_streak
                    );
                }
            }
        }
    }
}

async fn fal_fetch_result(
    http: &Client,
    api_key: &str,
    response_url: &str,
) -> Result<ModelFile> {
    let resp = http
        .get(response_url)
        .header("Authorization", format!("Key {}", api_key))
        .timeout(JSON_REQUEST_TIMEOUT)
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
    let body = read_capped(resp, MAX_JSON_BYTES, "result response").await?;
    let result: HunyuanResult =
        serde_json::from_slice(&body).context("Parsing result JSON")?;
    result.glb_file().ok_or_else(|| {
        anyhow!("Result JSON had no GLB URL (model_glb / model_urls.glb)")
    })
}

/// Download the GLB. `declared_size` is the provider's own claim about
/// the file, used only to refuse an oversized download before opening
/// the connection — `read_capped` is what actually enforces the limit.
async fn fal_download_glb(
    http: &Client,
    url: &str,
    declared_size: Option<u64>,
) -> Result<Vec<u8>> {
    // GLB downloads use the fal.ai CDN host (e.g. v3.fal.media). No
    // auth needed for these URLs; they're pre-signed and short-lived.
    validate_glb_url(url)?;
    if let Some(size) = declared_size {
        if size > MAX_GLB_BYTES {
            bail!(
                "Provider reports a {} byte GLB, over the {} byte limit",
                size,
                MAX_GLB_BYTES
            );
        }
    }
    let resp = http.get(url).send().await.context("HTTP download GLB")?;
    let status = resp.status();
    if !status.is_success() {
        bail!("Download {}", status.as_u16());
    }
    read_capped(resp, MAX_GLB_BYTES, "GLB").await
}

/// Best-effort remote cancel: `PUT` the queue cancel URL so fal.ai drops
/// a queued job or signals an in-progress runner to stop. Errors are
/// logged and swallowed — by the time we call this the local job is
/// already being torn down, and a failed cancel (e.g. the job just
/// completed: `400 ALREADY_COMPLETED`) shouldn't surface as a job error.
async fn fal_cancel(http: &Client, api_key: &str, cancel_url: &str) {
    match http
        .put(cancel_url)
        .header("Authorization", format!("Key {}", api_key))
        .send()
        .await
    {
        Ok(resp) => log::info!("fal.ai cancel requested -> {}", resp.status().as_u16()),
        Err(e) => log::warn!("fal.ai remote cancel failed: {}", e),
    }
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

    #[test]
    fn result_resolves_top_level_model_glb() {
        // Mirrors the documented fal.ai Hunyuan3D V3 output: the GLB
        // lives in `model_glb` (NOT `model_mesh`). Extra fields
        // (thumbnail / seed / file metadata) must be ignored.
        let json = r#"{
            "model_glb": {
                "file_size": 64724836,
                "file_name": "model.glb",
                "content_type": "model/gltf-binary",
                "url": "https://v3b.fal.media/files/b/abc/model.glb"
            },
            "thumbnail": { "url": "https://v3b.fal.media/files/b/abc/preview.png" },
            "model_urls": {
                "glb": { "url": "https://v3b.fal.media/files/b/abc/model.glb" },
                "obj": { "url": "https://v3b.fal.media/files/b/abc/model.obj" }
            },
            "seed": 42
        }"#;
        let result: HunyuanResult = serde_json::from_str(json).unwrap();
        let file = result.glb_file().expect("model_glb resolves");
        assert_eq!(file.url, "https://v3b.fal.media/files/b/abc/model.glb");
        // The declared size feeds the pre-download size check.
        assert_eq!(file.file_size, Some(64_724_836));
    }

    #[test]
    fn result_falls_back_to_model_urls_glb() {
        // If only the per-format map is present, resolve via
        // model_urls.glb rather than failing.
        let json = r#"{
            "model_urls": {
                "glb": { "url": "https://cdn/alt.glb" },
                "obj": { "url": "https://cdn/alt.obj" }
            }
        }"#;
        let result: HunyuanResult = serde_json::from_str(json).unwrap();
        assert_eq!(
            result.glb_file().map(|f| f.url).as_deref(),
            Some("https://cdn/alt.glb")
        );
    }

    #[test]
    fn result_without_any_glb_yields_none() {
        // No GLB anywhere -> None, so the caller surfaces a clean
        // "no GLB URL" error instead of downloading garbage.
        let json = r#"{ "thumbnail": { "url": "https://cdn/preview.png" }, "seed": 7 }"#;
        let result: HunyuanResult = serde_json::from_str(json).unwrap();
        assert!(result.glb_file().is_none());
    }

    #[test]
    fn poll_disposition_retries_only_what_can_change() {
        // 4xx (except the "come back later" pair) can't fix itself —
        // retrying just burns the 300 s budget and then reports a
        // misleading timeout instead of "bad key".
        assert!(matches!(poll_disposition(401), PollAction::Fail(_)));
        assert!(matches!(poll_disposition(403), PollAction::Fail(_)));
        assert!(matches!(poll_disposition(404), PollAction::Fail(_)));
        assert!(matches!(poll_disposition(422), PollAction::Fail(_)));
        assert!(matches!(poll_disposition(408), PollAction::Retry));
        assert!(matches!(poll_disposition(429), PollAction::Retry));
        assert!(matches!(poll_disposition(500), PollAction::Retry));
        assert!(matches!(poll_disposition(502), PollAction::Retry));
        assert!(matches!(poll_disposition(503), PollAction::Retry));
    }

    #[test]
    fn glb_url_must_be_https() {
        assert!(validate_glb_url("https://v3b.fal.media/files/a.glb").is_ok());
        // A different host is allowed (fal has moved this CDN before)
        // but logged.
        assert!(validate_glb_url("https://cdn.example.com/a.glb").is_ok());
        // Plain http is refused outright: the body feeds a parser.
        assert!(validate_glb_url("http://v3b.fal.media/files/a.glb").is_err());
        assert!(validate_glb_url("file:///etc/passwd").is_err());
        assert!(validate_glb_url("not a url").is_err());
    }
}
