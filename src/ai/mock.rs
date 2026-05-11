//! `MockProvider`: simulates a real AI provider for Phase 1 wiring
//! tests.
//!
//! Walks through every state the real pipeline will go through —
//! `Submitting` → `Polling { progress }` (interpolated over 3
//! seconds) → `Voxelizing` → `Done` — without making any network
//! calls or producing voxel data. Lets us verify the channel /
//! event-pump / UI plumbing end-to-end before Phase 2 adds the real
//! fal.ai client.
//!
//! Cancellation is honored at every sleep boundary.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use super::job::JobEvent;
use super::provider::{AiProvider, AiRequest};

/// Sleeps and emits faked progress events. Total wall time ~3 s when
/// not cancelled.
pub struct MockProvider;

impl MockProvider {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MockProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl AiProvider for MockProvider {
    fn name(&self) -> &str {
        "Mock (Phase 1 stub)"
    }

    fn submit(
        &self,
        request: AiRequest,
        runtime: &tokio::runtime::Handle,
        events_tx: mpsc::Sender<JobEvent>,
        cancel: Arc<AtomicBool>,
    ) {
        runtime.spawn(async move {
            // Cooperative cancel helper: returns true when the worker
            // should bail. The first three checkpoints are between
            // each fake stage; the polling phase has 10 internal
            // checkpoints (one per progress tick).
            let cancelled = |c: &AtomicBool| c.load(Ordering::Acquire);
            let send_failed = |tx: &mpsc::Sender<JobEvent>, msg: &str| {
                let _ = tx.send(JobEvent::Failed {
                    message: msg.into(),
                });
            };

            // Stage 1: submitting (~300 ms).
            tokio::time::sleep(Duration::from_millis(300)).await;
            if cancelled(&cancel) {
                send_failed(&events_tx, "Cancelled");
                return;
            }
            if events_tx.send(JobEvent::Submitted).is_err() {
                return;
            }

            // Stage 2: polling for ~2.5 s with 10 progress ticks.
            for i in 0..10 {
                tokio::time::sleep(Duration::from_millis(250)).await;
                if cancelled(&cancel) {
                    send_failed(&events_tx, "Cancelled");
                    return;
                }
                let progress = (i + 1) as f32 / 10.0;
                if events_tx.send(JobEvent::Progress(progress)).is_err() {
                    return;
                }
            }

            // Stage 3: GLB ready (fake — no real bytes).
            if events_tx
                .send(JobEvent::GlbReady { byte_count: 0 })
                .is_err()
            {
                return;
            }

            // Stage 4: voxelizing (~200 ms).
            tokio::time::sleep(Duration::from_millis(200)).await;
            if cancelled(&cancel) {
                send_failed(&events_tx, "Cancelled");
                return;
            }

            // Done. Echo the prompt in the summary so the user can see
            // their request round-tripped through the worker. Mock
            // produces no voxel data — `patch=None` keeps `App` from
            // pushing a no-op Command onto the undo history.
            let summary = format!(
                "Mock generated for: {} (resolution {}, image: {})",
                request.prompt,
                request.resolution,
                if request.image.is_some() { "yes" } else { "no" }
            );
            let _ = events_tx.send(JobEvent::Done {
                summary,
                patch: None,
            });
        });
    }
}
