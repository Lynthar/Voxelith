//! `AiProvider` trait + request / handle types.
//!
//! Providers encapsulate the differences between remote 3D-gen APIs
//! (fal.ai, Replicate, Tripo, etc.). The MVP only implements the
//! mock; Phase 2 adds the real fal.ai + Hunyuan3D V3 client. The
//! trait is intentionally minimal — each impl owns its own HTTP
//! client, request shaping, polling cadence, and result download —
//! and communicates back to the main thread through a sync channel
//! the caller already supplied.

use std::sync::atomic::AtomicBool;
use std::sync::{mpsc, Arc};

use super::job::JobEvent;

/// A single AI generation request. Only `prompt` is required for the
/// text-to-3D path; `image` (Phase 2+) and `resolution` (Phase 3) are
/// available when the user provides them.
#[derive(Debug, Clone)]
pub struct AiRequest {
    /// Free-form text prompt. Required for text-to-3D providers.
    pub prompt: String,
    /// Optional reference image bytes (PNG / JPEG). When present, the
    /// provider should switch to its image-to-3D endpoint if it has
    /// one. Phase 1 doesn't wire UI for this — Phase 2 adds the
    /// upload control.
    pub image: Option<Vec<u8>>,
    /// Voxelization resolution along the longest axis. Provider may
    /// request higher mesh detail if this is large. Currently only
    /// read by the voxelizer in Phase 3; the provider may ignore it.
    pub resolution: u32,
}

/// Owner of an in-flight job. Holding this struct keeps the worker
/// task alive; dropping it does **not** cancel — the worker uses
/// `cancel.load(Acquire)` at safe points. Caller should `cancel.store
/// (true, Release)` to abort gracefully (the next event will be
/// `Failed { message: "Cancelled" }`).
pub struct JobHandle {
    /// Cooperative-cancel flag. Worker checks at safe points
    /// (between HTTP polls, before voxelization, etc.).
    pub cancel: Arc<AtomicBool>,
}

impl JobHandle {
    pub fn new() -> Self {
        Self {
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Request cooperative cancellation. The next worker poll will
    /// observe this and emit a `Failed { "Cancelled" }` event before
    /// exiting.
    pub fn request_cancel(&self) {
        self.cancel
            .store(true, std::sync::atomic::Ordering::Release);
    }
}

impl Default for JobHandle {
    fn default() -> Self {
        Self::new()
    }
}

/// Implemented by every backend. Each `submit` call spawns its own
/// async task on the provided `tokio::runtime::Handle` and returns
/// immediately; events flow back through `events_tx`.
///
/// `Send + Sync + 'static` is required so providers can be stored
/// behind `Arc<dyn AiProvider>` and shared with worker tasks.
pub trait AiProvider: Send + Sync + 'static {
    /// Display name shown in the UI ("fal.ai · Hunyuan3D V3").
    fn name(&self) -> &str;

    /// Spawn the job. The worker will emit at least one terminal
    /// event (`Done` or `Failed`) before exiting; the caller should
    /// transition the UI accordingly.
    fn submit(
        &self,
        request: AiRequest,
        runtime: &tokio::runtime::Handle,
        events_tx: mpsc::Sender<JobEvent>,
        cancel: Arc<AtomicBool>,
    );
}
