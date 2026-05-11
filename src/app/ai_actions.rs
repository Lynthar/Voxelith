//! AI job lifecycle on the main thread:
//! - `start_ai_job`: build a request, spawn the worker, transition
//!   `AiJobState::Idle` → `Submitting`.
//! - `cancel_ai_job`: flip the cooperative-cancel flag.
//! - `tick_ai_job`: per-frame; drains worker events, advances
//!   `AiJobState`, and applies any `VoxelPatch` from a `Done` event
//!   through `CommandHistory::execute` so the result is undoable.
//!   Mirrors the shape of `app::preview::tick_preview`.

use std::sync::mpsc;

use voxelith::ai::{AiJobState, AiRequest, JobEvent, JobHandle};
use voxelith::editor::{Command, VoxelChange};
use voxelith::procgen::VoxelPatch;

use super::App;

impl App {
    /// Start a new AI generation job using the panel's current prompt
    /// and resolution. No-op when a job is already running.
    pub(super) fn start_ai_job(&mut self) {
        if self.ai_job.is_running() {
            return;
        }
        if self.ui.ai_prompt.trim().is_empty() {
            self.ui.set_status("AI: enter a prompt first");
            return;
        }
        if !self.ai_has_key {
            self.ui
                .set_status("AI: set your fal.ai API key in the AI panel first");
            return;
        }

        // Fresh channel + cancel token per job. Old ones (if any) are
        // dropped by the assignment below — the worker for any prior
        // job has already finished (we checked `is_running` above).
        let (tx, rx) = mpsc::channel();
        let handle = JobHandle::new();
        let cancel = handle.cancel.clone();

        let request = AiRequest {
            prompt: self.ui.ai_prompt.clone(),
            image: None,
            resolution: self.ui.ai_resolution,
        };

        self.ai_provider
            .submit(request, self.ai_runtime.handle(), tx, cancel);

        self.ai_event_rx = Some(rx);
        self.ai_handle = Some(handle);
        self.ai_job = AiJobState::Submitting;
        self.ui.set_status("AI: submitting");
    }

    /// Request cooperative cancellation of the active job. The worker
    /// will see the flag at its next checkpoint and emit a final
    /// `Failed { "Cancelled" }` event; `tick_ai_job` then transitions
    /// to `Failed` and clears the channel + handle.
    pub(super) fn cancel_ai_job(&mut self) {
        if let Some(handle) = &self.ai_handle {
            handle.request_cancel();
            self.ui.set_status("AI: cancelling…");
        }
    }

    /// Drain pending worker events and update `ai_job`. Called every
    /// frame from `RedrawRequested`. Cheap when no job is in flight.
    pub(super) fn tick_ai_job(&mut self) {
        // Collect into a local Vec so the immutable borrow on
        // `self.ai_event_rx` is dropped before we mutate `self` (e.g.
        // via `apply_ai_patch`).
        let events: Vec<JobEvent> = match &self.ai_event_rx {
            Some(rx) => rx.try_iter().collect(),
            None => return,
        };

        let mut terminal = false;
        for event in events {
            match event {
                JobEvent::Submitted => {
                    self.ai_job = AiJobState::Polling { progress: 0.0 };
                }
                JobEvent::Progress(p) => {
                    self.ai_job = AiJobState::Polling { progress: p };
                }
                JobEvent::GlbReady { byte_count: _ } => {
                    self.ai_job = AiJobState::Voxelizing;
                }
                JobEvent::Done { summary, patch } => {
                    if let Some(patch) = patch {
                        self.apply_ai_patch(patch);
                    }
                    self.ui.set_status(format!("AI: {}", summary));
                    self.ai_job = AiJobState::Done { summary };
                    terminal = true;
                }
                JobEvent::Failed { message } => {
                    self.ui.set_status(format!("AI failed: {}", message));
                    self.ai_job = AiJobState::Failed { message };
                    terminal = true;
                }
            }
        }

        if terminal {
            // Drop the receiver + cancel token so the next `start_ai_job`
            // gets a fresh pair. The terminal state remains visible in
            // the UI until the user clicks Generate or Dismiss.
            self.ai_event_rx = None;
            self.ai_handle = None;
        }
    }

    /// Land a finished AI patch into the world through `CommandHistory`
    /// so the user can Ctrl+Z it. Identity writes (cells already
    /// matching the new voxel) are filtered so a paint-over of an
    /// existing scene doesn't pollute the undo stack with no-ops.
    /// Phase 4 will polish placement (auto-center, auto-select).
    fn apply_ai_patch(&mut self, patch: VoxelPatch) {
        let changes: Vec<VoxelChange> = patch
            .voxels
            .iter()
            .filter_map(|&(pos, new_voxel)| {
                let old_voxel = self.world.get_voxel(pos.0, pos.1, pos.2);
                if old_voxel == new_voxel {
                    None
                } else {
                    Some(VoxelChange {
                        pos,
                        old_voxel,
                        new_voxel,
                    })
                }
            })
            .collect();
        if changes.is_empty() {
            return;
        }
        let cmd = Command::set_voxels(changes);
        self.editor.history.execute(cmd, &mut self.world);
    }

    /// Save a fresh API key to the keychain, refresh the cached flag.
    /// Called from the UI's "Save" button on the API key entry.
    pub(super) fn save_ai_key(&mut self, key: String) {
        let trimmed = key.trim();
        if trimmed.is_empty() {
            self.ui.set_status("AI: API key is empty");
            return;
        }
        match voxelith::ai::save_api_key("fal_ai", trimmed) {
            Ok(()) => {
                self.ai_has_key = true;
                self.ui.set_status("AI: API key saved to keychain");
            }
            Err(e) => {
                log::error!("Failed to save API key: {}", e);
                self.ui.set_status(format!("AI: save failed: {}", e));
            }
        }
    }

    /// Remove the stored API key. Used by the "Clear" button.
    pub(super) fn clear_ai_key(&mut self) {
        match voxelith::ai::clear_api_key("fal_ai") {
            Ok(()) => {
                self.ai_has_key = false;
                self.ui.set_status("AI: API key cleared");
            }
            Err(e) => {
                log::error!("Failed to clear API key: {}", e);
                self.ui.set_status(format!("AI: clear failed: {}", e));
            }
        }
    }
}
