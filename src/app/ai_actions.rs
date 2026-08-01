//! AI job lifecycle on the main thread:
//! - `start_ai_job`: build a request, spawn the worker, transition
//!   `AiJobState::Idle` → `Submitting`.
//! - `cancel_ai_job`: flip the cooperative-cancel flag.
//! - `tick_ai_job`: per-frame; drains worker events, advances
//!   `AiJobState`, and applies any `VoxelPatch` from a `Done` event
//!   through `CommandHistory::execute` so the result is undoable.
//!   Mirrors the shape of `app::preview::tick_preview`.

use std::sync::mpsc::{self, Receiver, TryRecvError};

use voxelith::ai::{AiJobState, AiRequest, JobEvent, JobHandle};
use voxelith::editor::{Command, Selection};
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

        // Record the prompt to the MRU at submit time — kept even if the
        // job later fails or is cancelled, since the user most likely
        // wants to retry or tweak it.
        let prompt = self.ui.ai_prompt.clone();
        let request = AiRequest {
            prompt: prompt.clone(),
            image: None,
            resolution: self.ui.ai_resolution,
        };

        self.ai_provider
            .submit(request, self.ai_runtime.handle(), tx, cancel);

        self.ai_event_rx = Some(rx);
        self.ai_handle = Some(handle);
        self.ai_job = AiJobState::Submitting;
        self.ui.set_status("AI: submitting");
        self.touch_recent_prompt(&prompt);
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
        let (events, disconnected) = match &self.ai_event_rx {
            Some(rx) => drain_events(rx),
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

        // The worker promises a terminal event before it goes away. If
        // the channel closed without one it died unexpectedly — a
        // panicked task is caught by tokio and simply drops its sender.
        // Left unhandled, the panel sat in "running" forever with
        // Generate disabled and Cancel inert; only a restart cleared it.
        if disconnected && !terminal {
            let message = "AI worker stopped unexpectedly (see the log)";
            log::error!("AI event channel disconnected with no terminal event");
            self.ui.set_status(format!("AI failed: {}", message));
            self.ai_job = AiJobState::Failed {
                message: message.to_string(),
            };
            terminal = true;
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
        // Dedupe by position + drop identity writes (see `patch_to_changes`).
        let changes = self.patch_to_changes(&patch);
        if changes.is_empty() {
            return;
        }
        // Remember the generated footprint for the "Frame Generated"
        // camera action.
        self.last_generated_bounds = super::bounds_of(patch.voxels.iter().map(|&(p, _)| p));
        let cmd = Command::set_voxels(changes);
        self.editor.history.execute(cmd, &mut self.world);

        // Placement polish: auto-select the result's AABB so it can be
        // moved / copied immediately (mirrors Paste's auto-select), and
        // frame it — the model lands at the world origin and is often
        // off-screen from where the user was working.
        if let Some((min, max)) = self.last_generated_bounds {
            self.editor.selection = Some(Selection::from_corners(min, max));
        }
        self.frame_generated();
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

/// Drain every queued worker event, reporting whether the channel is
/// also closed.
///
/// `Receiver::try_iter` can't tell "nothing right now" from "the sender
/// is gone", and the difference matters: a worker that dies without
/// sending a terminal event leaves the job state stuck forever. Buffered
/// events survive the sender being dropped, so a normal Done-then-drop
/// still delivers its payload before we see the disconnect.
fn drain_events(rx: &Receiver<JobEvent>) -> (Vec<JobEvent>, bool) {
    let mut events = Vec::new();
    loop {
        match rx.try_recv() {
            Ok(event) => events.push(event),
            Err(TryRecvError::Empty) => return (events, false),
            Err(TryRecvError::Disconnected) => return (events, true),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drain_reports_buffered_events_before_the_disconnect() {
        // The order matters: a worker that finishes normally sends its
        // terminal event and *then* drops the sender, so the drain has
        // to hand back the payload alongside the disconnect flag.
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(JobEvent::Submitted).unwrap();
        tx.send(JobEvent::Progress(0.5)).unwrap();
        drop(tx);

        let (events, disconnected) = drain_events(&rx);
        assert_eq!(events.len(), 2);
        assert!(disconnected);
    }

    #[test]
    fn drain_of_a_live_empty_channel_is_not_a_disconnect() {
        let (tx, rx) = std::sync::mpsc::channel::<JobEvent>();
        let (events, disconnected) = drain_events(&rx);
        assert!(events.is_empty());
        assert!(!disconnected, "sender is still alive");
        drop(tx);
    }

    #[test]
    fn drain_of_a_dead_empty_channel_reports_the_disconnect() {
        let (tx, rx) = std::sync::mpsc::channel::<JobEvent>();
        drop(tx);
        let (events, disconnected) = drain_events(&rx);
        assert!(events.is_empty());
        assert!(disconnected);
    }
}
