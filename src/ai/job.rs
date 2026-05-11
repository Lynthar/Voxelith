//! AI job state machine + events.
//!
//! The state machine lives on the main thread (in `App`); the worker
//! task on the tokio runtime emits [`JobEvent`]s through a sync
//! channel and the main thread translates each event into an
//! [`AiJobState`] transition during its per-frame `tick_ai_job`.

use crate::procgen::VoxelPatch;

/// What the AI generation pipeline is currently doing. Single value
/// owned by `App`; UI reads it to render the panel state.
#[derive(Debug, Clone)]
pub enum AiJobState {
    /// No active job. Generate button enabled.
    Idle,
    /// HTTP submission to the provider; waiting for the provider to
    /// accept and assign a job id. Cancel allowed.
    Submitting,
    /// Provider is running the job. `progress` is 0.0..=1.0; for
    /// providers that don't report fine-grained progress this stays
    /// at a fixed indeterminate value (e.g. 0.5) until the next stage.
    Polling { progress: f32 },
    /// GLB downloaded, mesh-to-voxel conversion in progress. Phase 3
    /// will populate the actual voxelizer here.
    Voxelizing,
    /// Pipeline finished successfully. `summary` is a short status-bar
    /// message (e.g. "Generated 4128 voxels"). Phase 4 attaches the
    /// `VoxelPatch` and applies it via `Command::set_voxels`.
    Done { summary: String },
    /// Pipeline aborted. `message` is the user-facing reason — keep it
    /// short, the status bar is the primary surface.
    Failed { message: String },
}

impl AiJobState {
    /// True when a job is active (i.e. Generate should be disabled,
    /// Cancel should be enabled).
    pub fn is_running(&self) -> bool {
        matches!(
            self,
            AiJobState::Submitting | AiJobState::Polling { .. } | AiJobState::Voxelizing
        )
    }

    /// True when no job is active and the user could start one.
    pub fn is_idle(&self) -> bool {
        matches!(self, AiJobState::Idle | AiJobState::Done { .. } | AiJobState::Failed { .. })
    }

    /// Best-effort progress fraction in 0..=1 for the progress bar.
    /// Indeterminate stages return a fixed midpoint value; the bar can
    /// also be rendered as a spinner when this returns `None`.
    pub fn progress(&self) -> Option<f32> {
        match self {
            AiJobState::Submitting => Some(0.05),
            AiJobState::Polling { progress } => Some(progress.clamp(0.05, 0.95)),
            AiJobState::Voxelizing => Some(0.95),
            AiJobState::Done { .. } => Some(1.0),
            _ => None,
        }
    }

    /// Short label suitable for the panel header / status bar.
    pub fn label(&self) -> &str {
        match self {
            AiJobState::Idle => "Ready",
            AiJobState::Submitting => "Submitting…",
            AiJobState::Polling { .. } => "Generating…",
            AiJobState::Voxelizing => "Voxelizing…",
            AiJobState::Done { .. } => "Done",
            AiJobState::Failed { .. } => "Failed",
        }
    }
}

impl Default for AiJobState {
    fn default() -> Self {
        AiJobState::Idle
    }
}

/// Events emitted by the worker task. The `App::tick_ai_job` drains
/// these via `mpsc::Receiver::try_iter`.
#[derive(Debug, Clone)]
pub enum JobEvent {
    /// Submission accepted by the provider.
    Submitted,
    /// Periodic progress update from polling.
    Progress(f32),
    /// Provider returned the GLB bytes; voxelization is starting.
    GlbReady { byte_count: usize },
    /// Pipeline finished successfully. `patch` is `Some` for providers
    /// that ran through voxelization (the real fal.ai client in
    /// Phase 3+); `None` for diagnostic / mock providers that only
    /// want to surface a status message without modifying the world.
    Done {
        summary: String,
        patch: Option<VoxelPatch>,
    },
    /// Pipeline failed at any stage.
    Failed { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_is_idle_not_running() {
        let s = AiJobState::Idle;
        assert!(s.is_idle());
        assert!(!s.is_running());
    }

    #[test]
    fn polling_is_running_not_idle() {
        let s = AiJobState::Polling { progress: 0.5 };
        assert!(!s.is_idle());
        assert!(s.is_running());
    }

    #[test]
    fn done_is_idle_again_so_user_can_start_a_new_job() {
        let s = AiJobState::Done {
            summary: "test".into(),
        };
        assert!(s.is_idle());
        assert!(!s.is_running());
    }

    #[test]
    fn failed_is_idle_again_so_user_can_retry() {
        let s = AiJobState::Failed {
            message: "test".into(),
        };
        assert!(s.is_idle());
    }

    #[test]
    fn progress_clamps_to_visible_range() {
        // The progress bar would look broken if a provider reported 0.0
        // or 1.0 mid-pipeline; clamp into a "definitely making progress"
        // range so the user always sees motion.
        let s = AiJobState::Polling { progress: 0.0 };
        assert_eq!(s.progress(), Some(0.05));
        let s = AiJobState::Polling { progress: 1.0 };
        assert_eq!(s.progress(), Some(0.95));
        let s = AiJobState::Polling { progress: 0.5 };
        assert_eq!(s.progress(), Some(0.5));
    }
}
