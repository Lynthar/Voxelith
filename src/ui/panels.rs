//! UI state and panel definitions.

use std::path::PathBuf;

use crate::editor::{Axis, Quarter};
use crate::mcp::bridge::Approval;
use crate::prefs::PanelVisibility;

use super::CameraView;

/// One-shot UI actions that need to be processed by the application.
///
/// Not `Copy` because `OpenRecent` carries a `PathBuf`. Actions are
/// taken by value via `UiState::take_actions`, so this is fine.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum UiAction {
    // File operations
    NewProject,
    OpenProject,
    /// Open a specific path from the recent-files MRU.
    OpenRecent(PathBuf),
    SaveProject,
    SaveAs,
    /// Re-read the open project from disk, discarding local edits.
    /// Raised by the disk-conflict strip, behind its confirm dialog.
    ReloadFromDisk,
    ImportVox,
    ExportVox,
    ExportObj,
    /// MC smoothed OBJ, no blur — preserves thin features
    ExportObjSmoothedLight,
    /// MC smoothed OBJ, 3×3×3 blur — clay-like, may dissolve thin features
    ExportObjSmoothedHeavy,
    ExportGlb,
    /// MC smoothed GLB, no blur
    ExportGlbSmoothedLight,
    /// MC smoothed GLB, 3×3×3 blur
    ExportGlbSmoothedHeavy,
    Exit,

    /// Unsaved-changes prompt: save the project, then run the action
    /// that was waiting on it. If the save doesn't happen (write error,
    /// or the user backs out of Save As) the pending action is dropped.
    UnsavedSave,
    /// Unsaved-changes prompt: run the pending action, losing the edits.
    UnsavedDiscard,
    /// Unsaved-changes prompt: forget the pending action.
    UnsavedCancel,

    // Edit operations
    Undo,
    Redo,
    /// Wipe the world, history and sockets. Not undoable, so the menu
    /// raises a confirm dialog and only dispatches this once the user
    /// accepts.
    ClearAll,

    // Selection / clipboard operations
    CopySelection,
    CutSelection,
    /// Paste at the selection's origin (or hovered cell when no
    /// selection exists). Ctrl+Shift+V's "always paste at cursor"
    /// is keyboard-only — UI buttons go through this default path.
    PasteClipboard,
    DeleteSelection,
    /// Set the selection to the AABB of every non-air voxel in
    /// the world.
    SelectAllSolid,
    /// Clear the active selection (Esc / Ctrl+D / Edit menu →
    /// Deselect / Tools panel button). Every entry point must route
    /// here, including ones holding `&mut Editor`: the selection also
    /// owns drag/move anchors and the translucent move ghost, which
    /// live on `App`. Writing `editor.selection = None` directly
    /// clears the marquee and strands the rest — see `App::deselect`.
    Deselect,
    /// Rotate the selection's voxel contents around `axis` by
    /// `quarter` (90° / -90° / 180°). Anchor is `selection.min`;
    /// the AABB may swap dimensions but its `min` corner stays put.
    /// One Ctrl+Z reverses the entire rotation.
    RotateSelection { axis: Axis, quarter: Quarter },
    /// Mirror the selection's voxel contents across the midplane
    /// perpendicular to `axis`. AABB unchanged.
    MirrorSelection { axis: Axis },

    // Generate operations
    GenerateTestCube,
    GenerateGround,
    GenerateSphere,
    GeneratePyramid,
    /// Run the procgen panel's currently-selected generator (terrain
    /// / tree / ...) and apply the result via CommandHistory (undo-able).
    GenerateProcedural,
    /// Run the pipeline graph and apply its output via CommandHistory.
    RunGraph,

    // Camera operations
    ResetCamera,
    SetCameraView(CameraView),
    /// Fit the camera to an AABB — center the target and pull back to
    /// the fit distance, keeping the current viewing angle. Three
    /// targets: the whole scene, the active selection, or the most
    /// recent generation.
    FrameAll,
    FrameSelected,
    FrameGenerated,

    // Crash recovery (in-app egui prompt; see `show_recovery_prompt`)
    /// Load the on-disk autosave into the editor.
    RecoverAutosave,
    /// Discard the on-disk autosave and keep the fresh default scene.
    DiscardAutosave,

    // AI operations
    /// Submit a new AI generation job using the current `ai_prompt` /
    /// `ai_resolution` from `App`. No-op when one is already running.
    AiGenerate,
    /// Cooperative cancel of the active job; the worker will emit a
    /// terminal `Failed { "Cancelled" }` event before stopping.
    AiCancel,
    /// Save the carried API key to the OS keychain. The key is moved
    /// out of the UI state immediately after saving so it doesn't
    /// linger in memory longer than necessary.
    AiSaveKey(String),
    /// Remove the stored API key.
    AiClearKey,

    // --- Agent bridge ---
    /// Start serving MCP on the given loopback port. `0` asks the OS for
    /// a free one.
    AgentStart(u16),
    /// Stop serving and close the socket.
    AgentStop,
    /// Switch between applying an agent's batches directly and being
    /// asked first.
    AgentApproval(Approval),
    /// Commit the batch waiting for approval.
    AgentAccept,
    /// Decline it. Nothing is applied and the agent is told why.
    AgentReject,
}

/// Display-ready summary of a completed export, shown in an in-app
/// dialog after a successful OBJ / GLB / VOX write so the user can
/// sanity-check a large export (triangle budget, file size, lost color
/// info) without digging through the transient status bar. `App` builds
/// it from the format's `*Stats` plus the written file's on-disk size;
/// the UI only lays it out. Optional fields are skipped in the dialog
/// when `None` — VOX has no triangle / chunk concept, so those stay
/// empty and only its palette / quantization lines show.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ExportReport {
    /// Human format name, e.g. "glTF Binary (.glb)".
    pub format: String,
    /// Written file's name (not the full path).
    pub filename: String,
    /// How the geometry was produced: "Greedy mesh", "Marching Cubes
    /// (light)", "Marching Cubes (heavy)", or "—" for formats with no
    /// meshing step (VOX).
    pub mesh_source: String,
    /// Triangle / vertex / chunk counts when the format meshes; all
    /// `None` for VOX.
    pub triangles: Option<usize>,
    pub vertices: Option<usize>,
    pub chunks: Option<usize>,
    /// On-disk size in bytes, read back after writing.
    pub file_size: Option<u64>,
    /// How colors are carried, e.g. "Per-vertex RGBA" or "254-color
    /// palette".
    pub color_model: String,
    /// Non-fatal notes worth surfacing, e.g. a palette-quantization
    /// count. One label per line in the dialog.
    pub notes: Vec<String>,
}

/// Human-readable byte size for the export report: `820` → `"820 B"`,
/// `4_096` → `"4.0 KiB"`, `5_242_880` → `"5.0 MiB"`. Binary units
/// (1024-based) since these are file sizes on disk.
pub fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let b = bytes as f64;
    if b < KIB {
        format!("{} B", bytes)
    } else if b < MIB {
        format!("{:.1} KiB", b / KIB)
    } else if b < GIB {
        format!("{:.1} MiB", b / MIB)
    } else {
        format!("{:.2} GiB", b / GIB)
    }
}

/// Group a count with thousands separators for the detailed export
/// report (the perf HUD uses the coarser `hud::compact_count`):
/// `1_234_567` → `"1,234,567"`, `999` → `"999"`.
pub fn group_thousands(n: usize) -> String {
    let s = n.to_string();
    let len = s.len();
    let mut out = String::with_capacity(len + len / 3);
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// A "this can't be undone — are you sure?" dialog, carrying the
/// action to dispatch if the user says yes.
#[derive(Debug, Clone)]
pub struct ConfirmPrompt {
    pub title: String,
    pub body: String,
    pub action: UiAction,
}

/// UI state.
///
/// Workspace panel toggles live in `panels`, the same struct prefs
/// persists — `App::new` assigns it wholesale from the loaded prefs
/// and `App::save_prefs` assigns it back, so neither end can drift
/// out of sync with the other. (There used to be a hand-written
/// `new()` carrying a second, contradicting set of defaults — it had
/// no callers.)
#[derive(Default)]
pub struct UiState {
    /// Open/closed state of the workspace panels. Persisted verbatim.
    pub panels: PanelVisibility,

    // Transient windows — not part of the saved layout.
    pub show_help: bool,
    pub show_about: bool,

    /// Crash-recovery prompt: an in-app egui dialog (NOT a native rfd
    /// modal — `rfd::MessageDialog` exits the process on this winit+wgpu
    /// setup). Set true at startup when an autosave is on disk; cleared
    /// when the user picks Recover or Discard.
    pub show_recovery_prompt: bool,

    /// Active file-operation error, shown as an in-app egui dialog
    /// (`(title, detail)`). Same reason as `show_recovery_prompt`: a
    /// native modal would crash the process on the very failure it's
    /// trying to report. `Some` while shown; cleared by the OK button.
    pub error_dialog: Option<(String, String)>,

    /// Report from the last successful export, shown as an in-app egui
    /// dialog (same click-to-dismiss contract as `error_dialog`). `Some`
    /// while shown; cleared by the dialog's Close button.
    pub export_report: Option<ExportReport>,

    /// Pending confirmation for a destructive, non-undoable action.
    /// Accepting dispatches the carried `UiAction`; cancelling drops it.
    /// In-app egui, never a native modal — see `show_recovery_prompt`.
    pub confirm: Option<ConfirmPrompt>,

    /// Unsaved-changes prompt. `Some(what)` while shown, where `what`
    /// describes the operation being held up ("open another project").
    /// Raised by `App::guard_then`; the Save / Don't Save / Cancel
    /// buttons dispatch the `Unsaved*` actions.
    pub unsaved_prompt: Option<String>,

    /// The open project changed on disk while there were local edits to
    /// lose, so the auto-reload was refused. `Some(file label)` for as
    /// long as that stands — written by `App::tick_disk_reload`, cleared
    /// when the two sides agree again (save / reload / open) or when the
    /// user dismisses the strip.
    ///
    /// A field rather than a status-bar line because the *state* lasts:
    /// every later write is refused too, and a message that scrolls away
    /// leaves the refusals looking like a broken feature.
    pub disk_conflict: Option<String>,

    // One-shot action queue
    pending_actions: Vec<UiAction>,

    // Status message for user feedback
    pub status_message: Option<(String, std::time::Instant)>,

    /// Buffer for the API key entry box in the AI panel. Held in UI
    /// state (not `App`) so it never crosses the main-thread boundary
    /// into a worker — once the user clicks "Save", the value is
    /// moved out into a `UiAction::AiSaveKey(_)` and the buffer is
    /// cleared.
    pub ai_key_input: String,

    /// Port the Agent panel will ask for when the bridge is started.
    /// A string rather than a `u16` because it is a text field, and an
    /// unparseable one has to disable the button rather than silently
    /// stand for some other port.
    pub agent_port_input: String,
}

impl UiState {
    /// Queue an action to be processed
    pub fn request(&mut self, action: UiAction) {
        if !self.pending_actions.contains(&action) {
            self.pending_actions.push(action);
        }
    }

    /// Take all pending actions (clears the queue)
    pub fn take_actions(&mut self) -> Vec<UiAction> {
        std::mem::take(&mut self.pending_actions)
    }

    /// Clear all pending actions
    pub fn clear_actions(&mut self) {
        self.pending_actions.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::{format_bytes, group_thousands};

    #[test]
    fn format_bytes_scales_binary_units() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(820), "820 B");
        assert_eq!(format_bytes(1024), "1.0 KiB");
        assert_eq!(format_bytes(1536), "1.5 KiB");
        assert_eq!(format_bytes(5 * 1024 * 1024), "5.0 MiB");
        assert_eq!(format_bytes(3 * 1024 * 1024 * 1024), "3.00 GiB");
    }

    #[test]
    fn group_thousands_inserts_separators() {
        assert_eq!(group_thousands(0), "0");
        assert_eq!(group_thousands(7), "7");
        assert_eq!(group_thousands(999), "999");
        assert_eq!(group_thousands(1_000), "1,000");
        assert_eq!(group_thousands(12_345), "12,345");
        assert_eq!(group_thousands(1_234_567), "1,234,567");
    }
}
