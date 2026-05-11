//! UI state and panel definitions.

use std::path::PathBuf;

use crate::editor::{Axis, Quarter};

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

    // Edit operations
    Undo,
    Redo,
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
    /// Deselect). Mirror of `editor.selection = None` for menu-
    /// bar contexts that don't get `&mut Editor`.
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
}

/// UI state
#[derive(Default)]
pub struct UiState {
    // Panel visibility (toggles, not one-shot actions)
    pub show_stats: bool,
    pub show_tools: bool,
    pub show_palette: bool,
    pub show_viewport_settings: bool,
    pub show_procgen: bool,
    pub show_graph: bool,
    pub show_help: bool,
    pub show_about: bool,
    pub show_ai: bool,

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
}

impl UiState {
    pub fn new() -> Self {
        Self {
            show_stats: true,
            show_tools: true,
            show_palette: true,
            show_viewport_settings: false,
            show_procgen: false,
            show_graph: false,
            show_help: false,
            show_about: false,
            show_ai: false,
            pending_actions: Vec::new(),
            status_message: None,
            ai_key_input: String::new(),
        }
    }

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
