//! UI state and panel definitions.

use super::CameraView;

/// One-shot UI actions that need to be processed by the application
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UiAction {
    // File operations
    NewProject,
    OpenProject,
    SaveProject,
    SaveAs,
    ImportVox,
    ExportVox,
    Exit,

    // Edit operations
    Undo,
    Redo,
    ClearAll,

    // Generate operations
    GenerateTestCube,
    GenerateGround,
    GenerateSphere,
    GeneratePyramid,

    // Camera operations
    ResetCamera,
    SetCameraView(CameraView),
}

/// UI state
#[derive(Default)]
pub struct UiState {
    // Panel visibility (toggles, not one-shot actions)
    pub show_stats: bool,
    pub show_tools: bool,
    pub show_palette: bool,
    pub show_viewport_settings: bool,
    pub show_help: bool,
    pub show_about: bool,

    // One-shot action queue
    pending_actions: Vec<UiAction>,

    // Status message for user feedback
    pub status_message: Option<(String, std::time::Instant)>,
}

impl UiState {
    pub fn new() -> Self {
        Self {
            show_stats: true,
            show_tools: true,
            show_palette: true,
            show_viewport_settings: false,
            show_help: false,
            show_about: false,
            pending_actions: Vec::new(),
            status_message: None,
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
