//! UI state and panel definitions.

/// UI state
#[derive(Default)]
pub struct UiState {
    // Panel visibility
    pub show_stats: bool,
    pub show_tools: bool,
    pub show_palette: bool,

    // One-shot action flags
    pub new_project_requested: bool,
    pub open_project_requested: bool,
    pub save_project_requested: bool,
    pub exit_requested: bool,
    pub undo_requested: bool,
    pub redo_requested: bool,
    pub generate_test_cube: bool,
    pub generate_ground: bool,
}

impl UiState {
    pub fn new() -> Self {
        Self {
            show_stats: true,
            show_tools: true,
            show_palette: true,
            ..Default::default()
        }
    }
}
