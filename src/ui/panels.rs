//! UI state and panel definitions.

use super::Tool;

/// UI state
#[derive(Default)]
pub struct UiState {
    // Panel visibility
    pub show_stats: bool,
    pub show_tools: bool,

    // Tool state
    pub tool: Tool,
    pub brush_color: [u8; 3],
    pub brush_size: u32,

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
            tool: Tool::Place,
            brush_color: [200, 100, 50], // Default orange-ish
            brush_size: 1,
            ..Default::default()
        }
    }
}
