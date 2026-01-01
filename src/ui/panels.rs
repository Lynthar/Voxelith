//! UI state and panel definitions.

use super::CameraView;

/// UI state
#[derive(Default)]
pub struct UiState {
    // Panel visibility
    pub show_stats: bool,
    pub show_tools: bool,
    pub show_palette: bool,
    pub show_viewport_settings: bool,
    pub show_help: bool,
    pub show_about: bool,

    // One-shot action flags
    pub new_project_requested: bool,
    pub open_project_requested: bool,
    pub save_project_requested: bool,
    pub save_as_requested: bool,
    pub import_vox_requested: bool,
    pub export_vox_requested: bool,
    pub exit_requested: bool,
    pub undo_requested: bool,
    pub redo_requested: bool,
    pub clear_all_requested: bool,
    pub generate_test_cube: bool,
    pub generate_ground: bool,
    pub generate_sphere: bool,
    pub generate_pyramid: bool,
    pub reset_camera_requested: bool,
    pub camera_view: Option<CameraView>,

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
            ..Default::default()
        }
    }
}
