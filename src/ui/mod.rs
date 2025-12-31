//! User interface components using egui.

mod panels;

pub use panels::UiState;

use egui::Context;

/// Main UI manager
pub struct Ui {
    pub state: UiState,
}

impl Ui {
    pub fn new() -> Self {
        Self {
            state: UiState::default(),
        }
    }

    /// Render the UI
    pub fn show(&mut self, ctx: &Context, stats: &RenderStats) {
        // Top menu bar
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("New").clicked() {
                        self.state.new_project_requested = true;
                        ui.close_menu();
                    }
                    if ui.button("Open...").clicked() {
                        self.state.open_project_requested = true;
                        ui.close_menu();
                    }
                    if ui.button("Save").clicked() {
                        self.state.save_project_requested = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Exit").clicked() {
                        self.state.exit_requested = true;
                    }
                });
                ui.menu_button("Edit", |ui| {
                    if ui.button("Undo").clicked() {
                        self.state.undo_requested = true;
                        ui.close_menu();
                    }
                    if ui.button("Redo").clicked() {
                        self.state.redo_requested = true;
                        ui.close_menu();
                    }
                });
                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.state.show_stats, "Show Stats");
                    ui.checkbox(&mut self.state.show_tools, "Show Tools");
                });
                ui.menu_button("Generate", |ui| {
                    if ui.button("Test Cube").clicked() {
                        self.state.generate_test_cube = true;
                        ui.close_menu();
                    }
                    if ui.button("Ground Plane").clicked() {
                        self.state.generate_ground = true;
                        ui.close_menu();
                    }
                });
            });
        });

        // Stats panel
        if self.state.show_stats {
            egui::Window::new("Stats")
                .default_pos([10.0, 40.0])
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("FPS: {:.1}", stats.fps));
                    ui.label(format!("Frame time: {:.2}ms", stats.frame_time_ms));
                    ui.label(format!("Triangles: {}", stats.triangles));
                    ui.label(format!("Chunks: {}", stats.chunks));
                    ui.separator();
                    ui.label(format!(
                        "Camera: ({:.1}, {:.1}, {:.1})",
                        stats.camera_pos.0, stats.camera_pos.1, stats.camera_pos.2
                    ));
                });
        }

        // Tools panel
        if self.state.show_tools {
            egui::Window::new("Tools")
                .default_pos([10.0, 200.0])
                .resizable(true)
                .show(ctx, |ui| {
                    ui.heading("Brush");
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.state.tool, Tool::Place, "Place");
                        ui.selectable_value(&mut self.state.tool, Tool::Remove, "Remove");
                        ui.selectable_value(&mut self.state.tool, Tool::Paint, "Paint");
                    });

                    ui.separator();
                    ui.heading("Color");
                    let mut color = [
                        self.state.brush_color[0] as f32 / 255.0,
                        self.state.brush_color[1] as f32 / 255.0,
                        self.state.brush_color[2] as f32 / 255.0,
                    ];
                    if ui.color_edit_button_rgb(&mut color).changed() {
                        self.state.brush_color = [
                            (color[0] * 255.0) as u8,
                            (color[1] * 255.0) as u8,
                            (color[2] * 255.0) as u8,
                        ];
                    }

                    ui.separator();
                    ui.heading("Brush Size");
                    ui.add(egui::Slider::new(&mut self.state.brush_size, 1..=10));
                });
        }

        // Status bar
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Voxelith v0.1.0");
                ui.separator();
                ui.label(format!("Tool: {:?}", self.state.tool));
            });
        });
    }

    /// Clear one-shot flags
    pub fn clear_flags(&mut self) {
        self.state.new_project_requested = false;
        self.state.open_project_requested = false;
        self.state.save_project_requested = false;
        self.state.exit_requested = false;
        self.state.undo_requested = false;
        self.state.redo_requested = false;
        self.state.generate_test_cube = false;
        self.state.generate_ground = false;
    }
}

impl Default for Ui {
    fn default() -> Self {
        Self::new()
    }
}

/// Render statistics for UI display
#[derive(Default)]
pub struct RenderStats {
    pub fps: f32,
    pub frame_time_ms: f32,
    pub triangles: usize,
    pub chunks: usize,
    pub camera_pos: (f32, f32, f32),
}

/// Available tools
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tool {
    #[default]
    Place,
    Remove,
    Paint,
}
