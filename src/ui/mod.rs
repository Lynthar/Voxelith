//! User interface components using egui.

mod panels;

pub use panels::UiState;

use crate::editor::{Editor, Tool};
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
    pub fn show(&mut self, ctx: &Context, stats: &RenderStats, editor: &mut Editor) {
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
                    let undo_text = if editor.can_undo() { "Undo (Ctrl+Z)" } else { "Undo" };
                    if ui.add_enabled(editor.can_undo(), egui::Button::new(undo_text)).clicked() {
                        self.state.undo_requested = true;
                        ui.close_menu();
                    }
                    let redo_text = if editor.can_redo() { "Redo (Ctrl+Y)" } else { "Redo" };
                    if ui.add_enabled(editor.can_redo(), egui::Button::new(redo_text)).clicked() {
                        self.state.redo_requested = true;
                        ui.close_menu();
                    }
                });
                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.state.show_stats, "Show Stats");
                    ui.checkbox(&mut self.state.show_tools, "Show Tools");
                    ui.checkbox(&mut self.state.show_palette, "Show Palette");
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
                    ui.separator();
                    ui.label(format!(
                        "History: {} undo, {} redo",
                        editor.history.undo_count(),
                        editor.history.redo_count()
                    ));
                });
        }

        // Tools panel
        if self.state.show_tools {
            egui::Window::new("Tools")
                .default_pos([10.0, 200.0])
                .resizable(true)
                .show(ctx, |ui| {
                    ui.heading("Tool");
                    ui.horizontal(|ui| {
                        if ui.selectable_label(editor.current_tool == Tool::Place, "Place (1)").clicked() {
                            editor.current_tool = Tool::Place;
                        }
                        if ui.selectable_label(editor.current_tool == Tool::Remove, "Remove (2)").clicked() {
                            editor.current_tool = Tool::Remove;
                        }
                    });
                    ui.horizontal(|ui| {
                        if ui.selectable_label(editor.current_tool == Tool::Paint, "Paint (3)").clicked() {
                            editor.current_tool = Tool::Paint;
                        }
                        if ui.selectable_label(editor.current_tool == Tool::Eyedropper, "Pick (4)").clicked() {
                            editor.current_tool = Tool::Eyedropper;
                        }
                    });
                    ui.horizontal(|ui| {
                        if ui.selectable_label(editor.current_tool == Tool::Fill, "Fill (5)").clicked() {
                            editor.current_tool = Tool::Fill;
                        }
                    });

                    ui.separator();
                    ui.heading("Brush Size");
                    let mut size = editor.brush_size as u32;
                    if ui.add(egui::Slider::new(&mut size, 1..=10)).changed() {
                        editor.brush_size = size as u8;
                    }

                    ui.separator();
                    ui.heading("Color");
                    let mut color = [
                        editor.brush_color.r as f32 / 255.0,
                        editor.brush_color.g as f32 / 255.0,
                        editor.brush_color.b as f32 / 255.0,
                    ];
                    if ui.color_edit_button_rgb(&mut color).changed() {
                        editor.brush_color = crate::core::Voxel::from_rgb(
                            (color[0] * 255.0) as u8,
                            (color[1] * 255.0) as u8,
                            (color[2] * 255.0) as u8,
                        );
                    }

                    // Show hovered voxel info
                    if let Some(hit) = &editor.hovered_voxel {
                        ui.separator();
                        ui.heading("Hovered Voxel");
                        ui.label(format!("Pos: ({}, {}, {})", hit.voxel_pos.0, hit.voxel_pos.1, hit.voxel_pos.2));
                        ui.label(format!("Normal: ({}, {}, {})", hit.normal.0, hit.normal.1, hit.normal.2));
                    }
                });
        }

        // Color palette panel
        if self.state.show_palette {
            egui::Window::new("Palette")
                .default_pos([10.0, 450.0])
                .resizable(true)
                .show(ctx, |ui| {
                    let palette = &editor.palette;
                    let cols = 5;

                    egui::Grid::new("palette_grid")
                        .spacing([4.0, 4.0])
                        .show(ui, |ui| {
                            for (i, voxel) in palette.iter().enumerate() {
                                let color = egui::Color32::from_rgb(voxel.r, voxel.g, voxel.b);
                                let is_selected = editor.brush_color.r == voxel.r
                                    && editor.brush_color.g == voxel.g
                                    && editor.brush_color.b == voxel.b;

                                let size = if is_selected { 24.0 } else { 20.0 };
                                let (rect, response) = ui.allocate_exact_size(
                                    egui::vec2(size, size),
                                    egui::Sense::click(),
                                );

                                if response.clicked() {
                                    editor.brush_color = *voxel;
                                }

                                ui.painter().rect_filled(rect, 2.0, color);
                                if is_selected {
                                    ui.painter().rect_stroke(rect, 2.0, egui::Stroke::new(2.0, egui::Color32::WHITE));
                                }

                                if (i + 1) % cols == 0 {
                                    ui.end_row();
                                }
                            }
                        });
                });
        }

        // Status bar
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Voxelith v0.1.0");
                ui.separator();
                ui.label(format!("Tool: {}", editor.current_tool.name()));
                ui.separator();
                ui.label(format!("Brush: {}px", editor.brush_size));
                if let Some(hit) = &editor.hovered_voxel {
                    ui.separator();
                    ui.label(format!("Voxel: ({}, {}, {})", hit.voxel_pos.0, hit.voxel_pos.1, hit.voxel_pos.2));
                }
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
