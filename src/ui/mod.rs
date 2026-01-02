//! User interface components using egui.

mod panels;

pub use panels::{UiAction, UiState};

use crate::editor::{Editor, Tool};
use egui::Context;

/// Viewport display settings
#[derive(Debug, Clone)]
pub struct ViewportSettings {
    pub show_grid: bool,
    pub show_axes: bool,
    pub wireframe_mode: bool,
    pub grid_size: i32,
    pub grid_spacing: f32,
}

impl Default for ViewportSettings {
    fn default() -> Self {
        Self {
            show_grid: true,
            show_axes: true,
            wireframe_mode: false,
            grid_size: 20,
            grid_spacing: 1.0,
        }
    }
}

/// Main UI manager
pub struct Ui {
    pub state: UiState,
    pub viewport: ViewportSettings,
}

impl Ui {
    pub fn new() -> Self {
        Self {
            state: UiState::default(),
            viewport: ViewportSettings::default(),
        }
    }

    /// Render the UI
    pub fn show(&mut self, ctx: &Context, stats: &RenderStats, editor: &mut Editor) {
        // Top menu bar
        self.show_menu_bar(ctx, editor);

        // Left side panel with tools
        self.show_toolbar(ctx, editor);

        // Stats panel
        if self.state.show_stats {
            self.show_stats_panel(ctx, stats, editor);
        }

        // Tools panel
        if self.state.show_tools {
            self.show_tools_panel(ctx, editor);
        }

        // Color palette panel
        if self.state.show_palette {
            self.show_palette_panel(ctx, editor);
        }

        // Viewport settings panel
        if self.state.show_viewport_settings {
            self.show_viewport_panel(ctx);
        }

        // Help panel
        if self.state.show_help {
            self.show_help_panel(ctx);
        }

        // Status bar
        self.show_status_bar(ctx, editor);
    }

    fn show_menu_bar(&mut self, ctx: &Context, editor: &Editor) {
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("New Project").clicked() {
                        self.state.request(UiAction::NewProject);
                        ui.close_menu();
                    }
                    if ui.button("Open...").clicked() {
                        self.state.request(UiAction::OpenProject);
                        ui.close_menu();
                    }
                    if ui.button("Save").clicked() {
                        self.state.request(UiAction::SaveProject);
                        ui.close_menu();
                    }
                    if ui.button("Save As...").clicked() {
                        self.state.request(UiAction::SaveAs);
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.menu_button("Import", |ui| {
                        if ui.button("MagicaVoxel (.vox)...").clicked() {
                            self.state.request(UiAction::ImportVox);
                            ui.close_menu();
                        }
                    });
                    ui.menu_button("Export", |ui| {
                        if ui.button("MagicaVoxel (.vox)...").clicked() {
                            self.state.request(UiAction::ExportVox);
                            ui.close_menu();
                        }
                    });
                    ui.separator();
                    if ui.button("Exit").clicked() {
                        self.state.request(UiAction::Exit);
                    }
                });

                ui.menu_button("Edit", |ui| {
                    let undo_text = if editor.can_undo() { "Undo  Ctrl+Z" } else { "Undo" };
                    if ui.add_enabled(editor.can_undo(), egui::Button::new(undo_text)).clicked() {
                        self.state.request(UiAction::Undo);
                        ui.close_menu();
                    }
                    let redo_text = if editor.can_redo() { "Redo  Ctrl+Y" } else { "Redo" };
                    if ui.add_enabled(editor.can_redo(), egui::Button::new(redo_text)).clicked() {
                        self.state.request(UiAction::Redo);
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Clear All").clicked() {
                        self.state.request(UiAction::ClearAll);
                        ui.close_menu();
                    }
                });

                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.state.show_stats, "Statistics");
                    ui.checkbox(&mut self.state.show_tools, "Tools Panel");
                    ui.checkbox(&mut self.state.show_palette, "Color Palette");
                    ui.checkbox(&mut self.state.show_viewport_settings, "Viewport Settings");
                    ui.separator();
                    ui.checkbox(&mut self.viewport.show_grid, "Show Grid");
                    ui.checkbox(&mut self.viewport.show_axes, "Show Axes");
                    ui.checkbox(&mut self.viewport.wireframe_mode, "Wireframe Mode");
                });

                ui.menu_button("Generate", |ui| {
                    if ui.button("Test Cube").clicked() {
                        self.state.request(UiAction::GenerateTestCube);
                        ui.close_menu();
                    }
                    if ui.button("Ground Plane").clicked() {
                        self.state.request(UiAction::GenerateGround);
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Sphere").clicked() {
                        self.state.request(UiAction::GenerateSphere);
                        ui.close_menu();
                    }
                    if ui.button("Pyramid").clicked() {
                        self.state.request(UiAction::GeneratePyramid);
                        ui.close_menu();
                    }
                });

                ui.menu_button("Help", |ui| {
                    if ui.button("Keyboard Shortcuts").clicked() {
                        self.state.show_help = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("About Voxelith").clicked() {
                        self.state.show_about = true;
                        ui.close_menu();
                    }
                });
            });
        });
    }

    fn show_toolbar(&mut self, ctx: &Context, editor: &mut Editor) {
        egui::SidePanel::left("toolbar")
            .resizable(false)
            .default_width(48.0)
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(8.0);

                    // Tool buttons
                    let tool_button = |ui: &mut egui::Ui, tool: Tool, current: Tool, icon: &str, tooltip: &str| -> bool {
                        let selected = tool == current;
                        ui.add(
                            egui::Button::new(icon)
                                .min_size(egui::vec2(36.0, 36.0))
                                .selected(selected)
                        )
                        .on_hover_text(tooltip)
                        .clicked()
                    };

                    if tool_button(ui, Tool::Place, editor.current_tool, "+", "Place (1)") {
                        editor.current_tool = Tool::Place;
                    }
                    if tool_button(ui, Tool::Remove, editor.current_tool, "-", "Remove (2)") {
                        editor.current_tool = Tool::Remove;
                    }
                    if tool_button(ui, Tool::Paint, editor.current_tool, "P", "Paint (3)") {
                        editor.current_tool = Tool::Paint;
                    }
                    if tool_button(ui, Tool::Eyedropper, editor.current_tool, "E", "Eyedropper (4)") {
                        editor.current_tool = Tool::Eyedropper;
                    }
                    if tool_button(ui, Tool::Fill, editor.current_tool, "F", "Fill (5)") {
                        editor.current_tool = Tool::Fill;
                    }

                    ui.add_space(16.0);
                    ui.separator();
                    ui.add_space(8.0);

                    // Current color preview
                    let color = egui::Color32::from_rgb(
                        editor.brush_color.r,
                        editor.brush_color.g,
                        editor.brush_color.b,
                    );
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(32.0, 32.0), egui::Sense::hover());
                    ui.painter().rect_filled(rect, 4.0, color);
                    ui.painter().rect_stroke(rect, 4.0, egui::Stroke::new(1.0, egui::Color32::WHITE));

                    ui.add_space(8.0);

                    // Brush size indicator
                    ui.label(format!("{}", editor.brush_size));
                });
            });
    }

    fn show_stats_panel(&self, ctx: &Context, stats: &RenderStats, editor: &Editor) {
        egui::Window::new("Statistics")
            .default_pos([60.0, 40.0])
            .resizable(false)
            .collapsible(true)
            .show(ctx, |ui| {
                egui::Grid::new("stats_grid")
                    .num_columns(2)
                    .spacing([20.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("FPS:");
                        ui.label(format!("{:.1}", stats.fps));
                        ui.end_row();

                        ui.label("Frame time:");
                        ui.label(format!("{:.2}ms", stats.frame_time_ms));
                        ui.end_row();

                        ui.label("Triangles:");
                        ui.label(format!("{}", stats.triangles));
                        ui.end_row();

                        ui.label("Chunks:");
                        ui.label(format!("{}", stats.chunks));
                        ui.end_row();

                        ui.label("History:");
                        ui.label(format!("{} / {}", editor.history.undo_count(), editor.history.redo_count()));
                        ui.end_row();
                    });

                ui.separator();

                ui.label(format!(
                    "Camera: ({:.1}, {:.1}, {:.1})",
                    stats.camera_pos.0, stats.camera_pos.1, stats.camera_pos.2
                ));
            });
    }

    fn show_tools_panel(&mut self, ctx: &Context, editor: &mut Editor) {
        egui::Window::new("Tools")
            .default_pos([60.0, 200.0])
            .resizable(true)
            .collapsible(true)
            .show(ctx, |ui| {
                // Tool selection
                ui.heading("Tool");
                egui::Grid::new("tool_grid")
                    .num_columns(3)
                    .spacing([4.0, 4.0])
                    .show(ui, |ui| {
                        if ui.selectable_label(editor.current_tool == Tool::Place, "Place").clicked() {
                            editor.current_tool = Tool::Place;
                        }
                        if ui.selectable_label(editor.current_tool == Tool::Remove, "Remove").clicked() {
                            editor.current_tool = Tool::Remove;
                        }
                        if ui.selectable_label(editor.current_tool == Tool::Paint, "Paint").clicked() {
                            editor.current_tool = Tool::Paint;
                        }
                        ui.end_row();

                        if ui.selectable_label(editor.current_tool == Tool::Eyedropper, "Pick").clicked() {
                            editor.current_tool = Tool::Eyedropper;
                        }
                        if ui.selectable_label(editor.current_tool == Tool::Fill, "Fill").clicked() {
                            editor.current_tool = Tool::Fill;
                        }
                        ui.end_row();
                    });

                ui.separator();

                // Brush size
                ui.heading("Brush Size");
                let mut size = editor.brush_size as u32;
                ui.add(egui::Slider::new(&mut size, 1..=10).show_value(true));
                editor.brush_size = size as u8;

                ui.separator();

                // Color
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

                // RGB values
                ui.horizontal(|ui| {
                    ui.label("RGB:");
                    ui.label(format!("{}, {}, {}", editor.brush_color.r, editor.brush_color.g, editor.brush_color.b));
                });

                // Show hovered voxel info
                if let Some(hit) = &editor.hovered_voxel {
                    ui.separator();
                    ui.heading("Hovered");
                    ui.label(format!("Position: ({}, {}, {})", hit.voxel_pos.0, hit.voxel_pos.1, hit.voxel_pos.2));
                    ui.label(format!("Face: ({}, {}, {})", hit.normal.0, hit.normal.1, hit.normal.2));
                }
            });
    }

    fn show_palette_panel(&mut self, ctx: &Context, editor: &mut Editor) {
        egui::Window::new("Palette")
            .default_pos([60.0, 450.0])
            .resizable(true)
            .collapsible(true)
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

                ui.separator();

                // Quick color buttons
                ui.horizontal(|ui| {
                    if ui.button("Add").clicked() {
                        // Add current color to palette (would need mutable palette)
                    }
                });
            });
    }

    fn show_viewport_panel(&mut self, ctx: &Context) {
        egui::Window::new("Viewport Settings")
            .default_pos([ctx.screen_rect().width() - 220.0, 40.0])
            .resizable(false)
            .collapsible(true)
            .show(ctx, |ui| {
                ui.heading("Display");
                ui.checkbox(&mut self.viewport.show_grid, "Show Grid");
                ui.checkbox(&mut self.viewport.show_axes, "Show Axes");
                ui.checkbox(&mut self.viewport.wireframe_mode, "Wireframe Mode");

                ui.separator();

                ui.heading("Grid");
                ui.add(egui::Slider::new(&mut self.viewport.grid_size, 5..=50).text("Size"));
                ui.add(egui::Slider::new(&mut self.viewport.grid_spacing, 0.5..=5.0).text("Spacing"));

                ui.separator();

                ui.heading("Camera");
                if ui.button("Reset Camera").clicked() {
                    self.state.request(UiAction::ResetCamera);
                }

                ui.horizontal(|ui| {
                    if ui.button("Top").clicked() {
                        self.state.request(UiAction::SetCameraView(CameraView::Top));
                    }
                    if ui.button("Front").clicked() {
                        self.state.request(UiAction::SetCameraView(CameraView::Front));
                    }
                    if ui.button("Side").clicked() {
                        self.state.request(UiAction::SetCameraView(CameraView::Side));
                    }
                });
            });
    }

    fn show_help_panel(&mut self, ctx: &Context) {
        egui::Window::new("Keyboard Shortcuts")
            .default_pos([ctx.screen_rect().width() / 2.0 - 150.0, 100.0])
            .resizable(false)
            .collapsible(false)
            .open(&mut self.state.show_help)
            .show(ctx, |ui| {
                egui::Grid::new("shortcuts_grid")
                    .num_columns(2)
                    .spacing([40.0, 4.0])
                    .show(ui, |ui| {
                        ui.heading("Tools");
                        ui.end_row();

                        ui.label("1");
                        ui.label("Place tool");
                        ui.end_row();

                        ui.label("2");
                        ui.label("Remove tool");
                        ui.end_row();

                        ui.label("3");
                        ui.label("Paint tool");
                        ui.end_row();

                        ui.label("4");
                        ui.label("Eyedropper");
                        ui.end_row();

                        ui.label("5");
                        ui.label("Fill tool");
                        ui.end_row();

                        ui.end_row();
                        ui.heading("Edit");
                        ui.end_row();

                        ui.label("Ctrl+Z");
                        ui.label("Undo");
                        ui.end_row();

                        ui.label("Ctrl+Y");
                        ui.label("Redo");
                        ui.end_row();

                        ui.label("Ctrl+Shift+Z");
                        ui.label("Redo");
                        ui.end_row();

                        ui.end_row();
                        ui.heading("Camera");
                        ui.end_row();

                        ui.label("WASD");
                        ui.label("Move camera");
                        ui.end_row();

                        ui.label("Q / Space");
                        ui.label("Move up");
                        ui.end_row();

                        ui.label("E / Shift");
                        ui.label("Move down");
                        ui.end_row();

                        ui.label("Middle Mouse");
                        ui.label("Orbit camera");
                        ui.end_row();

                        ui.label("Right Mouse");
                        ui.label("Pan camera");
                        ui.end_row();

                        ui.label("Scroll");
                        ui.label("Zoom");
                        ui.end_row();

                        ui.label("Escape");
                        ui.label("Release cursor");
                        ui.end_row();

                        ui.end_row();
                        ui.heading("File");
                        ui.end_row();

                        ui.label("Ctrl+N");
                        ui.label("New project");
                        ui.end_row();

                        ui.label("Ctrl+O");
                        ui.label("Open project");
                        ui.end_row();

                        ui.label("Ctrl+S");
                        ui.label("Save project");
                        ui.end_row();

                        ui.label("Ctrl+Shift+S");
                        ui.label("Save as...");
                        ui.end_row();

                        ui.end_row();
                        ui.heading("Actions");
                        ui.end_row();

                        ui.label("Left Click");
                        ui.label("Apply tool");
                        ui.end_row();
                    });
            });
    }

    fn show_status_bar(&mut self, ctx: &Context, editor: &Editor) {
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Show status message if recent (within 5 seconds)
                if let Some((msg, time)) = &self.state.status_message {
                    if time.elapsed().as_secs() < 5 {
                        ui.label(egui::RichText::new(msg).color(egui::Color32::YELLOW));
                        ui.separator();
                    } else {
                        self.state.status_message = None;
                    }
                }

                ui.label("Voxelith v0.1.0");
                ui.separator();
                ui.label(format!("Tool: {}", editor.current_tool.name()));
                ui.separator();
                ui.label(format!("Brush: {}px", editor.brush_size));
                ui.separator();
                ui.label(format!(
                    "Color: RGB({}, {}, {})",
                    editor.brush_color.r, editor.brush_color.g, editor.brush_color.b
                ));
                if let Some(hit) = &editor.hovered_voxel {
                    ui.separator();
                    ui.label(format!(
                        "Cursor: ({}, {}, {})",
                        hit.voxel_pos.0, hit.voxel_pos.1, hit.voxel_pos.2
                    ));
                }

                // Right-aligned viewport info
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.viewport.wireframe_mode {
                        ui.label("[Wireframe]");
                    }
                    if self.viewport.show_grid {
                        ui.label("[Grid]");
                    }
                    if self.viewport.show_axes {
                        ui.label("[Axes]");
                    }
                });
            });
        });
    }

    /// Set a status message to display
    pub fn set_status(&mut self, message: impl Into<String>) {
        self.state.status_message = Some((message.into(), std::time::Instant::now()));
    }

    /// Clear one-shot action flags
    pub fn clear_flags(&mut self) {
        self.state.clear_actions();
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

/// Preset camera views
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CameraView {
    Top,
    Front,
    Side,
}
