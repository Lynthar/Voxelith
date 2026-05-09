//! User interface components using egui.

mod panels;

pub use panels::{UiAction, UiState};

use crate::editor::{Editor, Tool};
use crate::procgen::{
    CombineOp, FilterPredicate, LSystemTree, MaskMode, NodeId, NodeKind,
    PerlinTerrain, PipelineGraph, WfcGenerator, WfcTileset,
};
use egui::Context;

/// Viewport display settings
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

/// Which generator the procgen panel is currently editing.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum GeneratorChoice {
    Terrain,
    Tree,
    Wfc,
}

impl GeneratorChoice {
    /// Display label used by the panel's combo box and status messages.
    pub fn label(self) -> &'static str {
        match self {
            Self::Terrain => "Perlin Terrain",
            Self::Tree => "L-System Tree",
            Self::Wfc => "WFC Tile Layout",
        }
    }
}

impl Default for GeneratorChoice {
    fn default() -> Self {
        Self::Terrain
    }
}

/// Live state for the procedural-generation panel.
///
/// Each generator's instance doubles as its parameter state — UI
/// sliders mutate the fields in place, then `UiAction::GenerateProcedural`
/// triggers `selected`'s `generate()` in the application layer.
///
/// `preview_enabled` and `graph_preview_enabled` independently drive
/// translucent overlays — the first for the selected single generator,
/// the second for the pipeline graph's output. Both share the renderer's
/// preview slot; when both are on, the graph wins on the slot since
/// its tick runs second.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ProcgenSettings {
    pub selected: GeneratorChoice,
    pub terrain: PerlinTerrain,
    pub tree: LSystemTree,
    pub wfc: WfcGenerator,
    pub preview_enabled: bool,
    #[serde(default)]
    pub graph_preview_enabled: bool,
}

/// Main UI manager
pub struct Ui {
    pub state: UiState,
    pub viewport: ViewportSettings,
    pub procgen: ProcgenSettings,
    /// Pipeline graph edited in the Graph panel. Persisted in prefs.
    pub graph: PipelineGraph,
    /// Currently-selected node in the visual graph editor. Drives
    /// the sidebar parameter editor. Cleared automatically when the
    /// node is removed.
    pub selected_node: Option<NodeId>,
    /// Active wire-creation drag: source node whose output socket was
    /// pressed. While set, the editor renders a live wire from that
    /// socket to the cursor; on release a hit-test against input
    /// sockets either snaps the wire to a target or discards it.
    pub dragging_wire: Option<NodeId>,
    /// Recent-files MRU mirrored from `prefs::Prefs::recent_files`.
    /// App syncs this whenever the prefs version changes (touch_recent
    /// + initial load).
    pub recent_files: Vec<std::path::PathBuf>,
    /// Mirror of `App::clipboard.is_some()` so the Tools panel can
    /// gray out the Paste button without `App::clipboard` leaking
    /// across the UI layer boundary. App syncs it before each frame.
    pub has_clipboard: bool,
}

impl Ui {
    pub fn new() -> Self {
        Self {
            state: UiState::default(),
            viewport: ViewportSettings::default(),
            procgen: ProcgenSettings::default(),
            graph: PipelineGraph::default(),
            selected_node: None,
            dragging_wire: None,
            recent_files: Vec::new(),
            has_clipboard: false,
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

        // Procedural generation panel
        if self.state.show_procgen {
            self.show_procgen_panel(ctx);
        }

        // Pipeline graph panel
        if self.state.show_graph {
            self.show_graph_panel(ctx);
        }

        // Help panel
        if self.state.show_help {
            self.show_help_panel(ctx);
        }

        // About dialog
        if self.state.show_about {
            self.show_about_dialog(ctx);
        }

        // Status bar
        self.show_status_bar(ctx, editor);
    }

    fn show_menu_bar(&mut self, ctx: &Context, editor: &Editor) {
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("New").clicked() {
                        self.state.request(UiAction::NewProject);
                        ui.close_menu();
                    }
                    if ui.button("Open...").clicked() {
                        self.state.request(UiAction::OpenProject);
                        ui.close_menu();
                    }
                    ui.menu_button("Open Recent", |ui| {
                        if self.recent_files.is_empty() {
                            ui.add_enabled(false, egui::Button::new("(empty)"));
                        } else {
                            for path in &self.recent_files {
                                let label = path
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .map(|s| s.to_string())
                                    .unwrap_or_else(|| path.display().to_string());
                                let resp = ui
                                    .button(label)
                                    .on_hover_text(path.display().to_string());
                                if resp.clicked() {
                                    self.state.request(UiAction::OpenRecent(path.clone()));
                                    ui.close_menu();
                                }
                            }
                        }
                    });
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
                        if ui.button("Wavefront OBJ (.obj)...").clicked() {
                            self.state.request(UiAction::ExportObj);
                            ui.close_menu();
                        }
                        if ui
                            .button("Wavefront OBJ — smoothed, light (.obj)...")
                            .on_hover_text(
                                "Marching Cubes over raw voxel density: \
                                 voxel surfaces with rounded edges. \
                                 Preserves thin features (tree branches, \
                                 sparse detail).",
                            )
                            .clicked()
                        {
                            self.state.request(UiAction::ExportObjSmoothedLight);
                            ui.close_menu();
                        }
                        if ui
                            .button("Wavefront OBJ — smoothed, heavy (.obj)...")
                            .on_hover_text(
                                "Marching Cubes after a 3×3×3 density \
                                 blur: clay-like blobs. Best for terrain \
                                 / large solid masses; thin features may \
                                 dissolve.",
                            )
                            .clicked()
                        {
                            self.state.request(UiAction::ExportObjSmoothedHeavy);
                            ui.close_menu();
                        }
                        if ui.button("glTF Binary (.glb)...").clicked() {
                            self.state.request(UiAction::ExportGlb);
                            ui.close_menu();
                        }
                        if ui
                            .button("glTF Binary — smoothed, light (.glb)...")
                            .on_hover_text(
                                "Marching Cubes over raw voxel density: \
                                 voxel surfaces with rounded edges. \
                                 Preserves thin features.",
                            )
                            .clicked()
                        {
                            self.state.request(UiAction::ExportGlbSmoothedLight);
                            ui.close_menu();
                        }
                        if ui
                            .button("glTF Binary — smoothed, heavy (.glb)...")
                            .on_hover_text(
                                "Marching Cubes after a 3×3×3 density \
                                 blur: clay-like blobs. Best for terrain.",
                            )
                            .clicked()
                        {
                            self.state.request(UiAction::ExportGlbSmoothedHeavy);
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
                    let has_sel = editor.selection.is_some();
                    let can_paste = self.has_clipboard;
                    if ui
                        .add_enabled(has_sel, egui::Button::new("Cut  Ctrl+X"))
                        .clicked()
                    {
                        self.state.request(UiAction::CutSelection);
                        ui.close_menu();
                    }
                    if ui
                        .add_enabled(has_sel, egui::Button::new("Copy  Ctrl+C"))
                        .clicked()
                    {
                        self.state.request(UiAction::CopySelection);
                        ui.close_menu();
                    }
                    if ui
                        .add_enabled(can_paste, egui::Button::new("Paste  Ctrl+V"))
                        .clicked()
                    {
                        self.state.request(UiAction::PasteClipboard);
                        ui.close_menu();
                    }
                    if ui
                        .add_enabled(has_sel, egui::Button::new("Delete  Del"))
                        .clicked()
                    {
                        self.state.request(UiAction::DeleteSelection);
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Select All  Ctrl+A").clicked() {
                        self.state.request(UiAction::SelectAllSolid);
                        ui.close_menu();
                    }
                    if ui
                        .add_enabled(has_sel, egui::Button::new("Deselect  Esc"))
                        .clicked()
                    {
                        self.state.request(UiAction::Deselect);
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
                    ui.checkbox(&mut self.state.show_procgen, "Procedural Generation");
                    ui.checkbox(&mut self.state.show_graph, "Pipeline Graph");
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
                    ui.separator();
                    if ui.button("Procedural Terrain...").clicked() {
                        self.state.show_procgen = true;
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

                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(8.0);

                    // Shape tools — click-anchor / drag / release.
                    if tool_button(ui, Tool::Line, editor.current_tool, "L", "Line (6)") {
                        editor.current_tool = Tool::Line;
                    }
                    if tool_button(ui, Tool::Box, editor.current_tool, "▢", "Box (7)") {
                        editor.current_tool = Tool::Box;
                    }
                    if tool_button(ui, Tool::Sphere, editor.current_tool, "○", "Sphere (8)") {
                        editor.current_tool = Tool::Sphere;
                    }
                    if tool_button(ui, Tool::Cylinder, editor.current_tool, "⌭", "Cylinder (9)") {
                        editor.current_tool = Tool::Cylinder;
                    }

                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(8.0);

                    // Selection — drag an AABB; Esc / Ctrl+D to clear.
                    if tool_button(
                        ui,
                        Tool::Select,
                        editor.current_tool,
                        "▭",
                        "Select (0)\nDrag to mark an AABB. Esc or Ctrl+D deselects.",
                    ) {
                        editor.current_tool = Tool::Select;
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
                // Tool selection — split into Brush (cell-by-cell) and
                // Shape (click-anchor / drag / release) groups so the
                // distinct interaction model is visually clear.
                ui.heading("Brush");
                egui::Grid::new("brush_tool_grid")
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

                ui.add_space(4.0);
                ui.heading("Shape");
                egui::Grid::new("shape_tool_grid")
                    .num_columns(3)
                    .spacing([4.0, 4.0])
                    .show(ui, |ui| {
                        if ui
                            .selectable_label(editor.current_tool == Tool::Line, "Line")
                            .on_hover_text("Drag from anchor to end (3D Bresenham line)")
                            .clicked()
                        {
                            editor.current_tool = Tool::Line;
                        }
                        if ui
                            .selectable_label(editor.current_tool == Tool::Box, "Box")
                            .on_hover_text("Drag corner to corner (filled AABB)")
                            .clicked()
                        {
                            editor.current_tool = Tool::Box;
                        }
                        if ui
                            .selectable_label(editor.current_tool == Tool::Sphere, "Sphere")
                            .on_hover_text("Drag bbox; ellipsoid fits in it")
                            .clicked()
                        {
                            editor.current_tool = Tool::Sphere;
                        }
                        ui.end_row();

                        if ui
                            .selectable_label(editor.current_tool == Tool::Cylinder, "Cylinder")
                            .on_hover_text(
                                "Drag bbox; cylinder axis runs along the longest dimension",
                            )
                            .clicked()
                        {
                            editor.current_tool = Tool::Cylinder;
                        }
                        ui.end_row();
                    });

                ui.add_space(4.0);
                ui.heading("Selection");
                if ui
                    .selectable_label(editor.current_tool == Tool::Select, "Box Select")
                    .on_hover_text(
                        "Drag corner-to-corner to mark an AABB region for batch \
                         operations. Esc or Ctrl+D deselects.",
                    )
                    .clicked()
                {
                    editor.current_tool = Tool::Select;
                }
                if let Some(sel) = editor.selection {
                    let (w, h, d) = sel.size();
                    ui.label(
                        egui::RichText::new(format!(
                            "Active: {}×{}×{} ({} cells)",
                            w,
                            h,
                            d,
                            sel.cell_count()
                        ))
                        .small()
                        .weak(),
                    );
                }
                let has_sel = editor.selection.is_some();
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(has_sel, egui::Button::new("Copy"))
                        .on_hover_text("Ctrl+C — copy non-air voxels into the clipboard")
                        .clicked()
                    {
                        self.state.request(UiAction::CopySelection);
                    }
                    if ui
                        .add_enabled(has_sel, egui::Button::new("Cut"))
                        .on_hover_text("Ctrl+X — copy then clear in one undoable Command")
                        .clicked()
                    {
                        self.state.request(UiAction::CutSelection);
                    }
                    let can_paste = self.has_clipboard;
                    if ui
                        .add_enabled(can_paste, egui::Button::new("Paste"))
                        .on_hover_text(
                            "Ctrl+V — paste at selection origin (or cursor cell if no \
                             selection). Ctrl+Shift+V always pastes at cursor.",
                        )
                        .clicked()
                    {
                        self.state.request(UiAction::PasteClipboard);
                    }
                });
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(has_sel, egui::Button::new("Delete"))
                        .on_hover_text("Del — clear non-air voxels inside the selection")
                        .clicked()
                    {
                        self.state.request(UiAction::DeleteSelection);
                    }
                    if ui
                        .button("Select All")
                        .on_hover_text("Ctrl+A — select the AABB of every non-air voxel")
                        .clicked()
                    {
                        self.state.request(UiAction::SelectAllSolid);
                    }
                    if ui
                        .add_enabled(has_sel, egui::Button::new("Deselect"))
                        .on_hover_text("Esc / Ctrl+D — clear the active selection")
                        .clicked()
                    {
                        editor.selection = None;
                    }
                });

                ui.separator();

                // Brush size
                ui.heading("Brush Size");
                let mut size = editor.brush_size as u32;
                ui.add(egui::Slider::new(&mut size, 1..=10).show_value(true));
                editor.brush_size = size as u8;

                ui.separator();

                // Symmetry
                ui.heading("Symmetry");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut editor.symmetry.x, "X")
                        .on_hover_text("Mirror brush across the x = 0 plane");
                    ui.checkbox(&mut editor.symmetry.y, "Y")
                        .on_hover_text("Mirror brush across the y = 0 plane");
                    ui.checkbox(&mut editor.symmetry.z, "Z")
                        .on_hover_text("Mirror brush across the z = 0 plane");
                });
                ui.label(
                    egui::RichText::new(
                        "Mirrors Place / Remove / Paint / Fill across enabled \
                         planes through the world origin. Eyedropper is exempt.",
                    )
                    .small()
                    .weak(),
                );

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
                        // Check if color already exists in palette
                        let color = editor.brush_color;
                        let exists = editor.palette.iter().any(|v| {
                            v.r == color.r && v.g == color.g && v.b == color.b
                        });
                        if !exists && editor.palette.len() < 32 {
                            editor.palette.push(color);
                        }
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

    fn show_procgen_panel(&mut self, ctx: &Context) {
        // Deferred-action pattern: `.open(...)` borrows self.state.show_procgen
        // and the closure borrows self.procgen, so we can't dispatch a UiAction
        // (which mutates self.state) until both are released.
        let mut generate = false;
        let procgen = &mut self.procgen;

        egui::Window::new("Procedural Generation")
            .default_pos([ctx.screen_rect().width() - 240.0, 200.0])
            .default_width(240.0)
            .resizable(true)
            .collapsible(true)
            .open(&mut self.state.show_procgen)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Generator");
                    egui::ComboBox::from_id_salt("procgen_selected")
                        .selected_text(procgen.selected.label())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut procgen.selected,
                                GeneratorChoice::Terrain,
                                GeneratorChoice::Terrain.label(),
                            );
                            ui.selectable_value(
                                &mut procgen.selected,
                                GeneratorChoice::Tree,
                                GeneratorChoice::Tree.label(),
                            );
                            ui.selectable_value(
                                &mut procgen.selected,
                                GeneratorChoice::Wfc,
                                GeneratorChoice::Wfc.label(),
                            );
                        });
                });

                ui.separator();

                match procgen.selected {
                    GeneratorChoice::Terrain => {
                        terrain_params_ui(ui, &mut procgen.terrain)
                    }
                    GeneratorChoice::Tree => {
                        tree_params_ui(ui, &mut procgen.tree)
                    }
                    GeneratorChoice::Wfc => {
                        wfc_params_ui(ui, &mut procgen.wfc)
                    }
                }

                ui.separator();

                ui.horizontal(|ui| {
                    ui.checkbox(&mut procgen.preview_enabled, "Preview")
                        .on_hover_text(
                            "Show a translucent overlay of the generator's \
                             current output (debounced ~150ms)",
                        );
                    if ui
                        .button("Generate")
                        .on_hover_text("Apply generated voxels (undo-able)")
                        .clicked()
                    {
                        generate = true;
                    }
                });
            });

        if generate {
            self.state.request(UiAction::GenerateProcedural);
        }
    }

    fn show_graph_panel(&mut self, ctx: &Context) {
        // Deferred actions: collected during the immediate-mode pass,
        // applied after the window closure releases its borrows on
        // `self.graph` and `self.state`.
        let mut run = false;
        let mut delete_id: Option<NodeId> = None;
        let mut add_kind: Option<NodeKind> = None;
        let mut auto_layout = false;
        let mut wire_action: Option<(NodeId, usize, Option<NodeId>)> = None;
        let mut wire_error: Option<String> = None;

        let graph = &mut self.graph;
        let selected = &mut self.selected_node;
        let drag_wire = &mut self.dragging_wire;
        let preview_enabled = &mut self.procgen.graph_preview_enabled;

        egui::Window::new("Pipeline Graph")
            .default_pos([240.0, 80.0])
            .default_size([960.0, 600.0])
            .min_size([520.0, 340.0])
            .resizable(true)
            .collapsible(true)
            .open(&mut self.state.show_graph)
            .show(ctx, |ui| {
                // ===== Top toolbar =====
                ui.horizontal(|ui| {
                    if ui
                        .button("▶ Run Pipeline")
                        .on_hover_text("Evaluate the graph and apply (undo-able)")
                        .clicked()
                    {
                        run = true;
                    }
                    ui.checkbox(preview_enabled, "Preview").on_hover_text(
                        "Show a translucent overlay of the graph's output \
                         (debounced ~150ms)",
                    );
                    ui.separator();
                    ui.menu_button("+ Add Node", |ui| {
                        for k in node_menu_options() {
                            if ui.button(k.0).clicked() {
                                add_kind = Some((k.1)());
                                ui.close_menu();
                            }
                            if k.2 {
                                ui.separator();
                            }
                        }
                    });
                    if ui.button("Auto Layout").on_hover_text("Re-grid all nodes").clicked()
                    {
                        auto_layout = true;
                    }
                    ui.separator();
                    ui.label(format!("Nodes: {}", graph.nodes.len()));
                });
                ui.separator();

                // ===== Split: canvas (left) + sidebar (right) =====
                let avail = ui.available_size();
                let sidebar_w = 280.0_f32.min(avail.x * 0.4).max(220.0);
                let canvas_w = (avail.x - sidebar_w - 12.0).max(200.0);

                ui.horizontal_top(|ui| {
                    // ---- Canvas ----
                    ui.allocate_ui_with_layout(
                        egui::vec2(canvas_w, avail.y),
                        egui::Layout::top_down(egui::Align::Min),
                        |ui| {
                            graph_canvas(
                                ui,
                                graph,
                                selected,
                                drag_wire,
                                &mut delete_id,
                                &mut wire_action,
                                &mut wire_error,
                            );
                        },
                    );
                    ui.separator();

                    // ---- Sidebar ----
                    ui.allocate_ui_with_layout(
                        egui::vec2(sidebar_w, avail.y),
                        egui::Layout::top_down(egui::Align::Min),
                        |ui| {
                            graph_sidebar(ui, graph, *selected);
                        },
                    );
                });
            });

        // ===== Apply deferred actions =====
        if let Some(id) = delete_id {
            graph.remove(id);
            if *selected == Some(id) {
                *selected = None;
            }
        }
        if let Some(kind) = add_kind {
            let id = graph.add(kind);
            *selected = Some(id);
        }
        if auto_layout {
            graph.relayout();
        }
        if let Some((target, slot, source)) = wire_action {
            if let Err(e) = graph.set_input(target, slot, source) {
                wire_error = Some(format!("{}", e));
            }
        }
        if let Some(msg) = wire_error {
            self.set_status(format!("Graph: {}", msg));
        }
        if run {
            self.state.request(UiAction::RunGraph);
        }
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

                        ui.label("6");
                        ui.label("Line shape");
                        ui.end_row();

                        ui.label("7");
                        ui.label("Box shape");
                        ui.end_row();

                        ui.label("8");
                        ui.label("Sphere shape");
                        ui.end_row();

                        ui.label("9");
                        ui.label("Cylinder shape");
                        ui.end_row();

                        ui.label("0");
                        ui.label("Box select tool");
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
                        ui.heading("Selection");
                        ui.end_row();

                        ui.label("Drag in selection");
                        ui.label("Move (single SetVoxels Command)");
                        ui.end_row();

                        ui.label("Drag outside");
                        ui.label("Create new selection");
                        ui.end_row();

                        ui.label("Ctrl+C / Ctrl+X");
                        ui.label("Copy / Cut non-air voxels");
                        ui.end_row();

                        ui.label("Ctrl+V");
                        ui.label("Paste at selection origin (or cursor)");
                        ui.end_row();

                        ui.label("Ctrl+Shift+V");
                        ui.label("Paste at cursor cell");
                        ui.end_row();

                        ui.label("Del");
                        ui.label("Delete non-air voxels in selection");
                        ui.end_row();

                        ui.label("Ctrl+A");
                        ui.label("Select all (AABB of all solid voxels)");
                        ui.end_row();

                        ui.label("Esc / Ctrl+D");
                        ui.label("Deselect");
                        ui.end_row();

                        ui.label("Arrows");
                        ui.label("Nudge selection on X / Z (Shift × 10)");
                        ui.end_row();

                        ui.label("Ctrl + Up/Down");
                        ui.label("Nudge selection on Y axis");
                        ui.end_row();

                        ui.end_row();
                        ui.heading("Camera");
                        ui.end_row();

                        ui.label("WASD");
                        ui.label("Move camera");
                        ui.end_row();

                        ui.label("Q");
                        ui.label("Move up");
                        ui.end_row();

                        ui.label("E");
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

    fn show_about_dialog(&mut self, ctx: &Context) {
        egui::Window::new("About Voxelith")
            .default_pos([ctx.screen_rect().width() / 2.0 - 150.0, ctx.screen_rect().height() / 2.0 - 100.0])
            .resizable(false)
            .collapsible(false)
            .open(&mut self.state.show_about)
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.heading("Voxelith");
                    ui.add_space(8.0);
                    ui.label(format!("Version {}", env!("CARGO_PKG_VERSION")));
                    ui.add_space(16.0);
                    ui.label("Procedural-first voxel asset creation tool");
                    ui.add_space(8.0);
                    ui.label("Built with Rust, wgpu, and egui");
                    ui.add_space(16.0);
                    ui.separator();
                    ui.add_space(8.0);
                    ui.label("MIT License");
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
                // Tool name highlighted: easy to miss in the previous flat
                // style — users have ended up confused about which tool is
                // active (especially Fill / Eyedropper, which behave
                // very differently from the brush tools).
                ui.label(
                    egui::RichText::new(format!(
                        "Tool: {}",
                        editor.current_tool.name()
                    ))
                    .strong()
                    .color(egui::Color32::LIGHT_BLUE),
                );
                ui.separator();
                ui.label(format!("Brush: {}px", editor.brush_size));
                if editor.symmetry.any() {
                    ui.separator();
                    let mut axes = String::new();
                    if editor.symmetry.x { axes.push('X'); }
                    if editor.symmetry.y { axes.push('Y'); }
                    if editor.symmetry.z { axes.push('Z'); }
                    ui.label(
                        egui::RichText::new(format!("Sym: {}", axes))
                            .color(egui::Color32::LIGHT_YELLOW),
                    );
                }
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
                if let Some(sel) = editor.selection {
                    let (w, h, d) = sel.size();
                    ui.separator();
                    ui.label(
                        egui::RichText::new(format!(
                            "Sel: {}×{}×{} ({} cells)",
                            w,
                            h,
                            d,
                            sel.cell_count()
                        ))
                        .color(egui::Color32::from_rgb(255, 230, 60)),
                    );
                }

                // Right-aligned viewport / preview info.
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
                    if self.procgen.preview_enabled
                        || self.procgen.graph_preview_enabled
                    {
                        ui.label(
                            egui::RichText::new("● Preview")
                                .color(egui::Color32::LIGHT_GREEN),
                        );
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

// ---- Procgen panel parameter editors ---------------------------------
//
// Free functions so the procgen panel's borrow on `self.procgen` can
// dispatch to the right editor without involving `&mut self`. They take
// only the generator's parameter struct.

fn terrain_params_ui(ui: &mut egui::Ui, t: &mut PerlinTerrain) {
    ui.heading(GeneratorChoice::Terrain.label());
    ui.add_space(4.0);

    egui::Grid::new("terrain_params")
        .num_columns(2)
        .spacing([10.0, 4.0])
        .show(ui, |ui| {
            ui.label("Seed");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut t.seed).speed(1.0));
                if ui
                    .button("Rand")
                    .on_hover_text("Randomize seed")
                    .clicked()
                {
                    t.seed = rand::random();
                }
            });
            ui.end_row();

            ui.label("Width");
            ui.add(egui::Slider::new(&mut t.width, 8..=256));
            ui.end_row();

            ui.label("Depth");
            ui.add(egui::Slider::new(&mut t.depth, 8..=256));
            ui.end_row();

            ui.label("Min Y");
            ui.add(egui::Slider::new(&mut t.min_height, -64..=64));
            ui.end_row();

            ui.label("Max Y");
            ui.add(egui::Slider::new(&mut t.max_height, -64..=128));
            ui.end_row();

            ui.label("Frequency");
            ui.add(
                egui::Slider::new(&mut t.frequency, 0.005..=0.5)
                    .logarithmic(true),
            );
            ui.end_row();

            ui.label("Octaves");
            ui.add(egui::Slider::new(&mut t.octaves, 1..=8));
            ui.end_row();
        });

    ui.label(format!(
        "{} × {} × {}",
        t.width,
        t.depth,
        (t.max_height - t.min_height).max(0)
    ));
}

fn tree_params_ui(ui: &mut egui::Ui, t: &mut LSystemTree) {
    ui.heading(GeneratorChoice::Tree.label());
    ui.add_space(4.0);

    egui::Grid::new("tree_params")
        .num_columns(2)
        .spacing([10.0, 4.0])
        .show(ui, |ui| {
            ui.label("Seed");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut t.seed).speed(1.0));
                if ui
                    .button("Rand")
                    .on_hover_text("Randomize seed")
                    .clicked()
                {
                    t.seed = rand::random();
                }
            });
            ui.end_row();

            ui.label("Iterations");
            ui.add(egui::Slider::new(&mut t.iterations, 1..=6));
            ui.end_row();

            ui.label("Angle (°)");
            ui.add(egui::Slider::new(&mut t.angle_deg, 5.0..=60.0));
            ui.end_row();

            ui.label("Init length");
            ui.add(egui::Slider::new(&mut t.initial_length, 1.0..=12.0));
            ui.end_row();

            ui.label("Length scale");
            ui.add(egui::Slider::new(&mut t.length_scale, 0.4..=1.0));
            ui.end_row();

            ui.label("Origin");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut t.origin.0).prefix("x:"));
                ui.add(egui::DragValue::new(&mut t.origin.1).prefix("y:"));
                ui.add(egui::DragValue::new(&mut t.origin.2).prefix("z:"));
            });
            ui.end_row();

            ui.label("Trunk");
            color_button_u8(ui, &mut t.trunk_color);
            ui.end_row();

            ui.label("Leaves");
            color_button_u8(ui, &mut t.leaf_color);
            ui.end_row();
        });
}

fn wfc_params_ui(ui: &mut egui::Ui, t: &mut WfcGenerator) {
    ui.heading(GeneratorChoice::Wfc.label());
    ui.add_space(4.0);

    egui::Grid::new("wfc_params")
        .num_columns(2)
        .spacing([10.0, 4.0])
        .show(ui, |ui| {
            ui.label("Seed");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut t.seed).speed(1.0));
                if ui
                    .button("Rand")
                    .on_hover_text("Randomize seed")
                    .clicked()
                {
                    t.seed = rand::random();
                }
            });
            ui.end_row();

            ui.label("Width (tiles)");
            ui.add(egui::Slider::new(&mut t.width, 2..=24));
            ui.end_row();

            ui.label("Depth (tiles)");
            ui.add(egui::Slider::new(&mut t.depth, 2..=24));
            ui.end_row();

            ui.label("Origin");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut t.origin.0).prefix("x:"));
                ui.add(egui::DragValue::new(&mut t.origin.1).prefix("y:"));
                ui.add(egui::DragValue::new(&mut t.origin.2).prefix("z:"));
            });
            ui.end_row();

            ui.label("Tileset");
            egui::ComboBox::from_id_salt("wfc_tileset")
                .selected_text(t.tileset.label())
                .show_ui(ui, |ui| {
                    for &option in WfcTileset::ALL {
                        ui.selectable_value(&mut t.tileset, option, option.label());
                    }
                });
            ui.end_row();
        });

    let s = crate::procgen::WFC_TILE_SIZE as i32;
    ui.label(format!(
        "≈ {} × {} × {} voxels",
        t.width as i32 * s,
        s,
        t.depth as i32 * s
    ));
}

// =============================================================
// Visual graph editor: layout constants + helpers
// =============================================================

const NODE_W: f32 = 168.0;
const NODE_H: f32 = 84.0;
const NODE_HEADER_H: f32 = 22.0;
const SOCKET_R: f32 = 6.0;
const SOCKET_HIT_R: f32 = SOCKET_R + 4.0;

/// Available node kinds in the "+ Add Node" menu.
/// Tuple is (label, factory, separator_after).
fn node_menu_options() -> Vec<(&'static str, fn() -> NodeKind, bool)> {
    vec![
        ("Source: Terrain", || NodeKind::Terrain(PerlinTerrain::default()), false),
        ("Source: Tree", || NodeKind::Tree(LSystemTree::default()), false),
        ("Source: WFC", || NodeKind::Wfc(WfcGenerator::default()), true),
        (
            "Translate",
            || NodeKind::Translate { input: None, dx: 0, dy: 0, dz: 0 },
            false,
        ),
        (
            "Filter",
            || NodeKind::Filter {
                input: None,
                predicate: FilterPredicate::default(),
            },
            false,
        ),
        (
            "Mask",
            || NodeKind::Mask {
                subject: None,
                mask: None,
                mode: MaskMode::default(),
            },
            false,
        ),
        (
            "Combine",
            || NodeKind::Combine {
                a: None,
                b: None,
                op: CombineOp::Union,
            },
            true,
        ),
        ("Output", || NodeKind::Output { input: None }, false),
    ]
}

/// Header tint per node kind — gives a quick visual key for source vs.
/// transform vs. sink.
fn node_header_color(kind: &NodeKind) -> egui::Color32 {
    match kind {
        NodeKind::Terrain(_) => egui::Color32::from_rgb(70, 110, 60),
        NodeKind::Tree(_) => egui::Color32::from_rgb(60, 100, 60),
        NodeKind::Wfc(_) => egui::Color32::from_rgb(100, 90, 50),
        NodeKind::Translate { .. } => egui::Color32::from_rgb(70, 80, 110),
        NodeKind::Filter { .. } => egui::Color32::from_rgb(80, 100, 110),
        NodeKind::Mask { .. } => egui::Color32::from_rgb(90, 110, 130),
        NodeKind::Combine { .. } => egui::Color32::from_rgb(110, 70, 110),
        NodeKind::Output { .. } => egui::Color32::from_rgb(120, 80, 60),
    }
}

/// One- or two-line summary shown under the header inside the node box.
fn node_summary(kind: &NodeKind) -> String {
    match kind {
        NodeKind::Terrain(t) => {
            format!("seed {} • {}×{}", t.seed, t.width, t.depth)
        }
        NodeKind::Tree(t) => {
            format!("seed {} • iter {}", t.seed, t.iterations)
        }
        NodeKind::Wfc(t) => {
            format!("seed {} • {}×{}", t.seed, t.width, t.depth)
        }
        NodeKind::Translate { dx, dy, dz, .. } => {
            format!("offset ({}, {}, {})", dx, dy, dz)
        }
        NodeKind::Filter { predicate, .. } => predicate.label(),
        NodeKind::Mask { mode, .. } => mode.label().to_string(),
        NodeKind::Combine { op, .. } => op.label().to_string(),
        NodeKind::Output { .. } => "pipeline result".to_string(),
    }
}

/// Screen-space bounding box of a node body.
fn node_screen_rect(canvas_min: egui::Pos2, node: &crate::procgen::GraphNode) -> egui::Rect {
    egui::Rect::from_min_size(
        canvas_min + egui::vec2(node.position[0], node.position[1]),
        egui::vec2(NODE_W, NODE_H),
    )
}

/// Center of an input socket in screen space. Combine nodes have
/// two inputs stacked vertically; everyone else has one centered.
fn input_socket_screen(
    canvas_min: egui::Pos2,
    node: &crate::procgen::GraphNode,
    slot: usize,
) -> egui::Pos2 {
    let body = node_screen_rect(canvas_min, node);
    match &node.kind {
        NodeKind::Combine { .. } => {
            let body_inner_top = body.min.y + NODE_HEADER_H + 14.0;
            let y = body_inner_top + slot as f32 * 22.0;
            egui::pos2(body.min.x, y)
        }
        _ => egui::pos2(body.min.x, body.center().y + 6.0),
    }
}

/// Center of a node's output socket (right edge).
fn output_socket_screen(
    canvas_min: egui::Pos2,
    node: &crate::procgen::GraphNode,
) -> egui::Pos2 {
    let body = node_screen_rect(canvas_min, node);
    egui::pos2(body.max.x, body.center().y + 6.0)
}

/// Sample a cubic Bezier at parameter `t ∈ [0, 1]`.
fn cubic_bezier_point(
    p0: egui::Pos2,
    p1: egui::Pos2,
    p2: egui::Pos2,
    p3: egui::Pos2,
    t: f32,
) -> egui::Pos2 {
    let omt = 1.0 - t;
    let omt2 = omt * omt;
    let omt3 = omt2 * omt;
    let t2 = t * t;
    let t3 = t2 * t;
    egui::pos2(
        omt3 * p0.x + 3.0 * omt2 * t * p1.x + 3.0 * omt * t2 * p2.x + t3 * p3.x,
        omt3 * p0.y + 3.0 * omt2 * t * p1.y + 3.0 * omt * t2 * p2.y + t3 * p3.y,
    )
}

/// Draw a wire from `from` (output socket) to `to` (input socket) as a
/// horizontally-bowed cubic Bezier — the standard look for node-graph
/// editors. Tessellated to a polyline so we don't depend on egui's
/// CubicBezierShape API across versions.
fn paint_wire(
    painter: &egui::Painter,
    from: egui::Pos2,
    to: egui::Pos2,
    color: egui::Color32,
) {
    let dx = (to.x - from.x).abs().max(40.0);
    let c1 = egui::pos2(from.x + dx * 0.5, from.y);
    let c2 = egui::pos2(to.x - dx * 0.5, to.y);

    const SEGMENTS: usize = 24;
    let mut pts = Vec::with_capacity(SEGMENTS + 1);
    for i in 0..=SEGMENTS {
        let t = i as f32 / SEGMENTS as f32;
        pts.push(cubic_bezier_point(from, c1, c2, to, t));
    }
    painter.add(egui::Shape::line(pts, egui::Stroke::new(2.0, color)));
}

/// Visual graph editor canvas. Renders nodes + wires, handles
/// click-select, body-drag, and socket-drag wire creation. Mutations
/// to the graph (input slot changes, deletion) are deferred via the
/// out-params so the caller can apply them outside of the borrow.
fn graph_canvas(
    ui: &mut egui::Ui,
    graph: &mut PipelineGraph,
    selected: &mut Option<NodeId>,
    drag_wire: &mut Option<NodeId>,
    delete_id: &mut Option<NodeId>,
    wire_action: &mut Option<(NodeId, usize, Option<NodeId>)>,
    wire_error: &mut Option<String>,
) {
    let avail = ui.available_size();
    let (canvas_rect, _bg) =
        ui.allocate_exact_size(avail, egui::Sense::hover());
    let painter = ui.painter_at(canvas_rect);

    // Background.
    painter.rect_filled(
        canvas_rect,
        0.0,
        egui::Color32::from_rgb(28, 28, 36),
    );

    // ===== Wires (drawn before nodes so they pass under boxes) =====
    for node in &graph.nodes {
        let in_count = PipelineGraph::input_count(&node.kind);
        for slot in 0..in_count {
            let input_id = match (slot, &node.kind) {
                (
                    0,
                    NodeKind::Translate { input, .. } | NodeKind::Output { input },
                ) => *input,
                (0, NodeKind::Combine { a, .. }) => *a,
                (1, NodeKind::Combine { b, .. }) => *b,
                _ => None,
            };
            let Some(src_id) = input_id else { continue };
            let Some(src) = graph.get(src_id) else { continue };
            let from = output_socket_screen(canvas_rect.min, src);
            let to = input_socket_screen(canvas_rect.min, node, slot);
            let highlighted = *selected == Some(node.id) || *selected == Some(src_id);
            let color = if highlighted {
                egui::Color32::from_rgb(180, 200, 255)
            } else {
                egui::Color32::from_rgb(140, 140, 160)
            };
            paint_wire(&painter, from, to, color);
        }
    }

    // ===== Live wire (while a socket-drag is active) =====
    if let Some(src_id) = *drag_wire {
        if let Some(src) = graph.get(src_id) {
            let from = output_socket_screen(canvas_rect.min, src);
            let to = ui
                .ctx()
                .input(|i| i.pointer.interact_pos())
                .unwrap_or(from);
            paint_wire(&painter, from, to, egui::Color32::YELLOW);
        }
    }

    // ===== Nodes =====
    // Two passes: first allocate all node body widgets so their drag
    // responses are registered, then draw + handle sockets. Splitting
    // keeps z-order predictable (sockets sit on top of body).
    //
    // First pass: register a click-and-drag interaction over each
    // node body so egui can route hover / click / drag events. We
    // capture the per-body response so the second pass can apply the
    // delta to the node's position without re-allocating.
    struct NodeFrame {
        body_resp: egui::Response,
        delta: egui::Vec2,
    }
    let mut frames: Vec<(NodeId, NodeFrame)> = Vec::with_capacity(graph.nodes.len());

    for node in &graph.nodes {
        let body = node_screen_rect(canvas_rect.min, node);
        let body_id = ui.id().with(("graph_node_body", node.id));
        let body_resp = ui.interact(body, body_id, egui::Sense::click_and_drag());
        let delta = if body_resp.dragged() {
            body_resp.drag_delta()
        } else {
            egui::Vec2::ZERO
        };
        frames.push((node.id, NodeFrame { body_resp, delta }));
    }

    // Apply body drags + clicks (mutates graph.position / selected).
    for (id, frame) in &frames {
        if frame.body_resp.clicked() {
            *selected = Some(*id);
        }
        if frame.body_resp.dragged() {
            *selected = Some(*id);
            if let Some(node) = graph.get_mut(*id) {
                node.position[0] += frame.delta.x;
                node.position[1] += frame.delta.y;
            }
        }
    }

    // Re-borrow `&graph.nodes` for visual + socket drawing. We use
    // the cached frames for body rects so mid-drag positions update
    // smoothly.
    let nodes_snapshot: Vec<crate::procgen::GraphNode> = graph.nodes.clone();
    for node in &nodes_snapshot {
        let body = node_screen_rect(canvas_rect.min, node);
        let is_selected = *selected == Some(node.id);

        // Body fill + outline.
        painter.rect_filled(body, 4.0, egui::Color32::from_rgb(50, 50, 60));
        let outline = if is_selected {
            egui::Stroke::new(2.0, egui::Color32::LIGHT_BLUE)
        } else {
            egui::Stroke::new(1.0, egui::Color32::from_gray(80))
        };
        painter.rect_stroke(body, 4.0, outline);

        // Header.
        let header = egui::Rect::from_min_max(
            body.min,
            egui::pos2(body.max.x, body.min.y + NODE_HEADER_H),
        );
        painter.rect_filled(header, 4.0, node_header_color(&node.kind));
        painter.text(
            header.min + egui::vec2(8.0, 3.0),
            egui::Align2::LEFT_TOP,
            format!("#{}  {}", node.id, node.kind.label()),
            egui::FontId::proportional(12.0),
            egui::Color32::WHITE,
        );

        // Delete × button (top-right corner of header).
        let close_size = 16.0;
        let close_rect = egui::Rect::from_min_size(
            egui::pos2(header.max.x - close_size - 2.0, header.min.y + 3.0),
            egui::vec2(close_size, close_size),
        );
        let close_id = ui.id().with(("graph_node_close", node.id));
        let close_resp =
            ui.interact(close_rect, close_id, egui::Sense::click());
        let close_color = if close_resp.hovered() {
            egui::Color32::from_rgb(255, 120, 120)
        } else {
            egui::Color32::from_gray(220)
        };
        painter.text(
            close_rect.center(),
            egui::Align2::CENTER_CENTER,
            "×",
            egui::FontId::proportional(14.0),
            close_color,
        );
        if close_resp.clicked() {
            *delete_id = Some(node.id);
        }

        // Summary text.
        painter.text(
            body.min + egui::vec2(8.0, NODE_HEADER_H + 6.0),
            egui::Align2::LEFT_TOP,
            node_summary(&node.kind),
            egui::FontId::proportional(11.0),
            egui::Color32::from_gray(200),
        );

        // Input sockets.
        for slot in 0..PipelineGraph::input_count(&node.kind) {
            let center = input_socket_screen(canvas_rect.min, node, slot);
            let hit_rect = egui::Rect::from_center_size(
                center,
                egui::vec2(SOCKET_HIT_R * 2.0, SOCKET_HIT_R * 2.0),
            );
            let in_id =
                ui.id().with(("graph_in_sock", node.id, slot));
            let in_resp = ui.interact(hit_rect, in_id, egui::Sense::hover());
            let hot = drag_wire.is_some() && in_resp.hovered();
            let color = if hot {
                egui::Color32::from_rgb(255, 230, 100)
            } else {
                egui::Color32::from_rgb(180, 180, 200)
            };
            painter.circle_filled(center, SOCKET_R, color);
            painter.circle_stroke(
                center,
                SOCKET_R,
                egui::Stroke::new(1.0, egui::Color32::BLACK),
            );
        }

        // Output socket.
        if PipelineGraph::has_output(&node.kind) {
            let center = output_socket_screen(canvas_rect.min, node);
            let hit_rect = egui::Rect::from_center_size(
                center,
                egui::vec2(SOCKET_HIT_R * 2.0, SOCKET_HIT_R * 2.0),
            );
            let out_id = ui.id().with(("graph_out_sock", node.id));
            let out_resp =
                ui.interact(hit_rect, out_id, egui::Sense::drag());
            painter.circle_filled(
                center,
                SOCKET_R,
                egui::Color32::from_rgb(220, 200, 100),
            );
            painter.circle_stroke(
                center,
                SOCKET_R,
                egui::Stroke::new(1.0, egui::Color32::BLACK),
            );
            if out_resp.drag_started() {
                *drag_wire = Some(node.id);
            }
            if out_resp.drag_stopped() && *drag_wire == Some(node.id) {
                // Hit-test cursor against every input socket.
                let p = ui.ctx().input(|i| i.pointer.interact_pos());
                let mut hit: Option<(NodeId, usize)> = None;
                if let Some(p) = p {
                    'outer: for target in &nodes_snapshot {
                        if target.id == node.id {
                            continue;
                        }
                        for slot in 0..PipelineGraph::input_count(&target.kind) {
                            let s = input_socket_screen(
                                canvas_rect.min,
                                target,
                                slot,
                            );
                            if (s - p).length() <= SOCKET_HIT_R {
                                hit = Some((target.id, slot));
                                break 'outer;
                            }
                        }
                    }
                }
                if let Some((target_id, slot)) = hit {
                    *wire_action = Some((target_id, slot, Some(node.id)));
                } else {
                    // Released into empty space → no-op (intentional cancel).
                    let _ = wire_error;
                }
                *drag_wire = None;
            }
        }
    }

    // If the user dragged and the cursor was released anywhere outside
    // the canvas (or pointer became unavailable), still cancel the
    // pending wire so we don't leave a stuck live wire on next frame.
    if drag_wire.is_some() && ui.ctx().input(|i| !i.pointer.any_down()) {
        *drag_wire = None;
    }

    // Empty-graph hint.
    if graph.nodes.is_empty() {
        painter.text(
            canvas_rect.center(),
            egui::Align2::CENTER_CENTER,
            "Empty pipeline.\nUse \"+ Add Node\" above.",
            egui::FontId::proportional(13.0),
            egui::Color32::from_gray(120),
        );
    }
}

/// Right-side parameter editor. Shows the selected node's params,
/// plus connection ComboBoxes (kept as a fallback to visual wiring,
/// useful for disconnecting / reading the current state).
fn graph_sidebar(
    ui: &mut egui::Ui,
    graph: &mut PipelineGraph,
    selected: Option<NodeId>,
) {
    ui.heading("Inspector");
    ui.add_space(4.0);

    let Some(id) = selected else {
        ui.label("Click a node in the canvas to edit its parameters.");
        return;
    };

    // Snapshot of node ids for input ComboBoxes (avoids holding an
    // immutable borrow on graph.nodes while we mutate one node below).
    let candidates: Vec<(NodeId, String)> = graph
        .nodes
        .iter()
        .map(|n| (n.id, format!("#{}: {}", n.id, n.kind.label())))
        .collect();

    let Some(node) = graph.get_mut(id) else {
        ui.label("(node not found)");
        return;
    };

    ui.label(
        egui::RichText::new(format!("#{}  {}", node.id, node.kind.label()))
            .strong(),
    );
    ui.separator();
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| match &mut node.kind {
            NodeKind::Terrain(t) => terrain_params_ui(ui, t),
            NodeKind::Tree(t) => tree_params_ui(ui, t),
            NodeKind::Wfc(t) => wfc_params_ui(ui, t),
            NodeKind::Translate { input, dx, dy, dz } => {
                input_slot(ui, "Input", input, &candidates, id);
                ui.horizontal(|ui| {
                    ui.label("Offset");
                    ui.add(egui::DragValue::new(dx).prefix("x:"));
                    ui.add(egui::DragValue::new(dy).prefix("y:"));
                    ui.add(egui::DragValue::new(dz).prefix("z:"));
                });
            }
            NodeKind::Filter { input, predicate } => {
                input_slot(ui, "Input", input, &candidates, id);
                filter_predicate_ui(ui, predicate, id);
            }
            NodeKind::Mask { subject, mask, mode } => {
                input_slot(ui, "Subject", subject, &candidates, id);
                input_slot(ui, "Mask", mask, &candidates, id);
                ui.horizontal(|ui| {
                    ui.label("Mode");
                    egui::ComboBox::from_id_salt(("mask_mode_sb", id))
                        .selected_text(mode.label())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                mode,
                                MaskMode::AboveColumn,
                                "Above column",
                            );
                            ui.selectable_value(
                                mode,
                                MaskMode::BelowColumn,
                                "Below column",
                            );
                        });
                });
                ui.label(
                    egui::RichText::new(
                        "Keeps subject voxels based on mask's column profile. \
                         Above-column → trees above terrain; Below-column → \
                         stalactites below ceilings.",
                    )
                    .small()
                    .weak(),
                );
            }
            NodeKind::Combine { a, b, op } => {
                input_slot(ui, "Input A", a, &candidates, id);
                input_slot(ui, "Input B", b, &candidates, id);
                ui.horizontal(|ui| {
                    ui.label("Operation");
                    egui::ComboBox::from_id_salt(("combine_op_sb", id))
                        .selected_text(op.label())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(op, CombineOp::Union, "Union");
                            ui.selectable_value(
                                op,
                                CombineOp::Difference,
                                "Difference",
                            );
                            ui.selectable_value(
                                op,
                                CombineOp::Intersect,
                                "Intersect",
                            );
                        });
                });
            }
            NodeKind::Output { input } => {
                input_slot(ui, "Input", input, &candidates, id);
            }
        });
}

/// Sidebar editor for a `Filter` node's predicate. Top combo switches
/// the predicate variant (resetting params to that variant's defaults
/// on change); the rows below it edit the current variant's params.
/// Variant switches discard the previous variant's params on purpose —
/// keeping a "remembered y threshold" across switches would surprise
/// the user more than help them.
fn filter_predicate_ui(
    ui: &mut egui::Ui,
    predicate: &mut FilterPredicate,
    node_id: NodeId,
) {
    // Variant selector. We compare via `matches!` rather than tag enums
    // to avoid carrying a parallel discriminator type.
    let cur_label = match predicate {
        FilterPredicate::YAbove(_) => "Y above",
        FilterPredicate::YBelow(_) => "Y below",
        FilterPredicate::MatchesColor(_) => "Color match",
        FilterPredicate::InsideBox { .. } => "Inside box",
    };
    ui.horizontal(|ui| {
        ui.label("Predicate");
        egui::ComboBox::from_id_salt(("filter_pred_kind", node_id))
            .selected_text(cur_label)
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(
                        matches!(predicate, FilterPredicate::YAbove(_)),
                        "Y above",
                    )
                    .clicked()
                    && !matches!(predicate, FilterPredicate::YAbove(_))
                {
                    *predicate = FilterPredicate::YAbove(0);
                }
                if ui
                    .selectable_label(
                        matches!(predicate, FilterPredicate::YBelow(_)),
                        "Y below",
                    )
                    .clicked()
                    && !matches!(predicate, FilterPredicate::YBelow(_))
                {
                    *predicate = FilterPredicate::YBelow(0);
                }
                if ui
                    .selectable_label(
                        matches!(predicate, FilterPredicate::MatchesColor(_)),
                        "Color match",
                    )
                    .clicked()
                    && !matches!(predicate, FilterPredicate::MatchesColor(_))
                {
                    *predicate = FilterPredicate::MatchesColor([200, 200, 200, 255]);
                }
                if ui
                    .selectable_label(
                        matches!(predicate, FilterPredicate::InsideBox { .. }),
                        "Inside box",
                    )
                    .clicked()
                    && !matches!(predicate, FilterPredicate::InsideBox { .. })
                {
                    *predicate = FilterPredicate::InsideBox {
                        min: (-8, 0, -8),
                        max: (8, 16, 8),
                    };
                }
            });
    });

    // Variant params.
    match predicate {
        FilterPredicate::YAbove(t) | FilterPredicate::YBelow(t) => {
            ui.horizontal(|ui| {
                ui.label("Threshold y");
                ui.add(egui::DragValue::new(t));
            });
        }
        FilterPredicate::MatchesColor(rgba) => {
            ui.horizontal(|ui| {
                ui.label("Color");
                let mut rgb = [rgba[0], rgba[1], rgba[2]];
                color_button_u8(ui, &mut rgb);
                rgba[0] = rgb[0];
                rgba[1] = rgb[1];
                rgba[2] = rgb[2];
                // Editor-placed voxels always have alpha 255; pin the
                // predicate's alpha to 255 too so a colour picked here
                // matches what's actually in the world.
                rgba[3] = 255;
            });
            ui.label(
                egui::RichText::new(
                    "Matches voxels with this exact RGB (alpha pinned to 255).",
                )
                .small()
                .weak(),
            );
        }
        FilterPredicate::InsideBox { min, max } => {
            ui.horizontal(|ui| {
                ui.label("Min");
                ui.add(egui::DragValue::new(&mut min.0).prefix("x:"));
                ui.add(egui::DragValue::new(&mut min.1).prefix("y:"));
                ui.add(egui::DragValue::new(&mut min.2).prefix("z:"));
            });
            ui.horizontal(|ui| {
                ui.label("Max");
                ui.add(egui::DragValue::new(&mut max.0).prefix("x:"));
                ui.add(egui::DragValue::new(&mut max.1).prefix("y:"));
                ui.add(egui::DragValue::new(&mut max.2).prefix("z:"));
            });
        }
    }
}

/// ComboBox for picking one of the graph's existing nodes as an input.
/// `self_id` is excluded from the list (a node can't connect to itself).
/// "(none)" is always an option for clearing the slot.
fn input_slot(
    ui: &mut egui::Ui,
    label: &str,
    input: &mut Option<NodeId>,
    candidates: &[(NodeId, String)],
    self_id: NodeId,
) {
    ui.horizontal(|ui| {
        ui.label(label);
        let current = match input {
            Some(id) => candidates
                .iter()
                .find(|(c, _)| *c == *id)
                .map(|(_, l)| l.as_str())
                .unwrap_or("(missing)"),
            None => "(none)",
        };
        egui::ComboBox::from_id_salt(("input_slot", label, self_id))
            .selected_text(current)
            .show_ui(ui, |ui| {
                ui.selectable_value(input, None, "(none)");
                for (cid, clabel) in candidates {
                    if *cid == self_id {
                        continue;
                    }
                    ui.selectable_value(input, Some(*cid), clabel);
                }
            });
    });
}

fn color_button_u8(ui: &mut egui::Ui, color: &mut [u8; 3]) {
    let mut f = [
        color[0] as f32 / 255.0,
        color[1] as f32 / 255.0,
        color[2] as f32 / 255.0,
    ];
    if ui.color_edit_button_rgb(&mut f).changed() {
        color[0] = (f[0] * 255.0).round() as u8;
        color[1] = (f[1] * 255.0).round() as u8;
        color[2] = (f[2] * 255.0).round() as u8;
    }
}
