//! Application state, event loop integration, and frame rendering.
//!
//! `App` owns every long-lived runtime resource (window, renderer, world,
//! editor, UI). The `winit` event loop drives it through the
//! `ApplicationHandler` impl in `handler.rs`. Behavior is split across
//! sibling submodules by responsibility:
//!
//! - `file_ops` — new/save/open/import/export
//! - `shapes`   — built-in sphere/pyramid generators
//! - `input`    — raycast, tool apply, keyboard shortcuts
//! - `ui_actions` — drains `UiAction`s queued by the egui layer
//! - `render`   — per-frame wgpu pass
//! - `handler`  — winit `ApplicationHandler`

mod file_ops;
mod handler;
mod input;
mod preview;
mod render;
mod shapes;
mod ui_actions;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;
use winit::{keyboard::ModifiersState, window::Window};

use voxelith::{
    core::{Voxel, World},
    editor::{BrushTool, Editor, EditorTool, Tool},
    mesh::{patch_to_mesh, Mesher, NaiveMesher},
    prefs::{EditorPrefs, PanelVisibility, Prefs, WindowPrefs},
    render::Renderer,
    ui::{RenderStats, Ui},
};

use preview::PreviewState;

/// Alpha applied to the brush hover overlay. Higher than the procgen
/// preview (0.5) so the brush hint stays legible against existing
/// voxels of similar color.
const BRUSH_PREVIEW_ALPHA: f32 = 0.75;

/// Main application state.
pub struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    egui_state: Option<egui_winit::State>,
    egui_renderer: Option<egui_wgpu::Renderer>,

    world: World,
    mesher: NaiveMesher,
    editor: Editor,
    ui: Ui,

    last_frame: Instant,
    frame_times: VecDeque<f32>,

    cursor_captured: bool,
    cursor_pos: (f32, f32),
    modifiers: ModifiersState,

    /// True between left-button press and release; gates drag-paint
    /// in `CursorMoved`.
    left_button_held: bool,
    /// Voxel position the most recent stroke step applied at, so we
    /// don't re-apply on every CursorMoved while the cursor sits on
    /// the same cell.
    last_stroke_voxel: Option<(i32, i32, i32)>,
    /// Screen-space position of the left-button press. Used as a
    /// dead-zone origin: drag-paint only kicks in once the cursor
    /// has moved past `DRAG_THRESHOLD_PX` pixels from here, so a
    /// single click with hand-tremor doesn't paint a streak.
    stroke_start_screen_pos: Option<(f32, f32)>,

    /// Current project file path (None = unsaved).
    project_path: Option<PathBuf>,

    /// Last grid settings (for detecting changes).
    last_grid_size: i32,
    last_grid_spacing: f32,

    /// Procgen preview state machine.
    preview: PreviewState,

    /// Cache key for the brush hover overlay so we don't regenerate
    /// its mesh on every CursorMoved when nothing meaningful changed.
    /// `(hovered voxel, tool, brush color, brush size)`.
    last_brush_preview_key: Option<((i32, i32, i32), Tool, Voxel, u8)>,

    /// Persisted user preferences. Loaded at startup, dehydrated and
    /// written back on close. The recent-files MRU lives here.
    prefs: Prefs,
}

impl App {
    pub fn new() -> Self {
        let prefs = Prefs::load();

        let mut editor = Editor::new();
        editor.brush_color = Voxel::from_rgba(
            prefs.editor.brush_color[0],
            prefs.editor.brush_color[1],
            prefs.editor.brush_color[2],
            prefs.editor.brush_color[3],
        );
        editor.brush_size = prefs.editor.brush_size.max(1);
        editor.current_tool = tool_from_index(prefs.editor.selected_tool);
        if !prefs.editor.palette.is_empty() {
            editor.palette = prefs
                .editor
                .palette
                .iter()
                .map(|c| Voxel::from_rgba(c[0], c[1], c[2], c[3]))
                .collect();
        }

        let mut ui = Ui::new();
        ui.state.show_stats = prefs.panels.show_stats;
        ui.state.show_tools = prefs.panels.show_tools;
        ui.state.show_palette = prefs.panels.show_palette;
        ui.state.show_viewport_settings = prefs.panels.show_viewport_settings;
        ui.state.show_procgen = prefs.panels.show_procgen;
        ui.state.show_graph = prefs.panels.show_graph;
        ui.viewport = prefs.viewport.clone();
        ui.procgen = prefs.procgen.clone();
        ui.graph = prefs.graph.clone();
        // Pre-position-field prefs deserialize every node at [0, 0].
        // Spread them out so the visual editor can see them.
        if ui.graph.all_at_origin() {
            ui.graph.relayout();
        }
        ui.recent_files = prefs.recent_files.clone();

        let last_grid_size = ui.viewport.grid_size;
        let last_grid_spacing = ui.viewport.grid_spacing;

        Self {
            window: None,
            renderer: None,
            egui_state: None,
            egui_renderer: None,
            world: World::new(),
            mesher: NaiveMesher::new(),
            editor,
            ui,
            last_frame: Instant::now(),
            frame_times: VecDeque::with_capacity(60),
            cursor_captured: false,
            cursor_pos: (0.0, 0.0),
            modifiers: ModifiersState::empty(),
            left_button_held: false,
            last_stroke_voxel: None,
            stroke_start_screen_pos: None,
            project_path: None,
            last_grid_size,
            last_grid_spacing,
            preview: PreviewState::new(),
            last_brush_preview_key: None,
            prefs,
        }
    }

    /// Initial window inner-size from prefs. Read by `handler::resumed`.
    ///
    /// Sanity-guarded: implausibly large values (older builds wrote
    /// physical pixels into the logical-size field, which then grew
    /// by scale_factor on every restart) fall back to a known-good
    /// default. The next `save_prefs` will overwrite the bad entry
    /// with a proper logical size.
    pub(super) fn initial_window_size(&self) -> (u32, u32) {
        let w = self.prefs.window.width;
        let h = self.prefs.window.height;
        if !(320..=2048).contains(&w) || !(240..=2048).contains(&h) {
            (1280, 720)
        } else {
            (w, h)
        }
    }

    /// Push the current path to the recent-files MRU. Called from
    /// file_ops after a successful open/save/import/export. Mirrors
    /// the updated list to `ui.recent_files` so the next frame's
    /// Open Recent menu reflects it.
    pub(super) fn touch_recent(&mut self, path: &std::path::Path) {
        self.prefs.touch_recent(path);
        self.ui.recent_files = self.prefs.recent_files.clone();
    }

    /// Snapshot live UI/editor/window state into `self.prefs`, then
    /// write the file. Called on app exit.
    pub(super) fn save_prefs(&mut self) {
        self.prefs.panels = PanelVisibility {
            show_stats: self.ui.state.show_stats,
            show_tools: self.ui.state.show_tools,
            show_palette: self.ui.state.show_palette,
            show_viewport_settings: self.ui.state.show_viewport_settings,
            show_procgen: self.ui.state.show_procgen,
            show_graph: self.ui.state.show_graph,
        };
        self.prefs.viewport = self.ui.viewport.clone();
        self.prefs.procgen = self.ui.procgen.clone();
        self.prefs.graph = self.ui.graph.clone();
        self.prefs.editor = EditorPrefs {
            brush_color: [
                self.editor.brush_color.r,
                self.editor.brush_color.g,
                self.editor.brush_color.b,
                self.editor.brush_color.a,
            ],
            brush_size: self.editor.brush_size,
            selected_tool: tool_to_index(self.editor.current_tool),
            palette: self
                .editor
                .palette
                .iter()
                .map(|v| [v.r, v.g, v.b, v.a])
                .collect(),
        };
        if let Some(window) = &self.window {
            // `inner_size()` returns physical pixels; `WindowPrefs` is
            // in logical pixels (matches how we restore via
            // `LogicalSize` in handler::resumed). Without this conversion
            // the window grows by `scale_factor` on every restart on
            // high-DPI displays, eventually larger than the monitor.
            let size = window.inner_size();
            let scale = window.scale_factor().max(0.1);
            let logical_w =
                ((size.width as f64 / scale).round() as u32).clamp(640, 4096);
            let logical_h =
                ((size.height as f64 / scale).round() as u32).clamp(480, 4096);
            self.prefs.window = WindowPrefs {
                width: logical_w,
                height: logical_h,
            };
        }
        if let Err(e) = self.prefs.save() {
            log::error!("Failed to save prefs: {}", e);
        }
    }

}

fn tool_from_index(idx: u8) -> Tool {
    match idx {
        0 => Tool::Place,
        1 => Tool::Remove,
        2 => Tool::Paint,
        3 => Tool::Eyedropper,
        4 => Tool::Fill,
        _ => Tool::Place,
    }
}

fn tool_to_index(t: Tool) -> u8 {
    match t {
        Tool::Place => 0,
        Tool::Remove => 1,
        Tool::Paint => 2,
        Tool::Eyedropper => 3,
        Tool::Fill => 4,
    }
}

impl App {

    /// Initialize the application with a window.
    pub(super) fn init(&mut self, window: Window) {
        let window = Arc::new(window);
        self.window = Some(window.clone());

        let renderer = pollster::block_on(Renderer::new(window.clone()))
            .expect("Failed to create renderer");

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx,
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        // egui is a 2D overlay — its render pass attaches no depth
        // texture, so its pipeline must not declare a depth format
        // either. Mismatch trips wgpu validation
        // ("Incompatible depth-stencil attachment format").
        let egui_renderer = egui_wgpu::Renderer::new(
            &renderer.device,
            renderer.config.format,
            None,
            1,
            false,
        );

        self.renderer = Some(renderer);
        self.egui_state = Some(egui_state);
        self.egui_renderer = Some(egui_renderer);

        self.create_initial_scene();
    }

    /// Create the initial test scene shown on startup.
    fn create_initial_scene(&mut self) {
        self.world.create_test_cube((0, 8, 0), 4);
        self.world.create_test_ground(20, 2);
        self.rebuild_all_meshes();
    }

    /// Rebuild meshes for all dirty chunks and upload them to the GPU.
    ///
    /// Mesh generation runs on rayon's thread pool. Uploads stay on
    /// the calling thread because wgpu device/queue handles aren't
    /// trivially shareable with workers and uploads are cheap
    /// relative to mesh construction.
    pub(super) fn rebuild_all_meshes(&mut self) {
        let Some(renderer) = &mut self.renderer else {
            return;
        };

        let dirty = self.world.dirty_chunks();
        if dirty.is_empty() {
            return;
        }

        // Concurrent reads only: mesher acquires read locks on the
        // dirty chunk + 6 neighbors. Multiple workers operating on
        // disjoint chunks share-read those neighbors fine.
        let mesher = &self.mesher;
        let world = &self.world;
        let meshes: Vec<_> = dirty
            .par_iter()
            .map(|&pos| mesher.generate(world, pos))
            .collect();

        for mesh in &meshes {
            renderer.upload_mesh(mesh);
        }

        self.world.clear_dirty_flags();
    }

    /// Refresh the translucent brush hover overlay. Called every
    /// frame; the cache key short-circuits when nothing meaningful
    /// changed so the cost is just a few field comparisons.
    ///
    /// Eyedropper / Fill don't show a preview — Eyedropper doesn't
    /// write voxels and Fill's footprint depends on a flood-fill
    /// traversal that we don't want to run per-frame.
    pub(super) fn update_brush_preview(&mut self) {
        let tool = self.editor.current_tool;
        // All tools that target a cell get a hover hint. Eyedropper
        // is excluded because brush_color != the sampled color (would
        // mislead). Fill gets one too — its preview marks the seed
        // cell, not the full flood region.
        let show = matches!(
            tool,
            Tool::Place | Tool::Remove | Tool::Paint | Tool::Fill
        );

        let key = if show {
            self.editor.hovered_voxel.map(|h| {
                (
                    h.voxel_pos,
                    tool,
                    self.editor.brush_color,
                    self.editor.brush_size,
                )
            })
        } else {
            None
        };

        if key == self.last_brush_preview_key {
            return;
        }
        self.last_brush_preview_key = key;

        let Some(hit) = self.editor.hovered_voxel.filter(|_| show) else {
            if let Some(r) = &mut self.renderer {
                r.clear_brush_preview();
            }
            return;
        };

        // Use the brush tool's own preview semantics: Place uses
        // adjacent_pos, Remove/Paint use the hovered cell.
        let brush = BrushTool::new(tool);
        let positions = brush.preview_positions(&hit, self.editor.brush_size);
        if positions.is_empty() {
            if let Some(r) = &mut self.renderer {
                r.clear_brush_preview();
            }
            return;
        }

        let color = self.editor.brush_color;
        let voxels: Vec<((i32, i32, i32), Voxel)> =
            positions.into_iter().map(|p| (p, color)).collect();

        let mesh = patch_to_mesh(&voxels, BRUSH_PREVIEW_ALPHA);
        if let Some(r) = &mut self.renderer {
            r.set_brush_preview_mesh(&mesh);
        }
    }

    /// Compute frame statistics for the UI overlay.
    pub(super) fn calculate_stats(&self) -> RenderStats {
        let avg_frame_time = if self.frame_times.is_empty() {
            16.67
        } else {
            self.frame_times.iter().sum::<f32>() / self.frame_times.len() as f32
        };

        let renderer = self.renderer.as_ref().unwrap();
        let camera_pos = renderer.camera.position;

        RenderStats {
            fps: 1000.0 / avg_frame_time,
            frame_time_ms: avg_frame_time,
            triangles: renderer.total_triangles(),
            chunks: self.world.chunk_count(),
            camera_pos: (camera_pos.x, camera_pos.y, camera_pos.z),
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}
