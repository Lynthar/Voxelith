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

mod ai_actions;
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
use std::time::{Duration, Instant};

use rayon::prelude::*;
use winit::{keyboard::ModifiersState, window::Window};

use std::collections::HashSet;

use voxelith::{
    ai::{AiJobState, AiProvider, AiRuntime, FalHunyuanProvider, JobEvent, JobHandle},
    core::{Voxel, World},
    editor::{
        box_voxels, cylinder_voxels, line_voxels, sphere_voxels, BrushTool, Clipboard, Editor,
        EditorTool, RaycastHit, Selection, SymmetryAxes, Tool,
    },
    mesh::{patch_to_mesh, GreedyMesher, Mesher},
    prefs::{EditorPrefs, PanelVisibility, Prefs, WindowPrefs},
    render::Renderer,
    ui::{RenderStats, Ui},
};

use preview::PreviewState;

/// Alpha applied to the brush hover overlay. Higher than the procgen
/// preview (0.5) so the brush hint stays legible against existing
/// voxels of similar color.
const BRUSH_PREVIEW_ALPHA: f32 = 0.75;

/// Alpha applied to the move-drag voxel ghost — the translucent copy
/// of a selection's content that follows the cursor while it's being
/// relocated. A touch lighter than the brush hint (0.75) so it reads
/// as "in transit" rather than already placed, while staying clearly
/// visible against the voxels it slides over.
const MOVE_GHOST_ALPHA: f32 = 0.55;

/// How often `tick_autosave` writes the crash-recovery file while there
/// are unsaved changes. Long enough that saving a big world doesn't
/// hitch editing, short enough that a crash loses little work.
const AUTOSAVE_INTERVAL: Duration = Duration::from_secs(60);

/// Inclusive AABB `(min, max)` enclosing a set of cell positions, or
/// `None` for an empty set. Used to remember a generation's footprint
/// for the "Frame Generated" camera action.
pub(super) fn bounds_of(
    positions: impl IntoIterator<Item = (i32, i32, i32)>,
) -> Option<((i32, i32, i32), (i32, i32, i32))> {
    let mut it = positions.into_iter();
    let first = it.next()?;
    let (mut min, mut max) = (first, first);
    for p in it {
        min = (min.0.min(p.0), min.1.min(p.1), min.2.min(p.2));
        max = (max.0.max(p.0), max.1.max(p.1), max.2.max(p.2));
    }
    Some((min, max))
}

/// Main application state.
pub struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    egui_state: Option<egui_winit::State>,
    egui_renderer: Option<egui_wgpu::Renderer>,

    world: World,
    mesher: GreedyMesher,
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
    /// `(active cell, tool, brush color, brush size, symmetry, shape
    /// drag key)`. The "active cell" is `hover.voxel_pos` for
    /// brush tools and `hover.adjacent_pos` for shape tools (so
    /// shapes lock to the ground-plane fallback when the world is
    /// empty). The trailing `Option<ShapeDragKey>` carries the
    /// shape drag's enough-to-detect-change snapshot during a
    /// Footprint or Height phase.
    last_brush_preview_key: Option<(
        (i32, i32, i32),
        Tool,
        Voxel,
        u8,
        SymmetryAxes,
        Option<ShapeDragKey>,
    )>,

    /// In-progress shape drag (Line / Box / Sphere / Cylinder).
    /// Two-phase: Footprint while the left button is held (cursor
    /// drags on a locked plane defining W×D), then Height after
    /// release (cursor's vertical screen-space delta defines H along
    /// the plane normal). A second click commits; Esc cancels.
    /// Replaces the prior single-anchor `shape_drag_anchor` so the
    /// 3D-bbox-from-two-raycast-points "flat shape" bug is gone:
    /// W/D come from a 1:1 ray-vs-plane projection on the locked
    /// face, H is its own dedicated screen-Y axis. See vengi
    /// `ShapeBrush` for the same idea.
    pub(super) shape_drag: Option<ShapeDrag>,

    /// Set when the left button is held with the Select tool active
    /// **outside** any existing selection — the anchor cell of a new
    /// selection drag. Finalized into `editor.selection` by
    /// `commit_selection` on mouse-up.
    pub(super) selection_drag_anchor: Option<(i32, i32, i32)>,

    /// Set when the left button is held with the Select tool active
    /// **inside** an existing selection — the cell the press landed
    /// on. While set, every cursor move computes `current - anchor`
    /// as a translation delta, and `commit_selection` on mouse-up
    /// runs `move_selection(delta)` so the selection's voxels
    /// translate as one undoable Command.
    pub(super) selection_move_anchor: Option<(i32, i32, i32)>,

    /// Snapshot of the selection's non-air voxels (world-space)
    /// captured when a move drag begins, so the per-frame ghost just
    /// translates this set by the live delta instead of re-reading the
    /// world each time the cursor crosses a cell. Empty when no move
    /// drag is active; only read while `selection_move_anchor` is
    /// `Some` and overwritten at the next pickup, so leftover data
    /// between drags is harmless.
    pub(super) move_ghost_voxels: Vec<((i32, i32, i32), Voxel)>,

    /// Cache key for the selection wireframe so we don't rebuild the
    /// 24-vertex line buffer on every `CursorMoved` when the AABB
    /// hasn't changed.
    last_selection_box: Option<Selection>,

    /// Companion cache discriminant to `last_selection_box` for the
    /// move-drag voxel ghost: `Some(delta)` while ghosting, `None`
    /// otherwise. Load-bearing on the commit frame — the drag's final
    /// box equals the committed selection box, so a box-only cache
    /// would early-out and strand the ghost mesh on screen after the
    /// move lands.
    last_ghost_delta: Option<(i32, i32, i32)>,

    /// Locked face plane for drag-paint. Captured on the first
    /// `apply_tool` of a brush stroke (Place / Remove / Paint) and
    /// cleared on left-button release. While set,
    /// `update_raycast` ray-casts against this plane instead of the
    /// voxel world — without it, each new voxel written would shift
    /// the next ray-vs-voxels hit toward the camera and the stroke
    /// would "stack" along the view direction (vengi-style fix; see
    /// `vengi/AABBBrush.cpp`).
    pub(super) stroke_plane: Option<StrokePlane>,

    /// Voxel data captured by the most recent Copy / Cut. Pasting
    /// composites these onto the world (only the non-air voxels;
    /// see `Clipboard` docs). Not persisted across sessions —
    /// matches the convention in MagicaVoxel / Goxel / vengi.
    pub(super) clipboard: Option<Clipboard>,

    /// Persisted user preferences. Loaded at startup, dehydrated and
    /// written back on close. The recent-files MRU lives here.
    prefs: Prefs,

    /// Tokio multi-thread runtime running on its own background OS
    /// thread. AI jobs run there to keep the winit main thread free.
    /// Lives the entire app lifetime; no shutdown path needed.
    pub(super) ai_runtime: AiRuntime,
    /// Active provider. Phase 1 = mock; Phase 2 swaps in the real
    /// fal.ai client.
    pub(super) ai_provider: Arc<dyn AiProvider>,
    /// Latest job state. Mirrored into `Ui` each frame so the panel
    /// can render it. Mutated only by `tick_ai_job` (drained from
    /// the channel) and the action dispatcher (UI button → submit /
    /// cancel).
    pub(super) ai_job: AiJobState,
    /// Receiver half of the worker → main-thread event channel. Set
    /// when a job is in flight, cleared on terminal event.
    pub(super) ai_event_rx: Option<std::sync::mpsc::Receiver<JobEvent>>,
    /// Cancel token for the current job. Held alongside `ai_event_rx`;
    /// dropping it doesn't cancel — the worker checks the AtomicBool
    /// at safe points (see `MockProvider`).
    pub(super) ai_handle: Option<JobHandle>,
    /// Cached "is an API key in the keychain?" so the UI doesn't hit
    /// the keyring every frame. Refreshed by save / clear actions.
    pub(super) ai_has_key: bool,

    /// Voxel data changed since the last time anything was persisted
    /// (manual save *or* autosave). Set from `rebuild_all_meshes` (dirty
    /// chunks ⟺ a voxel changed) and cleared by save / autosave / open /
    /// new / import / initial-scene. Drives whether `tick_autosave`
    /// bothers to write.
    pub(super) unsaved_changes: bool,
    /// When the last autosave ran. `tick_autosave` rate-limits writes to
    /// `AUTOSAVE_INTERVAL`.
    pub(super) last_autosave: Instant,

    /// World-space AABB (inclusive cell coords) of the most recent
    /// procgen / graph / AI generation, powering the "Frame Generated"
    /// camera action. `None` until something is generated this session;
    /// set at each generation chokepoint. Not cleared on undo — framing
    /// stale bounds just frames where the geometry was, and the action
    /// guards on `None`.
    pub(super) last_generated_bounds: Option<((i32, i32, i32), (i32, i32, i32))>,
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
        editor.symmetry = SymmetryAxes {
            x: prefs.editor.symmetry[0],
            y: prefs.editor.symmetry[1],
            z: prefs.editor.symmetry[2],
        };
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
            mesher: GreedyMesher::new(),
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
            shape_drag: None,
            selection_drag_anchor: None,
            selection_move_anchor: None,
            move_ghost_voxels: Vec::new(),
            last_selection_box: None,
            last_ghost_delta: None,
            stroke_plane: None,
            clipboard: None,
            prefs,
            ai_runtime: AiRuntime::new(),
            ai_provider: Arc::new(FalHunyuanProvider::new()),
            ai_job: AiJobState::Idle,
            ai_event_rx: None,
            ai_handle: None,
            ai_has_key: voxelith::ai::has_api_key("fal_ai"),
            unsaved_changes: false,
            last_autosave: Instant::now(),
            last_generated_bounds: None,
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
            symmetry: [
                self.editor.symmetry.x,
                self.editor.symmetry.y,
                self.editor.symmetry.z,
            ],
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

/// Expand `cells` with every symmetry mirror combination, deduped.
/// `Symmetry off` returns `cells` unchanged so the common path skips
/// the HashSet allocation. Used by both the live shape preview and
/// the shape commit path so they always render the same set.
fn expand_with_symmetry(
    cells: Vec<(i32, i32, i32)>,
    symmetry: SymmetryAxes,
) -> Vec<(i32, i32, i32)> {
    if !symmetry.any() {
        return cells;
    }
    let mut out: HashSet<(i32, i32, i32)> = HashSet::new();
    for cell in cells {
        for m in symmetry.mirror_positions(cell) {
            out.insert(m);
        }
    }
    out.into_iter().collect()
}

/// Locked face plane captured at the start of a brush stroke. The
/// stroke's drag-paint stays on this plane until release, so paint
/// doesn't stack along the view direction as new voxels occlude the
/// cursor's ray-vs-voxels hit.
///
/// The plane is axis-aligned (face normal is one of ±X / ±Y / ±Z),
/// stored as `axis` (which axis is the normal) plus `sign` (which
/// face). `plane_coord` is the world-space position of the plane
/// along `axis`. `anchor_along_axis` is the locked value of
/// `adjacent_pos[axis]` — every paint cell in the stroke pins this
/// component, so Place fills along the face, Remove / Paint stay on
/// the same layer.
#[derive(Debug, Clone, Copy)]
pub(super) struct StrokePlane {
    pub axis: usize,
    pub sign: i32,
    pub plane_coord: f32,
    pub anchor_along_axis: i32,
}

/// Build a `StrokePlane` from a raycast hit. Returns `None` when
/// the hit's normal isn't axis-aligned (e.g. starting inside a
/// voxel produces `(0, 0, 0)`); the caller falls back to the
/// existing per-cell ray-vs-voxels path.
pub(super) fn build_stroke_plane(hit: &RaycastHit) -> Option<StrokePlane> {
    let (nx, ny, nz) = hit.normal;
    let (axis, sign) = if nx != 0 && ny == 0 && nz == 0 {
        (0_usize, nx)
    } else if nx == 0 && ny != 0 && nz == 0 {
        (1_usize, ny)
    } else if nx == 0 && ny == 0 && nz != 0 {
        (2_usize, nz)
    } else {
        return None;
    };
    let ap = [hit.adjacent_pos.0, hit.adjacent_pos.1, hit.adjacent_pos.2];
    // The plane is the face *between* `voxel_pos` and `adjacent_pos`.
    // For sign > 0 the plane sits at `adjacent_pos[axis]` (its near
    // face); for sign < 0 it sits at `adjacent_pos[axis] + 1`
    // (its far face). Either way, every cell painted on this plane
    // has `adjacent_pos[axis] == anchor_along_axis`.
    let plane_coord = if sign > 0 {
        ap[axis] as f32
    } else {
        (ap[axis] + 1) as f32
    };
    Some(StrokePlane {
        axis,
        sign,
        plane_coord,
        anchor_along_axis: ap[axis],
    })
}

/// Pixels of vertical cursor movement per voxel of shape height in
/// the second phase of a shape drag. Tuned empirically; 8 px feels
/// responsive at the default camera distance.
pub(super) const SHAPE_HEIGHT_PIXELS_PER_VOXEL: f32 = 8.0;

/// In-progress shape drag (anchor + locked plane + current phase).
/// First phase is Footprint (left button held, cursor on the locked
/// plane defines W × D). Second phase is Height (left released, the
/// cursor's vertical screen-space movement defines H along the
/// plane normal until a second click commits).
#[derive(Debug, Clone, Copy)]
pub(super) struct ShapeDrag {
    /// First-press hit's `adjacent_pos`. Sits on the locked plane,
    /// so `anchor[plane.axis] == plane.anchor_along_axis`.
    pub anchor: (i32, i32, i32),
    /// Locked face plane — same `StrokePlane` shape brush stroke
    /// uses. All cells in the footprint have their `axis` component
    /// pinned to this plane.
    pub plane: StrokePlane,
    pub phase: ShapePhase,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum ShapePhase {
    /// Left button held; cursor's plane-locked hit is the
    /// footprint's other corner.
    Footprint,
    /// Left button released; cursor's vertical screen movement
    /// defines extruded height along the plane normal. A second
    /// click commits.
    Height {
        /// Footprint's other corner at the moment the user
        /// released the button (locked from then on — only height
        /// changes during this phase).
        end_on_plane: (i32, i32, i32),
        /// Cursor's screen-Y at release. Height = `(release_y -
        /// cursor_y) / SHAPE_HEIGHT_PIXELS_PER_VOXEL` (clamped to
        /// ≥ 0 since the user can't extrude *into* the face).
        release_screen_y: f32,
    },
}

/// Reduced cache key for `update_brush_preview` — drops the f32
/// `release_screen_y` (uses quantized integer height instead) so
/// the key implements `Eq` for the existing tuple-comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ShapeDragKey {
    Footprint {
        anchor: (i32, i32, i32),
        /// Current cursor's plane-locked cell. Without this in the
        /// key, dragging the cursor across cells in Footprint phase
        /// wouldn't invalidate the cache and the preview would
        /// freeze on the first cell.
        end_cell: (i32, i32, i32),
    },
    Height {
        anchor: (i32, i32, i32),
        end_on_plane: (i32, i32, i32),
        height: i32,
    },
}

impl ShapeDrag {
    /// Build the cache key for `update_brush_preview`. `hovered_cell`
    /// is the cursor's current plane-locked `adjacent_pos` (used in
    /// Footprint phase only; `None` falls back to anchor).
    pub fn cache_key(
        &self,
        cursor_y: f32,
        hovered_cell: Option<(i32, i32, i32)>,
    ) -> ShapeDragKey {
        match self.phase {
            ShapePhase::Footprint => ShapeDragKey::Footprint {
                anchor: self.anchor,
                end_cell: hovered_cell.unwrap_or(self.anchor),
            },
            ShapePhase::Height {
                end_on_plane,
                release_screen_y,
            } => ShapeDragKey::Height {
                anchor: self.anchor,
                end_on_plane,
                height: shape_height_from_cursor(release_screen_y, cursor_y),
            },
        }
    }

    /// 3D end corner of the shape after extrusion. Only valid in
    /// `Height` phase — `Footprint` callers should use the cursor's
    /// plane-locked `hovered_voxel.adjacent_pos` directly.
    pub fn extruded_end(&self, cursor_y: f32) -> Option<(i32, i32, i32)> {
        let ShapePhase::Height {
            end_on_plane,
            release_screen_y,
        } = self.phase
        else {
            return None;
        };
        let h = shape_height_from_cursor(release_screen_y, cursor_y);
        let mut e = [end_on_plane.0, end_on_plane.1, end_on_plane.2];
        e[self.plane.axis] += self.plane.sign * h;
        Some((e[0], e[1], e[2]))
    }
}

/// Pure helper: `(release_y - cursor_y) / SHAPE_HEIGHT_PIXELS_PER_VOXEL`,
/// clamped at 0 (negative would extrude into the face the plane was
/// captured on, which is never what the user means).
pub(super) fn shape_height_from_cursor(release_y: f32, cursor_y: f32) -> i32 {
    let dy = release_y - cursor_y; // screen up → positive
    (dy / SHAPE_HEIGHT_PIXELS_PER_VOXEL).round().max(0.0) as i32
}

fn tool_from_index(idx: u8) -> Tool {
    match idx {
        0 => Tool::Place,
        1 => Tool::Remove,
        2 => Tool::Paint,
        3 => Tool::Eyedropper,
        4 => Tool::Fill,
        5 => Tool::Line,
        6 => Tool::Box,
        7 => Tool::Sphere,
        8 => Tool::Cylinder,
        9 => Tool::Select,
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
        Tool::Line => 5,
        Tool::Box => 6,
        Tool::Sphere => 7,
        Tool::Cylinder => 8,
        Tool::Select => 9,
    }
}

impl App {

    /// Initialize the application with a window.
    pub(super) fn init(&mut self, window: Window) {
        let window = Arc::new(window);
        // Default cursor_pos to the screen center so a zoom-to-cursor
        // scroll BEFORE any cursor movement anchors at the screen
        // center (≈ camera target) instead of the (0,0) top-left
        // corner — the latter would shift the orbit pivot toward the
        // top-left of the world on the first scroll, which is
        // surprising. CursorMoved overwrites this on the first real
        // mouse move.
        let physical = window.inner_size();
        self.cursor_pos = (
            physical.width as f32 / 2.0,
            physical.height as f32 / 2.0,
        );
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

        // Always start on the default scene so the first frame has
        // something to draw, then defer any crash-recovery PROMPT to the
        // first `RedrawRequested`. Showing a native modal (rfd) here —
        // inside winit's `resumed` callback — exits the process with
        // code 1 on Windows (no Rust panic; confirmed it's the modal's
        // timing, not the file or its loading). By the first frame the
        // event loop is running and the window has presented, so the
        // dialog behaves like the in-loop file dialogs that already work.
        self.create_initial_scene();
        self.unsaved_changes = false;
        // If a crash-recovery autosave is on disk, the last session
        // didn't exit cleanly (a clean exit deletes it) — raise the
        // in-app recovery prompt. The default scene is already up behind
        // it. The prompt is egui, NOT a native `rfd::MessageDialog`:
        // showing one of those exits the process on this winit + wgpu
        // setup (it was the real cause of the "autosave bricks startup"
        // crash, not the file). See `Ui::show_recovery_prompt` and the
        // `RecoverAutosave` / `DiscardAutosave` actions.
        if Self::autosave_path().is_some_and(|p| p.exists()) {
            self.ui.state.show_recovery_prompt = true;
        }
    }

    /// Create the initial test scene shown on startup.
    fn create_initial_scene(&mut self) {
        self.world.create_test_cube((0, 8, 0), 4);
        self.world.create_test_ground(20, 2);
        self.rebuild_all_meshes();
        // Anchor the orbit pivot on the actual scene rather than the
        // hardcoded (0,0,0) target from `Camera::new`. Without this,
        // middle-mouse orbit circles a point underneath the model and
        // the visible cube swings through a wide arc each rotation.
        self.recenter_camera_on_scene();
    }

    /// Path of the crash-recovery autosave, next to `prefs.ron` in the
    /// platform config dir. `None` if the OS exposes no config dir.
    fn autosave_path() -> Option<PathBuf> {
        Prefs::config_path()
            .and_then(|p| p.parent().map(|d| d.join("autosave.vxlt")))
    }

    /// Per-frame autosave tick. Cheap when idle (one bool + one elapsed
    /// check). Writes at most once per `AUTOSAVE_INTERVAL`, and only when
    /// there are unsaved changes to a non-empty world. Clears
    /// `unsaved_changes` on a successful write so we don't rewrite an
    /// unchanged world every interval; a failed write is logged and
    /// retried next interval.
    pub(super) fn tick_autosave(&mut self) {
        if !self.unsaved_changes || self.last_autosave.elapsed() < AUTOSAVE_INTERVAL {
            return;
        }
        // Don't autosave (or offer to recover) an empty scene — e.g. just
        // after Clear All. Reset the timer so we don't re-check every frame.
        if self.world.scene_center().is_none() {
            self.unsaved_changes = false;
            self.last_autosave = Instant::now();
            return;
        }
        let Some(path) = Self::autosave_path() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let state = self.current_editor_state();
        // Atomic write: serialize to a temp file, then rename it over the
        // real autosave. A crash mid-write then leaves at most a stale
        // `autosave.tmp`, never a half-written `autosave.vxlt` — so
        // recovery always loads a COMPLETE last state. `fs::rename`
        // replaces the destination on Windows (MoveFileEx) as on POSIX,
        // and both files share the dir so it's a same-volume move.
        let tmp = path.with_extension("tmp");
        let result = voxelith::io::save_world_with_state(&self.world, state, &tmp)
            .and_then(|()| std::fs::rename(&tmp, &path).map_err(Into::into));
        match result {
            Ok(()) => {
                log::info!("Autosaved to {}", path.display());
                self.unsaved_changes = false;
            }
            Err(e) => {
                log::warn!("Autosave failed: {}", e);
                let _ = std::fs::remove_file(&tmp); // drop a partial temp
            }
        }
        self.last_autosave = Instant::now();
    }

    /// Remove the crash-recovery autosave. Called on a clean exit (so the
    /// next launch starts fresh) and when the user declines recovery.
    pub(super) fn delete_autosave(&self) {
        if let Some(path) = Self::autosave_path() {
            if path.exists() {
                if let Err(e) = std::fs::remove_file(&path) {
                    log::warn!("Failed to remove autosave {}: {}", path.display(), e);
                }
            }
        }
    }

    /// Snap `camera.target` to the world's scene-center (AABB of all
    /// non-air voxels), then re-derive controller yaw / pitch /
    /// distance from the new pose. Camera position itself is
    /// untouched — only the orbit pivot moves, so the user's current
    /// view direction smoothly rotates onto the scene rather than
    /// jumping.
    ///
    /// No-op when the world is empty (nothing meaningful to focus on).
    pub(super) fn recenter_camera_on_scene(&mut self) {
        let Some(center) = self.world.scene_center() else { return };
        let Some(renderer) = &mut self.renderer else { return };
        renderer.camera.target = center;
        renderer
            .camera_controller
            .sync_orbit_state_from_camera(&renderer.camera);
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

        // Dirty chunks this frame ⟺ voxel data changed (a write marks its
        // chunk dirty; boundary writes also mark neighbors). This is the
        // single chokepoint every edit / generation / AI / paste funnels
        // through, so it's where we flag the document for autosave. The
        // load / new / initial-scene paths clear the flag again after
        // their own rebuild.
        self.unsaved_changes = true;

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

    /// Refresh the translucent brush/shape hover overlay. Called every
    /// frame; the cache key short-circuits when nothing meaningful
    /// changed so the cost is just a few field comparisons.
    ///
    /// Three preview modes share this overlay slot:
    /// 1. **Brush tools** (Place/Remove/Paint/Fill): brush sphere at
    ///    the hovered cell, expanded by symmetry mirrors.
    /// 2. **Shape tools, idle** (no drag): single-cell anchor hint at
    ///    `adjacent_pos` (the cell where the next press would anchor).
    /// 3. **Shape tools, dragging** (left held with anchor set): full
    ///    shape voxel set from anchor to current cell, plus mirrors.
    ///
    /// Eyedropper has no preview (its color != the sampled color would
    /// mislead).
    pub(super) fn update_brush_preview(&mut self) {
        let tool = self.editor.current_tool;

        // If the user switched away from a shape tool while a drag
        // was in progress (e.g. via the toolbar mid-Footprint),
        // drop the drag so the next tool's preview isn't haunted by
        // the orphaned state.
        if !tool.is_shape() && self.shape_drag.is_some() {
            self.shape_drag = None;
        }

        let symmetry = self.editor.symmetry;
        let color = self.editor.brush_color;
        let size = self.editor.brush_size;
        let cursor_y = self.cursor_pos.1;

        // Eyedropper and Select skip the brush-style hover overlay
        // entirely. Eyedropper would mislead (brush color != sampled
        // color); Select draws its own AABB wireframe.
        let show = !matches!(tool, Tool::Eyedropper | Tool::Select);

        // Cache key. `cell` is hover-derived for non-shape tools and
        // for idle shapes; for an active ShapeDrag, `cell` is fixed
        // to `(0,0,0)` since the drag's own `cache_key` already
        // captures everything that affects the preview output
        // (including the current hovered cell in Footprint phase).
        let hovered_cell = self.editor.hovered_voxel.map(|h| h.adjacent_pos);
        let drag_key = self.shape_drag.map(|d| d.cache_key(cursor_y, hovered_cell));
        let key = if show {
            if drag_key.is_some() {
                Some((
                    (0, 0, 0),
                    tool,
                    color,
                    size,
                    symmetry,
                    drag_key,
                ))
            } else {
                self.editor.hovered_voxel.map(|h| {
                    let cell = if tool.is_shape() { h.adjacent_pos } else { h.voxel_pos };
                    (cell, tool, color, size, symmetry, None)
                })
            }
        } else {
            None
        };

        if key == self.last_brush_preview_key {
            return;
        }
        self.last_brush_preview_key = key;

        if !show {
            if let Some(r) = &mut self.renderer {
                r.clear_brush_preview();
            }
            return;
        }

        // Compute the preview cell list. Active shape drag has its
        // own dedicated branch (no dependency on `hovered_voxel` in
        // Height phase, since the cursor lives in screen space); all
        // other modes need a real hover.
        let positions: Vec<(i32, i32, i32)> = if let Some(drag) = self.shape_drag {
            let (anchor, end_3d) = match drag.phase {
                ShapePhase::Footprint => {
                    // Footprint: cursor's plane-locked hit is the
                    // other corner. No hit (cursor off-world) → no
                    // preview this frame.
                    let Some(hit) = self.editor.hovered_voxel else {
                        if let Some(r) = &mut self.renderer {
                            r.clear_brush_preview();
                        }
                        return;
                    };
                    (drag.anchor, hit.adjacent_pos)
                }
                ShapePhase::Height { .. } => {
                    // Height: extrude end_on_plane along the plane
                    // normal by the cursor-Y delta.
                    let end_3d = drag.extruded_end(cursor_y).expect("Height phase");
                    (drag.anchor, end_3d)
                }
            };
            let raw = match tool {
                Tool::Line => line_voxels(anchor, end_3d),
                Tool::Box => box_voxels(anchor, end_3d),
                Tool::Sphere => sphere_voxels(anchor, end_3d),
                Tool::Cylinder => cylinder_voxels(anchor, end_3d),
                _ => Vec::new(),
            };
            expand_with_symmetry(raw, symmetry)
        } else if tool.is_shape() {
            // Idle shape tool: hint at the anchor cell. Need a hit.
            let Some(hit) = self.editor.hovered_voxel else {
                if let Some(r) = &mut self.renderer {
                    r.clear_brush_preview();
                }
                return;
            };
            expand_with_symmetry(vec![hit.adjacent_pos], symmetry)
        } else {
            // Brush tool: BrushTool handles symmetry internally.
            let Some(hit) = self.editor.hovered_voxel else {
                if let Some(r) = &mut self.renderer {
                    r.clear_brush_preview();
                }
                return;
            };
            let brush = BrushTool::new(tool);
            brush.preview_positions(&hit, size, symmetry)
        };

        if positions.is_empty() {
            if let Some(r) = &mut self.renderer {
                r.clear_brush_preview();
            }
            return;
        }

        let voxels: Vec<((i32, i32, i32), Voxel)> =
            positions.into_iter().map(|p| (p, color)).collect();

        let mesh = patch_to_mesh(&voxels, BRUSH_PREVIEW_ALPHA);
        if let Some(r) = &mut self.renderer {
            r.set_brush_preview_mesh(&mesh);
        }
    }

    /// Refresh the box-selection wireframe **and** the move-drag voxel
    /// ghost. Both overlays are driven from the same four states and
    /// share one cache gate:
    ///
    /// 1. **New-selection drag** (`selection_drag_anchor` set):
    ///    live AABB from anchor → current cell. No ghost.
    /// 2. **Move-selection drag** (`selection_move_anchor` set):
    ///    existing AABB translated by `current - anchor`, plus a
    ///    translucent ghost of the picked-up voxels at the same delta.
    /// 3. **Idle with a committed selection**: static AABB, no ghost.
    /// 4. **Nothing**: clear both slots.
    ///
    /// Cached against `(last_selection_box, last_ghost_delta)` so
    /// dragging inside the same cell doesn't rebuild either buffer.
    /// The delta half of the key is what clears the ghost on the
    /// commit frame, where the wireframe box alone is unchanged.
    pub(super) fn update_selection_visualization(&mut self) {
        // Resolve the wireframe box and, for a move drag, the live
        // translation delta the ghost follows.
        let (preview, ghost_delta) = if let Some(anchor) = self.selection_drag_anchor {
            // New-selection drag — anchor → current end cell.
            let box_ = self
                .editor
                .hovered_voxel
                .map(|hit| Selection::from_corners(anchor, Self::select_anchor_pos(&hit)));
            (box_, None)
        } else if let Some(move_anchor) = self.selection_move_anchor {
            // Move drag — existing selection translated by the cursor
            // delta. Falls back to the un-translated selection if
            // there's no current hover (cursor off-world); the user
            // sees the box stay put rather than vanish.
            match (self.editor.selection, self.editor.hovered_voxel) {
                (Some(sel), Some(hit)) => {
                    let cur = Self::select_anchor_pos(&hit);
                    let delta = (
                        cur.0 - move_anchor.0,
                        cur.1 - move_anchor.1,
                        cur.2 - move_anchor.2,
                    );
                    (Some(sel.translated(delta)), Some(delta))
                }
                _ => (self.editor.selection, Some((0, 0, 0))),
            }
        } else {
            (self.editor.selection, None)
        };

        if (preview, ghost_delta) == (self.last_selection_box, self.last_ghost_delta) {
            return;
        }
        self.last_selection_box = preview;
        self.last_ghost_delta = ghost_delta;

        // Build the translated ghost mesh (move drag only) before
        // borrowing the renderer, so reading `move_ghost_voxels`
        // doesn't tangle with the `&mut renderer` borrow.
        let ghost_mesh = match ghost_delta {
            Some(delta) if !self.move_ghost_voxels.is_empty() => {
                let voxels: Vec<((i32, i32, i32), Voxel)> = self
                    .move_ghost_voxels
                    .iter()
                    .map(|&((x, y, z), v)| ((x + delta.0, y + delta.1, z + delta.2), v))
                    .collect();
                Some(patch_to_mesh(&voxels, MOVE_GHOST_ALPHA))
            }
            _ => None,
        };

        if let Some(r) = &mut self.renderer {
            match preview {
                Some(sel) => r.set_selection_mesh(sel.min, sel.max),
                None => r.clear_selection(),
            }
            match &ghost_mesh {
                Some(mesh) => r.set_move_ghost_mesh(mesh),
                None => r.clear_move_ghost(),
            }
        }
    }

    /// Snapshot the selection's non-air voxels (world-space) at the
    /// start of a move drag, so the per-frame ghost just translates
    /// the captured set by the live delta rather than re-reading the
    /// world every time the cursor crosses a cell. Extracts the same
    /// content as `copy_selection_to_clipboard`, but keeps absolute
    /// positions since the ghost renders in world space.
    pub(super) fn begin_move_ghost(&mut self, sel: Selection) {
        self.move_ghost_voxels = sel
            .iter_cells()
            .filter_map(|(x, y, z)| {
                let v = self.world.get_voxel(x, y, z);
                (!v.is_air()).then_some(((x, y, z), v))
            })
            .collect();
    }

    /// Resolve the cell a Select-tool gesture should anchor at for a
    /// given raycast hit. Real-voxel hits select the hit cell itself
    /// (so clicking a tree trunk grabs the trunk); virtual-ground
    /// hits use `adjacent_pos` (the cell *on* the plane, not the
    /// `(x, -1, z)` ghost below it) — otherwise an empty-world drag
    /// would silently put the selection one cell underground.
    pub(super) fn select_anchor_pos(hit: &RaycastHit) -> (i32, i32, i32) {
        if hit.virtual_ground {
            hit.adjacent_pos
        } else {
            hit.voxel_pos
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
