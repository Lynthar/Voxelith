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
//! - `agent_bridge` — serves an MCP client from this world
//! - `render`   — per-frame wgpu pass
//! - `handler`  — winit `ApplicationHandler`

mod agent_bridge;
mod file_ops;
mod handler;
mod hud;
mod input;
mod preview;
mod render;
mod runtime;
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
    core::{Voxel, World},
    editor::{
        box_voxels, cylinder_voxels, line_voxels, sphere_voxels, BrushTool, Clipboard, Editor,
        EditorTool, RaycastHit, Selection, SymmetryAxes, Tool,
    },
    mesh::{patch_to_mesh, GreedyMesher, Mesher},
    prefs::{EditorPrefs, Prefs, WindowPrefs},
    procgen::PipelineGraph,
    render::Renderer,
    ui::{RenderStats, Ui},
};

use agent_bridge::AgentBridgeState;
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

    /// `(milliseconds, chunks)` of the most recent non-empty
    /// dirty-chunk rebuild — mesh generation + GPU upload + dirty-flag
    /// clear, i.e. the cost a big edit adds to its frame. `None` until
    /// the first rebuild. Surfaced by the perf HUD via
    /// `calculate_stats`.
    last_rebuild: Option<(f32, usize)>,

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

    /// The in-editor MCP server and whatever it has parked awaiting
    /// approval. Off until the Agent panel starts it.
    agent: AgentBridgeState,

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

    /// Cache of the socket gizmo's geometry inputs — `(position,
    /// normal)` per socket — so `update_socket_visualization` rebuilds
    /// the line buffer only when sockets are placed / deleted / moved /
    /// loaded, not every frame. Names don't affect the gizmo, so
    /// renaming a socket doesn't invalidate this.
    last_socket_viz: Vec<([f32; 3], [f32; 3])>,

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
    /// thread, so the winit main thread never awaits. The agent
    /// bridge's HTTP server runs there. Lives the entire app lifetime;
    /// no shutdown path needed.
    pub(super) async_runtime: runtime::AsyncRuntime,

    /// Voxel data changed since the last time the user's *own* file was
    /// written. Set from `rebuild_all_meshes` (dirty chunks ⟺ a voxel
    /// changed) and cleared only by manual save / open / new / import /
    /// initial-scene. Drives the unsaved-changes guard on every path
    /// that would throw the scene away.
    ///
    /// Deliberately NOT cleared by autosave: the autosave is a crash net
    /// living in the config dir, and a clean exit deletes it. If it
    /// cleared this flag, the sequence "edit → autosave fires → close"
    /// would skip the prompt and then delete the only copy.
    pub(super) unsaved_changes: bool,
    /// Same signal, but scoped to the autosave timer, which *does* want
    /// "nothing changed since my last write" so it doesn't rewrite an
    /// identical world every interval.
    pub(super) autosave_pending: bool,
    /// When the last autosave ran. `tick_autosave` rate-limits writes to
    /// `AUTOSAVE_INTERVAL`.
    pub(super) last_autosave: Instant,

    /// Modification time of `project_path` as of the last time we wrote
    /// it or read it. `tick_disk_reload` compares the file against this
    /// to tell somebody else's write from our own — an agent driving the
    /// MCP server with `--checkpoint`, or a `voxelith exec --out` run.
    /// `None` whenever no project file is open.
    pub(super) watched_mtime: Option<std::time::SystemTime>,
    /// When the project file was last polled. See `DISK_POLL_INTERVAL`.
    pub(super) last_disk_poll: Instant,

    /// Action deferred while the unsaved-changes prompt is up. Answered
    /// by the `UnsavedGuard*` actions; see `App::guard_then`.
    pub(super) pending_guarded: Option<PendingAction>,
    /// Set when the app should shut down at the end of this frame. The
    /// actual `event_loop.exit()` happens in `handler`, which is the
    /// only place holding an `ActiveEventLoop`.
    pub(super) exit_requested: bool,

    /// World-space AABB (inclusive cell coords) of the most recent
    /// procgen / graph generation or GLB import, powering the "Frame Generated"
    /// camera action. `None` until something is generated this session;
    /// set at each generation chokepoint. Not cleared on undo — framing
    /// stale bounds just frames where the geometry was, and the action
    /// guards on `None`.
    pub(super) last_generated_bounds: Option<((i32, i32, i32), (i32, i32, i32))>,
}

/// An action that throws the current scene away, held until the
/// unsaved-changes prompt is answered. See `App::guard_then`.
pub(super) enum PendingAction {
    NewProject,
    /// File ▸ Open — the path picker hasn't run yet.
    OpenPicker,
    /// Open Recent — the path is already known.
    OpenPath(PathBuf),
    ImportVox,
    Exit,
    Generate(GenerateKind),
    /// Load the crash-recovery autosave over the current scene.
    ///
    /// Guarded like the rest because the recovery prompt is *non-modal*:
    /// the default scene is live behind it, so a few edits made there
    /// and then a click on Recover used to discard them without a word.
    RecoverAutosave,
}

impl PendingAction {
    /// What the prompt says is about to happen ("Open another project
    /// anyway?"). Written to fit after "…unsaved changes.".
    fn describe(&self) -> &'static str {
        match self {
            PendingAction::NewProject => "start a new project",
            PendingAction::OpenPicker | PendingAction::OpenPath(_) => {
                "open another project"
            }
            PendingAction::ImportVox => "import a .vox model",
            PendingAction::Exit => "quit",
            PendingAction::Generate(_) => "replace the scene",
            PendingAction::RecoverAutosave => "recover the auto-saved work",
        }
    }
}

/// Which built-in scene the Generate menu asked for. A plain marker so
/// the choice can sit in a `PendingAction` (a closure couldn't).
#[derive(Clone, Copy)]
pub(super) enum GenerateKind {
    TestCube,
    Ground,
    Sphere,
    Pyramid,
}

/// A stored `[r, g, b, a]` as a brush-ready voxel.
///
/// The alpha is dropped rather than restored, and that is the point:
/// every voxel that reaches the world is opaque, and the brush is one
/// step from the world. A stored alpha of 0 would hand the brush the
/// greedy mesher's "no visible face" sentinel — solid to every count,
/// invisible in every picture — and both files this reads from
/// (`prefs.ron`, `.vxlt`) are files something else can write.
pub(super) fn brush_from_stored(color: [u8; 4]) -> Voxel {
    Voxel::from_rgb(color[0], color[1], color[2])
}

impl App {
    pub fn new() -> Self {
        let mut prefs = Prefs::load();

        let mut editor = Editor::new();
        editor.brush_color = brush_from_stored(prefs.editor.brush_color);
        editor.brush_color.flags = prefs.editor.brush_flags;
        editor.brush_color.set_tint_zone(prefs.editor.brush_tint_zone);
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
                .map(|&c| brush_from_stored(c))
                .collect();
        }

        let mut ui = Ui::new();
        ui.state.panels = prefs.panels.clone();
        ui.viewport = prefs.viewport.clone();
        ui.procgen = prefs.procgen.clone();
        // The graph belongs to the project now, so a fresh session opens
        // with an empty one — except the first time a build that stores
        // it there meets a prefs file that still holds the user's graph.
        // That one gets carried in front of them, once: re-running the
        // migration every launch would drop the old graph on top of
        // whatever they had been editing since.
        if !prefs.graph_migrated && !prefs.graph.is_empty() {
            ui.graph = prefs.graph.to_graph();
            // Pre-position-field prefs deserialize every node at [0, 0].
            if ui.graph.all_at_origin() {
                ui.graph.relayout();
            }
            prefs.graph_migrated = true;
            ui.set_status(
                "Your pipeline graph now travels with the project — save this one to keep it",
            );
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
            last_rebuild: None,
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
            agent: AgentBridgeState::new(),
            last_brush_preview_key: None,
            shape_drag: None,
            selection_drag_anchor: None,
            selection_move_anchor: None,
            move_ghost_voxels: Vec::new(),
            last_selection_box: None,
            last_ghost_delta: None,
            last_socket_viz: Vec::new(),
            stroke_plane: None,
            clipboard: None,
            prefs,
            async_runtime: runtime::AsyncRuntime::new(),
            unsaved_changes: false,
            autosave_pending: false,
            last_autosave: Instant::now(),
            watched_mtime: None,
            last_disk_poll: Instant::now(),
            pending_guarded: None,
            exit_requested: false,
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
        // Range must match the clamp in `save_prefs` — otherwise a size
        // this side accepts but that side never writes (or the reverse)
        // makes a valid window silently reset. A 2560- or 3840-wide logical
        // window (a 2K / 4K display at scale 1.0) used to be rejected here
        // (old max 2048) yet saved fine, so it never restored (#9).
        if !(640..=4096).contains(&w) || !(480..=4096).contains(&h) {
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
        self.prefs.panels = self.ui.state.panels.clone();
        self.prefs.viewport = self.ui.viewport.clone();
        self.prefs.procgen = self.ui.procgen.clone();
        // `prefs.graph` is deliberately *not* written back: the live
        // graph goes into the project file now, and the copy in prefs is
        // a one-time migration source that keeps its old contents until
        // the field is removed a version from now. Only the flag moves.
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
            brush_flags: self.editor.brush_color.flags,
            brush_tint_zone: self.editor.brush_color.tint_zone(),
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
        10 => Tool::Socket,
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
        Tool::Socket => 10,
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
        self.autosave_pending = false;
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

    /// Reset every piece of session state that refers to the geometry
    /// of the scene being replaced.
    ///
    /// Every whole-scene replacement path — New, Open, Import, crash
    /// recovery, Clear All, Generate\* — must call this. They each used
    /// to open-code their own subset, and the parts they all forgot
    /// were the selection and the generated-bounds: project A's marquee
    /// stayed live over project B, so Delete / Ctrl+X / arrow-nudge hit
    /// B's voxels at A's coordinates, and Frame Generated flew off to
    /// where the *previous* scene's geometry had been.
    ///
    /// Callers that restore state from a file (open / recover) run this
    /// first and repopulate sockets afterwards.
    pub(super) fn reset_scene_session_state(&mut self) {
        self.editor.history.clear();
        // Clears the selection plus the drag / move anchors and the
        // move ghost — see `App::deselect`.
        self.deselect();
        self.editor.sockets.clear();
        // The graph is document data like the sockets, so it goes with
        // the scene. Open / reload put the incoming file's graph back
        // immediately after this call; New Scene is the path that wants
        // the empty one this leaves behind.
        self.ui.graph = PipelineGraph::default();
        self.shape_drag = None;
        self.stroke_plane = None;
        self.last_stroke_voxel = None;
        self.last_generated_bounds = None;
        // A batch parked for approval was built against the world that
        // is being thrown away: its `old_voxel`s describe cells that no
        // longer exist, so committing it here would write a change list
        // whose undo restores a scene nobody was ever looking at. The
        // history-depth check in `accept_agent_batch` doesn't catch this
        // one — `history.clear()` above can land on the same (0, 0) the
        // batch was parked at.
        self.drop_pending_review_for_new_scene();
        if let Some(renderer) = &mut self.renderer {
            renderer.chunk_meshes.clear();
        }
    }

    /// Run `action`, or park it until the user answers the
    /// unsaved-changes prompt.
    ///
    /// Lives on `App` rather than in the UiAction dispatch because
    /// Ctrl+N / Ctrl+O call `new_project` / `open_project` directly from
    /// the key handler and would sail straight past a guard installed
    /// only in the queue.
    pub(super) fn guard_then(&mut self, action: PendingAction) {
        if !self.unsaved_changes {
            self.run_guarded(action);
            return;
        }
        self.ui.state.unsaved_prompt = Some(action.describe().to_string());
        self.pending_guarded = Some(action);
    }

    /// Perform a guarded action now that it's cleared to run.
    pub(super) fn run_guarded(&mut self, action: PendingAction) {
        match action {
            PendingAction::NewProject => self.new_project(),
            PendingAction::OpenPicker => self.open_project(),
            PendingAction::OpenPath(path) => self.do_open_project(path),
            PendingAction::ImportVox => self.import_vox(),
            PendingAction::Exit => self.exit_requested = true,
            PendingAction::Generate(kind) => self.generate_scene(kind),
            PendingAction::RecoverAutosave => self.recover_autosave(),
        }
    }

    /// Give up on the parked action — the user cancelled, or the save
    /// they asked for first didn't happen.
    ///
    /// Recovery is the one action whose entry point closes behind it:
    /// `Ui::show_recovery_prompt` clears its own flag the moment it
    /// dispatches, so dropping the action silently would leave the
    /// autosave on disk with no way back to it this session — and a
    /// clean exit later deletes it. Put the prompt back instead.
    pub(super) fn drop_pending_guarded(&mut self) {
        if let Some(PendingAction::RecoverAutosave) = self.pending_guarded.take() {
            self.ui.state.show_recovery_prompt = true;
        }
    }

    /// Path of the crash-recovery autosave, next to `prefs.ron` in the
    /// platform config dir. `None` if the OS exposes no config dir.
    fn autosave_path() -> Option<PathBuf> {
        Prefs::config_path()
            .and_then(|p| p.parent().map(|d| d.join("autosave.vxlt")))
    }

    /// Per-frame autosave tick. Cheap when idle (one bool + one elapsed
    /// check). Writes at most once per `AUTOSAVE_INTERVAL`, and only when
    /// there are changes to a non-empty world. Clears `autosave_pending`
    /// on a successful write so we don't rewrite an unchanged world every
    /// interval; a failed write is logged and retried next interval.
    /// `unsaved_changes` is left alone — see its doc comment.
    pub(super) fn tick_autosave(&mut self) {
        if !self.autosave_pending || self.last_autosave.elapsed() < AUTOSAVE_INTERVAL {
            return;
        }
        // Don't autosave (or offer to recover) an empty scene — e.g. just
        // after Clear All. Reset the timer so we don't re-check every frame.
        if self.world.scene_center().is_none() {
            self.autosave_pending = false;
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
                self.autosave_pending = false;
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

    /// The document differs from the user's file, and the autosave
    /// timer owes it a write.
    ///
    /// Two flags on purpose, and neither is the other's shorthand:
    /// `unsaved_changes` is "dirty relative to the file the user owns"
    /// and only a manual save / open / new clears it; `autosave_pending`
    /// is the timer's own bookkeeping. Autosave must never clear the
    /// first, or "edit → autosave fires → close" would skip the guard
    /// and then delete the only copy on the way out.
    ///
    /// Callers are the two ways a document changes: voxels (through the
    /// mesh rebuild below, which every edit funnels into) and the
    /// pipeline graph, which reaches no chunk and so has to say so.
    pub(super) fn mark_document_modified(&mut self) {
        self.unsaved_changes = true;
        self.autosave_pending = true;
    }

    /// Rebuild meshes for all dirty chunks and upload them to the GPU.
    ///
    /// Mesh generation runs on rayon's thread pool. Uploads stay on
    /// the calling thread because wgpu device/queue handles aren't
    /// trivially shareable with workers and uploads are cheap
    /// relative to mesh construction.
    pub(super) fn rebuild_all_meshes(&mut self) {
        if self.renderer.is_none() {
            return;
        }

        let dirty = self.world.dirty_chunks();
        if dirty.is_empty() {
            return;
        }
        let started = Instant::now();

        // Dirty chunks this frame ⟺ voxel data changed (a write marks its
        // chunk dirty; boundary writes also mark neighbors). This is the
        // single chokepoint every edit / generation / import / paste funnels
        // through, so it's where we flag the document as modified. The
        // load / new / initial-scene paths clear the flags again after
        // their own rebuild.
        self.mark_document_modified();

        // Concurrent reads only: mesher acquires read locks on the dirty
        // chunk + its 26 Moore neighbors (3³−1 — per-vertex AO samples
        // diagonal chunks, not just the 6 faces; see mesh::neighbors).
        // Multiple workers on disjoint chunks share-read those fine.
        let mesher = &self.mesher;
        let world = &self.world;
        let meshes: Vec<_> = dirty
            .par_iter()
            .map(|&pos| mesher.generate(world, pos))
            .collect();

        // Checked at the top; taken here so the flag write above isn't
        // holding a borrow of it.
        let Some(renderer) = &mut self.renderer else {
            return;
        };
        for mesh in &meshes {
            renderer.upload_mesh(mesh);
        }

        self.world.clear_dirty_flags();

        self.last_rebuild = Some((
            started.elapsed().as_secs_f32() * 1000.0,
            dirty.len(),
        ));
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

        // Eyedropper, Select, and Socket skip the brush-style hover
        // overlay entirely. Eyedropper would mislead (brush color !=
        // sampled color); Select draws its own AABB wireframe; Socket
        // draws its own gizmo overlay (`update_socket_visualization`).
        let show = !matches!(tool, Tool::Eyedropper | Tool::Select | Tool::Socket);

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
                    // Key on the cell the preview is actually DRAWN at:
                    // Place (like the shape tools) previews on `adjacent_pos`
                    // — the empty cell in front of the hit face — the rest
                    // on `voxel_pos`. Keying Place on `voxel_pos` left the
                    // preview stale when the cursor crossed to another face
                    // of the same voxel (adjacent_pos moved but the key
                    // didn't, so no regen).
                    let cell = if tool.is_shape() || matches!(tool, Tool::Place) {
                        h.adjacent_pos
                    } else {
                        h.voxel_pos
                    };
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
                Tool::Cylinder => cylinder_voxels(anchor, end_3d, Some(drag.plane.axis)),
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

    /// Refresh the socket gizmo overlay from `editor.sockets`. Each
    /// socket renders as a directional pin through the line pipeline
    /// (a shaft + arrowhead along its outward normal, plus a base cross
    /// on the surface), so the orientation that export bakes is visible
    /// in-scene.
    ///
    /// Cached against the `(position, normal)` list: cheap to recompute
    /// each frame for the handful of sockets a scene carries, and only
    /// touches the GPU when that list actually changes (place / delete /
    /// load). Renames don't move the gizmo, so they don't rebuild it.
    pub(super) fn update_socket_visualization(&mut self) {
        let cur: Vec<([f32; 3], [f32; 3])> = self
            .editor
            .sockets
            .iter()
            .map(|s| (s.position, s.normal))
            .collect();
        if cur == self.last_socket_viz {
            return;
        }
        self.last_socket_viz = cur.clone();
        if let Some(r) = &mut self.renderer {
            r.set_socket_mesh(&cur);
        }
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
            last_rebuild: self.last_rebuild,
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

/// Shrink a restored window size that the monitor can't display.
///
/// Both sizes are logical pixels. A size saved on a bigger display
/// restores verbatim on a smaller one — unplug the 4K dock and the
/// 3840x2160 entry produces a window whose bottom edge, and with it
/// the status bar, hangs off screen; the title bar is the only part
/// you can grab, so it can't be dragged back into view either.
///
/// Sizes that already fit are returned untouched: the saved value is
/// the *inner* size, so a window filling its monitor is normal, and
/// trimming it every launch would shrink it a step at a time. Once
/// something doesn't fit, both axes get the 10% margin — winit sizes
/// the inner area only, decorations live outside it, and there's no
/// cross-platform work-area query to subtract a taskbar with. A
/// monitor reporting a zero dimension tells us nothing, so in that
/// case the saved size stands.
fn fit_window_to_monitor(
    (w, h): (u32, u32),
    (mon_w, mon_h): (u32, u32),
) -> (u32, u32) {
    if mon_w == 0 || mon_h == 0 || (w <= mon_w && h <= mon_h) {
        return (w, h);
    }
    let fit = |v: u32, max: u32| v.min((max as f64 * 0.9) as u32).max(1);
    (fit(w, mon_w), fit(h, mon_h))
}

#[cfg(test)]
mod tests {
    use super::fit_window_to_monitor;

    #[test]
    fn a_size_that_fits_is_left_alone() {
        // Including the exact-fit case: a maximized window saves its
        // full inner size, and it must survive round-trips unchanged.
        assert_eq!(fit_window_to_monitor((1280, 720), (1920, 1080)), (1280, 720));
        assert_eq!(fit_window_to_monitor((1920, 1080), (1920, 1080)), (1920, 1080));
    }

    #[test]
    fn a_size_from_a_bigger_display_is_brought_back_on_screen() {
        let (w, h) = fit_window_to_monitor((3840, 2160), (1920, 1080));
        assert!(w < 1920 && h < 1080, "got {w}x{h}");
        assert_eq!((w, h), (1728, 972));
    }

    #[test]
    fn overflow_on_one_axis_margins_both() {
        // The window has to move fully back inside the monitor, not
        // just clip the offending axis.
        assert_eq!(fit_window_to_monitor((1920, 2160), (1920, 1080)), (1728, 972));
    }

    #[test]
    fn an_unusable_monitor_size_changes_nothing() {
        assert_eq!(fit_window_to_monitor((1280, 720), (0, 0)), (1280, 720));
    }

    #[test]
    fn the_result_is_stable_under_repeated_launches() {
        // Save → restore → save must converge, or the window shrinks
        // a little on every start.
        let once = fit_window_to_monitor((3840, 2160), (1920, 1080));
        assert_eq!(fit_window_to_monitor(once, (1920, 1080)), once);
    }
}
