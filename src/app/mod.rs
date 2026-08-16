//! Application state and the frame loop. `App` owns every long-lived
//! resource — window, renderer, document, editor, UI — and the sibling
//! modules split its behavior by responsibility.

mod agent_bridge;
mod document;
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
    core::{CellAabb, Voxel},
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
use document::Document;
use preview::PreviewState;

/// Alpha applied to the brush hover overlay. Higher than the procgen
/// preview (0.5) so the brush hint stays legible against existing
/// voxels of similar color.
const BRUSH_PREVIEW_ALPHA: f32 = 0.75;

/// Alpha for the move-drag ghost — the translucent copy of a
/// selection that follows the cursor. Lighter than the brush hint, so
/// it reads as in transit rather than already placed.
const MOVE_GHOST_ALPHA: f32 = 0.55;

/// How often `tick_autosave` writes the crash-recovery file while there
/// are unsaved changes. Long enough that saving a big world doesn't
/// hitch editing, short enough that a crash loses little work.
const AUTOSAVE_INTERVAL: Duration = Duration::from_secs(60);

/// How long after the last input the app keeps rendering at full rate.
/// Covers the gap before the OS starts delivering key repeats and the
/// tail of any interaction, so activity never *feels* throttled.
const ACTIVE_GRACE: Duration = Duration::from_millis(1500);

/// Frame cadence when nothing is happening. Ten a second keeps every
/// per-frame tick honest while cutting a motionless editor's burn to
/// nothing; input wakes the loop immediately regardless.
const IDLE_FRAME_INTERVAL: Duration = Duration::from_millis(100);

/// Cells a shape gesture may write in one commit, matching the agent
/// layer's per-op ceiling. Uncapped, a glancing-angle drag near the
/// raycast reach freezes the frame loop before anything commits.
pub(super) const MAX_SHAPE_COMMIT_CELLS: i64 = 2_097_152;

/// Cells the live preview re-enumerates per cursor step. Lower than the
/// commit cap because it runs on every mouse move; past it the preview
/// goes dark, while the commit still works.
pub(super) const MAX_SHAPE_PREVIEW_CELLS: i64 = 262_144;

/// Cells a shape gesture costs, before symmetry. A line costs its
/// length, the volumetric tools their AABB. `i64` throughout — the i32
/// product wraps for exactly the drags this exists to refuse.
pub(super) fn shape_cell_cost(tool: Tool, a: (i32, i32, i32), b: (i32, i32, i32)) -> i64 {
    let extent = |p: i32, q: i32| (p as i64 - q as i64).abs() + 1;
    let (ex, ey, ez) = (extent(a.0, b.0), extent(a.1, b.1), extent(a.2, b.2));
    match tool {
        Tool::Line => ex.max(ey).max(ez),
        _ => ex * ey * ez,
    }
}

/// How many mirrored copies the active symmetry produces per cell
/// (1, 2, 4 or 8) — the multiplier on a shape's base cost.
pub(super) fn symmetry_factor(symmetry: SymmetryAxes) -> i64 {
    1 << (symmetry.x as u32 + symmetry.y as u32 + symmetry.z as u32)
}

/// Inclusive AABB `(min, max)` enclosing a set of cell positions, or
/// `None` for an empty set. Used to remember a generation's footprint
/// for the "Frame Generated" camera action.
pub(super) fn bounds_of(positions: impl IntoIterator<Item = (i32, i32, i32)>) -> Option<CellAabb> {
    let mut it = positions.into_iter();
    let first = it.next()?;
    let (mut min, mut max) = (first, first);
    for p in it {
        min = (min.0.min(p.0), min.1.min(p.1), min.2.min(p.2));
        max = (max.0.max(p.0), max.1.max(p.1), max.2.max(p.2));
    }
    Some((min, max))
}

/// Cache key for the brush hover overlay: `(active cell, tool, brush
/// color, brush size, symmetry, shape drag key)`. See the field doc on
/// [`App::last_brush_preview_key`] for what each part invalidates.
type BrushPreviewKey = (
    (i32, i32, i32),
    Tool,
    Voxel,
    u8,
    SymmetryAxes,
    Option<ShapeDragKey>,
);

/// Main application state.
pub struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    egui_state: Option<egui_winit::State>,
    egui_renderer: Option<egui_wgpu::Renderer>,

    /// The open project — world, sockets, pipeline graph, metadata and
    /// the revision marks that say whether it differs from the user's
    /// file. See [`Document`].
    document: Document,
    mesher: GreedyMesher,
    editor: Editor,
    ui: Ui,

    last_frame: Instant,
    frame_times: VecDeque<f32>,

    /// When the user last touched the app. Drives the frame scheduler:
    /// full rate for a grace window after the last touch, an idle
    /// heartbeat after that.
    pub(super) last_interaction: Instant,
    /// When the next frame is due. `about_to_wait` requests a redraw
    /// once this passes and parks the loop until then, rather than
    /// polling a motionless scene at the display's refresh rate.
    pub(super) next_frame_at: Instant,

    /// `(milliseconds, chunks)` of the most recent non-empty rebuild —
    /// the cost a big edit adds to its frame. `None` until the first
    /// one; surfaced by the perf HUD.
    last_rebuild: Option<(f32, usize)>,

    cursor_captured: bool,
    cursor_pos: (f32, f32),
    modifiers: ModifiersState,

    /// The one in-flight edit gesture. See [`EditInteraction`] — every
    /// press / release / cancel path reads and writes this instead of
    /// a set of parallel latch fields.
    pub(super) interaction: EditInteraction,

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

    /// Cache key for the hover overlay, so its mesh is regenerated only
    /// when something meaningful changed. The active cell is the
    /// hovered one for brush tools and the adjacent one for shapes.
    last_brush_preview_key: Option<BrushPreviewKey>,

    /// Cache key for the selection wireframe so we don't rebuild the
    /// 24-vertex line buffer on every `CursorMoved` when the AABB
    /// hasn't changed.
    last_selection_box: Option<Selection>,

    /// Companion cache key to `last_selection_box` for the move ghost.
    /// Load-bearing on the commit frame: the drag's final box equals the
    /// committed one, so a box-only cache would strand the ghost.
    last_ghost_delta: Option<(i32, i32, i32)>,

    /// Cache of the socket gizmo's geometry inputs, so the line buffer
    /// rebuilds only when sockets move rather than every frame. Names
    /// don't affect the gizmo, so a rename doesn't invalidate it.
    last_socket_viz: Vec<([f32; 3], [f32; 3])>,

    /// Voxel data from the most recent Copy or Cut. Paste composites
    /// the non-air voxels onto the world. Not persisted across
    /// sessions, matching every comparable editor.
    pub(super) clipboard: Option<Clipboard>,

    /// Persisted user preferences. Loaded at startup, dehydrated and
    /// written back on close. The recent-files MRU lives here.
    prefs: Prefs,

    /// Tokio runtime on its own background thread, so the winit main
    /// thread never awaits. The agent bridge's HTTP server runs there,
    /// for the whole app lifetime.
    pub(super) async_runtime: runtime::AsyncRuntime,

    /// When the last autosave ran. `tick_autosave` rate-limits writes to
    /// `AUTOSAVE_INTERVAL`.
    pub(super) last_autosave: Instant,

    /// Modification time of `project_path` as of the last read or write.
    /// `tick_disk_reload` compares against it to tell somebody else's
    /// write from our own. `None` when no project file is open.
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

    /// AABB of the most recent generation or import, for the "Frame
    /// Generated" action. `None` until something is generated; not
    /// cleared on undo, since stale bounds still frame where it was.
    pub(super) last_generated_bounds: Option<CellAabb>,
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
    /// Load the crash-recovery autosave over the current scene. Guarded
    /// like the rest, because the prompt is non-modal and the default
    /// scene is live behind it.
    RecoverAutosave,
}

impl PendingAction {
    /// What the prompt says is about to happen ("Open another project
    /// anyway?"). Written to fit after "…unsaved changes.".
    fn describe(&self) -> &'static str {
        match self {
            PendingAction::NewProject => "start a new project",
            PendingAction::OpenPicker | PendingAction::OpenPath(_) => "open another project",
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

/// A stored `[r, g, b, a]` as a brush-ready voxel. Alpha is dropped
/// rather than restored: every voxel that reaches the world is opaque,
/// and both files this reads from are externally writable.
pub(super) fn brush_from_stored(color: [u8; 4]) -> Voxel {
    Voxel::from_rgb(color[0], color[1], color[2])
}

impl App {
    pub fn new() -> Self {
        let prefs = Prefs::load();

        let mut editor = Editor::new();
        editor.brush_color = brush_from_stored(prefs.editor.brush_color);
        editor.brush_color.flags = prefs.editor.brush_flags;
        editor
            .brush_color
            .set_tint_zone(prefs.editor.brush_tint_zone);
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

        let document = Document::new();
        let mut ui = Ui::new();
        ui.state.panels = prefs.panels.clone();
        ui.viewport = prefs.viewport.clone();
        ui.procgen = prefs.procgen.clone();
        ui.recent_files = prefs.recent_files.clone();

        let last_grid_size = ui.viewport.grid_size;
        let last_grid_spacing = ui.viewport.grid_spacing;

        Self {
            window: None,
            renderer: None,
            egui_state: None,
            egui_renderer: None,
            document,
            mesher: GreedyMesher::new(),
            editor,
            ui,
            last_frame: Instant::now(),
            frame_times: VecDeque::with_capacity(60),
            last_interaction: Instant::now(),
            next_frame_at: Instant::now(),
            last_rebuild: None,
            cursor_captured: false,
            cursor_pos: (0.0, 0.0),
            modifiers: ModifiersState::empty(),
            interaction: EditInteraction::Idle,
            project_path: None,
            last_grid_size,
            last_grid_spacing,
            preview: PreviewState::new(),
            agent: AgentBridgeState::new(),
            last_brush_preview_key: None,
            last_selection_box: None,
            last_ghost_delta: None,
            last_socket_viz: Vec::new(),
            clipboard: None,
            prefs,
            async_runtime: runtime::AsyncRuntime::new(),
            last_autosave: Instant::now(),
            watched_mtime: None,
            last_disk_poll: Instant::now(),
            pending_guarded: None,
            exit_requested: false,
            last_generated_bounds: None,
        }
    }

    /// Initial window inner-size from prefs, read by
    /// `handler::resumed`. Implausible values fall back to a known-good
    /// default, which the next `save_prefs` writes back.
    pub(super) fn initial_window_size(&self) -> (u32, u32) {
        let w = self.prefs.window.width;
        let h = self.prefs.window.height;
        // The range must match the clamp in `save_prefs`: a size one
        // side accepts and the other never writes makes a valid window
        // silently reset on every restart.
        if !(640..=4096).contains(&w) || !(480..=4096).contains(&h) {
            (1280, 720)
        } else {
            (w, h)
        }
    }

    /// Push the current path to the recent-files MRU and mirror the
    /// list to `ui.recent_files`, so the next frame's Open Recent menu
    /// reflects it.
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
            // `inner_size()` is physical pixels and `WindowPrefs` is
            // logical, matching the restore path. Without the
            // conversion the window grows by `scale_factor` per restart.
            let size = window.inner_size();
            let scale = window.scale_factor().max(0.1);
            let logical_w = ((size.width as f64 / scale).round() as u32).clamp(640, 4096);
            let logical_h = ((size.height as f64 / scale).round() as u32).clamp(480, 4096);
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

/// Expand `cells` with every symmetry mirror, deduped; symmetry off
/// returns them unchanged. Used by both the live preview and the commit
/// path, so the two always agree.
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

/// The face plane locked at the start of a brush stroke, so drag-paint
/// stays on it until release rather than stacking along the view
/// direction. Axis-aligned, stored as an axis plus a sign.
#[derive(Debug, Clone, Copy)]
pub(super) struct StrokePlane {
    pub axis: usize,
    pub sign: i32,
    pub plane_coord: f32,
    pub anchor_along_axis: i32,
}

/// Build a `StrokePlane` from a raycast hit, or `None` when the hit's
/// normal isn't axis-aligned. The caller then falls back to the
/// per-cell ray path.
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
    // The plane is the face between `voxel_pos` and `adjacent_pos`:
    // near face for a positive sign, far face for a negative one.
    // Either way every painted cell pins `adjacent_pos[axis]`.
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

/// The one in-flight edit gesture, if any. One value makes illegal
/// combinations unrepresentable and gives every exit the same verb.
/// The camera stays outside it — orbiting mid-gesture is legal.
#[derive(Debug, Default)]
pub(super) enum EditInteraction {
    /// No gesture in flight.
    #[default]
    Idle,
    /// Left button held: a brush stroke for the painting tools, and for
    /// the click tools a plain hold that uses none of the fields but
    /// still counts as "the button is down".
    BrushStroke {
        /// Locked face plane for drag-paint, keeping a stroke on one
        /// face instead of stacking toward the camera. `None` until the
        /// first apply hits something and locks it.
        plane: Option<StrokePlane>,
        /// Cell the most recent stroke step applied at, so drag-paint
        /// doesn't re-apply while the cursor sits in the same cell.
        last_voxel: Option<(i32, i32, i32)>,
        /// Screen position of the press — the dead-zone origin that
        /// keeps single-click hand tremor from painting a streak.
        start_screen: (f32, f32),
    },
    /// Shape phase one: button held, the cursor's plane-locked hit is
    /// the footprint's other corner. `anchor` is the first press's
    /// `adjacent_pos`, sitting on `plane`.
    ShapeFootprint {
        anchor: (i32, i32, i32),
        plane: StrokePlane,
    },
    /// Shape phase two: button released, vertical cursor movement
    /// extrudes height along the plane normal. A second click commits
    /// and Esc cancels.
    ShapeHeight {
        anchor: (i32, i32, i32),
        plane: StrokePlane,
        /// Footprint's other corner at the moment the button was
        /// released (locked from then on — only height changes).
        end_on_plane: (i32, i32, i32),
        /// Cursor's screen-Y at release. Height = `(release_y -
        /// cursor_y) / SHAPE_HEIGHT_PIXELS_PER_VOXEL`, clamped to
        /// ≥ 0 (the user can't extrude *into* the face).
        release_screen_y: f32,
    },
    /// Select tool, dragging out a new marquee from `anchor`.
    /// Finalized into `editor.selection` on release.
    SelectDrag { anchor: (i32, i32, i32) },
    /// Select tool, dragging the existing selection's contents. Every
    /// cursor move renders the ghost at `current - anchor`; release
    /// runs `move_selection` with that delta as one undoable Command.
    SelectMove {
        anchor: (i32, i32, i32),
        /// The selection's non-air voxels, snapshotted at pick-up so the
        /// ghost translates this set rather than re-reading the world.
        /// Empty when the box was too big to sweep.
        ghost: Vec<((i32, i32, i32), Voxel)>,
    },
}

impl EditInteraction {
    /// A gesture is in flight — drives full-rate frame pacing and the
    /// "don't fight the drag" guards.
    pub(super) fn is_active(&self) -> bool {
        !matches!(self, EditInteraction::Idle)
    }

    /// The face plane the cursor's raycast is locked to, if any: a
    /// brush stroke after its first in-world apply, or either shape
    /// phase.
    pub(super) fn locked_plane(&self) -> Option<StrokePlane> {
        match self {
            EditInteraction::BrushStroke { plane, .. } => *plane,
            EditInteraction::ShapeFootprint { plane, .. }
            | EditInteraction::ShapeHeight { plane, .. } => Some(*plane),
            _ => None,
        }
    }

    /// The shape-gesture cache key for `update_brush_preview`, or
    /// `None` when no shape gesture is in flight. `hovered_cell` is the
    /// Footprint phase's current cell.
    pub(super) fn shape_cache_key(
        &self,
        cursor_y: f32,
        hovered_cell: Option<(i32, i32, i32)>,
    ) -> Option<ShapeDragKey> {
        match *self {
            EditInteraction::ShapeFootprint { anchor, .. } => Some(ShapeDragKey::Footprint {
                anchor,
                end_cell: hovered_cell.unwrap_or(anchor),
            }),
            EditInteraction::ShapeHeight {
                anchor,
                end_on_plane,
                release_screen_y,
                ..
            } => Some(ShapeDragKey::Height {
                anchor,
                end_on_plane,
                height: shape_height_from_cursor(release_screen_y, cursor_y),
            }),
            _ => None,
        }
    }

    /// 3D end corner of the shape after extrusion — `Some` only in
    /// Height phase. Footprint callers use the cursor's plane-locked
    /// `hovered_voxel.adjacent_pos` directly.
    pub(super) fn shape_extruded_end(&self, cursor_y: f32) -> Option<(i32, i32, i32)> {
        let EditInteraction::ShapeHeight {
            plane,
            end_on_plane,
            release_screen_y,
            ..
        } = *self
        else {
            return None;
        };
        let h = shape_height_from_cursor(release_screen_y, cursor_y);
        let mut e = [end_on_plane.0, end_on_plane.1, end_on_plane.2];
        e[plane.axis] += plane.sign * h;
        Some((e[0], e[1], e[2]))
    }
}

/// Reduced cache key for `update_brush_preview` — drops the f32
/// `release_screen_y` (uses quantized integer height instead) so
/// the key implements `Eq` for the existing tuple-comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ShapeDragKey {
    Footprint {
        anchor: (i32, i32, i32),
        /// The cursor's plane-locked cell. Without it in the key, a
        /// Footprint-phase drag wouldn't invalidate the cache and the
        /// preview would freeze on the first cell.
        end_cell: (i32, i32, i32),
    },
    Height {
        anchor: (i32, i32, i32),
        end_on_plane: (i32, i32, i32),
        height: i32,
    },
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
        // Default the cursor to the screen center, so a scroll before
        // any mouse movement anchors there rather than at the top-left
        // corner and drags the orbit pivot with it.
        let physical = window.inner_size();
        self.cursor_pos = (physical.width as f32 / 2.0, physical.height as f32 / 2.0);
        self.window = Some(window.clone());

        let renderer =
            pollster::block_on(Renderer::new(window.clone())).expect("Failed to create renderer");

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx,
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        // egui is a 2D overlay whose pass attaches no depth texture, so
        // its pipeline must not declare a depth format either or wgpu
        // validation refuses the mismatch.
        let egui_renderer =
            egui_wgpu::Renderer::new(&renderer.device, renderer.config.format, None, 1, false);

        self.renderer = Some(renderer);
        self.egui_state = Some(egui_state);
        self.egui_renderer = Some(egui_renderer);

        // Always start on the default scene, then defer the recovery
        // prompt to the first `RedrawRequested`: a dialog raised inside
        // `resumed` exits the process on the Windows setup.
        self.create_initial_scene();
        // The rebuild above bumped the revision; the default scene is
        // the baseline, not unsaved work.
        self.document.mark_saved();
        // An autosave on disk means the last session didn't exit
        // cleanly, so raise the recovery prompt. It is egui, never an
        // `rfd::MessageDialog` — that exits the process here.
        if Self::autosave_path().is_some_and(|p| p.exists()) {
            self.ui.state.show_recovery_prompt = true;
        }
    }

    /// Create the initial test scene shown on startup.
    fn create_initial_scene(&mut self) {
        self.document.world.create_test_cube((0, 8, 0), 4);
        self.document.world.create_test_ground(20, 2);
        self.rebuild_all_meshes();
        // Anchor the orbit pivot on the actual scene rather than the
        // origin, or orbit circles a point underneath the model and the
        // visible geometry swings through a wide arc.
        self.recenter_camera_on_scene();
    }

    /// Reset every piece of session state that refers to the geometry of
    /// the scene being replaced. Callers that restore from a file run
    /// this first and repopulate afterwards.
    ///
    /// # Safety
    /// Every whole-scene replacement path must call this, or a stale
    /// marquee edits the new world at the old coordinates.
    pub(super) fn reset_scene_session_state(&mut self) {
        self.editor.history.clear();
        // Clears the selection plus the drag / move anchors and the
        // move ghost — see `App::deselect`.
        self.deselect();
        self.document.sockets.clear();
        // The graph is document data like the sockets, so it goes with
        // the scene. Open and reload put the file's graph back right
        // after; New Scene wants the empty one this leaves.
        self.document.graph = PipelineGraph::default();
        self.cancel_interaction();
        self.last_generated_bounds = None;
        // A parked batch was built against the world being thrown away,
        // and the generation check can't catch this one: `history
        // .clear()` above can land on the same value it parked at.
        self.drop_pending_review_for_new_scene();
        if let Some(renderer) = &mut self.renderer {
            renderer.chunk_meshes.clear();
        }
    }

    /// Run `action`, or park it until the user answers the
    /// unsaved-changes prompt. On `App` rather than in the UiAction
    /// dispatch, because the key handler calls the file ops directly.
    pub(super) fn guard_then(&mut self, action: PendingAction) {
        if !self.document.unsaved() {
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

    /// Give up on the parked action. Recovery is the one whose entry
    /// point closes behind it, so dropping it silently would strand the
    /// autosave — the prompt goes back up instead.
    pub(super) fn drop_pending_guarded(&mut self) {
        if let Some(PendingAction::RecoverAutosave) = self.pending_guarded.take() {
            self.ui.state.show_recovery_prompt = true;
        }
    }

    /// Path of the crash-recovery autosave, next to `prefs.ron` in the
    /// platform config dir. `None` if the OS exposes no config dir.
    fn autosave_path() -> Option<PathBuf> {
        Prefs::config_path().and_then(|p| p.parent().map(|d| d.join("autosave.vxlt")))
    }

    /// Per-frame autosave tick, cheap when idle. Writes whenever the
    /// document changed, an empty world included — post-Clear-All
    /// emptiness *is* the document state. Only the autosave mark moves.
    pub(super) fn tick_autosave(&mut self) {
        if !self.document.autosave_due() || self.last_autosave.elapsed() < AUTOSAVE_INTERVAL {
            return;
        }
        let Some(path) = Self::autosave_path() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let state = self.current_editor_state();
        let metadata = self.document.metadata.clone();
        // Atomic write: serialize to a per-process temp, then rename it
        // over the autosave, so a crash mid-write leaves at most a stale
        // temp and recovery always loads a complete state.
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        let result =
            voxelith::io::save_world_with_state(&self.document.world, state, metadata, &tmp)
                .and_then(|()| std::fs::rename(&tmp, &path).map_err(Into::into));
        match result {
            Ok(()) => {
                log::info!("Autosaved to {}", path.display());
                self.document.mark_autosaved();
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

    /// Snap `camera.target` to the scene center and re-derive the
    /// controller's angles. The position is untouched, so only the orbit
    /// pivot moves. A no-op on an empty world.
    pub(super) fn recenter_camera_on_scene(&mut self) {
        let Some(center) = self.document.world.scene_center() else {
            return;
        };
        let Some(renderer) = &mut self.renderer else {
            return;
        };
        renderer.camera.target = center;
        renderer
            .camera_controller
            .sync_orbit_state_from_camera(&renderer.camera);
    }

    /// How long until the next frame is due — the scheduler's one policy
    /// decision. Full rate while the user is plausibly mid-something,
    /// the idle heartbeat otherwise.
    pub(super) fn frame_interval(&self) -> Duration {
        let navigating = self
            .renderer
            .as_ref()
            .is_some_and(|r| r.camera_controller.is_navigating());
        let gesturing = self.cursor_captured || self.interaction.is_active();
        let active = navigating || gesturing || self.last_interaction.elapsed() < ACTIVE_GRACE;
        if active {
            Duration::ZERO
        } else {
            IDLE_FRAME_INTERVAL
        }
    }

    /// Abandon whatever gesture is in flight — back to `Idle`, with any
    /// stroke's undo entry sealed. The one verb every exit point uses,
    /// so no caller needs to know which state it is cancelling.
    pub(super) fn cancel_interaction(&mut self) {
        self.editor.history.end_stroke();
        self.interaction = EditInteraction::Idle;
    }

    /// Rebuild meshes for all dirty chunks and upload them. Meshing runs
    /// on the rayon pool; uploads stay on the calling thread, where the
    /// wgpu handles live and the cost is small anyway.
    pub(super) fn rebuild_all_meshes(&mut self) {
        if self.renderer.is_none() {
            return;
        }

        let dirty = self.document.world.dirty_chunks();
        if dirty.is_empty() {
            return;
        }
        let started = Instant::now();

        // Dirty chunks mean voxel data changed, and this is the single
        // chokepoint every edit funnels into — so it is where the
        // revision moves. Non-voxel edits bump at their own sites.
        self.document.bump();

        // Concurrent reads only: each mesher read-locks its chunk and
        // the 26 neighbors AO samples, and workers on disjoint chunks
        // share those fine. See `mesh::neighbors`.
        let mesher = &self.mesher;
        let world = &self.document.world;
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

        self.document.world.clear_dirty_flags();

        self.last_rebuild = Some((started.elapsed().as_secs_f32() * 1000.0, dirty.len()));
    }

    /// Refresh the hover overlay: a brush sphere, an idle shape's anchor
    /// hint, or a dragging shape's full cell set, each with symmetry
    /// mirrors. Cached, so an unchanged frame costs a few comparisons.
    pub(super) fn update_brush_preview(&mut self) {
        let tool = self.effective_tool();

        // A tool switch cancels the gesture it orphans, reconciled here
        // because switches arrive through several doors. A shape gesture
        // survives within the shape family; a Select drag does not.
        match self.interaction {
            EditInteraction::ShapeFootprint { .. } | EditInteraction::ShapeHeight { .. }
                if !tool.is_shape() =>
            {
                self.cancel_interaction();
            }
            EditInteraction::SelectDrag { .. } | EditInteraction::SelectMove { .. }
                if tool != Tool::Select =>
            {
                self.cancel_interaction();
            }
            _ => {}
        }

        let symmetry = self.editor.symmetry;
        let color = self.editor.brush_color;
        let size = self.editor.brush_size;
        let cursor_y = self.cursor_pos.1;

        // Eyedropper, Select and Socket skip this overlay: the first
        // would show the brush color rather than the sampled one, and
        // the other two draw their own.
        let show = !matches!(tool, Tool::Eyedropper | Tool::Select | Tool::Socket);

        // `cell` is hover-derived for non-shape tools and idle shapes.
        // An active shape gesture pins it, since the gesture's own key
        // already captures everything affecting the output.
        let hovered_cell = self.editor.hovered_voxel.map(|h| h.adjacent_pos);
        let drag_key = self.interaction.shape_cache_key(cursor_y, hovered_cell);
        let key = if show {
            if drag_key.is_some() {
                Some(((0, 0, 0), tool, color, size, symmetry, drag_key))
            } else {
                self.editor.hovered_voxel.map(|h| {
                    // Key on the cell the preview is drawn at: Place uses
                    // `adjacent_pos`, the rest `voxel_pos`. Otherwise
                    // crossing to another face of the same voxel is stale.
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

        // An active shape gesture has its own branch — Height phase
        // reads screen space, not a hovered cell — while every other
        // mode needs a real hover.
        let shape_ends = match self.interaction {
            EditInteraction::ShapeFootprint { anchor, .. } => {
                // Footprint: cursor's plane-locked hit is the
                // other corner. No hit (cursor off-world) → no
                // preview this frame.
                let Some(hit) = self.editor.hovered_voxel else {
                    if let Some(r) = &mut self.renderer {
                        r.clear_brush_preview();
                    }
                    return;
                };
                Some((anchor, hit.adjacent_pos))
            }
            EditInteraction::ShapeHeight { anchor, .. } => {
                // Height: extrude end_on_plane along the plane
                // normal by the cursor-Y delta.
                let end_3d = self
                    .interaction
                    .shape_extruded_end(cursor_y)
                    .expect("Height phase");
                Some((anchor, end_3d))
            }
            _ => None,
        };
        let positions: Vec<(i32, i32, i32)> = if let Some((anchor, end_3d)) = shape_ends {
            // Budget check BEFORE enumerating: the enumeration is the
            // cost being bounded, and it re-runs on every cursor step.
            let cost =
                shape_cell_cost(tool, anchor, end_3d).saturating_mul(symmetry_factor(symmetry));
            if cost > MAX_SHAPE_PREVIEW_CELLS {
                self.ui.set_status(format!(
                    "Shape too large to preview ({cost} cells) — commits up to \
                     {MAX_SHAPE_COMMIT_CELLS}",
                ));
                if let Some(r) = &mut self.renderer {
                    r.clear_brush_preview();
                }
                return;
            }
            let plane_axis = self.interaction.locked_plane().map(|p| p.axis);
            let raw = match tool {
                Tool::Line => line_voxels(anchor, end_3d),
                Tool::Box => box_voxels(anchor, end_3d),
                Tool::Sphere => sphere_voxels(anchor, end_3d),
                Tool::Cylinder => cylinder_voxels(anchor, end_3d, plane_axis),
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

    /// Refresh the selection wireframe and the move ghost, both driven
    /// from the interaction state through one cache gate. The delta half
    /// of the key is what clears the ghost on the commit frame.
    pub(super) fn update_selection_visualization(&mut self) {
        // Resolve the wireframe box and, for a move drag, the live
        // translation delta the ghost follows.
        let (preview, ghost_delta) = match self.interaction {
            EditInteraction::SelectDrag { anchor } => {
                // New-selection drag — anchor → current end cell.
                let box_ = self
                    .editor
                    .hovered_voxel
                    .map(|hit| Selection::from_corners(anchor, Self::select_anchor_pos(&hit)));
                (box_, None)
            }
            EditInteraction::SelectMove { anchor, .. } => {
                // Move drag: the selection translated by the cursor
                // delta, falling back to the untranslated box when the
                // cursor is off-world, so it stays put rather than vanish.
                match (self.editor.selection, self.editor.hovered_voxel) {
                    (Some(sel), Some(hit)) => {
                        let cur = Self::select_anchor_pos(&hit);
                        let delta = (cur.0 - anchor.0, cur.1 - anchor.1, cur.2 - anchor.2);
                        (Some(sel.translated(delta)), Some(delta))
                    }
                    _ => (self.editor.selection, Some((0, 0, 0))),
                }
            }
            _ => (self.editor.selection, None),
        };

        if (preview, ghost_delta) == (self.last_selection_box, self.last_ghost_delta) {
            return;
        }
        self.last_selection_box = preview;
        self.last_ghost_delta = ghost_delta;

        // Build the translated ghost mesh (move drag only) before
        // borrowing the renderer, so reading the ghost snapshot
        // doesn't tangle with the `&mut renderer` borrow.
        let ghost_mesh = match (ghost_delta, &self.interaction) {
            (Some(delta), EditInteraction::SelectMove { ghost, .. }) if !ghost.is_empty() => {
                let voxels: Vec<((i32, i32, i32), Voxel)> = ghost
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

    /// Snapshot the selection's non-air voxels at the start of a move
    /// drag, so the ghost translates the captured set rather than
    /// re-reading the world. Positions stay absolute.
    pub(super) fn move_ghost_snapshot(&self, sel: Selection) -> Vec<((i32, i32, i32), Voxel)> {
        // The same dense-sweep bound the selection operations use: this
        // walks every cell of the AABB at pick-up. An oversized box
        // simply gets no ghost.
        let extent = |a: i32, b: i32| (b as i64 - a as i64) + 1;
        let cells = extent(sel.min.0, sel.max.0)
            * extent(sel.min.1, sel.max.1)
            * extent(sel.min.2, sel.max.2);
        if cells > Self::MAX_SELECTION_SWEEP_CELLS {
            return Vec::new();
        }
        sel.iter_cells()
            .filter_map(|(x, y, z)| {
                let v = self.document.world.get_voxel(x, y, z);
                (!v.is_air()).then_some(((x, y, z), v))
            })
            .collect()
    }

    /// Refresh the socket gizmos, so the orientation the export bakes is
    /// visible in-scene. Cached against the `(position, normal)` list,
    /// which a rename doesn't change.
    pub(super) fn update_socket_visualization(&mut self) {
        let cur: Vec<([f32; 3], [f32; 3])> = self
            .document
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

    /// The cell a Select gesture anchors at. A real hit selects the hit
    /// cell; a virtual ground hit uses `adjacent_pos`, or an empty-world
    /// drag puts the selection one cell underground.
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
            chunks: self.document.world.chunk_count(),
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

/// Shrink a restored window size the monitor can't display — a size
/// saved on a 4K dock otherwise hangs its status bar off a laptop
/// screen. Sizes that already fit are returned untouched.
fn fit_window_to_monitor((w, h): (u32, u32), (mon_w, mon_h): (u32, u32)) -> (u32, u32) {
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
        assert_eq!(
            fit_window_to_monitor((1280, 720), (1920, 1080)),
            (1280, 720)
        );
        assert_eq!(
            fit_window_to_monitor((1920, 1080), (1920, 1080)),
            (1920, 1080)
        );
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
        assert_eq!(
            fit_window_to_monitor((1920, 2160), (1920, 1080)),
            (1728, 972)
        );
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
