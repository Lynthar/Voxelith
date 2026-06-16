//! Editor tools for voxel manipulation.
//!
//! Provides different brush types and editing modes.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use super::{Command, CommandHistory, RaycastHit, SymmetryAxes, VoxelChange};
use crate::core::{Voxel, World};

/// Time window within which consecutive brush writes coalesce into a
/// single undo entry. Picked to match a reasonable drag/click cadence
/// (≈5 actions/sec) so paint strokes feel like one operation while
/// distinct user gestures stay separate.
pub const STROKE_MERGE_WINDOW: Duration = Duration::from_millis(200);

/// Maximum chebyshev distance (in voxels) that `flood_fill` will
/// expand from its start cell. Without this cap a fill in an unbounded
/// world could traverse arbitrarily far; the only existing limit was
/// `max_voxels`, which is a count cap, not a spatial one.
pub const MAX_FILL_DIST: i32 = 64;

/// Available editing tools.
///
/// Brush tools (`Place`/`Remove`/`Paint`/`Eyedropper`/`Fill`) act on
/// the hovered cell every click or drag-step. Shape tools (`Line`,
/// `Box`, `Sphere`, `Cylinder`) use a click-anchor / drag-extent /
/// release-commit gesture: the shape's full voxel set is committed
/// in one `Command` on mouse-up. The `Select` tool follows the same
/// click-drag-release gesture as shapes, but commits a `Selection`
/// AABB into `Editor::selection` instead of writing voxels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    /// Place voxels
    Place,
    /// Remove voxels
    Remove,
    /// Paint existing voxels (change color without adding/removing)
    Paint,
    /// Pick color from existing voxel
    Eyedropper,
    /// Fill region with voxels
    Fill,
    /// Line shape: drag from anchor to end, fills with brush color
    /// using 3D Bresenham.
    Line,
    /// Filled axis-aligned box: drag corner-to-corner.
    Box,
    /// Filled ellipsoid fitting in the drag bbox (use a square-ish
    /// drag for a uniform sphere).
    Sphere,
    /// Filled cylinder fitting in the drag bbox; axis = bbox's
    /// longest dimension, ellipse cross-section in the other two.
    Cylinder,
    /// Box selection: drag corner-to-corner to mark an AABB region
    /// for batch operations (copy / cut / paste / delete / move).
    Select,
    /// Place a named attachment point. Single click drops a socket at
    /// the center of the clicked face, oriented along the face normal;
    /// it carries no voxels and exports to glTF as an empty node. Kept
    /// **last** in the enum so the `current_tool as usize` discriminant
    /// in `.vxlt` / prefs stays stable for the existing tools.
    Socket,
}

impl Tool {
    /// Get display name
    pub fn name(&self) -> &'static str {
        match self {
            Tool::Place => "Place",
            Tool::Remove => "Remove",
            Tool::Paint => "Paint",
            Tool::Eyedropper => "Eyedropper",
            Tool::Fill => "Fill",
            Tool::Line => "Line",
            Tool::Box => "Box",
            Tool::Sphere => "Sphere",
            Tool::Cylinder => "Cylinder",
            Tool::Select => "Select",
            Tool::Socket => "Socket",
        }
    }

    /// Get keyboard shortcut hint
    pub fn shortcut(&self) -> &'static str {
        match self {
            Tool::Place => "1",
            Tool::Remove => "2",
            Tool::Paint => "3",
            Tool::Eyedropper => "4 / Alt",
            Tool::Fill => "5",
            Tool::Line => "6",
            Tool::Box => "7",
            Tool::Sphere => "8",
            Tool::Cylinder => "9",
            Tool::Select => "0",
            // No digit free; placed from the toolbar / Tools panel.
            Tool::Socket => "",
        }
    }

    /// Whether this tool uses click-anchor / drag-extent / release-
    /// commit semantics. Shape tools do; brush tools don't. `Select`
    /// shares the gesture but goes through its own commit path
    /// (writing into `Editor::selection`, not the world).
    pub fn is_shape(&self) -> bool {
        matches!(
            self,
            Tool::Line | Tool::Box | Tool::Sphere | Tool::Cylinder
        )
    }

    /// Whether this tool's gesture needs a release-time commit
    /// (latch anchor on press, finalize on release). Used by the
    /// event handler to dispatch between `commit_shape` /
    /// `commit_selection` / brush stroke-end on mouse-up.
    pub fn needs_release_commit(&self) -> bool {
        self.is_shape() || matches!(self, Tool::Select)
    }

    /// Whether this tool needs an anchor cell to operate. Place,
    /// every shape tool, and Select need one (so they can build /
    /// pick into an empty world via the y=0 ground-plane raycast
    /// fallback); brush tools that read the hovered cell
    /// (Remove/Paint/Eyedropper/Fill) need a real solid voxel and
    /// shouldn't engage the fallback.
    pub fn uses_ground_plane_fallback(&self) -> bool {
        // Socket joins this set so a socket can be dropped on the y=0
        // ground in an empty world (e.g. a spawn / origin marker), not
        // only on an existing voxel face.
        matches!(self, Tool::Place | Tool::Select | Tool::Socket) || self.is_shape()
    }
}

/// Context passed to tools during execution
pub struct ToolContext<'a> {
    pub world: &'a mut World,
    pub history: &'a mut CommandHistory,
    pub brush_color: Voxel,
    pub brush_size: u8,
    pub symmetry: SymmetryAxes,
}

/// Trait for tool implementations
pub trait EditorTool {
    /// Apply the tool at the given hit location
    fn apply(&self, ctx: &mut ToolContext, hit: &RaycastHit);

    /// Positions the brush hover overlay should highlight, including
    /// any symmetry-mirrored copies (deduped). Caller passes its
    /// current `symmetry` so a single source of truth drives both this
    /// preview and the matching `apply` call.
    fn preview_positions(
        &self,
        hit: &RaycastHit,
        brush_size: u8,
        symmetry: SymmetryAxes,
    ) -> Vec<(i32, i32, i32)>;
}

/// Brush tool for place/remove/paint operations
pub struct BrushTool {
    pub mode: Tool,
}

impl BrushTool {
    pub fn new(mode: Tool) -> Self {
        Self { mode }
    }

    /// Get affected positions for a spherical brush
    fn get_brush_positions(center: (i32, i32, i32), size: u8) -> Vec<(i32, i32, i32)> {
        let mut positions = Vec::new();
        let radius = (size as i32 - 1).max(0);
        let radius_sq = (radius as f32 + 0.5).powi(2);

        for dz in -radius..=radius {
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let dist_sq = (dx * dx + dy * dy + dz * dz) as f32;
                    if dist_sq <= radius_sq {
                        positions.push((
                            center.0 + dx,
                            center.1 + dy,
                            center.2 + dz,
                        ));
                    }
                }
            }
        }

        positions
    }
}

impl EditorTool for BrushTool {
    fn apply(&self, ctx: &mut ToolContext, hit: &RaycastHit) {
        let center = match self.mode {
            Tool::Place => hit.adjacent_pos,
            Tool::Remove | Tool::Paint => hit.voxel_pos,
            // Eyedropper / Fill go through input.rs's tool dispatch,
            // not BrushTool. Shape tools and Select have their own
            // click-anchor / drag / commit lifecycle and never call
            // this path.
            Tool::Eyedropper
            | Tool::Fill
            | Tool::Line
            | Tool::Box
            | Tool::Sphere
            | Tool::Cylinder
            | Tool::Select
            | Tool::Socket => return,
        };

        // Expand the brush sphere across symmetry mirrors. Spheres that
        // overlap near a symmetry plane would double-count cells, so we
        // dedup via HashSet — both for efficiency and so the resulting
        // change set has each position exactly once.
        let positions = Self::affected_positions(center, ctx.brush_size, ctx.symmetry);

        let changes: Vec<VoxelChange> = match self.mode {
            Tool::Place => positions
                .into_iter()
                .map(|pos| VoxelChange {
                    pos,
                    old_voxel: ctx.world.get_voxel(pos.0, pos.1, pos.2),
                    new_voxel: ctx.brush_color,
                })
                .filter(|c| c.old_voxel != c.new_voxel)
                .collect(),
            Tool::Remove => positions
                .into_iter()
                .filter_map(|pos| {
                    let old = ctx.world.get_voxel(pos.0, pos.1, pos.2);
                    if old.is_air() {
                        None
                    } else {
                        Some(VoxelChange { pos, old_voxel: old, new_voxel: Voxel::AIR })
                    }
                })
                .collect(),
            Tool::Paint => positions
                .into_iter()
                .filter_map(|pos| {
                    let old = ctx.world.get_voxel(pos.0, pos.1, pos.2);
                    if !old.is_air() && old != ctx.brush_color {
                        Some(VoxelChange { pos, old_voxel: old, new_voxel: ctx.brush_color })
                    } else {
                        None
                    }
                })
                .collect(),
            _ => return,
        };

        if !changes.is_empty() {
            let cmd = Command::set_voxels(changes);
            ctx.history.execute_merge(cmd, ctx.world, STROKE_MERGE_WINDOW);
        }
    }

    fn preview_positions(
        &self,
        hit: &RaycastHit,
        brush_size: u8,
        symmetry: SymmetryAxes,
    ) -> Vec<(i32, i32, i32)> {
        match self.mode {
            Tool::Place => Self::affected_positions(hit.adjacent_pos, brush_size, symmetry),
            Tool::Remove | Tool::Paint => {
                Self::affected_positions(hit.voxel_pos, brush_size, symmetry)
            }
            // Fill marks just the seed cell(s) — full flood region would
            // be too expensive to compute every frame.
            Tool::Fill => symmetry.mirror_positions(hit.voxel_pos),
            Tool::Eyedropper => vec![hit.voxel_pos],
            // Shape tools and Select have their own preview path
            // (App::update_brush_preview for shapes; the dedicated
            // selection-mesh slot for Select). BrushTool's preview is
            // bypassed for them. Empty here keeps the trait satisfied
            // without contributing stray cells if someone ever calls
            // this for a non-brush tool by mistake.
            Tool::Line | Tool::Box | Tool::Sphere | Tool::Cylinder | Tool::Select
            | Tool::Socket => Vec::new(),
        }
    }
}

impl BrushTool {
    /// Brush sphere positions centered at `center` plus every mirror
    /// implied by `symmetry`, deduped. Pulled out so both `apply` and
    /// `preview_positions` go through the same expansion path.
    fn affected_positions(
        center: (i32, i32, i32),
        brush_size: u8,
        symmetry: SymmetryAxes,
    ) -> Vec<(i32, i32, i32)> {
        if !symmetry.any() {
            // Common path: skip the HashSet allocation when no mirroring.
            return Self::get_brush_positions(center, brush_size);
        }
        let mut out: HashSet<(i32, i32, i32)> = HashSet::new();
        for c in symmetry.mirror_positions(center) {
            for p in Self::get_brush_positions(c, brush_size) {
                out.insert(p);
            }
        }
        out.into_iter().collect()
    }
}

/// Pick color from a voxel
pub fn eyedrop(world: &World, hit: &RaycastHit) -> Option<Voxel> {
    let voxel = world.get_voxel(hit.voxel_pos.0, hit.voxel_pos.1, hit.voxel_pos.2);
    if !voxel.is_air() {
        Some(voxel)
    } else {
        None
    }
}

/// Compute the changes a flood-fill would make from `start`, without
/// applying them. Pulled out of `flood_fill` so callers that need to
/// batch multiple fills into a single undo entry (notably the symmetric
/// fill path in `app::input::apply_tool`) can collect changes from
/// several seeds and submit one combined `Command`.
///
/// Returns an empty `Vec` if `start` already holds `new_voxel` or
/// would produce no writes for any reason.
pub fn compute_flood_fill_changes(
    world: &World,
    start: (i32, i32, i32),
    new_voxel: Voxel,
    max_voxels: usize,
) -> Vec<VoxelChange> {
    let target_voxel = world.get_voxel(start.0, start.1, start.2);
    if target_voxel == new_voxel {
        return Vec::new();
    }

    let mut changes = Vec::new();
    let mut visited = HashSet::new();
    let mut stack = vec![start];

    while let Some(pos) = stack.pop() {
        if visited.contains(&pos) {
            continue;
        }
        if changes.len() >= max_voxels {
            break;
        }
        // Spatial cap: skip cells outside the chebyshev radius around
        // `start`. Prevents runaway fills in unbounded worlds where
        // the connected region might extend far beyond what the user
        // intended to paint.
        if (pos.0 - start.0).abs() > MAX_FILL_DIST
            || (pos.1 - start.1).abs() > MAX_FILL_DIST
            || (pos.2 - start.2).abs() > MAX_FILL_DIST
        {
            continue;
        }

        let current = world.get_voxel(pos.0, pos.1, pos.2);
        if current != target_voxel {
            continue;
        }

        visited.insert(pos);
        changes.push(VoxelChange {
            pos,
            old_voxel: current,
            new_voxel,
        });

        // 6-connectivity expansion.
        let neighbors = [
            (pos.0 + 1, pos.1, pos.2),
            (pos.0 - 1, pos.1, pos.2),
            (pos.0, pos.1 + 1, pos.2),
            (pos.0, pos.1 - 1, pos.2),
            (pos.0, pos.1, pos.2 + 1),
            (pos.0, pos.1, pos.2 - 1),
        ];
        for neighbor in neighbors {
            if !visited.contains(&neighbor) {
                stack.push(neighbor);
            }
        }
    }

    changes
}

/// Flood fill from a single seed: thin wrapper that computes the
/// changes via `compute_flood_fill_changes` and pushes one `Command`
/// onto `history`. Returns the number of voxels written.
pub fn flood_fill(
    world: &mut World,
    history: &mut CommandHistory,
    start: (i32, i32, i32),
    new_voxel: Voxel,
    max_voxels: usize,
) -> usize {
    let changes = compute_flood_fill_changes(world, start, new_voxel, max_voxels);
    let count = changes.len();
    if !changes.is_empty() {
        let cmd = Command::set_voxels(changes);
        history.execute(cmd, world);
    }
    count
}

/// Flood fill from multiple seeds, batching all resulting writes into
/// a single `Command` so the whole symmetric stroke is one undo entry.
/// Each seed's flood is computed against the *original* world snapshot
/// (not the cumulative one), so two seeds spreading toward the same
/// region won't surprise each other; the per-position dedup keeps the
/// first occurrence (any later mirror writing the same cell would
/// produce the same `new_voxel` anyway, so the choice is benign).
pub fn flood_fill_multi(
    world: &mut World,
    history: &mut CommandHistory,
    starts: &[(i32, i32, i32)],
    new_voxel: Voxel,
    max_voxels: usize,
) -> usize {
    let mut combined: HashMap<(i32, i32, i32), VoxelChange> = HashMap::new();
    for &start in starts {
        // Skip air seeds defensively — Fill semantics don't extend air.
        if world.get_voxel(start.0, start.1, start.2).is_air() {
            continue;
        }
        for change in compute_flood_fill_changes(world, start, new_voxel, max_voxels) {
            combined.entry(change.pos).or_insert(change);
        }
    }
    let count = combined.len();
    if count > 0 {
        let cmd = Command::set_voxels(combined.into_values().collect());
        history.execute(cmd, world);
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brush_positions() {
        let positions = BrushTool::get_brush_positions((0, 0, 0), 1);
        assert_eq!(positions.len(), 1);
        assert!(positions.contains(&(0, 0, 0)));

        let positions = BrushTool::get_brush_positions((0, 0, 0), 2);
        assert!(positions.len() > 1);
    }

    #[test]
    fn test_flood_fill() {
        let mut world = World::new();
        let mut history = CommandHistory::new(100);

        // Create a small area to fill
        for x in 0..3 {
            for z in 0..3 {
                world.set_voxel(x, 0, z, Voxel::from_rgb(100, 100, 100));
            }
        }
        world.clear_dirty_flags();

        // Flood fill with new color
        let count = flood_fill(
            &mut world,
            &mut history,
            (1, 0, 1),
            Voxel::from_rgb(255, 0, 0),
            1000,
        );

        assert_eq!(count, 9);
        assert_eq!(world.get_voxel(0, 0, 0).r, 255);
    }

    #[test]
    fn test_flood_fill_bounding_box_caps() {
        // A long thin connected strip extending past MAX_FILL_DIST.
        // The fill must stop at the cap rather than traversing the
        // whole strip.
        let mut world = World::new();
        let mut history = CommandHistory::new(100);

        let strip_len = MAX_FILL_DIST + 50; // well beyond the cap
        let target = Voxel::from_rgb(100, 100, 100);
        for x in 0..strip_len {
            world.set_voxel(x, 0, 0, target);
        }
        world.clear_dirty_flags();

        let count = flood_fill(
            &mut world,
            &mut history,
            (0, 0, 0),
            Voxel::from_rgb(255, 0, 0),
            1_000_000, // generous voxel cap so spatial cap is what bites
        );

        // From start (0,0,0), reachable along +X is x ∈ [0, MAX_FILL_DIST].
        // -X is blocked at the world's edge (0 was the start).
        assert_eq!(count as i32, MAX_FILL_DIST + 1);

        // The cell just past the cap must not have been touched.
        assert_eq!(
            world.get_voxel(MAX_FILL_DIST + 1, 0, 0),
            target
        );
        // The cell at the cap was filled.
        assert_eq!(
            world.get_voxel(MAX_FILL_DIST, 0, 0).r,
            255
        );
    }
}
