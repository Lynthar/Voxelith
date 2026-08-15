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
/// `Box`, `Sphere`, `Cylinder`) use the two-phase gesture `app/input`
/// implements: drag a footprint on the locked face plane, release,
/// move the cursor vertically to set height, and a **second click**
/// commits the whole shape as one `Command` (Esc cancels). This doc
/// used to claim mouse-up committed — it doesn't, and the drift
/// between here, the README and the in-app help is exactly what
/// stale gesture descriptions cost. The `Select` tool is the one
/// that *does* commit on release: drag corner-to-corner, release,
/// and the `Selection` AABB lands in `Editor::selection`.
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
    /// Filled cylinder fitting in the drag bbox; axis of revolution =
    /// the locked footprint plane's normal (the height-extrude
    /// direction), ellipse cross-section in the other two.
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
            // Place only writes into empty cells — it never overwrites an
            // existing solid voxel, so brushing over a model (or a large
            // brush straddling one) can't punch its color through. Use
            // Paint to recolor solids.
            Tool::Place => positions
                .into_iter()
                .filter_map(|pos| {
                    let old = ctx.world.get_voxel(pos.0, pos.1, pos.2);
                    if old.is_air() {
                        Some(VoxelChange { pos, old_voxel: old, new_voxel: ctx.brush_color })
                    } else {
                        None
                    }
                })
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

/// What a flood fill did.
///
/// `truncated` is the half a bare count can't express: a fill that
/// wrote every cell of its region and one that hit a cap with more to
/// go report the same number, and Fill is a destructive edit the user
/// is entitled to be told about. It is set only when a cell that would
/// *genuinely have been written* was turned away by one of the two
/// caps — a flood running into air at the radius is the region ending
/// on its own, not a cap biting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FillOutcome {
    /// Cells actually written. A cell already equal to the brush voxel
    /// is traversed (to keep the region connected) but not counted.
    pub written: usize,
    /// A cell belonging to the region was left unfilled by the cell
    /// budget or by [`MAX_FILL_DIST`].
    pub truncated: bool,
}

/// The voxel at `pos` if it belongs to the seed's region, else `None`.
///
/// Membership is decided by RGBA only: cells that share the seed's
/// color belong to one region even when their emissive / metallic /
/// tint-zone bits differ, and each is rewritten with the brush's full
/// voxel (color + material bits). Matching on the whole 8-byte voxel
/// used to stop a fill at a same-colored-but-differently-flagged
/// neighbor and made the result depend on which cell was seeded.
///
/// Air is never part of a region, whatever its color. `Voxel::AIR
/// .color()` is `[0, 0, 0, 0]` — and so is a *solid* voxel built from a
/// fully-transparent palette entry, which `.vox` import can produce.
/// Without that guard, filling such a voxel matched every neighboring
/// air cell and flooded the empty space around it with solid geometry.
///
/// One function decides this so the caps' "was anything actually left
/// out?" test can't drift from the fill's own notion of the region.
fn region_voxel(world: &World, pos: (i32, i32, i32), target_rgba: [u8; 4]) -> Option<Voxel> {
    let v = world.get_voxel(pos.0, pos.1, pos.2);
    (!v.is_air() && v.color() == target_rgba).then_some(v)
}

/// Compute the changes a flood-fill would make from `start`, without
/// applying them. Pulled out of `flood_fill` so callers that need to
/// batch multiple fills into a single undo entry (notably the symmetric
/// fill path in `app::input::apply_tool`) can collect changes from
/// several seeds and submit one combined `Command`.
///
/// Region membership is decided by RGBA alone (see [`region_voxel`]),
/// so a cell whose color already equals `new_voxel`'s but whose
/// material flags or tint zone differ is still rewritten — "same color"
/// is not "no change" here. The result is empty only when the seed's
/// region yields no writes at all.
///
/// The returned flag is `truncated` in the [`FillOutcome`] sense: a
/// cell of the region was left out by a cap.
pub fn compute_flood_fill_changes(
    world: &World,
    start: (i32, i32, i32),
    new_voxel: Voxel,
    max_voxels: usize,
) -> (Vec<VoxelChange>, bool) {
    let target_rgba = world.get_voxel(start.0, start.1, start.2).color();

    let mut changes = Vec::new();
    let mut visited = HashSet::new();
    let mut stack = vec![start];
    let mut truncated = false;

    while let Some(pos) = stack.pop() {
        if visited.contains(&pos) {
            continue;
        }
        // Spatial cap: skip cells outside the chebyshev radius around
        // `start`. Prevents runaway fills in unbounded worlds where
        // the connected region might extend far beyond what the user
        // intended to paint.
        if (pos.0 - start.0).abs() > MAX_FILL_DIST
            || (pos.1 - start.1).abs() > MAX_FILL_DIST
            || (pos.2 - start.2).abs() > MAX_FILL_DIST
        {
            if region_voxel(world, pos, target_rgba).is_some() {
                truncated = true;
            }
            continue;
        }

        let Some(current) = region_voxel(world, pos, target_rgba) else {
            continue;
        };

        // Cap on cells *matched*, not writes emitted: a region already
        // holding `new_voxel` still needs bounding so the flood can't run
        // away in an unbounded world.
        //
        // Tested here rather than at the top of the loop so `truncated`
        // means "a real cell was left out". At the top it also fired on
        // the air neighbors still queued when a region ends exactly on
        // the budget, reporting a truncation that never happened.
        if visited.len() >= max_voxels {
            truncated = true;
            break;
        }

        visited.insert(pos);
        // Emit a write only where it changes something — a cell already
        // equal to the brush voxel is still traversed (to keep the region
        // connected) but doesn't bloat the undo entry or the count.
        if current != new_voxel {
            changes.push(VoxelChange {
                pos,
                old_voxel: current,
                new_voxel,
            });
        }

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

    (changes, truncated)
}

/// Flood fill from a single seed: thin wrapper that computes the
/// changes via `compute_flood_fill_changes` and pushes one `Command`
/// onto `history`.
pub fn flood_fill(
    world: &mut World,
    history: &mut CommandHistory,
    start: (i32, i32, i32),
    new_voxel: Voxel,
    max_voxels: usize,
) -> FillOutcome {
    let (changes, truncated) = compute_flood_fill_changes(world, start, new_voxel, max_voxels);
    let written = changes.len();
    if !changes.is_empty() {
        let cmd = Command::set_voxels(changes);
        history.execute(cmd, world);
    }
    FillOutcome { written, truncated }
}

/// Flood fill from multiple seeds, batching all resulting writes into
/// a single `Command` so the whole symmetric stroke is one undo entry.
/// Each seed's flood is computed against the *original* world snapshot
/// (not the cumulative one), so two seeds spreading toward the same
/// region won't surprise each other; the per-position dedup keeps the
/// first occurrence (any later mirror writing the same cell would
/// produce the same `new_voxel` anyway, so the choice is benign).
/// Each seed carries its own caps, so `truncated` is true when *any*
/// mirror hit one — the stroke as a whole came up short.
pub fn flood_fill_multi(
    world: &mut World,
    history: &mut CommandHistory,
    starts: &[(i32, i32, i32)],
    new_voxel: Voxel,
    max_voxels: usize,
) -> FillOutcome {
    let mut combined: HashMap<(i32, i32, i32), VoxelChange> = HashMap::new();
    let mut truncated = false;
    for &start in starts {
        // Skip air seeds defensively — Fill semantics don't extend air.
        if world.get_voxel(start.0, start.1, start.2).is_air() {
            continue;
        }
        let (changes, seed_truncated) =
            compute_flood_fill_changes(world, start, new_voxel, max_voxels);
        truncated |= seed_truncated;
        for change in changes {
            combined.entry(change.pos).or_insert(change);
        }
    }
    let written = combined.len();
    if written > 0 {
        let cmd = Command::set_voxels(combined.into_values().collect());
        history.execute(cmd, world);
    }
    FillOutcome { written, truncated }
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
        let outcome = flood_fill(
            &mut world,
            &mut history,
            (1, 0, 1),
            Voxel::from_rgb(255, 0, 0),
            1000,
        );

        assert_eq!(outcome.written, 9);
        assert!(!outcome.truncated, "the whole region fit");
        assert_eq!(world.get_voxel(0, 0, 0).r, 255);
    }

    #[test]
    fn test_flood_fill_never_spreads_into_air() {
        // A `.vox` palette entry with alpha 0 imports as a *solid*
        // voxel whose `color()` is [0,0,0,0] — the same bytes
        // `Voxel::AIR` reports. Region membership is decided on color,
        // so without an explicit air check every empty cell around such
        // a voxel matched the seed and the fill flooded open space with
        // geometry.
        let mut world = World::new();
        let mut history = CommandHistory::new(100);

        let transparent = Voxel::from_rgba(0, 0, 0, 0);
        assert!(transparent.is_solid(), "precondition: solid, not air");
        assert_eq!(transparent.color(), Voxel::AIR.color());
        world.set_voxel(0, 0, 0, transparent);
        world.clear_dirty_flags();

        let outcome = flood_fill(
            &mut world,
            &mut history,
            (0, 0, 0),
            Voxel::from_rgb(255, 0, 0),
            10_000,
        );

        assert_eq!(outcome.written, 1, "only the seed voxel itself is filled");
        assert!(
            !outcome.truncated,
            "the region is one cell — the air around it isn't a truncation"
        );
        for neighbor in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)] {
            assert!(
                world.get_voxel(neighbor.0, neighbor.1, neighbor.2).is_air(),
                "air at {neighbor:?} must stay air"
            );
        }
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

        let outcome = flood_fill(
            &mut world,
            &mut history,
            (0, 0, 0),
            Voxel::from_rgb(255, 0, 0),
            1_000_000, // generous voxel cap so spatial cap is what bites
        );

        // From start (0,0,0), reachable along +X is x ∈ [0, MAX_FILL_DIST].
        // -X is blocked at the world's edge (0 was the start).
        assert_eq!(outcome.written as i32, MAX_FILL_DIST + 1);
        assert!(
            outcome.truncated,
            "the strip continues past the radius — the user must be told"
        );

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

    #[test]
    fn place_only_fills_air_never_overwrites_solid() {
        let mut world = World::new();
        let existing = Voxel::from_rgb(0, 0, 255);
        world.set_voxel(0, 0, 0, existing); // a solid the brush must not punch through
        let mut history = CommandHistory::new(100);
        let brush = Voxel::from_rgb(255, 0, 0);
        let tool = BrushTool { mode: Tool::Place };

        // Place writes at adjacent_pos; aim it straight at the occupied cell.
        let hit_occupied = RaycastHit {
            voxel_pos: (0, 0, 0),
            adjacent_pos: (0, 0, 0),
            normal: (0, 1, 0),
            distance: 1.0,
            virtual_ground: false,
        };
        {
            let mut ctx = ToolContext {
                world: &mut world,
                history: &mut history,
                brush_color: brush,
                brush_size: 1,
                symmetry: SymmetryAxes::default(),
            };
            tool.apply(&mut ctx, &hit_occupied);
        }
        assert_eq!(world.get_voxel(0, 0, 0), existing, "Place must not overwrite a solid");

        // Aiming at an empty cell still places normally.
        let hit_air = RaycastHit {
            voxel_pos: (0, 0, 0),
            adjacent_pos: (0, 1, 0),
            normal: (0, 1, 0),
            distance: 1.0,
            virtual_ground: false,
        };
        {
            let mut ctx = ToolContext {
                world: &mut world,
                history: &mut history,
                brush_color: brush,
                brush_size: 1,
                symmetry: SymmetryAxes::default(),
            };
            tool.apply(&mut ctx, &hit_air);
        }
        assert_eq!(world.get_voxel(0, 1, 0), brush, "Place fills empty cells");
    }

    #[test]
    fn flood_fill_matches_on_rgba_ignoring_flags() {
        // Two adjacent cells share RGBA but differ in flags (one is
        // emissive). Region membership is color-only, so a fill seeded on
        // one flows into the other and rewrites both with the brush voxel.
        let mut world = World::new();
        let mut history = CommandHistory::new(100);
        let base = Voxel::from_rgb(50, 100, 150);
        let mut emissive = base;
        emissive.flags = 0b01; // same RGBA as `base`, emissive bit set
        world.set_voxel(0, 0, 0, base);
        world.set_voxel(1, 0, 0, emissive);
        world.clear_dirty_flags();

        let brush = Voxel::from_rgb(255, 0, 0);
        let outcome = flood_fill(&mut world, &mut history, (0, 0, 0), brush, 1000);

        assert_eq!(
            outcome.written, 2,
            "the same-RGBA neighbor must join the region"
        );
        assert_eq!(world.get_voxel(0, 0, 0), brush);
        assert_eq!(world.get_voxel(1, 0, 0), brush, "emissive neighbor recolored, flags reset");
    }

    #[test]
    fn flood_fill_stops_at_different_rgba() {
        let mut world = World::new();
        let mut history = CommandHistory::new(100);
        let a = Voxel::from_rgb(10, 20, 30);
        let wall = Voxel::from_rgb(200, 200, 200);
        world.set_voxel(0, 0, 0, a);
        world.set_voxel(1, 0, 0, wall); // a differently-colored wall
        world.set_voxel(2, 0, 0, a); // same color as seed but unreachable past the wall
        world.clear_dirty_flags();

        let outcome = flood_fill(&mut world, &mut history, (0, 0, 0), Voxel::from_rgb(255, 0, 0), 1000);

        assert_eq!(
            outcome.written, 1,
            "fill can't cross a different-colored cell"
        );
        assert_eq!(world.get_voxel(1, 0, 0), wall, "wall untouched");
        assert_eq!(world.get_voxel(2, 0, 0), a, "cell beyond the wall untouched");
    }

    #[test]
    fn flood_fill_budget_reports_truncation_only_when_it_bites() {
        // A 4-cell strip filled under a budget of exactly 4. The region
        // ends on the budget, so nothing was left out — but the air
        // neighbors of the last cell are still queued when the cap is
        // reached. Testing the cap before deciding those cells aren't
        // part of the region reported a truncation that never happened.
        let mut world = World::new();
        let mut history = CommandHistory::new(100);
        let target = Voxel::from_rgb(100, 100, 100);
        for x in 0..4 {
            world.set_voxel(x, 0, 0, target);
        }
        world.clear_dirty_flags();

        let exact = flood_fill(
            &mut world,
            &mut history,
            (0, 0, 0),
            Voxel::from_rgb(255, 0, 0),
            4,
        );
        assert_eq!(exact.written, 4);
        assert!(
            !exact.truncated,
            "the region ended on the budget — nothing was left out"
        );

        // One cell short: now the cap really does leave a cell behind.
        let mut world = World::new();
        let mut history = CommandHistory::new(100);
        for x in 0..4 {
            world.set_voxel(x, 0, 0, target);
        }
        world.clear_dirty_flags();

        let short = flood_fill(
            &mut world,
            &mut history,
            (0, 0, 0),
            Voxel::from_rgb(255, 0, 0),
            3,
        );
        assert_eq!(short.written, 3);
        assert!(short.truncated, "a cell of the region was refused");
    }
}
