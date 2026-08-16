//! Editor tools for voxel manipulation.
//!
//! Provides different brush types and editing modes.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use super::{Command, CommandHistory, RaycastHit, SymmetryAxes, VoxelChange};
use crate::core::{Voxel, World};

/// Time window within which consecutive brush writes coalesce into one
/// undo entry — about five actions a second, so a stroke feels like one
/// operation while separate gestures stay separate.
pub const STROKE_MERGE_WINDOW: Duration = Duration::from_millis(200);

/// Maximum Chebyshev distance `flood_fill` expands from its start cell.
/// `max_voxels` is a count cap, not a spatial one, so without this a
/// fill in an unbounded world could travel arbitrarily far.
pub const MAX_FILL_DIST: i32 = 64;

/// Available editing tools. Brush tools act on the hovered cell per
/// click or drag-step; shape tools run the two-phase footprint → height
/// gesture and commit on a second click. `Select` commits on release.
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
    /// Place a named attachment point at the center of the clicked
    /// face, oriented along its normal. Kept **last** in the enum so the
    /// discriminant stored in `.vxlt` and prefs stays stable.
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
            // No digit free; picked from the toolbar.
            Tool::Socket => "",
        }
    }

    /// Whether this tool uses click-anchor / drag-extent semantics.
    /// Shape tools do, brush tools don't, and `Select` shares the
    /// gesture but commits into `Editor::selection` rather than a world.
    pub fn is_shape(&self) -> bool {
        matches!(self, Tool::Line | Tool::Box | Tool::Sphere | Tool::Cylinder)
    }

    /// Whether this tool needs an anchor cell, and so may use the y=0
    /// ground-plane fallback to work in an empty world. Tools that read
    /// the hovered cell need a real voxel and must not engage it.
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

    /// Positions the hover overlay should highlight, symmetry copies
    /// included and deduped. The caller passes its own `symmetry`, so
    /// one source drives both this preview and the matching `apply`.
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
                        positions.push((center.0 + dx, center.1 + dy, center.2 + dz));
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
            // Eyedropper and Fill go through input.rs's dispatch rather
            // than BrushTool; shape tools and Select have their own
            // gesture lifecycle and never reach this path.
            Tool::Eyedropper
            | Tool::Fill
            | Tool::Line
            | Tool::Box
            | Tool::Sphere
            | Tool::Cylinder
            | Tool::Select
            | Tool::Socket => return,
        };

        // Expand the brush sphere across symmetry mirrors, deduped:
        // spheres overlapping near a plane would otherwise write the
        // same position twice.
        let positions = Self::affected_positions(center, ctx.brush_size, ctx.symmetry);

        let changes: Vec<VoxelChange> = match self.mode {
            // Place writes only into empty cells, so brushing over a
            // model can't punch its color through. Paint recolors
            // solids.
            Tool::Place => positions
                .into_iter()
                .filter_map(|pos| {
                    let old = ctx.world.get_voxel(pos.0, pos.1, pos.2);
                    if old.is_air() {
                        Some(VoxelChange {
                            pos,
                            old_voxel: old,
                            new_voxel: ctx.brush_color,
                        })
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
                        Some(VoxelChange {
                            pos,
                            old_voxel: old,
                            new_voxel: Voxel::AIR,
                        })
                    }
                })
                .collect(),
            Tool::Paint => positions
                .into_iter()
                .filter_map(|pos| {
                    let old = ctx.world.get_voxel(pos.0, pos.1, pos.2);
                    if !old.is_air() && old != ctx.brush_color {
                        Some(VoxelChange {
                            pos,
                            old_voxel: old,
                            new_voxel: ctx.brush_color,
                        })
                    } else {
                        None
                    }
                })
                .collect(),
            _ => return,
        };

        if !changes.is_empty() {
            let cmd = Command::set_voxels(changes);
            ctx.history
                .execute_merge(cmd, ctx.world, STROKE_MERGE_WINDOW);
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
            // Shape tools and Select have their own preview paths and
            // bypass this one. Empty keeps the trait satisfied without
            // contributing stray cells if it is ever called anyway.
            Tool::Line
            | Tool::Box
            | Tool::Sphere
            | Tool::Cylinder
            | Tool::Select
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

/// What a flood fill did. `truncated` is set only when a cell that
/// would genuinely have been written was turned away by a cap — a
/// region ending on its own is not a cap biting.
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
/// Membership is by RGBA alone, and air never belongs whatever its
/// color — one function, so the caps can't drift from the fill.
fn region_voxel(world: &World, pos: (i32, i32, i32), target_rgba: [u8; 4]) -> Option<Voxel> {
    let v = world.get_voxel(pos.0, pos.1, pos.2);
    (!v.is_air() && v.color() == target_rgba).then_some(v)
}

/// The changes a flood fill from `start` would make, without applying
/// them, so several seeds can batch into one undo entry. Membership is
/// by RGBA, so "same color" is not "no change"; the flag is `truncated`.
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
        // Spatial cap: skip cells outside the Chebyshev radius around
        // `start`, so a connected region in an unbounded world can't
        // run far past what the user meant to paint.
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

        // Cap on cells matched, not writes emitted, so a region already
        // holding `new_voxel` is still bounded. Tested here rather than
        // at the top, so `truncated` means a real cell was left out.
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

/// Flood fill from several seeds as one `Command`, so a symmetric
/// stroke is one undo entry. Each seed runs against the original world
/// and carries its own caps, so `truncated` is true if any hit one.
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
        // A solid voxel whose `color()` is [0,0,0,0] reports the same
        // bytes as air, and membership is decided on color — so without
        // an explicit air check the fill floods open space.
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
        assert_eq!(world.get_voxel(MAX_FILL_DIST + 1, 0, 0), target);
        // The cell at the cap was filled.
        assert_eq!(world.get_voxel(MAX_FILL_DIST, 0, 0).r, 255);
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
        assert_eq!(
            world.get_voxel(0, 0, 0),
            existing,
            "Place must not overwrite a solid"
        );

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
        assert_eq!(
            world.get_voxel(1, 0, 0),
            brush,
            "emissive neighbor recolored, flags reset"
        );
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

        let outcome = flood_fill(
            &mut world,
            &mut history,
            (0, 0, 0),
            Voxel::from_rgb(255, 0, 0),
            1000,
        );

        assert_eq!(
            outcome.written, 1,
            "fill can't cross a different-colored cell"
        );
        assert_eq!(world.get_voxel(1, 0, 0), wall, "wall untouched");
        assert_eq!(
            world.get_voxel(2, 0, 0),
            a,
            "cell beyond the wall untouched"
        );
    }

    #[test]
    fn flood_fill_budget_reports_truncation_only_when_it_bites() {
        // A 4-cell strip under a budget of exactly 4: the region ends on
        // the budget, so nothing was left out, even though air
        // neighbors are still queued when the cap is reached.
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
