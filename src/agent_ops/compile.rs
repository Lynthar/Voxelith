//! The executor: ops in, one change list out.
//!
//! Ops run against a deep copy of the session world so that each op
//! sees the ones before it (rotating a wall you built two ops ago has
//! to mean something) while the real world stays untouched until the
//! whole batch succeeds.
//!
//! There is no diff pass at the end. Every write goes through
//! [`Scratch::write`], which records `(original, latest)` per cell as
//! it goes — the first touch of a cell reads its pre-batch value
//! straight out of the scratch world, because a cell nobody has
//! touched still holds exactly that. Cost is proportional to what the
//! batch wrote, not to how big the document is, and there's one place
//! where limits, the write mask, and undo bookkeeping all live.

use std::collections::{HashMap, HashSet};

use crate::core::{ChunkPos, Voxel, World};
use crate::editor::{
    box_voxels, cylinder_voxels, line_voxels, mirror_selection_changes, rotate_selection_changes,
    sphere_voxels, Selection, VoxelChange,
};

use super::describe::solid_voxel_count;
use super::registry;
use super::schema::{quarter_from, Aabb, AxisSpec, Op, VoxelSpec, WriteMode};
use super::{
    ErrorCode, OpsError, MAX_BATCH_CELLS, MAX_COORD, MAX_NEW_CHUNKS, MAX_OP_REGION_CELLS,
    MAX_SET_VOXELS_PER_OP,
};

/// The six face neighbors, for shell and hollow tests.
const FACE_NEIGHBORS: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

pub(super) struct Scratch {
    world: World,
    /// `pos → (value before the batch, value now)`.
    changes: HashMap<(i32, i32, i32), (Voxel, Voxel)>,
    selection: Option<Selection>,
    /// Chunks the session already had, so the cap counts only what this
    /// batch allocates.
    base_chunks: usize,
    cells: u64,
    notes: Vec<String>,
}

pub(super) struct Outcome {
    pub changes: Vec<VoxelChange>,
    pub selection: Option<Selection>,
    pub notes: Vec<String>,
    pub voxel_count: u64,
    pub world_aabb: Option<Aabb>,
    /// The scratch world with every op applied. A real run drops it and
    /// commits `changes` instead; a preview hands it back as the world
    /// the batch *would* have produced, so the caller can describe or
    /// slice a result it hasn't accepted yet.
    pub world: World,
}

impl Scratch {
    pub fn new(world: &World, selection: Option<Selection>) -> Self {
        Self {
            world: world.deep_clone(),
            changes: HashMap::new(),
            selection,
            base_chunks: world.chunk_count(),
            cells: 0,
            notes: Vec::new(),
        }
    }

    pub fn finish(self) -> Outcome {
        let mut changes: Vec<VoxelChange> = self
            .changes
            .into_iter()
            .filter(|(_, (old, new))| old != new)
            .map(|(pos, (old_voxel, new_voxel))| VoxelChange {
                pos,
                old_voxel,
                new_voxel,
            })
            .collect();
        // `HashMap` iteration order is unspecified and re-seeded every
        // process. Sorting keeps a batch's command — and so its report
        // and any golden test built on it — identical run to run; x
        // varies fastest to match the chunk array's layout.
        changes.sort_unstable_by_key(|c| (c.pos.2, c.pos.1, c.pos.0));

        Outcome {
            voxel_count: solid_voxel_count(&self.world),
            world_aabb: self.world.scene_aabb().map(Aabb::from_pair),
            changes,
            selection: self.selection,
            notes: self.notes,
            world: self.world,
        }
    }

    pub fn run_op(&mut self, index: usize, op: &Op) -> Result<(), OpsError> {
        match op {
            Op::Box {
                min,
                max,
                voxel,
                filled,
                write_mode,
            } => {
                let region = checked_region(*min, *max)?;
                let cells = box_voxels(region.min, region.max);
                self.write_shape(cells, *filled, voxel, *write_mode)
            }

            Op::Sphere {
                center,
                radius,
                voxel,
                filled,
                write_mode,
            } => {
                let center = tuple(*center);
                check_coord(center)?;
                let r = checked_extent(*radius, "radius", 0)?;
                let region = checked_region(
                    [center.0 - r, center.1 - r, center.2 - r],
                    [center.0 + r, center.1 + r, center.2 + r],
                )?;
                let cells = sphere_voxels(region.min, region.max);
                self.write_shape(cells, *filled, voxel, *write_mode)
            }

            Op::Cylinder {
                base,
                radius,
                height,
                axis,
                voxel,
                filled,
                write_mode,
            } => {
                check_coord(tuple(*base))?;
                let r = checked_extent(*radius, "radius", 0)?;
                let h = checked_extent(*height, "height", 1)?;
                let (min, max) = cylinder_box(*base, r, h, *axis);
                let region = checked_region(min, max)?;
                // The axis hint keeps a short, wide cylinder standing up
                // instead of lying along its longest side.
                let cells = cylinder_voxels(region.min, region.max, Some(axis.index()));
                self.write_shape(cells, *filled, voxel, *write_mode)
            }

            Op::Line {
                from,
                to,
                voxel,
                write_mode,
            } => {
                let (from, to) = (tuple(*from), tuple(*to));
                check_coord(from)?;
                check_coord(to)?;
                check_line_length(from, to)?;
                let cells = line_voxels(from, to);
                // `filled` — a 1-cell line has no interior to remove.
                self.write_shape(cells, true, voxel, *write_mode)
            }

            Op::Hollow { min, max } => self.hollow(checked_region(*min, *max)?),

            Op::SetVoxels { voxels, write_mode } => {
                if voxels.len() > MAX_SET_VOXELS_PER_OP {
                    return Err(OpsError::new(
                        ErrorCode::TooManyVoxels,
                        format!(
                            "{} voxels in one set_voxels op; at most {} are allowed — use a shape or a generator for bulk work",
                            voxels.len(),
                            MAX_SET_VOXELS_PER_OP
                        ),
                    ));
                }
                for entry in voxels {
                    self.write((entry.0, entry.1, entry.2), entry.3.to_voxel()?, *write_mode)?;
                }
                Ok(())
            }

            Op::Generate {
                generator,
                params,
                translate,
                write_mode,
            } => self.generate(index, generator, params, *translate, *write_mode),

            Op::Select { min, max } => {
                self.selection = Some(checked_region(*min, *max)?);
                Ok(())
            }

            Op::Deselect => {
                self.selection = None;
                Ok(())
            }

            Op::Rotate {
                axis,
                quarters,
                region,
            } => {
                let quarter = quarter_from(*quarters)?;
                let (source, from_selection) = self.resolve_region(region)?;
                self.charge(source.cell_count() as u64)?;
                let (rotated, changes) =
                    rotate_selection_changes(&self.world, source, axis.to_axis(), quarter);
                for change in changes {
                    self.write(change.pos, change.new_voxel, WriteMode::Replace)?;
                }
                // The selection follows its own contents; an explicit
                // region is a one-off and leaves the marquee alone.
                if from_selection {
                    self.selection = Some(rotated);
                }
                Ok(())
            }

            Op::Mirror { axis, region } => {
                let (source, _) = self.resolve_region(region)?;
                self.charge(source.cell_count() as u64)?;
                let changes = mirror_selection_changes(&self.world, source, axis.to_axis());
                for change in changes {
                    self.write(change.pos, change.new_voxel, WriteMode::Replace)?;
                }
                Ok(())
            }

            Op::MirrorCopy {
                axis,
                plane,
                region,
                write_mode,
            } => self.mirror_copy(*axis, *plane, region, *write_mode),
        }
    }

    /// The single door every voxel write goes through.
    fn write(
        &mut self,
        pos: (i32, i32, i32),
        voxel: Voxel,
        mode: WriteMode,
    ) -> Result<(), OpsError> {
        check_coord(pos)?;
        // Masked-out cells are charged too: the work is in visiting
        // them, and an op that scans a region shouldn't get to do it
        // for free by writing nothing.
        self.charge(1)?;

        let current = self.world.get_voxel(pos.0, pos.1, pos.2);
        match mode {
            WriteMode::Replace => {}
            WriteMode::OnlyAir if current.is_solid() => return Ok(()),
            WriteMode::OnlySolid if current.is_air() => return Ok(()),
            _ => {}
        }
        if current == voxel {
            return Ok(());
        }
        if !self.world.has_chunk(ChunkPos::from_world_pos(pos.0, pos.1, pos.2))
            && self.world.chunk_count() >= self.base_chunks + MAX_NEW_CHUNKS
        {
            return Err(OpsError::new(
                ErrorCode::WorldTooLarge,
                format!(
                    "this batch would allocate more than {} new chunks (32³ voxels each); \
                     build closer together or split the work across batches",
                    MAX_NEW_CHUNKS
                ),
            ));
        }

        // Vacant means untouched, and an untouched cell still holds its
        // pre-batch value — so `current` is the right `old` to record
        // for undo. Later writes to the same cell only refresh `new`.
        self.changes.entry(pos).or_insert((current, current)).1 = voxel;
        self.world.set_voxel(pos.0, pos.1, pos.2, voxel);
        Ok(())
    }

    fn charge(&mut self, cells: u64) -> Result<(), OpsError> {
        self.cells = self.cells.saturating_add(cells);
        if self.cells > MAX_BATCH_CELLS {
            return Err(OpsError::new(
                ErrorCode::CellBudgetExceeded,
                format!(
                    "this batch touches more than {} cells; split it into several batches",
                    MAX_BATCH_CELLS
                ),
            ));
        }
        Ok(())
    }

    fn write_shape(
        &mut self,
        cells: Vec<(i32, i32, i32)>,
        filled: bool,
        spec: &VoxelSpec,
        mode: WriteMode,
    ) -> Result<(), OpsError> {
        let voxel = spec.to_voxel()?;
        let cells = if filled { cells } else { shell_of(cells) };
        for pos in cells {
            self.write(pos, voxel, mode)?;
        }
        Ok(())
    }

    /// Clear every cell in `region` whose six face neighbors are all
    /// solid.
    fn hollow(&mut self, region: Selection) -> Result<(), OpsError> {
        self.charge(region.cell_count() as u64)?;
        // Classify against the pre-op state, then clear. Clearing as we
        // scan would expose the next cell's neighbor and stop it from
        // qualifying, eroding one layer instead of removing the
        // interior.
        let interior: Vec<(i32, i32, i32)> = region
            .iter_cells()
            .filter(|&(x, y, z)| {
                self.world.get_voxel(x, y, z).is_solid()
                    && FACE_NEIGHBORS.iter().all(|(dx, dy, dz)| {
                        self.world.get_voxel(x + dx, y + dy, z + dz).is_solid()
                    })
            })
            .collect();
        for pos in interior {
            self.write(pos, Voxel::AIR, WriteMode::Replace)?;
        }
        Ok(())
    }

    fn generate(
        &mut self,
        index: usize,
        generator: &str,
        params: &serde_json::Value,
        translate: [i32; 3],
        mode: WriteMode,
    ) -> Result<(), OpsError> {
        check_coord(tuple(translate))?;
        let built = registry::build(generator, params)?;
        let patch = built.generate().map_err(|e| {
            OpsError::new(
                ErrorCode::GeneratorFailed,
                format!("generator {generator} failed: {e}"),
            )
        })?;
        // Patch notes are how a generator reports a degraded result —
        // WFC's over-constrained fallback, for one — and that is
        // exactly what an agent needs to see.
        for note in &patch.notes {
            self.notes.push(format!("op[{index}] {generator}: {note}"));
        }
        // Deduped for the same reason the editor's patch path dedupes:
        // a generator legitimately overwrites its own cell (trunk, then
        // leaf), and only the last value is the intended one.
        for (pos, voxel) in patch.dedup_last_write() {
            // Saturating, not wrapping: a generator handed an extreme
            // origin parameter can emit a position anywhere in `i32`,
            // and the sum must land somewhere `write` will reject
            // rather than wrap around into the middle of the scene.
            let pos = (
                pos.0.saturating_add(translate[0]),
                pos.1.saturating_add(translate[1]),
                pos.2.saturating_add(translate[2]),
            );
            self.write(pos, voxel, mode)?;
        }
        Ok(())
    }

    fn mirror_copy(
        &mut self,
        axis: AxisSpec,
        plane: Option<i32>,
        region: &Option<Aabb>,
        mode: WriteMode,
    ) -> Result<(), OpsError> {
        let (source, _) = self.resolve_region(region)?;
        self.charge(source.cell_count() as u64)?;
        let index = axis.index();
        let plane = match plane {
            Some(plane) => {
                if !(-MAX_COORD..=MAX_COORD).contains(&plane) {
                    return Err(OpsError::new(
                        ErrorCode::CoordinateOutOfRange,
                        format!("mirror plane {plane} is outside ±{MAX_COORD}"),
                    ));
                }
                plane
            }
            // The seam just past the region: the copy lands flush
            // against the original.
            None => component(source.max, index) + 1,
        };

        // Reflect first, write second: with a plane inside the region,
        // writing as we scan would re-read cells we just stamped.
        let stamps: Vec<((i32, i32, i32), Voxel)> = source
            .iter_cells()
            .filter_map(|pos| {
                let voxel = self.world.get_voxel(pos.0, pos.1, pos.2);
                // Air isn't copied — this stamps a shape onto the other
                // side, it doesn't erase what's already there.
                if voxel.is_air() {
                    return None;
                }
                Some((reflect(pos, index, plane), voxel))
            })
            .collect();
        for (pos, voxel) in stamps {
            self.write(pos, voxel, mode)?;
        }
        Ok(())
    }

    /// An op's region: the explicit one if given, else the session
    /// selection. The flag says which, because ops that move their
    /// contents follow the selection only when they *are* the
    /// selection.
    fn resolve_region(&self, region: &Option<Aabb>) -> Result<(Selection, bool), OpsError> {
        let (region, from_selection) = match region {
            Some(aabb) => (aabb.to_selection(), false),
            None => {
                let selection = self.selection.ok_or_else(|| {
                    OpsError::new(
                        ErrorCode::NoSelection,
                        "no region given and nothing is selected; pass \"region\" or run a `select` op first",
                    )
                })?;
                (selection, true)
            }
        };
        // Checked on both paths: `select` validates what it stores, but
        // `AgentSession::selection` is a public field, so a host that
        // sets it directly must not get an unbounded transform.
        check_region(region)?;
        Ok((region, from_selection))
    }
}

fn tuple(p: [i32; 3]) -> (i32, i32, i32) {
    (p[0], p[1], p[2])
}

fn component(p: (i32, i32, i32), index: usize) -> i32 {
    match index {
        0 => p.0,
        1 => p.1,
        _ => p.2,
    }
}

/// Reflect a cell across the seam at `plane`.
///
/// A voxel is the cell `[p, p+1)`, not the point `p`, so its mirror
/// image across the seam `x = plane` is `2·plane − 1 − p`. Using the
/// point formula (`2·plane − p`) shifts the copy one cell into the
/// original.
fn reflect(pos: (i32, i32, i32), index: usize, plane: i32) -> (i32, i32, i32) {
    let mirrored = 2 * plane - 1 - component(pos, index);
    match index {
        0 => (mirrored, pos.1, pos.2),
        1 => (pos.0, mirrored, pos.2),
        _ => (pos.0, pos.1, mirrored),
    }
}

fn cylinder_box(base: [i32; 3], radius: i32, height: i32, axis: AxisSpec) -> ([i32; 3], [i32; 3]) {
    let mut min = base;
    let mut max = base;
    for (i, (lo, hi)) in min.iter_mut().zip(max.iter_mut()).enumerate() {
        if i == axis.index() {
            *hi = base[i] + height - 1;
        } else {
            *lo = base[i] - radius;
            *hi = base[i] + radius;
        }
    }
    (min, max)
}

/// The 1-cell outer layer of a cell set: every cell with at least one
/// face neighbor outside it.
///
/// Shared by every shape's `filled: false`, so a hollow sphere is the
/// shell of exactly the sphere the filled version would have written —
/// no second rasterizer to disagree with the first.
fn shell_of(cells: Vec<(i32, i32, i32)>) -> Vec<(i32, i32, i32)> {
    let inside: HashSet<(i32, i32, i32)> = cells.iter().copied().collect();
    cells
        .into_iter()
        .filter(|&(x, y, z)| {
            FACE_NEIGHBORS
                .iter()
                .any(|(dx, dy, dz)| !inside.contains(&(x + dx, y + dy, z + dz)))
        })
        .collect()
}

fn check_coord(pos: (i32, i32, i32)) -> Result<(), OpsError> {
    // A range test rather than `abs()`: `i32::MIN.abs()` overflows — a
    // debug panic, and in release it wraps back to `i32::MIN`, which
    // would sail straight through a `> MAX_COORD` check.
    let out_of_range = |v: i32| !(-MAX_COORD..=MAX_COORD).contains(&v);
    if out_of_range(pos.0) || out_of_range(pos.1) || out_of_range(pos.2) {
        return Err(OpsError::new(
            ErrorCode::CoordinateOutOfRange,
            format!(
                "coordinate {:?} is outside ±{} on some axis",
                pos, MAX_COORD
            ),
        ));
    }
    Ok(())
}

/// Bound a line by the cells it visits, which is not the volume of its
/// bounding box.
///
/// `line_voxels` is a Bresenham walk: one cell per step along the
/// dominant axis. The diagonal from `[0,0,0]` to `[500,500,500]` is 501
/// writes, while its bounding box is 125 million cells — measuring it
/// the way a box is measured refused a long diagonal beam that costs
/// almost nothing, and said "region 501×501×501" while doing it.
fn check_line_length(a: (i32, i32, i32), b: (i32, i32, i32)) -> Result<(), OpsError> {
    // In i64 so the subtraction can't overflow on its own terms, rather
    // than only because `check_coord` happens to run first.
    let span = |p: i32, q: i32| (p as i64 - q as i64).unsigned_abs();
    let cells = span(b.0, a.0).max(span(b.1, a.1)).max(span(b.2, a.2)) + 1;
    if cells > MAX_OP_REGION_CELLS {
        return Err(OpsError::new(
            ErrorCode::RegionTooLarge,
            format!(
                "line spans {} cells; a single op may cover at most {}",
                cells, MAX_OP_REGION_CELLS
            ),
        ));
    }
    Ok(())
}

fn checked_extent(value: i32, name: &str, min: i32) -> Result<i32, OpsError> {
    if value < min || value > MAX_COORD {
        return Err(OpsError::new(
            ErrorCode::InvalidArgument,
            format!("{name} must be between {min} and {MAX_COORD}, got {value}"),
        ));
    }
    Ok(value)
}

/// Normalize a pair of corners into a region and check it fits the
/// per-op limits.
fn checked_region(a: [i32; 3], b: [i32; 3]) -> Result<Selection, OpsError> {
    let region = Selection::from_corners(tuple(a), tuple(b));
    check_region(region)?;
    Ok(region)
}

fn check_region(region: Selection) -> Result<(), OpsError> {
    check_coord(region.min)?;
    check_coord(region.max)?;
    let (w, h, d) = region.size();
    let cells = w as u64 * h as u64 * d as u64;
    if cells > MAX_OP_REGION_CELLS {
        return Err(OpsError::new(
            ErrorCode::RegionTooLarge,
            format!(
                "region {}×{}×{} is {} cells; a single op may cover at most {}",
                w, h, d, cells, MAX_OP_REGION_CELLS
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_ops::{AgentSession, ApplyReport, OpsBatch, MAX_OPS_PER_BATCH};

    fn parse(json: &str) -> OpsBatch {
        serde_json::from_str(json).expect("test batch should parse")
    }

    fn apply(session: &mut AgentSession, json: &str) -> ApplyReport {
        session
            .apply_ops(&parse(json))
            .unwrap_or_else(|e| panic!("batch should apply, got {e}"))
    }

    fn refuse(session: &mut AgentSession, json: &str) -> OpsError {
        session
            .apply_ops(&parse(json))
            .expect_err("batch should have been refused")
    }

    /// Every solid voxel, sorted — for byte-exact world comparisons.
    fn snapshot(world: &World) -> Vec<((i32, i32, i32), Voxel)> {
        let mut out = Vec::new();
        for pos in world.sorted_chunk_positions() {
            let chunk = world.get_chunk(pos).unwrap();
            let (ox, oy, oz) = pos.world_origin();
            for (local, voxel) in chunk.read().iter_solid() {
                out.push((
                    (ox + local.x as i32, oy + local.y as i32, oz + local.z as i32),
                    *voxel,
                ));
            }
        }
        out.sort_by_key(|(pos, _)| *pos);
        out
    }

    fn solid_count(world: &World) -> usize {
        snapshot(world).len()
    }

    // -------- shapes --------

    #[test]
    fn box_covers_the_closed_region() {
        let mut session = AgentSession::new();
        let report = apply(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"box","min":[0,0,0],"max":[1,1,1],"voxel":{"rgb":[10,20,30]}}
            ]}"#,
        );
        assert_eq!(report.changed_voxels, 8);
        assert_eq!(session.world.get_voxel(0, 0, 0).color(), [10, 20, 30, 255]);
        assert_eq!(session.world.get_voxel(1, 1, 1).color(), [10, 20, 30, 255]);
        assert_eq!(report.world_aabb, Some(Aabb { min: [0, 0, 0], max: [1, 1, 1] }));
    }

    #[test]
    fn unfilled_box_is_a_one_cell_shell() {
        let mut session = AgentSession::new();
        let report = apply(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"box","min":[0,0,0],"max":[4,4,4],"voxel":{"rgb":[1,2,3]},"filled":false}
            ]}"#,
        );
        // 5³ minus the 3³ interior.
        assert_eq!(report.changed_voxels, 125 - 27);
        assert!(session.world.get_voxel(2, 2, 2).is_air(), "interior must be empty");
        assert!(session.world.get_voxel(2, 2, 0).is_solid(), "face must be solid");
    }

    #[test]
    fn air_voxel_erases() {
        let mut session = AgentSession::new();
        apply(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"box","min":[0,0,0],"max":[3,3,3],"voxel":{"rgb":[1,2,3]}},
                {"op":"box","min":[1,1,1],"max":[2,2,2],"voxel":"air"}
            ]}"#,
        );
        assert_eq!(solid_count(&session.world), 64 - 8);
        assert!(session.world.get_voxel(1, 1, 1).is_air());
    }

    #[test]
    fn sphere_matches_the_editor_shape_tool() {
        // The op is center+radius; the tool is corner-to-corner. Same
        // rasterizer underneath, so the cell sets must be identical.
        let mut session = AgentSession::new();
        let report = apply(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"sphere","center":[0,0,0],"radius":3,"voxel":{"rgb":[9,9,9]}}
            ]}"#,
        );
        let expected = sphere_voxels((-3, -3, -3), (3, 3, 3));
        assert_eq!(report.changed_voxels, expected.len());
        for pos in expected {
            assert!(
                session.world.get_voxel(pos.0, pos.1, pos.2).is_solid(),
                "missing sphere cell {pos:?}"
            );
        }
    }

    #[test]
    fn cylinder_stands_along_its_axis() {
        let mut session = AgentSession::new();
        apply(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"cylinder","base":[0,0,0],"radius":1,"height":5,"voxel":{"rgb":[9,9,9]}}
            ]}"#,
        );
        for y in 0..5 {
            assert!(session.world.get_voxel(0, y, 0).is_solid(), "missing y={y}");
        }
        assert!(session.world.get_voxel(0, 5, 0).is_air(), "height is 5 cells, 0..=4");
    }

    #[test]
    fn cylinder_axis_x_extends_along_x() {
        let mut session = AgentSession::new();
        apply(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"cylinder","base":[0,0,0],"radius":1,"height":4,"axis":"x","voxel":{"rgb":[9,9,9]}}
            ]}"#,
        );
        for x in 0..4 {
            assert!(session.world.get_voxel(x, 0, 0).is_solid(), "missing x={x}");
        }
        assert!(session.world.get_voxel(4, 0, 0).is_air());
    }

    #[test]
    fn line_includes_both_endpoints() {
        let mut session = AgentSession::new();
        let report = apply(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"line","from":[0,0,0],"to":[4,0,0],"voxel":{"rgb":[9,9,9]}}
            ]}"#,
        );
        assert_eq!(report.changed_voxels, 5);
        assert!(session.world.get_voxel(0, 0, 0).is_solid());
        assert!(session.world.get_voxel(4, 0, 0).is_solid());
    }

    #[test]
    fn a_long_diagonal_line_costs_its_steps_not_its_bounding_box() {
        // A 501-cell beam used to come back refused as a "501×501×501
        // region" of 125 million cells. A Bresenham line visits one cell
        // per step along the dominant axis; the box it happens to span
        // is not what it costs.
        let mut session = AgentSession::new();
        let report = apply(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"line","from":[0,0,0],"to":[500,500,500],"voxel":{"rgb":[9,9,9]}}
            ]}"#,
        );
        assert_eq!(report.changed_voxels, 501);
        assert!(session.world.get_voxel(500, 500, 500).is_solid());
    }

    #[test]
    fn a_line_longer_than_the_per_op_cap_is_still_refused() {
        // The cap moved to the step count, so only a line spanning
        // essentially the whole coordinate range reaches it — but it is
        // still an explicit error rather than an allocation nobody
        // bounded.
        let mut session = AgentSession::new();
        let json = format!(
            r#"{{"version":1,"ops":[
                {{"op":"line","from":[{},0,0],"to":[{},0,0],"voxel":{{"rgb":[1,2,3]}}}}
            ]}}"#,
            -MAX_COORD, MAX_COORD
        );
        let error = refuse(&mut session, &json);
        assert_eq!(error.code, ErrorCode::RegionTooLarge);
        assert!(error.message.contains("line"), "got: {}", error.message);
        assert_eq!(session.world.chunk_count(), 0);
    }

    #[test]
    fn hollow_removes_the_whole_interior_at_once() {
        // The load-bearing detail: cells are classified against the
        // pre-op state. Clearing while scanning would expose the next
        // cell's neighbor and stop it from qualifying, so only a
        // fraction of the 27 interior cells would go.
        let mut session = AgentSession::new();
        apply(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"box","min":[0,0,0],"max":[4,4,4],"voxel":{"rgb":[1,2,3]}}
            ]}"#,
        );
        let report = apply(
            &mut session,
            r#"{"version":1,"ops":[{"op":"hollow","min":[0,0,0],"max":[4,4,4]}]}"#,
        );
        assert_eq!(report.changed_voxels, 27, "all 3³ interior cells must go");
        assert_eq!(solid_count(&session.world), 125 - 27);
    }

    // -------- write modes --------

    #[test]
    fn only_air_builds_around_what_is_there() {
        let mut session = AgentSession::new();
        apply(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"box","min":[0,0,0],"max":[0,0,0],"voxel":{"rgb":[255,0,0]}},
                {"op":"box","min":[0,0,0],"max":[2,0,0],"voxel":{"rgb":[0,255,0]},"write_mode":"only_air"}
            ]}"#,
        );
        assert_eq!(session.world.get_voxel(0, 0, 0).color(), [255, 0, 0, 255], "occupied cell kept");
        assert_eq!(session.world.get_voxel(1, 0, 0).color(), [0, 255, 0, 255]);
    }

    #[test]
    fn only_solid_repaints_without_growing() {
        let mut session = AgentSession::new();
        apply(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"box","min":[0,0,0],"max":[0,0,0],"voxel":{"rgb":[255,0,0]}},
                {"op":"box","min":[0,0,0],"max":[2,0,0],"voxel":{"rgb":[0,0,255]},"write_mode":"only_solid"}
            ]}"#,
        );
        assert_eq!(session.world.get_voxel(0, 0, 0).color(), [0, 0, 255, 255], "repainted");
        assert!(session.world.get_voxel(1, 0, 0).is_air(), "must not grow into air");
    }

    #[test]
    fn material_flags_and_tint_zone_reach_the_voxel() {
        let mut session = AgentSession::new();
        apply(
            &mut session,
            r#"{"version":1,"ops":[{"op":"set_voxels","voxels":[
                [0,0,0,{"rgb":[1,2,3],"emissive":true,"tint_zone":2}],
                [1,0,0,{"rgb":[4,5,6],"metallic":true,"material":7}]
            ]}]}"#,
        );
        let a = session.world.get_voxel(0, 0, 0);
        assert!(a.is_emissive() && !a.is_metallic());
        assert_eq!(a.tint_zone(), 2);
        let b = session.world.get_voxel(1, 0, 0);
        assert!(b.is_metallic());
        assert_eq!(b.material, 7);
    }

    #[test]
    fn set_voxels_over_the_cap_is_refused() {
        let entries: Vec<String> = (0..=MAX_SET_VOXELS_PER_OP)
            .map(|i| format!("[{i},0,0,{{\"rgb\":[1,2,3]}}]"))
            .collect();
        let json = format!(
            r#"{{"version":1,"ops":[{{"op":"set_voxels","voxels":[{}]}}]}}"#,
            entries.join(",")
        );
        let mut session = AgentSession::new();
        let error = refuse(&mut session, &json);
        assert_eq!(error.code, ErrorCode::TooManyVoxels);
        assert_eq!(error.op_index, Some(0));
    }

    // -------- selection, transforms --------

    #[test]
    fn rotate_uses_the_selection_and_moves_it_along() {
        let mut session = AgentSession::new();
        let report = apply(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"box","min":[0,0,0],"max":[3,0,0],"voxel":{"rgb":[1,2,3]}},
                {"op":"select","min":[0,0,0],"max":[3,0,0]},
                {"op":"rotate","axis":"y","quarters":1}
            ]}"#,
        );
        // 4×1×1 rotated about Y is 1×1×4, anchored at the same min.
        assert_eq!(report.selection, Some(Aabb { min: [0, 0, 0], max: [0, 0, 3] }));
        assert_eq!(session.selection.map(Aabb::from), report.selection);
        assert!(session.world.get_voxel(0, 0, 3).is_solid());
        assert!(session.world.get_voxel(3, 0, 0).is_air());
    }

    #[test]
    fn an_explicit_region_leaves_the_selection_alone() {
        let mut session = AgentSession::new();
        apply(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"box","min":[0,0,0],"max":[3,0,0],"voxel":{"rgb":[1,2,3]}},
                {"op":"select","min":[10,10,10],"max":[11,11,11]},
                {"op":"rotate","axis":"y","quarters":1,"region":{"min":[0,0,0],"max":[3,0,0]}}
            ]}"#,
        );
        assert_eq!(
            session.selection.map(Aabb::from),
            Some(Aabb { min: [10, 10, 10], max: [11, 11, 11] })
        );
    }

    #[test]
    fn a_transform_without_a_region_or_selection_says_so() {
        let mut session = AgentSession::new();
        let error = refuse(
            &mut session,
            r#"{"version":1,"ops":[{"op":"mirror","axis":"x"}]}"#,
        );
        assert_eq!(error.code, ErrorCode::NoSelection);
    }

    #[test]
    fn deselect_clears_the_selection() {
        let mut session = AgentSession::new();
        let report = apply(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"select","min":[0,0,0],"max":[1,1,1]},
                {"op":"deselect"}
            ]}"#,
        );
        assert!(report.selection.is_none());
        assert!(session.selection.is_none());
    }

    #[test]
    fn mirror_flips_the_region_in_place() {
        let mut session = AgentSession::new();
        apply(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"set_voxels","voxels":[[0,0,0,{"rgb":[255,0,0]}],[2,0,0,{"rgb":[0,0,255]}]]},
                {"op":"mirror","axis":"x","region":{"min":[0,0,0],"max":[2,0,0]}}
            ]}"#,
        );
        assert_eq!(session.world.get_voxel(0, 0, 0).color(), [0, 0, 255, 255]);
        assert_eq!(session.world.get_voxel(2, 0, 0).color(), [255, 0, 0, 255]);
    }

    #[test]
    fn mirror_copy_lands_flush_against_the_original() {
        let mut session = AgentSession::new();
        apply(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"box","min":[0,0,0],"max":[3,0,0],"voxel":{"rgb":[1,2,3]}},
                {"op":"mirror_copy","axis":"x","region":{"min":[0,0,0],"max":[3,0,0]}}
            ]}"#,
        );
        // Cells 0..=3 reflect to 7..=4 across the seam at x = 4: the
        // copy touches the original with no gap and no overlap.
        assert_eq!(solid_count(&session.world), 8);
        for x in 0..8 {
            assert!(session.world.get_voxel(x, 0, 0).is_solid(), "missing x={x}");
        }
    }

    #[test]
    fn mirror_copy_across_the_origin_matches_editor_symmetry() {
        // `SymmetryAxes` mirrors cell n to -n-1; plane 0 must agree, or
        // the two symmetry features in the app disagree by one cell.
        let mut session = AgentSession::new();
        apply(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"box","min":[0,0,0],"max":[2,0,0],"voxel":{"rgb":[1,2,3]}},
                {"op":"mirror_copy","axis":"x","plane":0,"region":{"min":[0,0,0],"max":[2,0,0]}}
            ]}"#,
        );
        for x in -3..3 {
            assert!(session.world.get_voxel(x, 0, 0).is_solid(), "missing x={x}");
        }
        assert!(session.world.get_voxel(-4, 0, 0).is_air());
    }

    #[test]
    fn mirror_copy_stamps_solids_and_leaves_air_alone() {
        let mut session = AgentSession::new();
        apply(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"set_voxels","voxels":[[0,0,0,{"rgb":[255,0,0]}],[5,0,0,{"rgb":[0,255,0]}]]},
                {"op":"mirror_copy","axis":"x","plane":3,"region":{"min":[0,0,0],"max":[2,0,0]}}
            ]}"#,
        );
        // Only (0,0,0) is solid in the region; it reflects to x = 5,
        // where something already sits. Air cells 1 and 2 reflect onto
        // 4 and 3 but must not erase anything.
        assert_eq!(session.world.get_voxel(5, 0, 0).color(), [255, 0, 0, 255]);
        assert!(session.world.get_voxel(3, 0, 0).is_air());
        assert!(session.world.get_voxel(4, 0, 0).is_air());
    }

    #[test]
    fn a_generated_tree_mirrors_to_exactly_twice_itself() {
        // Golden-shaped check without magic numbers: a disjoint mirror
        // copy doubles both the voxel count and the width. An
        // off-by-one in the reflection makes the halves overlap, and
        // the count comes in low.
        let mut session = AgentSession::new();
        let grown = apply(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"generate","generator":"builtin.lsystem_tree","params":{"seed":7,"iterations":3}}
            ]}"#,
        );
        let before = grown.world_aabb.expect("the tree should exist");
        let width = before.max[0] - before.min[0] + 1;

        let mirrored = apply(
            &mut session,
            &format!(
                r#"{{"version":1,"ops":[
                    {{"op":"mirror_copy","axis":"x","region":{{"min":[{},{},{}],"max":[{},{},{}]}}}}
                ]}}"#,
                before.min[0], before.min[1], before.min[2],
                before.max[0], before.max[1], before.max[2],
            ),
        );
        assert_eq!(mirrored.voxel_count, grown.voxel_count * 2);
        let after = mirrored.world_aabb.expect("still there");
        assert_eq!(after.max[0] - after.min[0] + 1, width * 2);
        assert_eq!(after.min[0], before.min[0], "the original half must not move");
    }

    // -------- generators --------

    #[test]
    fn generate_offsets_the_patch_by_translate() {
        let mut session = AgentSession::new();
        let plain = apply(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"generate","generator":"builtin.perlin_terrain","params":{"width":8,"depth":8}}
            ]}"#,
        );
        let mut moved_session = AgentSession::new();
        let moved = apply(
            &mut moved_session,
            r#"{"version":1,"ops":[
                {"op":"generate","generator":"builtin.perlin_terrain","params":{"width":8,"depth":8},
                 "translate":[100,0,0]}
            ]}"#,
        );
        assert_eq!(plain.voxel_count, moved.voxel_count);
        let (a, b) = (plain.world_aabb.unwrap(), moved.world_aabb.unwrap());
        assert_eq!(b.min[0] - a.min[0], 100);
        assert_eq!(b.min[1], a.min[1]);
    }

    #[test]
    fn a_generator_placed_past_the_edge_of_the_world_is_refused() {
        // The patch position and the offset are both `i32`; their sum
        // must not wrap the model back into the middle of the scene.
        let mut session = AgentSession::new();
        let error = refuse(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"generate","generator":"builtin.lsystem_tree","params":{"iterations":1},
                 "translate":[2147483647,0,0]}
            ]}"#,
        );
        assert_eq!(error.code, ErrorCode::CoordinateOutOfRange);
        assert_eq!(session.world.chunk_count(), 0);
    }

    #[test]
    fn an_unknown_generator_lists_the_real_ones() {
        let mut session = AgentSession::new();
        let error = refuse(
            &mut session,
            r#"{"version":1,"ops":[{"op":"generate","generator":"builtin.castle"}]}"#,
        );
        assert_eq!(error.code, ErrorCode::UnknownGenerator);
        assert!(
            error.message.contains("builtin.perlin_terrain"),
            "the message should name what is available, got: {}",
            error.message
        );
    }

    // -------- limits --------

    #[test]
    fn an_oversized_region_is_refused_before_it_allocates() {
        let mut session = AgentSession::new();
        let error = refuse(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"box","min":[0,0,0],"max":[200,200,200],"voxel":{"rgb":[1,2,3]}}
            ]}"#,
        );
        assert_eq!(error.code, ErrorCode::RegionTooLarge);
        assert_eq!(session.world.chunk_count(), 0);
    }

    #[test]
    fn a_far_away_coordinate_is_an_error_not_a_silent_write() {
        // `World::set_voxel` happily creates a chunk anywhere; the point
        // of the ceiling is that a runaway coordinate gets reported.
        let mut session = AgentSession::new();
        let error = refuse(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"set_voxels","voxels":[[99999999,0,0,{"rgb":[1,2,3]}]]}
            ]}"#,
        );
        assert_eq!(error.code, ErrorCode::CoordinateOutOfRange);
    }

    #[test]
    fn the_most_negative_coordinate_is_refused_without_overflowing() {
        // `i32::MIN.abs()` panics in debug and wraps back to `i32::MIN`
        // in release, which would walk straight through a naive range
        // test. Both builds must refuse it.
        let mut session = AgentSession::new();
        let json = format!(
            r#"{{"version":1,"ops":[{{"op":"set_voxels","voxels":[[{},0,0,{{"rgb":[1,2,3]}}]]}}]}}"#,
            i32::MIN
        );
        assert_eq!(
            refuse(&mut session, &json).code,
            ErrorCode::CoordinateOutOfRange
        );
    }

    #[test]
    fn a_selection_the_host_set_is_still_bounded() {
        // `AgentSession::selection` is public for the CLI and MCP hosts
        // to drive; a transform must not inherit an unchecked region
        // from it.
        let mut session = AgentSession::new();
        session.selection = Some(Selection::from_corners((0, 0, 0), (999, 999, 999)));
        let error = refuse(
            &mut session,
            r#"{"version":1,"ops":[{"op":"mirror","axis":"x"}]}"#,
        );
        assert_eq!(error.code, ErrorCode::RegionTooLarge);
    }

    #[test]
    fn the_cell_budget_stops_a_batch() {
        let mut scratch = Scratch::new(&World::new(), None);
        assert!(scratch.charge(MAX_BATCH_CELLS).is_ok());
        assert_eq!(
            scratch.charge(1).unwrap_err().code,
            ErrorCode::CellBudgetExceeded
        );
    }

    #[test]
    fn every_visited_cell_is_charged_including_masked_ones() {
        let mut scratch = Scratch::new(&World::new(), None);
        let voxel = Voxel::from_rgb(1, 2, 3);
        scratch.write((0, 0, 0), voxel, WriteMode::Replace).unwrap();
        // Masked out (the cell is air), but it was still visited.
        scratch.write((1, 0, 0), voxel, WriteMode::OnlySolid).unwrap();
        assert_eq!(scratch.cells, 2);
        assert_eq!(scratch.changes.len(), 1);
    }

    #[test]
    fn too_many_ops_is_refused_whole() {
        let ops: Vec<&str> = vec![r#"{"op":"deselect"}"#; MAX_OPS_PER_BATCH + 1];
        let json = format!(r#"{{"version":1,"ops":[{}]}}"#, ops.join(","));
        let mut session = AgentSession::new();
        let error = refuse(&mut session, &json);
        assert_eq!(error.code, ErrorCode::TooManyOps);
        assert!(error.op_index.is_none(), "envelope errors have no op index");
    }

    #[test]
    fn an_unknown_version_is_refused() {
        let mut session = AgentSession::new();
        let error = refuse(&mut session, r#"{"version":99,"ops":[{"op":"deselect"}]}"#);
        assert_eq!(error.code, ErrorCode::UnsupportedVersion);
    }

    #[test]
    fn an_empty_batch_is_a_mistake_worth_reporting() {
        let mut session = AgentSession::new();
        assert_eq!(
            refuse(&mut session, r#"{"version":1,"ops":[]}"#).code,
            ErrorCode::InvalidArgument
        );
    }

    // -------- atomicity, undo, dry run --------

    #[test]
    fn a_failed_batch_leaves_nothing_behind() {
        let mut session = AgentSession::new();
        apply(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"box","min":[0,0,0],"max":[1,1,1],"voxel":{"rgb":[1,2,3]}},
                {"op":"select","min":[0,0,0],"max":[1,1,1]}
            ]}"#,
        );
        let before = snapshot(&session.world);
        let undo_depth = session.history.undo_count();

        let error = refuse(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"box","min":[5,5,5],"max":[6,6,6],"voxel":{"rgb":[9,9,9]}},
                {"op":"rotate","axis":"y","quarters":9}
            ]}"#,
        );
        assert_eq!(error.op_index, Some(1), "the failing op is named");
        assert_eq!(snapshot(&session.world), before, "op 0 must not have landed");
        assert_eq!(session.history.undo_count(), undo_depth, "no undo entry pushed");
        assert_eq!(
            session.selection.map(Aabb::from),
            Some(Aabb { min: [0, 0, 0], max: [1, 1, 1] }),
            "selection must survive a failed batch"
        );
    }

    #[test]
    fn a_batch_is_one_undo_entry_and_restores_exactly() {
        let mut session = AgentSession::new();
        apply(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"box","min":[0,0,0],"max":[3,3,3],"voxel":{"rgb":[100,100,100]}}
            ]}"#,
        );
        let before = snapshot(&session.world);

        apply(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"box","min":[1,4,1],"max":[2,5,2],"voxel":{"rgb":[200,50,50]}},
                {"op":"hollow","min":[0,0,0],"max":[3,3,3]},
                {"op":"mirror_copy","axis":"x","region":{"min":[0,0,0],"max":[3,5,3]}}
            ]}"#,
        );
        assert_eq!(session.history.undo_count(), 2, "one entry per batch");
        assert_ne!(snapshot(&session.world), before);

        assert!(session.undo());
        assert_eq!(snapshot(&session.world), before, "undo must restore exactly");
        assert!(session.redo());
        assert_ne!(snapshot(&session.world), before);
    }

    #[test]
    fn a_repeated_batch_changes_nothing_the_second_time() {
        let batch = r#"{"version":1,"ops":[
            {"op":"box","min":[0,0,0],"max":[3,3,3],"voxel":{"rgb":[1,2,3]}}
        ]}"#;
        let mut session = AgentSession::new();
        assert_eq!(apply(&mut session, batch).changed_voxels, 64);
        let again = apply(&mut session, batch);
        assert_eq!(again.changed_voxels, 0, "identity writes aren't changes");
        assert_eq!(session.history.undo_count(), 1, "a no-op batch pushes no undo entry");
    }

    #[test]
    fn a_dry_run_reports_what_the_real_run_does_and_writes_nothing() {
        let batch = r#"{"version":1,"ops":[
            {"op":"box","min":[0,0,0],"max":[4,4,4],"voxel":{"rgb":[1,2,3]}},
            {"op":"hollow","min":[0,0,0],"max":[4,4,4]},
            {"op":"select","min":[0,0,0],"max":[4,4,4]}
        ]}"#;
        let dry_json = batch.replace(r#""ops""#, r#""options":{"dry_run":true},"ops""#);

        let mut session = AgentSession::new();
        let dry = session.apply_ops(&parse(&dry_json)).expect("dry run");
        assert_eq!(session.world.chunk_count(), 0, "a dry run must not write");
        assert!(session.selection.is_none(), "nor move the selection");
        assert!(!session.history.can_undo());

        let real = apply(&mut session, batch);
        let mut dry_json = serde_json::to_value(&dry).unwrap();
        dry_json["dry_run"] = serde_json::Value::Bool(false);
        assert_eq!(
            dry_json,
            serde_json::to_value(&real).unwrap(),
            "a dry run must predict the real run exactly"
        );
    }

    #[test]
    fn a_preview_hands_back_the_world_the_batch_would_leave() {
        // What a dry run is *for*: describing or slicing a result before
        // accepting it. Without the preview, a dry run reported the
        // world after the batch while a description taken alongside it
        // showed the world before — two answers in one envelope.
        let batch = parse(
            r#"{"version":1,"ops":[
                {"op":"box","min":[0,0,0],"max":[4,4,4],"voxel":{"rgb":[1,2,3]}},
                {"op":"hollow","min":[0,0,0],"max":[4,4,4]},
                {"op":"select","min":[0,0,0],"max":[4,4,4]}
            ]}"#,
        );
        let mut session = AgentSession::new();
        let preview = session.preview_ops(&batch).expect("preview should run");

        assert!(preview.report.dry_run, "a preview commits nothing");
        assert_eq!(session.world.chunk_count(), 0, "the session is untouched");
        assert!(session.selection.is_none());
        assert!(!session.history.can_undo());

        // The preview session answers "what would I be looking at" with
        // the same numbers the report gives.
        assert_eq!(
            preview.session.describe().voxel_count,
            preview.report.voxel_count
        );
        assert_eq!(
            preview.session.selection.map(Aabb::from),
            preview.report.selection
        );

        // …and running the same batch for real lands exactly there.
        let real = session.apply_ops(&batch).expect("real run should apply");
        assert_eq!(snapshot(&session.world), snapshot(&preview.session.world));
        assert_eq!(real.voxel_count, preview.report.voxel_count);
        assert_eq!(real.changed_voxels, preview.report.changed_voxels);
        assert_eq!(real.world_aabb, preview.report.world_aabb);
        assert_eq!(real.selection, preview.report.selection);
    }

    #[test]
    fn later_ops_see_what_earlier_ops_wrote() {
        // The whole reason ops run against a scratch *world* rather than
        // being compiled independently.
        let mut session = AgentSession::new();
        apply(
            &mut session,
            r#"{"version":1,"ops":[
                {"op":"box","min":[0,0,0],"max":[2,0,0],"voxel":{"rgb":[255,0,0]}},
                {"op":"box","min":[0,0,0],"max":[4,0,0],"voxel":{"rgb":[0,255,0]},"write_mode":"only_air"},
                {"op":"mirror","axis":"x","region":{"min":[0,0,0],"max":[4,0,0]}}
            ]}"#,
        );
        // only_air filled 3..=4 with green; the mirror then swaps the
        // row end for end.
        assert_eq!(session.world.get_voxel(0, 0, 0).color(), [0, 255, 0, 255]);
        assert_eq!(session.world.get_voxel(4, 0, 0).color(), [255, 0, 0, 255]);
    }

    #[test]
    fn the_change_list_is_ordered_the_same_way_every_run() {
        // `HashMap` iteration is re-seeded per process, so without the
        // sort in `finish` the command — and any golden built on it —
        // would shuffle between runs.
        let batch = r#"{"version":1,"ops":[
            {"op":"box","min":[0,0,0],"max":[5,5,5],"voxel":{"rgb":[1,2,3]}}
        ]}"#;
        let mut first = Scratch::new(&World::new(), None);
        first.run_op(0, &parse(batch).ops[0]).unwrap();
        let mut second = Scratch::new(&World::new(), None);
        second.run_op(0, &parse(batch).ops[0]).unwrap();

        let a: Vec<_> = first.finish().changes.iter().map(|c| c.pos).collect();
        let b: Vec<_> = second.finish().changes.iter().map(|c| c.pos).collect();
        assert_eq!(a, b);
        assert!(a.windows(2).all(|w| {
            (w[0].2, w[0].1, w[0].0) < (w[1].2, w[1].1, w[1].0)
        }));
    }
}
