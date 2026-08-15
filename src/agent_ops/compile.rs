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

use serde_json::Value;

use crate::core::{ChunkPos, Voxel, World};
use crate::editor::{
    box_voxels, cylinder_voxels, line_voxels, mirror_selection_changes, rotate_selection_changes,
    sphere_voxels, Selection, VoxelChange,
};
use crate::procgen::PipelineGraph;

use super::describe::solid_voxel_count;
use super::registry;
use super::schema::{quarter_from, Aabb, AxisSpec, GraphEdit, Op, VoxelSpec, WriteMode};
use super::{
    ErrorCode, OpsError, MAX_BATCH_CELLS, MAX_COORD, MAX_GRAPH_EDITS, MAX_NEW_CHUNKS,
    MAX_OP_REGION_CELLS, MAX_SET_VOXELS_PER_OP,
};

/// The six face neighbors, for shell and hollow tests.
pub(super) const FACE_NEIGHBORS: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

/// Step one cell from `pos`, or `None` when the step would leave `i32`.
///
/// The ops path never sees `None`: its coordinates came through
/// [`check_coord`] and sit within ±[`MAX_COORD`], nowhere near the edge.
/// The *measurement* path is the reason this exists — `describe` reads
/// whatever a `.vxlt` happens to hold, and a cell parked at `i32::MAX`
/// used to walk its own neighbor arithmetic straight into an overflow
/// panic (in the editor, on the frame loop's thread).
///
/// A neighbor that can't be represented is a neighbor that can't hold a
/// voxel, so reading `None` as "nothing there" is the honest answer, not
/// a fallback that papers over a failure.
pub(super) fn face_neighbor(
    pos: (i32, i32, i32),
    delta: (i32, i32, i32),
) -> Option<(i32, i32, i32)> {
    Some((
        pos.0.checked_add(delta.0)?,
        pos.1.checked_add(delta.1)?,
        pos.2.checked_add(delta.2)?,
    ))
}

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
    /// The document's graph as the batch found it. `graph_edit` starts
    /// from here when nothing earlier in the batch replaced it.
    base_graph: PipelineGraph,
    /// The graph the last `graph` / `graph_edit` op left, if any.
    /// Document data, not undo data — the caller assigns it alongside
    /// the changes, the same way it assigns the selection.
    graph: Option<PipelineGraph>,
}

pub(super) struct Outcome {
    pub changes: Vec<VoxelChange>,
    pub selection: Option<Selection>,
    pub notes: Vec<String>,
    pub voxel_count: u64,
    pub world_aabb: Option<Aabb>,
    /// The pipeline graph the batch left on the document, if it carried
    /// one.
    pub graph: Option<PipelineGraph>,
    /// The scratch world with every op applied. A real run drops it and
    /// commits `changes` instead; a preview hands it back as the world
    /// the batch *would* have produced, so the caller can describe or
    /// slice a result it hasn't accepted yet.
    pub world: World,
}

impl Scratch {
    pub fn new(input: super::BatchInput<'_>) -> Self {
        Self {
            world: input.world.deep_clone(),
            changes: HashMap::new(),
            selection: input.selection,
            base_chunks: input.world.chunk_count(),
            cells: 0,
            notes: Vec::new(),
            base_graph: input.graph.clone(),
            graph: None,
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
            graph: self.graph,
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
                    self.write(
                        (entry.0, entry.1, entry.2),
                        entry.3.to_voxel()?,
                        *write_mode,
                    )?;
                }
                Ok(())
            }

            Op::Generate {
                generator,
                params,
                translate,
                write_mode,
            } => self.generate(index, generator, params, *translate, *write_mode),

            Op::Graph {
                graph,
                apply,
                translate,
                write_mode,
            } => self.run_graph(index, graph, *apply, *translate, *write_mode),

            Op::GraphEdit {
                edits,
                apply,
                translate,
                write_mode,
            } => self.edit_graph(index, edits, *apply, *translate, *write_mode),

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
        if !self
            .world
            .has_chunk(ChunkPos::from_world_pos(pos.0, pos.1, pos.2))
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
            .filter(|&pos| is_enclosed(&self.world, pos))
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

    /// Store a pipeline graph on the document, and — unless `apply` is
    /// off — evaluate it and write what it produces.
    ///
    /// The write half is deliberately identical to [`Self::generate`]:
    /// a graph is a generator with a shape, so its patch goes through
    /// the same dedup and the same [`Self::write`] door, and inherits
    /// the same budget, coordinate ceiling and `write_mode`.
    ///
    /// Everything before that is the part a graph needs and a single
    /// generator doesn't. The graph arrives as raw JSON rather than a
    /// typed field for two reasons that happen to want the same thing:
    /// the generated tool schema stays a bare object instead of nine
    /// more definitions an agent carries in context every turn, and the
    /// unknown-key check below can name the field that was misspelled —
    /// which `#[serde(deny_unknown_fields)]` cannot do here, because
    /// these same types have to stay lenient for `.vxlt`.
    fn run_graph(
        &mut self,
        index: usize,
        sent: &Value,
        apply: bool,
        translate: [i32; 3],
        mode: WriteMode,
    ) -> Result<(), OpsError> {
        check_coord(tuple(translate))?;
        let graph: PipelineGraph = serde_json::from_value(sent.clone())
            .map_err(|e| OpsError::new(ErrorCode::InvalidGraph, e.to_string()))?;
        let understood = serde_json::to_value(&graph)
            .map_err(|e| OpsError::new(ErrorCode::InvalidGraph, e.to_string()))?;
        reject_unknown_graph_keys(sent, &understood, "graph")?;
        self.commit_graph(index, graph, apply, translate, mode)
    }

    /// Change the graph the document already has.
    ///
    /// Edits run against a copy and are only kept if every one of them
    /// lands — the same atomicity the batch itself has, one level down.
    /// A `connect` that would close a cycle is refused by the graph's
    /// own check, which is the point of editing through its methods
    /// rather than reaching into `nodes`.
    fn edit_graph(
        &mut self,
        index: usize,
        edits: &[GraphEdit],
        apply: bool,
        translate: [i32; 3],
        mode: WriteMode,
    ) -> Result<(), OpsError> {
        check_coord(tuple(translate))?;
        if edits.len() > MAX_GRAPH_EDITS {
            return Err(OpsError::new(
                ErrorCode::TooManyOps,
                format!(
                    "{} edits in one op; at most {} are allowed — a graph that needs more \
                     is quicker to send whole",
                    edits.len(),
                    MAX_GRAPH_EDITS
                ),
            ));
        }
        // Start from what an earlier op in this batch left, so
        // `graph` then `graph_edit` reads the way it looks.
        let mut graph = self
            .graph
            .clone()
            .unwrap_or_else(|| self.base_graph.clone());
        for (position, edit) in edits.iter().enumerate() {
            apply_graph_edit(&mut graph, edit).map_err(|mut e| {
                e.message = format!("edit[{position}]: {}", e.message);
                e
            })?;
        }
        self.commit_graph(index, graph, apply, translate, mode)
    }

    /// Validate a graph, optionally evaluate it into the world, and keep
    /// it on the document.
    ///
    /// Shared by `graph` and `graph_edit` so a graph that arrived whole
    /// and one that was edited into place are checked and written by the
    /// same code — the same reason `apply_ops` and `preview_ops` share
    /// [`run_batch`](super::run_batch).
    ///
    /// The write half is deliberately identical to [`Self::generate`]:
    /// a graph is a generator with a shape, so its patch goes through
    /// the same dedup and the same [`Self::write`] door, and inherits
    /// the budget, the coordinate ceiling and `write_mode` from it.
    fn commit_graph(
        &mut self,
        index: usize,
        mut graph: PipelineGraph,
        apply: bool,
        translate: [i32; 3],
        mode: WriteMode,
    ) -> Result<(), OpsError> {
        graph.normalize();
        check_graph(&graph)?;

        if apply {
            let patch = graph.evaluate().map_err(|e| {
                OpsError::new(ErrorCode::GeneratorFailed, format!("graph failed: {e}"))
            })?;
            for note in &patch.notes {
                self.notes.push(format!("op[{index}] graph: {note}"));
            }
            for (pos, voxel) in patch.dedup_last_write() {
                let pos = (
                    pos.0.saturating_add(translate[0]),
                    pos.1.saturating_add(translate[1]),
                    pos.2.saturating_add(translate[2]),
                );
                self.write(pos, voxel, mode)?;
            }
        }
        // Stored whether or not it ran: `apply: false` is how a graph
        // gets built up across batches, and a graph nobody kept is a
        // graph the human can't open.
        self.graph = Some(graph);
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

/// Apply one edit to a graph, in the graph's own terms.
///
/// Every arm goes through a `PipelineGraph` method rather than touching
/// `nodes` directly, so an agent's `connect` gets the same cycle check
/// and the same rollback a human dragging a wire in the panel does, and
/// `remove` clears the wires that pointed at the node for both of them.
fn apply_graph_edit(graph: &mut PipelineGraph, edit: &GraphEdit) -> Result<(), OpsError> {
    match edit {
        GraphEdit::AddNode { node } => {
            let mut parsed: crate::procgen::GraphNode = serde_json::from_value(node.clone())
                .map_err(|e| OpsError::new(ErrorCode::InvalidGraph, e.to_string()))?;
            let understood = serde_json::to_value(&parsed)
                .map_err(|e| OpsError::new(ErrorCode::InvalidGraph, e.to_string()))?;
            reject_unknown_graph_keys(node, &understood, "node")?;
            if graph.get(parsed.id).is_some() {
                return Err(OpsError::new(
                    ErrorCode::InvalidGraph,
                    format!("node {} already exists; pick an unused id", parsed.id),
                ));
            }
            // An agent sends no layout, and every node at [0, 0] is a
            // pile the human can't read. The whole-graph path re-lays
            // out on arrival; a single node joining a laid-out graph
            // takes its own cascade slot instead.
            if parsed.position == [0.0, 0.0] {
                parsed.position = PipelineGraph::place(parsed.id);
            }
            graph.nodes.push(parsed);
            Ok(())
        }
        GraphEdit::RemoveNode { id } => {
            require_node(graph, *id)?;
            graph.remove(*id);
            Ok(())
        }
        GraphEdit::SetParams { id, params } => set_node_params(graph, *id, params),
        GraphEdit::Connect {
            target,
            slot,
            source,
        } => {
            require_node(graph, *source)?;
            graph
                .set_input(*target, *slot, Some(*source))
                .map_err(graph_error)
        }
        GraphEdit::Disconnect { target, slot } => {
            graph.set_input(*target, *slot, None).map_err(graph_error)
        }
        GraphEdit::Clear => {
            *graph = PipelineGraph::default();
            Ok(())
        }
    }
}

fn require_node(graph: &PipelineGraph, id: crate::procgen::NodeId) -> Result<(), OpsError> {
    if graph.get(id).is_none() {
        return Err(OpsError::new(
            ErrorCode::InvalidGraph,
            format!("no node with id {id}"),
        ));
    }
    Ok(())
}

/// Merge `params` into a node's payload, keeping everything it doesn't
/// name.
///
/// Round-tripping the payload through JSON rather than matching on every
/// `NodeKind` variant: one implementation covers generator parameters
/// and transform fields alike, and it can't fall behind a variant added
/// later. Same merge-over-current semantics the registry gives a
/// `generate` op, and the same refusal of a name nothing reads.
fn set_node_params(
    graph: &mut PipelineGraph,
    id: crate::procgen::NodeId,
    params: &Value,
) -> Result<(), OpsError> {
    let Value::Object(overrides) = params else {
        return Err(OpsError::new(
            ErrorCode::InvalidGraph,
            "params must be a JSON object",
        ));
    };
    let node = graph
        .get_mut(id)
        .ok_or_else(|| OpsError::new(ErrorCode::InvalidGraph, format!("no node with id {id}")))?;
    let mut current = match serde_json::to_value(&node.kind) {
        Ok(Value::Object(map)) => map,
        _ => {
            return Err(OpsError::new(
                ErrorCode::InvalidGraph,
                "this node has no parameters to set",
            ))
        }
    };
    for (key, value) in overrides {
        if key == "kind" {
            return Err(OpsError::new(
                ErrorCode::InvalidGraph,
                "a node's kind can't be changed in place; remove it and add the one you meant",
            ));
        }
        if !current.contains_key(key) {
            let valid: Vec<&str> = current.keys().map(String::as_str).collect();
            return Err(OpsError::new(
                ErrorCode::InvalidGraph,
                format!(
                    "node {id} has no field {key:?}; it accepts {}",
                    valid.join(", ")
                ),
            ));
        }
        current.insert(key.clone(), value.clone());
    }
    node.kind = serde_json::from_value(Value::Object(current))
        .map_err(|e| OpsError::new(ErrorCode::InvalidGraph, e.to_string()))?;
    Ok(())
}

/// A graph problem in this protocol's terms. The split is the one an
/// agent acts on: a malformed graph gets edited, an oversized one gets
/// split.
/// Shape then size — see [`super::check_graph`], which is this under a
/// name the editor can reach.
pub(super) fn check_graph(graph: &PipelineGraph) -> Result<(), OpsError> {
    graph.validate().map_err(graph_error)?;
    registry::check_graph_sources(graph)
}

fn graph_error(error: crate::procgen::GraphError) -> OpsError {
    use crate::procgen::GraphError as G;
    let code = match error {
        G::TooManyNodes { .. } => ErrorCode::GraphTooLarge,
        _ => ErrorCode::InvalidGraph,
    };
    OpsError::new(code, error.to_string())
}

/// Refuse keys the graph types didn't understand, by comparing what was
/// sent against what survived a round trip through them.
///
/// The alternative — `#[serde(deny_unknown_fields)]` — isn't available:
/// the same types are the `.vxlt` storage format, which must ignore
/// fields a newer build wrote, and a node's payload is flattened, which
/// serde won't combine with the deny attribute anyway. Round-tripping
/// gets the strictness without a hand-written key list that would start
/// drifting from the structs the day it was written.
///
/// Serialization emits every field (nothing here is `skip_serializing`),
/// so a key present in the request and absent from the round trip is a
/// key nothing read. Arrays line up index for index — `nodes` comes back
/// in the order it went in.
fn reject_unknown_graph_keys(sent: &Value, understood: &Value, path: &str) -> Result<(), OpsError> {
    match (sent, understood) {
        (Value::Object(sent), Value::Object(understood)) => {
            for (key, value) in sent {
                let Some(counterpart) = understood.get(key) else {
                    let valid: Vec<&str> = understood.keys().map(String::as_str).collect();
                    return Err(OpsError::new(
                        ErrorCode::InvalidGraph,
                        format!(
                            "{path}: unknown field {key:?}; this node accepts {}",
                            valid.join(", ")
                        ),
                    ));
                };
                reject_unknown_graph_keys(value, counterpart, &format!("{path}.{key}"))?;
            }
            Ok(())
        }
        (Value::Array(sent), Value::Array(understood)) => {
            for (i, (sent, understood)) in sent.iter().zip(understood).enumerate() {
                reject_unknown_graph_keys(sent, understood, &format!("{path}[{i}]"))?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// A solid cell whose six face neighbors are all solid — the interior
/// `hollow` removes, and the count `describe` reports so an agent can
/// tell whether hollowing would buy anything before exporting.
pub(super) fn is_enclosed(world: &World, pos: (i32, i32, i32)) -> bool {
    world.get_voxel(pos.0, pos.1, pos.2).is_solid()
        && FACE_NEIGHBORS.iter().all(|&delta| {
            face_neighbor(pos, delta).is_some_and(|n| world.get_voxel(n.0, n.1, n.2).is_solid())
        })
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
    use crate::agent_ops::{
        AgentSession, ApplyReport, OpsBatch, MAX_GRAPH_SOURCES, MAX_OPS_PER_BATCH,
    };

    fn parse(json: &str) -> OpsBatch {
        serde_json::from_str(json).expect("test batch should parse")
    }

    /// A document with nothing in it, borrowed for the scratch tests
    /// that only exercise the budget.
    fn empty_document<'a>(
        world: &'a World,
        graph: &'a PipelineGraph,
    ) -> crate::agent_ops::BatchInput<'a> {
        crate::agent_ops::BatchInput {
            world,
            selection: None,
            graph,
        }
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
                    (
                        ox + local.x as i32,
                        oy + local.y as i32,
                        oz + local.z as i32,
                    ),
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
        assert_eq!(
            report.world_aabb,
            Some(Aabb {
                min: [0, 0, 0],
                max: [1, 1, 1]
            })
        );
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
        assert!(
            session.world.get_voxel(2, 2, 2).is_air(),
            "interior must be empty"
        );
        assert!(
            session.world.get_voxel(2, 2, 0).is_solid(),
            "face must be solid"
        );
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
        assert!(
            session.world.get_voxel(0, 5, 0).is_air(),
            "height is 5 cells, 0..=4"
        );
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
        assert_eq!(
            session.world.get_voxel(0, 0, 0).color(),
            [255, 0, 0, 255],
            "occupied cell kept"
        );
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
        assert_eq!(
            session.world.get_voxel(0, 0, 0).color(),
            [0, 0, 255, 255],
            "repainted"
        );
        assert!(
            session.world.get_voxel(1, 0, 0).is_air(),
            "must not grow into air"
        );
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
        assert_eq!(
            report.selection,
            Some(Aabb {
                min: [0, 0, 0],
                max: [0, 0, 3]
            })
        );
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
            Some(Aabb {
                min: [10, 10, 10],
                max: [11, 11, 11]
            })
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
                before.min[0],
                before.min[1],
                before.min[2],
                before.max[0],
                before.max[1],
                before.max[2],
            ),
        );
        assert_eq!(mirrored.voxel_count, grown.voxel_count * 2);
        let after = mirrored.world_aabb.expect("still there");
        assert_eq!(after.max[0] - after.min[0] + 1, width * 2);
        assert_eq!(
            after.min[0], before.min[0],
            "the original half must not move"
        );
    }

    // -------- graphs --------

    /// A three-node pipeline, as an agent would send it: no `next_id`,
    /// no positions, each source naming only the parameters it changes.
    const GRAPH_BATCH: &str = r#"{"version":1,"ops":[{"op":"graph","graph":{"nodes":[
        {"id":0,"kind":"builtin.perlin_terrain","width":8,"depth":8,"max_height":4},
        {"id":1,"kind":"filter","input":0,"predicate":{"y_above":1}},
        {"id":2,"kind":"output","input":1}
    ]}}]}"#;

    #[test]
    fn a_graph_writes_its_output_and_stays_on_the_document() {
        // The whole point of sending a graph instead of voxels: the
        // model lands *and* the recipe is still there to tune.
        let mut session = AgentSession::new();
        let report = apply(&mut session, GRAPH_BATCH);

        assert!(report.changed_voxels > 0, "the graph built something");
        assert!(session.world.scene_aabb().is_some());
        assert_eq!(session.graph.nodes.len(), 3);
        assert_eq!(
            session.graph.next_id, 3,
            "bookkeeping the agent never sent is derived on the way in"
        );
        // Every voxel came through the filter.
        let (min, _) = session
            .world
            .scene_aabb()
            .expect("the graph built something");
        assert!(min.1 >= 1, "filter leaked y={}", min.1);
    }

    #[test]
    fn a_graph_can_be_stored_without_being_run() {
        // `apply: false` is how a graph gets built up over several
        // batches — dry_run can't stand in for it, since a dry run
        // keeps nothing at all.
        let mut session = AgentSession::new();
        let report = apply(
            &mut session,
            &GRAPH_BATCH.replace(r#""graph":{"#, r#""apply":false,"graph":{"#),
        );
        assert_eq!(report.changed_voxels, 0, "nothing was written");
        assert_eq!(session.world.chunk_count(), 0);
        assert_eq!(session.graph.nodes.len(), 3, "but the graph was kept");
    }

    #[test]
    fn a_graph_op_offsets_its_output_like_generate_does() {
        let mut session = AgentSession::new();
        apply(
            &mut session,
            &GRAPH_BATCH.replace(r#""op":"graph""#, r#""op":"graph","translate":[100,0,0]"#),
        );
        let (min, _) = session
            .world
            .scene_aabb()
            .expect("the graph built something");
        assert!(
            min.0 >= 90,
            "translated x should be far from the origin: {min:?}"
        );
    }

    #[test]
    fn a_misspelled_node_field_is_named_rather_than_ignored() {
        // The graph types stay lenient — they are also the `.vxlt`
        // storage format — so this check lives in the protocol, and it
        // has to say *which* field and *where*.
        let mut session = AgentSession::new();
        let error = refuse(
            &mut session,
            r#"{"version":1,"ops":[{"op":"graph","graph":{"nodes":[
                {"id":0,"kind":"builtin.perlin_terrain","widht":8},
                {"id":1,"kind":"output","input":0}
            ]}}]}"#,
        );
        assert_eq!(error.code, ErrorCode::InvalidGraph);
        assert!(
            error.message.contains("widht") && error.message.contains("width"),
            "the message should name the bad key and the real ones, got: {}",
            error.message
        );
    }

    #[test]
    fn a_misspelled_nodes_key_is_still_refused() {
        // `PipelineGraph` carries a struct-level `#[serde(default)]` so
        // that a field added to it later can't lock users out of the
        // `.vxlt` files they already have. The price is that serde no
        // longer refuses a graph object with no `nodes` at all, which
        // used to be what caught this typo ("missing field `nodes`").
        // The protocol's own unknown-key check has to cover it instead.
        let mut session = AgentSession::new();
        let error = refuse(
            &mut session,
            r#"{"version":1,"ops":[{"op":"graph","graph":{"noeds":[
                {"id":0,"kind":"output"}
            ]}}]}"#,
        );
        assert_eq!(error.code, ErrorCode::InvalidGraph);
        assert!(
            error.message.contains("noeds"),
            "the message should name the bad key, got: {}",
            error.message
        );
    }

    #[test]
    fn a_malformed_graph_is_refused_with_a_code_the_agent_can_branch_on() {
        let mut session = AgentSession::new();
        let duplicate = refuse(
            &mut session,
            r#"{"version":1,"ops":[{"op":"graph","graph":{"nodes":[
                {"id":0,"kind":"builtin.perlin_terrain"},
                {"id":0,"kind":"output","input":0}
            ]}}]}"#,
        );
        assert_eq!(duplicate.code, ErrorCode::InvalidGraph);

        let no_output = refuse(
            &mut session,
            r#"{"version":1,"ops":[{"op":"graph","graph":{"nodes":[
                {"id":0,"kind":"builtin.perlin_terrain"}
            ]}}]}"#,
        );
        assert_eq!(no_output.code, ErrorCode::InvalidGraph);
    }

    #[test]
    fn a_graph_node_cannot_walk_past_the_generator_size_ceiling() {
        // The same ceiling `generate` enforces, reached by the other
        // door: a graph node holds an already-built generator, so
        // nothing in the registry's build path gets a look at it.
        let mut session = AgentSession::new();
        let error = refuse(
            &mut session,
            r#"{"version":1,"ops":[{"op":"graph","graph":{"nodes":[
                {"id":0,"kind":"builtin.perlin_terrain","width":100000,"depth":100000},
                {"id":1,"kind":"output","input":0}
            ]}}]}"#,
        );
        assert_eq!(error.code, ErrorCode::InvalidParams);
        assert!(
            error.message.contains("node 0"),
            "a graph names its nodes by id; the message must too, got: {}",
            error.message
        );
    }

    /// The editor loads a graph out of a `.vxlt` and evaluates it on
    /// the thread that draws, so the two ceilings have to be reachable
    /// from outside this module — and they have to fire *before* the
    /// evaluator recurses or the generator allocates. Both of these
    /// were measured taking the process: a chain this long overflowed
    /// the stack, and the terrain node's own `max - min` overflowed
    /// `i32` while sizing its buffer.
    #[test]
    fn check_graph_refuses_what_the_evaluator_cannot_survive() {
        let mut nodes: Vec<String> = (0..60_000)
            .map(|id| format!(r#"{{"id":{id},"kind":"translate","dx":1}}"#))
            .collect();
        nodes.push(r#"{"id":60000,"kind":"output"}"#.to_string());
        let graph: crate::procgen::PipelineGraph =
            serde_json::from_str(&format!(r#"{{"nodes":[{}]}}"#, nodes.join(",")))
                .expect("a .vxlt could hold this");
        let error = super::check_graph(&graph).expect_err("60,001 nodes is past the stack guard");
        assert_eq!(error.code, ErrorCode::GraphTooLarge);

        let graph: crate::procgen::PipelineGraph = serde_json::from_str(
            r#"{"nodes":[
                {"id":0,"kind":"builtin.perlin_terrain","width":1,"depth":1,
                 "min_height":-2000000000,"max_height":2000000000},
                {"id":1,"kind":"output","input":0}]}"#,
        )
        .expect("a .vxlt could hold this too");
        let error =
            super::check_graph(&graph).expect_err("a four-billion-cell span is past the ceiling");
        assert_eq!(error.code, ErrorCode::InvalidParams);
    }

    /// A generator's origin is a coordinate, and every other coordinate
    /// in this protocol is bounded before anything runs. These two were
    /// not: the generator stamped `origin + offset` while building its
    /// patch, which overflows on a build with overflow checks — the
    /// abort takes the process, and with it whatever an editor had
    /// unsaved. Release wrapped and the write door refused the result,
    /// so the code an agent sees is the same one; it just arrives before
    /// the work now, naming the field it wrote.
    #[test]
    fn a_generator_origin_past_the_coordinate_ceiling_is_refused() {
        for generator in ["builtin.wfc", "builtin.lsystem_tree"] {
            let mut session = AgentSession::new();
            let error = refuse(
                &mut session,
                &format!(
                    r#"{{"version":1,"ops":[{{"op":"generate","generator":"{generator}",
                       "params":{{"origin":[2147483647,0,0]}}}}]}}"#
                ),
            );
            assert_eq!(
                error.code,
                ErrorCode::CoordinateOutOfRange,
                "{generator} should refuse an extreme origin"
            );
            assert!(
                error.message.contains("origin"),
                "the message must name the parameter, got: {}",
                error.message
            );
        }
    }

    /// A Translate node's offset is checked for shape, not magnitude —
    /// the schema takes any `i32`, `validate` looks at wires, and the
    /// source ceiling looks at sizes. So the sum lands at the write
    /// door, and it has to *arrive* there: a bare `+` in the node's
    /// shift used to abort the process on a debug build instead.
    #[test]
    fn a_translate_node_offset_saturates_into_a_refusal() {
        let mut session = AgentSession::new();
        let error = refuse(
            &mut session,
            r#"{"version":1,"ops":[{"op":"graph","graph":{"nodes":[
                {"id":0,"kind":"builtin.perlin_terrain","width":4,"depth":4},
                {"id":1,"kind":"translate","input":0,"dx":2147483647},
                {"id":2,"kind":"output","input":1}
            ]}}]}"#,
        );
        assert_eq!(error.code, ErrorCode::CoordinateOutOfRange);
    }

    /// Same ceiling through the graph door, which holds an
    /// already-built generator the registry's build path never saw.
    #[test]
    fn a_graph_node_cannot_walk_past_the_origin_ceiling_either() {
        let mut session = AgentSession::new();
        let error = refuse(
            &mut session,
            r#"{"version":1,"ops":[{"op":"graph","graph":{"nodes":[
                {"id":0,"kind":"builtin.lsystem_tree","origin":[2147483647,0,0]},
                {"id":1,"kind":"output","input":0}
            ]}}]}"#,
        );
        assert_eq!(error.code, ErrorCode::CoordinateOutOfRange);
        assert!(
            error.message.contains("node 0"),
            "a graph names its nodes by id; the message must too, got: {}",
            error.message
        );
    }

    #[test]
    fn too_many_source_nodes_is_refused_before_evaluation() {
        // Each source is legal on its own; evaluation holds all of them
        // at once, and none of it has reached the cell budget yet.
        let mut nodes: Vec<String> = (0..=MAX_GRAPH_SOURCES)
            .map(|i| format!(r#"{{"id":{i},"kind":"builtin.perlin_terrain","width":8,"depth":8}}"#))
            .collect();
        nodes.push(format!(
            r#"{{"id":{},"kind":"output","input":0}}"#,
            MAX_GRAPH_SOURCES + 1
        ));
        let batch = format!(
            r#"{{"version":1,"ops":[{{"op":"graph","graph":{{"nodes":[{}]}}}}]}}"#,
            nodes.join(",")
        );
        let mut session = AgentSession::new();
        assert_eq!(refuse(&mut session, &batch).code, ErrorCode::GraphTooLarge);
    }

    #[test]
    fn a_dry_run_graph_keeps_neither_the_voxels_nor_the_graph() {
        let mut session = AgentSession::new();
        let report = apply(
            &mut session,
            &GRAPH_BATCH.replace(r#""ops":["#, r#""options":{"dry_run":true},"ops":["#),
        );
        assert!(report.dry_run);
        assert!(
            report.changed_voxels > 0,
            "it still reports what would land"
        );
        assert_eq!(session.world.chunk_count(), 0);
        assert!(
            session.graph.nodes.is_empty(),
            "a dry run commits nothing, the graph included"
        );
    }

    // -------- editing a graph in place --------

    /// Build a graph over two batches: nodes first without evaluating,
    /// then wire it up and run it. This is what `apply: false` is for.
    #[test]
    fn a_graph_can_be_built_up_by_edits_across_batches() {
        let mut session = AgentSession::new();
        apply(
            &mut session,
            r#"{"version":1,"ops":[{"op":"graph_edit","apply":false,"edits":[
                {"edit":"add_node","node":{"id":0,"kind":"builtin.perlin_terrain","width":8,"depth":8}},
                {"edit":"add_node","node":{"id":1,"kind":"output"}}
            ]}]}"#,
        );
        assert_eq!(session.graph.nodes.len(), 2);
        assert_eq!(session.world.chunk_count(), 0, "nothing was evaluated yet");

        let report = apply(
            &mut session,
            r#"{"version":1,"ops":[{"op":"graph_edit","edits":[
                {"edit":"connect","target":1,"slot":0,"source":0}
            ]}]}"#,
        );
        assert!(report.changed_voxels > 0, "wiring it up ran it");
        assert_eq!(session.graph.nodes.len(), 2, "the edit kept the graph");
    }

    #[test]
    fn set_params_keeps_the_fields_it_does_not_name() {
        let mut session = AgentSession::new();
        apply(&mut session, GRAPH_BATCH);
        apply(
            &mut session,
            r#"{"version":1,"ops":[{"op":"graph_edit","apply":false,"edits":[
                {"edit":"set_params","id":0,"params":{"width":16}}
            ]}]}"#,
        );
        match &session.graph.get(0).unwrap().kind {
            crate::procgen::NodeKind::Terrain(terrain) => {
                assert_eq!(terrain.width, 16, "the named field changed");
                assert_eq!(terrain.depth, 8, "the others kept the batch's values");
            }
            other => panic!("node 0 is {other:?}"),
        }
    }

    #[test]
    fn an_edit_that_fails_leaves_the_graph_exactly_as_it_was() {
        // Atomicity one level down from the batch: the first edit is
        // fine, the second closes a cycle. Neither may survive.
        let mut session = AgentSession::new();
        apply(&mut session, GRAPH_BATCH);
        let before = session.graph.clone();

        let error = refuse(
            &mut session,
            r#"{"version":1,"ops":[{"op":"graph_edit","edits":[
                {"edit":"set_params","id":0,"params":{"width":32}},
                {"edit":"connect","target":0,"slot":0,"source":2}
            ]}]}"#,
        );
        assert_eq!(error.code, ErrorCode::InvalidGraph);
        assert!(
            error.message.contains("edit[1]"),
            "the message should say which edit failed, got: {}",
            error.message
        );
        assert_eq!(session.graph, before, "not even the first edit stuck");
    }

    #[test]
    fn edits_name_the_field_or_node_that_was_wrong() {
        let mut session = AgentSession::new();
        apply(&mut session, GRAPH_BATCH);

        let bad_field = refuse(
            &mut session,
            r#"{"version":1,"ops":[{"op":"graph_edit","edits":[
                {"edit":"set_params","id":0,"params":{"widht":16}}
            ]}]}"#,
        );
        assert!(
            bad_field.message.contains("widht") && bad_field.message.contains("width"),
            "got: {}",
            bad_field.message
        );

        let missing = refuse(
            &mut session,
            r#"{"version":1,"ops":[{"op":"graph_edit","edits":[
                {"edit":"connect","target":2,"slot":0,"source":9}
            ]}]}"#,
        );
        assert!(missing.message.contains("9"), "got: {}", missing.message);

        let retyped = refuse(
            &mut session,
            r#"{"version":1,"ops":[{"op":"graph_edit","edits":[
                {"edit":"set_params","id":0,"params":{"kind":"output"}}
            ]}]}"#,
        );
        assert!(
            retyped.message.contains("kind"),
            "changing kind in place must be refused by name, got: {}",
            retyped.message
        );
    }

    #[test]
    fn removing_a_node_clears_the_wires_that_pointed_at_it() {
        let mut session = AgentSession::new();
        apply(&mut session, GRAPH_BATCH);
        apply(
            &mut session,
            r#"{"version":1,"ops":[{"op":"graph_edit","apply":false,"edits":[
                {"edit":"remove_node","id":0}
            ]}]}"#,
        );
        assert_eq!(session.graph.nodes.len(), 2);
        assert_eq!(
            session.graph.get_input(1, 0).unwrap(),
            None,
            "the filter's input must not still name a node that is gone"
        );
    }

    #[test]
    fn a_node_added_by_an_agent_is_laid_out_rather_than_piled_on_the_origin() {
        let mut session = AgentSession::new();
        apply(&mut session, GRAPH_BATCH);
        apply(
            &mut session,
            r#"{"version":1,"ops":[{"op":"graph_edit","apply":false,"edits":[
                {"edit":"add_node","node":{"id":9,"kind":"translate","dy":4}}
            ]}]}"#,
        );
        let added = session.graph.get(9).expect("the node was added");
        assert_ne!(added.position, [0.0, 0.0]);
        for node in &session.graph.nodes {
            if node.id != 9 {
                assert_ne!(node.position, added.position, "nodes overlap");
            }
        }
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
        let mut scratch = Scratch::new(empty_document(&World::new(), &PipelineGraph::default()));
        assert!(scratch.charge(MAX_BATCH_CELLS).is_ok());
        assert_eq!(
            scratch.charge(1).unwrap_err().code,
            ErrorCode::CellBudgetExceeded
        );
    }

    #[test]
    fn every_visited_cell_is_charged_including_masked_ones() {
        let mut scratch = Scratch::new(empty_document(&World::new(), &PipelineGraph::default()));
        let voxel = Voxel::from_rgb(1, 2, 3);
        scratch.write((0, 0, 0), voxel, WriteMode::Replace).unwrap();
        // Masked out (the cell is air), but it was still visited.
        scratch
            .write((1, 0, 0), voxel, WriteMode::OnlySolid)
            .unwrap();
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
        assert_eq!(
            snapshot(&session.world),
            before,
            "op 0 must not have landed"
        );
        assert_eq!(
            session.history.undo_count(),
            undo_depth,
            "no undo entry pushed"
        );
        assert_eq!(
            session.selection.map(Aabb::from),
            Some(Aabb {
                min: [0, 0, 0],
                max: [1, 1, 1]
            }),
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
        assert_eq!(
            snapshot(&session.world),
            before,
            "undo must restore exactly"
        );
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
        assert_eq!(
            session.history.undo_count(),
            1,
            "a no-op batch pushes no undo entry"
        );
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
        let mut first = Scratch::new(empty_document(&World::new(), &PipelineGraph::default()));
        first.run_op(0, &parse(batch).ops[0]).unwrap();
        let mut second = Scratch::new(empty_document(&World::new(), &PipelineGraph::default()));
        second.run_op(0, &parse(batch).ops[0]).unwrap();

        let a: Vec<_> = first.finish().changes.iter().map(|c| c.pos).collect();
        let b: Vec<_> = second.finish().changes.iter().map(|c| c.pos).collect();
        assert_eq!(a, b);
        assert!(a
            .windows(2)
            .all(|w| { (w[0].2, w[0].1, w[0].0) < (w[1].2, w[1].1, w[1].0) }));
    }
}
