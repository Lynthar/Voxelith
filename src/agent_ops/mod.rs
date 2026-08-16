//! The agent-facing edit protocol: a batch of JSON ops over the
//! editor's own primitives, answered with a report. Atomic, sequential,
//! one undo entry per batch.

use serde::Serialize;

use crate::core::World;
use crate::editor::{Command, CommandHistory, Selection, Socket, VoxelChange};
use crate::io::ProjectMetadata;
use crate::procgen::PipelineGraph;

mod compile;
mod describe;
mod registry;
mod schema;

pub use describe::{
    describe, slice, ColorCount, Description, DocumentView, LoosePart, SliceMode, SliceRequest,
    SocketInfo, Structure, SymmetryCheck,
};
pub use registry::{generator_infos, graph_template, GeneratorInfo};
pub use schema::{
    Aabb, AxisSpec, BatchOptions, GraphEdit, Op, OpsBatch, SolidVoxel, VoxelEntry, VoxelSpec,
    WriteMode,
};

/// Wire-format version an [`OpsBatch`] must declare. Bumped only for a
/// breaking change; additive fields keep version 1 and carry
/// `#[serde(default)]`.
pub const SCHEMA_VERSION: u32 = 1;

/// Undo depth of a session, matching the interactive editor's.
const HISTORY_DEPTH: usize = 100;

/// Ops per batch. A cap this side of "infinite" keeps one runaway
/// generation from becoming one un-reviewable undo entry.
pub const MAX_OPS_PER_BATCH: usize = 256;

/// Explicit voxels a single `set_voxels` op may carry. The op is the
/// per-voxel escape hatch for detailing; anything bigger wants a shape
/// or a generator.
pub const MAX_SET_VOXELS_PER_OP: usize = 4096;

/// Cells in a single op's region (128³). Bounds the work *and* the
/// temporary cell list a shape helper materializes.
pub const MAX_OP_REGION_CELLS: u64 = 128 * 128 * 128;

/// Cells one batch may touch in total, counting every cell an op
/// writes, scans or masks out. Upper-bound accounting on purpose — the
/// point is to bound work and memory, not to bill precisely.
pub const MAX_BATCH_CELLS: u64 = 8_388_608;

/// Coordinate ceiling on every axis. The world itself is unbounded, but
/// an agent that means `y = 5` and emits `y = 5000000` should hear about
/// it. Also keeps mirror and bbox math clear of `i32` overflow.
pub const MAX_COORD: i32 = 1 << 20;

/// Source nodes one graph may hold. Evaluation keeps only the working
/// front resident and the sources are the floor under it. A graph
/// needing more wants to run in stages — its output is in the world.
pub const MAX_GRAPH_SOURCES: usize = 8;

/// Edits one `graph_edit` op may carry. A graph tops out at 64 nodes,
/// so a change list longer than this is quicker to send as a whole
/// graph — and cheaper for the agent to get right.
pub const MAX_GRAPH_EDITS: usize = 64;

/// Chunks one batch may bring into existence. Each is a 256 KB
/// allocation, so scattered writes are the cheap way to ask for a lot of
/// memory. The cap is on *new* chunks, so large documents keep working.
pub const MAX_NEW_CHUNKS: usize = 2048;

/// Machine-readable failure reason. Agents branch on this; humans read
/// the message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// `version` isn't one this build speaks.
    UnsupportedVersion,
    /// A field is present and well-typed but its value isn't usable.
    InvalidArgument,
    /// More ops than [`MAX_OPS_PER_BATCH`].
    TooManyOps,
    /// A single op's region exceeds [`MAX_OP_REGION_CELLS`].
    RegionTooLarge,
    /// A `set_voxels` op exceeds [`MAX_SET_VOXELS_PER_OP`].
    TooManyVoxels,
    /// The batch exceeded [`MAX_BATCH_CELLS`].
    CellBudgetExceeded,
    /// A coordinate is outside ±[`MAX_COORD`].
    CoordinateOutOfRange,
    /// The batch would allocate more than [`MAX_NEW_CHUNKS`] chunks.
    WorldTooLarge,
    /// An op needs a region and got neither an explicit one nor a
    /// session selection.
    NoSelection,
    /// No generator with that id is registered.
    UnknownGenerator,
    /// Generator params don't fit the generator's parameter struct.
    InvalidParams,
    /// The generator ran and refused (its own validation).
    GeneratorFailed,
    /// The graph is malformed: a duplicate id, a wire to a node that
    /// isn't there, a cycle, no Output node or several.
    InvalidGraph,
    /// The graph is well formed but too big to evaluate — too many
    /// nodes, too many sources, or sources covering too many cells.
    GraphTooLarge,
    /// A requested slice is bigger than the ASCII view supports.
    SliceTooLarge,
}

impl ErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::UnsupportedVersion => "unsupported_version",
            ErrorCode::InvalidArgument => "invalid_argument",
            ErrorCode::TooManyOps => "too_many_ops",
            ErrorCode::RegionTooLarge => "region_too_large",
            ErrorCode::TooManyVoxels => "too_many_voxels",
            ErrorCode::CellBudgetExceeded => "cell_budget_exceeded",
            ErrorCode::CoordinateOutOfRange => "coordinate_out_of_range",
            ErrorCode::WorldTooLarge => "world_too_large",
            ErrorCode::NoSelection => "no_selection",
            ErrorCode::UnknownGenerator => "unknown_generator",
            ErrorCode::InvalidParams => "invalid_params",
            ErrorCode::GeneratorFailed => "generator_failed",
            ErrorCode::InvalidGraph => "invalid_graph",
            ErrorCode::GraphTooLarge => "graph_too_large",
            ErrorCode::SliceTooLarge => "slice_too_large",
        }
    }
}

/// A refused batch. `op_index` is the position of the op that failed
/// (absent for envelope-level failures), so an agent can fix that one
/// op and resend rather than guessing.
#[derive(Debug, Clone, Serialize)]
pub struct OpsError {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub op_index: Option<usize>,
    pub code: ErrorCode,
    pub message: String,
}

impl OpsError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            op_index: None,
            code,
            message: message.into(),
        }
    }

    /// Tag an error with the op it came from. Op executors raise errors
    /// without knowing their own index; the batch loop attaches it once
    /// here instead of threading the index through every call site.
    pub fn at(mut self, op_index: usize) -> Self {
        self.op_index = Some(op_index);
        self
    }
}

impl std::fmt::Display for OpsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.op_index {
            Some(i) => write!(f, "op[{}] {}: {}", i, self.code.as_str(), self.message),
            None => write!(f, "{}: {}", self.code.as_str(), self.message),
        }
    }
}

impl std::error::Error for OpsError {}

/// What a batch did. Identical in shape for a dry run and a real one —
/// that's the point of `dry_run`: an agent can ask "what would this
/// do?" and get the same numbers it would get from doing it.
#[derive(Debug, Clone, Serialize)]
pub struct ApplyReport {
    pub version: u32,
    pub dry_run: bool,
    pub applied_ops: usize,
    /// Voxels whose value actually changed — writes that land on a cell
    /// already holding that value don't count.
    pub changed_voxels: usize,
    /// Solid voxels in the whole world afterwards.
    pub voxel_count: u64,
    pub world_aabb: Option<Aabb>,
    pub selection: Option<Aabb>,
    /// Non-fatal diagnostics, mostly forwarded from generator patches
    /// (WFC's "N cells fell back to empty" and friends). Prefixed with
    /// the op that produced them.
    pub notes: Vec<String>,
}

/// A batch run against a copy of the session, committing nothing. The
/// caller can [`describe`](AgentSession::describe) or
/// [`slice`](AgentSession::slice) it before running it for real.
pub struct Preview {
    /// Always `dry_run: true` — a preview commits nothing, whatever the
    /// batch asked for.
    pub report: ApplyReport,
    /// The session as the batch would leave it: world and selection
    /// after the last op, sockets carried across. Its history is empty,
    /// so `describe()` on a preview reports depth 0.
    pub session: AgentSession,
}

/// A batch that has run, handed back for its caller to commit or throw
/// away. Both hosts — a session and the editor — reach it through
/// [`run_batch`], so neither can validate or report differently.
pub struct BatchOutcome {
    pub report: ApplyReport,
    /// Cells whose value actually changed, ready to become one
    /// [`Command::set_voxels`]: de-duplicated by position, identity
    /// writes dropped, sorted.
    pub changes: Vec<VoxelChange>,
    /// The selection after the last op, which is *not* undo-stack data
    /// on either host — a caller that commits `changes` assigns this
    /// alongside it, and one that discards them leaves its own alone.
    pub selection: Option<Selection>,
    /// The pipeline graph a `graph` op left on the document, if the
    /// batch carried one. Document data, assigned by whoever commits —
    /// the same contract as `selection`.
    pub graph: Option<PipelineGraph>,
    /// The world the batch produced. A committing caller ignores it and
    /// applies `changes`; one that only wants a look describes, slices
    /// or renders this rather than predicting from the report.
    pub world: World,
}

/// The document a batch runs against, borrowed from whoever owns it.
/// Growing a field here beats growing another positional parameter on
/// [`run_batch`] every time the document does.
pub struct BatchInput<'a> {
    pub world: &'a World,
    pub selection: Option<Selection>,
    /// The graph the document currently holds — the starting point a
    /// `graph_edit` op edits. A batch that never mentions graphs never
    /// reads it.
    pub graph: &'a PipelineGraph,
}

/// Everything that must hold before a graph is worth evaluating:
/// [`PipelineGraph::validate`] covers the shape, `check_graph_sources`
/// the size. Public because the editor evaluates graphs from files too.
pub fn check_graph(graph: &PipelineGraph) -> Result<(), OpsError> {
    compile::check_graph(graph)
}

/// Validate a batch and run it against a copy of the document,
/// committing nothing. The single execution path — every entry point
/// comes through here, so a preview and a real run cannot disagree.
pub fn run_batch(input: BatchInput<'_>, batch: &OpsBatch) -> Result<BatchOutcome, OpsError> {
    if batch.version != SCHEMA_VERSION {
        return Err(OpsError::new(
            ErrorCode::UnsupportedVersion,
            format!(
                "ops version {} is not supported; this build speaks version {}",
                batch.version, SCHEMA_VERSION
            ),
        ));
    }
    if batch.ops.is_empty() {
        return Err(OpsError::new(
            ErrorCode::InvalidArgument,
            "`ops` is empty; a batch must carry at least one op",
        ));
    }
    if batch.ops.len() > MAX_OPS_PER_BATCH {
        return Err(OpsError::new(
            ErrorCode::TooManyOps,
            format!(
                "{} ops in one batch; at most {} are allowed — split it",
                batch.ops.len(),
                MAX_OPS_PER_BATCH
            ),
        ));
    }

    let mut scratch = compile::Scratch::new(input);
    for (index, op) in batch.ops.iter().enumerate() {
        scratch.run_op(index, op).map_err(|e| e.at(index))?;
    }
    let mut outcome = scratch.finish();
    let report = report_of(batch.options.dry_run, batch.ops.len(), &mut outcome);
    Ok(BatchOutcome {
        report,
        changes: outcome.changes,
        selection: outcome.selection,
        graph: outcome.graph,
        world: outcome.world,
    })
}

/// A headless editing session: the document an agent operates on.
/// Fields are public because the CLI and the MCP server legitimately
/// swap the world in after a load and read the sockets out for export.
pub struct AgentSession {
    pub world: World,
    pub history: CommandHistory,
    /// Session selection. Like the editor's, it is *not* on the undo
    /// stack — it's a marquee, not document data.
    pub selection: Option<Selection>,
    /// Named attachment points. Document data (they persist in `.vxlt`
    /// and export to glTF), carried here so a load → edit → save round
    /// trip doesn't drop them. No v1 op modifies them.
    pub sockets: Vec<Socket>,
    /// The document's pipeline graph. Document data like the sockets,
    /// and here rather than only in the editor so a graph an agent built
    /// reaches the human who opens the project.
    pub graph: PipelineGraph,
    /// The project's identity (`name` / `author` / `created_at`).
    /// Loaded with the file and handed back to the save, so a headless
    /// load → edit → save round trip keeps it — no op modifies it.
    pub metadata: ProjectMetadata,
}

impl Default for AgentSession {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentSession {
    pub fn new() -> Self {
        Self {
            world: World::new(),
            history: CommandHistory::new(HISTORY_DEPTH),
            selection: None,
            sockets: Vec::new(),
            graph: PipelineGraph::default(),
            metadata: ProjectMetadata::default(),
        }
    }

    /// Validate and run a batch. On success the whole batch has been
    /// applied (or, for a dry run, would have been); on failure nothing
    /// changed.
    pub fn apply_ops(&mut self, batch: &OpsBatch) -> Result<ApplyReport, OpsError> {
        let outcome = run_batch(self.batch_input(), batch)?;

        if !batch.options.dry_run {
            // One command for the whole batch, so a graph it carried
            // rides inside and undo steps both back. A no-op batch
            // pushes nothing.
            let graph = outcome.graph.map(|after| crate::editor::GraphTransition {
                before: self.graph.clone(),
                after,
            });
            self.history.execute_with_graph(
                Command::set_voxels_with_graph(outcome.changes, graph),
                &mut self.world,
                &mut self.graph,
            );
            self.selection = outcome.selection;
        }
        Ok(outcome.report)
    }

    /// Run a batch and hand back the result instead of committing it.
    /// Same validation, executor and report as [`apply_ops`] — they
    /// share [`run_batch`] — but the session is left untouched.
    pub fn preview_ops(&self, batch: &OpsBatch) -> Result<Preview, OpsError> {
        let mut outcome = run_batch(self.batch_input(), batch)?;
        // Nothing was committed, whatever the batch asked for. A batch
        // sent here without `dry_run` set is still only being looked at.
        outcome.report.dry_run = true;
        Ok(Preview {
            report: outcome.report,
            session: AgentSession {
                world: outcome.world,
                history: CommandHistory::new(HISTORY_DEPTH),
                selection: outcome.selection,
                sockets: self.sockets.clone(),
                graph: outcome.graph.unwrap_or_else(|| self.graph.clone()),
                metadata: self.metadata.clone(),
            },
        })
    }

    /// This session's document, borrowed as the shape [`run_batch`]
    /// takes.
    pub fn batch_input(&self) -> BatchInput<'_> {
        BatchInput {
            world: &self.world,
            selection: self.selection,
            graph: &self.graph,
        }
    }

    /// This session's parts, borrowed as the shape the description
    /// and slice helpers take.
    pub fn view(&self) -> DocumentView<'_> {
        DocumentView {
            world: &self.world,
            selection: self.selection,
            sockets: &self.sockets,
            graph: &self.graph,
            undo_depth: self.history.undo_count(),
            redo_depth: self.history.redo_count(),
        }
    }

    /// Text summary of the document — the agent's cheapest look at what
    /// it just built.
    pub fn describe(&self) -> Description {
        describe::describe(self.view())
    }

    /// One axis-aligned plane as ASCII art.
    pub fn slice(&self, request: &SliceRequest) -> Result<String, OpsError> {
        describe::slice(&self.world, request)
    }

    pub fn undo(&mut self) -> bool {
        self.history.undo(&mut self.world, &mut self.graph)
    }

    pub fn redo(&mut self) -> bool {
        self.history.redo(&mut self.world, &mut self.graph)
    }
}

/// The one place an [`ApplyReport`] is built, so a preview and a real
/// run can't describe the same batch differently. Takes the outcome by
/// `&mut` only to move the notes out of it rather than copy them.
fn report_of(dry_run: bool, applied_ops: usize, outcome: &mut compile::Outcome) -> ApplyReport {
    ApplyReport {
        version: SCHEMA_VERSION,
        dry_run,
        applied_ops,
        changed_voxels: outcome.changes.len(),
        voxel_count: outcome.voxel_count,
        world_aabb: outcome.world_aabb,
        selection: outcome.selection.map(Aabb::from),
        notes: std::mem::take(&mut outcome.notes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::World;

    fn parse(json: &str) -> OpsBatch {
        serde_json::from_str(json).expect("test batch should parse")
    }

    /// Two ops, the second reading the first's output — enough that a
    /// host which validated or executed differently would show it.
    const BATCH: &str = r#"{"version":1,"ops":[
        {"op":"box","min":[0,0,0],"max":[4,4,4],"voxel":{"rgb":[120,90,60]}},
        {"op":"hollow","min":[0,0,0],"max":[4,4,4]}
    ]}"#;

    /// The editor's half of the contract: a batch arrives as a change
    /// list, the *user's* history commits it, and the world that lands
    /// is the one the agent's own session would have built.
    #[test]
    fn a_foreign_history_commits_the_world_the_session_would_have() {
        let batch = parse(BATCH);

        // The headless host: the session owns world and history both.
        let mut session = AgentSession::new();
        let session_report = session.apply_ops(&batch).expect("batch should apply");

        // The editor host: both belong to the user, and `run_batch` is
        // all it needs from this layer.
        let mut world = World::new();
        let mut history = CommandHistory::new(HISTORY_DEPTH);
        let graph = PipelineGraph::default();
        let outcome = run_batch(
            BatchInput {
                world: &world,
                selection: None,
                graph: &graph,
            },
            &batch,
        )
        .expect("batch should run");
        history.execute(Command::set_voxels(outcome.changes), &mut world);

        assert_eq!(outcome.report.changed_voxels, session_report.changed_voxels);
        assert_eq!(outcome.report.world_aabb, session_report.world_aabb);
        assert_eq!(outcome.report.voxel_count, session_report.voxel_count);
        for (x, y, z) in [(0, 0, 0), (2, 2, 0), (2, 2, 2), (4, 4, 4)] {
            assert_eq!(
                world.get_voxel(x, y, z),
                session.world.get_voxel(x, y, z),
                "cell ({x},{y},{z}) differs between the two hosts"
            );
        }

        // One entry for the batch, so one Ctrl+Z takes all of it back
        // out — the same guarantee the session gives, on the user's own
        // stack.
        assert_eq!(history.undo_count(), 1);
        assert!(history.undo(&mut world, &mut PipelineGraph::default()));
        assert!(
            world.get_voxel(2, 2, 0).is_air(),
            "undo must take the whole batch back out"
        );
    }

    /// Nothing is committed until the caller says so — the property the
    /// editor's approval mode rests on, since a batch awaiting a human's
    /// yes must not already be in their world.
    #[test]
    fn run_batch_leaves_its_input_world_alone() {
        let world = World::new();
        let graph = PipelineGraph::default();
        let outcome = run_batch(
            BatchInput {
                world: &world,
                selection: None,
                graph: &graph,
            },
            &parse(BATCH),
        )
        .expect("batch should run");
        assert!(outcome.report.voxel_count > 0, "the batch built something");
        assert_eq!(
            world.chunk_count(),
            0,
            "the caller's world must stay untouched until it commits"
        );
    }

    /// A preview commits nothing whatever the batch asked for, so its
    /// report says `dry_run` regardless — or the approval path claims
    /// an edit landed while it is still waiting for a human.
    #[test]
    fn a_preview_reports_a_dry_run_the_batch_never_asked_for() {
        let session = AgentSession::new();
        let preview = session
            .preview_ops(&parse(BATCH))
            .expect("batch should run");
        assert!(preview.report.dry_run);
    }

    /// A batch whose `graph` op replaced the document's graph. Uses the
    /// registry's own template so the graph is one the op provably
    /// accepts; `apply: false` keeps it a pure graph replacement.
    fn graph_batch() -> OpsBatch {
        let graph = crate::agent_ops::graph_template();
        parse(&format!(
            r#"{{"version":1,"ops":[{{"op":"graph","graph":{graph},"apply":false}}]}}"#
        ))
    }

    /// "One undo entry per batch" has to cover the graph half too, not
    /// just the voxels — otherwise a graph-only batch gets no undo entry
    /// and the graph it replaced is gone for good.
    #[test]
    fn a_graph_only_batch_is_one_undoable_entry() {
        let mut session = AgentSession::new();
        let before = session.graph.clone();
        session
            .apply_ops(&graph_batch())
            .expect("batch should apply");

        assert!(
            !session.graph.nodes.is_empty(),
            "the batch must have installed the template graph"
        );
        assert_eq!(
            session.history.undo_count(),
            1,
            "a graph replacement is a document edit and gets an undo entry"
        );

        let installed = session.graph.clone();
        assert!(session.undo(), "the entry must be steppable");
        assert_eq!(
            session.graph, before,
            "undo must restore the graph the batch replaced"
        );
        assert!(session.redo());
        assert_eq!(
            session.graph, installed,
            "redo must re-install the batch's graph"
        );
    }

    /// A mixed batch (voxels + graph) undoes as a whole: the voxels
    /// come back out AND the old graph comes back — not the half-state
    /// where the voxels are gone but the new graph lingers.
    #[test]
    fn undoing_a_mixed_batch_restores_the_graph_with_the_voxels() {
        let mut session = AgentSession::new();
        let before = session.graph.clone();

        let graph = crate::agent_ops::graph_template();
        let batch = parse(&format!(
            r#"{{"version":1,"ops":[
                {{"op":"box","min":[0,0,0],"max":[2,2,2],"voxel":{{"rgb":[10,20,30]}}}},
                {{"op":"graph","graph":{graph},"apply":false}}
            ]}}"#
        ));
        session.apply_ops(&batch).expect("batch should apply");
        assert_eq!(
            session.history.undo_count(),
            1,
            "one entry for the whole batch"
        );
        assert!(!session.world.get_voxel(1, 1, 1).is_air());

        assert!(session.undo());
        assert!(
            session.world.get_voxel(1, 1, 1).is_air(),
            "undo must take the voxels back out"
        );
        assert_eq!(
            session.graph, before,
            "…and the graph back to what the batch replaced"
        );
    }

    /// Re-sending the graph the document already holds changes nothing,
    /// so it must push nothing — the same no-op rule voxel batches
    /// follow.
    #[test]
    fn re_sending_the_same_graph_pushes_no_undo_entry() {
        let mut session = AgentSession::new();
        session.apply_ops(&graph_batch()).expect("first apply");
        assert_eq!(session.history.undo_count(), 1);
        session.apply_ops(&graph_batch()).expect("second apply");
        assert_eq!(
            session.history.undo_count(),
            1,
            "an identical graph is a no-op batch"
        );
    }
}
