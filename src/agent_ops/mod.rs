//! Agent-facing operation layer: a JSON edit protocol over the
//! editor's own primitives.
//!
//! An external agent (Claude, ChatGPT, …) can't click a viewport, and
//! asking an LLM to emit voxels one at a time is hopeless token
//! economics. So this module exposes the *high-level* primitives the
//! editor already has — boxes, spheres, cylinders, lines, hollowing,
//! selection transforms, parametric generators — as a batch of JSON
//! ops, and hands back a report the agent can read to decide what to
//! do next.
//!
//! ```text
//! OpsBatch (JSON) → validate → run on a scratch World → one
//!   Command::SetVoxels → CommandHistory → ApplyReport (JSON)
//! ```
//!
//! Three properties are load-bearing, in this order:
//!
//! 1. **Atomic.** Ops run against a [`World::deep_clone`] of the
//!    session world. Any failure discards the copy, so a batch either
//!    lands whole or changes nothing. An agent recovers from a clean
//!    failure far better than from half an edit.
//! 2. **Sequential.** Each op sees the results of the ones before it,
//!    which is what makes `rotate` / `mirror` / `hollow` mean anything
//!    inside a batch.
//! 3. **One undo entry per batch.** The accumulated changes commit as a
//!    single [`Command::set_voxels`], so a human at the editor undoes an
//!    agent's whole step with one Ctrl+Z.
//!
//! Errors are the product here as much as the edits are: every failure
//! carries the offending `op_index`, a stable machine-readable
//! [`ErrorCode`], and a message that says what to do instead. That is
//! the loop an agent iterates on.
//!
//! Dependency direction is one-way — `agent_ops → editor / procgen /
//! core` — the same discipline `ai → procgen` follows. Nothing here
//! knows about `app`, winit, or egui; file dialogs and other UI-bound
//! side effects are deliberately absent (an agent-triggered `rfd`
//! dialog would block the main thread with no one to click it).

use serde::Serialize;

use crate::core::World;
use crate::editor::{Command, CommandHistory, Selection, Socket};

mod compile;
mod describe;
mod registry;
mod schema;

pub use describe::{ColorCount, Description, SliceMode, SliceRequest, SocketInfo};
pub use registry::{generator_infos, GeneratorInfo};
pub use schema::{
    Aabb, AxisSpec, BatchOptions, Op, OpsBatch, SolidVoxel, VoxelEntry, VoxelSpec, WriteMode,
};

/// Wire-format version an [`OpsBatch`] must declare. Bumped only for a
/// breaking change; additive fields keep version 1 (new fields get
/// `#[serde(default)]`, the same forward-compat discipline `prefs.ron`
/// and `.vxlt` follow).
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
/// writes, scans, or has masked out by its `write_mode`. It's an
/// upper-bound accounting on purpose: the point is to bound work and
/// memory, not to bill precisely.
pub const MAX_BATCH_CELLS: u64 = 8_388_608;

/// Coordinate ceiling on every axis. The world itself is unbounded —
/// any `i32` cell is writable — but an agent that means `y = 5` and
/// emits `y = 5000000` should hear about it rather than silently
/// building 5 km up. It also keeps the mirror reflection (`2·plane −
/// 1 − p`) and shape bounding boxes far away from `i32` overflow.
pub const MAX_COORD: i32 = 1 << 20;

/// Chunks one batch may bring into existence beyond what the session
/// already had. Each is a 256 KB allocation, so scattered writes are
/// the cheap way to ask for a lot of memory: 4096 stray `set_voxels`
/// entries in 4096 distinct chunks is a gigabyte. The cap is on *new*
/// chunks, not total, so a legitimately large document keeps working.
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

/// A batch run against a copy of the session, committing nothing.
///
/// This is what makes a dry run an actual *preview* rather than a
/// prediction the agent has to take on faith: the caller can
/// [`describe`](AgentSession::describe) or [`slice`](AgentSession::slice)
/// the result before deciding to run it for real. Without it, a dry run
/// paired with a description reports two contradictory pictures of the
/// document in one breath — the report describing the world after the
/// batch, the description the world before it.
pub struct Preview {
    /// Always `dry_run: true` — a preview commits nothing, whatever the
    /// batch asked for.
    pub report: ApplyReport,
    /// The session as the batch would leave it: world and selection
    /// after the last op, sockets carried across. Its history is empty,
    /// because nothing was committed — `describe()` on a preview
    /// therefore reports depth 0 rather than the real session's.
    pub session: AgentSession,
}

/// A headless editing session: the document an agent operates on.
///
/// Fields are public because the consumers above this layer (the CLI in
/// P1, the MCP server in P2) legitimately need to swap the world in
/// after loading a `.vxlt` or read the sockets out for export.
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
        }
    }

    /// Validate and run a batch. On success the whole batch has been
    /// applied (or, for a dry run, would have been); on failure nothing
    /// changed.
    pub fn apply_ops(&mut self, batch: &OpsBatch) -> Result<ApplyReport, OpsError> {
        let mut outcome = self.run(batch)?;
        let report = report_of(batch.options.dry_run, batch.ops.len(), &mut outcome);

        if !batch.options.dry_run {
            // One command for the whole batch: one undo entry, and a
            // no-op batch pushes nothing (`CommandHistory::execute`
            // drops no-ops).
            self.history
                .execute(Command::set_voxels(outcome.changes), &mut self.world);
            self.selection = outcome.selection;
        }
        Ok(report)
    }

    /// Run a batch and hand back the result instead of committing it.
    ///
    /// Same validation, same executor, same report as [`apply_ops`] —
    /// they share [`run`](Self::run) precisely so a preview and the real
    /// thing can't drift apart — but the session is left untouched and
    /// the resulting world comes back inside a [`Preview`] to be looked
    /// at.
    pub fn preview_ops(&self, batch: &OpsBatch) -> Result<Preview, OpsError> {
        let mut outcome = self.run(batch)?;
        let report = report_of(true, batch.ops.len(), &mut outcome);
        Ok(Preview {
            report,
            session: AgentSession {
                world: outcome.world,
                history: CommandHistory::new(HISTORY_DEPTH),
                selection: outcome.selection,
                sockets: self.sockets.clone(),
            },
        })
    }

    /// Envelope checks, then every op against a scratch copy of the
    /// world. The one execution path, shared by the real run and the
    /// preview.
    fn run(&self, batch: &OpsBatch) -> Result<compile::Outcome, OpsError> {
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

        let mut scratch = compile::Scratch::new(&self.world, self.selection);
        for (index, op) in batch.ops.iter().enumerate() {
            scratch.run_op(index, op).map_err(|e| e.at(index))?;
        }
        Ok(scratch.finish())
    }

    /// Text summary of the document — the agent's cheapest look at what
    /// it just built.
    pub fn describe(&self) -> Description {
        describe::describe(self)
    }

    /// One axis-aligned plane as ASCII art.
    pub fn slice(&self, request: &SliceRequest) -> Result<String, OpsError> {
        describe::slice(&self.world, request)
    }

    pub fn undo(&mut self) -> bool {
        self.history.undo(&mut self.world)
    }

    pub fn redo(&mut self) -> bool {
        self.history.redo(&mut self.world)
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
