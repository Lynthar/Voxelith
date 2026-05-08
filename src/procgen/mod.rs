//! Procedural generation algorithms.
//!
//! This module hosts the unified entry point for both algorithmic
//! generators (noise, WFC, L-System, ...) and, eventually, AI
//! generators. They all implement [`VoxelGenerator`] and emit a
//! [`VoxelPatch`] — a list of voxel writes — rather than mutating a
//! `World` directly. Decoupling the output lets callers route the
//! result through [`CommandHistory`] (for undo), AI format converters,
//! or preview/scratch worlds without changing the generator.
//!
//! [`CommandHistory`]: crate::editor::CommandHistory

mod graph;
mod terrain;
mod tree;
mod wfc;

pub use graph::{
    CombineOp, FilterPredicate, GraphError, GraphNode, MaskMode, NodeId,
    NodeKind, PipelineGraph,
};
pub use terrain::PerlinTerrain;
pub use tree::LSystemTree;
pub use wfc::{WfcGenerator, WfcTileset, WFC_TILE_SIZE};

use std::time::Duration;
use thiserror::Error;

use crate::core::{Voxel, World};

/// Errors raised by generators.
#[derive(Debug, Error)]
pub enum GenError {
    #[error("Generation failed: {0}")]
    Failed(String),
    #[error("Invalid parameters: {0}")]
    InvalidParams(String),
    #[error("Generation timeout")]
    Timeout,
}

pub type GenResult<T> = Result<T, GenError>;

/// Coarse classification — drives default placement, palette hints,
/// and which UI panel groups the generator under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratorCategory {
    Terrain,
    Building,
    Character,
    Prop,
    Vegetation,
    General,
}

/// How the generator runs. AI variants are stubs today; the current
/// generators are all `Algorithmic`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratorBackend {
    Algorithmic,
    LocalModel,
    RemoteAPI,
    Hybrid,
}

/// Static metadata describing a generator. Concrete generators return
/// their own (usually compile-time-constant) metadata from
/// [`VoxelGenerator::metadata`].
#[derive(Debug, Clone, Copy)]
pub struct GeneratorMeta {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub category: GeneratorCategory,
    pub backend: GeneratorBackend,
}

/// A bundle of voxel writes produced by a generator.
///
/// Stored as a flat `Vec<(pos, voxel)>` rather than a dense buffer
/// because most generators are sparse (terrain heightmaps, scattered
/// trees) and the apply-via-`CommandHistory` path needs the same
/// representation anyway.
///
/// `notes` carries per-run diagnostics — non-fatal warnings the
/// generator wants the UI to surface (e.g. WFC's "N cells fell back
/// to empty"). Consumers that don't care can ignore the field; it
/// defaults to empty.
#[derive(Debug, Clone, Default)]
pub struct VoxelPatch {
    pub voxels: Vec<((i32, i32, i32), Voxel)>,
    pub notes: Vec<String>,
}

impl VoxelPatch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(n: usize) -> Self {
        Self {
            voxels: Vec::with_capacity(n),
            notes: Vec::new(),
        }
    }

    pub fn set(&mut self, x: i32, y: i32, z: i32, voxel: Voxel) {
        self.voxels.push(((x, y, z), voxel));
    }

    pub fn len(&self) -> usize {
        self.voxels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.voxels.is_empty()
    }

    /// Apply directly to a world (no undo). Prefer routing through
    /// `CommandHistory::execute(Command::set_voxels(...))` when the
    /// caller is inside an editor session — that path is reversible.
    pub fn apply(&self, world: &mut World) {
        for &((x, y, z), voxel) in &self.voxels {
            world.set_voxel(x, y, z, voxel);
        }
    }
}

/// Trait every voxel generator implements.
///
/// Inputs (seeds, prompts, dimensions, ...) live as fields on the
/// concrete type. The same struct that holds parameter state is the
/// thing that runs — UI panels can edit fields in place and call
/// [`generate`](VoxelGenerator::generate) without any glue.
///
/// `Send + Sync` is required so generators can be moved to a worker
/// thread (algorithmic ones are pure CPU; AI ones may block on I/O).
pub trait VoxelGenerator: Send + Sync {
    fn metadata(&self) -> GeneratorMeta;

    /// Run the generator and return its output patch.
    fn generate(&self) -> GenResult<VoxelPatch>;

    /// Whether the generator can produce output incrementally.
    /// Default: false.
    fn supports_incremental(&self) -> bool {
        false
    }

    /// Hint for UI progress display. Default: zero.
    fn estimate_duration(&self) -> Duration {
        Duration::ZERO
    }
}
