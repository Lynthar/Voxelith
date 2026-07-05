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

use std::collections::HashMap;
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

    /// Collapse duplicate positions, keeping the **last** write for each,
    /// returned in first-seen order. Generators legitimately overwrite a
    /// cell — `LSystemTree` rasterizes a trunk voxel, then draws a leaf
    /// cluster over the same branch tip — and the final write is the
    /// intended value.
    ///
    /// Load-bearing for undo determinism. Consumers turn a patch into a
    /// `SetVoxels` command by reading each position's *world* old-value
    /// once and dropping identity writes. If duplicates survive to that
    /// step, both records read the same world old-value, so which one the
    /// identity filter keeps depends on the current world — re-running the
    /// same generator then flips the cell (trunk↔leaf) every run. Deduping
    /// first makes the conversion idempotent.
    pub fn dedup_last_write(&self) -> Vec<((i32, i32, i32), Voxel)> {
        let mut final_voxel: HashMap<(i32, i32, i32), Voxel> =
            HashMap::with_capacity(self.voxels.len());
        let mut order: Vec<(i32, i32, i32)> = Vec::with_capacity(self.voxels.len());
        for &(pos, voxel) in &self.voxels {
            // First sighting fixes draw order; a repeat only refreshes the
            // stored value, so `order` stays unique and first-seen-ordered.
            if final_voxel.insert(pos, voxel).is_none() {
                order.push(pos);
            }
        }
        order.into_iter().map(|pos| (pos, final_voxel[&pos])).collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::World;

    #[test]
    fn dedup_last_write_keeps_last_value_in_first_seen_order() {
        // Mirrors the LSystemTree trunk-then-leaf case: one cell is
        // written twice and the LAST write (leaf) must win; distinct
        // cells keep their first-seen order and single value.
        let trunk = Voxel::from_rgb(139, 90, 43);
        let leaf = Voxel::from_rgb(76, 153, 0);
        let stone = Voxel::from_rgb(128, 128, 128);

        let mut patch = VoxelPatch::new();
        patch.set(0, 0, 0, trunk); // branch tip: trunk first...
        patch.set(5, 0, 0, stone); // an unrelated cell
        patch.set(0, 0, 0, leaf); //  ...then a leaf over it (last write)

        assert_eq!(
            patch.dedup_last_write(),
            vec![((0, 0, 0), leaf), ((5, 0, 0), stone)]
        );
    }

    #[test]
    fn dedup_stops_rerun_oscillation_through_the_identity_filter() {
        // Build the change list exactly like `App::patch_to_changes`:
        // read every position's old value from the pre-apply world, drop
        // identity writes, THEN apply. Reading all olds up front is what
        // makes an un-deduped duplicate flip the cell on each re-run.
        fn apply_deduped(patch: &VoxelPatch, world: &mut World) {
            let changes: Vec<_> = patch
                .dedup_last_write()
                .into_iter()
                .filter(|&(pos, new)| world.get_voxel(pos.0, pos.1, pos.2) != new)
                .collect();
            for (pos, v) in changes {
                world.set_voxel(pos.0, pos.1, pos.2, v);
            }
        }
        fn apply_undeduped(patch: &VoxelPatch, world: &mut World) {
            let changes: Vec<_> = patch
                .voxels
                .iter()
                .copied()
                .filter(|&(pos, new)| world.get_voxel(pos.0, pos.1, pos.2) != new)
                .collect();
            for (pos, v) in changes {
                world.set_voxel(pos.0, pos.1, pos.2, v);
            }
        }

        let trunk = Voxel::from_rgb(139, 90, 43);
        let leaf = Voxel::from_rgb(76, 153, 0);
        let mut patch = VoxelPatch::new();
        patch.set(0, 0, 0, trunk);
        patch.set(0, 0, 0, leaf); // duplicate cell — leaf is the final value

        // With the fix: first run shows leaf; re-run is a no-op.
        let mut world = World::new();
        apply_deduped(&patch, &mut world);
        assert_eq!(world.get_voxel(0, 0, 0), leaf);
        apply_deduped(&patch, &mut world);
        assert_eq!(
            world.get_voxel(0, 0, 0),
            leaf,
            "deduped re-run must be idempotent, not oscillate"
        );

        // Without the fix: the identity filter drops a different duplicate
        // each run, so the cell flips leaf → trunk on the second run.
        let mut buggy = World::new();
        apply_undeduped(&patch, &mut buggy);
        assert_eq!(buggy.get_voxel(0, 0, 0), leaf);
        apply_undeduped(&patch, &mut buggy);
        assert_eq!(
            buggy.get_voxel(0, 0, 0),
            trunk,
            "documents the pre-fix oscillation the dedup removes"
        );
    }
}
