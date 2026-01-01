//! World: Collection of chunks with spatial indexing.
//!
//! The World provides a unified interface for accessing voxels across
//! multiple chunks, handling chunk boundaries transparently.

use super::{Chunk, ChunkPos, Voxel, CHUNK_SIZE, CHUNK_SIZE_I32};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// A world containing multiple chunks.
///
/// Supports both bounded (fixed-size) and unbounded (infinite) modes.
/// Thread-safe access is provided through RwLock.
#[derive(Default)]
pub struct World {
    /// Chunks indexed by their position
    chunks: HashMap<ChunkPos, Arc<RwLock<Chunk>>>,
    /// World bounds (None = unbounded/infinite)
    bounds: Option<WorldBounds>,
    /// Flag for tracking if any chunk is dirty
    any_dirty: bool,
}

/// Bounds for a finite world
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WorldBounds {
    pub min: ChunkPos,
    pub max: ChunkPos,
}

impl WorldBounds {
    pub fn new(min: ChunkPos, max: ChunkPos) -> Self {
        Self { min, max }
    }

    /// Create bounds for a single-chunk world at origin
    pub fn single_chunk() -> Self {
        Self {
            min: ChunkPos::ZERO,
            max: ChunkPos::ZERO,
        }
    }

    /// Create bounds for a world of given size in chunks, centered at origin
    pub fn centered(half_size: i32) -> Self {
        Self {
            min: ChunkPos::new(-half_size, -half_size, -half_size),
            max: ChunkPos::new(half_size, half_size, half_size),
        }
    }

    /// Check if a chunk position is within bounds
    pub fn contains(&self, pos: ChunkPos) -> bool {
        pos.x >= self.min.x
            && pos.x <= self.max.x
            && pos.y >= self.min.y
            && pos.y <= self.max.y
            && pos.z >= self.min.z
            && pos.z <= self.max.z
    }

    /// Get size in chunks for each dimension
    pub fn size(&self) -> (u32, u32, u32) {
        (
            (self.max.x - self.min.x + 1) as u32,
            (self.max.y - self.min.y + 1) as u32,
            (self.max.z - self.min.z + 1) as u32,
        )
    }

    /// Get size in voxels for each dimension
    pub fn size_voxels(&self) -> (u32, u32, u32) {
        let (cx, cy, cz) = self.size();
        (
            cx * CHUNK_SIZE as u32,
            cy * CHUNK_SIZE as u32,
            cz * CHUNK_SIZE as u32,
        )
    }
}

impl World {
    /// Create a new empty unbounded world
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a bounded world with the given bounds
    pub fn bounded(bounds: WorldBounds) -> Self {
        Self {
            chunks: HashMap::new(),
            bounds: Some(bounds),
            any_dirty: false,
        }
    }

    /// Create a world with a single chunk at origin
    pub fn single_chunk() -> Self {
        let mut world = Self::bounded(WorldBounds::single_chunk());
        world.get_or_create_chunk(ChunkPos::ZERO);
        world
    }

    /// Get world bounds (None if unbounded)
    pub fn bounds(&self) -> Option<&WorldBounds> {
        self.bounds.as_ref()
    }

    /// Check if a chunk exists at the given position
    pub fn has_chunk(&self, pos: ChunkPos) -> bool {
        self.chunks.contains_key(&pos)
    }

    /// Get chunk at position (returns None if not loaded)
    pub fn get_chunk(&self, pos: ChunkPos) -> Option<Arc<RwLock<Chunk>>> {
        self.chunks.get(&pos).cloned()
    }

    /// Get or create chunk at position
    pub fn get_or_create_chunk(&mut self, pos: ChunkPos) -> Arc<RwLock<Chunk>> {
        // Check bounds if set
        if let Some(bounds) = &self.bounds {
            if !bounds.contains(pos) {
                // Return empty chunk for out-of-bounds access
                return Arc::new(RwLock::new(Chunk::new()));
            }
        }

        self.chunks
            .entry(pos)
            .or_insert_with(|| Arc::new(RwLock::new(Chunk::new())))
            .clone()
    }

    /// Get voxel at world position
    pub fn get_voxel(&self, x: i32, y: i32, z: i32) -> Voxel {
        let chunk_pos = ChunkPos::from_world_pos(x, y, z);
        if let Some(chunk) = self.get_chunk(chunk_pos) {
            let lx = x.rem_euclid(CHUNK_SIZE_I32) as usize;
            let ly = y.rem_euclid(CHUNK_SIZE_I32) as usize;
            let lz = z.rem_euclid(CHUNK_SIZE_I32) as usize;
            chunk.read().get(lx, ly, lz)
        } else {
            Voxel::AIR
        }
    }

    /// Set voxel at world position
    pub fn set_voxel(&mut self, x: i32, y: i32, z: i32, voxel: Voxel) {
        let chunk_pos = ChunkPos::from_world_pos(x, y, z);
        let chunk = self.get_or_create_chunk(chunk_pos);

        let lx = x.rem_euclid(CHUNK_SIZE_I32) as usize;
        let ly = y.rem_euclid(CHUNK_SIZE_I32) as usize;
        let lz = z.rem_euclid(CHUNK_SIZE_I32) as usize;

        chunk.write().set(lx, ly, lz, voxel);
        self.any_dirty = true;
    }

    /// Fill a region with a voxel
    pub fn fill_region(&mut self, min: (i32, i32, i32), max: (i32, i32, i32), voxel: Voxel) {
        for z in min.2..=max.2 {
            for y in min.1..=max.1 {
                for x in min.0..=max.0 {
                    self.set_voxel(x, y, z, voxel);
                }
            }
        }
    }

    /// Get all loaded chunk positions
    pub fn chunk_positions(&self) -> impl Iterator<Item = &ChunkPos> {
        self.chunks.keys()
    }

    /// Get all chunks
    pub fn chunks(&self) -> impl Iterator<Item = (&ChunkPos, &Arc<RwLock<Chunk>>)> {
        self.chunks.iter()
    }

    /// Get number of loaded chunks
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Check if any chunk needs mesh rebuild
    pub fn has_dirty_chunks(&self) -> bool {
        self.any_dirty
            || self
                .chunks
                .values()
                .any(|c| c.read().is_dirty())
    }

    /// Get all dirty chunks
    pub fn dirty_chunks(&self) -> Vec<ChunkPos> {
        self.chunks
            .iter()
            .filter(|(_, c)| c.read().is_dirty())
            .map(|(pos, _)| *pos)
            .collect()
    }

    /// Clear all dirty flags
    pub fn clear_dirty_flags(&mut self) {
        for chunk in self.chunks.values() {
            chunk.write().clear_dirty();
        }
        self.any_dirty = false;
    }

    /// Remove empty chunks to free memory
    pub fn prune_empty_chunks(&mut self) {
        self.chunks.retain(|_, chunk| !chunk.read().is_empty());
    }

    /// Clear all chunks
    pub fn clear(&mut self) {
        self.chunks.clear();
        self.any_dirty = true;
    }

    /// Create a simple test world with a ground plane
    pub fn create_test_ground(&mut self, size: i32, height: i32) {
        let half = size / 2;
        for z in -half..=half {
            for y in 0..height {
                for x in -half..=half {
                    // Grass on top, dirt below
                    let voxel = if y == height - 1 {
                        Voxel::from_rgb(76, 153, 0) // Grass green
                    } else {
                        Voxel::from_rgb(139, 90, 43) // Dirt brown
                    };
                    self.set_voxel(x, y, z, voxel);
                }
            }
        }
    }

    /// Create a simple colored cube for testing
    pub fn create_test_cube(&mut self, center: (i32, i32, i32), half_size: i32) {
        let colors = [
            (255, 0, 0),   // Red
            (0, 255, 0),   // Green
            (0, 0, 255),   // Blue
            (255, 255, 0), // Yellow
            (255, 0, 255), // Magenta
            (0, 255, 255), // Cyan
        ];

        for z in -half_size..=half_size {
            for y in -half_size..=half_size {
                for x in -half_size..=half_size {
                    // Choose color based on position
                    let color_idx = ((x + y + z).abs() as usize) % colors.len();
                    let (r, g, b) = colors[color_idx];
                    let voxel = Voxel::from_rgb(r, g, b);
                    self.set_voxel(
                        center.0 + x,
                        center.1 + y,
                        center.2 + z,
                        voxel,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_get_set() {
        let mut world = World::new();

        // Set and get voxel
        let voxel = Voxel::from_rgb(255, 0, 0);
        world.set_voxel(10, 20, 30, voxel);
        assert_eq!(world.get_voxel(10, 20, 30), voxel);

        // Unset voxel should be air
        assert!(world.get_voxel(0, 0, 0).is_air());
    }

    #[test]
    fn test_world_cross_chunk() {
        let mut world = World::new();

        // Set voxels in different chunks
        world.set_voxel(-1, 0, 0, Voxel::from_rgb(255, 0, 0));
        world.set_voxel(0, 0, 0, Voxel::from_rgb(0, 255, 0));
        world.set_voxel(32, 0, 0, Voxel::from_rgb(0, 0, 255));

        assert_eq!(world.chunk_count(), 3);
    }

    #[test]
    fn test_bounded_world() {
        let bounds = WorldBounds::centered(1);
        let mut world = World::bounded(bounds);

        // Inside bounds - should work
        world.set_voxel(0, 0, 0, Voxel::from_rgb(255, 0, 0));
        assert!(!world.get_voxel(0, 0, 0).is_air());

        // Way outside bounds - should return air (not crash)
        assert!(world.get_voxel(1000, 1000, 1000).is_air());
    }
}
