//! Chunk: A fixed-size 3D grid of voxels.
//!
//! Chunks are the basic unit of voxel storage and rendering.
//! They provide efficient spatial access and modification of voxels.

use super::Voxel;
use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};

// Note: Chunk does not derive Serialize/Deserialize because of the large voxel array.
// Custom serialization will be implemented in the io module.

/// Chunk size in each dimension (32³ = 32,768 voxels per chunk)
/// This is a good balance between memory usage and granularity.
/// For large worlds, consider 64³ for fewer chunks.
pub const CHUNK_SIZE: usize = 32;
pub const CHUNK_SIZE_I32: i32 = CHUNK_SIZE as i32;
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

/// Position of a chunk in world space (in chunk coordinates, not voxel coordinates)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct ChunkPos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkPos {
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };

    #[inline]
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Convert world voxel position to chunk position
    #[inline]
    pub fn from_world_pos(x: i32, y: i32, z: i32) -> Self {
        Self {
            x: x.div_euclid(CHUNK_SIZE_I32),
            y: y.div_euclid(CHUNK_SIZE_I32),
            z: z.div_euclid(CHUNK_SIZE_I32),
        }
    }

    /// Get the world position of this chunk's origin (min corner)
    #[inline]
    pub fn world_origin(&self) -> (i32, i32, i32) {
        (
            self.x * CHUNK_SIZE_I32,
            self.y * CHUNK_SIZE_I32,
            self.z * CHUNK_SIZE_I32,
        )
    }

    /// Get neighbor chunk position in the given direction
    #[inline]
    pub fn neighbor(&self, dx: i32, dy: i32, dz: i32) -> Self {
        Self {
            x: self.x + dx,
            y: self.y + dy,
            z: self.z + dz,
        }
    }
}

impl From<(i32, i32, i32)> for ChunkPos {
    fn from((x, y, z): (i32, i32, i32)) -> Self {
        Self { x, y, z }
    }
}

impl From<[i32; 3]> for ChunkPos {
    fn from([x, y, z]: [i32; 3]) -> Self {
        Self { x, y, z }
    }
}

/// Local position within a chunk (0 to CHUNK_SIZE-1 for each axis)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalPos {
    pub x: u8,
    pub y: u8,
    pub z: u8,
}

impl LocalPos {
    #[inline]
    pub fn new(x: u8, y: u8, z: u8) -> Self {
        debug_assert!((x as usize) < CHUNK_SIZE);
        debug_assert!((y as usize) < CHUNK_SIZE);
        debug_assert!((z as usize) < CHUNK_SIZE);
        Self { x, y, z }
    }

    /// Convert world position to local position within a chunk
    #[inline]
    pub fn from_world_pos(x: i32, y: i32, z: i32) -> Self {
        Self {
            x: x.rem_euclid(CHUNK_SIZE_I32) as u8,
            y: y.rem_euclid(CHUNK_SIZE_I32) as u8,
            z: z.rem_euclid(CHUNK_SIZE_I32) as u8,
        }
    }

    /// Convert to linear index for array access
    #[inline]
    pub fn to_index(self) -> usize {
        (self.x as usize)
            + (self.y as usize) * CHUNK_SIZE
            + (self.z as usize) * CHUNK_SIZE * CHUNK_SIZE
    }

    /// Convert from linear index
    #[inline]
    pub fn from_index(index: usize) -> Self {
        debug_assert!(index < CHUNK_VOLUME);
        Self {
            x: (index % CHUNK_SIZE) as u8,
            y: ((index / CHUNK_SIZE) % CHUNK_SIZE) as u8,
            z: (index / (CHUNK_SIZE * CHUNK_SIZE)) as u8,
        }
    }
}

/// A chunk containing a 3D grid of voxels.
///
/// Voxels are stored in a flat array for cache efficiency.
/// Layout: x + y*SIZE + z*SIZE*SIZE (x varies fastest)
#[derive(Clone)]
pub struct Chunk {
    /// Flat array of voxels (using Vec for serde compatibility)
    voxels: Vec<Voxel>,
    /// Number of non-air voxels (for quick empty check)
    solid_count: u32,
    /// Flag indicating mesh needs rebuild
    dirty: bool,
}

impl Default for Chunk {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunk {
    /// Create a new empty chunk (all air)
    pub fn new() -> Self {
        Self {
            voxels: vec![Voxel::AIR; CHUNK_VOLUME],
            solid_count: 0,
            dirty: true,
        }
    }

    /// Create a chunk filled with a single voxel type
    pub fn filled(voxel: Voxel) -> Self {
        let solid_count = if voxel.is_solid() {
            CHUNK_VOLUME as u32
        } else {
            0
        };
        Self {
            voxels: vec![voxel; CHUNK_VOLUME],
            solid_count,
            dirty: true,
        }
    }

    /// Check if chunk is completely empty (all air)
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.solid_count == 0
    }

    /// Check if chunk is completely filled (no air)
    #[inline]
    pub fn is_full(&self) -> bool {
        self.solid_count == CHUNK_VOLUME as u32
    }

    /// Get number of solid voxels
    #[inline]
    pub fn solid_count(&self) -> u32 {
        self.solid_count
    }

    /// Check if mesh needs to be rebuilt
    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Mark chunk as needing mesh rebuild
    #[inline]
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Clear dirty flag (call after mesh rebuild)
    #[inline]
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    /// Get voxel at local position
    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> Voxel {
        debug_assert!(x < CHUNK_SIZE && y < CHUNK_SIZE && z < CHUNK_SIZE);
        self.voxels[x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE]
    }

    /// Get voxel at local position (safe version with bounds check)
    #[inline]
    pub fn get_safe(&self, x: i32, y: i32, z: i32) -> Option<Voxel> {
        if x >= 0
            && x < CHUNK_SIZE_I32
            && y >= 0
            && y < CHUNK_SIZE_I32
            && z >= 0
            && z < CHUNK_SIZE_I32
        {
            Some(self.get(x as usize, y as usize, z as usize))
        } else {
            None
        }
    }

    /// Set voxel at local position
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, voxel: Voxel) {
        debug_assert!(x < CHUNK_SIZE && y < CHUNK_SIZE && z < CHUNK_SIZE);
        let index = x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE;
        let old = &mut self.voxels[index];

        // Update solid count
        if old.is_solid() && voxel.is_air() {
            self.solid_count -= 1;
        } else if old.is_air() && voxel.is_solid() {
            self.solid_count += 1;
        }

        *old = voxel;
        self.dirty = true;
    }

    /// Get raw voxel slice (for mesh generation)
    #[inline]
    pub fn voxels(&self) -> &[Voxel] {
        &self.voxels
    }

    /// Iterate over all voxels with their positions
    pub fn iter_voxels(&self) -> impl Iterator<Item = (LocalPos, &Voxel)> {
        self.voxels.iter().enumerate().map(|(i, v)| {
            (LocalPos::from_index(i), v)
        })
    }

    /// Iterate over all solid voxels with their positions
    pub fn iter_solid(&self) -> impl Iterator<Item = (LocalPos, &Voxel)> {
        self.iter_voxels().filter(|(_, v)| v.is_solid())
    }

    /// Fill a region with a voxel
    pub fn fill_region(
        &mut self,
        min: (usize, usize, usize),
        max: (usize, usize, usize),
        voxel: Voxel,
    ) {
        for z in min.2..=max.2.min(CHUNK_SIZE - 1) {
            for y in min.1..=max.1.min(CHUNK_SIZE - 1) {
                for x in min.0..=max.0.min(CHUNK_SIZE - 1) {
                    self.set(x, y, z, voxel);
                }
            }
        }
    }
}

impl Index<LocalPos> for Chunk {
    type Output = Voxel;

    #[inline]
    fn index(&self, pos: LocalPos) -> &Self::Output {
        &self.voxels[pos.to_index()]
    }
}

impl IndexMut<LocalPos> for Chunk {
    #[inline]
    fn index_mut(&mut self, pos: LocalPos) -> &mut Self::Output {
        self.dirty = true;
        &mut self.voxels[pos.to_index()]
    }
}

impl Index<(usize, usize, usize)> for Chunk {
    type Output = Voxel;

    #[inline]
    fn index(&self, (x, y, z): (usize, usize, usize)) -> &Self::Output {
        &self.voxels[x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_size() {
        // Chunk should be approximately 256KB (32³ * 8 bytes per voxel)
        assert_eq!(CHUNK_VOLUME * std::mem::size_of::<Voxel>(), 262144);
    }

    #[test]
    fn test_local_pos_index_roundtrip() {
        for i in 0..CHUNK_VOLUME {
            let pos = LocalPos::from_index(i);
            assert_eq!(pos.to_index(), i);
        }
    }

    #[test]
    fn test_chunk_solid_count() {
        let mut chunk = Chunk::new();
        assert_eq!(chunk.solid_count(), 0);
        assert!(chunk.is_empty());

        chunk.set(0, 0, 0, Voxel::from_rgb(255, 0, 0));
        assert_eq!(chunk.solid_count(), 1);

        chunk.set(0, 0, 0, Voxel::AIR);
        assert_eq!(chunk.solid_count(), 0);
    }

    #[test]
    fn test_chunk_pos_from_world() {
        assert_eq!(ChunkPos::from_world_pos(0, 0, 0), ChunkPos::new(0, 0, 0));
        assert_eq!(ChunkPos::from_world_pos(31, 31, 31), ChunkPos::new(0, 0, 0));
        assert_eq!(ChunkPos::from_world_pos(32, 0, 0), ChunkPos::new(1, 0, 0));
        assert_eq!(ChunkPos::from_world_pos(-1, 0, 0), ChunkPos::new(-1, 0, 0));
        assert_eq!(ChunkPos::from_world_pos(-32, 0, 0), ChunkPos::new(-1, 0, 0));
        assert_eq!(ChunkPos::from_world_pos(-33, 0, 0), ChunkPos::new(-2, 0, 0));
    }
}
