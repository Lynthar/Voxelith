//! Mesh generation from voxel data.
//!
//! This module converts voxel chunks into renderable triangle meshes.
//! Multiple meshing strategies are supported:
//! - Naive: Simple but generates many triangles
//! - Greedy: Optimized mesh with merged faces (TODO)
//! - Marching Cubes: Smooth surfaces (TODO)

mod vertex;
mod naive;

pub use vertex::{Vertex, ChunkMesh};
pub use naive::NaiveMesher;

use crate::core::{Chunk, ChunkPos};

/// Trait for mesh generation strategies
pub trait Mesher {
    /// Generate mesh for a chunk
    fn generate(&self, chunk: &Chunk, chunk_pos: ChunkPos) -> ChunkMesh;
}

/// Face direction for voxel faces
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Face {
    /// +X direction (right)
    PosX = 0,
    /// -X direction (left)
    NegX = 1,
    /// +Y direction (up)
    PosY = 2,
    /// -Y direction (down)
    NegY = 3,
    /// +Z direction (front)
    PosZ = 4,
    /// -Z direction (back)
    NegZ = 5,
}

impl Face {
    /// Get normal vector for this face
    pub fn normal(&self) -> [f32; 3] {
        match self {
            Face::PosX => [1.0, 0.0, 0.0],
            Face::NegX => [-1.0, 0.0, 0.0],
            Face::PosY => [0.0, 1.0, 0.0],
            Face::NegY => [0.0, -1.0, 0.0],
            Face::PosZ => [0.0, 0.0, 1.0],
            Face::NegZ => [0.0, 0.0, -1.0],
        }
    }

    /// Get direction offset
    pub fn offset(&self) -> (i32, i32, i32) {
        match self {
            Face::PosX => (1, 0, 0),
            Face::NegX => (-1, 0, 0),
            Face::PosY => (0, 1, 0),
            Face::NegY => (0, -1, 0),
            Face::PosZ => (0, 0, 1),
            Face::NegZ => (0, 0, -1),
        }
    }

    /// All six faces
    pub const ALL: [Face; 6] = [
        Face::PosX,
        Face::NegX,
        Face::PosY,
        Face::NegY,
        Face::PosZ,
        Face::NegZ,
    ];
}
