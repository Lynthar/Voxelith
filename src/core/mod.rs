//! Core voxel data structures and world management.
//!
//! This module provides the fundamental building blocks for voxel storage:
//! - `Voxel`: Individual voxel data (material, color)
//! - `Chunk`: Fixed-size 3D grid of voxels
//! - `World`: Collection of chunks with spatial indexing

mod chunk;
mod voxel;
mod world;

pub use chunk::{Chunk, ChunkPos, LocalPos, CHUNK_SIZE, CHUNK_SIZE_I32, CHUNK_VOLUME};
pub use voxel::Voxel;
pub use world::{CellAabb, World};
