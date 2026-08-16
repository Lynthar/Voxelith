//! Core voxel storage: `Voxel` (material and color), `Chunk` (a
//! fixed-size 3D grid) and `World` (chunks with spatial indexing).

mod chunk;
mod voxel;
mod world;

pub use chunk::{Chunk, ChunkPos, LocalPos, CHUNK_SIZE, CHUNK_SIZE_I32, CHUNK_VOLUME};
pub use voxel::Voxel;
pub use world::{CellAabb, World};
