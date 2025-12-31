//! Naive meshing: Generate one quad per visible voxel face.
//!
//! This is the simplest meshing approach. It's not optimized but
//! produces correct results and is easy to understand.

use super::{ChunkMesh, Face, Mesher, Vertex};
use crate::core::{Chunk, ChunkPos, CHUNK_SIZE, CHUNK_SIZE_I32};

/// Naive mesher that generates individual quads for each visible face.
pub struct NaiveMesher;

impl NaiveMesher {
    pub fn new() -> Self {
        Self
    }

    /// Check if a face should be rendered (neighbor is air)
    fn is_face_visible(chunk: &Chunk, x: i32, y: i32, z: i32, face: Face) -> bool {
        let (dx, dy, dz) = face.offset();
        let nx = x + dx;
        let ny = y + dy;
        let nz = z + dz;

        // If neighbor is outside chunk, assume visible (will be handled by adjacent chunk)
        if nx < 0 || nx >= CHUNK_SIZE_I32 || ny < 0 || ny >= CHUNK_SIZE_I32 || nz < 0 || nz >= CHUNK_SIZE_I32 {
            return true;
        }

        // Face is visible if neighbor is air
        chunk.get(nx as usize, ny as usize, nz as usize).is_air()
    }

    /// Generate vertices for a face at the given position
    fn generate_face_vertices(
        x: f32,
        y: f32,
        z: f32,
        face: Face,
        color: [f32; 4],
    ) -> [Vertex; 4] {
        let normal = face.normal();

        // Define vertex positions for each face
        // Vertices are ordered for counter-clockwise winding when viewed from outside
        match face {
            Face::PosX => [
                Vertex::new([x + 1.0, y, z], normal, color),
                Vertex::new([x + 1.0, y, z + 1.0], normal, color),
                Vertex::new([x + 1.0, y + 1.0, z + 1.0], normal, color),
                Vertex::new([x + 1.0, y + 1.0, z], normal, color),
            ],
            Face::NegX => [
                Vertex::new([x, y, z + 1.0], normal, color),
                Vertex::new([x, y, z], normal, color),
                Vertex::new([x, y + 1.0, z], normal, color),
                Vertex::new([x, y + 1.0, z + 1.0], normal, color),
            ],
            Face::PosY => [
                Vertex::new([x, y + 1.0, z], normal, color),
                Vertex::new([x + 1.0, y + 1.0, z], normal, color),
                Vertex::new([x + 1.0, y + 1.0, z + 1.0], normal, color),
                Vertex::new([x, y + 1.0, z + 1.0], normal, color),
            ],
            Face::NegY => [
                Vertex::new([x, y, z + 1.0], normal, color),
                Vertex::new([x + 1.0, y, z + 1.0], normal, color),
                Vertex::new([x + 1.0, y, z], normal, color),
                Vertex::new([x, y, z], normal, color),
            ],
            Face::PosZ => [
                Vertex::new([x + 1.0, y, z + 1.0], normal, color),
                Vertex::new([x, y, z + 1.0], normal, color),
                Vertex::new([x, y + 1.0, z + 1.0], normal, color),
                Vertex::new([x + 1.0, y + 1.0, z + 1.0], normal, color),
            ],
            Face::NegZ => [
                Vertex::new([x, y, z], normal, color),
                Vertex::new([x + 1.0, y, z], normal, color),
                Vertex::new([x + 1.0, y + 1.0, z], normal, color),
                Vertex::new([x, y + 1.0, z], normal, color),
            ],
        }
    }

    /// Apply simple ambient occlusion darkening based on face direction
    fn apply_face_shading(color: [f32; 4], face: Face) -> [f32; 4] {
        // Simple directional shading
        let shade = match face {
            Face::PosY => 1.0,      // Top - brightest
            Face::PosX | Face::NegZ => 0.85,  // Side faces
            Face::NegX | Face::PosZ => 0.75,  // Other side faces
            Face::NegY => 0.6,      // Bottom - darkest
        };

        [
            color[0] * shade,
            color[1] * shade,
            color[2] * shade,
            color[3],
        ]
    }
}

impl Default for NaiveMesher {
    fn default() -> Self {
        Self::new()
    }
}

impl Mesher for NaiveMesher {
    fn generate(&self, chunk: &Chunk, chunk_pos: ChunkPos) -> ChunkMesh {
        // Early exit for empty chunks
        if chunk.is_empty() {
            return ChunkMesh::new(chunk_pos);
        }

        // Estimate capacity: worst case is 6 faces * 4 vertices per solid voxel
        // But typically only ~1/6 of faces are visible, so we estimate lower
        let estimated_faces = chunk.solid_count() as usize;
        let mut mesh = ChunkMesh::with_capacity(
            chunk_pos,
            estimated_faces * 4,
            estimated_faces * 6,
        );

        // Calculate world offset for this chunk
        let (wx, wy, wz) = chunk_pos.world_origin();

        // Iterate over all voxels
        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let voxel = chunk.get(x, y, z);

                    // Skip air voxels
                    if voxel.is_air() {
                        continue;
                    }

                    let color = voxel.color_f32();

                    // World position of this voxel
                    let world_x = wx as f32 + x as f32;
                    let world_y = wy as f32 + y as f32;
                    let world_z = wz as f32 + z as f32;

                    // Check each face
                    for face in Face::ALL {
                        if Self::is_face_visible(chunk, x as i32, y as i32, z as i32, face) {
                            let shaded_color = Self::apply_face_shading(color, face);
                            let vertices = Self::generate_face_vertices(
                                world_x,
                                world_y,
                                world_z,
                                face,
                                shaded_color,
                            );
                            mesh.add_quad(vertices);
                        }
                    }
                }
            }
        }

        mesh
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_chunk_mesh() {
        let chunk = Chunk::new();
        let mesher = NaiveMesher::new();
        let mesh = mesher.generate(&chunk, ChunkPos::ZERO);

        assert!(mesh.is_empty());
    }

    #[test]
    fn test_single_voxel_mesh() {
        let mut chunk = Chunk::new();
        chunk.set(1, 1, 1, Voxel::from_rgb(255, 0, 0));

        let mesher = NaiveMesher::new();
        let mesh = mesher.generate(&chunk, ChunkPos::ZERO);

        // Single voxel should have 6 visible faces
        assert_eq!(mesh.triangle_count(), 12); // 6 faces * 2 triangles
        assert_eq!(mesh.vertex_count(), 24);   // 6 faces * 4 vertices
    }

    #[test]
    fn test_two_adjacent_voxels() {
        let mut chunk = Chunk::new();
        chunk.set(1, 1, 1, Voxel::from_rgb(255, 0, 0));
        chunk.set(2, 1, 1, Voxel::from_rgb(0, 255, 0));

        let mesher = NaiveMesher::new();
        let mesh = mesher.generate(&chunk, ChunkPos::ZERO);

        // Two adjacent voxels: 12 faces visible (6 each, minus 2 shared faces)
        // But wait - we're not culling between chunks, so internal faces ARE culled
        // Each voxel has 5 visible faces (the shared face is hidden)
        assert_eq!(mesh.triangle_count(), 20); // 10 faces * 2 triangles
    }
}
