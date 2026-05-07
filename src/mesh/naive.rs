//! Naive meshing: Generate one quad per visible voxel face.
//!
//! Boundary faces are culled against the six face-neighbor chunks, so
//! adjacent loaded chunks won't produce duplicate faces along their
//! shared planes. If a neighbor chunk isn't loaded the boundary face
//! is rendered (treated as facing air).

use parking_lot::RwLockReadGuard;

use super::{
    apply_face_shading, face_quad_vertices, ChunkMesh, Face, Mesher,
};
use crate::core::{Chunk, ChunkPos, World, CHUNK_SIZE, CHUNK_SIZE_I32};

/// Naive mesher that generates individual quads for each visible face.
pub struct NaiveMesher;

/// Read-locked face-neighbors in `Face` enum order:
/// `[+X, -X, +Y, -Y, +Z, -Z]`. Index with `face as usize`.
type NeighborGuards<'a> = [Option<RwLockReadGuard<'a, Chunk>>; 6];

impl NaiveMesher {
    pub fn new() -> Self {
        Self
    }

    /// Check if a voxel face should be rendered.
    ///
    /// Within the chunk: the neighbor must be air. Across a chunk
    /// boundary: query the appropriate neighbor chunk's edge slab. If
    /// the neighbor chunk isn't loaded we treat the cell as air.
    fn is_face_visible(
        chunk: &Chunk,
        neighbors: &NeighborGuards<'_>,
        x: i32,
        y: i32,
        z: i32,
        face: Face,
    ) -> bool {
        let (dx, dy, dz) = face.offset();
        let nx = x + dx;
        let ny = y + dy;
        let nz = z + dz;

        // Same chunk: direct array access.
        if nx >= 0
            && nx < CHUNK_SIZE_I32
            && ny >= 0
            && ny < CHUNK_SIZE_I32
            && nz >= 0
            && nz < CHUNK_SIZE_I32
        {
            return chunk.get(nx as usize, ny as usize, nz as usize).is_air();
        }

        // Cross-chunk: pick the neighbor and the wrap-around local coord.
        // Only one of dx/dy/dz is nonzero, so the other two stay in [0, CHUNK_SIZE).
        let last = CHUNK_SIZE - 1;
        let (lx, ly, lz) = match face {
            Face::PosX => (0, ny as usize, nz as usize),
            Face::NegX => (last, ny as usize, nz as usize),
            Face::PosY => (nx as usize, 0, nz as usize),
            Face::NegY => (nx as usize, last, nz as usize),
            Face::PosZ => (nx as usize, ny as usize, 0),
            Face::NegZ => (nx as usize, ny as usize, last),
        };

        match &neighbors[face as usize] {
            Some(guard) => guard.get(lx, ly, lz).is_air(),
            None => true, // No loaded neighbor: render the face.
        }
    }

}

impl Default for NaiveMesher {
    fn default() -> Self {
        Self::new()
    }
}

impl Mesher for NaiveMesher {
    fn generate(&self, world: &World, chunk_pos: ChunkPos) -> ChunkMesh {
        let Some(chunk_arc) = world.get_chunk(chunk_pos) else {
            return ChunkMesh::new(chunk_pos);
        };
        let chunk = chunk_arc.read();

        if chunk.is_empty() {
            return ChunkMesh::new(chunk_pos);
        }

        // Acquire read locks on the six face-neighbors. Order matches
        // `Face`: [+X, -X, +Y, -Y, +Z, -Z] so we can index with `face as usize`.
        let neighbor_arcs = [
            world.get_chunk(chunk_pos.neighbor(1, 0, 0)),
            world.get_chunk(chunk_pos.neighbor(-1, 0, 0)),
            world.get_chunk(chunk_pos.neighbor(0, 1, 0)),
            world.get_chunk(chunk_pos.neighbor(0, -1, 0)),
            world.get_chunk(chunk_pos.neighbor(0, 0, 1)),
            world.get_chunk(chunk_pos.neighbor(0, 0, -1)),
        ];
        let neighbors: NeighborGuards<'_> = [
            neighbor_arcs[0].as_ref().map(|a| a.read()),
            neighbor_arcs[1].as_ref().map(|a| a.read()),
            neighbor_arcs[2].as_ref().map(|a| a.read()),
            neighbor_arcs[3].as_ref().map(|a| a.read()),
            neighbor_arcs[4].as_ref().map(|a| a.read()),
            neighbor_arcs[5].as_ref().map(|a| a.read()),
        ];

        // Estimate capacity. Worst case is 6 quads per solid voxel; in
        // practice only the surface contributes, so we lean lower.
        let estimated_faces = chunk.solid_count() as usize;
        let mut mesh = ChunkMesh::with_capacity(
            chunk_pos,
            estimated_faces * 4,
            estimated_faces * 6,
        );

        let (wx, wy, wz) = chunk_pos.world_origin();

        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let voxel = chunk.get(x, y, z);
                    if voxel.is_air() {
                        continue;
                    }

                    let color = voxel.color_f32();
                    let world_x = wx as f32 + x as f32;
                    let world_y = wy as f32 + y as f32;
                    let world_z = wz as f32 + z as f32;

                    for face in Face::ALL {
                        if Self::is_face_visible(
                            &chunk,
                            &neighbors,
                            x as i32,
                            y as i32,
                            z as i32,
                            face,
                        ) {
                            let shaded = apply_face_shading(color, face);
                            let vertices = face_quad_vertices(
                                world_x, world_y, world_z, face, shaded,
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
    use crate::core::Voxel;

    #[test]
    fn test_empty_chunk_mesh() {
        let world = World::new();
        let mesher = NaiveMesher::new();
        let mesh = mesher.generate(&world, ChunkPos::ZERO);

        assert!(mesh.is_empty());
    }

    #[test]
    fn test_single_voxel_mesh() {
        let mut world = World::new();
        world.set_voxel(1, 1, 1, Voxel::from_rgb(255, 0, 0));

        let mesher = NaiveMesher::new();
        let mesh = mesher.generate(&world, ChunkPos::ZERO);

        // Isolated voxel: all 6 faces visible.
        assert_eq!(mesh.triangle_count(), 12);
        assert_eq!(mesh.vertex_count(), 24);
    }

    #[test]
    fn test_two_adjacent_voxels() {
        let mut world = World::new();
        world.set_voxel(1, 1, 1, Voxel::from_rgb(255, 0, 0));
        world.set_voxel(2, 1, 1, Voxel::from_rgb(0, 255, 0));

        let mesher = NaiveMesher::new();
        let mesh = mesher.generate(&world, ChunkPos::ZERO);

        // Each voxel hides one face against the other: 10 faces total.
        assert_eq!(mesh.triangle_count(), 20);
    }

    #[test]
    fn test_chunk_boundary_culling() {
        // Two voxels straddling the chunk (0,0,0) / (1,0,0) boundary.
        // The +X face of (31, 0, 0) and the -X face of (32, 0, 0)
        // should both be culled.
        let mut world = World::new();
        world.set_voxel(31, 0, 0, Voxel::from_rgb(255, 0, 0));
        world.set_voxel(32, 0, 0, Voxel::from_rgb(0, 255, 0));

        let mesher = NaiveMesher::new();
        let mesh_a = mesher.generate(&world, ChunkPos::new(0, 0, 0));
        let mesh_b = mesher.generate(&world, ChunkPos::new(1, 0, 0));

        // Each voxel shows 5 faces -> 10 triangles per chunk.
        assert_eq!(mesh_a.triangle_count(), 10);
        assert_eq!(mesh_b.triangle_count(), 10);
    }

    #[test]
    fn test_chunk_boundary_no_neighbor() {
        // A boundary voxel with no neighbor chunk loaded: the boundary
        // face is rendered (treated as facing air).
        let mut world = World::new();
        world.set_voxel(31, 0, 0, Voxel::from_rgb(255, 0, 0));

        let mesher = NaiveMesher::new();
        let mesh = mesher.generate(&world, ChunkPos::new(0, 0, 0));

        // All 6 faces visible.
        assert_eq!(mesh.triangle_count(), 12);
    }
}
