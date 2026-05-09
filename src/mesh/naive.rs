//! Naive meshing: Generate one quad per visible voxel face.
//!
//! Boundary faces are culled against the six face-neighbor chunks, so
//! adjacent loaded chunks won't produce duplicate faces along their
//! shared planes. If a neighbor chunk isn't loaded the boundary face
//! is rendered (treated as facing air).
//!
//! Per-vertex AO is computed for each emitted quad — the 4 corners
//! sample 3 cells each in the face's outside layer (12 samples per
//! face) via `mesh::neighbors::voxel_at_local`, which routes through
//! the 26-neighbor lock array. AO 0–3 maps to a brightness factor in
//! the fragment shader.

use super::neighbors::{
    lock_neighbors, neighbor_arcs, voxel_at_local, NeighborArcs, NeighborGuards,
};
use super::{
    ao_to_f32, apply_face_shading, compute_face_ao, face_quad_vertices_sized_ao,
    ChunkMesh, Face, Mesher,
};
use crate::core::{Chunk, ChunkPos, World, CHUNK_SIZE};

/// Naive mesher that generates individual quads for each visible face.
pub struct NaiveMesher;

impl NaiveMesher {
    pub fn new() -> Self {
        Self
    }

    /// Whether the cell at chunk-local `(x, y, z)` exposes a face in
    /// the given direction. Routes the neighbor lookup through
    /// `voxel_at_local` so face-edge and corner-edge cells use the
    /// same 26-neighbor lock array as AO sampling.
    fn is_face_visible(
        chunk: &Chunk,
        neighbors: &NeighborGuards,
        x: i32,
        y: i32,
        z: i32,
        face: Face,
    ) -> bool {
        let (dx, dy, dz) = face.offset();
        voxel_at_local(chunk, neighbors, x + dx, y + dy, z + dz).is_air()
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

        // Acquire `Arc`s + read locks for all 26 neighbors. Face
        // culling needs 6; AO sampling at chunk corners can need
        // up to 3-axis-diagonal neighbors. Missing neighbors → None.
        let arcs: NeighborArcs = neighbor_arcs(world, chunk_pos);
        let neighbors: NeighborGuards = lock_neighbors(&arcs);

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
                    let world_x = wx + x as i32;
                    let world_y = wy + y as i32;
                    let world_z = wz + z as i32;

                    for face in Face::ALL {
                        if !Self::is_face_visible(
                            &chunk,
                            &neighbors,
                            x as i32,
                            y as i32,
                            z as i32,
                            face,
                        ) {
                            continue;
                        }
                        let shaded = apply_face_shading(color, face);
                        // 4-corner AO via 12 voxel samples through
                        // the 26-neighbor lock array.
                        let ao_int = compute_face_ao(
                            (world_x, world_y, world_z),
                            face,
                            |p| {
                                let lx = p.0 - wx;
                                let ly = p.1 - wy;
                                let lz = p.2 - wz;
                                voxel_at_local(&chunk, &neighbors, lx, ly, lz)
                                    .is_solid()
                            },
                        );
                        let ao = [
                            ao_to_f32(ao_int[0]),
                            ao_to_f32(ao_int[1]),
                            ao_to_f32(ao_int[2]),
                            ao_to_f32(ao_int[3]),
                        ];
                        let vertices = face_quad_vertices_sized_ao(
                            world_x as f32,
                            world_y as f32,
                            world_z as f32,
                            face,
                            1.0,
                            1.0,
                            shaded,
                            ao,
                        );
                        mesh.add_quad_with_ao_flip(vertices);
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
    fn test_isolated_voxel_has_full_ao() {
        // Isolated voxel: no neighbors → every vertex AO = 1.0.
        let mut world = World::new();
        world.set_voxel(1, 1, 1, Voxel::from_rgb(255, 0, 0));

        let mesh = NaiveMesher::new().generate(&world, ChunkPos::ZERO);
        for v in &mesh.vertices {
            assert_eq!(v.ao, 1.0, "expected full AO for isolated voxel");
        }
    }

    #[test]
    fn test_corner_neighbor_darkens_face_corner() {
        // Two voxels:
        //   (0, 0, 0) — the cell whose PosY face we'll inspect
        //   (1, 1, 0) — solid neighbor at side1 of vertices 1 & 2
        // For PosY of (0, 0, 0):
        //   vertex 0 (du=-1, dv=-1) → AO = 3 (no occlusion)
        //   vertex 1 (du=+1, dv=-1) → AO = 2 (side1 occluded)
        //   vertex 2 (du=+1, dv=+1) → AO = 2 (side1 occluded)
        //   vertex 3 (du=-1, dv=+1) → AO = 3
        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgb(255, 0, 0));
        world.set_voxel(1, 1, 0, Voxel::from_rgb(0, 255, 0));
        world.clear_dirty_flags();

        let mesh = NaiveMesher::new().generate(&world, ChunkPos::ZERO);

        // Find PosY-face vertices on (0, 0, 0): they sit at y = 1.
        let posy_verts: Vec<_> = mesh
            .vertices
            .iter()
            .filter(|v| {
                v.normal == [0.0, 1.0, 0.0]
                    && v.position[0] >= 0.0
                    && v.position[0] <= 1.0
                    && v.position[2] >= 0.0
                    && v.position[2] <= 1.0
                    && (v.position[1] - 1.0).abs() < 1e-6
            })
            .collect();
        assert_eq!(posy_verts.len(), 4);

        // Verify the AO pattern: 2 vertices at AO 2/3, 2 at AO 3/3.
        let ao_count_full = posy_verts.iter().filter(|v| v.ao == 1.0).count();
        let ao_count_partial = posy_verts
            .iter()
            .filter(|v| (v.ao - 2.0 / 3.0).abs() < 1e-6)
            .count();
        assert_eq!(ao_count_full, 2);
        assert_eq!(ao_count_partial, 2);
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
