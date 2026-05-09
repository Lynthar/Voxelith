//! Greedy meshing: merge adjacent same-color same-AO same-direction
//! faces into larger quads, dramatically reducing triangle count.
//!
//! The algorithm follows Mikola Lysenko's classic greedy mesher
//! ([0fps.net 2012-06-30]) generalized to per-voxel RGBA via Eddie
//! Abbondanz's vertex-color extension, plus per-vertex AO via the
//! 0fps AO + greedy combination ([0fps.net 2013-07-03]).
//!
//! For each of the 6 face directions, we sweep slice by slice through
//! the chunk, build a 2D mask of `(packed_color, packed_ao)` keys
//! (or `0` for "no visible face"), and run a 2D greedy rectangle
//! cover on the mask to emit one quad per maximal monochromatic
//! rectangle that also has uniform 4-corner AO.
//!
//! Cross-chunk handling matches `NaiveMesher`: rectangles stop at
//! the chunk boundary (no merging across chunks) but face culling
//! and AO sampling consult the 26 neighbor chunks via read-locks.
//!
//! ### Mask key
//!
//! Each mask cell stores `(packed_rgba << 8) | packed_ao` as `u64`:
//! - `packed_rgba` (top 32 bits, with 24 bits padding above): the
//!   shaded RGBA color, packed via `pack_rgba`
//! - `packed_ao` (bottom 8 bits): 4 corner AO values, 2 bits each,
//!   packed via `mesh::ao::pack_ao`
//!
//! `0` is reserved as the "no visible face" sentinel — safe because
//! every editor-placed voxel has α = 255, so a non-air visible face
//! always packs to a non-zero `packed_rgba`. Two cells merge only
//! when the entire `u64` matches, including all 4 corner AO values
//! — without this, the merged quad's bilinear-interpolated AO would
//! disagree with per-cell AO at internal edges.

use super::ao::pack_ao;
use super::neighbors::{
    lock_neighbors, neighbor_arcs, voxel_at_local, NeighborArcs, NeighborGuards,
};
use super::{
    ao_to_f32, apply_face_shading, compute_face_ao, face_quad_vertices_sized_ao,
    unpack_ao, ChunkMesh, Face, Mesher,
};
use crate::core::{Chunk, ChunkPos, World, CHUNK_SIZE};

/// Greedy mesher: merges same-color same-AO same-direction adjacent faces.
pub struct GreedyMesher;

impl GreedyMesher {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GreedyMesher {
    fn default() -> Self {
        Self::new()
    }
}

impl Mesher for GreedyMesher {
    fn generate(&self, world: &World, chunk_pos: ChunkPos) -> ChunkMesh {
        let Some(chunk_arc) = world.get_chunk(chunk_pos) else {
            return ChunkMesh::new(chunk_pos);
        };
        let chunk = chunk_arc.read();
        if chunk.is_empty() {
            return ChunkMesh::new(chunk_pos);
        }

        // Lock all 26 neighbors. Face culling needs only 6, but AO
        // sampling at chunk corners can need diagonal neighbors.
        let arcs: NeighborArcs = neighbor_arcs(world, chunk_pos);
        let neighbors: NeighborGuards = lock_neighbors(&arcs);

        // Capacity hint: greedy generally emits far fewer quads than
        // `solid_count`, but allocating up to that cap costs nothing
        // and avoids worst-case re-allocation on jagged scenes.
        let estimated_faces = chunk.solid_count() as usize;
        let mut mesh = ChunkMesh::with_capacity(
            chunk_pos,
            estimated_faces * 4,
            estimated_faces * 6,
        );

        let world_origin = chunk_pos.world_origin();
        for face in Face::ALL {
            mesh_face_direction(&chunk, &neighbors, face, world_origin, &mut mesh);
        }
        mesh
    }
}

/// Mesh one face direction across all CHUNK_SIZE slices, emitting
/// merged quads to `mesh`. Stack-allocates a 1024-entry `u64` mask
/// which is rebuilt for each slice; allocator traffic stays at zero
/// on the hot path.
fn mesh_face_direction(
    chunk: &Chunk,
    neighbors: &NeighborGuards,
    face: Face,
    world_origin: (i32, i32, i32),
    mesh: &mut ChunkMesh,
) {
    const SIZE: usize = CHUNK_SIZE;
    // 0 = no face; non-zero = (packed_rgba << 8) | packed_ao.
    let mut mask = [0u64; SIZE * SIZE];

    for d in 0..SIZE {
        // ---- Build the mask for slice `d` ----
        for v_idx in 0..SIZE {
            for u_idx in 0..SIZE {
                let (cx, cy, cz) = cell_for(face, d, u_idx, v_idx);
                let voxel = chunk.get(cx, cy, cz);
                if voxel.is_air()
                    || !is_face_visible(chunk, neighbors, cx, cy, cz, face)
                {
                    mask[v_idx * SIZE + u_idx] = 0;
                    continue;
                }
                let shaded = apply_face_shading(voxel.color_f32(), face);
                let packed_color = pack_rgba(shaded);
                // 4-corner AO via 12 voxel samples through the
                // 26-neighbor lock array.
                let world_x = world_origin.0 + cx as i32;
                let world_y = world_origin.1 + cy as i32;
                let world_z = world_origin.2 + cz as i32;
                let ao_int = compute_face_ao(
                    (world_x, world_y, world_z),
                    face,
                    |p| {
                        let lx = p.0 - world_origin.0;
                        let ly = p.1 - world_origin.1;
                        let lz = p.2 - world_origin.2;
                        voxel_at_local(chunk, neighbors, lx, ly, lz).is_solid()
                    },
                );
                let packed_ao = pack_ao(ao_int);
                mask[v_idx * SIZE + u_idx] =
                    (packed_color as u64) << 8 | packed_ao as u64;
            }
        }

        // ---- Greedy rectangle cover on the mask ----
        let mut v_idx = 0;
        while v_idx < SIZE {
            let mut u_idx = 0;
            while u_idx < SIZE {
                let key = mask[v_idx * SIZE + u_idx];
                if key == 0 {
                    u_idx += 1;
                    continue;
                }
                // Width: extend along +u while key matches.
                let mut w = 1;
                while u_idx + w < SIZE && mask[v_idx * SIZE + u_idx + w] == key {
                    w += 1;
                }
                // Height: extend along +v while *every* cell in the
                // current row of width `w` matches.
                let mut h = 1;
                'extend_v: while v_idx + h < SIZE {
                    for k in 0..w {
                        if mask[(v_idx + h) * SIZE + u_idx + k] != key {
                            break 'extend_v;
                        }
                    }
                    h += 1;
                }

                emit_merged_quad(face, d, u_idx, v_idx, w, h, key, world_origin, mesh);

                // Zero out the consumed rectangle.
                for dh in 0..h {
                    for dw in 0..w {
                        mask[(v_idx + dh) * SIZE + u_idx + dw] = 0;
                    }
                }
                u_idx += w;
            }
            v_idx += 1;
        }
    }
}

/// Compute the 4 vertices of a `(w × h)` quad at slice `d`, mask
/// origin `(u, v)`, world-origin offset `world_origin`, and emit
/// to `mesh`. The mask key contains both color and 4 corner AO —
/// they apply uniformly across the merged rectangle (cells with
/// different AO can't merge).
fn emit_merged_quad(
    face: Face,
    d: usize,
    u: usize,
    v: usize,
    w: usize,
    h: usize,
    packed_key: u64,
    world_origin: (i32, i32, i32),
    mesh: &mut ChunkMesh,
) {
    let (cx, cy, cz) = cell_for(face, d, u, v);
    let world_x = world_origin.0 as f32 + cx as f32;
    let world_y = world_origin.1 as f32 + cy as f32;
    let world_z = world_origin.2 as f32 + cz as f32;

    let packed_color = (packed_key >> 8) as u32;
    let packed_ao = (packed_key & 0xFF) as u8;
    let color = unpack_rgba(packed_color);
    let ao_int = unpack_ao(packed_ao);
    let ao = [
        ao_to_f32(ao_int[0]),
        ao_to_f32(ao_int[1]),
        ao_to_f32(ao_int[2]),
        ao_to_f32(ao_int[3]),
    ];
    let vertices = face_quad_vertices_sized_ao(
        world_x,
        world_y,
        world_z,
        face,
        w as f32,
        h as f32,
        color,
        ao,
    );
    mesh.add_quad_with_ao_flip(vertices);
}

/// Map slice-local `(d, u, v)` indices to chunk-local `(x, y, z)`
/// for a given face direction. Convention matches
/// `face_quad_vertices_sized` in `mesh/mod.rs`.
#[inline]
fn cell_for(face: Face, d: usize, u: usize, v: usize) -> (usize, usize, usize) {
    match face {
        Face::PosX | Face::NegX => (d, v, u),
        Face::PosY | Face::NegY => (u, d, v),
        Face::PosZ | Face::NegZ => (u, v, d),
    }
}

/// Pack a `[f32; 4]` shaded RGBA into a `u32` mask key. Layout:
/// `RRGGBBAA` (R high). `0` is the "no visible face" sentinel.
#[inline]
fn pack_rgba(c: [f32; 4]) -> u32 {
    let r = (c[0].clamp(0.0, 1.0) * 255.0).round() as u32;
    let g = (c[1].clamp(0.0, 1.0) * 255.0).round() as u32;
    let b = (c[2].clamp(0.0, 1.0) * 255.0).round() as u32;
    let a = (c[3].clamp(0.0, 1.0) * 255.0).round() as u32;
    (r << 24) | (g << 16) | (b << 8) | a
}

#[inline]
fn unpack_rgba(p: u32) -> [f32; 4] {
    let r = ((p >> 24) & 0xFF) as f32 / 255.0;
    let g = ((p >> 16) & 0xFF) as f32 / 255.0;
    let b = ((p >> 8) & 0xFF) as f32 / 255.0;
    let a = (p & 0xFF) as f32 / 255.0;
    [r, g, b, a]
}

/// Whether the cell at chunk-local `(x, y, z)` exposes a face in
/// `face` direction. Routes through `voxel_at_local` (26-neighbor
/// lock) to handle chunk boundaries uniformly with AO sampling.
fn is_face_visible(
    chunk: &Chunk,
    neighbors: &NeighborGuards,
    x: usize,
    y: usize,
    z: usize,
    face: Face,
) -> bool {
    let (dx, dy, dz) = face.offset();
    voxel_at_local(
        chunk,
        neighbors,
        x as i32 + dx,
        y as i32 + dy,
        z as i32 + dz,
    )
    .is_air()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Voxel, CHUNK_SIZE_I32};

    #[test]
    fn test_empty_chunk_mesh() {
        let world = World::new();
        let mesh = GreedyMesher::new().generate(&world, ChunkPos::ZERO);
        assert!(mesh.is_empty());
    }

    #[test]
    fn test_single_voxel_emits_six_quads() {
        // Isolated voxel: 6 visible faces, none mergeable. Greedy
        // and naive are identical here — useful as a winding sanity
        // check.
        let mut world = World::new();
        world.set_voxel(1, 1, 1, Voxel::from_rgb(255, 0, 0));
        let mesh = GreedyMesher::new().generate(&world, ChunkPos::ZERO);
        assert_eq!(mesh.triangle_count(), 12);
        assert_eq!(mesh.vertex_count(), 24);
    }

    #[test]
    fn test_two_x_adjacent_merge() {
        // Two voxels along +X. Top, bottom, +Z, -Z faces span 2×1
        // and merge into one quad each (their AO is uniform — both
        // cells have all-air neighbors so all corners = 3). ±X
        // faces stay 1×1 each. Total 6 quads = 12 tris.
        let mut world = World::new();
        world.set_voxel(1, 1, 1, Voxel::from_rgb(100, 100, 100));
        world.set_voxel(2, 1, 1, Voxel::from_rgb(100, 100, 100));
        let mesh = GreedyMesher::new().generate(&world, ChunkPos::ZERO);
        assert_eq!(mesh.triangle_count(), 12);
        assert_eq!(mesh.vertex_count(), 24);
    }

    #[test]
    fn test_different_colors_dont_merge() {
        let mut world = World::new();
        world.set_voxel(1, 1, 1, Voxel::from_rgb(255, 0, 0));
        world.set_voxel(2, 1, 1, Voxel::from_rgb(0, 255, 0));
        let mesh = GreedyMesher::new().generate(&world, ChunkPos::ZERO);
        assert_eq!(mesh.triangle_count(), 20);
    }

    #[test]
    fn test_2x2x1_slab_merge() {
        let mut world = World::new();
        let c = Voxel::from_rgb(50, 100, 150);
        for x in 0..2 {
            for z in 0..2 {
                world.set_voxel(x, 0, z, c);
            }
        }
        let mesh = GreedyMesher::new().generate(&world, ChunkPos::ZERO);
        assert_eq!(mesh.triangle_count(), 12);
    }

    #[test]
    fn test_full_chunk_layer_merges_to_single_quad_per_visible_face() {
        // 32×32×1 plane. Top: AO uniform → 1 merged quad. Bottom:
        // AO uniform → 1 merged quad. Sides: each side is 32×1 but
        // the corner cells' AO differs from interior cells along
        // that side (they have neighbors along the run direction
        // that interior cells don't), so AO segmentation breaks
        // each side into a small number of quads. Triangle count
        // is **higher than the pre-AO version** as a result — that's
        // the cost of correct AO.
        let mut world = World::new();
        let c = Voxel::from_rgb(200, 50, 50);
        for x in 0..CHUNK_SIZE_I32 {
            for z in 0..CHUNK_SIZE_I32 {
                world.set_voxel(x, 0, z, c);
            }
        }
        let mesh = GreedyMesher::new().generate(&world, ChunkPos::ZERO);
        // Top + bottom are single quads each (AO uniform: all air
        // above/below). Sides may segment due to AO at the corners
        // — verify we're at least better than naive (which would
        // emit 32 quads per side × 4 sides = 128 quads + 2 = 130
        // quads for top/bottom = 132 quads = 264 tris).
        let tri_count = mesh.triangle_count();
        assert!(tri_count >= 12, "expected at least 12 tris (top+bottom)");
        assert!(
            tri_count < 264,
            "greedy with AO should beat naive, got {} tris",
            tri_count
        );
    }

    #[test]
    fn test_chessboard_no_merge() {
        let mut world = World::new();
        let c = Voxel::from_rgb(100, 100, 100);
        world.set_voxel(0, 0, 0, c);
        world.set_voxel(1, 0, 1, c);
        let mesh = GreedyMesher::new().generate(&world, ChunkPos::ZERO);
        assert_eq!(mesh.triangle_count(), 24);
    }

    #[test]
    fn test_chunk_boundary_culling() {
        let mut world = World::new();
        world.set_voxel(31, 0, 0, Voxel::from_rgb(255, 0, 0));
        world.set_voxel(32, 0, 0, Voxel::from_rgb(0, 255, 0));
        let mesher = GreedyMesher::new();
        let mesh_a = mesher.generate(&world, ChunkPos::new(0, 0, 0));
        let mesh_b = mesher.generate(&world, ChunkPos::new(1, 0, 0));
        assert_eq!(mesh_a.triangle_count(), 10);
        assert_eq!(mesh_b.triangle_count(), 10);
    }

    #[test]
    fn test_chunk_boundary_no_neighbor_renders_face() {
        let mut world = World::new();
        world.set_voxel(31, 0, 0, Voxel::from_rgb(255, 0, 0));
        let mesh = GreedyMesher::new().generate(&world, ChunkPos::new(0, 0, 0));
        assert_eq!(mesh.triangle_count(), 12);
    }

    #[test]
    fn test_pack_unpack_rgba_roundtrip() {
        let original = [0.5_f32, 0.25, 0.75, 1.0];
        let packed = pack_rgba(original);
        let recovered = unpack_rgba(packed);
        for i in 0..4 {
            let delta = (original[i] - recovered[i]).abs();
            assert!(delta < 1.0 / 255.0 + 1e-6);
        }
        assert_eq!(packed, pack_rgba(recovered));
    }

    #[test]
    fn test_pack_air_color_is_zero() {
        let air_color = Voxel::AIR.color_f32();
        assert_eq!(pack_rgba(air_color), 0);
    }

    #[test]
    fn test_isolated_voxel_has_full_ao() {
        let mut world = World::new();
        world.set_voxel(1, 1, 1, Voxel::from_rgb(255, 0, 0));
        let mesh = GreedyMesher::new().generate(&world, ChunkPos::ZERO);
        for v in &mesh.vertices {
            assert_eq!(v.ao, 1.0, "expected full AO for isolated voxel");
        }
    }
}
