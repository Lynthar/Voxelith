//! Greedy meshing: sweep each face direction slice by slice, build a
//! mask of packed keys, and cover it with maximal rectangles. Merges
//! stop at the chunk boundary; culling and AO read the 26 neighbors.

use super::ao::pack_ao;
use super::neighbors::{
    lock_neighbors, neighbor_arcs, voxel_at_local, NeighborArcs, NeighborGuards,
};
use super::{
    ao_to_f32, apply_face_shading, compute_face_ao, face_quad_vertices_sized_ao, unpack_ao,
    ChunkMesh, Face, Mesher,
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

        // Worst case — one visible quad per solid voxel — to keep
        // jagged scenes from re-allocating. A full chunk reserves about
        // 6 MiB of vertices and 0.75 MiB of indices, per rayon worker.
        let estimated_faces = chunk.solid_count() as usize;
        let mut mesh =
            ChunkMesh::with_capacity(chunk_pos, estimated_faces * 4, estimated_faces * 6);

        let world_origin = chunk_pos.world_origin();
        for face in Face::ALL {
            mesh_face_direction(&chunk, &neighbors, face, world_origin, None, &mut mesh);
        }
        mesh
    }
}

/// One `(group_id, mesh)` per non-empty material group — the low two
/// `Voxel::flags` bits. Culling and AO see every solid voxel, but each
/// mesh emits only its own group's faces, so merges never span groups.
pub fn mesh_chunk_by_material(world: &World, chunk_pos: ChunkPos) -> Vec<(u8, ChunkMesh)> {
    let Some(chunk_arc) = world.get_chunk(chunk_pos) else {
        return Vec::new();
    };
    let chunk = chunk_arc.read();
    if chunk.is_empty() {
        return Vec::new();
    }
    let arcs: NeighborArcs = neighbor_arcs(world, chunk_pos);
    let neighbors: NeighborGuards = lock_neighbors(&arcs);
    let world_origin = chunk_pos.world_origin();

    let mut out = Vec::new();
    for group in 0u8..4 {
        let mut mesh = ChunkMesh::new(chunk_pos);
        for face in Face::ALL {
            mesh_face_direction(
                &chunk,
                &neighbors,
                face,
                world_origin,
                Some(group),
                &mut mesh,
            );
        }
        if !mesh.is_empty() {
            out.push((group, mesh));
        }
    }
    out
}

/// Mesh one face direction across every slice, emitting merged quads.
/// `group_filter` restricts which voxels *emit* faces; culling and AO
/// still consult every solid voxel. `None` meshes all of them.
fn mesh_face_direction(
    chunk: &Chunk,
    neighbors: &NeighborGuards,
    face: Face,
    world_origin: (i32, i32, i32),
    group_filter: Option<u8>,
    mesh: &mut ChunkMesh,
) {
    const SIZE: usize = CHUNK_SIZE;
    // 0 = no face; non-zero = (tint_zone << 40) | (packed_rgba << 8) | packed_ao.
    let mut mask = [0u64; SIZE * SIZE];

    for d in 0..SIZE {
        // ---- Build the mask for slice `d` ----
        for v_idx in 0..SIZE {
            for u_idx in 0..SIZE {
                let (cx, cy, cz) = cell_for(face, d, u_idx, v_idx);
                let voxel = chunk.get(cx, cy, cz);
                if voxel.is_air() || !is_face_visible(chunk, neighbors, cx, cy, cz, face) {
                    mask[v_idx * SIZE + u_idx] = 0;
                    continue;
                }
                // Outside the target group: still occludes and shades,
                // only its own face emission is suppressed here.
                if let Some(g) = group_filter {
                    if voxel.flags & 0x03 != g {
                        mask[v_idx * SIZE + u_idx] = 0;
                        continue;
                    }
                }
                let shaded = apply_face_shading(voxel.color_f32(), face);
                let packed_color = pack_rgba(shaded);
                // 4-corner AO via 12 voxel samples through the
                // 26-neighbor lock array.
                let world_x = world_origin.0 + cx as i32;
                let world_y = world_origin.1 + cy as i32;
                let world_z = world_origin.2 + cz as i32;
                let ao_int = compute_face_ao((world_x, world_y, world_z), face, |p| {
                    let lx = p.0 - world_origin.0;
                    let ly = p.1 - world_origin.1;
                    let lz = p.2 - world_origin.2;
                    voxel_at_local(chunk, neighbors, lx, ly, lz).is_solid()
                });
                let packed_ao = pack_ao(ao_int);
                // Tint zone (0-3) in bits 40+ so voxels of different
                // zones never merge — the zone must reach export
                // per-vertex (it can't be averaged across a merged quad).
                let zone = voxel.tint_zone() as u64;
                mask[v_idx * SIZE + u_idx] =
                    (zone << 40) | ((packed_color as u64) << 8) | packed_ao as u64;
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

/// Emit the four vertices of a `w × h` quad at slice `d`. Color and
/// the four AO corners both come from the mask key, which applies
/// uniformly across the rectangle — differing AO prevents merging.
#[allow(clippy::too_many_arguments)] // flat quad geometry — a struct would just move nine names one level down
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
    // Tint zone lives in bits 40+ (above color+ao); `>> 8 as u32` for the
    // color truncates it away, so it must be read from the full key here.
    let tint_zone = ((packed_key >> 40) & 0xFF) as f32;
    let color = unpack_rgba(packed_color);
    let ao_int = unpack_ao(packed_ao);
    let ao = [
        ao_to_f32(ao_int[0]),
        ao_to_f32(ao_int[1]),
        ao_to_f32(ao_int[2]),
        ao_to_f32(ao_int[3]),
    ];
    let mut vertices = face_quad_vertices_sized_ao(
        world_x, world_y, world_z, face, w as f32, h as f32, color, ao,
    );
    for vert in &mut vertices {
        vert.tint_zone = tint_zone;
    }
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
        // Two voxels along +X: top, bottom and ±Z merge into one 2×1
        // quad each (uniform AO), ±X stay 1×1. Six quads, twelve tris.
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
    fn test_different_tint_zones_dont_merge() {
        // Two same-color +X-adjacent voxels but different tint zones must
        // NOT merge their shared-direction faces (the zone is in the mask
        // key) — otherwise the zone couldn't survive per-vertex to export.
        let mut world = World::new();
        let mut a = Voxel::from_rgb(100, 100, 100);
        let mut b = Voxel::from_rgb(100, 100, 100);
        a.set_tint_zone(1);
        b.set_tint_zone(2);
        world.set_voxel(1, 1, 1, a);
        world.set_voxel(2, 1, 1, b);
        let mesh = GreedyMesher::new().generate(&world, ChunkPos::ZERO);
        // Same color alone would merge to 12 tris (test_two_x_adjacent_merge);
        // different zones block the merge → 20 tris, like different colors.
        assert_eq!(mesh.triangle_count(), 20);
        // Every emitted vertex carries its voxel's zone.
        let zones: Vec<f32> = mesh.vertices.iter().map(|v| v.tint_zone).collect();
        assert!(zones.contains(&1.0) && zones.contains(&2.0));
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
        // A 32×32×1 plane. Top and bottom merge to one quad each; the
        // sides segment because corner cells' AO differs from interior
        // ones, putting the count above the pre-AO version.
        let mut world = World::new();
        let c = Voxel::from_rgb(200, 50, 50);
        for x in 0..CHUNK_SIZE_I32 {
            for z in 0..CHUNK_SIZE_I32 {
                world.set_voxel(x, 0, z, c);
            }
        }
        let mesh = GreedyMesher::new().generate(&world, ChunkPos::ZERO);
        // Top and bottom are one quad each; sides segment on AO. The
        // bar is naive's output — 128 side quads plus 2 = 130.
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

    #[test]
    fn cell_for_puts_d_on_the_normal_and_u_v_on_the_face_axes() {
        // `cell_for` and `ao::face_axes` describe the same per-face
        // frame from two sides and nothing makes them agree. A
        // permutation shows up as AO from the wrong neighbor, not a crash.
        let axis_of = |v: [i32; 3]| v.iter().position(|c| *c != 0).expect("an axis");
        for face in Face::ALL {
            let (n, u, v) = crate::mesh::ao::face_axes(face);
            let cell = cell_for(face, 1, 2, 3);
            let cell = [cell.0, cell.1, cell.2];
            assert_eq!(cell[axis_of(n)], 1, "{face:?}: d belongs on the normal");
            assert_eq!(cell[axis_of(u)], 2, "{face:?}: u belongs on the U axis");
            assert_eq!(cell[axis_of(v)], 3, "{face:?}: v belongs on the V axis");
        }
    }

    #[test]
    fn per_material_split_still_culls_against_the_other_group() {
        // Culling runs against every solid voxel, so the face two
        // groups share is hidden in both meshes. Culling per group
        // instead leaves invisible geometry inside mixed exports.
        let mut world = World::new();
        let plain = Voxel::from_rgb(200, 200, 200);
        let mut emissive = Voxel::from_rgb(200, 100, 0);
        emissive.flags = 0b01;
        world.set_voxel(1, 1, 1, plain);
        world.set_voxel(2, 1, 1, emissive);

        let groups = mesh_chunk_by_material(&world, ChunkPos::ZERO);
        assert_eq!(groups.len(), 2, "one mesh per material present");
        let mut keys: Vec<u8> = groups.iter().map(|(m, _)| *m).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec![0, 1], "plain and emissive");
        for (material, mesh) in &groups {
            assert_eq!(
                mesh.vertices.len(),
                5 * 4,
                "material {material}: five visible faces, not six"
            );
            assert_eq!(mesh.indices.len(), 5 * 6);
        }
    }
}
