//! Greedy meshing: merge adjacent same-color same-direction faces
//! into larger quads, dramatically reducing triangle count.
//!
//! The algorithm follows Mikola Lysenko's classic greedy mesher
//! ([0fps.net 2012-06-30]) generalized to per-voxel RGBA via Eddie
//! Abbondanz's vertex-color extension. For each of the 6 face
//! directions, we sweep slice by slice through the chunk, build a
//! 2D mask of "visible face color at this cell" (or `0` for "no
//! visible face"), and run a 2D greedy rectangle cover on the mask
//! to emit one quad per maximal monochromatic rectangle.
//!
//! Cross-chunk handling matches `NaiveMesher`: rectangles stop at
//! the chunk boundary (no merging across chunks) but face culling
//! consults the 6 neighbor chunks via read-locks. This keeps the
//! per-edit re-mesh scope to a single chunk, and the seam between
//! two coplanar greedy quads from adjacent chunks is geometrically
//! invisible (they meet edge-to-edge at integer cell boundaries).
//!
//! ### Mask key
//!
//! Each mask cell stores the post-shading RGBA packed as `u32`
//! (R/G/B/A in 8 bits each, R in the high byte). `0` is reserved
//! as the "no visible face" sentinel — safe because every editor-
//! placed voxel has α = 255, so a non-air visible face packs to
//! something non-zero. Within a single face-direction pass, the
//! shading factor is constant, so two cells with the same raw
//! color will produce the same shaded mask key and merge.
//!
//! ### Output equivalence with `NaiveMesher`
//!
//! The merged set of (visible-face-position, face-direction,
//! shaded-color) triples is exactly the same as Naive's per-face
//! emission — greedy just packs them into fewer larger quads.
//! `face_quad_vertices_sized` (in `mesh/mod.rs`) drives the winding
//! for both meshers from a single source of truth so merged quads
//! and adjacent unmerged quads from boundary chunks always share
//! consistent vertex orientation.

use parking_lot::RwLockReadGuard;

use super::{
    apply_face_shading, face_quad_vertices_sized, ChunkMesh, Face, Mesher,
};
use crate::core::{Chunk, ChunkPos, World, CHUNK_SIZE, CHUNK_SIZE_I32};

/// Greedy mesher: merges same-color same-direction adjacent faces.
pub struct GreedyMesher;

/// Read-locked face-neighbors in `Face` enum order:
/// `[+X, -X, +Y, -Y, +Z, -Z]`. Index with `face as usize`. (Same
/// shape as `NaiveMesher::NeighborGuards` so the boundary culling
/// path is verbatim.)
type NeighborGuards<'a> = [Option<RwLockReadGuard<'a, Chunk>>; 6];

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

        // Acquire neighbor guards in `Face` enum order so we can
        // index by `face as usize` later. Same pattern as Naive.
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

        // Capacity hint: greedy generally emits far fewer quads than
        // `solid_count`, but allocating up to that cap costs nothing
        // and avoids worst-case re-allocation on jagged scenes.
        let estimated_faces = chunk.solid_count() as usize;
        let mut mesh = ChunkMesh::with_capacity(
            chunk_pos,
            estimated_faces * 4,
            estimated_faces * 6,
        );

        let (wx, wy, wz) = chunk_pos.world_origin();
        for face in Face::ALL {
            mesh_face_direction(&chunk, &neighbors, face, (wx, wy, wz), &mut mesh);
        }
        mesh
    }
}

/// Mesh one face direction across all CHUNK_SIZE slices, emitting
/// merged quads to `mesh`. Stack-allocates a 1024-entry mask which
/// is rebuilt for each slice; allocator traffic stays at zero on
/// the hot path.
fn mesh_face_direction(
    chunk: &Chunk,
    neighbors: &NeighborGuards<'_>,
    face: Face,
    world_origin: (i32, i32, i32),
    mesh: &mut ChunkMesh,
) {
    const SIZE: usize = CHUNK_SIZE;
    // 0 = no face; non-zero = packed shaded RGBA. See module doc.
    let mut mask = [0u32; SIZE * SIZE];

    for d in 0..SIZE {
        // ---- Build the mask for slice `d` ----
        for v_idx in 0..SIZE {
            for u_idx in 0..SIZE {
                let (cx, cy, cz) = cell_for(face, d, u_idx, v_idx);
                let voxel = chunk.get(cx, cy, cz);
                if voxel.is_air() {
                    mask[v_idx * SIZE + u_idx] = 0;
                    continue;
                }
                if !is_face_visible(chunk, neighbors, cx, cy, cz, face) {
                    mask[v_idx * SIZE + u_idx] = 0;
                    continue;
                }
                let shaded = apply_face_shading(voxel.color_f32(), face);
                mask[v_idx * SIZE + u_idx] = pack_rgba(shaded);
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
                while u_idx + w < SIZE
                    && mask[v_idx * SIZE + u_idx + w] == key
                {
                    w += 1;
                }
                // Height: extend along +v while *every* cell in the
                // current row of width `w` matches. The moment any
                // single cell breaks, height stops.
                let mut h = 1;
                'extend_v: while v_idx + h < SIZE {
                    for k in 0..w {
                        if mask[(v_idx + h) * SIZE + u_idx + k] != key {
                            break 'extend_v;
                        }
                    }
                    h += 1;
                }

                // Emit one quad covering the (w × h) rectangle.
                emit_merged_quad(
                    face,
                    d,
                    u_idx,
                    v_idx,
                    w,
                    h,
                    key,
                    world_origin,
                    mesh,
                );

                // Zero out the consumed rectangle so the rest of
                // the scan doesn't re-emit overlapping cells.
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
/// to `mesh`. Defers winding to `face_quad_vertices_sized` so
/// greedy and naive stay in sync.
fn emit_merged_quad(
    face: Face,
    d: usize,
    u: usize,
    v: usize,
    w: usize,
    h: usize,
    packed_color: u32,
    world_origin: (i32, i32, i32),
    mesh: &mut ChunkMesh,
) {
    // Start cell in chunk-local coordinates is exactly the lowest-
    // (u, v) corner of the rectangle within this slice.
    let (cx, cy, cz) = cell_for(face, d, u, v);
    let world_x = world_origin.0 as f32 + cx as f32;
    let world_y = world_origin.1 as f32 + cy as f32;
    let world_z = world_origin.2 as f32 + cz as f32;

    let color = unpack_rgba(packed_color);
    let vertices = face_quad_vertices_sized(
        world_x,
        world_y,
        world_z,
        face,
        w as f32,
        h as f32,
        color,
    );
    mesh.add_quad(vertices);
}

/// Map slice-local `(d, u, v)` indices to chunk-local `(x, y, z)`
/// for a given face direction. The (u, v) ↔ (axis) mapping follows
/// the convention in `face_quad_vertices_sized`:
/// - PosX/NegX: D=X, U=Z, V=Y
/// - PosY/NegY: D=Y, U=X, V=Z
/// - PosZ/NegZ: D=Z, U=X, V=Y
#[inline]
fn cell_for(face: Face, d: usize, u: usize, v: usize) -> (usize, usize, usize) {
    match face {
        Face::PosX | Face::NegX => (d, v, u),
        Face::PosY | Face::NegY => (u, d, v),
        Face::PosZ | Face::NegZ => (u, v, d),
    }
}

/// Pack a `[f32; 4]` shaded RGBA (each component in `[0, 1]`) into
/// a `u32` for use as a merge key. Layout: `RRGGBBAA` (R high). `0`
/// is reserved as the "no visible face" sentinel — safe in practice
/// because editor-placed voxels have α = 255 so any visible face
/// packs to a non-zero value.
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

/// Whether the cell at chunk-local `(x, y, z)` should expose a face
/// in `face` direction. Same logic as `NaiveMesher::is_face_visible`,
/// duplicated here to avoid making that method `pub(crate)` and
/// coupling the two implementations.
fn is_face_visible(
    chunk: &Chunk,
    neighbors: &NeighborGuards<'_>,
    x: usize,
    y: usize,
    z: usize,
    face: Face,
) -> bool {
    let (dx, dy, dz) = face.offset();
    let nx = x as i32 + dx;
    let ny = y as i32 + dy;
    let nz = z as i32 + dz;

    if nx >= 0
        && nx < CHUNK_SIZE_I32
        && ny >= 0
        && ny < CHUNK_SIZE_I32
        && nz >= 0
        && nz < CHUNK_SIZE_I32
    {
        return chunk.get(nx as usize, ny as usize, nz as usize).is_air();
    }

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
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Voxel;

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
        // check (a wrong winding would either drop a face or render
        // it back-side and triangle count would change visibly).
        let mut world = World::new();
        world.set_voxel(1, 1, 1, Voxel::from_rgb(255, 0, 0));
        let mesh = GreedyMesher::new().generate(&world, ChunkPos::ZERO);
        assert_eq!(mesh.triangle_count(), 12);
        assert_eq!(mesh.vertex_count(), 24);
    }

    #[test]
    fn test_two_x_adjacent_merge() {
        // Two voxels along +X. Top, bottom, +Z, -Z faces span 2×1
        // and merge into one quad each. ±X faces stay 1×1 each.
        // Total 6 quads = 12 tris (vs naive's 10 quads / 20 tris).
        let mut world = World::new();
        world.set_voxel(1, 1, 1, Voxel::from_rgb(100, 100, 100));
        world.set_voxel(2, 1, 1, Voxel::from_rgb(100, 100, 100));
        let mesh = GreedyMesher::new().generate(&world, ChunkPos::ZERO);
        assert_eq!(mesh.triangle_count(), 12);
        assert_eq!(mesh.vertex_count(), 24);
    }

    #[test]
    fn test_different_colors_dont_merge() {
        // Same 2-voxel X-adjacent layout but different colors. Top
        // / bottom / ±Z faces can't merge (color differs), so we
        // get 5 quads per voxel × 2 voxels = 10 quads / 20 tris,
        // same as naive.
        let mut world = World::new();
        world.set_voxel(1, 1, 1, Voxel::from_rgb(255, 0, 0));
        world.set_voxel(2, 1, 1, Voxel::from_rgb(0, 255, 0));
        let mesh = GreedyMesher::new().generate(&world, ChunkPos::ZERO);
        assert_eq!(mesh.triangle_count(), 20);
    }

    #[test]
    fn test_2x2x1_slab_merge() {
        // 2×2×1 flat slab (4 voxels) all same color. Top and bottom
        // each merge to a single 2×2 quad. The 4 sides each merge
        // along the axis they extend (2×1 quads). Total 6 quads
        // = 12 tris. Naive emits 16 visible faces = 32 tris.
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
        // 32×32×1 plane filling the entire chunk's bottom layer.
        // Top: 1024 cells → 1 merged quad. Bottom: same. Sides:
        // each side is 32×1 along the chunk edge, but the side
        // faces span CHUNK_SIZE × 1 = 32 cells in the u axis at
        // a single v level, also merging to 1 quad. So total
        // visible faces collapse to 6 quads = 12 tris (same count
        // as a single voxel — that's the dramatic compression).
        let mut world = World::new();
        let c = Voxel::from_rgb(200, 50, 50);
        for x in 0..CHUNK_SIZE_I32 {
            for z in 0..CHUNK_SIZE_I32 {
                world.set_voxel(x, 0, z, c);
            }
        }
        let mesh = GreedyMesher::new().generate(&world, ChunkPos::ZERO);
        assert_eq!(mesh.triangle_count(), 12);
        assert_eq!(mesh.vertex_count(), 24);
    }

    #[test]
    fn test_chessboard_no_merge() {
        // A 2×1×2 checkerboard: voxels at (0,0,0), (1,0,1) — same
        // color, but diagonally placed so no two share a face plane.
        // Each emits 6 quads, no merging possible.
        let mut world = World::new();
        let c = Voxel::from_rgb(100, 100, 100);
        world.set_voxel(0, 0, 0, c);
        world.set_voxel(1, 0, 1, c);
        let mesh = GreedyMesher::new().generate(&world, ChunkPos::ZERO);
        assert_eq!(mesh.triangle_count(), 24); // 12 × 2
    }

    #[test]
    fn test_chunk_boundary_culling() {
        // Two voxels straddling the chunk seam: the ±X faces between
        // them are culled in both meshes. Each chunk emits 5 visible
        // faces × 2 tris = 10 tris, same as naive.
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
        // Voxel at the +X chunk edge, no neighbor chunk loaded:
        // the +X face is rendered (treated as facing air). Same
        // semantics as naive.
        let mut world = World::new();
        world.set_voxel(31, 0, 0, Voxel::from_rgb(255, 0, 0));
        let mesh = GreedyMesher::new().generate(&world, ChunkPos::new(0, 0, 0));
        assert_eq!(mesh.triangle_count(), 12);
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        // Sanity: pack → unpack → pack is stable and color-faithful
        // up to u8 quantization.
        let original = [0.5_f32, 0.25, 0.75, 1.0];
        let packed = pack_rgba(original);
        let recovered = unpack_rgba(packed);
        // Each component within 1/255 of original.
        for i in 0..4 {
            let delta = (original[i] - recovered[i]).abs();
            assert!(delta < 1.0 / 255.0 + 1e-6, "component {} drifted: {} vs {}", i, original[i], recovered[i]);
        }
        assert_eq!(packed, pack_rgba(recovered));
    }

    #[test]
    fn test_pack_air_color_is_zero() {
        // Voxel::AIR has all-zero RGBA; packs to 0 (the sentinel).
        // Verifies the sentinel is consistent — any non-zero packed
        // value is necessarily a non-air voxel face.
        let air_color = Voxel::AIR.color_f32();
        assert_eq!(pack_rgba(air_color), 0);
    }
}
