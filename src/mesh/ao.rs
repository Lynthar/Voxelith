//! Per-vertex ambient occlusion. Each face vertex samples three cells
//! in the 2×2 neighborhood one step along the face normal and yields a
//! 0–3 occlusion value.

use super::Face;

/// AO for one vertex: 0 fully occluded, 3 unoccluded. Two solid sides
/// short-circuit to 0 — the corner is shadowed whatever `corner` holds.
#[inline]
pub fn vertex_ao(side1: bool, side2: bool, corner: bool) -> u8 {
    if side1 && side2 {
        return 0;
    }
    3 - (side1 as u8 + side2 as u8 + corner as u8)
}

/// A 0–3 AO integer as an `f32` factor in `[0, 1]`. The fragment shader
/// remaps it to `ambient_min + (1 - ambient_min) * ao`.
#[inline]
pub fn ao_to_f32(ao: u8) -> f32 {
    ao as f32 / 3.0
}

/// Per-face `(N, U, V)`: outward normal plus the two axes spanning the
/// face plane. U×V need not equal N — orientation lives in
/// [`face_vertex_signs`].
#[inline]
pub fn face_axes(face: Face) -> ([i32; 3], [i32; 3], [i32; 3]) {
    match face {
        Face::PosX => ([1, 0, 0], [0, 0, 1], [0, 1, 0]),
        Face::NegX => ([-1, 0, 0], [0, 0, 1], [0, 1, 0]),
        Face::PosY => ([0, 1, 0], [1, 0, 0], [0, 0, 1]),
        Face::NegY => ([0, -1, 0], [1, 0, 0], [0, 0, 1]),
        Face::PosZ => ([0, 0, 1], [1, 0, 0], [0, 1, 0]),
        Face::NegZ => ([0, 0, -1], [1, 0, 0], [0, 1, 0]),
    }
}

/// `(du, dv) ∈ {-1, +1}²` per vertex, in `face_quad_vertices_sized`
/// order. Vertices 0/2 sit on the main diagonal and 1/3 on the
/// anti-diagonal — the pairs the diagonal flip picks between.
#[inline]
pub fn face_vertex_signs(face: Face) -> [(i32, i32); 4] {
    match face {
        Face::PosX => [(-1, -1), (1, -1), (1, 1), (-1, 1)],
        Face::NegX => [(1, -1), (-1, -1), (-1, 1), (1, 1)],
        Face::PosY => [(-1, -1), (1, -1), (1, 1), (-1, 1)],
        Face::NegY => [(-1, 1), (1, 1), (1, -1), (-1, -1)],
        Face::PosZ => [(1, -1), (-1, -1), (-1, 1), (1, 1)],
        Face::NegZ => [(-1, -1), (1, -1), (1, 1), (-1, 1)],
    }
}

/// Four corner AO values for `face` of the cell at `voxel_pos`. Each
/// vertex samples along U, along V and along U+V, one step out on the
/// normal; `is_solid` is the caller's voxel query.
pub fn compute_face_ao(
    voxel_pos: (i32, i32, i32),
    face: Face,
    is_solid: impl Fn((i32, i32, i32)) -> bool,
) -> [u8; 4] {
    let (n, u, v) = face_axes(face);
    let signs = face_vertex_signs(face);
    let mut out = [0u8; 4];
    for (i, &(du, dv)) in signs.iter().enumerate() {
        let side1 = (
            voxel_pos.0 + n[0] + du * u[0],
            voxel_pos.1 + n[1] + du * u[1],
            voxel_pos.2 + n[2] + du * u[2],
        );
        let side2 = (
            voxel_pos.0 + n[0] + dv * v[0],
            voxel_pos.1 + n[1] + dv * v[1],
            voxel_pos.2 + n[2] + dv * v[2],
        );
        let corner = (
            voxel_pos.0 + n[0] + du * u[0] + dv * v[0],
            voxel_pos.1 + n[1] + du * u[1] + dv * v[1],
            voxel_pos.2 + n[2] + du * u[2] + dv * v[2],
        );
        out[i] = vertex_ao(is_solid(side1), is_solid(side2), is_solid(corner));
    }
    out
}

/// Pack an `[u8; 4]` AO tuple (each value 0–3) into 8 bits. Used by
/// the greedy mesher's mask key.
#[inline]
pub fn pack_ao(ao: [u8; 4]) -> u8 {
    debug_assert!(ao.iter().all(|&a| a <= 3));
    (ao[0] << 6) | (ao[1] << 4) | (ao[2] << 2) | ao[3]
}

#[inline]
pub fn unpack_ao(packed: u8) -> [u8; 4] {
    [
        (packed >> 6) & 0b11,
        (packed >> 4) & 0b11,
        (packed >> 2) & 0b11,
        packed & 0b11,
    ]
}

/// Flip the triangle split to the 1-3 diagonal when the 0-2 pair is
/// brighter, so the fold runs through the darker corners. Reference
/// only — `add_quad_with_ao_flip` compares the `f32` values directly.
#[allow(dead_code)]
#[inline]
pub fn should_flip_diagonal(ao: [u8; 4]) -> bool {
    ao[0] as u32 + ao[2] as u32 > ao[1] as u32 + ao[3] as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_ao_full_occlusion_when_both_sides_solid() {
        // L-shape: both sides solid → 0 (max occlusion).
        assert_eq!(vertex_ao(true, true, false), 0);
        assert_eq!(vertex_ao(true, true, true), 0);
    }

    #[test]
    fn vertex_ao_no_occlusion_when_all_air() {
        assert_eq!(vertex_ao(false, false, false), 3);
    }

    #[test]
    fn vertex_ao_partial_values() {
        assert_eq!(vertex_ao(true, false, false), 2);
        assert_eq!(vertex_ao(false, true, false), 2);
        assert_eq!(vertex_ao(false, false, true), 2);
        assert_eq!(vertex_ao(true, false, true), 1);
        assert_eq!(vertex_ao(false, true, true), 1);
    }

    #[test]
    fn flip_when_main_diagonal_brighter() {
        // 0-2 brighter (3+3=6 > 1+1=2) → flip
        assert!(should_flip_diagonal([3, 1, 3, 1]));
        // 1-3 brighter → don't flip
        assert!(!should_flip_diagonal([1, 3, 1, 3]));
    }

    #[test]
    fn no_flip_when_uniform() {
        assert!(!should_flip_diagonal([3, 3, 3, 3]));
        assert!(!should_flip_diagonal([0, 0, 0, 0]));
    }

    #[test]
    fn isolated_voxel_face_ao_is_all_3() {
        // No solid neighbors anywhere: every face's 4 corners → 3.
        let ao = compute_face_ao((0, 0, 0), Face::PosY, |_| false);
        assert_eq!(ao, [3, 3, 3, 3]);
    }

    #[test]
    fn voxel_corner_neighbor_darkens_appropriate_corner() {
        // Solid voxel at (1, 1, 0) (the side1 cell along U for
        // PosY's vertex 1 + 2). Vertices 1 and 2 should darken to
        // 2; vertices 0 and 3 stay at 3.
        let solid: std::collections::HashSet<(i32, i32, i32)> = [(1, 1, 0)].into_iter().collect();
        let ao = compute_face_ao((0, 0, 0), Face::PosY, |p| solid.contains(&p));
        assert_eq!(ao, [3, 2, 2, 3]);
    }

    #[test]
    fn pack_unpack_ao_roundtrip() {
        for a0 in 0..=3u8 {
            for a1 in 0..=3u8 {
                for a2 in 0..=3u8 {
                    for a3 in 0..=3u8 {
                        let original = [a0, a1, a2, a3];
                        assert_eq!(unpack_ao(pack_ao(original)), original);
                    }
                }
            }
        }
    }

    #[test]
    fn ao_to_f32_endpoints() {
        assert_eq!(ao_to_f32(0), 0.0);
        assert_eq!(ao_to_f32(3), 1.0);
    }

    #[test]
    fn face_vertex_signs_walk_all_four_corners_with_opposite_diagonals() {
        // Every face must visit all four corners of {-1,+1}², and
        // vertices 0/2 and 1/3 must be opposite pairs — the diagonal
        // the AO flip picks between.
        for face in Face::ALL {
            let signs = face_vertex_signs(face);
            let mut corners = signs;
            corners.sort_unstable();
            assert_eq!(
                corners,
                [(-1, -1), (-1, 1), (1, -1), (1, 1)],
                "{face:?}: the four vertices must be the four corners"
            );
            assert_eq!(
                (signs[0].0 + signs[2].0, signs[0].1 + signs[2].1),
                (0, 0),
                "{face:?}: vertices 0 and 2 must be opposite corners"
            );
            assert_eq!(
                (signs[1].0 + signs[3].0, signs[1].1 + signs[3].1),
                (0, 0),
                "{face:?}: vertices 1 and 3 must be opposite corners"
            );
        }
    }
}
