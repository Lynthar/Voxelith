//! Convert a sparse voxel list (typically from `procgen::VoxelPatch`)
//! into a renderable mesh, with face culling between voxels in the
//! input. Used for the procgen preview overlay so we can show the
//! generator's output without writing to the world.

use std::collections::HashMap;

use crate::core::{ChunkPos, Voxel};

use super::{apply_face_shading, face_quad_vertices, ChunkMesh, Face};

/// Build a mesh from a flat voxel list.
///
/// `alpha` is baked into every vertex's color alpha so the same shader
/// can render the result transparently when the caller pairs it with
/// an alpha-blending pipeline.
///
/// The list may contain duplicate positions; later entries win
/// (consistent with `HashMap` insertion). Air voxels are skipped.
/// Faces between two solid voxels in the list are culled.
pub fn patch_to_mesh(
    voxels: &[((i32, i32, i32), Voxel)],
    alpha: f32,
) -> ChunkMesh {
    // Index for O(1) neighbor lookup. `chunk_pos: ChunkPos::ZERO` is a
    // placeholder — the preview path doesn't go through Renderer's
    // per-chunk mesh map, the field just exists on `ChunkMesh`.
    let map: HashMap<(i32, i32, i32), Voxel> =
        voxels.iter().copied().collect();

    let mut mesh = ChunkMesh::with_capacity(
        ChunkPos::ZERO,
        map.len() * 4,
        map.len() * 6,
    );

    for (&(x, y, z), &voxel) in &map {
        if voxel.is_air() {
            continue;
        }

        let mut color = voxel.color_f32();
        color[3] = alpha;

        for face in Face::ALL {
            let (dx, dy, dz) = face.offset();
            let neighbor = (x + dx, y + dy, z + dz);

            // A face is hidden only if there's a *solid* voxel in the
            // patch right next to it. Air or absent neighbor -> draw.
            let visible = match map.get(&neighbor) {
                Some(v) if v.is_solid() => false,
                _ => true,
            };

            if visible {
                let shaded = apply_face_shading(color, face);
                let vertices =
                    face_quad_vertices(x as f32, y as f32, z as f32, face, shaded);
                mesh.add_quad(vertices);
            }
        }
    }

    mesh
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_patch() {
        let mesh = patch_to_mesh(&[], 1.0);
        assert!(mesh.is_empty());
    }

    #[test]
    fn test_single_voxel_six_faces() {
        let voxels = [((0, 0, 0), Voxel::from_rgb(255, 0, 0))];
        let mesh = patch_to_mesh(&voxels, 1.0);
        // 6 faces × 2 triangles
        assert_eq!(mesh.triangle_count(), 12);
        assert_eq!(mesh.vertex_count(), 24);
    }

    #[test]
    fn test_adjacent_voxels_internal_face_culled() {
        let voxels = [
            ((0, 0, 0), Voxel::from_rgb(255, 0, 0)),
            ((1, 0, 0), Voxel::from_rgb(0, 255, 0)),
        ];
        let mesh = patch_to_mesh(&voxels, 1.0);
        // 5 faces visible per voxel = 10 faces total
        assert_eq!(mesh.triangle_count(), 20);
    }

    #[test]
    fn test_alpha_baked_into_vertex_color() {
        let voxels = [((0, 0, 0), Voxel::from_rgb(255, 255, 255))];
        let mesh = patch_to_mesh(&voxels, 0.5);
        for v in &mesh.vertices {
            assert!((v.color[3] - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_air_skipped() {
        let voxels = [
            ((0, 0, 0), Voxel::AIR),
            ((1, 0, 0), Voxel::from_rgb(255, 0, 0)),
        ];
        let mesh = patch_to_mesh(&voxels, 1.0);
        // Only the solid voxel emits faces; air produces nothing.
        assert_eq!(mesh.triangle_count(), 12);
    }
}
