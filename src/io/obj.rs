//! Wavefront OBJ export.
//!
//! Walks every chunk in the world, re-meshes it with the `NaiveMesher`
//! (the same path used at render time), and writes the combined
//! geometry to a single OBJ file. Vertex colors are emitted using the
//! `v x y z r g b` extension that Blender / MeshLab / most modern
//! voxel-aware tools recognize; tools that don't understand the
//! trailing RGB just ignore it and produce an uncolored mesh.
//!
//! Y is up (matches Voxelith's world axis), so importers using the
//! default OBJ axes get the orientation right out of the box. CCW
//! winding from outside is preserved end-to-end (mesher → OBJ); no
//! axis or winding flip needed.
//!
//! The exporter doesn't deduplicate vertices across chunks. Each
//! chunk's vertices are emitted independently and its triangle
//! indices are translated to global OBJ-1-indexed values. Greedy
//! meshing (TODO) would shrink output a lot more than dedup would.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use thiserror::Error;

use crate::core::World;
use crate::mesh::{Mesher, NaiveMesher};

#[derive(Debug, Error)]
pub enum ObjError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Summary stats from an OBJ export. Used by the UI to surface a
/// "wrote N tris from M chunks" status line.
#[derive(Debug, Clone, Copy, Default)]
pub struct ObjStats {
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub chunk_count: usize,
}

/// Export the current world to a Wavefront OBJ at `path`.
///
/// Returns counts of what was written. An empty world produces a valid
/// OBJ with header + object name but no geometry — readers should
/// import it as an empty mesh rather than choking.
pub fn export_obj(world: &World, path: &Path) -> Result<ObjStats, ObjError> {
    let mesher = NaiveMesher::new();

    // Generate meshes for every chunk and keep only non-empty ones so
    // air-only chunks don't bloat the output with `g` headers.
    let mut chunk_meshes = Vec::new();
    let mut stats = ObjStats::default();
    for (chunk_pos, _) in world.chunks() {
        let mesh = mesher.generate(world, *chunk_pos);
        if mesh.is_empty() {
            continue;
        }
        stats.vertex_count += mesh.vertex_count();
        stats.triangle_count += mesh.triangle_count();
        stats.chunk_count += 1;
        chunk_meshes.push(mesh);
    }

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "# Voxelith OBJ export")?;
    writeln!(
        writer,
        "# vertices: {}, triangles: {}, chunks: {}",
        stats.vertex_count, stats.triangle_count, stats.chunk_count
    )?;
    writeln!(writer, "o Voxelith")?;

    // Faces in OBJ are 1-indexed and reference global vertex / normal
    // counters that span the whole file. We emit per-chunk: first the
    // chunk's vertices, then its normals, then its faces translated
    // into the running global index space. OBJ allows the v / vn / f
    // sequence to interleave like this as long as referenced indices
    // are defined earlier in the file, which they always are here.
    let mut base: usize = 1;
    for mesh in &chunk_meshes {
        let cp = mesh.chunk_pos;
        writeln!(writer, "g chunk_{}_{}_{}", cp.x, cp.y, cp.z)?;

        for v in &mesh.vertices {
            // 4 decimal places on positions is exact for integer-aligned
            // voxel corners (every coordinate is an integer); keeps the
            // file parseable as plain text by humans inspecting it.
            // Colors get 3 decimals — anything finer is below the input
            // RGB-byte resolution so further precision is noise.
            writeln!(
                writer,
                "v {:.4} {:.4} {:.4} {:.3} {:.3} {:.3}",
                v.position[0],
                v.position[1],
                v.position[2],
                v.color[0],
                v.color[1],
                v.color[2],
            )?;
        }
        for v in &mesh.vertices {
            writeln!(
                writer,
                "vn {:.4} {:.4} {:.4}",
                v.normal[0], v.normal[1], v.normal[2]
            )?;
        }
        for tri in mesh.indices.chunks_exact(3) {
            let a = base + tri[0] as usize;
            let b = base + tri[1] as usize;
            let c = base + tri[2] as usize;
            // `f v//vn` — no texture coordinates emitted, so the slot is
            // empty between the slashes.
            writeln!(writer, "f {a}//{a} {b}//{b} {c}//{c}")?;
        }
        base += mesh.vertex_count();
    }

    writer.flush()?;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Voxel;
    use std::io::Read;

    #[test]
    fn test_export_empty_world_produces_header_only() {
        let world = World::new();
        let dir = std::env::temp_dir();
        let path = dir.join("voxelith_empty_export.obj");
        let stats = export_obj(&world, &path).unwrap();
        assert_eq!(stats.triangle_count, 0);
        assert_eq!(stats.vertex_count, 0);
        assert_eq!(stats.chunk_count, 0);

        let mut s = String::new();
        File::open(&path).unwrap().read_to_string(&mut s).unwrap();
        // Header + object name lines exist; no `f` faces.
        assert!(s.contains("# Voxelith OBJ export"));
        assert!(s.contains("o Voxelith"));
        assert!(!s.contains("\nf "));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_export_single_voxel_writes_24_vertices_12_tris() {
        // One isolated voxel exposes all 6 faces. Naive mesher emits
        // 4 verts × 6 faces = 24 verts and 2 tris × 6 faces = 12 tris.
        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgb(255, 0, 0));
        world.clear_dirty_flags();

        let dir = std::env::temp_dir();
        let path = dir.join("voxelith_one_voxel.obj");
        let stats = export_obj(&world, &path).unwrap();
        assert_eq!(stats.vertex_count, 24);
        assert_eq!(stats.triangle_count, 12);
        assert_eq!(stats.chunk_count, 1);

        let mut s = String::new();
        File::open(&path).unwrap().read_to_string(&mut s).unwrap();
        // 24 `v ` lines and 12 `f ` lines (counting line starts).
        let v_lines = s.lines().filter(|l| l.starts_with("v ")).count();
        let vn_lines = s.lines().filter(|l| l.starts_with("vn ")).count();
        let f_lines = s.lines().filter(|l| l.starts_with("f ")).count();
        assert_eq!(v_lines, 24);
        assert_eq!(vn_lines, 24);
        assert_eq!(f_lines, 12);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_export_two_voxels_share_no_faces() {
        // Two non-adjacent voxels: each contributes its full 6-face cube.
        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgb(255, 0, 0));
        world.set_voxel(5, 0, 0, Voxel::from_rgb(0, 0, 255));
        world.clear_dirty_flags();

        let dir = std::env::temp_dir();
        let path = dir.join("voxelith_two_voxels.obj");
        let stats = export_obj(&world, &path).unwrap();
        assert_eq!(stats.triangle_count, 24); // 2 × 12
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_export_adjacent_voxels_cull_shared_face() {
        // Two voxels sharing a face: the shared face is culled from
        // both, so the total is 12 - 2 = 10 visible faces × 2 tris = 20.
        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgb(255, 0, 0));
        world.set_voxel(1, 0, 0, Voxel::from_rgb(0, 0, 255));
        world.clear_dirty_flags();

        let dir = std::env::temp_dir();
        let path = dir.join("voxelith_two_adjacent.obj");
        let stats = export_obj(&world, &path).unwrap();
        assert_eq!(stats.triangle_count, 20);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_export_face_indices_within_vertex_range() {
        // Sanity: every f line's indices fall within the file's
        // declared vertex / normal counts. Catches off-by-one bugs in
        // the global index translation.
        let mut world = World::new();
        for x in 0..3 {
            for z in 0..3 {
                world.set_voxel(x, 0, z, Voxel::from_rgb(100, 200, 50));
            }
        }
        world.clear_dirty_flags();

        let dir = std::env::temp_dir();
        let path = dir.join("voxelith_3x3.obj");
        let stats = export_obj(&world, &path).unwrap();

        let mut s = String::new();
        File::open(&path).unwrap().read_to_string(&mut s).unwrap();
        let _ = std::fs::remove_file(&path);

        for line in s.lines().filter(|l| l.starts_with("f ")) {
            // Format: `f a//a b//b c//c`. Parse the three indices.
            for token in line[2..].split_whitespace() {
                let idx: usize = token
                    .split("//")
                    .next()
                    .unwrap()
                    .parse()
                    .unwrap();
                assert!(
                    idx >= 1 && idx <= stats.vertex_count,
                    "face index {} out of range [1, {}]",
                    idx,
                    stats.vertex_count
                );
            }
        }
    }
}
