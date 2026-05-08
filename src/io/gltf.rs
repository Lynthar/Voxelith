//! glTF 2.0 binary (.glb) export.
//!
//! Walks every chunk via the `GreedyMesher` (same path as render and
//! OBJ export), accumulates one combined mesh, and writes a single
//! .glb file containing both the JSON scene description and the
//! binary vertex/index buffers. Output is a valid glTF 2.0 file
//! that imports directly into Unity, Unreal, Godot, Blender, and
//! every model viewer that handles the standard.
//!
//! Vertex layout matches the existing `mesh::Vertex` struct exactly
//! — POSITION (vec3 f32), NORMAL (vec3 f32), COLOR_0 (vec4 f32) —
//! emitted as deinterleaved bufferViews so JSON descriptors stay
//! simple (no `byteStride` annotations needed). Indices are u32 so
//! large worlds aren't capped at 64k vertices.
//!
//! ### File structure (per glTF 2.0 spec, §3.4 GLB)
//!
//! ```text
//! +----------------+
//! | Header (12 B)  |  magic "glTF" + version=2 + total length
//! +----------------+
//! | JSON chunk hdr |  length + type "JSON"
//! +----------------+
//! | JSON payload   |  scene description, padded to 4-byte align with 0x20
//! +----------------+
//! | BIN chunk hdr  |  length + type "BIN\0"   (omitted for empty world)
//! +----------------+
//! | BIN payload    |  raw bytes: positions | normals | colors | indices,
//! |                |  padded to 4-byte align with 0x00
//! +----------------+
//! ```
//!
//! Both chunks must be 4-byte aligned per spec; JSON pads with
//! ASCII space (0x20), BIN pads with zero.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use serde_json::json;
use thiserror::Error;

use crate::core::World;
use crate::mesh::{GreedyMesher, Mesher, Vertex};

#[derive(Debug, Error)]
pub enum GlbError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON serialization failed: {0}")]
    Json(#[from] serde_json::Error),
}

/// Stats reported back to the UI after a successful export.
#[derive(Debug, Clone, Copy, Default)]
pub struct GlbStats {
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub chunk_count: usize,
    pub byte_size: usize,
}

// glTF / OpenGL constants, named here once so the JSON below reads
// cleanly. See glTF 2.0 §3.6.2.4 (component types) and §3.6.2.5
// (buffer view targets).
const COMPONENT_TYPE_FLOAT: u32 = 5126;
const COMPONENT_TYPE_UINT: u32 = 5125;
const TARGET_ARRAY_BUFFER: u32 = 34962;
const TARGET_ELEMENT_ARRAY_BUFFER: u32 = 34963;
const PRIMITIVE_MODE_TRIANGLES: u32 = 4;

/// Export the current world as a binary glTF 2.0 file at `path`.
pub fn export_glb(world: &World, path: &Path) -> Result<GlbStats, GlbError> {
    let mesher = GreedyMesher::new();

    // Accumulate all chunks' meshes into one flat vertex / index pair.
    // Each chunk's local indices are offset by the running vertex
    // base so they index into the combined buffer.
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut chunk_count = 0usize;
    for (chunk_pos, _) in world.chunks() {
        let mesh = mesher.generate(world, *chunk_pos);
        if mesh.is_empty() {
            continue;
        }
        chunk_count += 1;
        let base = vertices.len() as u32;
        vertices.extend_from_slice(&mesh.vertices);
        indices.extend(mesh.indices.iter().map(|&i| base + i));
    }

    // Build the BIN payload (deinterleaved: positions, then normals,
    // then colors, then indices) and remember each section's offset.
    let mut bin = Vec::<u8>::new();

    let pos_offset = bin.len();
    for v in &vertices {
        bin.extend_from_slice(bytemuck::bytes_of(&v.position));
    }
    let pos_len = bin.len() - pos_offset;

    let normal_offset = bin.len();
    for v in &vertices {
        bin.extend_from_slice(bytemuck::bytes_of(&v.normal));
    }
    let normal_len = bin.len() - normal_offset;

    let color_offset = bin.len();
    for v in &vertices {
        bin.extend_from_slice(bytemuck::bytes_of(&v.color));
    }
    let color_len = bin.len() - color_offset;

    let index_offset = bin.len();
    bin.extend_from_slice(bytemuck::cast_slice(&indices));
    let index_len = bin.len() - index_offset;

    // BIN must be 4-byte aligned. Pad with zeros (spec §3.4.2).
    while bin.len() % 4 != 0 {
        bin.push(0);
    }

    // POSITION accessor REQUIRES `min` / `max` per spec §3.6.2.5.
    // For empty meshes we don't emit the accessor at all so this
    // bound calculation is gated.
    let bounds = if vertices.is_empty() {
        None
    } else {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for v in &vertices {
            for axis in 0..3 {
                if v.position[axis] < min[axis] {
                    min[axis] = v.position[axis];
                }
                if v.position[axis] > max[axis] {
                    max[axis] = v.position[axis];
                }
            }
        }
        Some((min, max))
    };

    // Build the JSON scene graph. Empty world produces a valid but
    // geometry-free glTF; populated world fills meshes/accessors/
    // bufferViews/buffers.
    let json_value = if let Some((min, max)) = bounds {
        json!({
            "asset": { "version": "2.0", "generator": "Voxelith" },
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0, "name": "Voxelith" }],
            "meshes": [{
                "name": "Voxelith",
                "primitives": [{
                    "attributes": {
                        "POSITION": 0,
                        "NORMAL": 1,
                        "COLOR_0": 2,
                    },
                    "indices": 3,
                    "mode": PRIMITIVE_MODE_TRIANGLES,
                }],
            }],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": COMPONENT_TYPE_FLOAT,
                    "count": vertices.len(),
                    "type": "VEC3",
                    "min": [min[0], min[1], min[2]],
                    "max": [max[0], max[1], max[2]],
                },
                {
                    "bufferView": 1,
                    "componentType": COMPONENT_TYPE_FLOAT,
                    "count": vertices.len(),
                    "type": "VEC3",
                },
                {
                    "bufferView": 2,
                    "componentType": COMPONENT_TYPE_FLOAT,
                    "count": vertices.len(),
                    "type": "VEC4",
                },
                {
                    "bufferView": 3,
                    "componentType": COMPONENT_TYPE_UINT,
                    "count": indices.len(),
                    "type": "SCALAR",
                },
            ],
            "bufferViews": [
                {
                    "buffer": 0,
                    "byteOffset": pos_offset,
                    "byteLength": pos_len,
                    "target": TARGET_ARRAY_BUFFER,
                },
                {
                    "buffer": 0,
                    "byteOffset": normal_offset,
                    "byteLength": normal_len,
                    "target": TARGET_ARRAY_BUFFER,
                },
                {
                    "buffer": 0,
                    "byteOffset": color_offset,
                    "byteLength": color_len,
                    "target": TARGET_ARRAY_BUFFER,
                },
                {
                    "buffer": 0,
                    "byteOffset": index_offset,
                    "byteLength": index_len,
                    "target": TARGET_ELEMENT_ARRAY_BUFFER,
                },
            ],
            "buffers": [{ "byteLength": bin.len() }],
        })
    } else {
        json!({
            "asset": { "version": "2.0", "generator": "Voxelith" },
            "scene": 0,
            "scenes": [{ "nodes": [] }],
        })
    };

    let mut json_bytes = serde_json::to_vec(&json_value)?;
    // JSON chunk also 4-byte aligned. Pad with ASCII space (0x20).
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }

    // Emit BIN chunk only when there's actual geometry — keeps
    // empty exports tighter and matches what Khronos's reference
    // exporter does.
    let has_bin = !bin.is_empty() && !vertices.is_empty();

    // Total file length: header (12) + JSON chunk (8 + json_bytes) +
    // optional BIN chunk (8 + bin).
    let total_len: u32 = (12
        + 8
        + json_bytes.len()
        + if has_bin { 8 + bin.len() } else { 0 }) as u32;

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // ===== Header (12 bytes) =====
    writer.write_all(b"glTF")?;
    writer.write_all(&2u32.to_le_bytes())?;
    writer.write_all(&total_len.to_le_bytes())?;

    // ===== JSON chunk =====
    writer.write_all(&(json_bytes.len() as u32).to_le_bytes())?;
    writer.write_all(b"JSON")?;
    writer.write_all(&json_bytes)?;

    // ===== BIN chunk =====
    if has_bin {
        writer.write_all(&(bin.len() as u32).to_le_bytes())?;
        writer.write_all(b"BIN\0")?;
        writer.write_all(&bin)?;
    }

    writer.flush()?;

    Ok(GlbStats {
        vertex_count: vertices.len(),
        triangle_count: indices.len() / 3,
        chunk_count,
        byte_size: total_len as usize,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Voxel;
    use std::io::Read;

    /// Read an entire .glb file and parse out (json_bytes, bin_bytes).
    /// Test-only helper so individual tests don't repeat byte-slicing
    /// boilerplate.
    fn read_glb(path: &Path) -> (Vec<u8>, Option<Vec<u8>>) {
        let mut bytes = Vec::new();
        File::open(path).unwrap().read_to_end(&mut bytes).unwrap();

        // Header
        assert_eq!(&bytes[0..4], b"glTF", "bad magic");
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(version, 2, "bad version");
        let total_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        assert_eq!(total_len, bytes.len(), "header length mismatch");

        // JSON chunk
        let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        assert_eq!(&bytes[16..20], b"JSON", "first chunk must be JSON");
        let json_start = 20;
        let json_end = json_start + json_len;
        let json = bytes[json_start..json_end].to_vec();

        // Optional BIN chunk
        if json_end < bytes.len() {
            let bin_len =
                u32::from_le_bytes(bytes[json_end..json_end + 4].try_into().unwrap())
                    as usize;
            assert_eq!(
                &bytes[json_end + 4..json_end + 8],
                b"BIN\0",
                "second chunk must be BIN"
            );
            let bin_start = json_end + 8;
            let bin = bytes[bin_start..bin_start + bin_len].to_vec();
            (json, Some(bin))
        } else {
            (json, None)
        }
    }

    #[test]
    fn test_export_empty_world_produces_valid_glb_no_bin() {
        let world = World::new();
        let path = std::env::temp_dir().join("voxelith_empty.glb");
        let stats = export_glb(&world, &path).unwrap();
        assert_eq!(stats.vertex_count, 0);
        assert_eq!(stats.triangle_count, 0);

        let (json_bytes, bin) = read_glb(&path);
        assert!(bin.is_none(), "empty world should omit BIN chunk");

        let json: serde_json::Value = serde_json::from_slice(&json_bytes).unwrap();
        assert_eq!(json["asset"]["version"], "2.0");
        assert_eq!(json["scenes"][0]["nodes"].as_array().unwrap().len(), 0);
        assert!(json["meshes"].is_null(), "no meshes for empty world");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_export_single_voxel_glb_structure() {
        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgb(255, 0, 0));
        world.clear_dirty_flags();
        let path = std::env::temp_dir().join("voxelith_single.glb");
        let stats = export_glb(&world, &path).unwrap();

        // Single voxel: 6 quads, 24 verts, 12 tris (greedy can't merge).
        assert_eq!(stats.vertex_count, 24);
        assert_eq!(stats.triangle_count, 12);

        let (json_bytes, bin) = read_glb(&path);
        let bin = bin.expect("non-empty world must have BIN chunk");

        let json: serde_json::Value = serde_json::from_slice(&json_bytes).unwrap();

        // POSITION accessor's count and type.
        let pos_acc = &json["accessors"][0];
        assert_eq!(pos_acc["count"], 24);
        assert_eq!(pos_acc["type"], "VEC3");
        assert_eq!(pos_acc["componentType"], COMPONENT_TYPE_FLOAT);
        // POSITION must have min/max per spec.
        assert!(pos_acc["min"].is_array());
        assert!(pos_acc["max"].is_array());

        // INDICES accessor count = 12 tris × 3 verts/tri = 36.
        assert_eq!(json["accessors"][3]["count"], 36);
        assert_eq!(json["accessors"][3]["componentType"], COMPONENT_TYPE_UINT);

        // BIN size: 24 verts × (12 + 12 + 16) bytes + 36 indices × 4
        // bytes = 960 + 144 = 1104, padded to 4-byte alignment = 1104.
        let expected = 24 * (12 + 12 + 16) + 36 * 4;
        // Allow up to 3 padding bytes for alignment.
        assert!(
            bin.len() == expected || bin.len() == expected + 4 - (expected % 4) % 4,
            "unexpected BIN size: got {}, expected ≈ {}",
            bin.len(),
            expected
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_export_position_min_max_match_geometry() {
        // Place voxels at known positions; bounding box in JSON's
        // POSITION accessor should match the cells' world extents
        // (lower corner of the lowest cell to upper corner of the
        // highest cell).
        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgb(255, 0, 0));
        world.set_voxel(2, 1, 3, Voxel::from_rgb(0, 255, 0));
        world.clear_dirty_flags();
        let path = std::env::temp_dir().join("voxelith_bounds.glb");
        export_glb(&world, &path).unwrap();

        let (json_bytes, _) = read_glb(&path);
        let json: serde_json::Value = serde_json::from_slice(&json_bytes).unwrap();
        let min = &json["accessors"][0]["min"];
        let max = &json["accessors"][0]["max"];

        // Lower corner of voxel (0, 0, 0) is (0, 0, 0); upper corner
        // of voxel (2, 1, 3) is (3, 2, 4).
        assert_eq!(min[0].as_f64().unwrap(), 0.0);
        assert_eq!(min[1].as_f64().unwrap(), 0.0);
        assert_eq!(min[2].as_f64().unwrap(), 0.0);
        assert_eq!(max[0].as_f64().unwrap(), 3.0);
        assert_eq!(max[1].as_f64().unwrap(), 2.0);
        assert_eq!(max[2].as_f64().unwrap(), 4.0);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_export_chunk_alignment() {
        // Spec §3.4: every chunk's length is a multiple of 4. Confirm
        // by checking the JSON chunk length (recorded in the chunk
        // header) is 4-aligned.
        let mut world = World::new();
        world.set_voxel(5, 5, 5, Voxel::from_rgb(10, 20, 30));
        let path = std::env::temp_dir().join("voxelith_align.glb");
        export_glb(&world, &path).unwrap();

        let mut bytes = Vec::new();
        File::open(&path).unwrap().read_to_end(&mut bytes).unwrap();
        let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        assert_eq!(json_len % 4, 0, "JSON chunk must be 4-byte aligned");
        let bin_len = u32::from_le_bytes(
            bytes[20 + json_len..20 + json_len + 4]
                .try_into()
                .unwrap(),
        ) as usize;
        assert_eq!(bin_len % 4, 0, "BIN chunk must be 4-byte aligned");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_buffer_view_offsets_within_buffer() {
        // Sanity: every bufferView's (byteOffset + byteLength) must
        // fit within the parent buffer's byteLength. Catches off-by-
        // one bugs in offset accounting.
        let mut world = World::new();
        for x in 0..3 {
            for z in 0..3 {
                world.set_voxel(x, 0, z, Voxel::from_rgb(100, 100, 100));
            }
        }
        let path = std::env::temp_dir().join("voxelith_views.glb");
        export_glb(&world, &path).unwrap();

        let (json_bytes, bin) = read_glb(&path);
        let json: serde_json::Value = serde_json::from_slice(&json_bytes).unwrap();
        let buffer_len = json["buffers"][0]["byteLength"].as_u64().unwrap() as usize;
        // Buffer length should match (or be slightly less due to padding)
        // the actual BIN chunk we wrote.
        assert!(bin.unwrap().len() >= buffer_len);

        for view in json["bufferViews"].as_array().unwrap() {
            let offset = view["byteOffset"].as_u64().unwrap() as usize;
            let length = view["byteLength"].as_u64().unwrap() as usize;
            assert!(
                offset + length <= buffer_len,
                "view {:?} extends past buffer end",
                view
            );
        }

        let _ = std::fs::remove_file(&path);
    }
}
