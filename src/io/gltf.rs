//! glTF 2.0 binary (.glb) export.
//!
//! Walks every chunk via the `GreedyMesher` (same path as render and
//! OBJ export), accumulates one combined mesh, and writes a single
//! .glb file containing both the JSON scene description and the
//! binary vertex/index buffers. Output is a valid glTF 2.0 file
//! that imports directly into Unity, Unreal, Godot, Blender, and
//! every model viewer that handles the standard.
//!
//! Each primitive emits POSITION (vec3 f32), NORMAL (vec3 f32),
//! COLOR_0 (vec4 f32, with AO baked in), the custom `_TINTZONE`
//! (scalar f32 faction zone), and TEXCOORD_0 (vec2 f32 — the same
//! zone in `.x`, so Unity glTFast, which drops custom attributes,
//! can still read it). All deinterleaved so JSON descriptors stay
//! simple (no `byteStride` annotations needed). Indices are u32 so
//! large worlds aren't capped at 64k vertices. The full
//! engine-consumption contract is in `docs/ENGINE_CONTRACT.md`.
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
//! | BIN payload    |  per group, back to back: positions | normals |
//! |                |  colors | tintzones | texcoords | indices,
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
use crate::mesh::{mesh_chunk_by_material, mesh_world_smoothed, Vertex};

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

/// Where to move the asset's local origin before export, so a consumer
/// can place it at a world position predictably. `BaseCenter` puts the
/// XZ center of the footprint with the bottom (min Y) at the origin —
/// buildings sit on the ground; `feet` (a spec alias for `BaseCenter`)
/// is the same point for characters; `Center` centers all axes; `Origin`
/// leaves model space untouched. See `docs/GAME_PIPELINE_ROADMAP.md` §3.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pivot {
    /// Keep model-space coordinates as-is (identity).
    Origin,
    /// XZ center of the geometry bounds; Y at the bottom (min Y).
    BaseCenter,
    /// Center of the geometry bounds on every axis.
    Center,
}

/// Up-axis convention of the consuming engine. glTF is natively Y-up
/// (Unity glTFast / Godot convert on import), so `Y` is the identity;
/// `Z` adds a +90° rotation about X for Z-up engines (e.g. Unreal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpAxis {
    Y,
    Z,
}

/// Deterministic placement applied at export as a single root node that
/// wraps the mesh + socket nodes: pivot → translation, up-axis →
/// rotation, unit scale → scale. The default is the identity (`Origin` /
/// `Y` / `1.0`) and produces byte-identical output to the plain
/// `export_glb`, so a minimal bake reproduces the interactive export.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExportTransform {
    pub pivot: Pivot,
    pub up_axis: UpAxis,
    pub unit_scale: f32,
}

impl Default for ExportTransform {
    fn default() -> Self {
        Self {
            pivot: Pivot::Origin,
            up_axis: UpAxis::Y,
            unit_scale: 1.0,
        }
    }
}

impl ExportTransform {
    /// True when this transform changes nothing — no root node is emitted
    /// and the output matches the un-transformed export byte-for-byte.
    pub fn is_identity(&self) -> bool {
        self.pivot == Pivot::Origin
            && self.up_axis == UpAxis::Y
            && (self.unit_scale - 1.0).abs() < 1e-9
    }
}

/// A named attachment point to emit as an empty glTF node (no mesh).
///
/// `translation` is the socket's world position and `rotation` is a
/// unit quaternion in glTF `[x, y, z, w]` order. The caller
/// (`app::file_ops`) builds these from `editor::Socket` — deriving the
/// rotation via `Socket::rotation` — so the orientation convention
/// lives in one place and this module stays free of math + `editor`
/// dependencies.
#[derive(Debug, Clone)]
pub struct SocketNode {
    pub name: String,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
}

// glTF / OpenGL constants, named here once so the JSON below reads
// cleanly. See glTF 2.0 §3.6.2.4 (component types) and §3.6.2.5
// (buffer view targets).
const COMPONENT_TYPE_FLOAT: u32 = 5126;
const COMPONENT_TYPE_UINT: u32 = 5125;
const TARGET_ARRAY_BUFFER: u32 = 34962;
const TARGET_ELEMENT_ARRAY_BUFFER: u32 = 34963;
const PRIMITIVE_MODE_TRIANGLES: u32 = 4;

/// One material group's combined geometry for GLB export.
struct GroupBuffers {
    /// Material group id: bit0 emissive, bit1 metallic (0 = plain).
    group_id: u8,
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

impl GroupBuffers {
    fn new(group_id: u8) -> Self {
        Self {
            group_id,
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }
}

/// Human name for a material group → glTF `materials[].name`.
fn material_name(group_id: u8) -> &'static str {
    match group_id {
        1 => "Voxelith_emissive",
        2 => "Voxelith_metallic",
        3 => "Voxelith_emissive_metallic",
        _ => "Voxelith",
    }
}

/// Build the glTF material for a group. The glTF *default* material is
/// metallic (`metallicFactor` defaults to 1.0), so even plain voxels get
/// an explicit `metallicFactor = 0` to render as matte colored surfaces;
/// metallic voxels get `metallicFactor = 1` + a lower roughness; emissive
/// voxels get a constant white `emissiveFactor` (core glTF can't express
/// per-vertex emissive color — vertex color only multiplies base color).
fn material_json(group_id: u8) -> serde_json::Value {
    let metallic = group_id & 0b10 != 0;
    let emissive = group_id & 0b01 != 0;
    let mut m = json!({
        "name": material_name(group_id),
        "pbrMetallicRoughness": {
            "metallicFactor": if metallic { 1.0 } else { 0.0 },
            "roughnessFactor": if metallic { 0.4 } else { 1.0 },
        },
    });
    if emissive {
        m["emissiveFactor"] = json!([1.0, 1.0, 1.0]);
    }
    m
}

/// Export the current world as a binary glTF 2.0 file at `path`,
/// grouping geometry by material flag (plain / emissive / metallic /
/// both). Each present group becomes its own primitive + material, so
/// emissive and metallic voxels carry real glTF PBR materials. Plain
/// voxels get an explicit non-metallic material (the glTF default
/// material is metallic, which would otherwise render them as metal).
///
/// `sockets` are emitted as empty nodes alongside the mesh node (see
/// [`SocketNode`]); pass `&[]` for none.
pub fn export_glb(
    world: &World,
    sockets: &[SocketNode],
    path: &Path,
) -> Result<GlbStats, GlbError> {
    export_glb_with_transform(world, sockets, path, ExportTransform::default())
}

/// Like [`export_glb`] but applies a deterministic placement
/// [`ExportTransform`] (pivot / up-axis / uniform scale) as a single
/// root node wrapping the mesh and socket nodes. The default transform
/// is the identity and yields byte-identical output to [`export_glb`];
/// the headless bake ([`crate::bake`]) uses this to emit assets with a
/// consistent pivot + scale for the game engine (see
/// `docs/GAME_PIPELINE_ROADMAP.md` §3.5).
pub fn export_glb_with_transform(
    world: &World,
    sockets: &[SocketNode],
    path: &Path,
    transform: ExportTransform,
) -> Result<GlbStats, GlbError> {
    // Accumulate combined vertex / index buffers per material group.
    let mut groups: Vec<GroupBuffers> = (0u8..4).map(GroupBuffers::new).collect();
    let mut chunk_count = 0usize;
    for (chunk_pos, _) in world.chunks() {
        let per_material = mesh_chunk_by_material(world, *chunk_pos);
        if !per_material.is_empty() {
            chunk_count += 1;
        }
        for (gid, mesh) in per_material {
            let g = &mut groups[gid as usize];
            let base = g.vertices.len() as u32;
            g.vertices.extend_from_slice(&mesh.vertices);
            g.indices.extend(mesh.indices.iter().map(|&i| base + i));
        }
    }
    // Drop empty groups; the rest become primitives in id order.
    groups.retain(|g| !g.vertices.is_empty());
    write_glb_groups(&groups, sockets, chunk_count, path, transform)
}

/// Export the world as a glTF Binary with Marching-Cubes smoothing.
/// Counterpart to `export_obj_smoothed`: walks the entire world as a
/// single density field and runs MC to produce a continuous
/// interpolated surface. Per-vertex colors and gradient-based
/// normals carry through to the GLB unchanged from MC's output.
/// `chunk_count` is reported as 1 (single combined mesh).
///
/// `blur` matches `export_obj_smoothed`: `false` keeps thin features
/// at the cost of less organic curvature ("rounded cubes"); `true`
/// applies a 3×3×3 blur for clay-like terrain output but dissolves
/// sparse / 1-cell-wide detail.
///
/// `sockets` export identically to the non-smoothed path — they're
/// independent of the mesh source.
pub fn export_glb_smoothed(
    world: &World,
    sockets: &[SocketNode],
    path: &Path,
    blur: bool,
) -> Result<GlbStats, GlbError> {
    export_glb_smoothed_with_transform(world, sockets, path, blur, ExportTransform::default())
}

/// [`export_glb_smoothed`] with a deterministic placement
/// [`ExportTransform`] (see [`export_glb_with_transform`]).
pub fn export_glb_smoothed_with_transform(
    world: &World,
    sockets: &[SocketNode],
    path: &Path,
    blur: bool,
    transform: ExportTransform,
) -> Result<GlbStats, GlbError> {
    let mesh = mesh_world_smoothed(world, blur);
    let chunk_count = if mesh.is_empty() { 0 } else { 1 };
    // MC output carries no material flags — a single plain group.
    let groups = if mesh.is_empty() {
        Vec::new()
    } else {
        vec![GroupBuffers {
            group_id: 0,
            vertices: mesh.vertices,
            indices: mesh.indices,
        }]
    };
    write_glb_groups(&groups, sockets, chunk_count, path, transform)
}

/// Write one or more material groups to a binary glTF 2.0 file. Each
/// group becomes a primitive (POSITION / NORMAL / COLOR_0 / _TINTZONE /
/// TEXCOORD_0 / indices) plus a material; the BIN payload lays the groups
/// out back to back. An empty
/// `groups` slice produces a valid geometry-free glTF (no BIN chunk) —
/// which `sockets` can still populate with empty nodes. `chunk_count` is
/// passed through to the returned stats. Per-vertex AO is baked into the
/// exported color (see `Vertex::baked_color`).
fn write_glb_groups(
    groups: &[GroupBuffers],
    sockets: &[SocketNode],
    chunk_count: usize,
    path: &Path,
    transform: ExportTransform,
) -> Result<GlbStats, GlbError> {
    // Per-group byte sections within the BIN, plus POSITION bounds.
    struct Section {
        pos: (usize, usize),
        normal: (usize, usize),
        color: (usize, usize),
        tintzone: (usize, usize),
        texcoord: (usize, usize),
        index: (usize, usize),
        min: [f32; 3],
        max: [f32; 3],
    }

    let mut bin = Vec::<u8>::new();
    let mut sections: Vec<Section> = Vec::with_capacity(groups.len());
    let mut total_vertices = 0usize;
    let mut total_indices = 0usize;

    for g in groups {
        let pos_offset = bin.len();
        for v in &g.vertices {
            bin.extend_from_slice(bytemuck::bytes_of(&v.position));
        }
        let pos_len = bin.len() - pos_offset;

        let normal_offset = bin.len();
        for v in &g.vertices {
            bin.extend_from_slice(bytemuck::bytes_of(&v.normal));
        }
        let normal_len = bin.len() - normal_offset;

        let color_offset = bin.len();
        for v in &g.vertices {
            // Bake per-vertex AO into the exported color (see
            // `Vertex::baked_color`); MC-smoothed meshes carry ao = 1.0.
            bin.extend_from_slice(bytemuck::bytes_of(&v.baked_color()));
        }
        let color_len = bin.len() - color_offset;

        let tintzone_offset = bin.len();
        for v in &g.vertices {
            bin.extend_from_slice(bytemuck::bytes_of(&v.tint_zone));
        }
        let tintzone_len = bin.len() - tintzone_offset;

        // Tint zone ALSO as TEXCOORD_0 = vec2(zone, 0). Unity glTFast drops
        // custom attributes (so it can't read `_TINTZONE`) but imports UV
        // sets, so this is the channel a stock-Unity uber-shader reads. See
        // docs/ENGINE_CONTRACT.md §6.2 (incl. the glTFast UV-pruning caveat).
        let texcoord_offset = bin.len();
        for v in &g.vertices {
            bin.extend_from_slice(bytemuck::bytes_of(&[v.tint_zone, 0.0f32]));
        }
        let texcoord_len = bin.len() - texcoord_offset;

        let index_offset = bin.len();
        bin.extend_from_slice(bytemuck::cast_slice(&g.indices));
        let index_len = bin.len() - index_offset;

        // POSITION accessor REQUIRES `min` / `max` per spec §3.6.2.5.
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for v in &g.vertices {
            for axis in 0..3 {
                if v.position[axis] < min[axis] {
                    min[axis] = v.position[axis];
                }
                if v.position[axis] > max[axis] {
                    max[axis] = v.position[axis];
                }
            }
        }

        sections.push(Section {
            pos: (pos_offset, pos_len),
            normal: (normal_offset, normal_len),
            color: (color_offset, color_len),
            tintzone: (tintzone_offset, tintzone_len),
            texcoord: (texcoord_offset, texcoord_len),
            index: (index_offset, index_len),
            min,
            max,
        });
        total_vertices += g.vertices.len();
        total_indices += g.indices.len();
    }

    // BIN must be 4-byte aligned. Pad with zeros (spec §3.4.2).
    while bin.len() % 4 != 0 {
        bin.push(0);
    }

    // Build per-group geometry descriptors (empty when there's no
    // geometry): one primitive + material + 6 accessors / bufferViews
    // per group (POSITION, NORMAL, COLOR_0, _TINTZONE, TEXCOORD_0,
    // indices), all sharing buffer 0.
    let mut accessors = Vec::with_capacity(groups.len() * 6);
    let mut buffer_views = Vec::with_capacity(groups.len() * 6);
    let mut primitives = Vec::with_capacity(groups.len());
    let mut materials = Vec::with_capacity(groups.len());

    for (i, (g, s)) in groups.iter().zip(&sections).enumerate() {
        // Group i owns accessors / bufferViews [6i .. 6i+6):
        // POSITION, NORMAL, COLOR_0, _TINTZONE, TEXCOORD_0, indices.
        let base = (i * 6) as u32;
        accessors.push(json!({
            "bufferView": base,
            "componentType": COMPONENT_TYPE_FLOAT,
            "count": g.vertices.len(),
            "type": "VEC3",
            "min": [s.min[0], s.min[1], s.min[2]],
            "max": [s.max[0], s.max[1], s.max[2]],
        }));
        accessors.push(json!({
            "bufferView": base + 1,
            "componentType": COMPONENT_TYPE_FLOAT,
            "count": g.vertices.len(),
            "type": "VEC3",
        }));
        accessors.push(json!({
            "bufferView": base + 2,
            "componentType": COMPONENT_TYPE_FLOAT,
            "count": g.vertices.len(),
            "type": "VEC4",
        }));
        // _TINTZONE: per-vertex faction recolor zone (SCALAR f32).
        accessors.push(json!({
            "bufferView": base + 3,
            "componentType": COMPONENT_TYPE_FLOAT,
            "count": g.vertices.len(),
            "type": "SCALAR",
        }));
        // TEXCOORD_0: the same zone in .x (VEC2 f32) — the glTFast-readable
        // mirror of _TINTZONE (see docs/GAME_PIPELINE_ROADMAP.md §3.2).
        accessors.push(json!({
            "bufferView": base + 4,
            "componentType": COMPONENT_TYPE_FLOAT,
            "count": g.vertices.len(),
            "type": "VEC2",
        }));
        accessors.push(json!({
            "bufferView": base + 5,
            "componentType": COMPONENT_TYPE_UINT,
            "count": g.indices.len(),
            "type": "SCALAR",
        }));

        buffer_views.push(json!({ "buffer": 0, "byteOffset": s.pos.0, "byteLength": s.pos.1, "target": TARGET_ARRAY_BUFFER }));
        buffer_views.push(json!({ "buffer": 0, "byteOffset": s.normal.0, "byteLength": s.normal.1, "target": TARGET_ARRAY_BUFFER }));
        buffer_views.push(json!({ "buffer": 0, "byteOffset": s.color.0, "byteLength": s.color.1, "target": TARGET_ARRAY_BUFFER }));
        buffer_views.push(json!({ "buffer": 0, "byteOffset": s.tintzone.0, "byteLength": s.tintzone.1, "target": TARGET_ARRAY_BUFFER }));
        buffer_views.push(json!({ "buffer": 0, "byteOffset": s.texcoord.0, "byteLength": s.texcoord.1, "target": TARGET_ARRAY_BUFFER }));
        buffer_views.push(json!({ "buffer": 0, "byteOffset": s.index.0, "byteLength": s.index.1, "target": TARGET_ELEMENT_ARRAY_BUFFER }));

        primitives.push(json!({
            "attributes": { "POSITION": base, "NORMAL": base + 1, "COLOR_0": base + 2, "_TINTZONE": base + 3, "TEXCOORD_0": base + 4 },
            "indices": base + 5,
            "material": i,
            "mode": PRIMITIVE_MODE_TRIANGLES,
        }));

        materials.push(material_json(g.group_id));
    }

    // Assemble the scene's node list: the mesh node (node 0) when there
    // is geometry, then one empty node per socket — `name` +
    // `translation` + `rotation`, no `mesh` — which is the standard
    // glTF representation of an attachment point. Sockets export even
    // for an empty world (a sockets-only glTF is valid).
    let mut nodes: Vec<serde_json::Value> = Vec::new();
    let mut scene_nodes: Vec<usize> = Vec::new();
    if !groups.is_empty() {
        nodes.push(json!({ "mesh": 0, "name": "Voxelith" }));
        scene_nodes.push(0);
    }
    for sock in sockets {
        scene_nodes.push(nodes.len());
        nodes.push(json!({
            "name": sock.name,
            "translation": sock.translation,
            "rotation": sock.rotation,
        }));
    }

    // Deterministic placement (§3.5): for a non-identity transform, wrap
    // every scene root (mesh + sockets) under one parent node carrying the
    // pivot offset, up-axis rotation, and uniform scale, so geometry and
    // sockets move together and the asset's local origin becomes the chosen
    // pivot. An identity transform adds nothing, leaving the output
    // byte-for-byte identical to the plain export.
    if !transform.is_identity() && !scene_nodes.is_empty() {
        let bounds = sections.iter().fold(
            None,
            |acc: Option<([f32; 3], [f32; 3])>, s| match acc {
                None => Some((s.min, s.max)),
                Some((mut lo, mut hi)) => {
                    for a in 0..3 {
                        lo[a] = lo[a].min(s.min[a]);
                        hi[a] = hi[a].max(s.max[a]);
                    }
                    Some((lo, hi))
                }
            },
        );
        let root = root_transform_node(&scene_nodes, bounds, transform);
        scene_nodes = vec![nodes.len()];
        nodes.push(root);
    }

    // Base document; geometry-only keys (meshes/materials/accessors/
    // bufferViews/buffers) are attached only when groups exist, and
    // `nodes` only when there's at least one node (mesh or socket).
    let mut json_value = json!({
        "asset": { "version": "2.0", "generator": "Voxelith" },
        "scene": 0,
        "scenes": [{ "nodes": scene_nodes }],
    });
    if !nodes.is_empty() {
        json_value["nodes"] = json!(nodes);
    }
    if !groups.is_empty() {
        json_value["meshes"] = json!([{ "name": "Voxelith", "primitives": primitives }]);
        json_value["materials"] = json!(materials);
        json_value["accessors"] = json!(accessors);
        json_value["bufferViews"] = json!(buffer_views);
        json_value["buffers"] = json!([{ "byteLength": bin.len() }]);
    }

    let mut json_bytes = serde_json::to_vec(&json_value)?;
    // JSON chunk also 4-byte aligned. Pad with ASCII space (0x20).
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }

    // Emit BIN chunk only when there's actual geometry.
    let has_bin = !bin.is_empty() && !groups.is_empty();

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
        vertex_count: total_vertices,
        triangle_count: total_indices / 3,
        chunk_count,
        byte_size: total_len as usize,
    })
}

/// Build the parent node that applies an [`ExportTransform`] to all
/// `children` (the existing scene roots). Translation places the chosen
/// pivot at the local origin (after scale + rotation), rotation is the
/// up-axis conversion, and scale is uniform. `bounds` is the geometry
/// AABB in mesh space, or `None` for a geometry-free scene (pivot then
/// falls back to the origin). Vertex data is never touched — placement
/// lives entirely in this node, so the export stays lossless.
fn root_transform_node(
    children: &[usize],
    bounds: Option<([f32; 3], [f32; 3])>,
    t: ExportTransform,
) -> serde_json::Value {
    let s = t.unit_scale;
    let pivot = match (t.pivot, bounds) {
        (Pivot::BaseCenter, Some((lo, hi))) => {
            [(lo[0] + hi[0]) * 0.5, lo[1], (lo[2] + hi[2]) * 0.5]
        }
        (Pivot::Center, Some((lo, hi))) => [
            (lo[0] + hi[0]) * 0.5,
            (lo[1] + hi[1]) * 0.5,
            (lo[2] + hi[2]) * 0.5,
        ],
        // `Origin`, or any pivot with no geometry to measure.
        _ => [0.0, 0.0, 0.0],
    };
    let rotation = match t.up_axis {
        UpAxis::Y => [0.0, 0.0, 0.0, 1.0],
        // +90° about X maps model +Y onto world +Z, for Z-up engines.
        UpAxis::Z => [
            std::f32::consts::FRAC_1_SQRT_2,
            0.0,
            0.0,
            std::f32::consts::FRAC_1_SQRT_2,
        ],
    };
    // translation = -(R · (scale · pivot)) so the pivot lands at origin.
    let scaled = [pivot[0] * s, pivot[1] * s, pivot[2] * s];
    let rotated = rotate_vec_by_quat(scaled, rotation);
    let translation = [-rotated[0], -rotated[1], -rotated[2]];
    json!({
        "name": "Voxelith_root",
        "translation": translation,
        "rotation": rotation,
        "scale": [s, s, s],
        "children": children,
    })
}

/// Rotate a vector by a unit quaternion `[x, y, z, w]`
/// (`v' = v + 2·q_xyz × (q_xyz × v + w·v)`). Kept local so this module
/// stays free of a math-library dependency.
fn rotate_vec_by_quat(v: [f32; 3], q: [f32; 4]) -> [f32; 3] {
    let (qx, qy, qz, qw) = (q[0], q[1], q[2], q[3]);
    // t = 2 · (q_xyz × v)
    let tx = 2.0 * (qy * v[2] - qz * v[1]);
    let ty = 2.0 * (qz * v[0] - qx * v[2]);
    let tz = 2.0 * (qx * v[1] - qy * v[0]);
    // v' = v + qw · t + q_xyz × t
    [
        v[0] + qw * tx + (qy * tz - qz * ty),
        v[1] + qw * ty + (qz * tx - qx * tz),
        v[2] + qw * tz + (qx * ty - qy * tx),
    ]
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
        let stats = export_glb(&world, &[], &path).unwrap();
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
        let stats = export_glb(&world, &[], &path).unwrap();

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

        // INDICES is now the 6th accessor per group (POSITION, NORMAL,
        // COLOR_0, _TINTZONE, TEXCOORD_0, indices); count = 12 tris × 3 = 36.
        assert_eq!(json["accessors"][5]["count"], 36);
        assert_eq!(json["accessors"][5]["componentType"], COMPONENT_TYPE_UINT);

        // BIN size: 24 verts × (12 pos + 12 normal + 16 color + 4 tintzone
        // + 8 texcoord) bytes + 36 indices × 4 bytes.
        let expected = 24 * (12 + 12 + 16 + 4 + 8) + 36 * 4;
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
        export_glb(&world, &[], &path).unwrap();

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
        export_glb(&world, &[], &path).unwrap();

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
        export_glb(&world, &[], &path).unwrap();

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

    #[test]
    fn test_export_bakes_ao_into_colors() {
        // A floor (y=0) and a wall (x=0) meet at a concave right-angle
        // seam, which produces per-vertex AO < 1 along the inside corner.
        // The exporter must darken those vertices' RGB below the source
        // 1.0 — if AO weren't baked, every R would be exactly 1.0.
        let mut world = World::new();
        for a in 0..4 {
            for b in 0..4 {
                world.set_voxel(a, 0, b, Voxel::from_rgb(255, 0, 0)); // floor
                world.set_voxel(0, a, b, Voxel::from_rgb(255, 0, 0)); // wall
            }
        }
        world.clear_dirty_flags();
        let path = std::env::temp_dir().join("voxelith_ao_bake.glb");
        export_glb(&world, &[], &path).unwrap();

        let (json_bytes, bin) = read_glb(&path);
        let bin = bin.expect("non-empty world must have BIN chunk");
        let json: serde_json::Value = serde_json::from_slice(&json_bytes).unwrap();

        // COLOR_0 is accessor 2 → bufferView 2; read each vertex's R.
        let view = &json["bufferViews"][2];
        let off = view["byteOffset"].as_u64().unwrap() as usize;
        let len = view["byteLength"].as_u64().unwrap() as usize;
        let mut min_r = f32::INFINITY;
        for rgba in bin[off..off + len].chunks_exact(16) {
            let r = f32::from_le_bytes(rgba[0..4].try_into().unwrap());
            min_r = min_r.min(r);
        }
        // Source red is 1.0; an occluded corner must come out darker,
        // but never below the ambient floor.
        assert!(
            min_r < 0.999,
            "expected AO to darken some vertex below R=1.0, got min R={min_r}"
        );
        assert!(
            min_r >= 0.5 - 1e-6,
            "R darkened below the ambient floor: {min_r}"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_export_groups_by_material_into_primitives_and_materials() {
        let mut world = World::new();
        // Plain, emissive, and metallic voxels — spaced apart so each is
        // its own visible region. Three material groups → three primitives.
        world.set_voxel(0, 0, 0, Voxel::from_rgb(200, 200, 200));
        let mut glow = Voxel::from_rgb(0, 100, 255);
        glow.set_emissive(true);
        world.set_voxel(5, 0, 0, glow);
        let mut metal = Voxel::from_rgb(180, 180, 190);
        metal.set_metallic(true);
        world.set_voxel(10, 0, 0, metal);
        world.clear_dirty_flags();

        let path = std::env::temp_dir().join("voxelith_materials.glb");
        export_glb(&world, &[], &path).unwrap();
        let (json_bytes, _) = read_glb(&path);
        let json: serde_json::Value = serde_json::from_slice(&json_bytes).unwrap();

        let mats = json["materials"].as_array().expect("materials present");
        let prims = json["meshes"][0]["primitives"].as_array().unwrap();
        assert_eq!(mats.len(), 3, "plain + emissive + metallic = 3 materials");
        assert_eq!(prims.len(), 3, "one primitive per material group");

        // Every primitive references a material by index.
        for p in prims {
            assert!(p["material"].is_number());
        }
        // Exactly one emissive material (emissiveFactor present).
        let emissive = mats
            .iter()
            .filter(|m| m.get("emissiveFactor").is_some())
            .count();
        assert_eq!(emissive, 1, "one emissive material");
        // Exactly one metallic material (metallicFactor == 1.0).
        let metallic = mats
            .iter()
            .filter(|m| m["pbrMetallicRoughness"]["metallicFactor"].as_f64() == Some(1.0))
            .count();
        assert_eq!(metallic, 1, "one metallic material");
        // The plain material must be non-metallic (fixes the glTF
        // default-metallic trap) and non-emissive.
        let plain = mats.iter().any(|m| {
            m.get("emissiveFactor").is_none()
                && m["pbrMetallicRoughness"]["metallicFactor"].as_f64() == Some(0.0)
        });
        assert!(plain, "plain material must be non-metallic, non-emissive");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_export_writes_tint_zone_attribute() {
        let mut world = World::new();
        // Two same-color plain voxels with different tint zones: same
        // material group (one primitive), kept unmerged by the zone mask
        // key, so the primitive's vertices carry both zones.
        let mut a = Voxel::from_rgb(150, 150, 150);
        a.set_tint_zone(1);
        let mut b = Voxel::from_rgb(150, 150, 150);
        b.set_tint_zone(2);
        world.set_voxel(0, 0, 0, a);
        world.set_voxel(2, 0, 0, b);
        world.clear_dirty_flags();

        let path = std::env::temp_dir().join("voxelith_tintzone.glb");
        export_glb(&world, &[], &path).unwrap();
        let (json_bytes, bin) = read_glb(&path);
        let bin = bin.expect("non-empty world must have BIN chunk");
        let json: serde_json::Value = serde_json::from_slice(&json_bytes).unwrap();

        // The primitive declares a _TINTZONE attribute, SCALAR f32.
        let prim = &json["meshes"][0]["primitives"][0];
        let tz = prim["attributes"]["_TINTZONE"]
            .as_u64()
            .expect("_TINTZONE attribute present") as usize;
        let acc = &json["accessors"][tz];
        assert_eq!(acc["type"], "SCALAR");
        assert_eq!(acc["componentType"], COMPONENT_TYPE_FLOAT);

        // Both zone values (1 and 2) survive to the exported buffer.
        let view = &json["bufferViews"][acc["bufferView"].as_u64().unwrap() as usize];
        let off = view["byteOffset"].as_u64().unwrap() as usize;
        let len = view["byteLength"].as_u64().unwrap() as usize;
        let (mut seen1, mut seen2) = (false, false);
        for z in bin[off..off + len].chunks_exact(4) {
            match f32::from_le_bytes(z.try_into().unwrap()) as i32 {
                1 => seen1 = true,
                2 => seen2 = true,
                _ => {}
            }
        }
        assert!(seen1 && seen2, "both tint zones should be exported");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_export_mirrors_tint_zone_into_texcoord0() {
        // The tint zone is ALSO written into TEXCOORD_0.x (a vec2 UV) so
        // Unity glTFast — which drops custom attributes like _TINTZONE —
        // can still read it. Same two-zone setup as the _TINTZONE test.
        let mut world = World::new();
        let mut a = Voxel::from_rgb(150, 150, 150);
        a.set_tint_zone(1);
        let mut b = Voxel::from_rgb(150, 150, 150);
        b.set_tint_zone(2);
        world.set_voxel(0, 0, 0, a);
        world.set_voxel(2, 0, 0, b);
        world.clear_dirty_flags();

        let path = std::env::temp_dir().join("voxelith_texcoord_zone.glb");
        export_glb(&world, &[], &path).unwrap();
        let (json_bytes, bin) = read_glb(&path);
        let bin = bin.expect("non-empty world must have BIN chunk");
        let json: serde_json::Value = serde_json::from_slice(&json_bytes).unwrap();

        // The primitive declares a TEXCOORD_0 attribute, VEC2 f32.
        let prim = &json["meshes"][0]["primitives"][0];
        let tc = prim["attributes"]["TEXCOORD_0"]
            .as_u64()
            .expect("TEXCOORD_0 attribute present") as usize;
        let acc = &json["accessors"][tc];
        assert_eq!(acc["type"], "VEC2");
        assert_eq!(acc["componentType"], COMPONENT_TYPE_FLOAT);

        // Both zone values survive in the .x of the uv pairs; .y is a 0 pad.
        let view = &json["bufferViews"][acc["bufferView"].as_u64().unwrap() as usize];
        let off = view["byteOffset"].as_u64().unwrap() as usize;
        let len = view["byteLength"].as_u64().unwrap() as usize;
        let (mut seen1, mut seen2) = (false, false);
        for uv in bin[off..off + len].chunks_exact(8) {
            let x = f32::from_le_bytes(uv[0..4].try_into().unwrap());
            let y = f32::from_le_bytes(uv[4..8].try_into().unwrap());
            assert_eq!(y, 0.0, "TEXCOORD_0.y is an unused 0 pad");
            match x as i32 {
                1 => seen1 = true,
                2 => seen2 = true,
                _ => {}
            }
        }
        assert!(seen1 && seen2, "both tint zones should reach TEXCOORD_0.x");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_export_writes_socket_nodes() {
        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgb(200, 200, 200));
        world.clear_dirty_flags();
        let sockets = vec![SocketNode {
            name: "muzzle".to_string(),
            translation: [0.5, 1.0, 0.5],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }];

        let path = std::env::temp_dir().join("voxelith_socket_node.glb");
        export_glb(&world, &sockets, &path).unwrap();
        let (json_bytes, _) = read_glb(&path);
        let json: serde_json::Value = serde_json::from_slice(&json_bytes).unwrap();

        let nodes = json["nodes"].as_array().expect("nodes present");
        // Mesh node (node 0) + one socket node.
        assert_eq!(nodes.len(), 2);
        let socket = nodes
            .iter()
            .find(|n| n["name"] == "muzzle")
            .expect("socket node present");
        // A socket is an EMPTY node — no mesh attached.
        assert!(socket.get("mesh").is_none(), "socket must not carry a mesh");
        assert_eq!(socket["translation"], serde_json::json!([0.5, 1.0, 0.5]));
        assert_eq!(
            socket["rotation"].as_array().map(|a| a.len()),
            Some(4),
            "rotation is a 4-component quaternion"
        );
        // The scene references both the mesh node and the socket node.
        let scene_nodes = json["scenes"][0]["nodes"].as_array().unwrap();
        assert_eq!(scene_nodes.len(), 2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_export_sockets_only_empty_world() {
        // A world with no geometry but a socket still produces a valid
        // glTF: a single empty node, no mesh / BIN chunk.
        let world = World::new();
        let sockets = vec![SocketNode {
            name: "origin".to_string(),
            translation: [1.0, 2.0, 3.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }];

        let path = std::env::temp_dir().join("voxelith_socket_only.glb");
        export_glb(&world, &sockets, &path).unwrap();
        let (json_bytes, bin) = read_glb(&path);
        assert!(bin.is_none(), "no geometry → no BIN chunk");
        let json: serde_json::Value = serde_json::from_slice(&json_bytes).unwrap();

        assert!(json["meshes"].is_null(), "no meshes for a sockets-only export");
        let nodes = json["nodes"].as_array().unwrap();
        assert_eq!(nodes.len(), 1);
        assert!(nodes[0].get("mesh").is_none());
        assert_eq!(nodes[0]["name"], "origin");
        assert_eq!(json["scenes"][0]["nodes"].as_array().unwrap().len(), 1);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_identity_transform_matches_plain_export() {
        // A default (identity) ExportTransform must produce byte-for-byte
        // the same file as the plain export, so a minimal bake reproduces
        // the interactive export exactly.
        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgb(200, 100, 50));
        world.set_voxel(1, 0, 0, Voxel::from_rgb(50, 100, 200));
        world.clear_dirty_flags();

        let p1 = std::env::temp_dir().join("voxelith_xform_plain.glb");
        let p2 = std::env::temp_dir().join("voxelith_xform_identity.glb");
        export_glb(&world, &[], &p1).unwrap();
        export_glb_with_transform(&world, &[], &p2, ExportTransform::default()).unwrap();

        let a = std::fs::read(&p1).unwrap();
        let b = std::fs::read(&p2).unwrap();
        assert_eq!(a, b, "identity transform must not change output bytes");

        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
    }

    #[test]
    fn test_base_center_pivot_wraps_root_node() {
        // A base-center pivot wraps the mesh node under a "Voxelith_root"
        // node whose translation centers the footprint in XZ and puts the
        // bottom (min Y) at the origin. POSITION accessor data is unchanged
        // (lossless — placement lives entirely in the node transform).
        let mut world = World::new();
        // Cells (0..2, 0, 0..2): mesh spans x,z ∈ [0,2], y ∈ [0,1].
        // Base-center pivot = (1, 0, 1).
        for x in 0..2 {
            for z in 0..2 {
                world.set_voxel(x, 0, z, Voxel::from_rgb(180, 180, 180));
            }
        }
        world.clear_dirty_flags();

        let path = std::env::temp_dir().join("voxelith_pivot.glb");
        let t = ExportTransform {
            pivot: Pivot::BaseCenter,
            up_axis: UpAxis::Y,
            unit_scale: 1.0,
        };
        export_glb_with_transform(&world, &[], &path, t).unwrap();

        let (json_bytes, _) = read_glb(&path);
        let json: serde_json::Value = serde_json::from_slice(&json_bytes).unwrap();

        // Scene has a single root: the transform node.
        let scene_nodes = json["scenes"][0]["nodes"].as_array().unwrap();
        assert_eq!(scene_nodes.len(), 1, "one scene root (the transform node)");
        let root_idx = scene_nodes[0].as_u64().unwrap() as usize;
        let root = &json["nodes"][root_idx];
        assert_eq!(root["name"], "Voxelith_root");

        // Translation = -pivot = (-1, 0, -1).
        let tr = root["translation"].as_array().unwrap();
        assert!((tr[0].as_f64().unwrap() + 1.0).abs() < 1e-5);
        assert!(tr[1].as_f64().unwrap().abs() < 1e-5);
        assert!((tr[2].as_f64().unwrap() + 1.0).abs() < 1e-5);

        // The mesh node (index 0) is a child of the root.
        let children = root["children"].as_array().unwrap();
        assert!(children.iter().any(|c| c.as_u64() == Some(0)));

        // POSITION bounds untouched by the pivot (lossless).
        assert_eq!(json["accessors"][0]["min"][0].as_f64().unwrap(), 0.0);
        assert_eq!(json["accessors"][0]["max"][0].as_f64().unwrap(), 2.0);

        let _ = std::fs::remove_file(&path);
    }
}
