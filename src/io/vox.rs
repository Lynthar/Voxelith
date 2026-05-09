//! MagicaVoxel VOX format import/export.
//!
//! VOX is the native format for MagicaVoxel, a popular voxel editor.
//! Supports reading both **v150** (MagicaVoxel 0.97/0.98) and
//! **v200** (0.99.7+) files. Writing always produces **v150** —
//! every MagicaVoxel version reads it, and our `World` data model
//! has no use for v200's scene graph / materials / layers.
//!
//! v200 reading flattens multi-model scene-graph files into the
//! `World`'s single voxel grid: each `nSHP` model is placed at the
//! position determined by the cumulative `nTRN` transform along
//! its scene-tree path. Material / layer / camera / render-object
//! chunks are read and discarded.
//!
//! Format spec:
//! - v150 (basic): <https://github.com/ephtracy/voxel-model/blob/master/MagicaVoxel-file-format-vox.txt>
//! - v200 extension: <https://github.com/ephtracy/voxel-model/blob/master/MagicaVoxel-file-format-vox-extension.txt>

use crate::core::{Voxel, World};
use std::collections::HashMap;
use std::io::{self, Read, Write};
use thiserror::Error;

/// VOX file magic number: "VOX "
const VOX_MAGIC: [u8; 4] = [b'V', b'O', b'X', b' '];
/// Version we write for export. v150 is the universal reader format.
const VOX_VERSION_WRITE: i32 = 150;
/// Versions we accept on read. v150 = basic format, v200 = extended
/// format with scene graph + materials (we read the geometry +
/// transforms, ignore the materials/layers/etc).
const VOX_VERSIONS_SUPPORTED: &[i32] = &[150, 200];

/// Maximum dimension size for VOX format (256)
const MAX_VOX_SIZE: u32 = 256;

/// Errors that can occur when reading/writing VOX files
#[derive(Debug, Error)]
pub enum VoxError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Invalid magic number, expected 'VOX '")]
    InvalidMagic,
    #[error("Unsupported VOX version: {0}")]
    UnsupportedVersion(i32),
    #[error("Invalid chunk ID: {0:?}")]
    InvalidChunkId([u8; 4]),
    #[error("Model too large for VOX format (max 256x256x256)")]
    ModelTooLarge,
    #[error("No voxel data found")]
    NoVoxelData,
    #[error("Invalid palette index: {0}")]
    InvalidPaletteIndex(u8),
}

/// Default MagicaVoxel palette (256 colors)
pub fn default_palette() -> [[u8; 4]; 256] {
    let mut palette = [[0u8; 4]; 256];

    // Initialize with a reasonable default palette
    // First entry is always transparent/empty
    palette[0] = [0, 0, 0, 0];

    // Generate a varied color palette
    for i in 1..256 {
        let idx = i as u8;
        // Create varied colors based on index
        let r = ((idx.wrapping_mul(37)) ^ (idx >> 2)).wrapping_add(idx);
        let g = ((idx.wrapping_mul(73)) ^ (idx >> 3)).wrapping_add(idx.wrapping_mul(2));
        let b = ((idx.wrapping_mul(149)) ^ (idx >> 1)).wrapping_add(idx.wrapping_mul(3));
        palette[i] = [r, g, b, 255];
    }

    // Override with some common colors at the start
    palette[1] = [255, 255, 255, 255]; // White
    palette[2] = [255, 0, 0, 255];     // Red
    palette[3] = [0, 255, 0, 255];     // Green
    palette[4] = [0, 0, 255, 255];     // Blue
    palette[5] = [255, 255, 0, 255];   // Yellow
    palette[6] = [255, 0, 255, 255];   // Magenta
    palette[7] = [0, 255, 255, 255];   // Cyan
    palette[8] = [128, 128, 128, 255]; // Gray
    palette[9] = [255, 128, 0, 255];   // Orange
    palette[10] = [128, 0, 255, 255];  // Purple
    palette[11] = [0, 128, 0, 255];    // Dark green
    palette[12] = [139, 90, 43, 255];  // Brown
    palette[13] = [76, 153, 0, 255];   // Grass green

    palette
}

/// Read a VOX-format STRING (`int32` byte count + raw bytes, no
/// null terminator). Used by v200 dict values.
fn read_vox_string<R: Read>(reader: &mut R) -> io::Result<String> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    let len = i32::from_le_bytes(buf).max(0) as usize;
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Read a VOX-format DICT (`int32` num pairs + N × {STRING key,
/// STRING value}). All values are stored as strings — caller parses
/// numeric ones via `str::parse`.
fn read_vox_dict<R: Read>(reader: &mut R) -> io::Result<HashMap<String, String>> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    let n = i32::from_le_bytes(buf).max(0) as usize;
    let mut out = HashMap::with_capacity(n);
    for _ in 0..n {
        let key = read_vox_string(reader)?;
        let value = read_vox_string(reader)?;
        out.insert(key, value);
    }
    Ok(out)
}

/// Decode the v200 rotation byte into a 3×3 integer rotation matrix.
///
/// Bit layout (per vox-extension spec):
/// - bits 0-1: column index of the non-zero entry in row 1 (0-2)
/// - bits 2-3: column index of the non-zero entry in row 2
/// - bit 4: row 1 sign (0 = +, 1 = -)
/// - bit 5: row 2 sign
/// - bit 6: row 3 sign
///
/// Row 3's column is whichever of {0, 1, 2} isn't claimed by rows
/// 1 and 2 (rotation matrix has exactly one ±1 per row and column).
///
/// `0x04 = 0b00000100` is the identity (row1=col0, row2=col1,
/// row3=col2, all positive).
fn decode_rotation_byte(rot: u8) -> [[i32; 3]; 3] {
    let row1_col = (rot & 0b11) as usize;
    let row2_col = ((rot >> 2) & 0b11) as usize;
    let row1_sign: i32 = if rot & (1 << 4) != 0 { -1 } else { 1 };
    let row2_sign: i32 = if rot & (1 << 5) != 0 { -1 } else { 1 };
    let row3_sign: i32 = if rot & (1 << 6) != 0 { -1 } else { 1 };
    // Row 3 column is the one missing from {row1_col, row2_col}.
    // Defensive fallback for malformed bytes (both rows pointing at
    // the same column): pick column 2 — caller will produce a
    // singular matrix, but at least no panic.
    let row3_col = if row1_col != row2_col {
        3 - row1_col - row2_col
    } else {
        2
    };
    let mut m = [[0i32; 3]; 3];
    if row1_col < 3 {
        m[0][row1_col] = row1_sign;
    }
    if row2_col < 3 {
        m[1][row2_col] = row2_sign;
    }
    if row3_col < 3 {
        m[2][row3_col] = row3_sign;
    }
    m
}

/// Apply a 3×3 integer rotation matrix to a vector. Rotation
/// matrices in this format are signed permutations, so the result
/// is exact integer (no rounding).
fn apply_rotation(m: [[i32; 3]; 3], v: (i32, i32, i32)) -> (i32, i32, i32) {
    let arr = [v.0, v.1, v.2];
    (
        m[0][0] * arr[0] + m[0][1] * arr[1] + m[0][2] * arr[2],
        m[1][0] * arr[0] + m[1][1] * arr[1] + m[1][2] * arr[2],
        m[2][0] * arr[0] + m[2][1] * arr[1] + m[2][2] * arr[2],
    )
}

/// 3×3 matrix multiplication (a × b, applied right-to-left so
/// composing parent × child gives the world-space transform).
fn rotation_compose(a: [[i32; 3]; 3], b: [[i32; 3]; 3]) -> [[i32; 3]; 3] {
    let mut out = [[0i32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

/// Identity rotation matrix.
const ROT_IDENTITY: [[i32; 3]; 3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

/// One node in the v200 scene graph. nTRN (Transform) carries a
/// translation + rotation and a single child id. nGRP groups N
/// child nodes. nSHP references one or more model ids. The graph
/// is a DAG with a single root nTRN at id 0 (per MagicaVoxel
/// convention).
#[derive(Debug, Clone)]
enum SceneNode {
    Transform {
        child_id: i32,
        translation: (i32, i32, i32),
        rotation: [[i32; 3]; 3],
    },
    Group {
        children: Vec<i32>,
    },
    Shape {
        model_ids: Vec<i32>,
    },
}

/// VOX chunk header
struct ChunkHeader {
    id: [u8; 4],
    content_size: i32,
    children_size: i32,
}

impl ChunkHeader {
    fn read<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut id = [0u8; 4];
        reader.read_exact(&mut id)?;

        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        let content_size = i32::from_le_bytes(buf);

        reader.read_exact(&mut buf)?;
        let children_size = i32::from_le_bytes(buf);

        Ok(Self {
            id,
            content_size,
            children_size,
        })
    }

    fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&self.id)?;
        writer.write_all(&self.content_size.to_le_bytes())?;
        writer.write_all(&self.children_size.to_le_bytes())?;
        Ok(())
    }
}

/// One model's geometry within a v200 scene (size + voxel list,
/// no palette — palette is shared at the scene level).
#[derive(Debug, Clone)]
struct VoxModelData {
    size: (u32, u32, u32),
    /// `(x, y, z, palette_index)` — same layout as `VoxModel.voxels`.
    voxels: Vec<(u8, u8, u8, u8)>,
}

/// A whole VOX file's contents: multiple models + palette + scene
/// graph. v150 files produce a `VoxScene` with a single model and
/// no scene graph; v200 files may have many models composed by
/// `nTRN`/`nGRP`/`nSHP` nodes.
///
/// `to_world` flattens the scene graph: each `nSHP`'s models are
/// placed in the world according to the cumulative `nTRN`
/// transform along the path from the root, with each model
/// rotated around its own center.
struct VoxScene {
    models: Vec<VoxModelData>,
    palette: [[u8; 4]; 256],
    nodes: HashMap<i32, SceneNode>,
}

impl VoxScene {
    /// Read a v150 or v200 VOX file. Multi-model + scene graph
    /// are preserved; ignored chunks (`MATL`, `LAYR`, `IMAP`,
    /// `rOBJ`, `rCAM`, `NOTE`, `INFO`, `PACK`, `MATT`) are skipped
    /// by their declared content size.
    pub fn read<R: Read>(reader: &mut R) -> Result<Self, VoxError> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != VOX_MAGIC {
            return Err(VoxError::InvalidMagic);
        }

        let mut version_buf = [0u8; 4];
        reader.read_exact(&mut version_buf)?;
        let version = i32::from_le_bytes(version_buf);
        if !VOX_VERSIONS_SUPPORTED.contains(&version) {
            log::warn!(
                "VOX version {} (supported {:?}), attempting to read anyway",
                version,
                VOX_VERSIONS_SUPPORTED
            );
        }

        let main_header = ChunkHeader::read(reader)?;
        if &main_header.id != b"MAIN" {
            return Err(VoxError::InvalidChunkId(main_header.id));
        }

        let mut models: Vec<VoxModelData> = Vec::new();
        let mut palette = default_palette();
        let mut nodes: HashMap<i32, SceneNode> = HashMap::new();
        // SIZE/XYZI come in pairs; a SIZE chunk publishes a pending
        // size that the next XYZI chunk consumes when writing into
        // a fresh `VoxModelData`.
        let mut pending_size: Option<(u32, u32, u32)> = None;

        let mut bytes_read = 0i32;
        while bytes_read < main_header.children_size {
            let chunk_header = ChunkHeader::read(reader)?;
            bytes_read +=
                12 + chunk_header.content_size + chunk_header.children_size;

            match &chunk_header.id {
                b"SIZE" => {
                    let mut buf = [0u8; 4];
                    reader.read_exact(&mut buf)?;
                    let x = u32::from_le_bytes(buf);
                    reader.read_exact(&mut buf)?;
                    let y = u32::from_le_bytes(buf);
                    reader.read_exact(&mut buf)?;
                    let z = u32::from_le_bytes(buf);
                    pending_size = Some((x, y, z));
                }
                b"XYZI" => {
                    let mut buf = [0u8; 4];
                    reader.read_exact(&mut buf)?;
                    let num_voxels = i32::from_le_bytes(buf) as usize;
                    let mut voxels = Vec::with_capacity(num_voxels);
                    for _ in 0..num_voxels {
                        let mut voxel_data = [0u8; 4];
                        reader.read_exact(&mut voxel_data)?;
                        voxels.push((
                            voxel_data[0],
                            voxel_data[1],
                            voxel_data[2],
                            voxel_data[3],
                        ));
                    }
                    let size = pending_size.take().unwrap_or((1, 1, 1));
                    models.push(VoxModelData { size, voxels });
                }
                b"RGBA" => {
                    for i in 0..256 {
                        let mut color = [0u8; 4];
                        reader.read_exact(&mut color)?;
                        // VOX file index 0..254 maps to palette
                        // 1..255 (palette[0] is reserved as
                        // empty/transparent). File index 255 is
                        // unused.
                        let palette_index = if i == 255 { 0 } else { i + 1 };
                        palette[palette_index] = color;
                    }
                }
                b"nTRN" => {
                    let node = read_ntrn_chunk(reader, chunk_header.content_size)?;
                    if let Some((id, n)) = node {
                        nodes.insert(id, n);
                    }
                }
                b"nGRP" => {
                    let node = read_ngrp_chunk(reader, chunk_header.content_size)?;
                    if let Some((id, n)) = node {
                        nodes.insert(id, n);
                    }
                }
                b"nSHP" => {
                    let node = read_nshp_chunk(reader, chunk_header.content_size)?;
                    if let Some((id, n)) = node {
                        nodes.insert(id, n);
                    }
                }
                _ => {
                    // Skip MATL / LAYR / IMAP / rOBJ / rCAM / NOTE /
                    // INFO / PACK / MATT / unknowns by reading
                    // exactly `content_size` bytes — the spec
                    // guarantees the header tells the truth.
                    let to_skip = chunk_header.content_size.max(0) as usize;
                    let mut skip_buf = vec![0u8; to_skip];
                    reader.read_exact(&mut skip_buf)?;
                }
            }

            // Children section. v200 chunks (nTRN/nGRP/nSHP/MATL/…)
            // all declare children_size = 0; v150 also doesn't use
            // nested chunks under MAIN's children. Skip defensively.
            if chunk_header.children_size > 0 {
                let mut skip_buf =
                    vec![0u8; chunk_header.children_size as usize];
                reader.read_exact(&mut skip_buf)?;
            }
        }

        if models.is_empty() {
            return Err(VoxError::NoVoxelData);
        }

        Ok(Self {
            models,
            palette,
            nodes,
        })
    }

    /// Flatten the scene graph into a `World`. Walks the tree from
    /// the root `nTRN` (MagicaVoxel convention: id 0), accumulates
    /// translation and rotation, and at each `nSHP` places the
    /// referenced model's voxels rotated around the model's center.
    ///
    /// If the scene has no `nTRN` nodes (v150 single-model files
    /// or v200 files we read before the scene graph existed), every
    /// model is placed at the origin — same behavior as the old
    /// single-model reader.
    pub fn to_world(&self) -> World {
        let mut world = World::new();
        if self.nodes.is_empty() || !self.nodes.contains_key(&0) {
            // No scene graph: write model voxels directly into
            // world coords (no center pivot). This matches v150
            // semantics (model voxel `(x, y, z)` → world `(x, y,
            // z)`) so a v150 export → v150 import round-trip is
            // identity. Multi-model v200 files without a scene
            // graph (rare, malformed) get every model overlapped
            // at the origin; users would notice and fix the source.
            for model in &self.models {
                for &(x, y, z, color_idx) in &model.voxels {
                    if color_idx == 0 {
                        continue;
                    }
                    let color = self.palette[color_idx as usize];
                    let voxel =
                        Voxel::from_rgba(color[0], color[1], color[2], color[3]);
                    world.set_voxel(x as i32, y as i32, z as i32, voxel);
                }
            }
            return world;
        }

        // DFS from root id 0.
        let mut visited: std::collections::HashSet<i32> =
            std::collections::HashSet::new();
        self.flatten_node(&mut world, 0, (0, 0, 0), ROT_IDENTITY, &mut visited);
        world
    }

    fn flatten_node(
        &self,
        world: &mut World,
        node_id: i32,
        translation: (i32, i32, i32),
        rotation: [[i32; 3]; 3],
        visited: &mut std::collections::HashSet<i32>,
    ) {
        // Cycle / repeat guard. MagicaVoxel scene graphs are DAGs
        // (often trees), but a malformed file could loop. Visiting
        // each node at most once keeps `to_world` linear.
        if !visited.insert(node_id) {
            return;
        }
        let Some(node) = self.nodes.get(&node_id) else {
            return;
        };
        match node {
            SceneNode::Transform {
                child_id,
                translation: local_t,
                rotation: local_r,
            } => {
                // Apply parent rotation to local translation, then
                // add to parent translation. Rotation composes as
                // parent × local.
                let rotated_t = apply_rotation(rotation, *local_t);
                let new_t = (
                    translation.0 + rotated_t.0,
                    translation.1 + rotated_t.1,
                    translation.2 + rotated_t.2,
                );
                let new_r = rotation_compose(rotation, *local_r);
                self.flatten_node(world, *child_id, new_t, new_r, visited);
            }
            SceneNode::Group { children } => {
                for &child_id in children {
                    self.flatten_node(world, child_id, translation, rotation, visited);
                }
            }
            SceneNode::Shape { model_ids } => {
                for &model_id in model_ids {
                    if let Some(model) =
                        self.models.get(model_id.max(0) as usize)
                    {
                        place_model(world, model, &self.palette, translation, rotation);
                    }
                }
            }
        }
    }
}

/// Place one model into the world at `translation`, rotated by
/// `rotation` around the model's geometric center. Skips palette
/// index 0 (empty/transparent).
fn place_model(
    world: &mut World,
    model: &VoxModelData,
    palette: &[[u8; 4]; 256],
    translation: (i32, i32, i32),
    rotation: [[i32; 3]; 3],
) {
    // Model's center (integer floor — matches MagicaVoxel's pivot
    // for even-sized models; odd sizes still center on the cell
    // closest to the geometric middle).
    let cx = (model.size.0 as i32) / 2;
    let cy = (model.size.1 as i32) / 2;
    let cz = (model.size.2 as i32) / 2;
    for &(x, y, z, color_idx) in &model.voxels {
        if color_idx == 0 {
            continue;
        }
        let local = (x as i32 - cx, y as i32 - cy, z as i32 - cz);
        let rotated = apply_rotation(rotation, local);
        let world_pos = (
            translation.0 + rotated.0,
            translation.1 + rotated.1,
            translation.2 + rotated.2,
        );
        let color = palette[color_idx as usize];
        let voxel = Voxel::from_rgba(color[0], color[1], color[2], color[3]);
        world.set_voxel(world_pos.0, world_pos.1, world_pos.2, voxel);
    }
}

/// Read an `nTRN` chunk's body. Returns `Some((id, node))` on
/// success; `None` on malformed input (caller treats as no-op).
///
/// Layout (per vox-extension spec):
/// - `i32` node id
/// - DICT node attributes (we don't use them)
/// - `i32` child node id
/// - `i32` reserved (== -1)
/// - `i32` layer id
/// - `i32` num frames (≥ 1; we use frame 0)
/// - per frame: DICT with optional `_r` (rotation byte string),
///   `_t` (translation "x y z"), `_f` (frame index)
fn read_ntrn_chunk<R: Read>(
    reader: &mut R,
    _content_size: i32,
) -> Result<Option<(i32, SceneNode)>, VoxError> {
    let mut i32buf = [0u8; 4];
    reader.read_exact(&mut i32buf)?;
    let node_id = i32::from_le_bytes(i32buf);
    let _attrs = read_vox_dict(reader)?;
    reader.read_exact(&mut i32buf)?;
    let child_id = i32::from_le_bytes(i32buf);
    reader.read_exact(&mut i32buf)?;
    let _reserved = i32::from_le_bytes(i32buf);
    reader.read_exact(&mut i32buf)?;
    let _layer_id = i32::from_le_bytes(i32buf);
    reader.read_exact(&mut i32buf)?;
    let num_frames = i32::from_le_bytes(i32buf).max(0);

    // Use the first frame as the static transform; ignore animation
    // (Voxelith has no time axis).
    let mut translation = (0i32, 0i32, 0i32);
    let mut rotation = ROT_IDENTITY;
    for f in 0..num_frames {
        let dict = read_vox_dict(reader)?;
        if f == 0 {
            if let Some(t_str) = dict.get("_t") {
                // "_t" value format: "x y z" — three space-separated ints.
                let parts: Vec<&str> = t_str.split_whitespace().collect();
                if parts.len() == 3 {
                    let x = parts[0].parse::<i32>().unwrap_or(0);
                    let y = parts[1].parse::<i32>().unwrap_or(0);
                    let z = parts[2].parse::<i32>().unwrap_or(0);
                    translation = (x, y, z);
                }
            }
            if let Some(r_str) = dict.get("_r") {
                // "_r" value is a single byte stored as decimal text.
                if let Ok(byte) = r_str.parse::<u8>() {
                    rotation = decode_rotation_byte(byte);
                }
            }
        }
    }

    Ok(Some((
        node_id,
        SceneNode::Transform {
            child_id,
            translation,
            rotation,
        },
    )))
}

/// Read an `nGRP` chunk. Layout: `i32` node id + DICT + `i32` num
/// children + N × `i32` child node ids.
fn read_ngrp_chunk<R: Read>(
    reader: &mut R,
    _content_size: i32,
) -> Result<Option<(i32, SceneNode)>, VoxError> {
    let mut i32buf = [0u8; 4];
    reader.read_exact(&mut i32buf)?;
    let node_id = i32::from_le_bytes(i32buf);
    let _attrs = read_vox_dict(reader)?;
    reader.read_exact(&mut i32buf)?;
    let num_children = i32::from_le_bytes(i32buf).max(0) as usize;
    let mut children = Vec::with_capacity(num_children);
    for _ in 0..num_children {
        reader.read_exact(&mut i32buf)?;
        children.push(i32::from_le_bytes(i32buf));
    }
    Ok(Some((node_id, SceneNode::Group { children })))
}

/// Read an `nSHP` chunk. Layout: `i32` node id + DICT + `i32` num
/// models + N × {`i32` model id, DICT model-attrs}.
fn read_nshp_chunk<R: Read>(
    reader: &mut R,
    _content_size: i32,
) -> Result<Option<(i32, SceneNode)>, VoxError> {
    let mut i32buf = [0u8; 4];
    reader.read_exact(&mut i32buf)?;
    let node_id = i32::from_le_bytes(i32buf);
    let _attrs = read_vox_dict(reader)?;
    reader.read_exact(&mut i32buf)?;
    let num_models = i32::from_le_bytes(i32buf).max(0) as usize;
    let mut model_ids = Vec::with_capacity(num_models);
    for _ in 0..num_models {
        reader.read_exact(&mut i32buf)?;
        model_ids.push(i32::from_le_bytes(i32buf));
        let _model_attrs = read_vox_dict(reader)?;
    }
    Ok(Some((node_id, SceneNode::Shape { model_ids })))
}

/// Voxel data for VOX format
pub struct VoxModel {
    /// Size of the model (x, y, z)
    pub size: (u32, u32, u32),
    /// Voxel positions and palette indices
    pub voxels: Vec<(u8, u8, u8, u8)>, // x, y, z, color_index
    /// Color palette (256 colors, RGBA)
    pub palette: [[u8; 4]; 256],
    /// Number of distinct world colors that didn't fit in the
    /// 255-slot palette and were quantized to the nearest existing
    /// entry. Caller can surface this in the UI so the user knows
    /// the export was lossy. Always 0 for `read`-loaded models.
    pub palette_overflow: u32,
}

impl VoxModel {
    /// Create empty model
    pub fn new(size: (u32, u32, u32)) -> Self {
        Self {
            size,
            voxels: Vec::new(),
            palette: default_palette(),
            palette_overflow: 0,
        }
    }

    /// Create model from world
    pub fn from_world(world: &World) -> Result<Self, VoxError> {
        // Find bounding box of all voxels
        let mut min_x = i32::MAX;
        let mut min_y = i32::MAX;
        let mut min_z = i32::MAX;
        let mut max_x = i32::MIN;
        let mut max_y = i32::MIN;
        let mut max_z = i32::MIN;

        // First pass: find bounds
        for (chunk_pos, chunk_lock) in world.chunks() {
            let chunk = chunk_lock.read();
            let (ox, oy, oz) = chunk_pos.world_origin();

            for (local_pos, _) in chunk.iter_solid() {
                let x = ox + local_pos.x as i32;
                let y = oy + local_pos.y as i32;
                let z = oz + local_pos.z as i32;

                min_x = min_x.min(x);
                min_y = min_y.min(y);
                min_z = min_z.min(z);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
                max_z = max_z.max(z);
            }
        }

        // Handle empty world
        if min_x > max_x {
            return Ok(Self::new((1, 1, 1)));
        }

        // Calculate size
        let size_x = (max_x - min_x + 1) as u32;
        let size_y = (max_y - min_y + 1) as u32;
        let size_z = (max_z - min_z + 1) as u32;

        // Check size limits
        if size_x > MAX_VOX_SIZE || size_y > MAX_VOX_SIZE || size_z > MAX_VOX_SIZE {
            return Err(VoxError::ModelTooLarge);
        }

        // Build color palette from unique colors
        let mut color_to_index: HashMap<[u8; 3], u8> = HashMap::new();
        let mut palette = default_palette();
        let mut next_index = 1u8; // 0 is reserved for empty
        // Distinct colors we had to quantize because the palette filled.
        let mut overflow_colors: std::collections::HashSet<[u8; 3]> =
            std::collections::HashSet::new();

        let mut voxels = Vec::new();

        // Second pass: collect voxels and build palette
        for (chunk_pos, chunk_lock) in world.chunks() {
            let chunk = chunk_lock.read();
            let (ox, oy, oz) = chunk_pos.world_origin();

            for (local_pos, voxel) in chunk.iter_solid() {
                let x = ox + local_pos.x as i32 - min_x;
                let y = oy + local_pos.y as i32 - min_y;
                let z = oz + local_pos.z as i32 - min_z;

                let color = [voxel.r, voxel.g, voxel.b];

                let color_index = if let Some(&idx) = color_to_index.get(&color) {
                    idx
                } else if next_index < 255 {
                    let idx = next_index;
                    color_to_index.insert(color, idx);
                    palette[idx as usize] = [color[0], color[1], color[2], 255];
                    next_index += 1;
                    idx
                } else {
                    // Palette full — quantize to the nearest existing
                    // entry. Track *distinct* lossy colors so the UI
                    // can report something meaningful (multiple voxels
                    // sharing the same lost color count as one).
                    overflow_colors.insert(color);
                    find_closest_color(&palette, color)
                };

                voxels.push((x as u8, y as u8, z as u8, color_index));
            }
        }

        Ok(Self {
            size: (size_x, size_y, size_z),
            voxels,
            palette,
            palette_overflow: overflow_colors.len() as u32,
        })
    }

    /// Convert to world
    pub fn to_world(&self) -> World {
        let mut world = World::new();

        for &(x, y, z, color_index) in &self.voxels {
            if color_index > 0 {
                let color = self.palette[color_index as usize];
                let voxel = Voxel::from_rgba(color[0], color[1], color[2], color[3]);
                world.set_voxel(x as i32, y as i32, z as i32, voxel);
            }
        }

        world
    }

    /// Write to VOX file
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<(), VoxError> {
        // Write header
        writer.write_all(&VOX_MAGIC)?;
        writer.write_all(&VOX_VERSION_WRITE.to_le_bytes())?;

        // Calculate chunk sizes
        let size_content = 12; // 3 x i32
        let xyzi_content = 4 + (self.voxels.len() * 4) as i32; // count + voxels
        let rgba_content = 256 * 4; // 256 colors x 4 bytes

        let children_size =
            12 + size_content +  // SIZE chunk
            12 + xyzi_content +  // XYZI chunk
            12 + rgba_content;   // RGBA chunk

        // Write MAIN chunk header
        ChunkHeader {
            id: *b"MAIN",
            content_size: 0,
            children_size,
        }.write(writer)?;

        // Write SIZE chunk
        ChunkHeader {
            id: *b"SIZE",
            content_size: size_content,
            children_size: 0,
        }.write(writer)?;
        writer.write_all(&(self.size.0 as i32).to_le_bytes())?;
        writer.write_all(&(self.size.1 as i32).to_le_bytes())?;
        writer.write_all(&(self.size.2 as i32).to_le_bytes())?;

        // Write XYZI chunk
        ChunkHeader {
            id: *b"XYZI",
            content_size: xyzi_content,
            children_size: 0,
        }.write(writer)?;
        writer.write_all(&(self.voxels.len() as i32).to_le_bytes())?;
        for &(x, y, z, c) in &self.voxels {
            writer.write_all(&[x, y, z, c])?;
        }

        // Write RGBA chunk
        ChunkHeader {
            id: *b"RGBA",
            content_size: rgba_content,
            children_size: 0,
        }.write(writer)?;
        // VOX format: palette index 1-255 maps to file indices 0-254,
        // file index 255 is unused
        for i in 1..=255 {
            writer.write_all(&self.palette[i])?;
        }
        writer.write_all(&[0, 0, 0, 0])?; // Unused entry

        Ok(())
    }
}

/// Find closest color in palette
fn find_closest_color(palette: &[[u8; 4]; 256], color: [u8; 3]) -> u8 {
    let mut best_index = 1u8;
    let mut best_dist = u32::MAX;

    for i in 1..256 {
        let p = palette[i];
        let dr = (color[0] as i32 - p[0] as i32).abs() as u32;
        let dg = (color[1] as i32 - p[1] as i32).abs() as u32;
        let db = (color[2] as i32 - p[2] as i32).abs() as u32;
        let dist = dr * dr + dg * dg + db * db;

        if dist < best_dist {
            best_dist = dist;
            best_index = i as u8;
        }
    }

    best_index
}

/// Export world to VOX file. Returns the number of distinct world
/// colors that didn't fit in the 255-slot palette and were quantized
/// to the nearest existing entry — 0 means a lossless export.
pub fn export_vox<W: Write>(world: &World, writer: &mut W) -> Result<u32, VoxError> {
    let model = VoxModel::from_world(world)?;
    model.write(writer)?;
    Ok(model.palette_overflow)
}

/// Import world from VOX file. Supports both v150 (single-model)
/// and v200 (multi-model + scene graph) — v200 files are flattened
/// into the unified `World` voxel grid, with each `nSHP`'s models
/// placed at their cumulative `nTRN` transform along the path
/// from the scene root.
pub fn import_vox<R: Read>(reader: &mut R) -> Result<World, VoxError> {
    let scene = VoxScene::read(reader)?;
    Ok(scene.to_world())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgb(255, 0, 0));
        world.set_voxel(1, 0, 0, Voxel::from_rgb(0, 255, 0));
        world.set_voxel(0, 1, 0, Voxel::from_rgb(0, 0, 255));

        let mut buffer = Vec::new();
        let overflow = export_vox(&world, &mut buffer).unwrap();
        assert_eq!(overflow, 0, "3 colors should fit in the 255-slot palette");

        let imported = import_vox(&mut buffer.as_slice()).unwrap();

        assert!(imported.get_voxel(0, 0, 0).is_solid());
        assert!(imported.get_voxel(1, 0, 0).is_solid());
        assert!(imported.get_voxel(0, 1, 0).is_solid());
    }

    // ---- v200 helpers / unit tests ---------------------------------

    fn write_vox_string(buf: &mut Vec<u8>, s: &str) {
        let bytes = s.as_bytes();
        buf.extend_from_slice(&(bytes.len() as i32).to_le_bytes());
        buf.extend_from_slice(bytes);
    }

    fn write_vox_dict(buf: &mut Vec<u8>, pairs: &[(&str, &str)]) {
        buf.extend_from_slice(&(pairs.len() as i32).to_le_bytes());
        for (k, v) in pairs {
            write_vox_string(buf, k);
            write_vox_string(buf, v);
        }
    }

    fn build_ntrn_content(
        node_id: i32,
        child_id: i32,
        translation: (i32, i32, i32),
        rotation_byte: Option<u8>,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&node_id.to_le_bytes());
        write_vox_dict(&mut buf, &[]); // empty attrs
        buf.extend_from_slice(&child_id.to_le_bytes());
        buf.extend_from_slice(&(-1i32).to_le_bytes()); // reserved
        buf.extend_from_slice(&(-1i32).to_le_bytes()); // layer id
        buf.extend_from_slice(&1i32.to_le_bytes()); // num frames
        let t_str = format!("{} {} {}", translation.0, translation.1, translation.2);
        let r_str = rotation_byte.map(|b| b.to_string());
        let mut frame_pairs: Vec<(&str, &str)> = vec![("_t", &t_str)];
        if let Some(ref s) = r_str {
            frame_pairs.push(("_r", s));
        }
        write_vox_dict(&mut buf, &frame_pairs);
        buf
    }

    fn build_ngrp_content(node_id: i32, children: &[i32]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&node_id.to_le_bytes());
        write_vox_dict(&mut buf, &[]);
        buf.extend_from_slice(&(children.len() as i32).to_le_bytes());
        for &c in children {
            buf.extend_from_slice(&c.to_le_bytes());
        }
        buf
    }

    fn build_nshp_content(node_id: i32, model_ids: &[i32]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&node_id.to_le_bytes());
        write_vox_dict(&mut buf, &[]);
        buf.extend_from_slice(&(model_ids.len() as i32).to_le_bytes());
        for &id in model_ids {
            buf.extend_from_slice(&id.to_le_bytes());
            write_vox_dict(&mut buf, &[]); // model attrs
        }
        buf
    }

    fn build_chunk(id: &[u8; 4], content: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(id);
        buf.extend_from_slice(&(content.len() as i32).to_le_bytes());
        buf.extend_from_slice(&0i32.to_le_bytes()); // children_size
        buf.extend_from_slice(content);
        buf
    }

    #[test]
    fn rotation_byte_identity() {
        // 0x04 = 0b00000100 → row1 col 0, row2 col 1, all positive → identity
        assert_eq!(decode_rotation_byte(0x04), ROT_IDENTITY);
    }

    #[test]
    fn rotation_byte_negate_y() {
        // bits: row1 col 0 (00), row2 col 1 (01), row1+ row2- row3+
        // = 0b0010_0100 = 0x24
        let r = decode_rotation_byte(0x24);
        assert_eq!(r, [[1, 0, 0], [0, -1, 0], [0, 0, 1]]);
        assert_eq!(apply_rotation(r, (3, 5, 7)), (3, -5, 7));
    }

    #[test]
    fn rotation_byte_swap_xy() {
        // row1 col 1, row2 col 0, all positive
        // bits 0-1 = 01, bits 2-3 = 00, signs all 0
        // = 0b0000_0001 = 0x01
        let r = decode_rotation_byte(0x01);
        assert_eq!(r, [[0, 1, 0], [1, 0, 0], [0, 0, 1]]);
        assert_eq!(apply_rotation(r, (3, 5, 7)), (5, 3, 7));
    }

    #[test]
    fn rotation_compose_double_swap_is_identity() {
        // Two consecutive 90° X-axis rotations (or any rotation
        // composed with itself twice) should bring identity back
        // for the involutive ones. Simple sanity: identity composed
        // with anything = that thing.
        let r = decode_rotation_byte(0x24);
        let composed = rotation_compose(ROT_IDENTITY, r);
        assert_eq!(composed, r);
    }

    #[test]
    fn v200_ntrn_translation_offsets_single_model() {
        // Minimal v200 file:
        //   model 0: 1×1×1 voxel at (0,0,0) color idx 1
        //   nTRN id=0 (root) → child=1, translation (5, 0, 0)
        //   nGRP id=1 → child=2
        //   nSHP id=2 → model 0
        // Expect: voxel placed at world (5, 0, 0).
        let mut chunks = Vec::new();

        // SIZE
        let mut size = Vec::new();
        size.extend_from_slice(&1u32.to_le_bytes());
        size.extend_from_slice(&1u32.to_le_bytes());
        size.extend_from_slice(&1u32.to_le_bytes());
        chunks.extend_from_slice(&build_chunk(b"SIZE", &size));

        // XYZI
        let mut xyzi = Vec::new();
        xyzi.extend_from_slice(&1i32.to_le_bytes()); // num voxels
        xyzi.extend_from_slice(&[0, 0, 0, 1]); // voxel (0, 0, 0, color_idx=1)
        chunks.extend_from_slice(&build_chunk(b"XYZI", &xyzi));

        // RGBA: index 0 in file = palette index 1 (red)
        let mut rgba = Vec::with_capacity(1024);
        rgba.extend_from_slice(&[255u8, 0, 0, 255]);
        for _ in 0..255 {
            rgba.extend_from_slice(&[0u8, 0, 0, 0]);
        }
        chunks.extend_from_slice(&build_chunk(b"RGBA", &rgba));

        // nTRN id=0, child=1, translation (5, 0, 0)
        chunks.extend_from_slice(&build_chunk(
            b"nTRN",
            &build_ntrn_content(0, 1, (5, 0, 0), None),
        ));
        // nGRP id=1, children=[2]
        chunks.extend_from_slice(&build_chunk(b"nGRP", &build_ngrp_content(1, &[2])));
        // nSHP id=2, model 0
        chunks.extend_from_slice(&build_chunk(b"nSHP", &build_nshp_content(2, &[0])));

        let mut buf = Vec::new();
        buf.extend_from_slice(&VOX_MAGIC);
        buf.extend_from_slice(&200i32.to_le_bytes()); // version 200
        buf.extend_from_slice(b"MAIN");
        buf.extend_from_slice(&0i32.to_le_bytes());
        buf.extend_from_slice(&(chunks.len() as i32).to_le_bytes());
        buf.extend_from_slice(&chunks);

        let world = import_vox(&mut buf.as_slice()).expect("v200 import");
        let v = world.get_voxel(5, 0, 0);
        assert!(v.is_solid(), "expected solid voxel at (5, 0, 0)");
        assert_eq!((v.r, v.g, v.b), (255, 0, 0));
    }

    #[test]
    fn v200_skips_unknown_chunks_safely() {
        // Same as above but with unknown chunks (MATL, LAYR, NOTE)
        // interspersed before the scene graph. Reader must skip
        // them via content_size and not corrupt subsequent chunks.
        let mut chunks = Vec::new();

        // SIZE / XYZI / RGBA (single model)
        let mut size = Vec::new();
        size.extend_from_slice(&1u32.to_le_bytes());
        size.extend_from_slice(&1u32.to_le_bytes());
        size.extend_from_slice(&1u32.to_le_bytes());
        chunks.extend_from_slice(&build_chunk(b"SIZE", &size));
        let mut xyzi = Vec::new();
        xyzi.extend_from_slice(&1i32.to_le_bytes());
        xyzi.extend_from_slice(&[0, 0, 0, 1]);
        chunks.extend_from_slice(&build_chunk(b"XYZI", &xyzi));
        let mut rgba = Vec::with_capacity(1024);
        rgba.extend_from_slice(&[0u8, 255, 0, 255]); // green at idx 1
        for _ in 0..255 {
            rgba.extend_from_slice(&[0u8, 0, 0, 0]);
        }
        chunks.extend_from_slice(&build_chunk(b"RGBA", &rgba));

        // Unknown chunks — fill with arbitrary bytes; reader must
        // skip exactly content_size each.
        chunks.extend_from_slice(&build_chunk(b"MATL", &[0xAB; 32]));
        chunks.extend_from_slice(&build_chunk(b"LAYR", &[0xCD; 16]));
        chunks.extend_from_slice(&build_chunk(b"NOTE", &[0xEF; 8]));
        chunks.extend_from_slice(&build_chunk(b"rOBJ", &[0x12; 64]));

        let mut buf = Vec::new();
        buf.extend_from_slice(&VOX_MAGIC);
        buf.extend_from_slice(&200i32.to_le_bytes());
        buf.extend_from_slice(b"MAIN");
        buf.extend_from_slice(&0i32.to_le_bytes());
        buf.extend_from_slice(&(chunks.len() as i32).to_le_bytes());
        buf.extend_from_slice(&chunks);

        let world = import_vox(&mut buf.as_slice())
            .expect("v200 with unknown chunks should still import");
        // No scene graph in this test, so fallback path: voxel at
        // (0, 0, 0) in world.
        let v = world.get_voxel(0, 0, 0);
        assert!(v.is_solid(), "voxel survived through unknown chunks");
        assert_eq!((v.r, v.g, v.b), (0, 255, 0));
    }

    #[test]
    fn v200_multi_model_with_separate_translations() {
        // Two 1×1×1 models (red + blue), each in its own nSHP,
        // both children of a single nGRP under a root nTRN. Each
        // shape is wrapped in its own nTRN with a different
        // translation.
        //
        //   nTRN id=0 (root, identity) → child=1
        //   nGRP id=1 → children=[2, 4]
        //   nTRN id=2 → child=3, translate (10, 0, 0)
        //   nSHP id=3 → model 0 (red)
        //   nTRN id=4 → child=5, translate (-10, 0, 0)
        //   nSHP id=5 → model 1 (blue)
        let mut chunks = Vec::new();

        // Two models
        for _ in 0..2 {
            let mut size = Vec::new();
            size.extend_from_slice(&1u32.to_le_bytes());
            size.extend_from_slice(&1u32.to_le_bytes());
            size.extend_from_slice(&1u32.to_le_bytes());
            chunks.extend_from_slice(&build_chunk(b"SIZE", &size));
        }
        // First XYZI uses color idx 1 (red), second uses idx 2 (blue)
        for color_idx in [1u8, 2] {
            let mut xyzi = Vec::new();
            xyzi.extend_from_slice(&1i32.to_le_bytes());
            xyzi.extend_from_slice(&[0, 0, 0, color_idx]);
            chunks.extend_from_slice(&build_chunk(b"XYZI", &xyzi));
        }

        // Wait — VOX format requires SIZE / XYZI to interleave per
        // model: SIZE0 XYZI0 SIZE1 XYZI1. Rebuild correctly.
        chunks.clear();
        // SIZE 0 + XYZI 0 (red)
        let mut s0 = Vec::new();
        s0.extend_from_slice(&1u32.to_le_bytes());
        s0.extend_from_slice(&1u32.to_le_bytes());
        s0.extend_from_slice(&1u32.to_le_bytes());
        chunks.extend_from_slice(&build_chunk(b"SIZE", &s0));
        let mut x0 = Vec::new();
        x0.extend_from_slice(&1i32.to_le_bytes());
        x0.extend_from_slice(&[0, 0, 0, 1]);
        chunks.extend_from_slice(&build_chunk(b"XYZI", &x0));
        // SIZE 1 + XYZI 1 (blue)
        let mut s1 = Vec::new();
        s1.extend_from_slice(&1u32.to_le_bytes());
        s1.extend_from_slice(&1u32.to_le_bytes());
        s1.extend_from_slice(&1u32.to_le_bytes());
        chunks.extend_from_slice(&build_chunk(b"SIZE", &s1));
        let mut x1 = Vec::new();
        x1.extend_from_slice(&1i32.to_le_bytes());
        x1.extend_from_slice(&[0, 0, 0, 2]);
        chunks.extend_from_slice(&build_chunk(b"XYZI", &x1));

        // RGBA: idx 1 = red, idx 2 = blue
        let mut rgba = Vec::with_capacity(1024);
        rgba.extend_from_slice(&[255u8, 0, 0, 255]); // file idx 0 → palette 1 (red)
        rgba.extend_from_slice(&[0u8, 0, 255, 255]); // file idx 1 → palette 2 (blue)
        for _ in 0..254 {
            rgba.extend_from_slice(&[0u8, 0, 0, 0]);
        }
        chunks.extend_from_slice(&build_chunk(b"RGBA", &rgba));

        // Scene graph
        chunks.extend_from_slice(&build_chunk(
            b"nTRN",
            &build_ntrn_content(0, 1, (0, 0, 0), None),
        ));
        chunks.extend_from_slice(&build_chunk(b"nGRP", &build_ngrp_content(1, &[2, 4])));
        chunks.extend_from_slice(&build_chunk(
            b"nTRN",
            &build_ntrn_content(2, 3, (10, 0, 0), None),
        ));
        chunks.extend_from_slice(&build_chunk(b"nSHP", &build_nshp_content(3, &[0])));
        chunks.extend_from_slice(&build_chunk(
            b"nTRN",
            &build_ntrn_content(4, 5, (-10, 0, 0), None),
        ));
        chunks.extend_from_slice(&build_chunk(b"nSHP", &build_nshp_content(5, &[1])));

        let mut buf = Vec::new();
        buf.extend_from_slice(&VOX_MAGIC);
        buf.extend_from_slice(&200i32.to_le_bytes());
        buf.extend_from_slice(b"MAIN");
        buf.extend_from_slice(&0i32.to_le_bytes());
        buf.extend_from_slice(&(chunks.len() as i32).to_le_bytes());
        buf.extend_from_slice(&chunks);

        let world = import_vox(&mut buf.as_slice()).expect("multi-model v200");

        // Red voxel translated to (10, 0, 0)
        let red = world.get_voxel(10, 0, 0);
        assert!(red.is_solid(), "red model should be at (10, 0, 0)");
        assert_eq!((red.r, red.g, red.b), (255, 0, 0));

        // Blue voxel translated to (-10, 0, 0)
        let blue = world.get_voxel(-10, 0, 0);
        assert!(blue.is_solid(), "blue model should be at (-10, 0, 0)");
        assert_eq!((blue.r, blue.g, blue.b), (0, 0, 255));
    }

    #[test]
    fn test_palette_overflow_reported() {
        // 256 distinct world colors. VOX palette has 254 usable slots
        // (index 0 is empty/transparent and index 255 is reserved by
        // our writer), so at least 2 distinct colors must be quantized.
        let mut world = World::new();
        for i in 0..256u32 {
            let r = i as u8;
            let g = ((i.wrapping_mul(7)) & 0xFF) as u8;
            let b = ((i.wrapping_mul(13)) & 0xFF) as u8;
            world.set_voxel(
                i as i32 % 16,
                0,
                i as i32 / 16,
                Voxel::from_rgb(r, g, b),
            );
        }
        let mut buffer = Vec::new();
        let overflow = export_vox(&world, &mut buffer).unwrap();
        assert!(
            overflow >= 1,
            "expected at least one overflow color, got {}",
            overflow
        );
    }
}
