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
/// Caps for `Vec`/`HashMap` capacity *hints* taken from file-declared
/// counts. Those counts come from untrusted files; the hint is only a
/// preallocation optimization, so capping it prevents a bogus count
/// from requesting a multi-gigabyte eager allocation. The read loops
/// still consume the full declared count and fail cleanly via
/// `read_exact` on a short stream. Real models / dicts are far smaller.
const MAX_VOXEL_HINT: usize = 1 << 20;
const MAX_DICT_HINT: usize = 256;
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
    #[error("Chunk {0:?} declares an out-of-range size")]
    InvalidChunkSize([u8; 4]),
}

/// Validate a file-declared chunk length. Sizes are `i32` on disk;
/// anything negative means the header is corrupt, and we refuse rather
/// than clamping — a clamped length silently desynchronises every
/// chunk that follows.
fn chunk_len(id: &[u8; 4], size: i32) -> Result<u64, VoxError> {
    u64::try_from(size).map_err(|_| VoxError::InvalidChunkSize(*id))
}

/// Record a parsed scene node, warning on a duplicate id (last one
/// wins, as before — but silently doing so hid malformed files).
fn insert_node(nodes: &mut HashMap<i32, SceneNode>, node: Option<(i32, SceneNode)>) {
    let Some((id, n)) = node else { return };
    if nodes.insert(id, n).is_some() {
        log::warn!("VOX: duplicate scene-graph node id {id}; keeping the last");
    }
}

/// MagicaVoxel's default palette, used when a file omits its `RGBA`
/// chunk (the spec says such a file is colored by this table, and
/// MagicaVoxel writes files that rely on it).
///
/// Transcribed from `default_palette[256]` in the format spec, whose
/// entries are `0xAABBGGRR`. Index order matches ours directly:
/// entry 0 is the reserved empty/transparent slot and 1..=255 are the
/// palette indices a voxel's `color_idx` refers to.
///
/// This used to be a home-grown hash of the index, which meant every
/// `.vox` without an `RGBA` chunk imported in entirely invented colors.
#[rustfmt::skip]
const DEFAULT_PALETTE: [[u8; 4]; 256] = [
    [0,0,0,0], [255,255,255,255], [255,255,204,255], [255,255,153,255],
    [255,255,102,255], [255,255,51,255], [255,255,0,255], [255,204,255,255],
    [255,204,204,255], [255,204,153,255], [255,204,102,255], [255,204,51,255],
    [255,204,0,255], [255,153,255,255], [255,153,204,255], [255,153,153,255],
    [255,153,102,255], [255,153,51,255], [255,153,0,255], [255,102,255,255],
    [255,102,204,255], [255,102,153,255], [255,102,102,255], [255,102,51,255],
    [255,102,0,255], [255,51,255,255], [255,51,204,255], [255,51,153,255],
    [255,51,102,255], [255,51,51,255], [255,51,0,255], [255,0,255,255],
    [255,0,204,255], [255,0,153,255], [255,0,102,255], [255,0,51,255],
    [255,0,0,255], [204,255,255,255], [204,255,204,255], [204,255,153,255],
    [204,255,102,255], [204,255,51,255], [204,255,0,255], [204,204,255,255],
    [204,204,204,255], [204,204,153,255], [204,204,102,255], [204,204,51,255],
    [204,204,0,255], [204,153,255,255], [204,153,204,255], [204,153,153,255],
    [204,153,102,255], [204,153,51,255], [204,153,0,255], [204,102,255,255],
    [204,102,204,255], [204,102,153,255], [204,102,102,255], [204,102,51,255],
    [204,102,0,255], [204,51,255,255], [204,51,204,255], [204,51,153,255],
    [204,51,102,255], [204,51,51,255], [204,51,0,255], [204,0,255,255],
    [204,0,204,255], [204,0,153,255], [204,0,102,255], [204,0,51,255],
    [204,0,0,255], [153,255,255,255], [153,255,204,255], [153,255,153,255],
    [153,255,102,255], [153,255,51,255], [153,255,0,255], [153,204,255,255],
    [153,204,204,255], [153,204,153,255], [153,204,102,255], [153,204,51,255],
    [153,204,0,255], [153,153,255,255], [153,153,204,255], [153,153,153,255],
    [153,153,102,255], [153,153,51,255], [153,153,0,255], [153,102,255,255],
    [153,102,204,255], [153,102,153,255], [153,102,102,255], [153,102,51,255],
    [153,102,0,255], [153,51,255,255], [153,51,204,255], [153,51,153,255],
    [153,51,102,255], [153,51,51,255], [153,51,0,255], [153,0,255,255],
    [153,0,204,255], [153,0,153,255], [153,0,102,255], [153,0,51,255],
    [153,0,0,255], [102,255,255,255], [102,255,204,255], [102,255,153,255],
    [102,255,102,255], [102,255,51,255], [102,255,0,255], [102,204,255,255],
    [102,204,204,255], [102,204,153,255], [102,204,102,255], [102,204,51,255],
    [102,204,0,255], [102,153,255,255], [102,153,204,255], [102,153,153,255],
    [102,153,102,255], [102,153,51,255], [102,153,0,255], [102,102,255,255],
    [102,102,204,255], [102,102,153,255], [102,102,102,255], [102,102,51,255],
    [102,102,0,255], [102,51,255,255], [102,51,204,255], [102,51,153,255],
    [102,51,102,255], [102,51,51,255], [102,51,0,255], [102,0,255,255],
    [102,0,204,255], [102,0,153,255], [102,0,102,255], [102,0,51,255],
    [102,0,0,255], [51,255,255,255], [51,255,204,255], [51,255,153,255],
    [51,255,102,255], [51,255,51,255], [51,255,0,255], [51,204,255,255],
    [51,204,204,255], [51,204,153,255], [51,204,102,255], [51,204,51,255],
    [51,204,0,255], [51,153,255,255], [51,153,204,255], [51,153,153,255],
    [51,153,102,255], [51,153,51,255], [51,153,0,255], [51,102,255,255],
    [51,102,204,255], [51,102,153,255], [51,102,102,255], [51,102,51,255],
    [51,102,0,255], [51,51,255,255], [51,51,204,255], [51,51,153,255],
    [51,51,102,255], [51,51,51,255], [51,51,0,255], [51,0,255,255],
    [51,0,204,255], [51,0,153,255], [51,0,102,255], [51,0,51,255],
    [51,0,0,255], [0,255,255,255], [0,255,204,255], [0,255,153,255],
    [0,255,102,255], [0,255,51,255], [0,255,0,255], [0,204,255,255],
    [0,204,204,255], [0,204,153,255], [0,204,102,255], [0,204,51,255],
    [0,204,0,255], [0,153,255,255], [0,153,204,255], [0,153,153,255],
    [0,153,102,255], [0,153,51,255], [0,153,0,255], [0,102,255,255],
    [0,102,204,255], [0,102,153,255], [0,102,102,255], [0,102,51,255],
    [0,102,0,255], [0,51,255,255], [0,51,204,255], [0,51,153,255],
    [0,51,102,255], [0,51,51,255], [0,51,0,255], [0,0,255,255],
    [0,0,204,255], [0,0,153,255], [0,0,102,255], [0,0,51,255],
    [238,0,0,255], [221,0,0,255], [187,0,0,255], [170,0,0,255],
    [136,0,0,255], [119,0,0,255], [85,0,0,255], [68,0,0,255],
    [34,0,0,255], [17,0,0,255], [0,238,0,255], [0,221,0,255],
    [0,187,0,255], [0,170,0,255], [0,136,0,255], [0,119,0,255],
    [0,85,0,255], [0,68,0,255], [0,34,0,255], [0,17,0,255],
    [0,0,238,255], [0,0,221,255], [0,0,187,255], [0,0,170,255],
    [0,0,136,255], [0,0,119,255], [0,0,85,255], [0,0,68,255],
    [0,0,34,255], [0,0,17,255], [238,238,238,255], [221,221,221,255],
    [187,187,187,255], [170,170,170,255], [136,136,136,255], [119,119,119,255],
    [85,85,85,255], [68,68,68,255], [34,34,34,255], [17,17,17,255],
];

/// Default MagicaVoxel palette (256 colors)
pub fn default_palette() -> [[u8; 4]; 256] {
    DEFAULT_PALETTE
}

/// Build the editor voxel for a palette entry.
///
/// Alpha is forced opaque. MagicaVoxel palettes can carry `a = 0`
/// entries, and passing those through produced a *solid* voxel whose
/// `color()` is `[0,0,0,0]` — identical to air. That broke the flood
/// fill's region test, violated the "every voxel in the world has
/// α = 255" assumption the greedy mesher's zero-key sentinel is built
/// on, and shipped an invisible-in-engine `COLOR_0.a` into every glTF
/// export. Nothing downstream reads voxel alpha, so no information is
/// lost; MagicaVoxel expresses real transparency through `MATL`
/// chunks, which we don't read anyway.
#[inline]
fn voxel_from_palette(color: [u8; 4]) -> Voxel {
    Voxel::from_rgba(color[0], color[1], color[2], 255)
}

/// Read a VOX-format STRING (`int32` byte count + raw bytes, no
/// null terminator). Used by v200 dict values.
fn read_vox_string<R: Read>(reader: &mut R) -> io::Result<String> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    let len = i32::from_le_bytes(buf).max(0) as usize;
    let bytes = super::read_exact_vec(reader, len)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Read a VOX-format DICT (`int32` num pairs + N × {STRING key,
/// STRING value}). All values are stored as strings — caller parses
/// numeric ones via `str::parse`.
fn read_vox_dict<R: Read>(reader: &mut R) -> io::Result<HashMap<String, String>> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    let n = i32::from_le_bytes(buf).max(0) as usize;
    // Cap the capacity hint — `n` is attacker-controlled. Dicts are
    // tiny in practice; the loop reads exactly `n` pairs and fails via
    // read_exact if the stream is short.
    let mut out = HashMap::with_capacity(n.min(MAX_DICT_HINT));
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
///
/// Returns `None` for a byte that doesn't describe a rotation — the
/// two column indices must both be in 0..=2 and must differ. Those
/// bytes exist in the wild only through corruption, but they can't be
/// papered over: `3 - row1 - row2` underflows whenever the columns
/// differ yet one of them is the illegal value 3 (64 of the 256 bytes),
/// which panicked outright in a debug build and produced a degenerate
/// matrix in a release one.
fn decode_rotation_byte(rot: u8) -> Option<[[i32; 3]; 3]> {
    let row1_col = (rot & 0b11) as usize;
    let row2_col = ((rot >> 2) & 0b11) as usize;
    if row1_col > 2 || row2_col > 2 || row1_col == row2_col {
        return None;
    }
    let row1_sign: i32 = if rot & (1 << 4) != 0 { -1 } else { 1 };
    let row2_sign: i32 = if rot & (1 << 5) != 0 { -1 } else { 1 };
    let row3_sign: i32 = if rot & (1 << 6) != 0 { -1 } else { 1 };
    // The remaining column, now provably in 0..=2: a rotation matrix
    // has exactly one ±1 per row and per column.
    let row3_col = 3 - row1_col - row2_col;
    let mut m = [[0i32; 3]; 3];
    m[0][row1_col] = row1_sign;
    m[1][row2_col] = row2_sign;
    m[2][row3_col] = row3_sign;
    Some(m)
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
        /// First model id only — see `read_nshp_chunk`.
        model_id: i32,
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

        // Chunk sizes are file-declared `i32`s: negatives are corrupt,
        // and the running total has to be `u64` + checked so a header
        // near `i32::MAX` can't overflow the accumulator (a debug-build
        // panic, a wrapped byte count in release).
        let total_children = chunk_len(&main_header.id, main_header.children_size)?;
        let mut bytes_read: u64 = 0;
        while bytes_read < total_children {
            let chunk_header = ChunkHeader::read(reader)?;
            let content_size = chunk_len(&chunk_header.id, chunk_header.content_size)?;
            let children_size = chunk_len(&chunk_header.id, chunk_header.children_size)?;
            bytes_read = bytes_read
                .checked_add(12 + content_size)
                .and_then(|n| n.checked_add(children_size))
                .ok_or(VoxError::InvalidChunkSize(chunk_header.id))?;

            // Hand each parser a reader limited to its own body. Two
            // things fall out of that. A parser that reads less than
            // the chunk declares — any chunk carrying fields added by a
            // newer MagicaVoxel — leaves the remainder to be drained
            // below instead of desynchronising the stream and turning
            // the next header into garbage. A parser that tries to read
            // *more* (a self-inconsistent length, e.g. an `XYZI` voxel
            // count that overruns its own chunk) hits EOF inside the
            // `Take` and fails cleanly rather than eating the file.
            {
                let mut body = reader.by_ref().take(content_size);
                match &chunk_header.id {
                    b"SIZE" => {
                        let mut buf = [0u8; 4];
                        body.read_exact(&mut buf)?;
                        let x = u32::from_le_bytes(buf);
                        body.read_exact(&mut buf)?;
                        let y = u32::from_le_bytes(buf);
                        body.read_exact(&mut buf)?;
                        let z = u32::from_le_bytes(buf);
                        if pending_size.is_some() {
                            log::warn!("VOX: SIZE chunk without a matching XYZI");
                        }
                        pending_size = Some((x, y, z));
                    }
                    b"XYZI" => {
                        let mut buf = [0u8; 4];
                        body.read_exact(&mut buf)?;
                        // `.max(0)` first: a negative count would
                        // sign-extend to a colossal usize and make
                        // `with_capacity` abort.
                        let num_voxels = i32::from_le_bytes(buf).max(0) as usize;
                        let mut voxels =
                            Vec::with_capacity(num_voxels.min(MAX_VOXEL_HINT));
                        for _ in 0..num_voxels {
                            let mut voxel_data = [0u8; 4];
                            body.read_exact(&mut voxel_data)?;
                            voxels.push((
                                voxel_data[0],
                                voxel_data[1],
                                voxel_data[2],
                                voxel_data[3],
                            ));
                        }
                        let size = match pending_size.take() {
                            Some(s) => s,
                            None => {
                                // No preceding SIZE. Assume 1³ so the
                                // read still completes, but say so —
                                // the wrong size shifts this model's
                                // rotation pivot.
                                log::warn!(
                                    "VOX: XYZI chunk with no preceding SIZE; \
                                     assuming 1×1×1"
                                );
                                (1, 1, 1)
                            }
                        };
                        models.push(VoxModelData { size, voxels });
                    }
                    b"RGBA" => {
                        for i in 0..256 {
                            let mut color = [0u8; 4];
                            body.read_exact(&mut color)?;
                            // VOX file index 0..254 maps to palette
                            // 1..255. File index 255 is unused, and
                            // palette[0] is our reserved empty slot —
                            // don't let the file overwrite it.
                            if i < 255 {
                                palette[i + 1] = color;
                            }
                        }
                    }
                    b"nTRN" => {
                        insert_node(&mut nodes, read_ntrn_chunk(&mut body)?);
                    }
                    b"nGRP" => {
                        insert_node(&mut nodes, read_ngrp_chunk(&mut body)?);
                    }
                    b"nSHP" => {
                        insert_node(&mut nodes, read_nshp_chunk(&mut body)?);
                    }
                    _ => {
                        // MATL / LAYR / IMAP / rOBJ / rCAM / NOTE /
                        // INFO / PACK / MATT / unknowns: the drain
                        // below streams past them.
                    }
                }
                // Whatever the parser left — trailing fields from a
                // newer format revision, or a whole skipped chunk.
                let unread = body.limit();
                super::skip_bytes(&mut body, unread)?;
            }

            // Children section. v200 chunks (nTRN/nGRP/nSHP/MATL/…)
            // all declare children_size = 0; v150 also doesn't use
            // nested chunks under MAIN's children. Skip defensively.
            super::skip_bytes(reader, children_size)?;
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

    /// Flatten the scene graph into a `World`, then (when `convert_axes`)
    /// rotate the whole model from MagicaVoxel's Z-up convention into
    /// Voxelith's Y-up one. Flattening runs in MagicaVoxel's native space
    /// so the per-node `nTRN` rotations compose correctly; the up-axis
    /// swap is a single global rotation over the finished grid (see
    /// [`mv_to_voxelith`]).
    pub fn to_world(&self, convert_axes: bool) -> World {
        let native = self.to_world_native();
        if convert_axes {
            rotate_world_z_up_to_y_up(&native)
        } else {
            native
        }
    }

    /// Flatten the scene graph into a `World` in MagicaVoxel's native
    /// (Z-up) coordinates. Walks the tree from the root `nTRN`
    /// (MagicaVoxel convention: id 0), accumulates translation and
    /// rotation, and at each `nSHP` places the referenced model's voxels
    /// rotated around the model's center.
    ///
    /// If the scene has no `nTRN` nodes (v150 single-model files
    /// or v200 files we read before the scene graph existed), every
    /// model is placed at the origin — same behavior as the old
    /// single-model reader.
    fn to_world_native(&self) -> World {
        let mut world = World::new();
        if !self.nodes.is_empty() && !self.nodes.contains_key(&0) {
            log::warn!(
                "VOX: scene graph has no root node 0; placing every model \
                 at the origin instead"
            );
        }
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
                    let voxel = voxel_from_palette(self.palette[color_idx as usize]);
                    world.set_voxel(x as i32, y as i32, z as i32, voxel);
                }
            }
            return world;
        }

        // DFS from root id 0.
        let mut limits = FlattenLimits::default();
        let mut path = std::collections::HashSet::new();
        self.flatten_node(
            &mut world,
            0,
            (0, 0, 0),
            ROT_IDENTITY,
            &mut path,
            0,
            &mut limits,
        );
        world
    }

    #[allow(clippy::too_many_arguments)]
    fn flatten_node(
        &self,
        world: &mut World,
        node_id: i32,
        translation: (i32, i32, i32),
        rotation: [[i32; 3]; 3],
        path: &mut std::collections::HashSet<i32>,
        depth: usize,
        limits: &mut FlattenLimits,
    ) {
        if limits.spent() {
            return;
        }
        limits.visits += 1;
        if limits.visits > MAX_SCENE_VISITS {
            limits.stop("scene graph has too many nodes");
            return;
        }
        if depth > MAX_SCENE_DEPTH {
            limits.stop("scene graph is nested too deeply");
            return;
        }
        // Cycle guard, scoped to the CURRENT path. A global
        // visited-once set would also swallow the second and later
        // arrivals at a shared subtree — and a shared subtree is
        // exactly how a DAG says "this part appears in several places",
        // so those files silently lost geometry.
        if !path.insert(node_id) {
            log::warn!("VOX: scene graph cycle through node {node_id}; pruning");
            return;
        }
        if let Some(node) = self.nodes.get(&node_id) {
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
                    self.flatten_node(
                        world,
                        *child_id,
                        new_t,
                        new_r,
                        path,
                        depth + 1,
                        limits,
                    );
                }
                SceneNode::Group { children } => {
                    for &child_id in children {
                        self.flatten_node(
                            world,
                            child_id,
                            translation,
                            rotation,
                            path,
                            depth + 1,
                            limits,
                        );
                    }
                }
                SceneNode::Shape { model_id } => {
                    if let Some(model) = self.models.get(*model_id as usize) {
                        place_model(world, model, &self.palette, translation, rotation);
                        if world.chunk_count() > MAX_SCENE_CHUNKS {
                            limits.stop("scene expands to too much of the world");
                        }
                    } else {
                        log::warn!("VOX: nSHP references missing model {model_id}");
                    }
                }
            }
        }
        path.remove(&node_id);
    }
}

/// Depth cap for the scene-graph walk. Real graphs are a handful of
/// levels deep; this only exists so a hand-crafted chain of `nTRN`
/// nodes can't recurse the main thread into a stack overflow.
const MAX_SCENE_DEPTH: usize = 4096;
/// Cap on total node visits. With DAG sharing restored (see
/// `flatten_node`), a few hundred bytes of nodes can describe
/// exponentially many paths.
const MAX_SCENE_VISITS: u32 = 1 << 20;
/// Cap on chunks the flattened scene may occupy. Every isolated voxel
/// claims a whole 32³ chunk (256 KiB), so a tiny file scattering models
/// across distant translations could otherwise ask for gigabytes.
const MAX_SCENE_CHUNKS: usize = 4096;

/// Budget tracking for `flatten_node`. A `.vox` is untrusted input, and
/// all three limits above are about refusing to be talked into an
/// out-of-memory abort or a stack overflow by a small file.
#[derive(Default)]
struct FlattenLimits {
    visits: u32,
    exhausted: bool,
}

impl FlattenLimits {
    fn spent(&self) -> bool {
        self.exhausted
    }

    /// Abandon the rest of the walk, keeping whatever was placed so
    /// far — a partial model the user can see beats a hard failure or
    /// a hang.
    fn stop(&mut self, why: &str) {
        if !self.exhausted {
            log::warn!("VOX: {why}; import stopped early");
            self.exhausted = true;
        }
    }
}

/// MagicaVoxel stores models Z-up; Voxelith is Y-up. Rotate a MagicaVoxel
/// coordinate into Voxelith space with a right-handed -90° turn about X:
/// `(x, y, z) -> (x, z, -y)`. A rotation (determinant +1), not a mirror,
/// so chirality is preserved — asymmetric models and text don't come out
/// flipped. [`voxelith_to_mv`] is its exact inverse.
#[inline]
fn mv_to_voxelith(p: (i32, i32, i32)) -> (i32, i32, i32) {
    (p.0, p.2, -p.1)
}

/// Inverse of [`mv_to_voxelith`]: Voxelith (Y-up) back to MagicaVoxel
/// (Z-up), a right-handed +90° about X, `(x, y, z) -> (x, -z, y)`.
#[inline]
fn voxelith_to_mv(p: (i32, i32, i32)) -> (i32, i32, i32) {
    (p.0, -p.2, p.1)
}

/// Rotate every solid voxel of `src` from MagicaVoxel's Z-up space into
/// Voxelith's Y-up space (see [`mv_to_voxelith`]), returning a fresh
/// world. Applied once on import after the scene graph is flattened, so a
/// model that stands upright in MagicaVoxel stands upright here.
fn rotate_world_z_up_to_y_up(src: &World) -> World {
    let mut out = World::new();
    for chunk_pos in src.sorted_chunk_positions() {
        let Some(chunk_lock) = src.get_chunk(chunk_pos) else {
            continue;
        };
        let chunk = chunk_lock.read();
        let (ox, oy, oz) = chunk_pos.world_origin();
        for (local_pos, voxel) in chunk.iter_solid() {
            let p = (
                ox + local_pos.x as i32,
                oy + local_pos.y as i32,
                oz + local_pos.z as i32,
            );
            let (nx, ny, nz) = mv_to_voxelith(p);
            out.set_voxel(nx, ny, nz, *voxel);
        }
    }
    out
}

/// Map a model-space cell index through a signed-permutation rotation.
///
/// The subtlety is the sign. A rotation matrix applied to a *point* is
/// just `R · p`, but a voxel is a *cell* spanning `[p, p+1)`, and
/// reflecting that interval about the model's middle sends it to
/// `[size-1-p, size-p)` — so a mirrored axis contributes `size-1-p`,
/// not `-p`. Getting that wrong shifts the whole model one cell along
/// every mirrored axis whenever that axis has an even size (for odd
/// sizes the two formulas coincide, which is why every existing
/// 1×1×1 test passed). Every 90°/180° rotation MagicaVoxel's UI
/// produces has at least one negative row, and its default 40³
/// workspace is even, so real files hit this constantly.
///
/// The pivot (`size / 2`, matching MagicaVoxel's `floor(size/2)`) is
/// subtracted per output axis afterwards; the reflection above already
/// keeps the model inside its own box, so the placement stays
/// box-preserving for both parities.
fn rotate_cell(
    rotation: [[i32; 3]; 3],
    p: [i32; 3],
    size: [i32; 3],
) -> (i32, i32, i32) {
    let mut out = [0i32; 3];
    for (row, out_axis) in out.iter_mut().enumerate() {
        // Exactly one column per row is non-zero (validated in
        // `decode_rotation_byte`); a zero row can only come from
        // `ROT_IDENTITY`-shaped data we built ourselves.
        let Some(col) = (0..3).find(|&c| rotation[row][c] != 0) else {
            continue;
        };
        let src = if rotation[row][col] < 0 {
            size[col] - 1 - p[col]
        } else {
            p[col]
        };
        *out_axis = src - size[col] / 2;
    }
    (out[0], out[1], out[2])
}

/// Place one model into the world at `translation`, rotated by
/// `rotation` around the model's pivot. Skips palette index 0
/// (empty/transparent).
fn place_model(
    world: &mut World,
    model: &VoxModelData,
    palette: &[[u8; 4]; 256],
    translation: (i32, i32, i32),
    rotation: [[i32; 3]; 3],
) {
    let size = [
        model.size.0 as i32,
        model.size.1 as i32,
        model.size.2 as i32,
    ];
    for &(x, y, z, color_idx) in &model.voxels {
        if color_idx == 0 {
            continue;
        }
        let rotated = rotate_cell(rotation, [x as i32, y as i32, z as i32], size);
        let world_pos = (
            translation.0 + rotated.0,
            translation.1 + rotated.1,
            translation.2 + rotated.2,
        );
        let voxel = voxel_from_palette(palette[color_idx as usize]);
        world.set_voxel(world_pos.0, world_pos.1, world_pos.2, voxel);
    }
}

/// Parse an `nTRN` frame's `_t` value ("x y z"). All three components
/// must be present and valid, or the translation is rejected whole.
fn parse_translation(s: &str) -> Option<(i32, i32, i32)> {
    let mut parts = s.split_whitespace();
    let x = parts.next()?.parse::<i32>().ok()?;
    let y = parts.next()?.parse::<i32>().ok()?;
    let z = parts.next()?.parse::<i32>().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((x, y, z))
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
                // "_t" value format: "x y z" — three space-separated
                // ints, taken all-or-nothing. Defaulting a single bad
                // component to 0 would silently drag the subtree part
                // of the way to the origin, which is harder to notice
                // than it landing at the origin outright.
                match parse_translation(t_str) {
                    Some(t) => translation = t,
                    None => log::warn!(
                        "VOX: nTRN {node_id} has malformed _t {t_str:?}; \
                         placing at the origin"
                    ),
                }
            }
            if let Some(r_str) = dict.get("_r") {
                // "_r" value is a single byte stored as decimal text.
                match r_str.parse::<u8>().ok().and_then(decode_rotation_byte) {
                    Some(m) => rotation = m,
                    None => log::warn!(
                        "VOX: nTRN {node_id} has malformed _r {r_str:?}; \
                         using no rotation"
                    ),
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
) -> Result<Option<(i32, SceneNode)>, VoxError> {
    let mut i32buf = [0u8; 4];
    reader.read_exact(&mut i32buf)?;
    let node_id = i32::from_le_bytes(i32buf);
    let _attrs = read_vox_dict(reader)?;
    reader.read_exact(&mut i32buf)?;
    let num_children = i32::from_le_bytes(i32buf).max(0) as usize;
    // Cap the hint — count is untrusted; 64K node ids already dwarfs any
    // real scene graph. The loop still reads the full count.
    let mut children = Vec::with_capacity(num_children.min(1 << 16));
    for _ in 0..num_children {
        reader.read_exact(&mut i32buf)?;
        children.push(i32::from_le_bytes(i32buf));
    }
    Ok(Some((node_id, SceneNode::Group { children })))
}

/// Read an `nSHP` chunk. Layout: `i32` node id + DICT + `i32` num
/// models + N × {`i32` model id, DICT model-attrs}.
///
/// Several model ids means an animation (0.99.7+ keyframes), not
/// several parts. Only the first is kept — placing all of them stacked
/// every frame of an animated `.vox` into one blob, which is not what
/// the file says. Same "first frame wins" rule as `read_ntrn_chunk`.
fn read_nshp_chunk<R: Read>(
    reader: &mut R,
) -> Result<Option<(i32, SceneNode)>, VoxError> {
    let mut i32buf = [0u8; 4];
    reader.read_exact(&mut i32buf)?;
    let node_id = i32::from_le_bytes(i32buf);
    let _attrs = read_vox_dict(reader)?;
    reader.read_exact(&mut i32buf)?;
    let num_models = i32::from_le_bytes(i32buf).max(0) as usize;
    // Cap the hint — count is untrusted (same rationale as nGRP).
    let mut first: Option<i32> = None;
    for _ in 0..num_models {
        reader.read_exact(&mut i32buf)?;
        let id = i32::from_le_bytes(i32buf);
        // Read every model's attribute dict regardless — they're part
        // of this chunk's byte stream.
        let _model_attrs = read_vox_dict(reader)?;
        first.get_or_insert(id);
    }
    let Some(model_id) = first else {
        log::warn!("VOX: nSHP {node_id} references no models");
        return Ok(None);
    };
    if model_id < 0 {
        log::warn!("VOX: nSHP {node_id} has negative model id {model_id}; skipping");
        return Ok(None);
    }
    Ok(Some((node_id, SceneNode::Shape { model_id })))
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
    /// 254-slot palette and were quantized to the nearest existing
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
    pub fn from_world(world: &World, convert_axes: bool) -> Result<Self, VoxError> {
        // Find bounding box of all voxels
        let mut min_x = i32::MAX;
        let mut min_y = i32::MAX;
        let mut min_z = i32::MAX;
        let mut max_x = i32::MIN;
        let mut max_y = i32::MIN;
        let mut max_z = i32::MIN;

        // First pass: find bounds
        for chunk_pos in world.sorted_chunk_positions() {
            let Some(chunk_lock) = world.get_chunk(chunk_pos) else {
                continue;
            };
            let chunk = chunk_lock.read();
            let (ox, oy, oz) = chunk_pos.world_origin();

            for (local_pos, _) in chunk.iter_solid() {
                let raw = (
                    ox + local_pos.x as i32,
                    oy + local_pos.y as i32,
                    oz + local_pos.z as i32,
                );
                let (x, y, z) = if convert_axes { voxelith_to_mv(raw) } else { raw };

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
        for chunk_pos in world.sorted_chunk_positions() {
            let Some(chunk_lock) = world.get_chunk(chunk_pos) else {
                continue;
            };
            let chunk = chunk_lock.read();
            let (ox, oy, oz) = chunk_pos.world_origin();

            for (local_pos, voxel) in chunk.iter_solid() {
                let raw = (
                    ox + local_pos.x as i32,
                    oy + local_pos.y as i32,
                    oz + local_pos.z as i32,
                );
                let mv = if convert_axes { voxelith_to_mv(raw) } else { raw };
                let x = mv.0 - min_x;
                let y = mv.1 - min_y;
                let z = mv.2 - min_z;

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
                    // Only slots we actually filled are candidates.
                    // Searching the whole 255 could snap a voxel to a
                    // leftover default-palette color that appears
                    // nowhere in this model — and slot 255 in
                    // particular never gets written (`next_index < 255`)
                    // yet still ships in the file.
                    find_closest_color(&palette, color, next_index)
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
                let voxel = voxel_from_palette(self.palette[color_index as usize]);
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

/// Nearest palette entry to `color`, searching only slots `1..end`
/// (the ones the export has actually assigned).
fn find_closest_color(palette: &[[u8; 4]; 256], color: [u8; 3], end: u8) -> u8 {
    let mut best_index = 1u8;
    let mut best_dist = u32::MAX;

    for i in 1..(end as usize).max(2) {
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
/// colors that didn't fit in the 254-slot palette and were quantized
/// to the nearest existing entry — 0 means a lossless export.
///
/// When `convert_axes` is set the world is rotated from Voxelith's Y-up
/// convention into MagicaVoxel's Z-up one (the inverse of the import
/// conversion) so the exported model opens upright in MagicaVoxel; pass
/// `false` to write voxel coordinates through verbatim.
pub fn export_vox<W: Write>(
    world: &World,
    writer: &mut W,
    convert_axes: bool,
) -> Result<u32, VoxError> {
    let model = VoxModel::from_world(world, convert_axes)?;
    model.write(writer)?;
    Ok(model.palette_overflow)
}

/// Import world from VOX file. Supports both v150 (single-model)
/// and v200 (multi-model + scene graph) — v200 files are flattened
/// into the unified `World` voxel grid, with each `nSHP`'s models
/// placed at their cumulative `nTRN` transform along the path
/// from the scene root.
///
/// When `convert_axes` is set (the interactive default) the flattened
/// model is rotated from MagicaVoxel's Z-up convention into Voxelith's
/// Y-up one, so a model that stands upright in MagicaVoxel stands upright
/// here. Pass `false` to read coordinates through verbatim (e.g. for a
/// file already authored Y-up).
pub fn import_vox<R: Read>(reader: &mut R, convert_axes: bool) -> Result<World, VoxError> {
    let scene = VoxScene::read(reader)?;
    Ok(scene.to_world(convert_axes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_roundtrip() {
        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgb(255, 0, 0));
        world.set_voxel(1, 0, 0, Voxel::from_rgb(0, 255, 0));
        world.set_voxel(0, 1, 0, Voxel::from_rgb(0, 0, 255));

        let mut buffer = Vec::new();
        let overflow = export_vox(&world, &mut buffer, false).unwrap();
        assert_eq!(overflow, 0, "3 colors should fit in the 254-slot palette");

        let imported = import_vox(&mut buffer.as_slice(), false).unwrap();

        assert!(imported.get_voxel(0, 0, 0).is_solid());
        assert!(imported.get_voxel(1, 0, 0).is_solid());
        assert!(imported.get_voxel(0, 1, 0).is_solid());
    }

    #[test]
    fn axis_helpers_are_inverse_and_map_up_correctly() {
        // MagicaVoxel up is +Z; Voxelith up is +Y.
        assert_eq!(mv_to_voxelith((0, 0, 1)), (0, 1, 0)); // MV +Z (up) -> Voxelith +Y (up)
        assert_eq!(voxelith_to_mv((0, 1, 0)), (0, 0, 1)); // and back
        // Exact inverses in both directions, for arbitrary points.
        for p in [(1, 2, 3), (-4, 5, -6), (0, 0, 0), (7, -8, 9)] {
            assert_eq!(mv_to_voxelith(voxelith_to_mv(p)), p);
            assert_eq!(voxelith_to_mv(mv_to_voxelith(p)), p);
        }
    }

    #[test]
    fn vox_axis_conversion_roundtrips_and_preserves_up() {
        // A vertical pair (one voxel two cells above another) exercises
        // orientation: exported with conversion it stands along MagicaVoxel's
        // Z axis, and re-importing with conversion must bring the "up" voxel
        // back *above* the base, not beside it.
        let mut world = World::new();
        let base = Voxel::from_rgb(200, 50, 50);
        let up = Voxel::from_rgb(50, 200, 50);
        world.set_voxel(0, 0, 0, base);
        world.set_voxel(0, 2, 0, up); // two cells up in Voxelith (+Y)

        let mut buffer = Vec::new();
        export_vox(&world, &mut buffer, true).unwrap();
        let imported = import_vox(&mut buffer.as_slice(), true).unwrap();

        // Same colors, same vertical relationship preserved.
        assert_eq!(imported.get_voxel(0, 0, 0).color(), base.color());
        assert_eq!(imported.get_voxel(0, 2, 0).color(), up.color());
        // Nothing leaked sideways where a wrong axis would have placed it.
        assert!(imported.get_voxel(0, 0, 2).is_air());
        assert!(imported.get_voxel(2, 0, 0).is_air());
    }

    #[test]
    fn vox_roundtrip_without_conversion_keeps_axes() {
        // Conversion off: a Y-up column stays a Y-column across the
        // round-trip (verbatim coordinates, no rotation).
        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgb(10, 20, 30));
        world.set_voxel(0, 3, 0, Voxel::from_rgb(40, 50, 60));

        let mut buffer = Vec::new();
        export_vox(&world, &mut buffer, false).unwrap();
        let imported = import_vox(&mut buffer.as_slice(), false).unwrap();

        assert!(imported.get_voxel(0, 0, 0).is_solid());
        assert!(imported.get_voxel(0, 3, 0).is_solid());
        assert!(imported.get_voxel(0, 0, 3).is_air());
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

    /// Total solid voxels across the world.
    fn solid_voxels(world: &World) -> u32 {
        world
            .chunks()
            .map(|(_, c)| c.read().solid_count())
            .sum()
    }

    /// Wrap chunk bytes in a v200 `MAIN` container.
    fn build_v200_file(chunks: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&VOX_MAGIC);
        buf.extend_from_slice(&200i32.to_le_bytes());
        buf.extend_from_slice(b"MAIN");
        buf.extend_from_slice(&0i32.to_le_bytes());
        buf.extend_from_slice(&(chunks.len() as i32).to_le_bytes());
        buf.extend_from_slice(chunks);
        buf
    }

    /// SIZE + XYZI for a model of `size` holding `voxels`
    /// (`(x, y, z, color_idx)`).
    fn build_model(size: (u32, u32, u32), voxels: &[(u8, u8, u8, u8)]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut size_buf = Vec::new();
        size_buf.extend_from_slice(&size.0.to_le_bytes());
        size_buf.extend_from_slice(&size.1.to_le_bytes());
        size_buf.extend_from_slice(&size.2.to_le_bytes());
        out.extend_from_slice(&build_chunk(b"SIZE", &size_buf));
        let mut xyzi = Vec::new();
        xyzi.extend_from_slice(&(voxels.len() as i32).to_le_bytes());
        for &(x, y, z, c) in voxels {
            xyzi.extend_from_slice(&[x, y, z, c]);
        }
        out.extend_from_slice(&build_chunk(b"XYZI", &xyzi));
        out
    }

    /// SIZE + XYZI for a 1³ model holding one voxel of `color_idx`.
    fn build_unit_model(color_idx: u8) -> Vec<u8> {
        build_model((1, 1, 1), &[(0, 0, 0, color_idx)])
    }

    /// A v200 file placing `model` under a root nTRN carrying
    /// `translation` and `rotation_byte`.
    fn build_v200_placed(
        model: &[u8],
        translation: (i32, i32, i32),
        rotation_byte: Option<u8>,
    ) -> Vec<u8> {
        let mut chunks = model.to_vec();
        chunks.extend_from_slice(&build_rgba_chunk([255, 0, 0, 255]));
        chunks.extend_from_slice(&build_chunk(
            b"nTRN",
            &build_ntrn_content(0, 1, translation, rotation_byte),
        ));
        chunks.extend_from_slice(&build_chunk(b"nGRP", &build_ngrp_content(1, &[2])));
        chunks.extend_from_slice(&build_chunk(b"nSHP", &build_nshp_content(2, &[0])));
        build_v200_file(&chunks)
    }

    /// Every cell of a filled `size` box, run through `rotate_cell`.
    fn rotated_box(rotation: [[i32; 3]; 3], size: [i32; 3]) -> HashSet<(i32, i32, i32)> {
        let mut out = HashSet::new();
        for x in 0..size[0] {
            for y in 0..size[1] {
                for z in 0..size[2] {
                    out.insert(rotate_cell(rotation, [x, y, z], size));
                }
            }
        }
        out
    }

    #[test]
    fn rotation_keeps_a_model_inside_its_own_box() {
        // The invariant that pins the cell-vs-point distinction: a
        // rotation permutes the model's box onto itself, so a filled
        // box must come out as the same filled box with axes permuted
        // — never shifted. Treating a voxel as a point instead of a
        // cell slid every mirrored *even*-length axis by one, which
        // this catches for size 2 and 4 while leaving the odd sizes
        // (where both formulas agree) as a control.
        for size in [[2, 3, 4], [1, 1, 1], [4, 4, 4], [3, 3, 3]] {
            for byte in 0u8..=255 {
                let Some(m) = decode_rotation_byte(byte) else {
                    continue;
                };
                let cells = rotated_box(m, size);
                // Expected box: output axis `row` spans the source
                // axis it draws from, offset by that axis's pivot.
                let mut expected = HashSet::new();
                let col: Vec<usize> = (0..3)
                    .map(|r| (0..3).find(|&c| m[r][c] != 0).unwrap())
                    .collect();
                for a in 0..size[col[0]] {
                    for b in 0..size[col[1]] {
                        for c in 0..size[col[2]] {
                            expected.insert((
                                a - size[col[0]] / 2,
                                b - size[col[1]] / 2,
                                c - size[col[2]] / 2,
                            ));
                        }
                    }
                }
                assert_eq!(
                    cells, expected,
                    "size {size:?} rotation {byte:#04x} left its box"
                );
            }
        }
    }

    #[test]
    fn v200_mirrored_axis_places_even_and_odd_models_consistently() {
        // 0x14 = row1←col0 negated, row2←col1, row3←col2: a plain
        // mirror along X, the shape every 90°/180° MagicaVoxel
        // rotation contains at least one of.
        const MIRROR_X: u8 = 0x14;
        assert_eq!(
            decode_rotation_byte(MIRROR_X),
            Some([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        );

        // Even length: the model is 2 wide with only its low cell
        // filled. Mirroring must land that cell on the box's high
        // side, i.e. at pivot-relative 0 → world (10, 0, 0).
        let even = build_v200_placed(
            &build_model((2, 1, 1), &[(0, 0, 0, 1)]),
            (10, 0, 0),
            Some(MIRROR_X),
        );
        let world = import_vox(&mut even.as_slice(), false).expect("import");
        assert!(
            world.get_voxel(10, 0, 0).is_solid(),
            "even-sized mirror landed off by one"
        );
        assert_eq!(solid_voxels(&world), 1);

        // Odd length: both the old and the corrected formula agree
        // here, so this pins that the fix didn't move odd models.
        let odd = build_v200_placed(
            &build_model((3, 1, 1), &[(0, 0, 0, 1)]),
            (10, 0, 0),
            Some(MIRROR_X),
        );
        let world = import_vox(&mut odd.as_slice(), false).expect("import");
        assert!(world.get_voxel(11, 0, 0).is_solid(), "odd-sized mirror moved");
        assert_eq!(solid_voxels(&world), 1);
    }

    #[test]
    fn v200_unrotated_placement_is_unchanged() {
        // Control for the two above: with no rotation, a cell sits at
        // `translation + p - pivot` regardless of parity.
        let even = build_v200_placed(
            &build_model((2, 1, 1), &[(0, 0, 0, 1)]),
            (10, 0, 0),
            None,
        );
        let world = import_vox(&mut even.as_slice(), false).expect("import");
        assert!(world.get_voxel(9, 0, 0).is_solid());
    }

    /// An `RGBA` chunk whose file entry 0 (palette index 1) is
    /// `color`; everything else is zeroed.
    fn build_rgba_chunk(color: [u8; 4]) -> Vec<u8> {
        let mut rgba = Vec::with_capacity(1024);
        rgba.extend_from_slice(&color);
        for _ in 0..255 {
            rgba.extend_from_slice(&[0u8; 4]);
        }
        build_chunk(b"RGBA", &rgba)
    }

    #[test]
    fn v200_places_a_shared_subtree_once_per_parent() {
        // The scene graph is a DAG: one part referenced from two
        // transforms means that part appears twice. A global
        // visited-once set silently dropped every appearance after the
        // first.
        let mut chunks = build_unit_model(1);
        chunks.extend_from_slice(&build_rgba_chunk([255, 0, 0, 255]));
        // root nTRN 0 → nGRP 1 → { nTRN 2, nTRN 3 } → both to nSHP 10.
        chunks.extend_from_slice(&build_chunk(
            b"nTRN",
            &build_ntrn_content(0, 1, (0, 0, 0), None),
        ));
        chunks
            .extend_from_slice(&build_chunk(b"nGRP", &build_ngrp_content(1, &[2, 3])));
        chunks.extend_from_slice(&build_chunk(
            b"nTRN",
            &build_ntrn_content(2, 10, (0, 0, 0), None),
        ));
        chunks.extend_from_slice(&build_chunk(
            b"nTRN",
            &build_ntrn_content(3, 10, (5, 0, 0), None),
        ));
        chunks.extend_from_slice(&build_chunk(b"nSHP", &build_nshp_content(10, &[0])));

        let world = import_vox(&mut build_v200_file(&chunks).as_slice(), false)
            .expect("v200 import");
        assert!(world.get_voxel(0, 0, 0).is_solid(), "first instance missing");
        assert!(world.get_voxel(5, 0, 0).is_solid(), "second instance missing");
    }

    #[test]
    fn v200_nshp_places_only_the_first_model() {
        // Several model ids in one nSHP is a keyframe animation, not
        // several parts — placing them all stacked every frame into one
        // blob.
        let mut chunks = build_unit_model(1);
        chunks.extend_from_slice(&build_unit_model(1));
        chunks.extend_from_slice(&build_rgba_chunk([255, 0, 0, 255]));
        chunks.extend_from_slice(&build_chunk(
            b"nTRN",
            &build_ntrn_content(0, 1, (0, 0, 0), None),
        ));
        chunks.extend_from_slice(&build_chunk(b"nGRP", &build_ngrp_content(1, &[2])));
        chunks
            .extend_from_slice(&build_chunk(b"nSHP", &build_nshp_content(2, &[0, 1])));

        let world = import_vox(&mut build_v200_file(&chunks).as_slice(), false)
            .expect("v200 import");
        // Both models are 1³ at the origin, so "only the first" shows
        // up as exactly one solid voxel in the world.
        assert_eq!(solid_voxels(&world), 1);
    }

    #[test]
    fn v200_negative_model_id_is_skipped() {
        // `-1` used to be clamped to 0, placing an unrelated model
        // instead of skipping the corrupt reference.
        let mut chunks = build_unit_model(1);
        chunks.extend_from_slice(&build_rgba_chunk([255, 0, 0, 255]));
        chunks.extend_from_slice(&build_chunk(
            b"nTRN",
            &build_ntrn_content(0, 1, (0, 0, 0), None),
        ));
        chunks.extend_from_slice(&build_chunk(b"nGRP", &build_ngrp_content(1, &[2])));
        chunks.extend_from_slice(&build_chunk(b"nSHP", &build_nshp_content(2, &[-1])));

        let world = import_vox(&mut build_v200_file(&chunks).as_slice(), false)
            .expect("v200 import");
        assert_eq!(solid_voxels(&world), 0);
    }

    #[test]
    fn chunk_with_trailing_fields_does_not_desync_the_stream() {
        // A newer MagicaVoxel may append fields to a chunk we already
        // know. Reading by our own idea of the layout instead of the
        // chunk's declared length left the stream mid-chunk, and every
        // header after it was garbage.
        let mut chunks = build_unit_model(1);
        chunks.extend_from_slice(&build_rgba_chunk([255, 0, 0, 255]));
        let mut ntrn = build_ntrn_content(0, 1, (5, 0, 0), None);
        ntrn.extend_from_slice(&[0xAB, 0xCD, 0xEF, 0x01]); // future field
        chunks.extend_from_slice(&build_chunk(b"nTRN", &ntrn));
        chunks.extend_from_slice(&build_chunk(b"nGRP", &build_ngrp_content(1, &[2])));
        chunks.extend_from_slice(&build_chunk(b"nSHP", &build_nshp_content(2, &[0])));

        let world = import_vox(&mut build_v200_file(&chunks).as_slice(), false)
            .expect("import must survive an unknown trailing field");
        assert!(
            world.get_voxel(5, 0, 0).is_solid(),
            "the chunks after the padded one must still parse"
        );
    }

    #[test]
    fn corrupt_chunk_lengths_error_instead_of_panicking() {
        let model = build_unit_model(1);

        // Negative content size: refuse rather than clamp, since a
        // clamped length desynchronises everything that follows.
        let mut bad = Vec::new();
        bad.extend_from_slice(b"XYZI");
        bad.extend_from_slice(&(-1i32).to_le_bytes());
        bad.extend_from_slice(&0i32.to_le_bytes());
        assert!(matches!(
            import_vox(&mut build_v200_file(&bad).as_slice(), false),
            Err(VoxError::InvalidChunkSize(_))
        ));

        // A size near i32::MAX must not overflow the running byte
        // count (a debug-build panic before) — it just runs out of data.
        let mut huge = model.clone();
        huge.extend_from_slice(b"MATL");
        huge.extend_from_slice(&i32::MAX.to_le_bytes());
        huge.extend_from_slice(&i32::MAX.to_le_bytes());
        assert!(import_vox(&mut build_v200_file(&huge).as_slice(), false).is_err());

        // An XYZI claiming more voxels than its own chunk holds must
        // fail cleanly instead of reading into the next chunk.
        let mut lying = Vec::new();
        lying.extend_from_slice(&1000i32.to_le_bytes());
        lying.extend_from_slice(&[0, 0, 0, 1]);
        let mut chunks = Vec::new();
        chunks.extend_from_slice(&build_chunk(b"XYZI", &lying));
        assert!(import_vox(&mut build_v200_file(&chunks).as_slice(), false).is_err());
    }

    #[test]
    fn palette_alpha_is_forced_opaque_on_import() {
        // A transparent palette entry would otherwise produce a solid
        // voxel whose color() is [0,0,0,0] — indistinguishable from air
        // to the flood fill and the greedy mesher's zero-key sentinel.
        let mut chunks = build_unit_model(1);
        chunks.extend_from_slice(&build_rgba_chunk([10, 20, 30, 0]));
        let world = import_vox(&mut build_v200_file(&chunks).as_slice(), false)
            .expect("v150-style import");
        let v = world.get_voxel(0, 0, 0);
        assert!(v.is_solid());
        assert_eq!(v.color(), [10, 20, 30, 255]);
    }

    #[test]
    fn rotation_byte_identity() {
        // 0x04 = 0b00000100 → row1 col 0, row2 col 1, all positive → identity
        assert_eq!(decode_rotation_byte(0x04), Some(ROT_IDENTITY));
    }

    #[test]
    fn rotation_byte_negate_y() {
        // bits: row1 col 0 (00), row2 col 1 (01), row1+ row2- row3+
        // = 0b0010_0100 = 0x24
        let r = decode_rotation_byte(0x24).unwrap();
        assert_eq!(r, [[1, 0, 0], [0, -1, 0], [0, 0, 1]]);
        assert_eq!(apply_rotation(r, (3, 5, 7)), (3, -5, 7));
    }

    #[test]
    fn rotation_byte_swap_xy() {
        // row1 col 1, row2 col 0, all positive
        // bits 0-1 = 01, bits 2-3 = 00, signs all 0
        // = 0b0000_0001 = 0x01
        let r = decode_rotation_byte(0x01).unwrap();
        assert_eq!(r, [[0, 1, 0], [1, 0, 0], [0, 0, 1]]);
        assert_eq!(apply_rotation(r, (3, 5, 7)), (5, 3, 7));
    }

    #[test]
    fn rotation_compose_double_swap_is_identity() {
        // Two consecutive 90° X-axis rotations (or any rotation
        // composed with itself twice) should bring identity back
        // for the involutive ones. Simple sanity: identity composed
        // with anything = that thing.
        let r = decode_rotation_byte(0x24).unwrap();
        let composed = rotation_compose(ROT_IDENTITY, r);
        assert_eq!(composed, r);
    }

    #[test]
    fn rotation_byte_never_panics_and_rejects_junk() {
        // Every byte must be decodable or cleanly rejected — this test
        // sweeps all 256, so a debug build would panic on the bad ones
        // rather than fail an assert.
        //
        // Bits 0-3 pick the two column indices; only 6 of those 16
        // pairs are legal (3 choices × 2 remaining), so 10 pairs × 16
        // sign/spare-bit combinations = 160 bytes are rejected. The 64
        // that used to panic outright are the subset where the columns
        // differ but sum above 3, underflowing `3 - row1 - row2`.
        let mut rejected = 0;
        let mut underflow_shaped = 0;
        for b in 0u8..=255 {
            let (c1, c2) = ((b & 0b11) as usize, ((b >> 2) & 0b11) as usize);
            if c1 != c2 && c1 + c2 > 3 {
                underflow_shaped += 1;
                assert!(
                    decode_rotation_byte(b).is_none(),
                    "byte {b:#04x} would underflow and must be rejected"
                );
            }
            match decode_rotation_byte(b) {
                Some(m) => {
                    // A signed permutation: exactly one ±1 per row and
                    // per column.
                    for row in 0..3 {
                        assert_eq!(
                            m[row].iter().filter(|v| **v != 0).count(),
                            1,
                            "byte {b:#04x} row {row} is not a permutation"
                        );
                    }
                    for col in 0..3 {
                        assert_eq!(
                            (0..3).filter(|r| m[*r][col] != 0).count(),
                            1,
                            "byte {b:#04x} col {col} is not a permutation"
                        );
                    }
                }
                None => rejected += 1,
            }
        }
        assert_eq!(underflow_shaped, 64);
        assert_eq!(rejected, 160, "expected exactly the malformed bytes");
    }

    #[test]
    fn translation_parse_is_all_or_nothing() {
        assert_eq!(parse_translation("1 2 3"), Some((1, 2, 3)));
        assert_eq!(parse_translation("-4  5\t-6"), Some((-4, 5, -6)));
        // A single bad component rejects the whole triple rather than
        // quietly zeroing that axis.
        assert_eq!(parse_translation("1 x 3"), None);
        assert_eq!(parse_translation("1 2"), None);
        assert_eq!(parse_translation("1 2 3 4"), None);
        assert_eq!(parse_translation(""), None);
    }

    #[test]
    fn default_palette_matches_the_spec_table() {
        let p = default_palette();
        // Reserved empty slot, then the spec's first/last entries
        // (0xAABBGGRR: 0xffffffff white, 0xff111111 near-black).
        assert_eq!(p[0], [0, 0, 0, 0]);
        assert_eq!(p[1], [255, 255, 255, 255]);
        assert_eq!(p[2], [255, 255, 204, 255]);
        assert_eq!(p[255], [17, 17, 17, 255]);
        // Every real entry is opaque, and only slot 0 is empty.
        assert!(p[1..].iter().all(|c| c[3] == 255));
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

        let world = import_vox(&mut buf.as_slice(), false).expect("v200 import");
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

        let world = import_vox(&mut buf.as_slice(), false)
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

        let world = import_vox(&mut buf.as_slice(), false).expect("multi-model v200");

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
        let overflow = export_vox(&world, &mut buffer, false).unwrap();
        assert!(
            overflow >= 1,
            "expected at least one overflow color, got {}",
            overflow
        );
    }
}
