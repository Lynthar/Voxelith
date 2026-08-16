//! Project save/load: a gzip container holding metadata, world chunks
//! and editor state.

use crate::core::{Chunk, ChunkPos, Voxel, World, CHUNK_SIZE, CHUNK_VOLUME};
use crate::procgen::PipelineGraph;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use std::io::{self, Read, Write};
use thiserror::Error;

/// Project file magic bytes
const PROJECT_MAGIC: [u8; 4] = *b"VXLT";
/// Current project format version. Version 0 was never written by any
/// build, so a zero here is damage rather than an old file.
const PROJECT_VERSION: u32 = 1;
/// Cap for the chunk-vector capacity hint read from the header.
/// `chunk_count` is untrusted; the read loop still consumes the full
/// declared count and errors cleanly if the stream runs short.
const MAX_CHUNK_HINT: usize = 4096;

/// Ceiling on the whole decompressed stream. gzip expands up to
/// ~1000:1, so without it a file of a few megabytes can declare
/// gigabytes. Hitting the cap surfaces as `UnexpectedEof`.
const MAX_DECOMPRESSED_BYTES: u64 = 1 << 30;

/// Ceiling on the JSON header. Real headers are kilobytes, so this
/// leaves three orders of headroom while still refusing a header that
/// is itself the decompression bomb.
const MAX_HEADER_BYTES: usize = 64 << 20;

/// Ceiling on one chunk's RLE payload: a run is 10 bytes and can cover
/// as little as one voxel, so no honest chunk needs more than one run
/// per cell.
const MAX_RLE_BYTES: usize = CHUNK_VOLUME * 10;

/// Ceiling on chunk coordinates read from a file, per axis. Cells are
/// `chunk * 32 + 0..=31`, which overflows near `i32::MAX`; ±2^24 chunks
/// leaves headroom for every AABB derived from one.
const MAX_CHUNK_COORD: i32 = 1 << 24;

/// Errors that can occur when reading/writing project files
#[derive(Debug, Error)]
pub enum ProjectError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Invalid project magic number")]
    InvalidMagic,
    #[error("Unsupported project version: {0}")]
    UnsupportedVersion(u32),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Invalid chunk data")]
    InvalidChunkData,
    #[error("Decompression error")]
    DecompressionError,
    /// A declared size or coordinate is past what any real project
    /// needs — the file is corrupt or hostile, not merely big.
    #[error("File exceeds format limits: {0}")]
    LimitExceeded(&'static str),
    /// The gzip stream holds data past the last declared chunk. A
    /// well-formed writer never produces this, so the length fields and
    /// the stream disagree about where the file ends.
    #[error("Data past the end of the declared content")]
    TrailingData,
}

/// Project metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMetadata {
    /// Project name
    pub name: String,
    /// Project description
    pub description: String,
    /// Author name
    pub author: String,
    /// Creation timestamp (Unix epoch seconds)
    pub created_at: u64,
    /// Last modified timestamp
    pub modified_at: u64,
    /// Voxelith version that created this project
    pub app_version: String,
}

impl Default for ProjectMetadata {
    fn default() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            name: "Untitled Project".to_string(),
            description: String::new(),
            author: String::new(),
            created_at: now,
            modified_at: now,
            app_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// Editor state that can be saved with the project
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditorState {
    /// Camera position
    pub camera_position: [f32; 3],
    /// Camera target
    pub camera_target: [f32; 3],
    /// Current brush color
    pub brush_color: [u8; 4],
    /// Color palette
    pub palette: Vec<[u8; 4]>,
    /// Selected tool index
    pub selected_tool: usize,
    /// Named attachment points placed in the scene. `#[serde(default)]`
    /// so files written before sockets existed still load.
    #[serde(default)]
    pub sockets: Vec<SocketData>,
    /// Brush material flags (bit 0 emissive, bit 1 metallic) at save
    /// time. `#[serde(default)]`; round-tripping it is what stops a load
    /// from clearing the mode, since `from_rgba` zeroes `flags`.
    #[serde(default)]
    pub brush_flags: u8,
    /// Brush tint zone at save time, under the same `#[serde(default)]`
    /// forward-compat contract as `brush_flags`.
    #[serde(default)]
    pub brush_tint_zone: u8,
    /// The procedural pipeline graph, if this project has one. Document
    /// data — how the model was made — so it travels with the project
    /// rather than the machine. `#[serde(default)]`.
    #[serde(default)]
    pub graph: PipelineGraph,
}

/// Where the editor's camera starts on a new scene, so a project built
/// headless still opens looking at something. `Renderer::new` starts
/// from this same constant, so the two cannot drift.
pub const DEFAULT_CAMERA_POSITION: [f32; 3] = [0.0, 20.0, 40.0];

impl Default for EditorState {
    fn default() -> Self {
        Self {
            // Not `[0.0; 3]`: position equal to target is a degenerate
            // look-at, which opens as a blank viewport and leaves the
            // orbit controls deriving yaw and pitch from a zero vector.
            camera_position: DEFAULT_CAMERA_POSITION,
            camera_target: [0.0; 3],
            brush_color: Default::default(),
            palette: Default::default(),
            selected_tool: Default::default(),
            sockets: Default::default(),
            brush_flags: Default::default(),
            brush_tint_zone: Default::default(),
            graph: Default::default(),
        }
    }
}

/// Serializable form of an `editor::Socket`. Plain data here so `io`
/// doesn't depend on `editor`; `app::file_ops` converts at the
/// boundary, like camera / brush / palette.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SocketData {
    pub name: String,
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

/// Serializable chunk data
#[derive(Serialize, Deserialize)]
struct ChunkData {
    /// Chunk position
    pos: ChunkPos,
    /// Run-length encoded voxel data
    rle_data: Vec<u8>,
}

/// Complete project data
#[derive(Serialize, Deserialize)]
pub struct Project {
    /// Project metadata
    pub metadata: ProjectMetadata,
    /// Editor state
    pub editor_state: EditorState,
    /// Chunk data (serialized separately)
    #[serde(skip)]
    chunks: Vec<ChunkData>,
}

impl Project {
    /// Create a new empty project
    pub fn new() -> Self {
        Self {
            metadata: ProjectMetadata::default(),
            editor_state: EditorState::default(),
            chunks: Vec::new(),
        }
    }

    /// Create project from world
    pub fn from_world(world: &World) -> Self {
        Self::from_world_with_state(world, EditorState::default())
    }

    /// Create project from world with editor state
    pub fn from_world_with_state(world: &World, editor_state: EditorState) -> Self {
        let mut project = Self::new();
        project.editor_state = editor_state;

        // Deterministic chunk order, so identical content serializes to
        // the same structure whatever the insertion order. Whole-file
        // bytes still move — `modified_at` does that on purpose.
        for pos in world.sorted_chunk_positions() {
            let Some(chunk_lock) = world.get_chunk(pos) else {
                continue;
            };
            let chunk = chunk_lock.read();
            if !chunk.is_empty() {
                let rle_data = rle_encode_chunk(&chunk);
                project.chunks.push(ChunkData { pos, rle_data });
            }
        }

        project
    }

    /// Convert project to world. A chunk that doesn't decode is an
    /// error, not a gap: loading past one leaves the model silently
    /// short a wall.
    pub fn to_world(&self) -> Result<World, ProjectError> {
        let mut world = World::new();

        for chunk_data in &self.chunks {
            let chunk = rle_decode_chunk(&chunk_data.rle_data)?;
            *world.get_or_create_chunk(chunk_data.pos).write() = chunk;
        }

        Ok(world)
    }

    /// Save project to writer
    pub fn save<W: Write>(&self, writer: &mut W) -> Result<(), ProjectError> {
        // Write magic and version
        writer.write_all(&PROJECT_MAGIC)?;
        writer.write_all(&PROJECT_VERSION.to_le_bytes())?;

        // Create compressed stream
        let mut encoder = GzEncoder::new(writer, Compression::default());

        // Serialize metadata and editor state as JSON
        let header_json = serde_json::to_string(&(&self.metadata, &self.editor_state))?;
        let header_bytes = header_json.as_bytes();
        encoder.write_all(&(header_bytes.len() as u32).to_le_bytes())?;
        encoder.write_all(header_bytes)?;

        // Write chunk count
        encoder.write_all(&(self.chunks.len() as u32).to_le_bytes())?;

        // Write each chunk
        for chunk_data in &self.chunks {
            // Write position
            encoder.write_all(&chunk_data.pos.x.to_le_bytes())?;
            encoder.write_all(&chunk_data.pos.y.to_le_bytes())?;
            encoder.write_all(&chunk_data.pos.z.to_le_bytes())?;

            // Write RLE data
            encoder.write_all(&(chunk_data.rle_data.len() as u32).to_le_bytes())?;
            encoder.write_all(&chunk_data.rle_data)?;
        }

        encoder.finish()?;
        Ok(())
    }

    /// Load a project. Strict by design: every declared size is capped,
    /// chunk coordinates are bounded, and the stream drains to gzip EOF
    /// so the CRC trailer is verified.
    pub fn load<R: Read>(reader: &mut R) -> Result<Self, ProjectError> {
        // Read and verify magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != PROJECT_MAGIC {
            return Err(ProjectError::InvalidMagic);
        }

        // Read version. 0 is rejected alongside the future: no build
        // ever wrote it (the constant has been 1 from the first commit),
        // so it can only mean a damaged version field.
        let mut version_buf = [0u8; 4];
        reader.read_exact(&mut version_buf)?;
        let version = u32::from_le_bytes(version_buf);
        if version != PROJECT_VERSION {
            return Err(ProjectError::UnsupportedVersion(version));
        }

        // Decompress, under a total-output cap — gzip's ratio makes the
        // decompressed size otherwise the file author's choice, not ours.
        let mut decoder = GzDecoder::new(reader).take(MAX_DECOMPRESSED_BYTES);

        // Read header JSON
        let mut len_buf = [0u8; 4];
        decoder.read_exact(&mut len_buf)?;
        let header_len = u32::from_le_bytes(len_buf) as usize;
        if header_len > MAX_HEADER_BYTES {
            return Err(ProjectError::LimitExceeded("header size"));
        }
        let header_bytes = super::read_exact_vec(&mut decoder, header_len)?;

        let (metadata, editor_state): (ProjectMetadata, EditorState) =
            serde_json::from_slice(&header_bytes)?;

        // Read chunk count
        decoder.read_exact(&mut len_buf)?;
        let chunk_count = u32::from_le_bytes(len_buf) as usize;

        // Cap the capacity hint so a bogus count can't request a huge
        // eager allocation; the loop still reads the full declared count
        // and fails via read_exact if the data runs short.
        let mut chunks = Vec::with_capacity(chunk_count.min(MAX_CHUNK_HINT));
        let mut seen = std::collections::HashSet::with_capacity(chunk_count.min(MAX_CHUNK_HINT));
        for _ in 0..chunk_count {
            // Read position
            let mut pos_buf = [0u8; 4];
            decoder.read_exact(&mut pos_buf)?;
            let x = i32::from_le_bytes(pos_buf);
            decoder.read_exact(&mut pos_buf)?;
            let y = i32::from_le_bytes(pos_buf);
            decoder.read_exact(&mut pos_buf)?;
            let z = i32::from_le_bytes(pos_buf);
            if x.unsigned_abs() > MAX_CHUNK_COORD as u32
                || y.unsigned_abs() > MAX_CHUNK_COORD as u32
                || z.unsigned_abs() > MAX_CHUNK_COORD as u32
            {
                return Err(ProjectError::LimitExceeded("chunk position"));
            }
            // The writer emits each position once (it iterates a map's
            // sorted keys), so a duplicate means the count and the
            // stream disagree — refusing beats a silent last-one-wins.
            if !seen.insert((x, y, z)) {
                return Err(ProjectError::InvalidChunkData);
            }

            // Read RLE data
            decoder.read_exact(&mut len_buf)?;
            let rle_len = u32::from_le_bytes(len_buf) as usize;
            if rle_len > MAX_RLE_BYTES {
                return Err(ProjectError::LimitExceeded("chunk RLE size"));
            }
            // Runs are 10 bytes each. Refused structurally here as well
            // as in `rle_decode_chunk`, so a `Project` never holds data
            // it will later refuse.
            if !rle_len.is_multiple_of(10) {
                return Err(ProjectError::InvalidChunkData);
            }
            let rle_data = super::read_exact_vec(&mut decoder, rle_len)?;

            chunks.push(ChunkData {
                pos: ChunkPos::new(x, y, z),
                rle_data,
            });
        }

        drain_to_eof(&mut decoder)?;

        Ok(Self {
            metadata,
            editor_state,
            chunks,
        })
    }

    /// Update metadata modified timestamp
    pub fn touch(&mut self) {
        self.metadata.modified_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
    }
}

impl Default for Project {
    fn default() -> Self {
        Self::new()
    }
}

/// Read past the last declared chunk, which is where flate2 verifies the
/// gzip CRC-32 and length trailer. Any byte still there means the stream
/// and the length fields disagree.
///
/// # Errors
/// `LimitExceeded` on a spent budget, `TrailingData` on leftover bytes.
fn drain_to_eof<R: Read>(reader: &mut io::Take<R>) -> Result<(), ProjectError> {
    let mut tail = [0u8; 64];
    match reader.read(&mut tail) {
        // A spent `Take` budget answers `Ok(0)` without ever reaching
        // the decoder, so the trailer would go unverified.
        Ok(0) if reader.limit() == 0 => Err(ProjectError::LimitExceeded("decompressed size")),
        Ok(0) => Ok(()),
        Ok(_) => Err(ProjectError::TrailingData),
        Err(e) => Err(e.into()),
    }
}

/// Run-length encode chunk voxels
fn rle_encode_chunk(chunk: &Chunk) -> Vec<u8> {
    let voxels = chunk.voxels();
    let mut result = Vec::new();

    if voxels.is_empty() {
        return result;
    }

    let mut current = voxels[0];
    let mut count = 1u16;

    for voxel in voxels.iter().skip(1) {
        if *voxel == current && count < 65535 {
            count += 1;
        } else {
            // Write run
            write_rle_run(&mut result, current, count);
            current = *voxel;
            count = 1;
        }
    }

    // Write final run
    write_rle_run(&mut result, current, count);

    result
}

/// Write a single RLE run
fn write_rle_run(buf: &mut Vec<u8>, voxel: Voxel, count: u16) {
    // Count as 2 bytes
    buf.extend_from_slice(&count.to_le_bytes());
    // Voxel data as 8 bytes
    buf.extend_from_slice(bytemuck::bytes_of(&voxel));
}

/// Run-length decode chunk voxels. Strict on every axis the encoder is
/// exact about: whole 10-byte runs, no empty run, summing to exactly
/// 32³ cells. Padding or truncating instead hides a corrupt file.
fn rle_decode_chunk(data: &[u8]) -> Result<Chunk, ProjectError> {
    if !data.len().is_multiple_of(10) {
        return Err(ProjectError::InvalidChunkData);
    }
    let mut decoded: Vec<Voxel> = Vec::with_capacity(CHUNK_VOLUME);

    let mut offset = 0;
    while offset + 10 <= data.len() {
        // Read count (2 bytes)
        let count = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;
        // A zero-length run encodes nothing; the encoder never emits
        // one, so it can only be damage.
        if count == 0 {
            return Err(ProjectError::InvalidChunkData);
        }

        // Read voxel (8 bytes)
        let voxel_bytes: [u8; 8] = data[offset..offset + 8]
            .try_into()
            .map_err(|_| ProjectError::InvalidChunkData)?;
        let mut voxel: Voxel = *bytemuck::from_bytes(&voxel_bytes);
        offset += 8;
        // Every voxel that reaches the world is opaque — bytemuck hands
        // back whatever the file said, and a solid voxel with a zero
        // color is the mesher's "no visible face" sentinel.
        if voxel.is_solid() {
            voxel.a = 255;
        }

        // Overflowing the chunk means the runs and the format disagree.
        if decoded.len() + count > CHUNK_VOLUME {
            return Err(ProjectError::InvalidChunkData);
        }
        for _ in 0..count {
            decoded.push(voxel);
        }
    }

    // Exactly one chunk of cells — short data used to be padded with
    // air, which reads as "part of the model quietly missing".
    if decoded.len() != CHUNK_VOLUME {
        return Err(ProjectError::InvalidChunkData);
    }

    // Create chunk with decoded voxels
    let mut chunk = Chunk::new();
    for (i, voxel) in decoded.into_iter().enumerate() {
        let x = i % CHUNK_SIZE;
        let y = (i / CHUNK_SIZE) % CHUNK_SIZE;
        let z = i / (CHUNK_SIZE * CHUNK_SIZE);
        if voxel.is_solid() {
            chunk.set(x, y, z, voxel);
        }
    }

    Ok(chunk)
}

/// Save world, editor state and metadata atomically and durably. The
/// metadata comes from the caller, which keeps identity stable across a
/// round trip; `modified_at` moves to now here.
pub fn save_world_with_state(
    world: &World,
    editor_state: EditorState,
    metadata: ProjectMetadata,
    path: &std::path::Path,
) -> Result<(), ProjectError> {
    let mut project = Project::from_world_with_state(world, editor_state);
    project.metadata = metadata;
    project.touch();
    write_project_atomic(&project, path)
}

/// Serialize a project to `path` atomically and durably: write a
/// per-process sibling temp, fsync it, then rename it over the target.
/// A failure removes the temp rather than leaving it behind.
fn write_project_atomic(project: &Project, path: &std::path::Path) -> Result<(), ProjectError> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(format!(".tmp{}", std::process::id()));
    let tmp = std::path::PathBuf::from(tmp);

    // Phase 1: write + fsync the temp. On any error, drop the partial
    // temp so we never leave stray `.tmp` files behind.
    if let Err(e) = write_temp_then_sync(project, &tmp) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }

    // Phase 2: atomically replace the target with the complete temp.
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.into());
    }

    // Phase 3 (POSIX): fsync the parent directory, where the rename
    // itself lives. Best-effort — a directory that can't be synced
    // doesn't fail a save whose bytes are already safe.
    #[cfg(unix)]
    if let Some(dir) = path.parent() {
        if let Ok(dir) = std::fs::File::open(dir) {
            let _ = dir.sync_all();
        }
    }
    Ok(())
}

/// Write `project` to `tmp`, flush it (surfacing the error `BufWriter`'s
/// `Drop` would swallow) and fsync it. Split out so the caller has one
/// cleanup path for every early return.
fn write_temp_then_sync(project: &Project, tmp: &std::path::Path) -> Result<(), ProjectError> {
    let file = std::fs::File::create(tmp)?;
    let mut writer = std::io::BufWriter::new(file);
    project.save(&mut writer)?;
    // `into_inner` flushes the buffer and hands back the File; on a flush
    // error it yields an `IntoInnerError` we unwrap to the io::Error.
    let file = writer.into_inner().map_err(|e| e.into_error())?;
    file.sync_all()?;
    Ok(())
}

/// Quick load world from file path
pub fn load_world(path: &std::path::Path) -> Result<World, ProjectError> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    let project = Project::load(&mut reader)?;
    project.to_world()
}

/// Load world, editor state and metadata. The metadata travels out so
/// the host can hand it back to [`save_world_with_state`], which keeps
/// `name` / `author` / `created_at` stable across open → save.
pub fn load_world_with_state(
    path: &std::path::Path,
) -> Result<(World, EditorState, ProjectMetadata), ProjectError> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    let project = Project::load(&mut reader)?;
    Ok((project.to_world()?, project.editor_state, project.metadata))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decompression cap and a real end-of-stream both surface as
    /// `Ok(0)`, and only the second one means flate2 checked the CRC
    /// trailer. Told apart by the budget rather than run at 1 GiB.
    #[test]
    fn a_spent_decompression_budget_is_not_a_clean_eof() {
        let data = [0u8; 100];

        let mut spent = Read::take(&data[..], 0);
        assert!(
            matches!(
                drain_to_eof(&mut spent),
                Err(ProjectError::LimitExceeded("decompressed size"))
            ),
            "a spent budget must be refused, not read as a verified trailer"
        );

        let empty: &[u8] = &[];
        let mut clean = Read::take(empty, 64);
        assert!(drain_to_eof(&mut clean).is_ok(), "a real EOF still passes");

        let mut leftover = Read::take(&data[..], 64);
        assert!(matches!(
            drain_to_eof(&mut leftover),
            Err(ProjectError::TrailingData)
        ));
    }

    #[test]
    fn test_project_roundtrip() {
        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgb(255, 0, 0));
        world.set_voxel(1, 1, 1, Voxel::from_rgb(0, 255, 0));
        world.set_voxel(31, 31, 31, Voxel::from_rgb(0, 0, 255));

        let project = Project::from_world(&world);

        let mut buffer = Vec::new();
        project.save(&mut buffer).unwrap();

        let loaded = Project::load(&mut buffer.as_slice()).unwrap();
        let loaded_world = loaded.to_world().unwrap();

        assert!(loaded_world.get_voxel(0, 0, 0).is_solid());
        assert_eq!(loaded_world.get_voxel(0, 0, 0).r, 255);
        assert!(loaded_world.get_voxel(1, 1, 1).is_solid());
        assert_eq!(loaded_world.get_voxel(1, 1, 1).g, 255);
    }

    #[test]
    fn the_pipeline_graph_survives_a_save_and_load() {
        // The graph is document data: it says how this model was made,
        // and a round trip that dropped it would delete the recipe while
        // keeping the result.
        use crate::procgen::{NodeKind, PerlinTerrain};

        let mut state = EditorState::default();
        let src = state.graph.add(NodeKind::Terrain(PerlinTerrain {
            width: 24,
            ..Default::default()
        }));
        state.graph.add(NodeKind::Output { input: Some(src) });

        let project = Project::from_world_with_state(&World::new(), state.clone());
        let mut buffer = Vec::new();
        project.save(&mut buffer).unwrap();
        let loaded = Project::load(&mut buffer.as_slice()).unwrap();

        assert_eq!(loaded.editor_state.graph, state.graph);
    }

    #[test]
    fn a_project_written_before_graphs_still_loads() {
        // `#[serde(default)]`, stated as a test: the header is JSON, and
        // an older build's header simply has no `graph` key.
        let header = r#"[{"name":"old","description":"","author":"","created_at":0,
            "modified_at":0,"app_version":"0.0.1"},
            {"camera_position":[0.0,20.0,40.0],"camera_target":[0.0,0.0,0.0],
             "brush_color":[1,2,3,255],"palette":[],"selected_tool":0}]"#;
        let (_, state): (ProjectMetadata, EditorState) =
            serde_json::from_str(header).expect("an older header must still parse");
        assert!(state.graph.nodes.is_empty());
        assert_eq!(state.brush_color, [1, 2, 3, 255]);
    }

    #[test]
    fn save_bytes_are_chunk_order_independent() {
        // The chunk store is a HashMap with per-process iteration order,
        // so two worlds with identical content could serialize
        // differently if the writer didn't sort chunks.
        let cells: Vec<(i32, i32, i32)> = (0..8)
            .map(|i| (i * 40 - 120, (i % 3) * 5, (i % 5) * 40 - 80))
            .collect();
        let color = Voxel::from_rgb(123, 45, 67);

        let mut forward = World::new();
        for &(x, y, z) in &cells {
            forward.set_voxel(x, y, z, color);
        }
        let mut reverse = World::new();
        for &(x, y, z) in cells.iter().rev() {
            reverse.set_voxel(x, y, z, color);
        }

        let mut a = Vec::new();
        Project::from_world(&forward).save(&mut a).unwrap();
        let mut b = Vec::new();
        Project::from_world(&reverse).save(&mut b).unwrap();
        assert_eq!(a, b, "chunk order must not affect the saved bytes");
    }

    #[test]
    fn test_roundtrip_preserves_editor_state_and_cross_chunk_voxels() {
        // Pins full EditorState equality and exact voxel round-trip
        // across negative coordinates and several chunks. Alpha is the
        // one field that does not survive verbatim.
        let mut world = World::new();
        let samples = [
            ((0, 0, 0), Voxel::from_rgb(255, 0, 0)),
            ((-1, -1, -1), Voxel::from_rgb(0, 255, 0)), // chunk (-1,-1,-1)
            ((31, 31, 31), Voxel::from_rgb(0, 0, 255)), // far corner of (0,0,0)
            ((32, 5, -33), Voxel::from_rgb(10, 20, 30)), // chunk (1,0,-2)
        ];
        for ((x, y, z), v) in samples {
            world.set_voxel(x, y, z, v);
        }

        let state = EditorState {
            camera_position: [1.5, -2.0, 3.25],
            camera_target: [0.0, 4.0, -1.0],
            brush_color: [12, 34, 56, 200],
            palette: vec![[1, 2, 3, 4], [255, 254, 253, 252]],
            selected_tool: 4,
            brush_flags: 0b11,  // emissive + metallic both set
            brush_tint_zone: 2, // secondary faction zone
            sockets: vec![
                SocketData {
                    name: "muzzle".to_string(),
                    position: [2.5, 1.0, -3.5],
                    normal: [0.0, 1.0, 0.0],
                },
                SocketData {
                    name: "Socket_2".to_string(),
                    position: [-1.0, 0.5, 4.0],
                    normal: [1.0, 0.0, 0.0],
                },
            ],
            graph: Default::default(),
        };

        let project = Project::from_world_with_state(&world, state.clone());
        let mut buffer = Vec::new();
        project.save(&mut buffer).unwrap();
        let loaded = Project::load(&mut buffer.as_slice()).unwrap();

        // EditorState round-trips field-for-field.
        let es = &loaded.editor_state;
        assert_eq!(es.camera_position, state.camera_position);
        assert_eq!(es.camera_target, state.camera_target);
        assert_eq!(es.brush_color, state.brush_color);
        assert_eq!(es.palette, state.palette);
        assert_eq!(es.selected_tool, state.selected_tool);
        assert_eq!(es.sockets, state.sockets);
        assert_eq!(es.brush_flags, state.brush_flags);
        assert_eq!(es.brush_tint_zone, state.brush_tint_zone);

        // Every set voxel survives — negatives, far chunks, exact rgba.
        let loaded_world = loaded.to_world().unwrap();
        for ((x, y, z), v) in samples {
            assert_eq!(
                loaded_world.get_voxel(x, y, z),
                v,
                "voxel ({}, {}, {}) did not round-trip",
                x,
                y,
                z
            );
        }
    }

    /// A solid voxel whose color is `[0, 0, 0, 0]` is the mesher's "no
    /// visible face" sentinel and reads as air to the flood fill. A
    /// `.vxlt` is no less external than a `.vox`.
    #[test]
    fn a_solid_voxel_arrives_opaque_whatever_the_file_said() {
        let dir = std::env::temp_dir().join("voxelith_alpha_normalize");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("proj.vxlt");

        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgba(10, 20, 30, 0));
        world.set_voxel(1, 0, 0, Voxel::from_rgba(40, 50, 60, 128));
        save_world_with_state(&world, EditorState::default(), Default::default(), &path).unwrap();

        let (loaded, _, _) = load_world_with_state(&path).unwrap();
        assert_eq!(loaded.get_voxel(0, 0, 0), Voxel::from_rgb(10, 20, 30));
        assert_eq!(loaded.get_voxel(1, 0, 0), Voxel::from_rgb(40, 50, 60));
        // Air is left alone: it is the one voxel that is *supposed* to
        // be fully transparent, and normalizing it would make every
        // empty cell a solid black one.
        assert!(loaded.get_voxel(2, 0, 0).is_air());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn editor_state_without_sockets_field_still_loads() {
        // A file written before sockets existed has no `sockets` key.
        // `#[serde(default)]` must fill it rather than failing the whole
        // header parse, which would brick every old project.
        let json = r#"{
            "camera_position": [0.0, 0.0, 0.0],
            "camera_target": [0.0, 0.0, 0.0],
            "brush_color": [10, 20, 30, 255],
            "palette": [],
            "selected_tool": 2
        }"#;
        let es: EditorState = serde_json::from_str(json).unwrap();
        assert_eq!(es.selected_tool, 2);
        assert!(es.sockets.is_empty());
        // brush_flags / brush_tint_zone are likewise absent in pre-existing
        // files; `#[serde(default)]` must fill them with 0 (a plain brush).
        assert_eq!(es.brush_flags, 0);
        assert_eq!(es.brush_tint_zone, 0);
    }

    #[test]
    fn load_truncated_never_panics() {
        // A crash mid-write can truncate a `.vxlt` at any offset.
        // Loading any prefix must return Ok or Err — never panic — so a
        // damaged autosave can't brick startup.
        let mut world = World::new();
        for i in 0..40 {
            world.set_voxel(
                i,
                i % 8,
                (i * 2) % 16,
                Voxel::from_rgb((i * 6) as u8, 100, 200),
            );
        }
        world.set_voxel(40, 0, -40, Voxel::from_rgb(1, 2, 3)); // forces a 2nd chunk
        let project = Project::from_world(&world);
        let mut buf = Vec::new();
        project.save(&mut buf).unwrap();
        assert!(buf.len() > 16);

        // Every prefix loads without panicking (the loop itself is the
        // assertion — a panic here fails the test).
        for len in 0..=buf.len() {
            let mut r = &buf[..len];
            let _ = Project::load(&mut r);
        }

        // Header-only (magic + version, no gzip stream) errors cleanly.
        let mut r8 = &buf[..8];
        assert!(Project::load(&mut r8).is_err());
        // A cut deep in the compressed body also errors, not loads garbage.
        let mut rmid = &buf[..(8 + (buf.len() - 8) / 2)];
        assert!(Project::load(&mut rmid).is_err());
        // The intact buffer still round-trips.
        let mut rfull = buf.as_slice();
        assert!(Project::load(&mut rfull).is_ok());
    }

    #[test]
    fn save_world_with_state_writes_loadable_file_atomically() {
        // The roundtrip tests serialize into memory, so they never
        // exercise the real file path. This drives it: load back,
        // replace an existing file, and leave no `.tmp` behind.
        let dir = std::env::temp_dir().join("voxelith_atomic_save");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("proj.vxlt");
        // What the helper writes — the temp name carries the pid so two
        // processes saving the same project can't share (and truncate)
        // one temp file.
        let tmp = path.with_file_name(format!("proj.vxlt.tmp{}", std::process::id()));

        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgb(255, 0, 0));
        world.set_voxel(40, 2, -3, Voxel::from_rgb(1, 2, 3)); // forces a 2nd chunk
        let state = EditorState {
            brush_color: [12, 34, 56, 200],
            brush_flags: 0b11, // emissive + metallic
            brush_tint_zone: 2,
            ..Default::default()
        };

        save_world_with_state(&world, state.clone(), Default::default(), &path).unwrap();
        assert!(path.exists(), "save produced no file");
        assert!(
            !tmp.exists(),
            "temp file left behind after a successful save"
        );

        let (loaded_world, loaded_state, _) = load_world_with_state(&path).unwrap();
        assert_eq!(loaded_world.get_voxel(0, 0, 0).r, 255);
        assert_eq!(loaded_world.get_voxel(40, 2, -3), Voxel::from_rgb(1, 2, 3));
        assert_eq!(loaded_state.brush_color, [12, 34, 56, 200]);
        assert_eq!(loaded_state.brush_flags, 0b11);
        assert_eq!(loaded_state.brush_tint_zone, 2);

        // A second save over the SAME path must replace it wholesale — the
        // old red voxel is gone and the new green one is present (proves the
        // rename swapped the file rather than appending / partially writing).
        let mut world2 = World::new();
        world2.set_voxel(5, 5, 5, Voxel::from_rgb(0, 255, 0));
        save_world_with_state(&world2, EditorState::default(), Default::default(), &path).unwrap();
        let (loaded2, _, _) = load_world_with_state(&path).unwrap();
        assert!(
            loaded2.get_voxel(0, 0, 0).is_air(),
            "old content survived the overwrite"
        );
        assert_eq!(loaded2.get_voxel(5, 5, 5).g, 255);
        assert!(!tmp.exists(), "temp file left behind after overwrite");

        let _ = std::fs::remove_dir_all(&dir);
    }

    type RawChunk = ((i32, i32, i32), Vec<u8>);

    /// Hand-assemble a `.vxlt` from parts, mirroring `Project::save`'s
    /// layout, so the strictness tests can put damage exactly where
    /// they mean to.
    fn raw_project(chunks: &[RawChunk]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&PROJECT_MAGIC);
        out.extend_from_slice(&PROJECT_VERSION.to_le_bytes());
        let mut enc = GzEncoder::new(&mut out, Compression::default());
        let header =
            serde_json::to_string(&(ProjectMetadata::default(), EditorState::default())).unwrap();
        enc.write_all(&(header.len() as u32).to_le_bytes()).unwrap();
        enc.write_all(header.as_bytes()).unwrap();
        enc.write_all(&(chunks.len() as u32).to_le_bytes()).unwrap();
        for ((x, y, z), rle) in chunks {
            enc.write_all(&x.to_le_bytes()).unwrap();
            enc.write_all(&y.to_le_bytes()).unwrap();
            enc.write_all(&z.to_le_bytes()).unwrap();
            enc.write_all(&(rle.len() as u32).to_le_bytes()).unwrap();
            enc.write_all(rle).unwrap();
        }
        enc.finish().unwrap();
        out
    }

    /// One full chunk of a single solid color, as the encoder writes it:
    /// runs of u16::MAX voxels plus a remainder, summing to 32³.
    fn full_chunk_rle() -> Vec<u8> {
        let mut chunk = Chunk::new();
        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    chunk.set(x, y, z, Voxel::from_rgb(9, 8, 7));
                }
            }
        }
        rle_encode_chunk(&chunk)
    }

    #[test]
    fn a_file_with_its_gzip_trailer_cut_off_is_refused() {
        // The exact corruption a crash mid-copy produces: everything
        // present but the final CRC-32 and length trailer. A loader that
        // stops at the last chunk reports this as loading cleanly.
        let bytes = raw_project(&[((0, 0, 0), full_chunk_rle())]);
        let truncated = &bytes[..bytes.len() - 8];
        assert!(
            Project::load(&mut &truncated[..]).is_err(),
            "a truncated gzip stream must not load"
        );
        // The intact stream still does — proving the drain-to-EOF
        // doesn't reject well-formed files.
        assert!(Project::load(&mut &bytes[..]).is_ok());
    }

    /// The whole read path a real open takes: parse the container, then
    /// decode every chunk. Content-level RLE damage surfaces in the
    /// second half.
    fn load_bytes(bytes: &[u8]) -> Result<World, ProjectError> {
        Project::load(&mut &bytes[..]).and_then(|p| p.to_world())
    }

    #[test]
    fn corrupt_rle_payloads_are_refused_not_papered_over() {
        // Trailing fragment: not a whole number of 10-byte runs.
        let mut fragment = full_chunk_rle();
        fragment.extend_from_slice(&[1, 2, 3]);
        let bytes = raw_project(&[((0, 0, 0), fragment)]);
        assert!(load_bytes(&bytes).is_err(), "fragment accepted");

        // Short: runs sum to less than a chunk. Used to be padded with
        // air — part of the model quietly missing.
        let mut short = Vec::new();
        short.extend_from_slice(&5u16.to_le_bytes());
        short.extend_from_slice(bytemuck::bytes_of(&Voxel::from_rgb(1, 2, 3)));
        let bytes = raw_project(&[((0, 0, 0), short)]);
        assert!(load_bytes(&bytes).is_err(), "short chunk accepted");

        // Long: runs sum past a chunk. Used to be silently truncated.
        let mut long = full_chunk_rle();
        long.extend_from_slice(&1u16.to_le_bytes());
        long.extend_from_slice(bytemuck::bytes_of(&Voxel::from_rgb(1, 2, 3)));
        let bytes = raw_project(&[((0, 0, 0), long)]);
        assert!(load_bytes(&bytes).is_err(), "overlong chunk accepted");

        // A zero-count run encodes nothing; the encoder never writes one.
        let mut zero = Vec::new();
        zero.extend_from_slice(&0u16.to_le_bytes());
        zero.extend_from_slice(bytemuck::bytes_of(&Voxel::from_rgb(1, 2, 3)));
        // Pad with a second, valid run so the %10 structural check
        // alone can't be what rejects it.
        zero.extend_from_slice(&7u16.to_le_bytes());
        zero.extend_from_slice(bytemuck::bytes_of(&Voxel::from_rgb(1, 2, 3)));
        let bytes = raw_project(&[((0, 0, 0), zero)]);
        assert!(load_bytes(&bytes).is_err(), "zero-run accepted");

        // And the counter-case: a well-formed chunk still loads whole.
        let bytes = raw_project(&[((0, 0, 0), full_chunk_rle())]);
        let world = load_bytes(&bytes).expect("well-formed chunk must load");
        assert_eq!(world.get_voxel(0, 0, 0), Voxel::from_rgb(9, 8, 7));
        assert_eq!(world.get_voxel(31, 31, 31), Voxel::from_rgb(9, 8, 7));
    }

    #[test]
    fn hostile_positions_counts_and_versions_are_refused() {
        // A chunk coordinate the mesher's `chunk * 32` math can't hold.
        let bytes = raw_project(&[((i32::MAX, 0, 0), full_chunk_rle())]);
        assert!(matches!(
            Project::load(&mut &bytes[..]),
            Err(ProjectError::LimitExceeded(_))
        ));

        // The same coordinate twice: the writer never does this, so the
        // count and the stream disagree.
        let bytes = raw_project(&[((0, 0, 0), full_chunk_rle()), ((0, 0, 0), full_chunk_rle())]);
        assert!(
            Project::load(&mut &bytes[..]).is_err(),
            "duplicate accepted"
        );

        // Version 0 was never written by any build.
        let mut bytes = raw_project(&[]);
        bytes[4..8].copy_from_slice(&0u32.to_le_bytes());
        assert!(matches!(
            Project::load(&mut &bytes[..]),
            Err(ProjectError::UnsupportedVersion(0))
        ));
    }

    #[test]
    fn saving_over_a_project_keeps_its_metadata() {
        // Open → save must not reset identity. The host loads the
        // metadata, holds it, and hands it back; `modified_at` alone
        // moves.
        let dir = std::env::temp_dir().join("voxelith_metadata_carry");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("proj.vxlt");

        // A project with distinctive metadata, written directly.
        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgb(1, 2, 3));
        let mut project = Project::from_world(&world);
        project.metadata.name = "Fortress".to_string();
        project.metadata.author = "Aqua".to_string();
        project.metadata.created_at = 1_600_000_000;
        project.metadata.modified_at = 1_600_000_000;
        write_project_atomic(&project, &path).unwrap();

        // The load → save round trip every host performs (the editor's
        // Document, exec --out, the MCP checkpoint).
        let (world, state, metadata) = load_world_with_state(&path).unwrap();
        save_world_with_state(&world, state, metadata, &path).unwrap();

        let file = std::fs::File::open(&path).unwrap();
        let loaded = Project::load(&mut std::io::BufReader::new(file)).unwrap();
        assert_eq!(loaded.metadata.name, "Fortress");
        assert_eq!(loaded.metadata.author, "Aqua");
        assert_eq!(loaded.metadata.created_at, 1_600_000_000);
        assert!(
            loaded.metadata.modified_at > 1_600_000_000,
            "modified_at is the one field a save should move"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_rle_encoding() {
        let mut chunk = Chunk::new();
        // Fill with same color to test RLE efficiency
        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    chunk.set(x, y, z, Voxel::from_rgb(128, 64, 32));
                }
            }
        }

        let encoded = rle_encode_chunk(&chunk);
        // Should be much smaller than raw data due to RLE
        assert!(encoded.len() < CHUNK_VOLUME * 8);

        let decoded = rle_decode_chunk(&encoded).unwrap();
        assert_eq!(decoded.get(0, 0, 0).r, 128);
        assert_eq!(decoded.get(15, 15, 15).g, 64);
    }
}
