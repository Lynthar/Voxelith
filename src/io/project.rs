//! Project save/load functionality.
//!
//! Projects are saved as compressed binary files containing:
//! - Project metadata (name, description, version)
//! - World data (chunks with voxel data)
//! - Editor state (camera position, tool settings, palette)

use crate::core::{Chunk, ChunkPos, Voxel, World, CHUNK_SIZE, CHUNK_VOLUME};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use std::io::{self, Read, Write};
use thiserror::Error;

/// Project file magic bytes
const PROJECT_MAGIC: [u8; 4] = [b'V', b'X', b'L', b'T'];
/// Current project format version
const PROJECT_VERSION: u32 = 1;
/// Cap for the chunk-vector capacity *hint* read from the file header.
/// `chunk_count` is untrusted; the hint is only a preallocation
/// optimization, so bounding it stops a corrupt file from requesting a
/// giant eager allocation. The read loop still consumes the full
/// declared count and errors cleanly if the stream is short. 4096 chunks
/// covers a 512³ world; larger ones just grow the Vec a few times.
const MAX_CHUNK_HINT: usize = 4096;

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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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
    /// Named attachment points (sockets) placed in the scene. `#[serde
    /// (default)]` so `.vxlt` files written before sockets existed still
    /// load (missing field → no sockets); the project format version
    /// doesn't need a bump because the addition is purely additive.
    #[serde(default)]
    pub sockets: Vec<SocketData>,
    /// Brush material flags (`Voxel::flags`: bit0 emissive / bit1
    /// metallic) captured at save time. `#[serde(default)]` so older
    /// `.vxlt` files (which never stored this) still load — missing → 0,
    /// a plain brush. Round-tripping it is what stops open / crash-
    /// recovery from silently clearing the brush's emissive / metallic
    /// mode: the load path rebuilds `brush_color` via `Voxel::from_rgba`,
    /// which zeroes `flags`.
    #[serde(default)]
    pub brush_flags: u8,
    /// Brush tint zone (`Voxel::tint_zone`, stored in `_reserved`:
    /// 0 none / 1 primary / 2 secondary / 3 reserved) captured at save
    /// time. Same `#[serde(default)]` forward-compat + anti-zeroing
    /// contract as `brush_flags`.
    #[serde(default)]
    pub brush_tint_zone: u8,
}

/// Serializable form of an `editor::Socket` (name + position + outward
/// normal). Kept as plain data here so the `io` layer doesn't depend on
/// `editor`; `app::file_ops` converts to/from `editor::Socket` at the
/// boundary, exactly like camera / brush / palette.
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

        // Deterministic chunk order so the `.vxlt` bytes are reproducible
        // across runs (HashMap iteration is per-process random) — matters
        // for backup dedup / content-addressing (#11).
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

    /// Convert project to world
    pub fn to_world(&self) -> World {
        let mut world = World::new();

        for chunk_data in &self.chunks {
            if let Some(chunk) = rle_decode_chunk(&chunk_data.rle_data) {
                // For unbounded worlds, get_or_create_chunk always returns Some
                if let Some(chunk_lock) = world.get_or_create_chunk(chunk_data.pos) {
                    *chunk_lock.write() = chunk;
                }
            }
        }

        world
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

    /// Load project from reader
    pub fn load<R: Read>(reader: &mut R) -> Result<Self, ProjectError> {
        // Read and verify magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != PROJECT_MAGIC {
            return Err(ProjectError::InvalidMagic);
        }

        // Read version
        let mut version_buf = [0u8; 4];
        reader.read_exact(&mut version_buf)?;
        let version = u32::from_le_bytes(version_buf);
        if version > PROJECT_VERSION {
            return Err(ProjectError::UnsupportedVersion(version));
        }

        // Decompress
        let mut decoder = GzDecoder::new(reader);

        // Read header JSON
        let mut len_buf = [0u8; 4];
        decoder.read_exact(&mut len_buf)?;
        let header_len = u32::from_le_bytes(len_buf) as usize;
        let header_bytes = super::read_exact_vec(&mut decoder, header_len)?;

        let (metadata, editor_state): (ProjectMetadata, EditorState) =
            serde_json::from_slice(&header_bytes)?;

        // Read chunk count
        decoder.read_exact(&mut len_buf)?;
        let chunk_count = u32::from_le_bytes(len_buf) as usize;

        // Read chunks. Cap the capacity hint so a bogus chunk_count from
        // a corrupt file can't request a huge eager allocation; the loop
        // below still reads the full count and fails via read_exact if
        // the data runs short.
        let mut chunks = Vec::with_capacity(chunk_count.min(MAX_CHUNK_HINT));
        for _ in 0..chunk_count {
            // Read position
            let mut pos_buf = [0u8; 4];
            decoder.read_exact(&mut pos_buf)?;
            let x = i32::from_le_bytes(pos_buf);
            decoder.read_exact(&mut pos_buf)?;
            let y = i32::from_le_bytes(pos_buf);
            decoder.read_exact(&mut pos_buf)?;
            let z = i32::from_le_bytes(pos_buf);

            // Read RLE data
            decoder.read_exact(&mut len_buf)?;
            let rle_len = u32::from_le_bytes(len_buf) as usize;
            let rle_data = super::read_exact_vec(&mut decoder, rle_len)?;

            chunks.push(ChunkData {
                pos: ChunkPos::new(x, y, z),
                rle_data,
            });
        }

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

/// Run-length decode chunk voxels
fn rle_decode_chunk(data: &[u8]) -> Option<Chunk> {
    let mut decoded: Vec<Voxel> = Vec::with_capacity(CHUNK_VOLUME);

    let mut offset = 0;
    while offset + 10 <= data.len() {
        // Read count (2 bytes)
        let count = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;

        // Read voxel (8 bytes)
        let voxel_bytes: [u8; 8] = data[offset..offset + 8].try_into().ok()?;
        let voxel: Voxel = *bytemuck::from_bytes(&voxel_bytes);
        offset += 8;

        // Add voxels
        for _ in 0..count {
            if decoded.len() >= CHUNK_VOLUME {
                break;
            }
            decoded.push(voxel);
        }
    }

    // Fill remaining with air if needed
    while decoded.len() < CHUNK_VOLUME {
        decoded.push(Voxel::AIR);
    }

    // Create chunk with decoded voxels
    let mut chunk = Chunk::new();
    for (i, voxel) in decoded.into_iter().enumerate().take(CHUNK_VOLUME) {
        let x = i % CHUNK_SIZE;
        let y = (i / CHUNK_SIZE) % CHUNK_SIZE;
        let z = i / (CHUNK_SIZE * CHUNK_SIZE);
        if voxel.is_solid() {
            chunk.set(x, y, z, voxel);
        }
    }

    Some(chunk)
}

/// Quick save world to file path (atomic + durable — see
/// [`write_project_atomic`]).
pub fn save_world(world: &World, path: &std::path::Path) -> Result<(), ProjectError> {
    write_project_atomic(&Project::from_world(world), path)
}

/// Save world with editor state to file path (atomic + durable — see
/// [`write_project_atomic`]).
pub fn save_world_with_state(
    world: &World,
    editor_state: EditorState,
    path: &std::path::Path,
) -> Result<(), ProjectError> {
    write_project_atomic(&Project::from_world_with_state(world, editor_state), path)
}

/// Serialize a project to `path` **atomically and durably**.
///
/// Writes to a sibling `<path>.tmp`, forces it to disk, then renames it
/// over the target. This closes two data-loss holes the plain
/// `File::create` + `BufWriter` path had:
///
/// - **Silent truncation on a flush error (#5).** `BufWriter`'s `Drop`
///   flushes but *ignores* any error, so a small project whose whole
///   gzip stream still sat in the 8 KiB buffer could report `Ok` having
///   written nothing to disk. `into_inner()` performs the final flush and
///   surfaces its error; `sync_all()` then forces the bytes to physical
///   storage before we treat the save as done.
/// - **A half-written target (#6).** `File::create` truncates the
///   destination up front, so a crash mid-write would destroy the
///   previous good save. Writing a temp then `fs::rename`-ing it over the
///   target (which replaces an existing file on Windows via `MoveFileExW`
///   as on POSIX; same directory ⇒ same volume) means the target is only
///   ever the complete old file or the complete new one.
///
/// On any failure the partial temp is removed rather than left behind.
fn write_project_atomic(project: &Project, path: &std::path::Path) -> Result<(), ProjectError> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
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
    Ok(())
}

/// Write `project` to `tmp`, flush it (surfacing the error `BufWriter`'s
/// `Drop` would swallow), and fsync it to physical disk. Split out so
/// [`write_project_atomic`] has a single `?`-using body with one cleanup
/// path for every early return.
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
    Ok(project.to_world())
}

/// Load world with editor state from file path
pub fn load_world_with_state(path: &std::path::Path) -> Result<(World, EditorState), ProjectError> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    let project = Project::load(&mut reader)?;
    Ok((project.to_world(), project.editor_state))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let loaded_world = loaded.to_world();

        assert!(loaded_world.get_voxel(0, 0, 0).is_solid());
        assert_eq!(loaded_world.get_voxel(0, 0, 0).r, 255);
        assert!(loaded_world.get_voxel(1, 1, 1).is_solid());
        assert_eq!(loaded_world.get_voxel(1, 1, 1).g, 255);
    }

    #[test]
    fn save_bytes_are_chunk_order_independent() {
        // The chunk store is a HashMap (per-instance, per-process random
        // iteration order), so two worlds with identical content built in
        // different orders could serialize to different bytes if the writer
        // didn't sort chunks. `sorted_chunk_positions` guarantees a
        // byte-identical `.vxlt` regardless of insertion order (#11 —
        // matters for backup dedup / content-addressing).
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
        // `test_project_roundtrip` checks a couple of voxels in one chunk.
        // This pins the two things most likely to regress if RLE / chunk-
        // index / header handling drifts: (a) full EditorState equality,
        // and (b) exact voxel round-trip across negative coordinates and
        // several chunks (incl. alpha).
        let mut world = World::new();
        let samples = [
            ((0, 0, 0), Voxel::from_rgb(255, 0, 0)),
            ((-1, -1, -1), Voxel::from_rgb(0, 255, 0)), // chunk (-1,-1,-1)
            ((31, 31, 31), Voxel::from_rgb(0, 0, 255)), // far corner of (0,0,0)
            ((32, 5, -33), Voxel::from_rgba(10, 20, 30, 200)), // chunk (1,0,-2)
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
            brush_flags: 0b11, // emissive + metallic both set
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
        let loaded_world = loaded.to_world();
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

    #[test]
    fn editor_state_without_sockets_field_still_loads() {
        // A `.vxlt` written before sockets existed has no `sockets` key
        // in its EditorState JSON. `#[serde(default)]` must fill it with
        // an empty Vec rather than failing the whole header parse —
        // otherwise the addition would silently brick every old project.
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
        // A crash (or force-kill) mid-write can leave a `.vxlt` truncated
        // at any offset. Loading ANY prefix of a valid file must return
        // Ok or Err — never panic — so a damaged autosave falls back to
        // the default scene instead of bricking startup.
        let mut world = World::new();
        for i in 0..40 {
            world.set_voxel(i, i % 8, (i * 2) % 16, Voxel::from_rgb((i * 6) as u8, 100, 200));
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
        // The roundtrip tests above serialize into an in-memory `Vec`, so
        // they never exercised the real File + BufWriter path where the #5
        // flush bug lived, nor the #6 temp-then-rename. This drives
        // `save_world_with_state` against an actual file: it must produce a
        // file that loads back (incl. brush flags/zone through the header),
        // atomically REPLACE an existing file on a second save, and leave
        // no `.tmp` behind.
        let dir = std::env::temp_dir().join("voxelith_atomic_save");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("proj.vxlt");
        let tmp = path.with_file_name("proj.vxlt.tmp"); // what the helper writes

        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgb(255, 0, 0));
        world.set_voxel(40, 2, -3, Voxel::from_rgba(1, 2, 3, 200)); // forces a 2nd chunk
        let state = EditorState {
            brush_color: [12, 34, 56, 200],
            brush_flags: 0b11, // emissive + metallic
            brush_tint_zone: 2,
            ..Default::default()
        };

        save_world_with_state(&world, state.clone(), &path).unwrap();
        assert!(path.exists(), "save produced no file");
        assert!(!tmp.exists(), "temp file left behind after a successful save");

        let (loaded_world, loaded_state) = load_world_with_state(&path).unwrap();
        assert_eq!(loaded_world.get_voxel(0, 0, 0).r, 255);
        assert_eq!(
            loaded_world.get_voxel(40, 2, -3),
            Voxel::from_rgba(1, 2, 3, 200)
        );
        assert_eq!(loaded_state.brush_color, [12, 34, 56, 200]);
        assert_eq!(loaded_state.brush_flags, 0b11);
        assert_eq!(loaded_state.brush_tint_zone, 2);

        // A second save over the SAME path must replace it wholesale — the
        // old red voxel is gone and the new green one is present (proves the
        // rename swapped the file rather than appending / partially writing).
        let mut world2 = World::new();
        world2.set_voxel(5, 5, 5, Voxel::from_rgb(0, 255, 0));
        save_world_with_state(&world2, EditorState::default(), &path).unwrap();
        let (loaded2, _) = load_world_with_state(&path).unwrap();
        assert!(
            loaded2.get_voxel(0, 0, 0).is_air(),
            "old content survived the overwrite"
        );
        assert_eq!(loaded2.get_voxel(5, 5, 5).g, 255);
        assert!(!tmp.exists(), "temp file left behind after overwrite");

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
