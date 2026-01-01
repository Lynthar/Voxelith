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
        let mut project = Self::new();

        for (pos, chunk_lock) in world.chunks() {
            let chunk = chunk_lock.read();
            if !chunk.is_empty() {
                let rle_data = rle_encode_chunk(&chunk);
                project.chunks.push(ChunkData {
                    pos: *pos,
                    rle_data,
                });
            }
        }

        project
    }

    /// Convert project to world
    pub fn to_world(&self) -> World {
        let mut world = World::new();

        for chunk_data in &self.chunks {
            if let Some(chunk) = rle_decode_chunk(&chunk_data.rle_data) {
                let chunk_lock = world.get_or_create_chunk(chunk_data.pos);
                *chunk_lock.write() = chunk;
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
        let mut header_bytes = vec![0u8; header_len];
        decoder.read_exact(&mut header_bytes)?;

        let (metadata, editor_state): (ProjectMetadata, EditorState) =
            serde_json::from_slice(&header_bytes)?;

        // Read chunk count
        decoder.read_exact(&mut len_buf)?;
        let chunk_count = u32::from_le_bytes(len_buf) as usize;

        // Read chunks
        let mut chunks = Vec::with_capacity(chunk_count);
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
            let mut rle_data = vec![0u8; rle_len];
            decoder.read_exact(&mut rle_data)?;

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

/// Quick save world to file path
pub fn save_world(world: &World, path: &std::path::Path) -> Result<(), ProjectError> {
    let project = Project::from_world(world);
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    project.save(&mut writer)
}

/// Quick load world from file path
pub fn load_world(path: &std::path::Path) -> Result<World, ProjectError> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    let project = Project::load(&mut reader)?;
    Ok(project.to_world())
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
