//! MagicaVoxel VOX format import/export.
//!
//! VOX is the native format for MagicaVoxel, a popular voxel editor.
//! This implementation supports reading and writing VOX files for
//! compatibility with the MagicaVoxel ecosystem.
//!
//! Format specification: https://github.com/ephtracy/voxel-model/blob/master/MagicaVoxel-file-format-vox.txt

use crate::core::{Voxel, World};
use std::collections::HashMap;
use std::io::{self, Read, Write};
use thiserror::Error;

/// VOX file magic number: "VOX "
const VOX_MAGIC: [u8; 4] = [b'V', b'O', b'X', b' '];
/// Supported VOX version
const VOX_VERSION: i32 = 150;

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

/// Voxel data for VOX format
pub struct VoxModel {
    /// Size of the model (x, y, z)
    pub size: (u32, u32, u32),
    /// Voxel positions and palette indices
    pub voxels: Vec<(u8, u8, u8, u8)>, // x, y, z, color_index
    /// Color palette (256 colors, RGBA)
    pub palette: [[u8; 4]; 256],
}

impl VoxModel {
    /// Create empty model
    pub fn new(size: (u32, u32, u32)) -> Self {
        Self {
            size,
            voxels: Vec::new(),
            palette: default_palette(),
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
                    // Palette full, find closest color
                    find_closest_color(&palette, color)
                };

                voxels.push((x as u8, y as u8, z as u8, color_index));
            }
        }

        Ok(Self {
            size: (size_x, size_y, size_z),
            voxels,
            palette,
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

    /// Read from VOX file
    pub fn read<R: Read>(reader: &mut R) -> Result<Self, VoxError> {
        // Read magic number
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != VOX_MAGIC {
            return Err(VoxError::InvalidMagic);
        }

        // Read version
        let mut version_buf = [0u8; 4];
        reader.read_exact(&mut version_buf)?;
        let version = i32::from_le_bytes(version_buf);
        if version != VOX_VERSION {
            // Try to read anyway, most versions are compatible
            log::warn!("VOX version {} (expected {}), attempting to read anyway", version, VOX_VERSION);
        }

        // Read MAIN chunk
        let main_header = ChunkHeader::read(reader)?;
        if &main_header.id != b"MAIN" {
            return Err(VoxError::InvalidChunkId(main_header.id));
        }

        let mut size: Option<(u32, u32, u32)> = None;
        let mut voxels: Vec<(u8, u8, u8, u8)> = Vec::new();
        let mut palette = default_palette();

        // Read child chunks
        let mut bytes_read = 0i32;
        while bytes_read < main_header.children_size {
            let chunk_header = ChunkHeader::read(reader)?;
            bytes_read += 12 + chunk_header.content_size + chunk_header.children_size;

            match &chunk_header.id {
                b"SIZE" => {
                    let mut buf = [0u8; 4];
                    reader.read_exact(&mut buf)?;
                    let x = u32::from_le_bytes(buf);
                    reader.read_exact(&mut buf)?;
                    let y = u32::from_le_bytes(buf);
                    reader.read_exact(&mut buf)?;
                    let z = u32::from_le_bytes(buf);
                    size = Some((x, y, z));
                }
                b"XYZI" => {
                    let mut buf = [0u8; 4];
                    reader.read_exact(&mut buf)?;
                    let num_voxels = i32::from_le_bytes(buf) as usize;

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
                }
                b"RGBA" => {
                    // Read 256 colors (last one is unused in some versions)
                    for i in 0..256 {
                        let mut color = [0u8; 4];
                        reader.read_exact(&mut color)?;
                        // VOX stores as RGBA, we keep it as RGBA
                        // Index 0 in file maps to index 1 in palette (0 is empty)
                        let palette_index = if i == 255 { 0 } else { i + 1 };
                        palette[palette_index] = color;
                    }
                }
                _ => {
                    // Skip unknown chunks
                    let mut skip_buf = vec![0u8; chunk_header.content_size as usize];
                    reader.read_exact(&mut skip_buf)?;
                }
            }

            // Skip children if any
            if chunk_header.children_size > 0 {
                let mut skip_buf = vec![0u8; chunk_header.children_size as usize];
                reader.read_exact(&mut skip_buf)?;
            }
        }

        let size = size.ok_or(VoxError::NoVoxelData)?;

        Ok(Self {
            size,
            voxels,
            palette,
        })
    }

    /// Write to VOX file
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<(), VoxError> {
        // Write header
        writer.write_all(&VOX_MAGIC)?;
        writer.write_all(&VOX_VERSION.to_le_bytes())?;

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

/// Export world to VOX file
pub fn export_vox<W: Write>(world: &World, writer: &mut W) -> Result<(), VoxError> {
    let model = VoxModel::from_world(world)?;
    model.write(writer)
}

/// Import world from VOX file
pub fn import_vox<R: Read>(reader: &mut R) -> Result<World, VoxError> {
    let model = VoxModel::read(reader)?;
    Ok(model.to_world())
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
        export_vox(&world, &mut buffer).unwrap();

        let imported = import_vox(&mut buffer.as_slice()).unwrap();

        assert!(imported.get_voxel(0, 0, 0).is_solid());
        assert!(imported.get_voxel(1, 0, 0).is_solid());
        assert!(imported.get_voxel(0, 1, 0).is_solid());
    }
}
