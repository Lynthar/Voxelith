//! Vertex and mesh data structures for rendering.

use bytemuck::{Pod, Zeroable};
use crate::core::ChunkPos;

/// Vertex format for voxel rendering.
///
/// Layout optimized for GPU:
/// - Position: 3 floats (12 bytes)
/// - Normal: 3 floats (12 bytes)
/// - Color: 4 floats (16 bytes)
/// Total: 40 bytes per vertex
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct Vertex {
    /// World position
    pub position: [f32; 3],
    /// Surface normal
    pub normal: [f32; 3],
    /// RGBA color (normalized 0.0-1.0)
    pub color: [f32; 4],
}

impl Vertex {
    /// Create a new vertex
    pub fn new(position: [f32; 3], normal: [f32; 3], color: [f32; 4]) -> Self {
        Self {
            position,
            normal,
            color,
        }
    }

    /// Get the vertex buffer layout for wgpu
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Normal
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Color
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

/// Generated mesh for a single chunk
#[derive(Debug, Clone)]
pub struct ChunkMesh {
    /// Chunk position this mesh belongs to
    pub chunk_pos: ChunkPos,
    /// Vertex data
    pub vertices: Vec<Vertex>,
    /// Triangle indices
    pub indices: Vec<u32>,
}

impl ChunkMesh {
    /// Create an empty mesh
    pub fn new(chunk_pos: ChunkPos) -> Self {
        Self {
            chunk_pos,
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Create mesh with pre-allocated capacity
    pub fn with_capacity(chunk_pos: ChunkPos, vertex_capacity: usize, index_capacity: usize) -> Self {
        Self {
            chunk_pos,
            vertices: Vec::with_capacity(vertex_capacity),
            indices: Vec::with_capacity(index_capacity),
        }
    }

    /// Check if mesh is empty
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// Get number of triangles
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Get number of vertices
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Add a quad (two triangles) to the mesh
    pub fn add_quad(&mut self, vertices: [Vertex; 4]) {
        let base = self.vertices.len() as u32;

        // Add vertices
        self.vertices.extend_from_slice(&vertices);

        // Add indices for two triangles (counter-clockwise winding)
        // Triangle 1: 0, 1, 2
        // Triangle 2: 0, 2, 3
        self.indices.extend_from_slice(&[
            base,
            base + 1,
            base + 2,
            base,
            base + 2,
            base + 3,
        ]);
    }

    /// Clear all mesh data
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    /// Get vertex data as bytes for GPU upload
    pub fn vertex_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.vertices)
    }

    /// Get index data as bytes for GPU upload
    pub fn index_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_size() {
        assert_eq!(std::mem::size_of::<Vertex>(), 40);
    }

    #[test]
    fn test_chunk_mesh_quad() {
        let mut mesh = ChunkMesh::new(ChunkPos::ZERO);

        let v = Vertex::new([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]);
        mesh.add_quad([v, v, v, v]);

        assert_eq!(mesh.vertex_count(), 4);
        assert_eq!(mesh.triangle_count(), 2);
        assert_eq!(mesh.indices.len(), 6);
    }
}
