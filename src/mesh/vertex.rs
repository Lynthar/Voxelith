//! Vertex and mesh data structures for rendering.

use bytemuck::{Pod, Zeroable};
use crate::core::ChunkPos;

/// Vertex format for voxel rendering.
///
/// Layout optimized for GPU:
/// - Position: 3 floats (12 bytes)
/// - Normal: 3 floats (12 bytes)
/// - Color: 4 floats (16 bytes)
/// - AO: 1 float (4 bytes) — 0 = fully occluded, 1 = no occlusion
/// Total: 44 bytes per vertex
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct Vertex {
    /// World position
    pub position: [f32; 3],
    /// Surface normal
    pub normal: [f32; 3],
    /// RGBA color (normalized 0.0-1.0)
    pub color: [f32; 4],
    /// Per-vertex Ambient Occlusion in `[0, 1]`. 0 = fully
    /// occluded (sharp inside corner), 1 = no occlusion (open
    /// space). Mapped to brightness in the fragment shader as
    /// `ambient_min + (1 - ambient_min) * ao`. Defaults to 1.0
    /// when `Vertex::new` is used (back-compat for code that
    /// hasn't computed AO yet — patch previews, tests, etc.).
    pub ao: f32,
}

impl Vertex {
    /// Create a new vertex with no occlusion (`ao = 1.0`). Code
    /// paths that haven't yet computed AO (procgen preview, tests)
    /// can use this; the mesher emits via `new_with_ao` instead.
    pub fn new(position: [f32; 3], normal: [f32; 3], color: [f32; 4]) -> Self {
        Self::new_with_ao(position, normal, color, 1.0)
    }

    /// Create a new vertex with explicit AO factor (0 = fully
    /// occluded, 1 = none).
    pub fn new_with_ao(
        position: [f32; 3],
        normal: [f32; 3],
        color: [f32; 4],
        ao: f32,
    ) -> Self {
        Self {
            position,
            normal,
            color,
            ao,
        }
    }

    /// Get the vertex buffer layout for wgpu
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Position @ offset 0
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Normal @ offset 12
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Color @ offset 24
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // AO @ offset 40
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 10]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32,
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

    /// Add a quad (two triangles) to the mesh with the default
    /// 0–2 diagonal split. Fine for AO-uniform faces (or AO-less
    /// previews); for AO-shaded faces use `add_quad_with_ao_flip`
    /// so the diagonal can flip to follow the dark pair.
    pub fn add_quad(&mut self, vertices: [Vertex; 4]) {
        self.push_quad(vertices, false);
    }

    /// Add a quad picking the triangle split based on the 4 corner
    /// AO values (0fps standard rule). Flips the diagonal when the
    /// 0-2 corners are brighter than 1-3 so the visible fold runs
    /// through the darker pair — visually less jarring than a
    /// bright/dark split across one triangle.
    ///
    /// Both splits go through `push_quad`, which reverses the
    /// natural ABCD walk order so the resulting triangles are
    /// CCW-from-outside (cross product parallel to face normal).
    /// See `push_quad` for the full winding rationale.
    pub fn add_quad_with_ao_flip(&mut self, vertices: [Vertex; 4]) {
        let flip =
            vertices[0].ao + vertices[2].ao > vertices[1].ao + vertices[3].ao;
        self.push_quad(vertices, flip);
    }

    fn push_quad(&mut self, vertices: [Vertex; 4], flip: bool) {
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&vertices);
        // Triangle index order is REVERSED from the visual ABCD
        // walk so each triangle's cross product comes out parallel
        // to the face normal — i.e., vertices appear CCW *from
        // outside* in world space, matching the wgpu / glTF /
        // standard convention. `face_quad_vertices_sized` emits
        // vertices in a walk order that's CW from outside; the
        // reversed triangle indices below flip that to CCW. Don't
        // change without verifying with `test_winding_*` tests.
        if flip {
            // Diagonal split along 1-3 (B-D): triangles
            // (A, D, B) + (B, D, C).
            self.indices.extend_from_slice(&[
                base,
                base + 3,
                base + 1,
                base + 1,
                base + 3,
                base + 2,
            ]);
        } else {
            // Default split along 0-2 (A-C): triangles
            // (A, C, B) + (A, D, C).
            self.indices.extend_from_slice(&[
                base,
                base + 2,
                base + 1,
                base,
                base + 3,
                base + 2,
            ]);
        }
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
        assert_eq!(std::mem::size_of::<Vertex>(), 44);
    }

    #[test]
    fn test_default_ao_is_one() {
        let v = Vertex::new([0.0; 3], [0.0; 3], [1.0; 4]);
        assert_eq!(v.ao, 1.0);
    }

    #[test]
    fn test_add_quad_with_ao_flip_picks_correct_diagonal() {
        let mut mesh = ChunkMesh::new(ChunkPos::ZERO);
        // ao[0]=1, ao[1]=0, ao[2]=1, ao[3]=0 → 0+2 > 1+3 → flip
        let v0 = Vertex::new_with_ao([0.0; 3], [0.0; 3], [1.0; 4], 1.0);
        let v1 = Vertex::new_with_ao([1.0, 0.0, 0.0], [0.0; 3], [1.0; 4], 0.0);
        let v2 = Vertex::new_with_ao([1.0, 1.0, 0.0], [0.0; 3], [1.0; 4], 1.0);
        let v3 = Vertex::new_with_ao([0.0, 1.0, 0.0], [0.0; 3], [1.0; 4], 0.0);
        mesh.add_quad_with_ao_flip([v0, v1, v2, v3]);
        // Flipped split (1-3 diagonal), reversed for CCW-from-outside:
        // triangles (A,D,B) + (B,D,C) → indices 0,3,1, 1,3,2
        assert_eq!(mesh.indices, vec![0, 3, 1, 1, 3, 2]);
    }

    #[test]
    fn test_add_quad_with_ao_no_flip_uses_default_diagonal() {
        let mut mesh = ChunkMesh::new(ChunkPos::ZERO);
        // ao[0]=0, ao[1]=1, ao[2]=0, ao[3]=1 → 0+2 < 1+3 → no flip
        let v0 = Vertex::new_with_ao([0.0; 3], [0.0; 3], [1.0; 4], 0.0);
        let v1 = Vertex::new_with_ao([1.0, 0.0, 0.0], [0.0; 3], [1.0; 4], 1.0);
        let v2 = Vertex::new_with_ao([1.0, 1.0, 0.0], [0.0; 3], [1.0; 4], 0.0);
        let v3 = Vertex::new_with_ao([0.0, 1.0, 0.0], [0.0; 3], [1.0; 4], 1.0);
        mesh.add_quad_with_ao_flip([v0, v1, v2, v3]);
        // Default split (0-2 diagonal), reversed for CCW-from-outside:
        // triangles (A,C,B) + (A,D,C) → indices 0,2,1, 0,3,2
        assert_eq!(mesh.indices, vec![0, 2, 1, 0, 3, 2]);
    }

    #[test]
    fn test_winding_cross_parallel_to_face_normal() {
        // The wgpu / glTF / standard convention is "vertices CCW
        // from outside" — meaning each triangle's cross product
        // (v1-v0) × (v2-v0) should be PARALLEL to the face normal.
        // After `perspective_rh × look_at_rh` (det negative) this
        // becomes CCW in NDC = front per `FrontFace::Ccw`.
        //
        // This test runs through ALL 6 face directions for both
        // split paths and verifies cross direction.
        use crate::mesh::{face_quad_vertices_sized, Face};

        for face in Face::ALL {
            let normal = face.normal();
            let verts = face_quad_vertices_sized(0.0, 0.0, 0.0, face, 1.0, 1.0, [1.0; 4]);

            // Run through both paths and check each triangle.
            for ao_uniform in [false, true] {
                let mut mesh = ChunkMesh::new(ChunkPos::ZERO);
                let mut quad = verts;
                if ao_uniform {
                    // No flip: ao all 0
                    for v in &mut quad {
                        v.ao = 0.0;
                    }
                } else {
                    // Trigger flip: ao[0]=ao[2]=1, ao[1]=ao[3]=0
                    quad[0].ao = 1.0;
                    quad[2].ao = 1.0;
                    quad[1].ao = 0.0;
                    quad[3].ao = 0.0;
                }
                mesh.add_quad_with_ao_flip(quad);

                for tri in 0..2 {
                    let i0 = mesh.indices[tri * 3] as usize;
                    let i1 = mesh.indices[tri * 3 + 1] as usize;
                    let i2 = mesh.indices[tri * 3 + 2] as usize;
                    let v0 = mesh.vertices[i0].position;
                    let v1 = mesh.vertices[i1].position;
                    let v2 = mesh.vertices[i2].position;
                    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
                    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
                    let cross = [
                        e1[1] * e2[2] - e1[2] * e2[1],
                        e1[2] * e2[0] - e1[0] * e2[2],
                        e1[0] * e2[1] - e1[1] * e2[0],
                    ];
                    // Cross dot normal should be POSITIVE (parallel,
                    // not anti-parallel) for CCW-from-outside.
                    let dot = cross[0] * normal[0]
                        + cross[1] * normal[1]
                        + cross[2] * normal[2];
                    assert!(
                        dot > 0.0,
                        "Face {:?} triangle {} (ao_uniform={}): cross {:?} not parallel to normal {:?}, dot={}",
                        face,
                        tri,
                        ao_uniform,
                        cross,
                        normal,
                        dot,
                    );
                }
            }
        }
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
