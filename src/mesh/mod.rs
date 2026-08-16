//! Voxel chunks to triangle meshes: greedy (the render and export
//! default), naive (reference, shares the quad helpers), and Marching
//! Cubes (export only). `patch_to_mesh` renders a sparse voxel list.

mod ao;
mod greedy;
mod marching_cubes;
mod naive;
mod neighbors;
mod patch;
mod vertex;

pub use greedy::{mesh_chunk_by_material, GreedyMesher};
pub use marching_cubes::{mesh_world_smoothed, SmoothMeshError};
pub use naive::NaiveMesher;
pub use patch::patch_to_mesh;
pub use vertex::{ChunkMesh, Vertex};

pub(crate) use ao::{ao_to_f32, compute_face_ao, unpack_ao};

use crate::core::{ChunkPos, World};

/// A mesh generation strategy. Implementations get the world plus a
/// chunk position so they can read neighbors for boundary culling.
pub trait Mesher {
    /// Generate the mesh for the chunk at `chunk_pos`. Returns an empty
    /// mesh if the chunk doesn't exist or contains only air.
    fn generate(&self, world: &World, chunk_pos: ChunkPos) -> ChunkMesh;
}

/// Face direction for voxel faces
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Face {
    /// +X direction (right)
    PosX = 0,
    /// -X direction (left)
    NegX = 1,
    /// +Y direction (up)
    PosY = 2,
    /// -Y direction (down)
    NegY = 3,
    /// +Z direction (front)
    PosZ = 4,
    /// -Z direction (back)
    NegZ = 5,
}

impl Face {
    /// Get normal vector for this face
    pub fn normal(&self) -> [f32; 3] {
        match self {
            Face::PosX => [1.0, 0.0, 0.0],
            Face::NegX => [-1.0, 0.0, 0.0],
            Face::PosY => [0.0, 1.0, 0.0],
            Face::NegY => [0.0, -1.0, 0.0],
            Face::PosZ => [0.0, 0.0, 1.0],
            Face::NegZ => [0.0, 0.0, -1.0],
        }
    }

    /// Get direction offset
    pub fn offset(&self) -> (i32, i32, i32) {
        match self {
            Face::PosX => (1, 0, 0),
            Face::NegX => (-1, 0, 0),
            Face::PosY => (0, 1, 0),
            Face::NegY => (0, -1, 0),
            Face::PosZ => (0, 0, 1),
            Face::NegZ => (0, 0, -1),
        }
    }

    /// All six faces
    pub const ALL: [Face; 6] = [
        Face::PosX,
        Face::NegX,
        Face::PosY,
        Face::NegY,
        Face::PosZ,
        Face::NegZ,
    ];
}

/// Build the 4 vertices of a unit-cube face at integer voxel position
/// `(x, y, z)`. Vertices are CCW-wound when viewed from outside.
pub(crate) fn face_quad_vertices(
    x: f32,
    y: f32,
    z: f32,
    face: Face,
    color: [f32; 4],
) -> [Vertex; 4] {
    face_quad_vertices_sized(x, y, z, face, 1.0, 1.0, color)
}

/// The four vertices of a `w × h` face at cell `(x, y, z)`, walked
/// clockwise from outside; `ChunkMesh::push_quad` reverses the indices
/// to give CCW-from-outside, which `test_winding_*` pins.
pub(crate) fn face_quad_vertices_sized(
    x: f32,
    y: f32,
    z: f32,
    face: Face,
    w: f32,
    h: f32,
    color: [f32; 4],
) -> [Vertex; 4] {
    let normal = face.normal();

    match face {
        Face::PosX => [
            Vertex::new([x + 1.0, y, z], normal, color),
            Vertex::new([x + 1.0, y, z + w], normal, color),
            Vertex::new([x + 1.0, y + h, z + w], normal, color),
            Vertex::new([x + 1.0, y + h, z], normal, color),
        ],
        Face::NegX => [
            Vertex::new([x, y, z + w], normal, color),
            Vertex::new([x, y, z], normal, color),
            Vertex::new([x, y + h, z], normal, color),
            Vertex::new([x, y + h, z + w], normal, color),
        ],
        Face::PosY => [
            Vertex::new([x, y + 1.0, z], normal, color),
            Vertex::new([x + w, y + 1.0, z], normal, color),
            Vertex::new([x + w, y + 1.0, z + h], normal, color),
            Vertex::new([x, y + 1.0, z + h], normal, color),
        ],
        Face::NegY => [
            Vertex::new([x, y, z + h], normal, color),
            Vertex::new([x + w, y, z + h], normal, color),
            Vertex::new([x + w, y, z], normal, color),
            Vertex::new([x, y, z], normal, color),
        ],
        Face::PosZ => [
            Vertex::new([x + w, y, z + 1.0], normal, color),
            Vertex::new([x, y, z + 1.0], normal, color),
            Vertex::new([x, y + h, z + 1.0], normal, color),
            Vertex::new([x + w, y + h, z + 1.0], normal, color),
        ],
        Face::NegZ => [
            Vertex::new([x, y, z], normal, color),
            Vertex::new([x + w, y, z], normal, color),
            Vertex::new([x + w, y + h, z], normal, color),
            Vertex::new([x, y + h, z], normal, color),
        ],
    }
}

/// Cheap directional shading: top brightest, bottom darkest, sides in
/// between. Alpha passes through unchanged.
pub(crate) fn apply_face_shading(color: [f32; 4], face: Face) -> [f32; 4] {
    let shade = match face {
        Face::PosY => 1.0,
        Face::PosX | Face::NegZ => 0.85,
        Face::NegX | Face::PosZ => 0.75,
        Face::NegY => 0.6,
    };
    [
        color[0] * shade,
        color[1] * shade,
        color[2] * shade,
        color[3],
    ]
}

/// As [`face_quad_vertices_sized`], with explicit per-vertex AO in
/// `face_vertex_signs` order.
#[allow(clippy::too_many_arguments)] // flat quad geometry — a struct would just move eight names one level down
pub(crate) fn face_quad_vertices_sized_ao(
    x: f32,
    y: f32,
    z: f32,
    face: Face,
    w: f32,
    h: f32,
    color: [f32; 4],
    ao: [f32; 4],
) -> [Vertex; 4] {
    let mut verts = face_quad_vertices_sized(x, y, z, face, w, h, color);
    for (i, v) in verts.iter_mut().enumerate() {
        v.ao = ao[i];
    }
    verts
}
