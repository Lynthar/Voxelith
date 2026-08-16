//! Voxelith — procedural-first voxel asset creation. Voxel storage,
//! meshing, rendering, procgen, I/O, and the headless entry points an
//! agent drives.

pub mod agent_ops;
pub mod bake;
pub mod core;
pub mod editor;
pub mod eval;
pub mod exec;
pub mod io;
pub mod mesh;
pub mod procgen;

// CPU ray-cast views. Beside `mesh` rather than inside `render`,
// because it needs no GPU and no window — which is the whole point.
pub mod view;

// The MCP server. Gated because it carries rmcp + schemars, which a
// build that only ever runs `bake` or `exec` has no use for.
#[cfg(feature = "mcp")]
pub mod mcp;

// The editor half, gated on `gui` so `--no-default-features` builds
// without winit / wgpu / egui in the tree. `prefs` is here because it
// is the editor's saved workspace — nothing headless reads it.
#[cfg(feature = "gui")]
pub mod prefs;
#[cfg(feature = "gui")]
pub mod render;
#[cfg(feature = "gui")]
pub mod ui;

// Re-export commonly used types
pub use core::{Chunk, ChunkPos, Voxel, World};
pub use editor::Editor;
pub use mesh::{ChunkMesh, Mesher, NaiveMesher};
#[cfg(feature = "gui")]
pub use render::Renderer;
#[cfg(feature = "gui")]
pub use ui::Ui;
