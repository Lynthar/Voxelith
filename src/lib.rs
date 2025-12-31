//! Voxelith - Procedural-first voxel asset creation tool
//!
//! This library provides core functionality for:
//! - Voxel data storage and manipulation
//! - Mesh generation from voxel data
//! - GPU rendering with wgpu
//! - User interface with egui
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │              User Interface             │
//! │              (ui module)                │
//! ├─────────────────────────────────────────┤
//! │           Application Logic             │
//! │         (editor, commands)              │
//! ├─────────────────────────────────────────┤
//! │             Core Engine                 │
//! │   (core, mesh, render, procgen)         │
//! └─────────────────────────────────────────┘
//! ```

pub mod core;
pub mod mesh;
pub mod render;
pub mod ui;
pub mod editor;
pub mod io;
pub mod procgen;

// Re-export commonly used types
pub use core::{Voxel, Chunk, ChunkPos, World};
pub use mesh::{ChunkMesh, NaiveMesher, Mesher};
pub use render::Renderer;
pub use ui::Ui;
pub use editor::Editor;
