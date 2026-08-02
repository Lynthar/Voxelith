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

pub mod agent_ops;
pub mod ai;
pub mod bake;
pub mod exec;
pub mod core;
pub mod mesh;
pub mod editor;
pub mod io;
pub mod procgen;

// The editor half. Gated on `gui` so `--no-default-features` builds the
// library and the headless subcommands without winit / wgpu / egui in
// the tree at all. `prefs` sits here rather than below because it is
// exactly the editor's saved workspace (window, panels, viewport,
// recent files) — nothing headless reads it.
#[cfg(feature = "gui")]
pub mod prefs;
#[cfg(feature = "gui")]
pub mod render;
#[cfg(feature = "gui")]
pub mod ui;

// Re-export commonly used types
pub use core::{Voxel, Chunk, ChunkPos, World};
pub use mesh::{ChunkMesh, NaiveMesher, Mesher};
pub use editor::Editor;
#[cfg(feature = "gui")]
pub use render::Renderer;
#[cfg(feature = "gui")]
pub use ui::Ui;
