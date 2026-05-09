//! File I/O: project save/load, import/export.
//!
//! Supported formats:
//! - Native project format (.vxlt) - compressed binary with metadata
//! - MagicaVoxel (.vox) - import/export
//! - Wavefront OBJ (.obj) - export (geometry + vertex colors)
//! - glTF Binary (.glb) - export (single-file, native vertex colors)

mod gltf;
mod obj;
mod project;
mod vox;

pub use gltf::{export_glb, export_glb_smoothed, GlbError, GlbStats};
pub use obj::{export_obj, export_obj_smoothed, ObjError, ObjStats};
pub use project::{
    EditorState, Project, ProjectError, ProjectMetadata,
    load_world, load_world_with_state, save_world, save_world_with_state,
};
pub use vox::{
    VoxError, VoxModel, default_palette,
    export_vox, import_vox,
};

use std::path::Path;

/// Supported file formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    /// Native Voxelith project (.vxlt)
    Project,
    /// MagicaVoxel (.vox)
    Vox,
    /// Wavefront OBJ (.obj) — export only
    Obj,
    /// glTF Binary (.glb) — export only
    Glb,
}

impl FileFormat {
    /// Detect format from file extension
    pub fn from_extension(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_lowercase();
        match ext.as_str() {
            "vxlt" | "voxelith" => Some(Self::Project),
            "vox" => Some(Self::Vox),
            "obj" => Some(Self::Obj),
            "glb" => Some(Self::Glb),
            _ => None,
        }
    }

    /// Get default file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Project => "vxlt",
            Self::Vox => "vox",
            Self::Obj => "obj",
            Self::Glb => "glb",
        }
    }

    /// Get format name for display
    pub fn name(&self) -> &'static str {
        match self {
            Self::Project => "Voxelith Project",
            Self::Vox => "MagicaVoxel",
            Self::Obj => "Wavefront OBJ",
            Self::Glb => "glTF Binary",
        }
    }

    /// Get file filter for file dialogs
    pub fn filter(&self) -> (&'static str, &'static [&'static str]) {
        match self {
            Self::Project => ("Voxelith Project", &["vxlt", "voxelith"]),
            Self::Vox => ("MagicaVoxel", &["vox"]),
            Self::Obj => ("Wavefront OBJ", &["obj"]),
            Self::Glb => ("glTF Binary", &["glb"]),
        }
    }
}

/// All supported import formats
pub fn import_formats() -> Vec<FileFormat> {
    vec![FileFormat::Project, FileFormat::Vox]
}

/// All supported export formats
pub fn export_formats() -> Vec<FileFormat> {
    vec![
        FileFormat::Project,
        FileFormat::Vox,
        FileFormat::Obj,
        FileFormat::Glb,
    ]
}
