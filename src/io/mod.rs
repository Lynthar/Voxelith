//! File I/O: project save/load, import/export.
//!
//! Supported formats:
//! - Native project format (.vxlt) - compressed binary with metadata
//! - MagicaVoxel (.vox) - import/export
//! - (Future) GLTF (.gltf, .glb) - export
//! - (Future) OBJ (.obj) - export

mod project;
mod vox;

pub use project::{
    EditorState, Project, ProjectError, ProjectMetadata,
    load_world, save_world,
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
}

impl FileFormat {
    /// Detect format from file extension
    pub fn from_extension(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_lowercase();
        match ext.as_str() {
            "vxlt" | "voxelith" => Some(Self::Project),
            "vox" => Some(Self::Vox),
            _ => None,
        }
    }

    /// Get default file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Project => "vxlt",
            Self::Vox => "vox",
        }
    }

    /// Get format name for display
    pub fn name(&self) -> &'static str {
        match self {
            Self::Project => "Voxelith Project",
            Self::Vox => "MagicaVoxel",
        }
    }

    /// Get file filter for file dialogs
    pub fn filter(&self) -> (&'static str, &'static [&'static str]) {
        match self {
            Self::Project => ("Voxelith Project", &["vxlt", "voxelith"]),
            Self::Vox => ("MagicaVoxel", &["vox"]),
        }
    }
}

/// All supported import formats
pub fn import_formats() -> Vec<FileFormat> {
    vec![FileFormat::Project, FileFormat::Vox]
}

/// All supported export formats
pub fn export_formats() -> Vec<FileFormat> {
    vec![FileFormat::Project, FileFormat::Vox]
}
