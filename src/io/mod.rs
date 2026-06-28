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

pub use gltf::{
    export_glb, export_glb_smoothed, export_glb_smoothed_with_transform,
    export_glb_with_transform, ExportTransform, GlbError, GlbStats, Pivot, SocketNode,
    UpAxis,
};
pub use obj::{export_obj, export_obj_smoothed, ObjError, ObjStats};
pub use project::{
    EditorState, Project, ProjectError, ProjectMetadata, SocketData,
    load_world, load_world_with_state, save_world, save_world_with_state,
};
pub use vox::{
    VoxError, VoxModel, default_palette,
    export_vox, import_vox,
};

use std::io::{self, Read};
use std::path::Path;

/// Read exactly `len` bytes from `reader` without trusting `len` enough
/// to pre-allocate it.
///
/// The voxel importers read length / count fields straight out of files
/// that may be corrupt or hostile. `vec![0u8; len]` would eagerly
/// reserve whatever the file claims — up to ~4 GiB from one bogus `u32`
/// — and the process aborts on that allocation before ever discovering
/// the stream is short. `Read::take(len).read_to_end` instead grows the
/// buffer only to the bytes actually present, so peak allocation tracks
/// real data, not the declared length. We still require the full `len`
/// bytes (matching `read_exact`) and report a short stream as
/// `UnexpectedEof`.
pub(super) fn read_exact_vec<R: Read>(reader: &mut R, len: usize) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    let read = reader.take(len as u64).read_to_end(&mut buf)?;
    if read != len {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "declared length exceeds available data",
        ));
    }
    Ok(buf)
}

/// Skip exactly `n` bytes of `reader` by streaming them to a sink, so a
/// huge declared length can't trigger a huge allocation (unlike
/// `vec![0u8; n]` + `read_exact`). Errors as `UnexpectedEof` if the
/// stream ends early.
pub(super) fn skip_bytes<R: Read>(reader: &mut R, n: u64) -> io::Result<()> {
    let copied = io::copy(&mut reader.take(n), &mut io::sink())?;
    if copied != n {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "declared chunk length exceeds available data",
        ));
    }
    Ok(())
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn read_exact_vec_reads_full_len() {
        let mut c = Cursor::new(vec![1u8, 2, 3, 4, 5]);
        assert_eq!(read_exact_vec(&mut c, 3).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn read_exact_vec_errors_on_short_stream_without_huge_alloc() {
        // Declares 4 GiB but only 4 bytes exist. Must surface
        // UnexpectedEof rather than attempt a 4 GiB allocation — this
        // is the whole point of routing importer reads through here.
        let mut c = Cursor::new(vec![0u8; 4]);
        let err = read_exact_vec(&mut c, 4 * 1024 * 1024 * 1024).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn skip_bytes_advances_then_reads() {
        let mut c = Cursor::new(vec![1u8, 2, 3, 4, 5]);
        skip_bytes(&mut c, 2).unwrap();
        assert_eq!(read_exact_vec(&mut c, 3).unwrap(), vec![3, 4, 5]);
    }

    #[test]
    fn skip_bytes_errors_on_short_stream() {
        let mut c = Cursor::new(vec![0u8; 4]);
        let err = skip_bytes(&mut c, 9_999_999_999).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }
}
