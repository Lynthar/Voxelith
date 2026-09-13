//! File I/O: the native `.vxlt` project format, `.vox` import/export,
//! and `.obj` / `.glb` export. The two glTF halves stay separate —
//! `gltf` writes by hand, `voxelize` parses with the `gltf` crate.

mod gltf;
mod obj;
mod project;
mod vox;
mod voxelize;

pub use gltf::{
    export_glb, export_glb_smoothed, export_glb_smoothed_with_transform, export_glb_with_transform,
    ExportTransform, GlbError, GlbStats, Pivot, SocketNode, UpAxis,
};
pub use obj::{export_obj, export_obj_smoothed, ObjError, ObjStats};
pub use project::{
    load_world, load_world_with_state, save_world_with_state, EditorState, Project, ProjectError,
    ProjectMetadata, SocketData, DEFAULT_CAMERA_POSITION,
};
pub use vox::{default_palette, export_vox, import_vox, VoxError, VoxImport, VoxModel};
pub use voxelize::voxelize_glb;

use std::io::{self, Read};
use std::path::Path;

use crate::core::World;

/// Read exactly `len` bytes without trusting `len` enough to
/// pre-allocate it: the buffer grows to the bytes actually present, so
/// peak allocation tracks real data rather than a declared length.
///
/// # Errors
/// `UnexpectedEof` when the stream holds fewer than `len` bytes.
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

/// Skip exactly `n` bytes by streaming them to a sink, so a huge
/// declared length can't trigger a huge allocation. `UnexpectedEof` if
/// the stream ends early.
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

/// How an exported mesh is built. The Export dialog, the bake spec and
/// the export reports all spell their three options from here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Surface {
    /// Greedy mesh — the voxels as they render.
    Blocky,
    /// Marching Cubes on the raw 0/1 density — rounded, keeps thin
    /// features.
    SmoothLight,
    /// Marching Cubes after a 3×3×3 blur — clay-like, may dissolve
    /// 1-cell features.
    SmoothHeavy,
}

impl Surface {
    pub const ALL: [Surface; 3] = [Surface::Blocky, Surface::SmoothLight, Surface::SmoothHeavy];

    /// The `smoothing` keyword a bake spec writes for this surface.
    pub fn bake_keyword(self) -> &'static str {
        match self {
            Surface::Blocky => "none",
            Surface::SmoothLight => "light",
            Surface::SmoothHeavy => "heavy",
        }
    }

    /// The surface a bake spec's `smoothing` keyword names.
    pub fn from_bake_keyword(keyword: &str) -> Option<Surface> {
        Self::ALL.into_iter().find(|s| s.bake_keyword() == keyword)
    }

    /// Geometry-source label for the export reports.
    pub fn mesh_source_label(self) -> &'static str {
        match self {
            Surface::Blocky => "Greedy mesh",
            Surface::SmoothLight => "Marching Cubes (light)",
            Surface::SmoothHeavy => "Marching Cubes (heavy)",
        }
    }

    /// `None` for the greedy mesh; `Some(blur)` for Marching Cubes,
    /// with or without the 3×3×3 density blur.
    fn blur(self) -> Option<bool> {
        match self {
            Surface::Blocky => None,
            Surface::SmoothLight => Some(false),
            Surface::SmoothHeavy => Some(true),
        }
    }
}

/// Export `world` as `.obj` with the given surface.
pub fn export_obj_surface(
    world: &World,
    path: &Path,
    surface: Surface,
) -> Result<ObjStats, ObjError> {
    match surface.blur() {
        None => export_obj(world, path),
        Some(blur) => export_obj_smoothed(world, path, blur),
    }
}

/// Export `world` and `sockets` as `.glb` with the given surface and
/// placement.
pub fn export_glb_surface(
    world: &World,
    sockets: &[SocketNode],
    path: &Path,
    surface: Surface,
    transform: ExportTransform,
) -> Result<GlbStats, GlbError> {
    match surface.blur() {
        None => export_glb_with_transform(world, sockets, path, transform),
        Some(blur) => export_glb_smoothed_with_transform(world, sockets, path, blur, transform),
    }
}

/// A mesh or voxel export target — one row per format. The Export
/// dialog's radios, the CLI's extension check and the report labels
/// all read this table, so a new format is one variant here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExportFormat {
    Glb,
    Obj,
    Vox,
}

impl ExportFormat {
    /// In dialog order, which is also how the CLI lists them.
    pub const ALL: [ExportFormat; 3] = [ExportFormat::Glb, ExportFormat::Obj, ExportFormat::Vox];

    pub fn extension(self) -> &'static str {
        match self {
            ExportFormat::Glb => "glb",
            ExportFormat::Obj => "obj",
            ExportFormat::Vox => "vox",
        }
    }

    /// The format's name, as the save dialog's file-type filter.
    pub fn name(self) -> &'static str {
        match self {
            ExportFormat::Glb => "glTF Binary",
            ExportFormat::Obj => "Wavefront OBJ",
            ExportFormat::Vox => "MagicaVoxel",
        }
    }

    /// `name (.ext)`: the dialog's radio label and the reports'
    /// `format` field.
    pub fn label(self) -> String {
        format!("{} (.{})", self.name(), self.extension())
    }

    /// The format `path`'s extension names, case-insensitively.
    pub fn from_path(path: &Path) -> Option<ExportFormat> {
        let ext = path.extension()?.to_string_lossy().to_lowercase();
        Self::ALL.into_iter().find(|f| f.extension() == ext)
    }
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

    #[test]
    fn every_export_format_round_trips_through_its_extension() {
        for format in ExportFormat::ALL {
            let path = Path::new("model").with_extension(format.extension());
            assert_eq!(ExportFormat::from_path(&path), Some(format));
            let shouty = path.with_extension(format.extension().to_uppercase());
            assert_eq!(ExportFormat::from_path(&shouty), Some(format));
        }
        assert_eq!(ExportFormat::from_path(Path::new("model.gltf")), None);
        assert_eq!(ExportFormat::from_path(Path::new("model")), None);
    }

    #[test]
    fn every_surface_round_trips_through_its_bake_keyword() {
        for surface in Surface::ALL {
            assert_eq!(
                Surface::from_bake_keyword(surface.bake_keyword()),
                Some(surface)
            );
        }
        assert_eq!(Surface::from_bake_keyword("medium"), None);
    }
}
