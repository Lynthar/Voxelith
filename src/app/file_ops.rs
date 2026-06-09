//! File operations: project new/save/open and VOX import/export.

use std::path::{Path, PathBuf};

use voxelith::{core::Voxel, editor::Tool, io};

use super::App;

impl App {
    /// Create a new empty project.
    pub(super) fn new_project(&mut self) {
        self.world.clear();
        self.editor.history.clear();
        self.project_path = None;
        self.unsaved_changes = false;
        if let Some(renderer) = &mut self.renderer {
            renderer.chunk_meshes.clear();
        }
        self.ui.set_status("New project created");
    }

    /// Snapshot the camera + brush / palette / tool into an
    /// `io::EditorState` for embedding in a saved or autosaved project.
    /// Falls back to defaults before the renderer exists. Shared by
    /// `do_save_project` and `App::tick_autosave`.
    pub(super) fn current_editor_state(&self) -> io::EditorState {
        let Some(renderer) = &self.renderer else {
            return io::EditorState::default();
        };
        io::EditorState {
            camera_position: [
                renderer.camera.position.x,
                renderer.camera.position.y,
                renderer.camera.position.z,
            ],
            camera_target: [
                renderer.camera.target.x,
                renderer.camera.target.y,
                renderer.camera.target.z,
            ],
            brush_color: [
                self.editor.brush_color.r,
                self.editor.brush_color.g,
                self.editor.brush_color.b,
                self.editor.brush_color.a,
            ],
            palette: self
                .editor
                .palette
                .iter()
                .map(|v| [v.r, v.g, v.b, v.a])
                .collect(),
            selected_tool: self.editor.current_tool as usize,
        }
    }

    /// Load a crash-recovery autosave into the editor. Mirrors
    /// `do_open_project`'s restore, but leaves `project_path` None (the
    /// recovery copy isn't the user's real file, so the next Save prompts
    /// for a location) and doesn't touch the recent-files MRU. Returns
    /// false — caller falls back to the default scene — if the file is
    /// unreadable.
    pub(super) fn recover_from_autosave(&mut self, path: &Path) -> bool {
        let (world, editor_state) = match io::load_world_with_state(path) {
            Ok(v) => v,
            Err(e) => {
                log::warn!("Failed to load autosave {}: {}", path.display(), e);
                return false;
            }
        };
        self.world = world;
        self.editor.history.clear();
        self.project_path = None;
        self.editor.brush_color = Voxel::from_rgba(
            editor_state.brush_color[0],
            editor_state.brush_color[1],
            editor_state.brush_color[2],
            editor_state.brush_color[3],
        );
        self.editor.palette = editor_state
            .palette
            .iter()
            .map(|c| Voxel::from_rgba(c[0], c[1], c[2], c[3]))
            .collect();
        self.editor.current_tool = match editor_state.selected_tool {
            0 => Tool::Place,
            1 => Tool::Remove,
            2 => Tool::Paint,
            3 => Tool::Eyedropper,
            4 => Tool::Fill,
            _ => Tool::Place,
        };
        if let Some(renderer) = &mut self.renderer {
            renderer.chunk_meshes.clear();
            renderer.camera.position = glam::Vec3::new(
                editor_state.camera_position[0],
                editor_state.camera_position[1],
                editor_state.camera_position[2],
            );
            renderer.camera.target = glam::Vec3::new(
                editor_state.camera_target[0],
                editor_state.camera_target[1],
                editor_state.camera_target[2],
            );
            renderer
                .camera_controller
                .sync_orbit_state_from_camera(&renderer.camera);
        }
        self.rebuild_all_meshes();
        self.ui
            .set_status("Recovered unsaved work — use Save As to keep it");
        true
    }

    /// Save to the current path, or prompt if there isn't one.
    pub(super) fn save_project(&mut self) {
        if let Some(path) = self.project_path.clone() {
            self.do_save_project(path);
        } else {
            self.save_project_as();
        }
    }

    /// Prompt for a path and save.
    pub(super) fn save_project_as(&mut self) {
        let dialog = rfd::FileDialog::new()
            .add_filter("Voxelith Project", &["vxlt"])
            .set_title("Save Project As");

        if let Some(path) = dialog.save_file() {
            self.do_save_project(path);
        }
    }

    fn do_save_project(&mut self, path: PathBuf) {
        let editor_state = self.current_editor_state();

        match io::save_world_with_state(&self.world, editor_state, &path) {
            Ok(_) => {
                self.project_path = Some(path.clone());
                self.unsaved_changes = false;
                self.touch_recent(&path);
                let filename = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("project");
                self.ui.set_status(format!("Saved: {}", filename));
            }
            Err(e) => {
                log::error!("Failed to save project {:?}: {}", path, e);
                self.show_write_error("Save failed", &path, "save", &e);
                self.ui.set_status(format!(
                    "Save failed: {} — your work is NOT saved",
                    file_label(&path)
                ));
            }
        }
    }

    /// Prompt for a path and open a project.
    pub(super) fn open_project(&mut self) {
        let dialog = rfd::FileDialog::new()
            .add_filter("Voxelith Project", &["vxlt"])
            .add_filter("All Files", &["*"])
            .set_title("Open Project");

        let Some(path) = dialog.pick_file() else {
            return;
        };
        self.do_open_project(path);
    }

    /// Open a project from a known path (used by `open_project` and
    /// the Open Recent menu). Touches the recent-files MRU on success.
    pub(super) fn do_open_project(&mut self, path: PathBuf) {
        match io::load_world_with_state(&path) {
            Ok((world, editor_state)) => {
                self.world = world;
                self.editor.history.clear();
                self.project_path = Some(path.clone());

                self.editor.brush_color = Voxel::from_rgba(
                    editor_state.brush_color[0],
                    editor_state.brush_color[1],
                    editor_state.brush_color[2],
                    editor_state.brush_color[3],
                );
                self.editor.palette = editor_state
                    .palette
                    .iter()
                    .map(|c| Voxel::from_rgba(c[0], c[1], c[2], c[3]))
                    .collect();
                self.editor.current_tool = match editor_state.selected_tool {
                    0 => Tool::Place,
                    1 => Tool::Remove,
                    2 => Tool::Paint,
                    3 => Tool::Eyedropper,
                    4 => Tool::Fill,
                    _ => Tool::Place,
                };

                if let Some(renderer) = &mut self.renderer {
                    renderer.chunk_meshes.clear();
                    renderer.camera.position = glam::Vec3::new(
                        editor_state.camera_position[0],
                        editor_state.camera_position[1],
                        editor_state.camera_position[2],
                    );
                    renderer.camera.target = glam::Vec3::new(
                        editor_state.camera_target[0],
                        editor_state.camera_target[1],
                        editor_state.camera_target[2],
                    );
                    // Full sync (yaw / pitch / distance) — setting only
                    // distance here used to leave yaw/pitch stale, so a
                    // post-load scroll or Reset Camera would teleport
                    // the camera (same root cause as the startup-state
                    // mismatch fixed in `Renderer::new`).
                    renderer
                        .camera_controller
                        .sync_orbit_state_from_camera(&renderer.camera);
                }

                self.rebuild_all_meshes();
                self.unsaved_changes = false;
                self.touch_recent(&path);
                let filename = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("project");
                self.ui.set_status(format!("Opened: {}", filename));
            }
            Err(e) => {
                log::error!("Failed to open project {:?}: {}", path, e);
                let (short, detail) = describe_project_open_error(&e, &path);
                self.show_error_dialog("Open failed", &detail);
                self.ui.set_status(short);
            }
        }
    }

    /// Prompt for a VOX file and import it.
    pub(super) fn import_vox(&mut self) {
        let dialog = rfd::FileDialog::new()
            .add_filter("MagicaVoxel", &["vox"])
            .set_title("Import MagicaVoxel File");

        let Some(path) = dialog.pick_file() else {
            return;
        };

        match std::fs::File::open(&path) {
            Ok(mut file) => match io::import_vox(&mut file) {
                Ok(world) => {
                    self.world = world;
                    self.editor.history.clear();
                    if let Some(renderer) = &mut self.renderer {
                        renderer.chunk_meshes.clear();
                    }
                    self.rebuild_all_meshes();
                    // Imported world replaces everything; the previous
                    // camera target is now meaningless. Anchor orbit
                    // pivot on the imported scene so middle-orbit
                    // immediately circles the new model. (`do_open_project`
                    // doesn't do this because it restores the saved
                    // camera pose verbatim — but .vox files don't carry
                    // camera state.)
                    self.recenter_camera_on_scene();
                    self.unsaved_changes = false;
                    self.touch_recent(&path);
                    let filename = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("file");
                    self.ui.set_status(format!("Imported: {}", filename));
                }
                Err(e) => {
                    log::error!("Failed to import VOX from {:?}: {}", path, e);
                    let (short, detail) = describe_vox_import_error(&e, &path);
                    self.show_error_dialog("Import failed", &detail);
                    self.ui.set_status(short);
                }
            },
            Err(e) => {
                log::error!("Failed to open file {:?}: {}", path, e);
                let detail = format!(
                    "Couldn't open \"{}\" — {}.\n\nCheck the file still exists \
                     and isn't locked by another app.",
                    file_label(&path),
                    e
                );
                self.show_error_dialog("Import failed", &detail);
                self.ui.set_status(format!("Import failed: {}", e));
            }
        }
    }

    /// OBJ export with Marching Cubes smoothing. `blur` selects the
    /// strength: `false` keeps thin features by running MC on the
    /// raw 0/1 density (rounded-cube look); `true` runs a 3×3×3 blur
    /// first for clay-like terrain output but dissolves sparse
    /// 1-cell features.
    pub(super) fn export_obj_smoothed(&mut self, blur: bool) {
        let title = if blur {
            "Export Smoothed OBJ (heavy / clay)"
        } else {
            "Export Smoothed OBJ (light / preserve detail)"
        };
        let dialog = rfd::FileDialog::new()
            .add_filter("Wavefront OBJ", &["obj"])
            .set_title(title);

        let Some(path) = dialog.save_file() else {
            return;
        };

        match io::export_obj_smoothed(&self.world, &path, blur) {
            Ok(stats) => {
                self.touch_recent(&path);
                let filename = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("file");
                let mode = if blur { "heavy" } else { "light" };
                let msg = if stats.triangle_count == 0 {
                    format!("Exported: {} (empty — no geometry)", filename)
                } else {
                    format!(
                        "Exported (smoothed, {}): {} ({} tris)",
                        mode, filename, stats.triangle_count
                    )
                };
                self.ui.set_status(msg);
            }
            Err(e) => {
                log::error!("Failed to export smoothed OBJ: {}", e);
                self.show_write_error("Export failed", &path, "export", &e);
                self.ui
                    .set_status(format!("Export failed: {}", file_label(&path)));
            }
        }
    }

    /// GLB export with Marching Cubes smoothing. `blur` matches
    /// `export_obj_smoothed`: light (no blur) preserves detail,
    /// heavy (3×3×3 blur) is clay-like and best for terrain.
    pub(super) fn export_glb_smoothed(&mut self, blur: bool) {
        let title = if blur {
            "Export Smoothed glTF Binary (heavy / clay)"
        } else {
            "Export Smoothed glTF Binary (light / preserve detail)"
        };
        let dialog = rfd::FileDialog::new()
            .add_filter("glTF Binary", &["glb"])
            .set_title(title);

        let Some(path) = dialog.save_file() else {
            return;
        };

        match io::export_glb_smoothed(&self.world, &path, blur) {
            Ok(stats) => {
                self.touch_recent(&path);
                let filename = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("file");
                let mode = if blur { "heavy" } else { "light" };
                let msg = if stats.triangle_count == 0 {
                    format!("Exported: {} (empty — no geometry)", filename)
                } else {
                    let kib = (stats.byte_size as f32) / 1024.0;
                    format!(
                        "Exported (smoothed, {}): {} ({} tris, {:.1} KiB)",
                        mode, filename, stats.triangle_count, kib
                    )
                };
                self.ui.set_status(msg);
            }
            Err(e) => {
                log::error!("Failed to export smoothed GLB: {}", e);
                self.show_write_error("Export failed", &path, "export", &e);
                self.ui
                    .set_status(format!("Export failed: {}", file_label(&path)));
            }
        }
    }

    /// Prompt for a path and export to glTF Binary (.glb). Same
    /// mesh-collection path as OBJ (greedy meshing across all
    /// chunks), but writes a single self-contained .glb that imports
    /// directly into Unity / Unreal / Godot / Blender. Status bar
    /// reports vertex / triangle / chunk counts and the resulting
    /// file size so the user can sanity-check large exports.
    pub(super) fn export_glb(&mut self) {
        let dialog = rfd::FileDialog::new()
            .add_filter("glTF Binary", &["glb"])
            .set_title("Export as glTF Binary");

        let Some(path) = dialog.save_file() else {
            return;
        };

        match io::export_glb(&self.world, &path) {
            Ok(stats) => {
                self.touch_recent(&path);
                let filename = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("file");
                let msg = if stats.triangle_count == 0 {
                    format!("Exported: {} (empty — no geometry)", filename)
                } else {
                    let kib = (stats.byte_size as f32) / 1024.0;
                    format!(
                        "Exported: {} ({} tris, {} chunks, {:.1} KiB)",
                        filename, stats.triangle_count, stats.chunk_count, kib
                    )
                };
                self.ui.set_status(msg);
            }
            Err(e) => {
                log::error!("Failed to export GLB: {}", e);
                self.show_write_error("Export failed", &path, "export", &e);
                self.ui
                    .set_status(format!("Export failed: {}", file_label(&path)));
            }
        }
    }

    /// Prompt for a path and export to OBJ. Walks every chunk, runs
    /// the greedy mesher to capture currently-visible geometry, and
    /// writes a single .obj with vertex colors. Touches the recent-
    /// files MRU on success and surfaces triangle counts in the status
    /// bar so the user knows the export wasn't silently empty.
    pub(super) fn export_obj(&mut self) {
        let dialog = rfd::FileDialog::new()
            .add_filter("Wavefront OBJ", &["obj"])
            .set_title("Export as Wavefront OBJ");

        let Some(path) = dialog.save_file() else {
            return;
        };

        match io::export_obj(&self.world, &path) {
            Ok(stats) => {
                self.touch_recent(&path);
                let filename = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("file");
                let msg = if stats.triangle_count == 0 {
                    format!("Exported: {} (empty — no geometry)", filename)
                } else {
                    format!(
                        "Exported: {} ({} tris, {} chunks)",
                        filename, stats.triangle_count, stats.chunk_count
                    )
                };
                self.ui.set_status(msg);
            }
            Err(e) => {
                log::error!("Failed to export OBJ: {}", e);
                self.show_write_error("Export failed", &path, "export", &e);
                self.ui
                    .set_status(format!("Export failed: {}", file_label(&path)));
            }
        }
    }

    /// Prompt for a path and export to VOX.
    pub(super) fn export_vox(&mut self) {
        let dialog = rfd::FileDialog::new()
            .add_filter("MagicaVoxel", &["vox"])
            .set_title("Export as MagicaVoxel");

        let Some(path) = dialog.save_file() else {
            return;
        };

        match std::fs::File::create(&path) {
            Ok(mut file) => match io::export_vox(&self.world, &mut file) {
                Ok(overflow) => {
                    self.touch_recent(&path);
                    let filename = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("file");
                    let msg = if overflow > 0 {
                        format!(
                            "Exported: {} ({} colors quantized — VOX is 255-color)",
                            filename, overflow
                        )
                    } else {
                        format!("Exported: {}", filename)
                    };
                    self.ui.set_status(msg);
                }
                Err(e) => {
                    log::error!("Failed to export VOX: {}", e);
                    self.show_write_error("Export failed", &path, "export", &e);
                    self.ui
                        .set_status(format!("Export failed: {}", file_label(&path)));
                }
            },
            Err(e) => {
                log::error!("Failed to create file {:?}: {}", path, e);
                self.show_write_error("Export failed", &path, "create", &e);
                self.ui
                    .set_status(format!("Export failed: {}", file_label(&path)));
            }
        }
    }
}

/// File name for messages, or a neutral fallback.
fn file_label(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("the file")
        .to_string()
}

impl App {
    /// Raise the in-app error dialog for a failed file operation. This is
    /// an egui window (see `Ui::show`), NOT a native `rfd::MessageDialog`
    /// — the latter exits the process on this winit + wgpu setup, which
    /// would turn every save/open/import error into a hard crash exactly
    /// when the user most needs the message. The `detail` carries the
    /// "why + recovery action"; callers also set a status-bar one-liner.
    pub(super) fn show_error_dialog(&mut self, title: &str, detail: &str) {
        self.ui.state.error_dialog = Some((title.to_string(), detail.to_string()));
    }

    /// Generic "couldn't write the file" error for save / export failures
    /// (usually permission / disk / path, not bad content). `verb` is the
    /// action word, e.g. "save" or "export".
    pub(super) fn show_write_error(
        &mut self,
        title: &str,
        path: &Path,
        verb: &str,
        err: &dyn std::fmt::Display,
    ) {
        let detail = format!(
            "Couldn't {} \"{}\" — {}.\n\nCheck you have write permission and free \
             disk space, then try a different location.",
            verb,
            file_label(path),
            err
        );
        self.show_error_dialog(title, &detail);
    }
}

/// Map a `VoxError` to (status-bar one-liner, dialog detail + recovery
/// action). The reason is specific so the user can tell a wrong-file from
/// an unsupported-version from a corrupt one.
fn describe_vox_import_error(e: &io::VoxError, path: &Path) -> (String, String) {
    let (reason, action): (String, &str) = match e {
        io::VoxError::InvalidMagic => (
            "not a MagicaVoxel .vox file (bad magic bytes)".to_string(),
            "Make sure you picked a .vox file exported from MagicaVoxel.",
        ),
        io::VoxError::UnsupportedVersion(v) => (
            format!("unsupported VOX version {} (Voxelith reads v150 and v200)", v),
            "Re-export the model as v150 from MagicaVoxel, then import again.",
        ),
        io::VoxError::ModelTooLarge => (
            "a model larger than the 256×256×256 VOX limit".to_string(),
            "Split or downscale the model below 256 on each axis.",
        ),
        io::VoxError::NoVoxelData => (
            "no voxel models in the file".to_string(),
            "The .vox has no SIZE/XYZI data — check how it was exported.",
        ),
        io::VoxError::InvalidChunkId(id) => (
            format!("an unexpected chunk tag {:?}", id),
            "The file is likely corrupt or uses an unsupported extension.",
        ),
        io::VoxError::InvalidPaletteIndex(i) => (
            format!("an invalid palette index {}", i),
            "The palette / voxel data is inconsistent — re-export the file.",
        ),
        io::VoxError::Io(inner) if inner.kind() == std::io::ErrorKind::UnexpectedEof => (
            "a truncated or corrupt file (ran out of data)".to_string(),
            "The .vox looks incomplete — re-download or re-export it.",
        ),
        io::VoxError::Io(inner) => (
            format!("a read error: {}", inner),
            "Check the file still exists and isn't locked by another app.",
        ),
    };
    let short = format!("Import failed: {}", reason);
    let detail = format!(
        "Couldn't import \"{}\" — {}.\n\n{}",
        file_label(path),
        reason,
        action
    );
    (short, detail)
}

/// Map a `ProjectError` to (status one-liner, dialog detail + action).
fn describe_project_open_error(e: &io::ProjectError, path: &Path) -> (String, String) {
    let (reason, action): (String, &str) = match e {
        io::ProjectError::InvalidMagic => (
            "not a Voxelith .vxlt project (bad magic bytes)".to_string(),
            "Pick a .vxlt project, or use File \u{25B8} Import for .vox models.",
        ),
        io::ProjectError::UnsupportedVersion(v) => (
            format!("saved in a newer project format (version {})", v),
            "Update Voxelith to open this project.",
        ),
        io::ProjectError::Json(inner) => (
            format!("a corrupt project header ({})", inner),
            "The header is damaged — try a backup or autosave copy.",
        ),
        io::ProjectError::Io(inner) if inner.kind() == std::io::ErrorKind::UnexpectedEof => (
            "truncated or corrupt (ran out of data)".to_string(),
            "The project looks incomplete — try a backup or autosave copy.",
        ),
        io::ProjectError::Io(inner) => (
            format!("a read error: {}", inner),
            "Check the file still exists and isn't locked by another app.",
        ),
        io::ProjectError::InvalidChunkData | io::ProjectError::DecompressionError => (
            "corrupt voxel data".to_string(),
            "The project body is damaged — try a backup or autosave copy.",
        ),
    };
    let short = format!("Open failed: {}", reason);
    let detail = format!(
        "Couldn't open \"{}\" — {}.\n\n{}",
        file_label(path),
        reason,
        action
    );
    (short, detail)
}
