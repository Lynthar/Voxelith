//! File operations: project new/save/open and VOX import/export.

use std::path::PathBuf;

use voxelith::{core::Voxel, editor::Tool, io};

use super::App;

impl App {
    /// Create a new empty project.
    pub(super) fn new_project(&mut self) {
        self.world.clear();
        self.editor.history.clear();
        self.project_path = None;
        if let Some(renderer) = &mut self.renderer {
            renderer.chunk_meshes.clear();
        }
        self.ui.set_status("New project created");
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
        let editor_state = if let Some(renderer) = &self.renderer {
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
        } else {
            io::EditorState::default()
        };

        match io::save_world_with_state(&self.world, editor_state, &path) {
            Ok(_) => {
                self.project_path = Some(path.clone());
                self.touch_recent(&path);
                let filename = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("project");
                self.ui.set_status(format!("Saved: {}", filename));
            }
            Err(e) => {
                log::error!("Failed to save project: {}", e);
                self.ui.set_status(format!("Save failed: {}", e));
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
                    renderer.camera_controller.distance =
                        (renderer.camera.position - renderer.camera.target).length();
                }

                self.rebuild_all_meshes();
                self.touch_recent(&path);
                let filename = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("project");
                self.ui.set_status(format!("Opened: {}", filename));
            }
            Err(e) => {
                log::error!("Failed to open project: {}", e);
                self.ui.set_status(format!("Open failed: {}", e));
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
                    self.touch_recent(&path);
                    let filename = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("file");
                    self.ui.set_status(format!("Imported: {}", filename));
                }
                Err(e) => {
                    log::error!("Failed to import VOX: {}", e);
                    self.ui.set_status(format!("Import failed: {}", e));
                }
            },
            Err(e) => {
                log::error!("Failed to open file: {}", e);
                self.ui.set_status(format!("Open failed: {}", e));
            }
        }
    }

    /// Prompt for a path and export to OBJ. Walks every chunk, runs
    /// the naive mesher to capture currently-visible geometry, and
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
                self.ui.set_status(format!("Export failed: {}", e));
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
                    self.ui.set_status(format!("Export failed: {}", e));
                }
            },
            Err(e) => {
                log::error!("Failed to create file: {}", e);
                self.ui.set_status(format!("Create file failed: {}", e));
            }
        }
    }
}
