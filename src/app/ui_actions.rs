//! UiAction dispatch: drains the queue produced by the egui layer
//! and applies each action to the world/editor/renderer.

use voxelith::editor::{Command, VoxelChange};
use voxelith::procgen::{GenResult, VoxelGenerator, VoxelPatch};
use voxelith::ui::{CameraView, GeneratorChoice, UiAction};

use super::App;

impl App {
    /// Process all queued UI actions for this frame.
    pub(super) fn handle_ui_actions(&mut self) {
        for action in self.ui.state.take_actions() {
            match action {
                UiAction::Exit => {
                    self.save_prefs();
                    std::process::exit(0)
                }
                UiAction::Undo => {
                    self.editor.undo(&mut self.world);
                }
                UiAction::Redo => {
                    self.editor.redo(&mut self.world);
                }
                UiAction::ClearAll => {
                    self.world.clear();
                    self.editor.history.clear();
                    if let Some(renderer) = &mut self.renderer {
                        renderer.chunk_meshes.clear();
                    }
                }
                UiAction::GenerateTestCube => {
                    self.world.clear();
                    self.editor.history.clear();
                    self.world.create_test_cube((0, 8, 0), 4);
                    self.rebuild_all_meshes();
                }
                UiAction::GenerateGround => {
                    self.world.clear();
                    self.editor.history.clear();
                    self.world.create_test_ground(20, 2);
                    self.rebuild_all_meshes();
                }
                UiAction::GenerateSphere => {
                    self.world.clear();
                    self.editor.history.clear();
                    self.create_sphere((0, 10, 0), 6);
                    self.rebuild_all_meshes();
                }
                UiAction::GeneratePyramid => {
                    self.world.clear();
                    self.editor.history.clear();
                    self.create_pyramid((0, 0, 0), 10);
                    self.rebuild_all_meshes();
                }
                UiAction::ResetCamera => {
                    if let Some(renderer) = &mut self.renderer {
                        renderer.camera.target = glam::Vec3::ZERO;
                        renderer.camera_controller.distance = 40.0;
                        renderer.camera_controller.yaw = 0.0;
                        renderer.camera_controller.pitch = 0.5;
                        // Apply immediately so camera.position matches the
                        // new orbit state — without this, the camera would
                        // appear "stuck" until the next orbit drag, and
                        // that drag would start with a visible teleport.
                        renderer
                            .camera_controller
                            .update_camera_position(&mut renderer.camera);
                    }
                }
                UiAction::SetCameraView(view) => {
                    if let Some(renderer) = &mut self.renderer {
                        match view {
                            CameraView::Top => {
                                renderer.camera_controller.pitch = 1.5;
                                renderer.camera_controller.yaw = 0.0;
                            }
                            CameraView::Front => {
                                renderer.camera_controller.pitch = 0.0;
                                renderer.camera_controller.yaw = 0.0;
                            }
                            CameraView::Side => {
                                renderer.camera_controller.pitch = 0.0;
                                renderer.camera_controller.yaw =
                                    std::f32::consts::FRAC_PI_2;
                            }
                        }
                        // Same rationale as ResetCamera: apply now so the
                        // first orbit drag continues from this view rather
                        // than snapping from a stale spherical state.
                        renderer
                            .camera_controller
                            .update_camera_position(&mut renderer.camera);
                    }
                }
                UiAction::NewProject => self.new_project(),
                UiAction::OpenProject => self.open_project(),
                UiAction::OpenRecent(path) => self.do_open_project(path),
                UiAction::SaveProject => self.save_project(),
                UiAction::SaveAs => self.save_project_as(),
                UiAction::ImportVox => self.import_vox(),
                UiAction::ExportVox => self.export_vox(),
                UiAction::ExportObj => self.export_obj(),
                UiAction::GenerateProcedural => self.run_selected_generator(),
                UiAction::RunGraph => self.run_graph(),
            }
        }
    }

    /// Evaluate the pipeline graph and apply its output through
    /// `CommandHistory` so it's undo-able. Errors / fallback notes
    /// are surfaced via the status bar.
    fn run_graph(&mut self) {
        let result = self.ui.graph.evaluate();
        let patch = match result {
            Ok(p) => p,
            Err(e) => {
                log::error!("Graph evaluation failed: {}", e);
                self.ui.set_status(format!("Graph error: {}", e));
                return;
            }
        };

        if patch.is_empty() {
            self.ui.set_status("Graph produced no voxels");
            return;
        }

        let changes: Vec<VoxelChange> = patch
            .voxels
            .iter()
            .filter_map(|&(pos, new_voxel)| {
                let old_voxel = self.world.get_voxel(pos.0, pos.1, pos.2);
                if old_voxel == new_voxel {
                    None
                } else {
                    Some(VoxelChange {
                        pos,
                        old_voxel,
                        new_voxel,
                    })
                }
            })
            .collect();

        if changes.is_empty() {
            self.ui
                .set_status("Graph: no changes (output matches existing voxels)");
            return;
        }

        let count = changes.len();
        let cmd = Command::set_voxels(changes);
        self.editor.history.execute(cmd, &mut self.world);

        let mut status = format!("Graph: {} voxels", count);
        if !patch.notes.is_empty() {
            status.push_str(" (");
            status.push_str(&patch.notes.join("; "));
            status.push(')');
        }
        self.ui.set_status(status);

        self.invalidate_preview();
    }

    /// Run the procgen panel's currently-selected generator and apply
    /// the patch through `CommandHistory` so it's undo-able.
    fn run_selected_generator(&mut self) {
        // Dispatch by the panel's combo box. Each generator's params
        // live as fields on its concrete type, so we just call
        // `.generate()` on whichever the user picked.
        let result: GenResult<VoxelPatch> = match self.ui.procgen.selected {
            GeneratorChoice::Terrain => self.ui.procgen.terrain.generate(),
            GeneratorChoice::Tree => self.ui.procgen.tree.generate(),
            GeneratorChoice::Wfc => self.ui.procgen.wfc.generate(),
        };

        let patch = match result {
            Ok(p) => p,
            Err(e) => {
                log::error!("Generation failed: {}", e);
                self.ui.set_status(format!("Generation failed: {}", e));
                return;
            }
        };

        if patch.is_empty() {
            self.ui.set_status("Generation produced no voxels");
            return;
        }

        // Convert patch -> set_voxels command. The current voxel at
        // each position becomes `old_voxel` so undo restores the
        // pre-generation state. Identity writes are dropped so we
        // don't push a no-op command (e.g. re-running an unchanged
        // generator over the same world).
        let changes: Vec<VoxelChange> = patch
            .voxels
            .iter()
            .filter_map(|&(pos, new_voxel)| {
                let old_voxel = self.world.get_voxel(pos.0, pos.1, pos.2);
                if old_voxel == new_voxel {
                    None
                } else {
                    Some(VoxelChange {
                        pos,
                        old_voxel,
                        new_voxel,
                    })
                }
            })
            .collect();

        if changes.is_empty() {
            self.ui
                .set_status("No changes (output matches existing voxels)");
            return;
        }

        let count = changes.len();
        // Capture the static label before set_status takes &mut self.ui.
        let label = self.ui.procgen.selected.label();
        // `changes` was built by cloning out of patch.voxels, so patch
        // is still owned here — we can read its notes after building cmd.
        let cmd = Command::set_voxels(changes);
        self.editor.history.execute(cmd, &mut self.world);

        let mut status = format!("{}: {} voxels", label, count);
        if !patch.notes.is_empty() {
            status.push_str(" (");
            status.push_str(&patch.notes.join("; "));
            status.push(')');
        }
        self.ui.set_status(status);

        // The just-applied geometry would otherwise double-render with
        // the preview overlay on top of it. Clear the preview; it'll
        // regenerate on the next param change if still enabled.
        self.invalidate_preview();
    }
}
