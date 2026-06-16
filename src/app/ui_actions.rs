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
                    self.delete_autosave();
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
                    self.editor.sockets.clear();
                    if let Some(renderer) = &mut self.renderer {
                        renderer.chunk_meshes.clear();
                    }
                }
                UiAction::CopySelection => self.copy_selection(),
                UiAction::CutSelection => self.cut_selection(),
                UiAction::PasteClipboard => self.paste_clipboard(false),
                UiAction::DeleteSelection => self.delete_selection(),
                UiAction::SelectAllSolid => self.select_all_solid(),
                UiAction::Deselect => {
                    self.selection_drag_anchor = None;
                    self.selection_move_anchor = None;
                    self.move_ghost_voxels.clear();
                    self.editor.selection = None;
                }
                UiAction::RotateSelection { axis, quarter } => {
                    self.rotate_selection(axis, quarter);
                }
                UiAction::MirrorSelection { axis } => {
                    self.mirror_selection(axis);
                }
                // Each Generate* replaces the whole scene. `replace_scene`
                // wipes world + history + stale GPU meshes before building
                // the new geometry (see its doc comment for why the mesh
                // wipe matters).
                UiAction::GenerateTestCube => {
                    self.replace_scene(|app| app.world.create_test_cube((0, 8, 0), 4));
                }
                UiAction::GenerateGround => {
                    self.replace_scene(|app| app.world.create_test_ground(20, 2));
                }
                UiAction::GenerateSphere => {
                    self.replace_scene(|app| app.create_sphere((0, 10, 0), 6));
                }
                UiAction::GeneratePyramid => {
                    self.replace_scene(|app| app.create_pyramid((0, 0, 0), 10));
                }
                UiAction::ResetCamera => {
                    // Reset camera target to the scene's AABB center
                    // (or origin if the world is empty) so the default
                    // view always faces the model. Pre-fix this set
                    // target to ZERO unconditionally, which placed the
                    // orbit pivot underground for any scene whose
                    // voxels sit above y=0.
                    let target = self
                        .world
                        .scene_center()
                        .unwrap_or(glam::Vec3::ZERO);
                    if let Some(renderer) = &mut self.renderer {
                        renderer.camera.target = target;
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
                UiAction::FrameAll => self.frame_all(),
                UiAction::FrameSelected => self.frame_selected(),
                UiAction::FrameGenerated => self.frame_generated(),
                UiAction::RecoverAutosave => {
                    if let Some(path) = Self::autosave_path() {
                        if self.recover_from_autosave(&path) {
                            self.unsaved_changes = false;
                            self.last_autosave = std::time::Instant::now();
                        } else {
                            // Corrupt / unreadable: drop it, keep the
                            // default scene already on screen.
                            self.delete_autosave();
                            self.ui
                                .set_status("Couldn't recover autosave — starting fresh");
                        }
                    }
                }
                UiAction::DiscardAutosave => {
                    self.delete_autosave();
                    self.ui.set_status("Discarded recovered work");
                }
                UiAction::NewProject => self.new_project(),
                UiAction::OpenProject => self.open_project(),
                UiAction::OpenRecent(path) => self.do_open_project(path),
                UiAction::SaveProject => self.save_project(),
                UiAction::SaveAs => self.save_project_as(),
                UiAction::ImportVox => self.import_vox(),
                UiAction::ExportVox => self.export_vox(),
                UiAction::ExportObj => self.export_obj(),
                UiAction::ExportObjSmoothedLight => self.export_obj_smoothed(false),
                UiAction::ExportObjSmoothedHeavy => self.export_obj_smoothed(true),
                UiAction::ExportGlb => self.export_glb(),
                UiAction::ExportGlbSmoothedLight => self.export_glb_smoothed(false),
                UiAction::ExportGlbSmoothedHeavy => self.export_glb_smoothed(true),
                UiAction::GenerateProcedural => self.run_selected_generator(),
                UiAction::RunGraph => self.run_graph(),
                UiAction::AiGenerate => self.start_ai_job(),
                UiAction::AiCancel => self.cancel_ai_job(),
                UiAction::AiSaveKey(key) => self.save_ai_key(key),
                UiAction::AiClearKey => self.clear_ai_key(),
            }
        }
    }

    /// Wholesale-replace the scene with freshly-built geometry: wipe
    /// the world, undo history, **and the stale GPU chunk meshes**,
    /// run `build`, re-mesh the new chunks, and re-anchor the orbit
    /// pivot on the new scene.
    ///
    /// The `chunk_meshes.clear()` is the load-bearing step. `World::
    /// clear()` only drops the chunks; `rebuild_all_meshes()` then
    /// re-meshes the *new* world's dirty chunks. Any chunk position the
    /// previous scene occupied but the new one doesn't is never visited
    /// again, so without this wipe its GPU mesh lingers and renders as
    /// ghost geometry over an otherwise-correct world. The file-ops
    /// paths (new/open/import) and ClearAll already do this; the
    /// Generate* menu items used to skip it.
    fn replace_scene(&mut self, build: impl FnOnce(&mut Self)) {
        self.world.clear();
        self.editor.history.clear();
        self.editor.sockets.clear();
        if let Some(renderer) = &mut self.renderer {
            renderer.chunk_meshes.clear();
        }
        build(self);
        self.rebuild_all_meshes();
        self.recenter_camera_on_scene();
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
        // Remember the generated footprint for the "Frame Generated"
        // camera action (uses the full patch, not just changed cells).
        self.last_generated_bounds = super::bounds_of(patch.voxels.iter().map(|&(p, _)| p));
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
        // Remember the generated footprint for the "Frame Generated"
        // camera action (uses the full patch, not just changed cells).
        self.last_generated_bounds = super::bounds_of(patch.voxels.iter().map(|&(p, _)| p));
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
