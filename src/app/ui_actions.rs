//! UiAction dispatch: drains the queue produced by the egui layer
//! and applies each action to the world/editor/renderer.

use voxelith::editor::{Command, VoxelChange};
use voxelith::procgen::VoxelPatch;
use voxelith::ui::{CameraView, UiAction};

use super::{App, GenerateKind, PendingAction};

impl App {
    /// Process all queued UI actions for this frame.
    pub(super) fn handle_ui_actions(&mut self) {
        for action in self.ui.state.take_actions() {
            match action {
                // Exit routes through the same guard and shutdown path
                // as the window's close button, so the menu item and
                // the X can't behave differently.
                UiAction::Exit => self.guard_then(PendingAction::Exit),
                UiAction::Undo => {
                    if self
                        .editor
                        .undo(&mut self.document.world, &mut self.document.graph)
                    {
                        // Voxel entries are flagged by the mesh rebuild;
                        // a graph-only transition reaches no chunk, so
                        // the step has to say so itself.
                        self.document.bump();
                    }
                }
                UiAction::Redo => {
                    if self
                        .editor
                        .redo(&mut self.document.world, &mut self.document.graph)
                    {
                        self.document.bump();
                    }
                }
                // Confirmed by the caller (the menu raises a confirm
                // dialog first) — this arm just does it.
                UiAction::ClearAll => {
                    // `world.clear()` drops chunks without dirtying any,
                    // so the rebuild notices nothing — flag the document
                    // here, but only when there was something to clear.
                    let had_content = self.document.world.scene_center().is_some()
                        || !self.document.sockets.is_empty()
                        || !self.document.graph.nodes.is_empty();
                    self.document.world.clear();
                    self.reset_scene_session_state();
                    if had_content {
                        self.document.bump();
                    }
                }
                UiAction::CopySelection => self.copy_selection(),
                UiAction::CutSelection => self.cut_selection(),
                UiAction::PasteClipboard { at_cursor } => self.paste_clipboard(at_cursor),
                UiAction::DeleteSelection => self.delete_selection(),
                UiAction::SelectAllSolid => self.select_all_solid(),
                UiAction::Deselect => self.deselect(),
                UiAction::RotateSelection { axis, quarter } => {
                    self.rotate_selection(axis, quarter);
                }
                UiAction::MirrorSelection { axis } => {
                    self.mirror_selection(axis);
                }
                // Each Generate* discards the whole scene, so it goes
                // through the unsaved-changes guard like New / Open.
                UiAction::GenerateTestCube => {
                    self.guard_then(PendingAction::Generate(GenerateKind::TestCube));
                }
                UiAction::GenerateGround => {
                    self.guard_then(PendingAction::Generate(GenerateKind::Ground));
                }
                UiAction::GenerateSphere => {
                    self.guard_then(PendingAction::Generate(GenerateKind::Sphere));
                }
                UiAction::GeneratePyramid => {
                    self.guard_then(PendingAction::Generate(GenerateKind::Pyramid));
                }
                UiAction::ResetCamera => {
                    // Reset Camera recenters the pivot while keeping the
                    // view direction; snapping to a fixed orientation is
                    // `SetCameraView`'s job. A no-op on an empty world.
                    self.recenter_camera_on_scene();
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
                                renderer.camera_controller.yaw = std::f32::consts::FRAC_PI_2;
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
                // Recovery replaces the scene, so it goes through the
                // guard like New and Open: its prompt is non-modal, and
                // edits made behind it are real work.
                UiAction::RecoverAutosave => {
                    self.guard_then(PendingAction::RecoverAutosave);
                }
                UiAction::DiscardAutosave => {
                    self.delete_autosave();
                    self.ui.set_status("Discarded recovered work");
                }
                UiAction::NewProject => self.guard_then(PendingAction::NewProject),
                UiAction::OpenProject => self.guard_then(PendingAction::OpenPicker),
                UiAction::OpenRecent(path) => {
                    self.guard_then(PendingAction::OpenPath(path));
                }
                UiAction::SaveProject => self.save_project(),
                UiAction::SaveAs => self.save_project_as(),
                UiAction::ReloadFromDisk => self.reload_project_from_disk(),
                UiAction::ImportVox => self.guard_then(PendingAction::ImportVox),
                UiAction::ImportGlb => self.import_glb(),
                UiAction::Export(kind) => self.do_export(kind),
                UiAction::RunGraph => self.run_graph(),
                UiAction::GraphEdited => self.document.bump(),
                UiAction::SocketsEdited => self.document.bump(),
                UiAction::AgentStart(port) => self.start_agent_bridge(port),
                UiAction::AgentStop => self.stop_agent_bridge(),
                UiAction::AgentApproval(approval) => self.set_agent_approval(approval),
                UiAction::AgentAccept => self.accept_agent_batch(),
                UiAction::AgentReject => self.reject_agent_batch(),

                // --- unsaved-changes prompt answers ---
                UiAction::UnsavedSave => {
                    self.save_project();
                    // `do_save_project` clears the flag on success, so a
                    // flag still set means the write failed or the
                    // picker was cancelled — don't destroy the scene.
                    if self.document.unsaved() {
                        self.drop_pending_guarded();
                        self.ui.set_status("Not saved — your work is still open");
                    } else if let Some(action) = self.pending_guarded.take() {
                        self.run_guarded(action);
                    }
                }
                UiAction::UnsavedDiscard => {
                    if let Some(action) = self.pending_guarded.take() {
                        self.run_guarded(action);
                    }
                }
                UiAction::UnsavedCancel => {
                    self.drop_pending_guarded();
                }
            }
        }
    }

    /// Replace the scene with a built-in demo shape. The wipe in
    /// `reset_scene_session_state` includes the GPU meshes: a chunk the
    /// new world doesn't occupy would otherwise linger as ghost geometry.
    pub(super) fn generate_scene(&mut self, kind: GenerateKind) {
        self.document.world.clear();
        self.reset_scene_session_state();
        match kind {
            GenerateKind::TestCube => self.document.world.create_test_cube((0, 8, 0), 4),
            GenerateKind::Ground => self.document.world.create_test_ground(20, 2),
            GenerateKind::Sphere => self.create_sphere((0, 10, 0), 6),
            GenerateKind::Pyramid => self.create_pyramid((0, 0, 0), 10),
        }
        self.rebuild_all_meshes();
        self.recenter_camera_on_scene();
    }

    /// Evaluate the pipeline graph and apply its output through
    /// `CommandHistory` so it's undo-able. Errors / fallback notes
    /// are surfaced via the status bar.
    fn run_graph(&mut self) {
        // The same two ceilings an agent's graph goes through, applied
        // to the panel's. This runs on the drawing thread, so a graph
        // out of a `.vxlt` doesn't fail here — it takes the editor.
        if let Err(refusal) = voxelith::agent_ops::check_graph(&self.document.graph) {
            log::warn!("Graph refused before evaluation: {}", refusal.message);
            self.ui.set_status(format!("Graph: {}", refusal.message));
            return;
        }
        let result = self.document.graph.evaluate();
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

        let changes = self.patch_to_changes(&patch);

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
        self.editor.history.execute(cmd, &mut self.document.world);

        let mut status = format!("Graph: {} voxels", count);
        if !patch.notes.is_empty() {
            status.push_str(" (");
            status.push_str(&patch.notes.join("; "));
            status.push(')');
        }
        self.ui.set_status(status);

        self.invalidate_preview();
    }

    /// Turn a generated [`VoxelPatch`] into an undoable change list,
    /// shared by the graph and glTF import: dedupe by position with the
    /// last write winning, then drop identity writes.
    pub(super) fn patch_to_changes(&self, patch: &VoxelPatch) -> Vec<VoxelChange> {
        patch
            .dedup_last_write()
            .into_iter()
            .filter_map(|(pos, new_voxel)| {
                let old_voxel = self.document.world.get_voxel(pos.0, pos.1, pos.2);
                (old_voxel != new_voxel).then_some(VoxelChange {
                    pos,
                    old_voxel,
                    new_voxel,
                })
            })
            .collect()
    }
}
