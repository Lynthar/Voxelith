//! UiAction dispatch: drains the queue produced by the egui layer
//! and applies each action to the world/editor/renderer.

use voxelith::editor::{Command, VoxelChange};
use voxelith::procgen::{GenResult, VoxelGenerator, VoxelPatch};
use voxelith::ui::{CameraView, GeneratorChoice, UiAction};

use super::{App, GenerateKind, PendingAction};

impl App {
    /// Process all queued UI actions for this frame.
    pub(super) fn handle_ui_actions(&mut self) {
        for action in self.ui.state.take_actions() {
            match action {
                // Exit routes through the same guard and the same
                // shutdown path as the window's close button, instead
                // of the `process::exit` it used to call — that skipped
                // every destructor and made the menu item behave
                // differently from the X.
                UiAction::Exit => self.guard_then(PendingAction::Exit),
                UiAction::Undo => {
                    self.editor.undo(&mut self.world);
                }
                UiAction::Redo => {
                    self.editor.redo(&mut self.world);
                }
                // Confirmed by the caller (the menu raises a confirm
                // dialog first) — this arm just does it.
                UiAction::ClearAll => {
                    self.world.clear();
                    self.reset_scene_session_state();
                }
                UiAction::CopySelection => self.copy_selection(),
                UiAction::CutSelection => self.cut_selection(),
                UiAction::PasteClipboard => self.paste_clipboard(false),
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
                    // "Reset Camera" recenters the orbit pivot on the
                    // scene while preserving the current view direction —
                    // identical to the F key (both route through
                    // recenter_camera_on_scene). Snapping back to a fixed
                    // default orientation is the job of SetCameraView
                    // (Top/Front/Side), not this action. No-op on an empty
                    // world (nothing to focus on).
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
                // Recovery replaces the scene, so it goes through the
                // unsaved-changes guard like New / Open — the prompt it
                // came from is non-modal, and edits made behind it are
                // real work the user never agreed to lose.
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
                UiAction::ExportVox => self.export_vox(),
                UiAction::ExportObj => self.export_obj(),
                UiAction::ExportObjSmoothedLight => self.export_obj_smoothed(false),
                UiAction::ExportObjSmoothedHeavy => self.export_obj_smoothed(true),
                UiAction::ExportGlb => self.export_glb(),
                UiAction::ExportGlbSmoothedLight => self.export_glb_smoothed(false),
                UiAction::ExportGlbSmoothedHeavy => self.export_glb_smoothed(true),
                UiAction::GenerateProcedural => self.run_selected_generator(),
                UiAction::RunGraph => self.run_graph(),
                UiAction::GraphEdited => self.mark_document_modified(),
                UiAction::AgentStart(port) => self.start_agent_bridge(port),
                UiAction::AgentStop => self.stop_agent_bridge(),
                UiAction::AgentApproval(approval) => self.set_agent_approval(approval),
                UiAction::AgentAccept => self.accept_agent_batch(),
                UiAction::AgentReject => self.reject_agent_batch(),

                // --- unsaved-changes prompt answers ---
                UiAction::UnsavedSave => {
                    self.save_project();
                    // `do_save_project` clears the flag on success. If
                    // it's still set the write failed or the user backed
                    // out of the Save As picker — either way, don't go
                    // ahead and destroy the scene.
                    if self.unsaved_changes {
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

    /// Replace the scene with one of the built-in demo shapes.
    ///
    /// `reset_scene_session_state` does the wipe, including the GPU
    /// chunk meshes — that part is load-bearing. `World::clear()` only
    /// drops the chunks; `rebuild_all_meshes()` then re-meshes the
    /// *new* world's dirty ones. Any chunk position the previous scene
    /// occupied but the new one doesn't is never visited again, so
    /// without the wipe its GPU mesh lingers and renders as ghost
    /// geometry over an otherwise-correct world.
    pub(super) fn generate_scene(&mut self, kind: GenerateKind) {
        self.world.clear();
        self.reset_scene_session_state();
        match kind {
            GenerateKind::TestCube => self.world.create_test_cube((0, 8, 0), 4),
            GenerateKind::Ground => self.world.create_test_ground(20, 2),
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
        // to the one in the panel. Not politeness: this runs on the
        // thread that draws, the evaluator descends recursively, and a
        // source generator sizes its buffer from its own parameters
        // before anything downstream gets a look — so a graph out of a
        // `.vxlt` (an external file, which can hold a chain long enough
        // to overflow the stack or a terrain node whose height span
        // overflows `i32`) doesn't fail here, it takes the editor.
        if let Err(refusal) = voxelith::agent_ops::check_graph(&self.ui.graph) {
            log::warn!("Graph refused before evaluation: {}", refusal.message);
            self.ui.set_status(format!("Graph: {}", refusal.message));
            return;
        }
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

    /// Turn a generated [`VoxelPatch`] into an undoable `SetVoxels`
    /// change list. Shared by the procgen panel, the graph and the
    /// glTF import so all three treat duplicate and identity writes
    /// identically:
    /// 1. **Dedupe by position, last write wins** (`VoxelPatch::
    ///    dedup_last_write`) — otherwise re-running a generator that
    ///    overwrites a cell flips it each run (see that method's docs).
    /// 2. **Drop identity writes** (cell already equals the final value)
    ///    so re-running over an unchanged world pushes no no-op undo entry.
    pub(super) fn patch_to_changes(&self, patch: &VoxelPatch) -> Vec<VoxelChange> {
        patch
            .dedup_last_write()
            .into_iter()
            .filter_map(|(pos, new_voxel)| {
                let old_voxel = self.world.get_voxel(pos.0, pos.1, pos.2);
                (old_voxel != new_voxel).then_some(VoxelChange {
                    pos,
                    old_voxel,
                    new_voxel,
                })
            })
            .collect()
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

        // Convert patch -> undoable set_voxels command: dedupe by position
        // then drop identity writes (see `patch_to_changes`).
        let changes = self.patch_to_changes(&patch);

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
        // the preview overlay on top of it. Clear the preview; if the
        // generator panel is still open the debounced (~150 ms) preview
        // refresh rebuilds it on its next tick (it re-evaluates on a timer,
        // not only when a parameter changes).
        self.invalidate_preview();
    }
}
