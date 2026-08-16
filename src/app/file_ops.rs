//! File operations: project new/save/open, VOX and GLB import, VOX
//! export, plus the poll that notices somebody else writing the open
//! project.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use voxelith::{
    editor::Command,
    editor::Socket,
    io,
    procgen::PipelineGraph,
    ui::{ExportKind, ExportReport, Surface},
};

use super::App;

/// How often `tick_disk_reload` stats the open project file. Short
/// enough that an agent's step shows up as it happens, long enough that
/// a burst doesn't re-mesh the scene per frame.
const DISK_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// What a poll of the open project file should lead to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DiskPoll {
    /// Nobody else has written it, or we couldn't read it this time.
    Ignore,
    /// It changed under us and there is nothing local to lose.
    Reload,
    /// It changed under us, but reloading would discard edits the user
    /// hasn't saved. Their copy wins; say so and leave it alone.
    WarnStale,
}

/// Decide what a poll found. An unreadable file is deliberately not
/// news: a save elsewhere is a temp-then-rename, and a poll landing in
/// that window sees the target briefly missing.
pub(super) fn classify_disk_poll(
    watched: Option<SystemTime>,
    on_disk: Option<SystemTime>,
    unsaved_changes: bool,
) -> DiskPoll {
    match (watched, on_disk) {
        (Some(watched), Some(on_disk)) if watched != on_disk => match unsaved_changes {
            true => DiskPoll::WarnStale,
            false => DiskPoll::Reload,
        },
        _ => DiskPoll::Ignore,
    }
}

/// Last-modified time of `path`, or `None` if it can't be read right
/// now — see `classify_disk_poll` for why that isn't an error here.
fn file_mtime(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

/// The camera pose a project carries, or `None` when it isn't one the
/// viewport can use. A position equal to its target is a degenerate
/// look-at that opens blank; non-finite coordinates are refused too.
pub(super) fn camera_from_state(state: &io::EditorState) -> Option<(glam::Vec3, glam::Vec3)> {
    let position = glam::Vec3::from_array(state.camera_position);
    let target = glam::Vec3::from_array(state.camera_target);
    let usable =
        position.is_finite() && target.is_finite() && position.distance_squared(target) > 1e-6;
    usable.then_some((position, target))
}

/// Rebuild the live `editor::Socket` list from a loaded `EditorState`.
/// Inverse of `current_editor_state`'s socket mapping; shared by the
/// open-project and crash-recovery restore paths.
fn sockets_from_state(state: &io::EditorState) -> Vec<Socket> {
    state
        .sockets
        .iter()
        .map(|s| Socket::new(s.name.clone(), s.position, s.normal))
        .collect()
}

/// Take the pipeline graph out of a loaded `EditorState`, normalized and
/// laid out. Loading keeps whatever the file holds, evaluable or not —
/// dropping a graph here would delete the recipe to protect a run.
fn graph_from_state(state: &io::EditorState) -> PipelineGraph {
    let mut graph = state.graph.clone();
    graph.normalize();
    if graph.all_at_origin() {
        graph.relayout();
    }
    graph
}

/// How much of a loaded `EditorState` to apply — the one place the
/// restore paths differ. Document state is always restored; the camera
/// and workspace echo only on open, never on a reload under the user.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum LoadKind {
    /// Open / crash recovery: the user asked for this file, so
    /// everything it carries applies.
    Open,
    /// The file changed under the open project (an agent's checkpoint):
    /// only the document itself follows.
    Reload,
}

/// Which remembered directory a file dialog should open in. Keeping
/// the mapping in one place stops each dialog site from inventing its
/// own idea of "where should this start".
pub(super) enum DialogStart {
    /// Project open / save — the folder of the most recent project.
    Project,
    /// Any asset export — where the user last wrote one.
    Export,
    /// `.vox` import — where the user last read one.
    Import,
}

impl App {
    /// Build a native file dialog owned by the main window.
    ///
    /// # Safety
    /// Every dialog must come from here. An ownerless one can open
    /// behind the app, whose modal loop then reads as a hang.
    pub(super) fn file_dialog(&self, start: DialogStart) -> rfd::FileDialog {
        let mut dialog = rfd::FileDialog::new();
        if let Some(window) = &self.window {
            dialog = dialog.set_parent(window.as_ref());
        }
        let dir = match start {
            DialogStart::Project => self
                .prefs
                .recent_files
                .first()
                .and_then(|p| p.parent())
                .map(|p| p.to_path_buf()),
            DialogStart::Export => self.prefs.last_export_dir.clone(),
            DialogStart::Import => self.prefs.last_import_dir.clone(),
        };
        // Skip a remembered directory that has since been deleted or
        // unmounted; rfd would land somewhere arbitrary rather than at
        // its own sensible default.
        if let Some(dir) = dir.filter(|d| d.is_dir()) {
            dialog = dialog.set_directory(dir);
        }
        dialog
    }

    /// Create a new empty project. Guarded by `App::guard_then` —
    /// callers must not invoke this directly.
    pub(super) fn new_project(&mut self) {
        self.document.world.clear();
        self.reset_scene_session_state();
        self.document.metadata = voxelith::io::ProjectMetadata::default();
        self.project_path = None;
        self.note_project_mtime();
        self.document.mark_saved();
        self.ui.set_status("New project created");
    }

    /// Snapshot the camera, brush, palette and tool into an
    /// `io::EditorState` for a save or autosave. Falls back to defaults
    /// before the renderer exists.
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
            // Carry the brush's material flags + tint zone so open /
            // crash-recovery can restore them; without these the load
            // path's `from_rgba` silently zeroes the brush mode (#8).
            brush_flags: self.editor.brush_color.flags,
            brush_tint_zone: self.editor.brush_color.tint_zone(),
            sockets: self
                .document
                .sockets
                .iter()
                .map(|s| io::SocketData {
                    name: s.name.clone(),
                    position: s.position,
                    normal: s.normal,
                })
                .collect(),
            // Document data, like the sockets above: the graph says how
            // this model was made, so it belongs to the project rather
            // than to whoever's machine the editor is running on.
            graph: self.document.graph.clone(),
        }
    }

    /// Build the glTF socket-node list (name + translation + derived
    /// rotation) from the live sockets, for the GLB export paths. The
    /// `+Y → normal` rotation convention lives in `Socket::rotation`.
    fn socket_export_nodes(&self) -> Vec<io::SocketNode> {
        self.document
            .sockets
            .iter()
            .map(|s| io::SocketNode {
                name: s.name.clone(),
                translation: s.position,
                rotation: s.rotation(),
            })
            .collect()
    }

    /// Apply a loaded `EditorState` per [`LoadKind`]. Returns whether a
    /// usable camera was applied, so the caller can frame the scene
    /// after its rebuild when there wasn't one.
    pub(super) fn apply_editor_state(&mut self, state: &io::EditorState, kind: LoadKind) -> bool {
        // Document state — every kind.
        self.document.sockets = sockets_from_state(state);
        self.document.graph = graph_from_state(state);
        if kind == LoadKind::Reload {
            return true; // camera untouched = nothing to re-frame
        }

        // Workspace echo — Open only. Rebuilding the brush from a color
        // zeroes its flags, so the material mode / tint zone ride along
        // explicitly (#8).
        self.editor.brush_color = super::brush_from_stored(state.brush_color);
        self.editor.brush_color.flags = state.brush_flags;
        self.editor.brush_color.set_tint_zone(state.brush_tint_zone);
        self.editor.palette = state
            .palette
            .iter()
            .map(|&c| super::brush_from_stored(c))
            .collect();
        self.editor.current_tool = super::tool_from_index(state.selected_tool as u8);

        // View state — Open only, and only when the pose is usable
        // (`camera_from_state` refuses the degenerate ones old files
        // and headless writers can hold).
        let camera = camera_from_state(state);
        if let (Some(renderer), Some((position, target))) = (&mut self.renderer, camera) {
            renderer.camera.position = position;
            renderer.camera.target = target;
            // Full sync (yaw / pitch / distance) — setting only part
            // used to leave the controller stale, so a post-load scroll
            // or Reset Camera teleported the camera.
            renderer
                .camera_controller
                .sync_orbit_state_from_camera(&renderer.camera);
        }
        camera.is_some()
    }

    /// Answer the recovery prompt's Recover button, once the
    /// unsaved-changes guard has cleared it (see
    /// `PendingAction::RecoverAutosave`).
    pub(super) fn recover_autosave(&mut self) {
        let Some(path) = Self::autosave_path() else {
            return;
        };
        if self.recover_from_autosave(&path) {
            // Recovered work has no file of its own, so it is unsaved
            // relative to anything on disk — otherwise a clean exit
            // sails through and deletes the autosave holding it.
            self.document.mark_recovered();
            self.last_autosave = std::time::Instant::now();
        } else {
            // Corrupt: set it aside rather than delete it. This may be
            // the only copy of the crashed session, and unreadable to
            // this loader is not unrecoverable by hand.
            let quarantine = path.with_extension("vxlt.corrupt");
            match std::fs::rename(&path, &quarantine) {
                Ok(()) => {
                    log::warn!(
                        "Autosave could not be read; kept at {}",
                        quarantine.display()
                    );
                    self.ui.set_status(format!(
                        "Couldn't read the autosave — a copy was kept as {}",
                        quarantine
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("autosave.vxlt.corrupt")
                    ));
                }
                Err(e) => {
                    log::warn!("Could not set the corrupt autosave aside: {e}");
                    self.ui
                        .set_status("Couldn't recover autosave — starting fresh");
                }
            }
        }
    }

    /// Load a crash-recovery autosave. Mirrors an open, but leaves
    /// `project_path` None — the recovery copy isn't the user's file —
    /// and skips the MRU. False means the caller should use the default.
    pub(super) fn recover_from_autosave(&mut self, path: &Path) -> bool {
        let (world, editor_state, metadata) = match io::load_world_with_state(path) {
            Ok(v) => v,
            Err(e) => {
                log::warn!("Failed to load autosave {}: {}", path.display(), e);
                return false;
            }
        };
        self.document.world = world;
        self.document.metadata = metadata;
        self.reset_scene_session_state();
        self.project_path = None;
        self.note_project_mtime();
        let had_camera = self.apply_editor_state(&editor_state, LoadKind::Open);
        self.rebuild_all_meshes();
        if !had_camera {
            self.recenter_camera_on_scene();
        }
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
        let dialog = self
            .file_dialog(DialogStart::Project)
            .add_filter("Voxelith Project", &["vxlt"])
            .set_title("Save Project As");

        if let Some(path) = dialog.save_file() {
            self.do_save_project(path);
        }
    }

    fn do_save_project(&mut self, path: PathBuf) {
        let editor_state = self.current_editor_state();

        let metadata = self.document.metadata.clone();
        match io::save_world_with_state(&self.document.world, editor_state, metadata, &path) {
            Ok(_) => {
                self.project_path = Some(path.clone());
                // Our own write — mark it, or the next poll reads it as
                // somebody else's and reloads what we just saved.
                self.note_project_mtime();
                self.document.mark_saved();
                self.touch_recent(&path);
                let filename = file_label(&path);
                self.ui.set_status(format!("Saved: {}", filename));
            }
            Err(e) => {
                log::error!("Failed to save project {:?}: {}", path, e);
                self.show_write_error("Save failed", &path, "save", &e, true);
                self.ui.set_status(format!(
                    "Save failed: {} — your work is NOT saved",
                    file_label(&path)
                ));
            }
        }
    }

    /// Prompt for a path and open a project.
    pub(super) fn open_project(&mut self) {
        let dialog = self
            .file_dialog(DialogStart::Project)
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
            Ok((world, editor_state, metadata)) => {
                self.document.world = world;
                self.document.metadata = metadata;
                self.reset_scene_session_state();
                self.project_path = Some(path.clone());

                let had_camera = self.apply_editor_state(&editor_state, LoadKind::Open);
                self.rebuild_all_meshes();
                // A project with no usable camera — one built headless —
                // keeps the viewport where it is and aims it at what
                // just loaded, rather than opening blank.
                if !had_camera {
                    self.recenter_camera_on_scene();
                }
                self.note_project_mtime();
                self.document.mark_saved();
                self.touch_recent(&path);
                let filename = file_label(&path);
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
        // Read the UI's Z-up→Y-up toggle (default on) up front so the
        // borrow doesn't tangle with the `&mut self` writes below.
        let convert_axes = self.ui.convert_vox_axes;
        let dialog = self
            .file_dialog(DialogStart::Import)
            .add_filter("MagicaVoxel", &["vox"])
            .set_title("Import MagicaVoxel File");

        let Some(path) = dialog.pick_file() else {
            return;
        };

        match std::fs::File::open(&path) {
            Ok(mut file) => match io::import_vox(&mut file, convert_axes) {
                Ok(world) => {
                    self.document.world = world;
                    self.reset_scene_session_state();
                    self.document.metadata = voxelith::io::ProjectMetadata::default();
                    // Detach from any open `.vxlt`: the imported model is
                    // a new document, so a later Save must prompt rather
                    // than overwrite the project that was open before.
                    self.project_path = None;
                    self.note_project_mtime();
                    self.rebuild_all_meshes();
                    // The imported world replaces everything, so anchor
                    // the orbit pivot on it. An open restores the saved
                    // pose instead, but `.vox` carries no camera.
                    self.recenter_camera_on_scene();
                    self.document.mark_saved();
                    // Imports seed the next import dialog rather than
                    // the project MRU — see `Prefs::touch_recent`.
                    self.prefs.remember_import_dir(&path);
                    let filename = file_label(&path);
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

    /// Import a `.glb` by voxelizing its mesh. Unlike `.vox` import this
    /// **adds to** the document as one undoable command, which is why it
    /// needs no unsaved-changes guard — Ctrl+Z puts it back.
    pub(super) fn import_glb(&mut self) {
        let resolution = self.ui.import_resolution;
        let dialog = self
            .file_dialog(DialogStart::Import)
            .add_filter("glTF Binary", &["glb", "gltf"])
            .set_title("Import glTF Mesh");

        let Some(path) = dialog.pick_file() else {
            return;
        };

        // Size gate before the whole-file read: everything downstream is
        // budgeted, but the read itself wasn't, so a mispicked
        // multi-gigabyte file landed in memory before anything spoke.
        const MAX_IMPORT_BYTES: u64 = 512 * 1024 * 1024;
        match std::fs::metadata(&path) {
            Ok(meta) if meta.len() > MAX_IMPORT_BYTES => {
                let detail = format!(
                    "\"{}\" is {} MiB; glTF import reads at most {} MiB.\n\n\
                     A mesh headed for a voxel grid doesn't need a file this \
                     size — re-export it smaller.",
                    file_label(&path),
                    meta.len() / (1024 * 1024),
                    MAX_IMPORT_BYTES / (1024 * 1024),
                );
                self.show_error_dialog("Import failed", &detail);
                self.ui.set_status("Import failed: file too large");
                return;
            }
            _ => {}
        }

        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) => {
                log::error!("Failed to read {:?}: {}", path, e);
                let detail = format!(
                    "Couldn't open \"{}\" — {}.\n\nCheck the file still exists \
                     and isn't locked by another app.",
                    file_label(&path),
                    e
                );
                self.show_error_dialog("Import failed", &detail);
                self.ui.set_status(format!("Import failed: {}", e));
                return;
            }
        };

        let patch = match io::voxelize_glb(&bytes, resolution) {
            Ok(patch) => patch,
            Err(e) => {
                log::error!("Failed to voxelize {:?}: {:#}", path, e);
                // A `.gltf` with sidecar buffers is the textbook export
                // and exactly what this refuses — it reads no file but
                // the one picked. Saying so beats a generic error.
                let sidecars = path
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("gltf"));
                let hint = match sidecars {
                    true => {
                        "\n\nA `.gltf` is only readable here when it carries its buffers \
                         and images inside itself. One that references a `.bin` or a `.png` \
                         beside it can't be: nothing outside the file you picked is read. \
                         Export it as a single `.glb` instead."
                    }
                    false => "",
                };
                let detail = format!(
                    "Couldn't turn \"{}\" into voxels — {:#}.\n\nThe file has to \
                     be a glTF binary containing at least one triangle mesh.{}",
                    file_label(&path),
                    e,
                    hint
                );
                self.show_error_dialog("Import failed", &detail);
                self.ui.set_status("Import failed: not a usable mesh");
                return;
            }
        };

        let changes = self.patch_to_changes(&patch);
        if changes.is_empty() {
            self.ui
                .set_status("Import: no changes (mesh matches existing voxels)");
            return;
        }

        let count = changes.len();
        self.last_generated_bounds = super::bounds_of(patch.voxels.iter().map(|&(p, _)| p));
        self.editor
            .history
            .execute(Command::set_voxels(changes), &mut self.document.world);
        self.rebuild_all_meshes();
        self.recenter_camera_on_scene();
        self.prefs.remember_import_dir(&path);
        let mut status = format!("Imported {} voxels from {}", count, file_label(&path));
        // A file that hit one of the walk's limits gives back part of
        // itself, and saying so is the difference between "that came in
        // wrong" and a model quietly missing half its geometry.
        if !patch.notes.is_empty() {
            status.push_str(" (");
            status.push_str(&patch.notes.join("; "));
            status.push(')');
        }
        self.ui.set_status(status);
        self.invalidate_preview();
    }

    /// Remember the project file's modification time, so the disk poll
    /// can tell somebody else's write from our own.
    ///
    /// # Safety
    /// Every write to the open project file must call this, or the poll
    /// reads the editor's own save as a foreign one and reloads it.
    pub(super) fn note_project_mtime(&mut self) {
        self.watched_mtime = self.project_path.as_deref().and_then(file_mtime);
        // Saving, opening or starting a new project all settle whatever
        // disagreement the strip was reporting: from here on this editor
        // and that file agree, or the file isn't ours any more.
        self.ui.state.disk_conflict = None;
    }

    /// Per-frame check for the open project changing underneath us — the
    /// human's half of headless agent editing. One writer at a time: the
    /// user's unsaved edits win and the reload is refused.
    pub(super) fn tick_disk_reload(&mut self) {
        let Some(path) = self.project_path.clone() else {
            return;
        };
        if self.last_disk_poll.elapsed() < DISK_POLL_INTERVAL {
            return;
        }
        self.last_disk_poll = Instant::now();

        let on_disk = file_mtime(&path);
        let verdict = classify_disk_poll(self.watched_mtime, on_disk, self.document.unsaved());
        // Take the new time once this poll is settled, so a refused
        // reload warns once. A *failed* one isn't settled — recording it
        // would retire a version the editor never read.
        let settled = match verdict {
            DiskPoll::Reload => self.reload_from_disk(&path),
            _ => true,
        };
        if let (Some(seen), true) = (on_disk, settled) {
            self.watched_mtime = Some(seen);
        }

        match verdict {
            DiskPoll::Ignore | DiskPoll::Reload => {}
            DiskPoll::WarnStale => {
                log::info!(
                    "{} changed on disk; not reloading over unsaved changes",
                    path.display()
                );
                // The strip, not just a status line: the refusal holds
                // for every later write too, so the user needs to be
                // able to find out *why* long after this frame.
                self.ui.state.disk_conflict = Some(file_label(&path));
            }
        }
    }

    /// Re-read the open project, restoring **only** the document — the
    /// camera and workspace stay where the user left them. Returns
    /// whether the file was actually read.
    fn reload_from_disk(&mut self, path: &Path) -> bool {
        let (world, state, metadata) = match io::load_world_with_state(path) {
            Ok(loaded) => loaded,
            Err(e) => {
                // Saves are temp-then-rename, so a torn read shouldn't
                // happen — but a project truncated by something else
                // must not silently empty the editor's copy.
                log::warn!("Reload of {} failed: {}", path.display(), e);
                self.ui.set_status(format!(
                    "{} changed on disk but couldn't be read — showing the last \
                     good version",
                    file_label(path)
                ));
                return false;
            }
        };
        self.document.world = world;
        self.document.metadata = metadata;
        self.reset_scene_session_state();
        self.apply_editor_state(&state, LoadKind::Reload);
        self.rebuild_all_meshes();
        // `rebuild_all_meshes` bumped the revision — true of an edit,
        // wrong here: this world came *out* of the user's file, so it is
        // exactly what the file holds.
        self.document.mark_saved();
        // Whatever the strip was reporting is settled: this editor now
        // shows what the file holds.
        self.ui.state.disk_conflict = None;
        self.ui
            .set_status(format!("Reloaded: {} (changed on disk)", file_label(path)));
        true
    }

    /// The strip's Reload button — take the file and drop the local
    /// edits. Confirmed in the UI before it gets here, because that is
    /// exactly what it discards.
    pub(super) fn reload_project_from_disk(&mut self) {
        let Some(path) = self.project_path.clone() else {
            return;
        };
        if self.reload_from_disk(&path) {
            // The poll's mark, brought up to date by hand: this version
            // has now been read. A failed read leaves it alone on
            // purpose — see `tick_disk_reload`.
            self.note_project_mtime();
        }
    }

    /// Run one export request. The dialog, the io call and the report
    /// all key off `kind`, so a new format is one arm here — and an
    /// illegal pairing can't be written down.
    pub(super) fn do_export(&mut self, kind: ExportKind) {
        let dialog = self
            .file_dialog(DialogStart::Export)
            .add_filter(kind_filter_name(kind), &[kind_extension(kind)])
            .set_title(kind_dialog_title(kind));
        let Some(path) = dialog.save_file() else {
            return;
        };
        match kind {
            ExportKind::Vox => self.export_vox_to(&path),
            ExportKind::Obj(surface) => self.export_obj_to(&path, surface),
            ExportKind::Glb(surface) => self.export_glb_to(&path, surface),
        }
    }

    /// Status-bar line for a mesh export, shared by OBJ and GLB. An
    /// empty scene says so instead of reporting a 0-triangle success.
    fn export_status(surface: Surface, filename: &str, triangles: usize, detail: &str) -> String {
        if triangles == 0 {
            return format!("Exported: {} (empty — no geometry)", filename);
        }
        match surface {
            Surface::Blocky => format!("Exported: {} ({})", filename, detail),
            Surface::SmoothLight => {
                format!("Exported (smoothed, light): {} ({})", filename, detail)
            }
            Surface::SmoothHeavy => {
                format!("Exported (smoothed, heavy): {} ({})", filename, detail)
            }
        }
    }

    fn export_obj_to(&mut self, path: &Path, surface: Surface) {
        let result = match surface {
            Surface::Blocky => io::export_obj(&self.document.world, path),
            Surface::SmoothLight => io::export_obj_smoothed(&self.document.world, path, false),
            Surface::SmoothHeavy => io::export_obj_smoothed(&self.document.world, path, true),
        };
        match result {
            Ok(stats) => {
                self.prefs.remember_export_dir(path);
                let detail = format!(
                    "{} tris, {} chunks",
                    stats.triangle_count, stats.chunk_count
                );
                self.ui.set_status(Self::export_status(
                    surface,
                    &file_label(path),
                    stats.triangle_count,
                    &detail,
                ));
                if stats.triangle_count > 0 {
                    self.set_export_report(
                        path,
                        ExportReport {
                            format: "Wavefront OBJ (.obj)".into(),
                            mesh_source: mesh_source_label(surface).into(),
                            triangles: Some(stats.triangle_count),
                            vertices: Some(stats.vertex_count),
                            chunks: Some(stats.chunk_count),
                            color_model: "Per-vertex RGBA".into(),
                            ..Default::default()
                        },
                    );
                }
            }
            Err(e) => {
                log::error!("Failed to export OBJ: {}", e);
                self.show_write_error(
                    "Export failed",
                    path,
                    "export",
                    &e,
                    matches!(e, io::ObjError::Io(_)),
                );
                self.ui
                    .set_status(format!("Export failed: {}", file_label(path)));
            }
        }
    }

    fn export_glb_to(&mut self, path: &Path, surface: Surface) {
        let sockets = self.socket_export_nodes();
        let result = match surface {
            Surface::Blocky => io::export_glb(&self.document.world, &sockets, path),
            Surface::SmoothLight => {
                io::export_glb_smoothed(&self.document.world, &sockets, path, false)
            }
            Surface::SmoothHeavy => {
                io::export_glb_smoothed(&self.document.world, &sockets, path, true)
            }
        };
        match result {
            Ok(stats) => {
                self.prefs.remember_export_dir(path);
                let kib = (stats.byte_size as f32) / 1024.0;
                let detail = match surface {
                    // The blocky line reports chunks, the smoothed ones
                    // don't (an MC mesh isn't per-chunk) — kept as the
                    // per-variant messages always were.
                    Surface::Blocky => format!(
                        "{} tris, {} chunks, {:.1} KiB",
                        stats.triangle_count, stats.chunk_count, kib
                    ),
                    _ => format!("{} tris, {:.1} KiB", stats.triangle_count, kib),
                };
                self.ui.set_status(Self::export_status(
                    surface,
                    &file_label(path),
                    stats.triangle_count,
                    &detail,
                ));
                if stats.triangle_count > 0 {
                    self.set_export_report(
                        path,
                        ExportReport {
                            format: "glTF Binary (.glb)".into(),
                            mesh_source: mesh_source_label(surface).into(),
                            triangles: Some(stats.triangle_count),
                            vertices: Some(stats.vertex_count),
                            chunks: Some(stats.chunk_count),
                            color_model: "Per-vertex RGBA".into(),
                            notes: socket_note(sockets.len()),
                            ..Default::default()
                        },
                    );
                }
            }
            Err(e) => {
                log::error!("Failed to export GLB: {}", e);
                self.show_write_error(
                    "Export failed",
                    path,
                    "export",
                    &e,
                    matches!(e, io::GlbError::Io(_)),
                );
                self.ui
                    .set_status(format!("Export failed: {}", file_label(path)));
            }
        }
    }

    fn export_vox_to(&mut self, path: &Path) {
        // Mirror the import convention on the way out (default on) so a
        // model exported to .vox opens upright in MagicaVoxel.
        let convert_axes = self.ui.convert_vox_axes;
        match std::fs::File::create(path) {
            Ok(mut file) => match io::export_vox(&self.document.world, &mut file, convert_axes) {
                Ok(overflow) => {
                    self.prefs.remember_export_dir(path);
                    let filename = file_label(path);
                    let msg = if overflow > 0 {
                        format!(
                            "Exported: {} ({} colors quantized — the VOX palette \
                                 holds 254)",
                            filename, overflow
                        )
                    } else {
                        format!("Exported: {}", filename)
                    };
                    self.ui.set_status(msg);
                    let mut notes = Vec::new();
                    if overflow > 0 {
                        notes.push(format!(
                            "{} colors quantized to the nearest of 254 \
                                 palette slots",
                            overflow
                        ));
                    }
                    self.set_export_report(
                        path,
                        ExportReport {
                            format: "MagicaVoxel (.vox)".into(),
                            color_model: "254-color palette".into(),
                            notes,
                            ..Default::default()
                        },
                    );
                }
                Err(e) => {
                    log::error!("Failed to export VOX: {}", e);
                    self.show_write_error(
                        "Export failed",
                        path,
                        "export",
                        &e,
                        matches!(e, io::VoxError::Io(_)),
                    );
                    self.ui
                        .set_status(format!("Export failed: {}", file_label(path)));
                }
            },
            Err(e) => {
                log::error!("Failed to create file {:?}: {}", path, e);
                self.show_write_error("Export failed", path, "create", &e, true);
                self.ui
                    .set_status(format!("Export failed: {}", file_label(path)));
            }
        }
    }
}

/// The save dialog's title for an export request.
fn kind_dialog_title(kind: ExportKind) -> &'static str {
    match kind {
        ExportKind::Vox => "Export as MagicaVoxel",
        ExportKind::Obj(Surface::Blocky) => "Export as Wavefront OBJ",
        ExportKind::Obj(Surface::SmoothLight) => "Export Smoothed OBJ (light / preserve detail)",
        ExportKind::Obj(Surface::SmoothHeavy) => "Export Smoothed OBJ (heavy / clay)",
        ExportKind::Glb(Surface::Blocky) => "Export as glTF Binary",
        ExportKind::Glb(Surface::SmoothLight) => {
            "Export Smoothed glTF Binary (light / preserve detail)"
        }
        ExportKind::Glb(Surface::SmoothHeavy) => "Export Smoothed glTF Binary (heavy / clay)",
    }
}

/// The save dialog's file-type filter name.
fn kind_filter_name(kind: ExportKind) -> &'static str {
    match kind {
        ExportKind::Vox => "MagicaVoxel",
        ExportKind::Obj(_) => "Wavefront OBJ",
        ExportKind::Glb(_) => "glTF Binary",
    }
}

/// The format's file extension.
fn kind_extension(kind: ExportKind) -> &'static str {
    match kind {
        ExportKind::Vox => "vox",
        ExportKind::Obj(_) => "obj",
        ExportKind::Glb(_) => "glb",
    }
}

/// Geometry-source label for the export report.
fn mesh_source_label(surface: Surface) -> &'static str {
    match surface {
        Surface::Blocky => "Greedy mesh",
        Surface::SmoothLight => "Marching Cubes (light)",
        Surface::SmoothHeavy => "Marching Cubes (heavy)",
    }
}

/// Export-report note line for emitted sockets, or empty when there
/// are none. Keeps the post-export summary honest about the empty
/// nodes that went into the .glb alongside the mesh.
fn socket_note(count: usize) -> Vec<String> {
    if count == 0 {
        Vec::new()
    } else {
        vec![format!(
            "{} named socket{} exported as glTF empty node{}",
            count,
            if count == 1 { "" } else { "s" },
            if count == 1 { "" } else { "s" },
        )]
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
    /// Raise the in-app error dialog for a failed file operation — an
    /// egui window, never an `rfd::MessageDialog`, which exits the
    /// process here. `detail` carries the why and the recovery action.
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
        writable_hint_applies: bool,
    ) {
        // The recovery hint is about the filesystem, so offer it only
        // for filesystem failures. A scene too spread out to smooth
        // already says what to do; adding "check permissions" misleads.
        let detail = if writable_hint_applies {
            format!(
                "Couldn't {} \"{}\" — {}.\n\nCheck you have write permission and \
                 free disk space, then try a different location.",
                verb,
                file_label(path),
                err
            )
        } else {
            format!("Couldn't {} \"{}\" — {}.", verb, file_label(path), err)
        };
        self.show_error_dialog(title, &detail);
    }

    /// Stash an [`ExportReport`] for the post-export dialog, filling in
    /// the file name and the on-disk size. Callers pass the
    /// format-specific fields and leave those two at their defaults.
    pub(super) fn set_export_report(&mut self, path: &Path, mut report: ExportReport) {
        report.filename = file_label(path);
        report.file_size = std::fs::metadata(path).ok().map(|m| m.len());
        self.ui.state.export_report = Some(report);
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
            format!(
                "unsupported VOX version {} (Voxelith reads v150 and v200)",
                v
            ),
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
        io::VoxError::InvalidChunkSize(id) => (
            format!("a corrupt {:?} chunk header (bad length)", id),
            "The .vox is damaged — re-download or re-export it.",
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
        // Version 0 was never written by any build, so it's damage, not
        // age — the "update Voxelith" advice would send the user
        // chasing a release that doesn't exist.
        io::ProjectError::UnsupportedVersion(0) => (
            "a corrupt version field".to_string(),
            "The file is damaged — try a backup or autosave copy.",
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
        io::ProjectError::LimitExceeded(what) => (
            format!("a {} past what any project can hold", what),
            "The file is corrupt (or not really a project) — try a backup or \
             autosave copy.",
        ),
        io::ProjectError::TrailingData => (
            "extra data after the model (inconsistent file)".to_string(),
            "The file is damaged — try a backup or autosave copy.",
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

#[cfg(test)]
mod tests {
    use super::*;

    fn at(secs: u64) -> Option<SystemTime> {
        Some(SystemTime::UNIX_EPOCH + Duration::from_secs(secs))
    }

    #[test]
    fn an_unchanged_file_is_left_alone() {
        assert_eq!(classify_disk_poll(at(10), at(10), false), DiskPoll::Ignore);
        assert_eq!(classify_disk_poll(at(10), at(10), true), DiskPoll::Ignore);
    }

    #[test]
    fn somebody_elses_write_reloads_when_there_is_nothing_to_lose() {
        assert_eq!(classify_disk_poll(at(10), at(11), false), DiskPoll::Reload);
    }

    #[test]
    fn unsaved_work_outranks_the_file_on_disk() {
        // Single-writer: this can see that the file moved, not merge two
        // versions of it, so the copy the user is still editing wins.
        assert_eq!(
            classify_disk_poll(at(10), at(11), true),
            DiskPoll::WarnStale
        );
    }

    #[test]
    fn a_file_that_cannot_be_read_right_now_is_not_a_change() {
        // A save elsewhere renames a temp over the target; a poll landing
        // inside that window sees nothing there. Reloading on that would
        // read a file that is about to be replaced.
        assert_eq!(classify_disk_poll(at(10), None, false), DiskPoll::Ignore);
        assert_eq!(classify_disk_poll(at(10), None, true), DiskPoll::Ignore);
    }

    #[test]
    fn a_project_built_headless_opens_with_a_usable_camera() {
        // `EditorState::default()` is what `exec` and the MCP server
        // write when they start from an empty world, so this is the
        // camera every agent-built project arrives with.
        let (position, target) =
            camera_from_state(&io::EditorState::default()).expect("must be usable");
        assert_ne!(position, target, "position == target is a blank viewport");
        assert_eq!(position.to_array(), io::DEFAULT_CAMERA_POSITION);
    }

    #[test]
    fn a_degenerate_camera_is_refused_rather_than_applied() {
        // Projects written before the default was fixed are on disk, and
        // a .vxlt is an external file that can hold anything at all.
        let zeroed = io::EditorState {
            camera_position: [0.0; 3],
            camera_target: [0.0; 3],
            ..Default::default()
        };
        assert!(camera_from_state(&zeroed).is_none());

        let not_a_number = io::EditorState {
            camera_position: [f32::NAN, 20.0, 40.0],
            camera_target: [0.0; 3],
            ..Default::default()
        };
        assert!(camera_from_state(&not_a_number).is_none());
    }

    #[test]
    fn a_real_saved_camera_is_kept_exactly() {
        let saved = io::EditorState {
            camera_position: [1.5, -2.0, 3.25],
            camera_target: [0.0, 4.0, -1.0],
            ..Default::default()
        };
        let (position, target) = camera_from_state(&saved).expect("must be usable");
        assert_eq!(position.to_array(), [1.5, -2.0, 3.25]);
        assert_eq!(target.to_array(), [0.0, 4.0, -1.0]);
    }

    #[test]
    fn a_reload_keeps_the_users_hands_where_they_are() {
        // The LoadKind split, end to end: a Reload applies the file's
        // document (sockets, graph) but must not touch the workspace —
        // brush, palette, tool stay as the user left them.
        let mut app = super::super::App::new();
        app.editor.brush_color = voxelith::core::Voxel::from_rgb(9, 9, 9);
        app.editor.select_tool(voxelith::editor::Tool::Fill);
        let state = io::EditorState {
            brush_color: [1, 2, 3, 255],
            selected_tool: 0, // Place
            sockets: vec![io::SocketData {
                name: "muzzle".into(),
                position: [1.0, 2.0, 3.0],
                normal: [0.0, 1.0, 0.0],
            }],
            ..Default::default()
        };

        app.apply_editor_state(&state, LoadKind::Reload);
        assert_eq!(
            app.document.sockets.len(),
            1,
            "document state follows the file"
        );
        assert_eq!(app.editor.brush_color.r, 9, "workspace stays the user's");
        assert_eq!(app.editor.current_tool, voxelith::editor::Tool::Fill);

        app.apply_editor_state(&state, LoadKind::Open);
        assert_eq!(
            app.editor.brush_color.r, 1,
            "an open applies the whole file"
        );
        assert_eq!(app.editor.current_tool, voxelith::editor::Tool::Place);
    }

    #[test]
    fn nothing_happens_before_the_first_mark() {
        // No mark means no project file open, or one whose time we never
        // managed to read — either way there is nothing to compare.
        assert_eq!(classify_disk_poll(None, at(11), false), DiskPoll::Ignore);
        assert_eq!(classify_disk_poll(None, None, false), DiskPoll::Ignore);
    }
}
