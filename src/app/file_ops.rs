//! File operations: project new/save/open, VOX and GLB import, VOX
//! export, plus the poll that notices somebody else writing the open
//! project.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use voxelith::{
    editor::Command, editor::Socket, io, procgen::PipelineGraph, ui::ExportReport,
};

use super::App;

/// How often `tick_disk_reload` stats the open project file. One
/// `metadata` call, so the cost is negligible either way; half a second
/// is short enough that an agent's step shows up as it happens and long
/// enough that a burst of them doesn't re-mesh the scene per frame.
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

/// Decide what a poll found. Split out from `tick_disk_reload` because
/// the interesting part is this table, and the rest is a clock and an
/// `fs::metadata` call.
///
/// An unreadable file is deliberately *not* news: a save elsewhere is a
/// write-temp-then-rename, and a poll landing in that window sees the
/// target briefly missing. Treating that as a change would reload from
/// a file that is about to be replaced anyway.
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

/// The camera pose a project carries, or `None` if it isn't one the
/// viewport can be pointed with.
///
/// A position equal to its target is a degenerate look-at: the view
/// matrix collapses and the viewport goes blank — no scene, no grid —
/// while the orbit controller derives yaw / pitch from a zero vector.
/// `EditorState::default()` no longer produces that, but projects
/// written before it stopped are on disk already, and a `.vxlt` is an
/// external file that can hold anything. Non-finite coordinates are
/// rejected on the same grounds.
pub(super) fn camera_from_state(state: &io::EditorState) -> Option<(glam::Vec3, glam::Vec3)> {
    let position = glam::Vec3::from_array(state.camera_position);
    let target = glam::Vec3::from_array(state.camera_target);
    let usable = position.is_finite()
        && target.is_finite()
        && position.distance_squared(target) > 1e-6;
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

/// Take the pipeline graph out of a loaded `EditorState`, ready for the
/// Graph panel.
///
/// `normalize` because the file is an external input — its `next_id` is
/// whatever wrote it, and a stale one hands the next added node an id
/// that is already taken. The re-layout covers the two writers that
/// have no business inventing panel coordinates: a build older than
/// `position`, and an agent, which sends nodes with no layout at all.
/// Loading keeps whatever the file holds, including a graph that can't
/// be evaluated: dropping one here would delete the recipe a model was
/// built from to protect a run nobody asked for yet. What makes
/// evaluation safe is checked where evaluation happens — `run_graph`
/// and the preview tick, both of which call `agent_ops::check_graph`
/// first.
fn graph_from_state(state: &io::EditorState) -> PipelineGraph {
    let mut graph = state.graph.clone();
    graph.normalize();
    if graph.all_at_origin() {
        graph.relayout();
    }
    graph
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
    /// `set_parent` is load-bearing, not cosmetic. An ownerless
    /// `IFileDialog` on Windows is an independent top-level window with
    /// no guarantee of sitting above ours, and it regularly opened
    /// *behind* the main window. Because the dialog runs its own modal
    /// message loop, the app stops rendering and ignores input while
    /// it's up — so what the user saw was a frozen program with no
    /// dialog anywhere on screen, indistinguishable from a hang, with
    /// force-quit (and every unsaved edit) as the obvious next step.
    /// Every dialog goes through here so a new call site can't
    /// reintroduce it.
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
        self.world.clear();
        self.reset_scene_session_state();
        self.project_path = None;
        self.note_project_mtime();
        self.unsaved_changes = false;
        self.autosave_pending = false;
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
            // Carry the brush's material flags + tint zone so open /
            // crash-recovery can restore them; without these the load
            // path's `from_rgba` silently zeroes the brush mode (#8).
            brush_flags: self.editor.brush_color.flags,
            brush_tint_zone: self.editor.brush_color.tint_zone(),
            sockets: self
                .editor
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
            graph: self.ui.graph.clone(),
        }
    }

    /// Build the glTF socket-node list (name + translation + derived
    /// rotation) from the live sockets, for the GLB export paths. The
    /// `+Y → normal` rotation convention lives in `Socket::rotation`.
    fn socket_export_nodes(&self) -> Vec<io::SocketNode> {
        self.editor
            .sockets
            .iter()
            .map(|s| io::SocketNode {
                name: s.name.clone(),
                translation: s.position,
                rotation: s.rotation(),
            })
            .collect()
    }

    /// Answer the recovery prompt's Recover button, once the
    /// unsaved-changes guard has cleared it (see
    /// `PendingAction::RecoverAutosave`).
    pub(super) fn recover_autosave(&mut self) {
        let Some(path) = Self::autosave_path() else {
            return;
        };
        if self.recover_from_autosave(&path) {
            // Recovered work has no file of its own (`project_path`
            // stays None), so relative to anything on disk it *is*
            // unsaved — say so, or the guard would wave a clean exit
            // through and the exit would then delete the autosave that
            // held the only copy.
            self.unsaved_changes = true;
            self.autosave_pending = false;
            self.last_autosave = std::time::Instant::now();
        } else {
            // Corrupt / unreadable: drop it, keep the default scene
            // already on screen.
            self.delete_autosave();
            self.ui
                .set_status("Couldn't recover autosave — starting fresh");
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
        self.reset_scene_session_state();
        self.project_path = None;
        self.note_project_mtime();
        self.editor.brush_color = super::brush_from_stored(editor_state.brush_color);
        // Same brush flag / tint-zone restore as `do_open_project` (#8) —
        // otherwise crash recovery drops the emissive / metallic /
        // faction-zone mode the user had when the autosave was written.
        self.editor.brush_color.flags = editor_state.brush_flags;
        self.editor
            .brush_color
            .set_tint_zone(editor_state.brush_tint_zone);
        self.editor.palette = editor_state
            .palette
            .iter()
            .map(|&c| super::brush_from_stored(c))
            .collect();
        self.editor.current_tool = super::tool_from_index(editor_state.selected_tool as u8);
        self.editor.sockets = sockets_from_state(&editor_state);
        self.ui.graph = graph_from_state(&editor_state);
        let camera = camera_from_state(&editor_state);
        if let (Some(renderer), Some((position, target))) = (&mut self.renderer, camera) {
            renderer.camera.position = position;
            renderer.camera.target = target;
            renderer
                .camera_controller
                .sync_orbit_state_from_camera(&renderer.camera);
        }
        self.rebuild_all_meshes();
        if camera.is_none() {
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

        match io::save_world_with_state(&self.world, editor_state, &path) {
            Ok(_) => {
                self.project_path = Some(path.clone());
                // Our own write — mark it, or the next poll reads it as
                // somebody else's and reloads what we just saved.
                self.note_project_mtime();
                self.unsaved_changes = false;
                self.autosave_pending = false;
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
            Ok((world, editor_state)) => {
                self.world = world;
                self.reset_scene_session_state();
                self.project_path = Some(path.clone());

                self.editor.brush_color = super::brush_from_stored(editor_state.brush_color);
                // Restore the brush's material flags / tint zone too —
                // rebuilding it from a color zeroes them, which used to
                // silently clear the emissive / metallic / faction-zone
                // mode on every open (#8). Round-trips via EditorState now.
                self.editor.brush_color.flags = editor_state.brush_flags;
                self.editor
                    .brush_color
                    .set_tint_zone(editor_state.brush_tint_zone);
                self.editor.palette = editor_state
                    .palette
                    .iter()
                    .map(|&c| super::brush_from_stored(c))
                    .collect();
                self.editor.current_tool =
                    super::tool_from_index(editor_state.selected_tool as u8);
                self.editor.sockets = sockets_from_state(&editor_state);
                self.ui.graph = graph_from_state(&editor_state);

                let camera = camera_from_state(&editor_state);
                if let (Some(renderer), Some((position, target))) =
                    (&mut self.renderer, camera)
                {
                    renderer.camera.position = position;
                    renderer.camera.target = target;
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
                // A project with no usable camera of its own — one an
                // agent built headless — keeps the viewport where it is
                // and aims it at what was just loaded, rather than
                // opening on a blank screen.
                if camera.is_none() {
                    self.recenter_camera_on_scene();
                }
                self.note_project_mtime();
                self.unsaved_changes = false;
                self.autosave_pending = false;
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
                    self.world = world;
                    self.reset_scene_session_state();
                    // Detach from any previously-open .vxlt: the imported
                    // model is a new document, so a later Save must prompt
                    // for a location instead of silently overwriting the
                    // project that was open before the import (#7). The
                    // source .vox stays on disk, so — like open/new —
                    // `unsaved_changes` is left false; the first edit arms
                    // autosave.
                    self.project_path = None;
                    self.note_project_mtime();
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
                    self.autosave_pending = false;
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

    /// Import a `.glb` / `.gltf` by voxelizing its mesh.
    ///
    /// Unlike `.vox` import this **adds to** the open document instead of
    /// replacing it, and lands as one undoable `SetVoxels` — the same
    /// path a generator's patch takes. That's why it needs no unsaved-
    /// changes guard: nothing is thrown away, and Ctrl+Z puts it back.
    ///
    /// The file is untrusted the way any opened file is, so this goes
    /// through `io::voxelize_glb`, which bounds what a GLB can make it
    /// allocate (see that module).
    pub(super) fn import_glb(&mut self) {
        let resolution = self.ui.import_resolution;
        let dialog = self
            .file_dialog(DialogStart::Import)
            .add_filter("glTF Binary", &["glb", "gltf"])
            .set_title("Import glTF Mesh");

        let Some(path) = dialog.pick_file() else {
            return;
        };

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
                // A `.gltf` that keeps its buffers and images in
                // sidecar files is the textbook export, and it is
                // exactly what this importer refuses: it reads no file
                // but the one the user picked, on purpose. Saying so
                // beats "Reading GLB buffers" and a shrug.
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
            .execute(Command::set_voxels(changes), &mut self.world);
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

    /// Remember the open project file's modification time, so the disk
    /// poll can tell somebody else's write from our own. Called wherever
    /// `project_path` changes or we write that file ourselves; with no
    /// project open it clears the mark, which is what stops a poll from
    /// firing against a path we no longer hold.
    pub(super) fn note_project_mtime(&mut self) {
        self.watched_mtime = self.project_path.as_deref().and_then(file_mtime);
        // Saving, opening or starting a new project all settle whatever
        // disagreement the strip was reporting: from here on this editor
        // and that file agree, or the file isn't ours any more.
        self.ui.state.disk_conflict = None;
    }

    /// Per-frame check for the open project changing underneath us.
    ///
    /// This is the human's half of agent editing: an agent working
    /// through `voxelith mcp --checkpoint` (or a shell loop running
    /// `voxelith exec --out`) writes the same `.vxlt` the editor has
    /// open, and the editor follows along step by step instead of
    /// showing a world that stopped being true several batches ago.
    ///
    /// One writer at a time, though. If the user has their own unsaved
    /// edits, theirs win and the reload is refused — this can tell that
    /// the file moved, not how to merge two versions of it.
    pub(super) fn tick_disk_reload(&mut self) {
        let Some(path) = self.project_path.clone() else {
            return;
        };
        if self.last_disk_poll.elapsed() < DISK_POLL_INTERVAL {
            return;
        }
        self.last_disk_poll = Instant::now();

        let on_disk = file_mtime(&path);
        let verdict = classify_disk_poll(self.watched_mtime, on_disk, self.unsaved_changes);
        // Take the new time when this poll is settled — a refused
        // reload warns once rather than on every poll from here on. A
        // reload that *failed* is not settled: the file is still one
        // this editor hasn't read, and recording its time would retire
        // that version for good, so a lock held for one poll interval
        // (a virus scanner or an indexer between rename and close) would
        // cost the user an agent's whole batch, announced by one status
        // line that scrolls away.
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

    /// Re-read the open project from disk, keeping the user where they
    /// were.
    ///
    /// Unlike `do_open_project` this restores **only** the world and its
    /// sockets: camera, brush, palette and tool stay as the user left
    /// them. The file carries all of those, but applying them here would
    /// yank the camera back to wherever it was pointing when the project
    /// was last saved — every time an agent finished a batch, which is
    /// the whole reason this path exists.
    ///
    /// The scene *is* replaced, so this goes through
    /// `reset_scene_session_state` like every other path that throws one
    /// away: a selection or an undo entry left over from the old world
    /// addresses cells that may no longer be there.
    ///
    /// Returns whether the file was actually read, so the caller knows
    /// whether this version has been dealt with.
    fn reload_from_disk(&mut self, path: &Path) -> bool {
        let (world, state) = match io::load_world_with_state(path) {
            Ok(loaded) => loaded,
            Err(e) => {
                // Saves are write-temp-then-rename, so a torn read
                // shouldn't be possible — but a project truncated by
                // something else must not turn the editor's copy into an
                // empty scene without a word.
                log::warn!("Reload of {} failed: {}", path.display(), e);
                self.ui.set_status(format!(
                    "{} changed on disk but couldn't be read — showing the last \
                     good version",
                    file_label(path)
                ));
                return false;
            }
        };
        self.world = world;
        self.reset_scene_session_state();
        self.editor.sockets = sockets_from_state(&state);
        // The graph joins the world and the sockets on the "restore"
        // side of this split rather than the camera's: an agent that
        // rewrote the recipe means for the human to see the new one.
        self.ui.graph = graph_from_state(&state);
        self.rebuild_all_meshes();
        // `rebuild_all_meshes` flags the document modified — true of an
        // edit, wrong here: this world came *out* of the user's file, so
        // it is exactly what the file holds.
        self.unsaved_changes = false;
        self.autosave_pending = false;
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
            // The poll's mark, brought up to date by hand: this editor
            // has now read that version, so the next poll has nothing
            // to report. A failed read leaves the mark alone on purpose
            // — see `tick_disk_reload`.
            self.note_project_mtime();
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
        let dialog = self
            .file_dialog(DialogStart::Export)
            .add_filter("Wavefront OBJ", &["obj"])
            .set_title(title);

        let Some(path) = dialog.save_file() else {
            return;
        };

        match io::export_obj_smoothed(&self.world, &path, blur) {
            Ok(stats) => {
                self.prefs.remember_export_dir(&path);
                let filename = file_label(&path);
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
                if stats.triangle_count > 0 {
                    self.set_export_report(
                        &path,
                        ExportReport {
                            format: "Wavefront OBJ (.obj)".into(),
                            mesh_source: smoothed_mesh_source(blur).into(),
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
                log::error!("Failed to export smoothed OBJ: {}", e);
                self.show_write_error("Export failed", &path, "export", &e, matches!(e, io::ObjError::Io(_)));
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
        let dialog = self
            .file_dialog(DialogStart::Export)
            .add_filter("glTF Binary", &["glb"])
            .set_title(title);

        let Some(path) = dialog.save_file() else {
            return;
        };

        let sockets = self.socket_export_nodes();
        match io::export_glb_smoothed(&self.world, &sockets, &path, blur) {
            Ok(stats) => {
                self.prefs.remember_export_dir(&path);
                let filename = file_label(&path);
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
                if stats.triangle_count > 0 {
                    self.set_export_report(
                        &path,
                        ExportReport {
                            format: "glTF Binary (.glb)".into(),
                            mesh_source: smoothed_mesh_source(blur).into(),
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
                log::error!("Failed to export smoothed GLB: {}", e);
                self.show_write_error("Export failed", &path, "export", &e, matches!(e, io::GlbError::Io(_)));
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
        let dialog = self
            .file_dialog(DialogStart::Export)
            .add_filter("glTF Binary", &["glb"])
            .set_title("Export as glTF Binary");

        let Some(path) = dialog.save_file() else {
            return;
        };

        let sockets = self.socket_export_nodes();
        match io::export_glb(&self.world, &sockets, &path) {
            Ok(stats) => {
                self.prefs.remember_export_dir(&path);
                let filename = file_label(&path);
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
                if stats.triangle_count > 0 {
                    self.set_export_report(
                        &path,
                        ExportReport {
                            format: "glTF Binary (.glb)".into(),
                            mesh_source: "Greedy mesh".into(),
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
                self.show_write_error("Export failed", &path, "export", &e, matches!(e, io::GlbError::Io(_)));
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
        let dialog = self
            .file_dialog(DialogStart::Export)
            .add_filter("Wavefront OBJ", &["obj"])
            .set_title("Export as Wavefront OBJ");

        let Some(path) = dialog.save_file() else {
            return;
        };

        match io::export_obj(&self.world, &path) {
            Ok(stats) => {
                self.prefs.remember_export_dir(&path);
                let filename = file_label(&path);
                let msg = if stats.triangle_count == 0 {
                    format!("Exported: {} (empty — no geometry)", filename)
                } else {
                    format!(
                        "Exported: {} ({} tris, {} chunks)",
                        filename, stats.triangle_count, stats.chunk_count
                    )
                };
                self.ui.set_status(msg);
                if stats.triangle_count > 0 {
                    self.set_export_report(
                        &path,
                        ExportReport {
                            format: "Wavefront OBJ (.obj)".into(),
                            mesh_source: "Greedy mesh".into(),
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
                self.show_write_error("Export failed", &path, "export", &e, matches!(e, io::ObjError::Io(_)));
                self.ui
                    .set_status(format!("Export failed: {}", file_label(&path)));
            }
        }
    }

    /// Prompt for a path and export to VOX.
    pub(super) fn export_vox(&mut self) {
        // Mirror the import convention on the way out (default on) so a
        // model exported to .vox opens upright in MagicaVoxel.
        let convert_axes = self.ui.convert_vox_axes;
        let dialog = self
            .file_dialog(DialogStart::Export)
            .add_filter("MagicaVoxel", &["vox"])
            .set_title("Export as MagicaVoxel");

        let Some(path) = dialog.save_file() else {
            return;
        };

        match std::fs::File::create(&path) {
            Ok(mut file) => match io::export_vox(&self.world, &mut file, convert_axes) {
                Ok(overflow) => {
                    self.prefs.remember_export_dir(&path);
                    let filename = file_label(&path);
                    let msg = if overflow > 0 {
                        format!(
                            "Exported: {} ({} colors quantized — VOX is 255-color)",
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
                        &path,
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
                    self.show_write_error("Export failed", &path, "export", &e, matches!(e, io::VoxError::Io(_)));
                    self.ui
                        .set_status(format!("Export failed: {}", file_label(&path)));
                }
            },
            Err(e) => {
                log::error!("Failed to create file {:?}: {}", path, e);
                self.show_write_error("Export failed", &path, "create", &e, true);
                self.ui
                    .set_status(format!("Export failed: {}", file_label(&path)));
            }
        }
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

/// Geometry-source label for the export report's smoothed (Marching
/// Cubes) variants; `blur` is the heavy-vs-light flag the menu passes.
fn smoothed_mesh_source(blur: bool) -> &'static str {
    if blur {
        "Marching Cubes (heavy)"
    } else {
        "Marching Cubes (light)"
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
        writable_hint_applies: bool,
    ) {
        // The recovery hint is about the filesystem, so only offer it
        // for filesystem failures. Exports can also fail for reasons
        // that have nothing to do with the disk — a scene too spread
        // out to smooth, a mesh past GLB's 4 GiB ceiling — and those
        // errors already say what to do; telling the user to check
        // their permissions on top of that just misleads.
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
    /// the bits every caller shares: the file name and the on-disk size
    /// (read back so it reflects what's actually on disk). Callers pass
    /// the format-specific fields and leave `filename` / `file_size` at
    /// their `Default` via `..Default::default()`.
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
        assert_eq!(classify_disk_poll(at(10), at(11), true), DiskPoll::WarnStale);
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
    fn nothing_happens_before_the_first_mark() {
        // No mark means no project file open, or one whose time we never
        // managed to read — either way there is nothing to compare.
        assert_eq!(classify_disk_poll(None, at(11), false), DiskPoll::Ignore);
        assert_eq!(classify_disk_poll(None, None, false), DiskPoll::Ignore);
    }
}
