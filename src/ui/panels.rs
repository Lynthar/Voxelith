//! UI state and panel definitions.

use std::path::PathBuf;

use crate::editor::{Axis, Quarter};
use crate::mcp::bridge::Approval;
use crate::prefs::PanelVisibility;

use super::CameraView;

/// How the exported mesh is built.
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

/// One export request: format × surface, with the pairings that don't
/// exist unrepresentable — `.vox` stores voxels, so there is no
/// smoothed variant to ask for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExportKind {
    Vox,
    Obj(Surface),
    Glb(Surface),
}

/// The format half of the Export… dialog's choice. A separate enum
/// rather than `ExportKind` itself because the dialog keeps a surface
/// selection alive even while `.vox` has it grayed out — switching
/// back to a mesh format finds the choice where the user left it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    Vox,
    Obj,
    Glb,
}

/// The Export… dialog's in-progress format × surface choice. Lives on
/// `UiState` beside the dialog's visibility flag (not inside an
/// `Option` with it) so closing the dialog doesn't reset it — the
/// common loop is re-exporting with the same settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExportChoice {
    pub format: ExportFormat,
    pub surface: Surface,
}

impl Default for ExportChoice {
    /// glTF Binary, blocky: the designated game-asset path.
    fn default() -> Self {
        Self {
            format: ExportFormat::Glb,
            surface: Surface::Blocky,
        }
    }
}

impl ExportChoice {
    /// The request this choice stands for. `.vox` ignores the surface
    /// half — which is why the dialog grays it out rather than hiding
    /// it: the selection is kept, it just doesn't apply.
    pub fn kind(self) -> ExportKind {
        match self.format {
            ExportFormat::Vox => ExportKind::Vox,
            ExportFormat::Obj => ExportKind::Obj(self.surface),
            ExportFormat::Glb => ExportKind::Glb(self.surface),
        }
    }
}

/// One-shot UI actions that need to be processed by the application.
///
/// Not `Copy` because `OpenRecent` carries a `PathBuf`. Actions are
/// taken by value via `UiState::take_actions`, so this is fine.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum UiAction {
    // File operations
    NewProject,
    OpenProject,
    /// Open a specific path from the recent-files MRU.
    OpenRecent(PathBuf),
    SaveProject,
    SaveAs,
    /// Re-read the open project from disk, discarding local edits.
    /// Raised by the disk-conflict strip, behind its confirm dialog.
    ReloadFromDisk,
    ImportVox,
    /// Voxelize a glTF mesh into the open document. Adds rather than
    /// replaces, and is undoable, so unlike `ImportVox` it needs no
    /// unsaved-changes guard.
    ImportGlb,
    /// Export the scene as `kind`. One action for every format ×
    /// surface pairing — raised by the Export… dialog (which replaced
    /// the menu's seven entries) with the `ExportKind` it built.
    Export(ExportKind),
    Exit,

    /// Unsaved-changes prompt: save the project, then run the action
    /// that was waiting on it. If the save doesn't happen (write error,
    /// or the user backs out of Save As) the pending action is dropped.
    UnsavedSave,
    /// Unsaved-changes prompt: run the pending action, losing the edits.
    UnsavedDiscard,
    /// Unsaved-changes prompt: forget the pending action.
    UnsavedCancel,

    // Edit operations
    Undo,
    Redo,
    /// Wipe the world, history and sockets. Not undoable, so the menu
    /// raises a confirm dialog and only dispatches this once the user
    /// accepts.
    ClearAll,

    // Selection / clipboard operations
    CopySelection,
    CutSelection,
    /// Paste at the selection's origin (or hovered cell when no
    /// selection exists). Ctrl+Shift+V's "always paste at cursor"
    /// is keyboard-only — UI buttons go through this default path.
    /// `at_cursor` pastes at the hovered cell (Ctrl+Shift+V) instead
    /// of the selection origin.
    PasteClipboard {
        at_cursor: bool,
    },
    DeleteSelection,
    /// Set the selection to the AABB of every non-air voxel in
    /// the world.
    SelectAllSolid,
    /// Clear the active selection (Esc / Ctrl+D / Edit menu →
    /// Deselect / Inspector button). Every entry point must route
    /// here, including ones holding `&mut Editor`: the selection also
    /// owns drag/move anchors and the translucent move ghost, which
    /// live on `App`. Writing `editor.selection = None` directly
    /// clears the marquee and strands the rest — see `App::deselect`.
    Deselect,
    /// Rotate the selection's voxel contents around `axis` by
    /// `quarter` (90° / -90° / 180°). Anchor is `selection.min`;
    /// the AABB may swap dimensions but its `min` corner stays put.
    /// One Ctrl+Z reverses the entire rotation.
    RotateSelection {
        axis: Axis,
        quarter: Quarter,
    },
    /// Mirror the selection's voxel contents across the midplane
    /// perpendicular to `axis`. AABB unchanged.
    MirrorSelection {
        axis: Axis,
    },

    // Generate operations
    GenerateTestCube,
    GenerateGround,
    GenerateSphere,
    GeneratePyramid,
    /// Run the pipeline graph and apply its output via CommandHistory.
    /// The generator presets in the Generate menu don't need an action
    /// of their own: they edit the graph (raising `GraphEdited`) and
    /// leave running it to this.
    RunGraph,
    /// The Graph panel changed the pipeline graph — a node, a wire, a
    /// parameter or a position.
    ///
    /// The graph is document data: it rides in the `.vxlt` and it is
    /// the recipe the model was built from. Voxel edits mark the
    /// document modified through the mesh rebuild, which a graph edit
    /// never reaches, so it has to say so itself — otherwise building a
    /// pipeline and quitting without running it loses the pipeline with
    /// no prompt, and the disk poll reloads a checkpoint straight over
    /// it ("nothing local to lose").
    GraphEdited,
    /// A socket was placed, renamed, deleted, or all of them cleared —
    /// same contract as [`UiAction::GraphEdited`], and for the same
    /// reason: sockets are document data (they ride in the `.vxlt` and
    /// export as glTF nodes) that no mesh rebuild ever notices, so
    /// renaming one and quitting used to lose it without a prompt.
    SocketsEdited,

    // Camera operations
    ResetCamera,
    SetCameraView(CameraView),
    /// Fit the camera to an AABB — center the target and pull back to
    /// the fit distance, keeping the current viewing angle. Three
    /// targets: the whole scene, the active selection, or the most
    /// recent generation.
    FrameAll,
    FrameSelected,
    FrameGenerated,

    // Crash recovery (in-app egui prompt; see `show_recovery_prompt`)
    /// Load the on-disk autosave into the editor.
    RecoverAutosave,
    /// Discard the on-disk autosave and keep the fresh default scene.
    DiscardAutosave,

    // --- Agent bridge ---
    /// Start serving MCP on the given loopback port. `0` asks the OS for
    /// a free one.
    AgentStart(u16),
    /// Stop serving and close the socket.
    AgentStop,
    /// Switch between applying an agent's batches directly and being
    /// asked first.
    AgentApproval(Approval),
    /// Commit the batch waiting for approval.
    AgentAccept,
    /// Decline it. Nothing is applied and the agent is told why.
    AgentReject,
}

/// Display-ready summary of a completed export, shown in an in-app
/// dialog after a successful OBJ / GLB / VOX write so the user can
/// sanity-check a large export (triangle budget, file size, lost color
/// info) without digging through the transient status bar. `App` builds
/// it from the format's `*Stats` plus the written file's on-disk size;
/// the UI only lays it out. Optional fields are skipped in the dialog
/// when `None` — VOX has no triangle / chunk concept, so those stay
/// empty and only its palette / quantization lines show.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ExportReport {
    /// Human format name, e.g. "glTF Binary (.glb)".
    pub format: String,
    /// Written file's name (not the full path).
    pub filename: String,
    /// How the geometry was produced: "Greedy mesh", "Marching Cubes
    /// (light)", "Marching Cubes (heavy)", or "—" for formats with no
    /// meshing step (VOX).
    pub mesh_source: String,
    /// Triangle / vertex / chunk counts when the format meshes; all
    /// `None` for VOX.
    pub triangles: Option<usize>,
    pub vertices: Option<usize>,
    pub chunks: Option<usize>,
    /// On-disk size in bytes, read back after writing.
    pub file_size: Option<u64>,
    /// How colors are carried, e.g. "Per-vertex RGBA" or "254-color
    /// palette".
    pub color_model: String,
    /// Non-fatal notes worth surfacing, e.g. a palette-quantization
    /// count. One label per line in the dialog.
    pub notes: Vec<String>,
}

/// Human-readable byte size for the export report: `820` → `"820 B"`,
/// `4_096` → `"4.0 KiB"`, `5_242_880` → `"5.0 MiB"`. Binary units
/// (1024-based) since these are file sizes on disk.
pub fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let b = bytes as f64;
    if b < KIB {
        format!("{} B", bytes)
    } else if b < MIB {
        format!("{:.1} KiB", b / KIB)
    } else if b < GIB {
        format!("{:.1} MiB", b / MIB)
    } else {
        format!("{:.2} GiB", b / GIB)
    }
}

/// Group a count with thousands separators for the detailed export
/// report (the perf HUD uses the coarser `hud::compact_count`):
/// `1_234_567` → `"1,234,567"`, `999` → `"999"`.
pub fn group_thousands(n: usize) -> String {
    let s = n.to_string();
    let len = s.len();
    let mut out = String::with_capacity(len + len / 3);
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// A "this can't be undone — are you sure?" dialog, carrying the
/// action to dispatch if the user says yes.
#[derive(Debug, Clone)]
pub struct ConfirmPrompt {
    pub title: String,
    pub body: String,
    pub action: UiAction,
}

/// UI state.
///
/// Workspace panel toggles live in `panels`, the same struct prefs
/// persists — `App::new` assigns it wholesale from the loaded prefs
/// and `App::save_prefs` assigns it back, so neither end can drift
/// out of sync with the other. (There used to be a hand-written
/// `new()` carrying a second, contradicting set of defaults — it had
/// no callers.)
#[derive(Default)]
pub struct UiState {
    /// Open/closed state of the workspace panels. Persisted verbatim.
    pub panels: PanelVisibility,

    // Transient windows — not part of the saved layout.
    pub show_help: bool,
    pub show_about: bool,

    /// Export… dialog (File ▸ Export…). The choice sits in its own
    /// field so it survives the dialog closing; see `ExportChoice`.
    pub show_export_dialog: bool,
    pub export_choice: ExportChoice,

    /// Crash-recovery prompt: an in-app egui dialog (NOT a native rfd
    /// modal — `rfd::MessageDialog` exits the process on this winit+wgpu
    /// setup). Set true at startup when an autosave is on disk; cleared
    /// when the user picks Recover or Discard.
    pub show_recovery_prompt: bool,

    /// Active file-operation error, shown as an in-app egui dialog
    /// (`(title, detail)`). Same reason as `show_recovery_prompt`: a
    /// native modal would crash the process on the very failure it's
    /// trying to report. `Some` while shown; cleared by the OK button.
    pub error_dialog: Option<(String, String)>,

    /// Report from the last successful export, shown as an in-app egui
    /// dialog (same click-to-dismiss contract as `error_dialog`). `Some`
    /// while shown; cleared by the dialog's Close button.
    pub export_report: Option<ExportReport>,

    /// Pending confirmation for a destructive, non-undoable action.
    /// Accepting dispatches the carried `UiAction`; cancelling drops it.
    /// In-app egui, never a native modal — see `show_recovery_prompt`.
    pub confirm: Option<ConfirmPrompt>,

    /// Unsaved-changes prompt. `Some(what)` while shown, where `what`
    /// describes the operation being held up ("open another project").
    /// Raised by `App::guard_then`; the Save / Don't Save / Cancel
    /// buttons dispatch the `Unsaved*` actions.
    pub unsaved_prompt: Option<String>,

    /// The open project changed on disk while there were local edits to
    /// lose, so the auto-reload was refused. `Some(file label)` for as
    /// long as that stands — written by `App::tick_disk_reload`, cleared
    /// when the two sides agree again (save / reload / open) or when the
    /// user dismisses the strip.
    ///
    /// A field rather than a status-bar line because the *state* lasts:
    /// every later write is refused too, and a message that scrolls away
    /// leaves the refusals looking like a broken feature.
    pub disk_conflict: Option<String>,

    // One-shot action queue
    pending_actions: Vec<UiAction>,

    // Status message for user feedback
    pub status_message: Option<(String, std::time::Instant)>,

    /// Port the Agent panel will ask for when the bridge is started.
    /// A string rather than a `u16` because it is a text field, and an
    /// unparseable one has to disable the button rather than silently
    /// stand for some other port.
    pub agent_port_input: String,
}

impl UiState {
    /// Queue an action to be processed
    pub fn request(&mut self, action: UiAction) {
        if !self.pending_actions.contains(&action) {
            self.pending_actions.push(action);
        }
    }

    /// Take all pending actions (clears the queue)
    pub fn take_actions(&mut self) -> Vec<UiAction> {
        std::mem::take(&mut self.pending_actions)
    }

    /// Clear all pending actions
    pub fn clear_actions(&mut self) {
        self.pending_actions.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::{format_bytes, group_thousands, ExportChoice, ExportFormat, ExportKind, Surface};

    /// The Export… dialog replaced seven menu entries; this pins that
    /// its format × surface grid still reaches every one of the seven
    /// `ExportKind`s and nothing else — `.vox` collapses its (grayed
    /// out) surface column into the single surfaceless kind.
    #[test]
    fn export_dialog_choices_cover_exactly_the_seven_kinds() {
        let formats = [ExportFormat::Vox, ExportFormat::Obj, ExportFormat::Glb];
        let surfaces = [Surface::Blocky, Surface::SmoothLight, Surface::SmoothHeavy];
        let mut kinds: Vec<ExportKind> = formats
            .iter()
            .flat_map(|&format| {
                surfaces
                    .iter()
                    .map(move |&surface| ExportChoice { format, surface }.kind())
            })
            .collect();
        kinds.dedup(); // the three Vox rows collapse to one
        kinds.sort_by_key(|k| format!("{:?}", k));
        kinds.dedup();
        assert_eq!(
            kinds.len(),
            7,
            "format × surface must span all seven exports"
        );
    }

    /// The dialog opens on the designated game-asset path.
    #[test]
    fn export_choice_defaults_to_blocky_glb() {
        assert_eq!(
            ExportChoice::default().kind(),
            ExportKind::Glb(Surface::Blocky)
        );
    }

    #[test]
    fn format_bytes_scales_binary_units() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(820), "820 B");
        assert_eq!(format_bytes(1024), "1.0 KiB");
        assert_eq!(format_bytes(1536), "1.5 KiB");
        assert_eq!(format_bytes(5 * 1024 * 1024), "5.0 MiB");
        assert_eq!(format_bytes(3 * 1024 * 1024 * 1024), "3.00 GiB");
    }

    #[test]
    fn group_thousands_inserts_separators() {
        assert_eq!(group_thousands(0), "0");
        assert_eq!(group_thousands(7), "7");
        assert_eq!(group_thousands(999), "999");
        assert_eq!(group_thousands(1_000), "1,000");
        assert_eq!(group_thousands(12_345), "12,345");
        assert_eq!(group_thousands(1_234_567), "1,234,567");
    }
}
