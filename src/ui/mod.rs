//! User interface components using egui.

pub mod hud;
mod panels;

pub use hud::HudState;
pub use panels::{ConfirmPrompt, ExportReport, UiAction, UiState};

use crate::editor::{Axis, Editor, Quarter, Tool};
use crate::mcp::bridge::{Approval, DEFAULT_PORT};
use crate::procgen::{
    CombineOp, FilterPredicate, LSystemTree, MaskMode, NodeId, NodeKind,
    PerlinTerrain, PipelineGraph, WfcGenerator, WfcTileset,
};
use egui::Context;

/// Hover text on every disabled wireframe control. The GPU feature is
/// `POLYGON_MODE_LINE`; GL backends and some integrated GPUs lack it.
const WIREFRAME_UNSUPPORTED: &str =
    "This GPU doesn't support line polygon mode, so wireframe is unavailable";

/// Cap on user-added palette entries. Keeps the palette grid a fixed,
/// scannable size (and `EditorPrefs::palette` small).
const MAX_PALETTE_COLORS: usize = 32;

/// Viewport display settings
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ViewportSettings {
    pub show_grid: bool,
    pub show_axes: bool,
    pub wireframe_mode: bool,
    pub grid_size: i32,
    pub grid_spacing: f32,
    /// Viewport HUD (bottom-left tool / gesture readout).
    pub show_hud: bool,
    /// Performance HUD (bottom-right FPS / tris / rebuild readout).
    /// Default off — stats overlays are opt-in everywhere (Blender /
    /// Unreal / Maya all ship them disabled).
    pub show_perf_hud: bool,
}

impl Default for ViewportSettings {
    fn default() -> Self {
        Self {
            show_grid: true,
            show_axes: true,
            wireframe_mode: false,
            grid_size: 20,
            grid_spacing: 1.0,
            show_hud: true,
            show_perf_hud: false,
        }
    }
}

/// Which generator the procgen panel is currently editing.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum GeneratorChoice {
    Terrain,
    Tree,
    Wfc,
}

impl GeneratorChoice {
    /// Display label used by the panel's combo box and status messages.
    pub fn label(self) -> &'static str {
        match self {
            Self::Terrain => "Perlin Terrain",
            Self::Tree => "L-System Tree",
            Self::Wfc => "WFC Tile Layout",
        }
    }
}

impl Default for GeneratorChoice {
    fn default() -> Self {
        Self::Terrain
    }
}

/// Live state for the procedural-generation panel.
///
/// Each generator's instance doubles as its parameter state — UI
/// sliders mutate the fields in place, then `UiAction::GenerateProcedural`
/// triggers `selected`'s `generate()` in the application layer.
///
/// `preview_enabled` and `graph_preview_enabled` independently drive
/// translucent overlays — the first for the selected single generator,
/// the second for the pipeline graph's output. Both share the renderer's
/// preview slot; when both are on, the graph wins on the slot since
/// its tick runs second.
/// Struct-level `#[serde(default)]`, not just field-level: this rides in
/// `prefs.ron`, and without it the *next* field added here without its
/// own default makes every existing prefs file fail to parse — which
/// `Prefs::load` handles by logging a warning and silently discarding
/// the user's whole workspace (window, palette, MRU).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ProcgenSettings {
    pub selected: GeneratorChoice,
    pub terrain: PerlinTerrain,
    pub tree: LSystemTree,
    pub wfc: WfcGenerator,
    pub preview_enabled: bool,
    #[serde(default)]
    pub graph_preview_enabled: bool,
}

/// Main UI manager
pub struct Ui {
    pub state: UiState,
    pub viewport: ViewportSettings,
    pub procgen: ProcgenSettings,
    /// Pipeline graph edited in the Graph panel.
    ///
    /// Document data, not workspace state: it rides in the `.vxlt`
    /// (`EditorState.graph`) because it is the recipe the model was
    /// built from, and it used to live in `prefs.ron`, where it
    /// followed the machine instead of the work. It sits on `Ui`
    /// because the panel edits it directly — which is why that panel
    /// tells `App` when it changed (`UiAction::GraphEdited`).
    pub graph: PipelineGraph,
    /// Currently-selected node in the visual graph editor. Drives
    /// the sidebar parameter editor. Cleared automatically when the
    /// node is removed.
    pub selected_node: Option<NodeId>,
    /// Active wire-creation drag: source node whose output socket was
    /// pressed. While set, the editor renders a live wire from that
    /// socket to the cursor; on release a hit-test against input
    /// sockets either snaps the wire to a target or discards it.
    pub dragging_wire: Option<NodeId>,
    /// Recent-files MRU mirrored from `prefs::Prefs::recent_files`.
    /// App syncs this whenever the prefs version changes (touch_recent
    /// + initial load).
    pub recent_files: Vec<std::path::PathBuf>,
    /// Mirror of `App::clipboard.is_some()` so the Tools panel can
    /// gray out the Paste button without `App::clipboard` leaking
    /// across the UI layer boundary. App syncs it before each frame.
    pub has_clipboard: bool,
    /// Mirror of `Renderer::wireframe_supported` (the GPU exposes
    /// `POLYGON_MODE_LINE`). Every wireframe control must gate on this:
    /// the render path already falls back to solid when the pipeline is
    /// missing, so without the gate the checkboxes stay tickable, the
    /// status bar announces `[Wireframe]`, and nothing whatsoever
    /// changes on screen — the UI lying about what the app is doing.
    /// App syncs it before each frame.
    pub wireframe_supported: bool,

    /// Voxelization resolution along the longest axis (32 / 64 / 128)
    /// for `File ▸ Import` of a GLB. Owned by the UI so the control's
    /// state lives next to its widget.
    pub import_resolution: u32,

    /// Whether `.vox` import/export converts between MagicaVoxel's Z-up
    /// convention and Voxelith's Y-up one (default on). Owned by the UI
    /// (a File ▸ Import checkbox binds to it); `App::import_vox` /
    /// `App::export_vox` read it at transfer time. A model authored in
    /// MagicaVoxel stands upright when this is on; turn it off for a
    /// `.vox` already authored Y-up.
    pub convert_vox_axes: bool,

    /// Mirror of the in-editor MCP bridge's state, synced each frame the
    /// same way `has_clipboard` is.
    pub agent: AgentView,
}

/// What the Agent panel and its approval strip draw from.
///
/// The bridge itself lives on `App` — a listening socket, a channel and
/// possibly a batch parked mid-call — and none of that belongs on the UI
/// side of the line. This is the display-ready summary App mirrors
/// across each frame, the same shape `has_clipboard` takes.
#[derive(Debug, Clone, Default)]
pub struct AgentView {
    /// The URL to hand a client, while the bridge is listening.
    pub url: Option<String>,
    pub approval: Approval,
    /// Batches committed since it came up — the panel's evidence that
    /// something is happening on the other end.
    pub applied: usize,
    /// What a batch waiting for approval would do, phrased for the
    /// person deciding ("writes 240 voxels, clears 18").
    pub pending: Option<String>,
}

impl Ui {
    pub fn new() -> Self {
        Self {
            state: UiState::default(),
            viewport: ViewportSettings::default(),
            procgen: ProcgenSettings::default(),
            graph: PipelineGraph::default(),
            selected_node: None,
            dragging_wire: None,
            recent_files: Vec::new(),
            has_clipboard: false,
            wireframe_supported: false,
            import_resolution: 64,
            convert_vox_axes: true,
            agent: AgentView::default(),
        }
    }

    /// Render the UI. `hud` is the App-built per-frame snapshot for
    /// the viewport HUD overlay (gesture phase, locked plane, …).
    pub fn show(
        &mut self,
        ctx: &Context,
        stats: &RenderStats,
        editor: &mut Editor,
        hud: &HudState,
    ) {
        // Top menu bar
        self.show_menu_bar(ctx, editor);

        // The open project changed on disk and the reload was refused —
        // shown directly under the menu bar until it's resolved.
        if self.state.disk_conflict.is_some() {
            self.show_disk_conflict_bar(ctx);
        }

        // An agent's batch is waiting to be approved. Same placement and
        // the same reasoning as the strip above: a state that lasts, and
        // that the user has to be able to work around rather than be
        // trapped by.
        if self.agent.pending.is_some() {
            self.show_agent_review_bar(ctx);
        }

        // Left side panel with tools
        self.show_toolbar(ctx, editor);

        // Stats panel
        if self.state.panels.show_stats {
            self.show_stats_panel(ctx, stats, editor);
        }

        // Tools panel
        if self.state.panels.show_tools {
            self.show_tools_panel(ctx, editor);
        }

        // Color palette panel
        if self.state.panels.show_palette {
            self.show_palette_panel(ctx, editor);
        }

        // Viewport settings panel
        if self.state.panels.show_viewport_settings {
            self.show_viewport_panel(ctx);
        }

        // Procedural generation panel
        if self.state.panels.show_procgen {
            self.show_procgen_panel(ctx);
        }

        // Pipeline graph panel
        if self.state.panels.show_graph {
            self.show_graph_panel(ctx);
        }

        // Agent bridge panel
        if self.state.panels.show_agent {
            self.show_agent_panel(ctx);
        }

        // Help panel
        if self.state.show_help {
            self.show_help_panel(ctx);
        }

        // About dialog
        if self.state.show_about {
            self.show_about_dialog(ctx);
        }

        // Status bar
        self.show_status_bar(ctx, editor);

        // Viewport HUD — after the status bar so every panel has
        // claimed its screen edge and `ctx.available_rect()` is the
        // true viewport rect the HUD anchors inside.
        if self.viewport.show_hud {
            hud::show_hud_overlay(ctx, hud);
        }

        // Performance HUD — bottom-right counterpart, same rules.
        if self.viewport.show_perf_hud {
            hud::show_perf_overlay(ctx, stats);
        }

        // Crash-recovery prompt, rendered last so it sits on top. This
        // is an in-app egui dialog, NOT a native `rfd::MessageDialog` —
        // the latter exits the process on this winit + wgpu setup
        // (regardless of whether it's shown during `resumed` or a later
        // frame), so the recovery flow can't use it.
        if self.state.show_recovery_prompt {
            self.show_recovery_prompt(ctx);
        }

        // File-operation error dialog (also in-app egui, not native rfd
        // — same crash reason; see `show_recovery_prompt`).
        if self.state.error_dialog.is_some() {
            self.show_error_dialog(ctx);
        }

        // Export report (in-app egui, same dismiss contract) — shown
        // after a successful export so the user can sanity-check the
        // triangle budget / file size without chasing the status bar.
        if self.state.export_report.is_some() {
            self.show_export_report(ctx);
        }

        // Confirmation for a destructive action, and the unsaved-changes
        // guard. Last, and in this order, so the guard sits on top of
        // everything else — it's the one blocking the user's request.
        if self.state.confirm.is_some() {
            self.show_confirm_dialog(ctx);
        }
        if self.state.unsaved_prompt.is_some() {
            self.show_unsaved_prompt(ctx);
        }
    }

    /// Somebody else wrote the open project — an agent running with
    /// `--checkpoint`, a `voxelith exec --out`, a `git checkout` — while
    /// there were unsaved edits here. The editor keeps the user's copy
    /// and says so here until they decide.
    ///
    /// A strip rather than a modal on purpose. The writer is typically an
    /// agent working through a batch at a time, so a dialog would reopen
    /// on every step and make the editor unusable; a strip states the
    /// situation, offers both ways out, and lets the user keep working
    /// in the meantime.
    fn show_disk_conflict_bar(&mut self, ctx: &Context) {
        let Some(file) = self.state.disk_conflict.clone() else {
            return;
        };
        egui::TopBottomPanel::top("disk_conflict").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    egui::RichText::new(format!("\u{26A0} {} changed on disk.", file))
                        .strong()
                        .color(ui.visuals().warn_fg_color),
                );
                ui.label("Your unsaved edits are kept.");
                if ui
                    .button("Reload")
                    .on_hover_text("Take the version on disk and discard the edits here")
                    .clicked()
                {
                    // Confirmed first: this throws away work the user
                    // hasn't saved, and the strip's whole point is that
                    // the two versions both exist.
                    self.state.confirm = Some(ConfirmPrompt {
                        title: "Reload from disk".to_string(),
                        body: format!(
                            "Load \"{}\" as it is on disk?\n\nThe unsaved changes in \
                             this editor will be lost.",
                            file
                        ),
                        action: UiAction::ReloadFromDisk,
                    });
                }
                if ui
                    .button("Keep mine")
                    .on_hover_text("Dismiss this. Save when you're ready to overwrite the file")
                    .clicked()
                {
                    self.state.disk_conflict = None;
                }
            });
        });
    }

    /// An agent's batch is on screen as translucent geometry, waiting
    /// for a yes or no.
    ///
    /// A strip and not a modal, and here the reason is sharper than it
    /// was for the disk-conflict one: the question is about geometry the
    /// user has to be able to orbit around and look at before answering.
    /// A modal would cover the very thing it is asking about.
    fn show_agent_review_bar(&mut self, ctx: &Context) {
        let Some(summary) = self.agent.pending.clone() else {
            return;
        };
        egui::TopBottomPanel::top("agent_review").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new(format!("Agent batch: {summary}.")).strong());
                ui.label("Shown in the viewport, not applied yet.");
                if ui
                    .button("Apply")
                    .on_hover_text("Commit it. One Ctrl+Z takes the whole batch back out")
                    .clicked()
                {
                    self.state.request(UiAction::AgentAccept);
                }
                if ui
                    .button("Discard")
                    .on_hover_text("Leave the project as it is, and tell the agent why")
                    .clicked()
                {
                    self.state.request(UiAction::AgentReject);
                }
            });
        });
    }

    /// The in-editor MCP server: switch it on, hand an agent the URL,
    /// and decide whether its batches land as they arrive or wait for a
    /// yes.
    fn show_agent_panel(&mut self, ctx: &Context) {
        // Deferred-action pattern (same as `show_graph_panel`): `.open(...)`
        // borrows `self.state.panels.show_agent` and the closure borrows
        // other `self.*` fields, so intents are collected into a local
        // and dispatched after the closure releases the borrow.
        let mut action: Option<UiAction> = None;

        // An empty field would read as "no port" and leave Start dead
        // with nothing saying why. Filling it here rather than in
        // `UiState::default` keeps the default in one place — this panel
        // is the only thing that has an opinion about it.
        if self.state.agent_port_input.is_empty() {
            self.state.agent_port_input = DEFAULT_PORT.to_string();
        }
        let agent = &self.agent;
        let port = self.state.agent_port_input.parse::<u16>().ok();
        let port_input = &mut self.state.agent_port_input;

        egui::Window::new("Agent Bridge")
            .default_pos([ctx.screen_rect().width() / 2.0 - 180.0, 140.0])
            .default_width(360.0)
            .open(&mut self.state.panels.show_agent)
            .show(ctx, |ui| {
                match &agent.url {
                    Some(url) => {
                        ui.horizontal(|ui| {
                            ui.colored_label(
                                egui::Color32::from_rgb(80, 200, 120),
                                "listening",
                            );
                            if ui
                                .button("Stop")
                                .on_hover_text("Close the socket. Anything waiting is told why")
                                .clicked()
                            {
                                action = Some(UiAction::AgentStop);
                            }
                        });
                        ui.label("Point an MCP client at:");
                        ui.monospace(url);
                        ui.label(
                            egui::RichText::new(format!(
                                "claude mcp add --transport http voxelith {url}"
                            ))
                            .small()
                            .weak(),
                        );
                        ui.label(format!(
                            "{} batch{} applied since it started",
                            agent.applied,
                            if agent.applied == 1 { "" } else { "es" }
                        ));
                    }
                    None => {
                        ui.label("Let an agent edit this project directly, instead of \
                                  passing a file back and forth.");
                        ui.horizontal(|ui| {
                            ui.label("Port");
                            ui.add(
                                egui::TextEdit::singleline(port_input)
                                    .desired_width(64.0)
                                    .hint_text("8737"),
                            );
                            if ui
                                .add_enabled(port.is_some(), egui::Button::new("Start"))
                                .on_hover_text(
                                    "Serve MCP on 127.0.0.1 — this machine only, never the \
                                     network",
                                )
                                .clicked()
                            {
                                if let Some(port) = port {
                                    action = Some(UiAction::AgentStart(port));
                                }
                            }
                        });
                        if port.is_none() {
                            ui.colored_label(
                                egui::Color32::from_rgb(220, 180, 80),
                                "Not a port number (0–65535). 0 asks the OS for a free one.",
                            );
                        }
                    }
                }

                ui.separator();
                ui.label(egui::RichText::new("When an agent sends a batch").strong());
                let mut approval = agent.approval;
                ui.radio_value(&mut approval, Approval::Auto, Approval::Auto.label())
                    .on_hover_text(
                        "It lands as it arrives and you watch it happen. Ctrl+Z takes any \
                         step back out — an agent's edits go on your undo stack, not a \
                         separate one.",
                    );
                ui.radio_value(&mut approval, Approval::Review, Approval::Review.label())
                    .on_hover_text(
                        "It goes up as translucent geometry and the agent waits until you \
                         apply or discard it.",
                    );
                if approval != agent.approval {
                    action = Some(UiAction::AgentApproval(approval));
                }

                if let Some(pending) = &agent.pending {
                    ui.separator();
                    ui.label(format!("Waiting on you: {pending}"));
                }
            });

        if let Some(action) = action {
            self.state.request(action);
        }
    }

    /// Confirmation for an action that can't be undone. Accepting
    /// dispatches `ConfirmAccepted`; App knows which action that is.
    fn show_confirm_dialog(&mut self, ctx: &Context) {
        let Some(prompt) = self.state.confirm.clone() else {
            return;
        };
        egui::Window::new(prompt.title)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.label(prompt.body);
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        self.state.confirm = None;
                    }
                    if ui.button("Continue").clicked() {
                        self.state.request(prompt.action);
                        self.state.confirm = None;
                    }
                });
            });
    }

    /// The unsaved-changes guard. Raised by `App::guard_then` before
    /// anything that would throw the current scene away.
    fn show_unsaved_prompt(&mut self, ctx: &Context) {
        let Some(what) = self.state.unsaved_prompt.clone() else {
            return;
        };
        egui::Window::new("Unsaved changes")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.label(format!(
                    "This project has changes that haven't been saved.\n\
                     Save them before you {}?",
                    what
                ));
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        self.state.request(UiAction::UnsavedSave);
                        self.state.unsaved_prompt = None;
                    }
                    if ui.button("Don't Save").clicked() {
                        self.state.request(UiAction::UnsavedDiscard);
                        self.state.unsaved_prompt = None;
                    }
                    if ui.button("Cancel").clicked() {
                        self.state.request(UiAction::UnsavedCancel);
                        self.state.unsaved_prompt = None;
                    }
                });
            });
    }

    /// In-app error dialog for failed file operations: centered window
    /// with the actionable detail and an OK button that dismisses it.
    fn show_error_dialog(&mut self, ctx: &Context) {
        let Some((title, detail)) = self.state.error_dialog.clone() else {
            return;
        };
        let mut dismiss = false;
        egui::Window::new(&title)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.label(&detail);
                ui.add_space(8.0);
                if ui.button("OK").clicked() {
                    dismiss = true;
                }
            });
        if dismiss {
            self.state.error_dialog = None;
        }
    }

    /// In-app export report: a centered, dismissable summary shown after
    /// a successful export. Mirrors `show_error_dialog`'s structure (one
    /// Close button clears the state). Rows for counts the format doesn't
    /// carry are skipped — VOX has no triangle / vertex / chunk numbers,
    /// so only its file size, colors, and quantization note show.
    fn show_export_report(&mut self, ctx: &Context) {
        let Some(report) = self.state.export_report.clone() else {
            return;
        };
        let mut dismiss = false;
        egui::Window::new("Export complete")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                egui::Grid::new("export_report_grid")
                    .num_columns(2)
                    .spacing([16.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("File");
                        ui.label(&report.filename);
                        ui.end_row();

                        ui.label("Format");
                        ui.label(&report.format);
                        ui.end_row();

                        if !report.mesh_source.is_empty() {
                            ui.label("Geometry");
                            ui.label(&report.mesh_source);
                            ui.end_row();
                        }
                        if let Some(t) = report.triangles {
                            ui.label("Triangles");
                            ui.label(panels::group_thousands(t));
                            ui.end_row();
                        }
                        if let Some(v) = report.vertices {
                            ui.label("Vertices");
                            ui.label(panels::group_thousands(v));
                            ui.end_row();
                        }
                        if let Some(c) = report.chunks {
                            ui.label("Chunks");
                            ui.label(panels::group_thousands(c));
                            ui.end_row();
                        }
                        if let Some(sz) = report.file_size {
                            ui.label("File size");
                            ui.label(panels::format_bytes(sz));
                            ui.end_row();
                        }
                        if !report.color_model.is_empty() {
                            ui.label("Colors");
                            ui.label(&report.color_model);
                            ui.end_row();
                        }
                    });

                // Lost-info / quantization notes in amber, like the
                // status bar's warning coding.
                if !report.notes.is_empty() {
                    ui.add_space(6.0);
                    for note in &report.notes {
                        ui.label(
                            egui::RichText::new(note)
                                .color(egui::Color32::from_rgb(255, 200, 80)),
                        );
                    }
                }

                ui.add_space(8.0);
                if ui.button("Close").clicked() {
                    dismiss = true;
                }
            });
        if dismiss {
            self.state.export_report = None;
        }
    }

    /// In-app crash-recovery prompt: a centered, non-closable window with
    /// Recover / Discard. Both dispatch a `UiAction` and clear the flag.
    fn show_recovery_prompt(&mut self, ctx: &Context) {
        egui::Window::new("Recover unsaved work?")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.label(
                    "Voxelith may have closed unexpectedly last time.\n\
                     Recover your last auto-saved work, or discard it and \
                     start fresh?",
                );
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Recover").clicked() {
                        self.state.request(UiAction::RecoverAutosave);
                        self.state.show_recovery_prompt = false;
                    }
                    if ui.button("Discard").clicked() {
                        self.state.request(UiAction::DiscardAutosave);
                        self.state.show_recovery_prompt = false;
                    }
                });
            });
    }

    fn show_menu_bar(&mut self, ctx: &Context, editor: &Editor) {
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("New").clicked() {
                        self.state.request(UiAction::NewProject);
                        ui.close_menu();
                    }
                    if ui.button("Open...").clicked() {
                        self.state.request(UiAction::OpenProject);
                        ui.close_menu();
                    }
                    ui.menu_button("Open Recent", |ui| {
                        if self.recent_files.is_empty() {
                            ui.add_enabled(false, egui::Button::new("(empty)"));
                        } else {
                            for path in &self.recent_files {
                                let label = path
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .map(|s| s.to_string())
                                    .unwrap_or_else(|| path.display().to_string());
                                let resp = ui
                                    .button(label)
                                    .on_hover_text(path.display().to_string());
                                if resp.clicked() {
                                    self.state.request(UiAction::OpenRecent(path.clone()));
                                    ui.close_menu();
                                }
                            }
                        }
                    });
                    if ui.button("Save").clicked() {
                        self.state.request(UiAction::SaveProject);
                        ui.close_menu();
                    }
                    if ui.button("Save As...").clicked() {
                        self.state.request(UiAction::SaveAs);
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.menu_button("Import", |ui| {
                        if ui.button("MagicaVoxel (.vox)...").clicked() {
                            self.state.request(UiAction::ImportVox);
                            ui.close_menu();
                        }
                        if ui
                            .button("glTF mesh (.glb)...")
                            .on_hover_text(
                                "Voxelize a triangle mesh into the open scene. \
                                 Adds to what's there and is undoable, unlike \
                                 .vox import which replaces the document.",
                            )
                            .clicked()
                        {
                            self.state.request(UiAction::ImportGlb);
                            ui.close_menu();
                        }
                        ui.horizontal(|ui| {
                            ui.label("Mesh resolution");
                            // 32 / 64 / 128 rather than a free slider:
                            // each step is roughly 8× the voxel count and
                            // the values between aren't useful in practice.
                            egui::ComboBox::from_id_salt("import_resolution")
                                .selected_text(format!("{}³", self.import_resolution))
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut self.import_resolution,
                                        32,
                                        "32³ (icon)",
                                    );
                                    ui.selectable_value(
                                        &mut self.import_resolution,
                                        64,
                                        "64³ (default)",
                                    );
                                    ui.selectable_value(
                                        &mut self.import_resolution,
                                        128,
                                        "128³ (detail)",
                                    );
                                });
                        });
                        ui.separator();
                        ui.checkbox(&mut self.convert_vox_axes, "Convert Z-up ↔ Y-up")
                            .on_hover_text(
                                "MagicaVoxel is Z-up, Voxelith is Y-up. When on, \
                                 .vox import and export rotate between the two so \
                                 models stay upright. Turn off for a .vox already \
                                 authored Y-up.",
                            );
                    });
                    ui.menu_button("Export", |ui| {
                        if ui.button("MagicaVoxel (.vox)...").clicked() {
                            self.state.request(UiAction::ExportVox);
                            ui.close_menu();
                        }
                        if ui.button("Wavefront OBJ (.obj)...").clicked() {
                            self.state.request(UiAction::ExportObj);
                            ui.close_menu();
                        }
                        if ui
                            .button("Wavefront OBJ — smoothed, light (.obj)...")
                            .on_hover_text(
                                "Marching Cubes over raw voxel density: \
                                 voxel surfaces with rounded edges. \
                                 Preserves thin features (tree branches, \
                                 sparse detail).",
                            )
                            .clicked()
                        {
                            self.state.request(UiAction::ExportObjSmoothedLight);
                            ui.close_menu();
                        }
                        if ui
                            .button("Wavefront OBJ — smoothed, heavy (.obj)...")
                            .on_hover_text(
                                "Marching Cubes after a 3×3×3 density \
                                 blur: clay-like blobs. Best for terrain \
                                 / large solid masses; thin features may \
                                 dissolve.",
                            )
                            .clicked()
                        {
                            self.state.request(UiAction::ExportObjSmoothedHeavy);
                            ui.close_menu();
                        }
                        if ui.button("glTF Binary (.glb)...").clicked() {
                            self.state.request(UiAction::ExportGlb);
                            ui.close_menu();
                        }
                        if ui
                            .button("glTF Binary — smoothed, light (.glb)...")
                            .on_hover_text(
                                "Marching Cubes over raw voxel density: \
                                 voxel surfaces with rounded edges. \
                                 Preserves thin features.",
                            )
                            .clicked()
                        {
                            self.state.request(UiAction::ExportGlbSmoothedLight);
                            ui.close_menu();
                        }
                        if ui
                            .button("glTF Binary — smoothed, heavy (.glb)...")
                            .on_hover_text(
                                "Marching Cubes after a 3×3×3 density \
                                 blur: clay-like blobs. Best for terrain.",
                            )
                            .clicked()
                        {
                            self.state.request(UiAction::ExportGlbSmoothedHeavy);
                            ui.close_menu();
                        }
                    });
                    ui.separator();
                    if ui.button("Exit").clicked() {
                        self.state.request(UiAction::Exit);
                    }
                });

                ui.menu_button("Edit", |ui| {
                    let undo_text = if editor.can_undo() { "Undo  Ctrl+Z" } else { "Undo" };
                    if ui.add_enabled(editor.can_undo(), egui::Button::new(undo_text)).clicked() {
                        self.state.request(UiAction::Undo);
                        ui.close_menu();
                    }
                    let redo_text = if editor.can_redo() { "Redo  Ctrl+Y" } else { "Redo" };
                    if ui.add_enabled(editor.can_redo(), egui::Button::new(redo_text)).clicked() {
                        self.state.request(UiAction::Redo);
                        ui.close_menu();
                    }
                    ui.separator();
                    let has_sel = editor.selection.is_some();
                    let can_paste = self.has_clipboard;
                    if ui
                        .add_enabled(has_sel, egui::Button::new("Cut  Ctrl+X"))
                        .clicked()
                    {
                        self.state.request(UiAction::CutSelection);
                        ui.close_menu();
                    }
                    if ui
                        .add_enabled(has_sel, egui::Button::new("Copy  Ctrl+C"))
                        .clicked()
                    {
                        self.state.request(UiAction::CopySelection);
                        ui.close_menu();
                    }
                    if ui
                        .add_enabled(can_paste, egui::Button::new("Paste  Ctrl+V"))
                        .clicked()
                    {
                        self.state.request(UiAction::PasteClipboard);
                        ui.close_menu();
                    }
                    if ui
                        .add_enabled(has_sel, egui::Button::new("Delete  Del"))
                        .clicked()
                    {
                        self.state.request(UiAction::DeleteSelection);
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Select All  Ctrl+A").clicked() {
                        self.state.request(UiAction::SelectAllSolid);
                        ui.close_menu();
                    }
                    if ui
                        .add_enabled(has_sel, egui::Button::new("Deselect  Esc"))
                        .clicked()
                    {
                        self.state.request(UiAction::Deselect);
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Clear All").clicked() {
                        // Everything else in this menu is undoable, and
                        // this sits one row from Deselect — a misclick
                        // used to silently take the whole scene with no
                        // way back (it clears the undo history too).
                        self.state.confirm = Some(ConfirmPrompt {
                            title: "Clear everything?".into(),
                            body: "This deletes every voxel and socket, \
                                   and can't be undone."
                                .into(),
                            action: UiAction::ClearAll,
                        });
                        ui.close_menu();
                    }
                });

                ui.menu_button("Selection", |ui| {
                    let has_sel = editor.selection.is_some();
                    // Each Rotate submenu hosts CW / CCW / 180°.
                    // Anchor is selection.min — the rotated AABB
                    // extends from the same min, so a 4×1×2 region
                    // becomes 2×1×4 spreading toward +Z.
                    ui.menu_button("Rotate around X", |ui| {
                        if ui
                            .add_enabled(has_sel, egui::Button::new("90°"))
                            .clicked()
                        {
                            self.state.request(UiAction::RotateSelection {
                                axis: Axis::X,
                                quarter: Quarter::Cw,
                            });
                            ui.close_menu();
                        }
                        if ui
                            .add_enabled(has_sel, egui::Button::new("-90°"))
                            .clicked()
                        {
                            self.state.request(UiAction::RotateSelection {
                                axis: Axis::X,
                                quarter: Quarter::Ccw,
                            });
                            ui.close_menu();
                        }
                        if ui
                            .add_enabled(has_sel, egui::Button::new("180°"))
                            .clicked()
                        {
                            self.state.request(UiAction::RotateSelection {
                                axis: Axis::X,
                                quarter: Quarter::Half,
                            });
                            ui.close_menu();
                        }
                    });
                    ui.menu_button("Rotate around Y", |ui| {
                        if ui
                            .add_enabled(has_sel, egui::Button::new("90° (R)"))
                            .clicked()
                        {
                            self.state.request(UiAction::RotateSelection {
                                axis: Axis::Y,
                                quarter: Quarter::Cw,
                            });
                            ui.close_menu();
                        }
                        if ui
                            .add_enabled(has_sel, egui::Button::new("-90° (Shift+R)"))
                            .clicked()
                        {
                            self.state.request(UiAction::RotateSelection {
                                axis: Axis::Y,
                                quarter: Quarter::Ccw,
                            });
                            ui.close_menu();
                        }
                        if ui
                            .add_enabled(has_sel, egui::Button::new("180°"))
                            .clicked()
                        {
                            self.state.request(UiAction::RotateSelection {
                                axis: Axis::Y,
                                quarter: Quarter::Half,
                            });
                            ui.close_menu();
                        }
                    });
                    ui.menu_button("Rotate around Z", |ui| {
                        if ui
                            .add_enabled(has_sel, egui::Button::new("90°"))
                            .clicked()
                        {
                            self.state.request(UiAction::RotateSelection {
                                axis: Axis::Z,
                                quarter: Quarter::Cw,
                            });
                            ui.close_menu();
                        }
                        if ui
                            .add_enabled(has_sel, egui::Button::new("-90°"))
                            .clicked()
                        {
                            self.state.request(UiAction::RotateSelection {
                                axis: Axis::Z,
                                quarter: Quarter::Ccw,
                            });
                            ui.close_menu();
                        }
                        if ui
                            .add_enabled(has_sel, egui::Button::new("180°"))
                            .clicked()
                        {
                            self.state.request(UiAction::RotateSelection {
                                axis: Axis::Z,
                                quarter: Quarter::Half,
                            });
                            ui.close_menu();
                        }
                    });
                    ui.separator();
                    if ui
                        .add_enabled(has_sel, egui::Button::new("Flip X (M)"))
                        .clicked()
                    {
                        self.state
                            .request(UiAction::MirrorSelection { axis: Axis::X });
                        ui.close_menu();
                    }
                    if ui
                        .add_enabled(has_sel, egui::Button::new("Flip Y"))
                        .clicked()
                    {
                        self.state
                            .request(UiAction::MirrorSelection { axis: Axis::Y });
                        ui.close_menu();
                    }
                    if ui
                        .add_enabled(has_sel, egui::Button::new("Flip Z"))
                        .clicked()
                    {
                        self.state
                            .request(UiAction::MirrorSelection { axis: Axis::Z });
                        ui.close_menu();
                    }
                });

                let wireframe_supported = self.wireframe_supported;
                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.state.panels.show_stats, "Statistics");
                    ui.checkbox(&mut self.state.panels.show_tools, "Tools Panel");
                    ui.checkbox(&mut self.state.panels.show_palette, "Color Palette");
                    ui.checkbox(&mut self.state.panels.show_viewport_settings, "Viewport Settings");
                    ui.checkbox(&mut self.state.panels.show_procgen, "Procedural Generation");
                    ui.checkbox(&mut self.state.panels.show_graph, "Pipeline Graph");
                    ui.checkbox(&mut self.state.panels.show_agent, "Agent Bridge");
                    ui.separator();
                    ui.checkbox(&mut self.viewport.show_grid, "Show Grid");
                    ui.checkbox(&mut self.viewport.show_axes, "Show Axes");
                    ui.add_enabled(
                        wireframe_supported,
                        egui::Checkbox::new(
                            &mut self.viewport.wireframe_mode,
                            "Wireframe Mode",
                        ),
                    )
                    .on_disabled_hover_text(WIREFRAME_UNSUPPORTED);
                    ui.checkbox(&mut self.viewport.show_hud, "Viewport HUD");
                    ui.checkbox(&mut self.viewport.show_perf_hud, "Performance HUD");
                });

                ui.menu_button("Generate", |ui| {
                    if ui.button("Test Cube").clicked() {
                        self.state.request(UiAction::GenerateTestCube);
                        ui.close_menu();
                    }
                    if ui.button("Ground Plane").clicked() {
                        self.state.request(UiAction::GenerateGround);
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Sphere").clicked() {
                        self.state.request(UiAction::GenerateSphere);
                        ui.close_menu();
                    }
                    if ui.button("Pyramid").clicked() {
                        self.state.request(UiAction::GeneratePyramid);
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Procedural Terrain...").clicked() {
                        // Select the generator the menu item names, not
                        // just the panel that hosts it. Opening the
                        // panel alone left it on whatever was last used
                        // — click "Procedural Terrain" and get the WFC
                        // or Tree parameters, which reads as the wrong
                        // panel rather than a stale selection.
                        self.procgen.selected = GeneratorChoice::Terrain;
                        self.state.panels.show_procgen = true;
                        ui.close_menu();
                    }
                });

                ui.menu_button("Help", |ui| {
                    if ui.button("Keyboard Shortcuts").clicked() {
                        self.state.show_help = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("About Voxelith").clicked() {
                        self.state.show_about = true;
                        ui.close_menu();
                    }
                });
            });
        });
    }

    fn show_toolbar(&mut self, ctx: &Context, editor: &mut Editor) {
        egui::SidePanel::left("toolbar")
            .resizable(false)
            .default_width(48.0)
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(8.0);

                    // Tool buttons. The tooltip's name and shortcut come
                    // from `Tool` itself — spelling them out per button
                    // meant eleven copies of the key map drifting away
                    // from the one the keyboard handler actually uses.
                    // `note` adds per-button detail where there is any.
                    let tool_button = |ui: &mut egui::Ui, tool: Tool, current: Tool, icon: &str, note: &str| -> bool {
                        let mut tooltip = tool.name().to_string();
                        if !tool.shortcut().is_empty() {
                            tooltip.push_str(&format!(" ({})", tool.shortcut()));
                        }
                        if !note.is_empty() {
                            tooltip.push('\n');
                            tooltip.push_str(note);
                        }
                        let selected = tool == current;
                        ui.add(
                            egui::Button::new(icon)
                                .min_size(egui::vec2(36.0, 36.0))
                                .selected(selected)
                        )
                        .on_hover_text(tooltip)
                        .clicked()
                    };

                    if tool_button(ui, Tool::Place, editor.current_tool, "+", "") {
                        editor.select_tool(Tool::Place);
                    }
                    if tool_button(ui, Tool::Remove, editor.current_tool, "-", "") {
                        editor.select_tool(Tool::Remove);
                    }
                    if tool_button(ui, Tool::Paint, editor.current_tool, "P", "") {
                        editor.select_tool(Tool::Paint);
                    }
                    if tool_button(ui, Tool::Eyedropper, editor.current_tool, "E", "") {
                        editor.select_tool(Tool::Eyedropper);
                    }
                    if tool_button(ui, Tool::Fill, editor.current_tool, "F", "") {
                        editor.select_tool(Tool::Fill);
                    }

                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(8.0);

                    // Shape tools — click-anchor / drag / release.
                    if tool_button(ui, Tool::Line, editor.current_tool, "L", "") {
                        editor.select_tool(Tool::Line);
                    }
                    if tool_button(ui, Tool::Box, editor.current_tool, "▢", "") {
                        editor.select_tool(Tool::Box);
                    }
                    if tool_button(ui, Tool::Sphere, editor.current_tool, "○", "") {
                        editor.select_tool(Tool::Sphere);
                    }
                    if tool_button(ui, Tool::Cylinder, editor.current_tool, "⌭", "") {
                        editor.select_tool(Tool::Cylinder);
                    }

                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(8.0);

                    // Selection — drag an AABB; Esc / Ctrl+D to clear.
                    if tool_button(
                        ui,
                        Tool::Select,
                        editor.current_tool,
                        "▭",
                        "Drag to mark an AABB. Esc or Ctrl+D deselects.",
                    ) {
                        editor.select_tool(Tool::Select);
                    }

                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(8.0);

                    // Socket — drop a named attachment point on a face.
                    if tool_button(
                        ui,
                        Tool::Socket,
                        editor.current_tool,
                        "⚓",
                        "Click a voxel face (or the ground) to drop a named \
                         attachment point. Exports to glTF as an empty node.",
                    ) {
                        editor.select_tool(Tool::Socket);
                    }

                    ui.add_space(16.0);
                    ui.separator();
                    ui.add_space(8.0);

                    // Current color preview
                    let color = egui::Color32::from_rgb(
                        editor.brush_color.r,
                        editor.brush_color.g,
                        editor.brush_color.b,
                    );
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(32.0, 32.0), egui::Sense::hover());
                    ui.painter().rect_filled(rect, 4.0, color);
                    ui.painter().rect_stroke(rect, 4.0, egui::Stroke::new(1.0, egui::Color32::WHITE));

                    ui.add_space(8.0);

                    // Brush size indicator
                    ui.label(format!("{}", editor.brush_size));
                });
            });
    }

    fn show_stats_panel(
        &mut self,
        ctx: &Context,
        stats: &RenderStats,
        editor: &Editor,
    ) {
        egui::Window::new("Statistics")
            .default_pos([60.0, 40.0])
            .resizable(false)
            .collapsible(true)
            .open(&mut self.state.panels.show_stats)
            .show(ctx, |ui| {
                egui::Grid::new("stats_grid")
                    .num_columns(2)
                    .spacing([20.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("FPS:");
                        ui.label(format!("{:.1}", stats.fps));
                        ui.end_row();

                        ui.label("Frame time:");
                        ui.label(format!("{:.2}ms", stats.frame_time_ms));
                        ui.end_row();

                        ui.label("Triangles:");
                        ui.label(format!("{}", stats.triangles));
                        ui.end_row();

                        ui.label("Chunks:");
                        ui.label(format!("{}", stats.chunks));
                        ui.end_row();

                        ui.label("History:");
                        ui.label(format!("{} / {}", editor.history.undo_count(), editor.history.redo_count()));
                        ui.end_row();
                    });

                ui.separator();

                ui.label(format!(
                    "Camera: ({:.1}, {:.1}, {:.1})",
                    stats.camera_pos.0, stats.camera_pos.1, stats.camera_pos.2
                ));
            });
    }

    fn show_tools_panel(&mut self, ctx: &Context, editor: &mut Editor) {
        // The close button's flag rides a local: `.open()` would borrow
        // `self.state.panels` for the whole window, while the closure
        // needs `self.state` for `request(...)`. Written back once both
        // borrows are released. Same shape in `show_viewport_panel`.
        let mut open = self.state.panels.show_tools;
        egui::Window::new("Tools")
            .default_pos([60.0, 200.0])
            .resizable(true)
            .collapsible(true)
            // This panel's content runs ~750px — taller than the whole
            // 1280x720 default window. Without its own scrollbar egui
            // clamps the window to the screen and everything below
            // Symmetry is simply unreachable.
            .vscroll(true)
            .open(&mut open)
            .show(ctx, |ui| {
                // Tool selection — split into Brush (cell-by-cell) and
                // Shape (click-anchor / drag / release) groups so the
                // distinct interaction model is visually clear.
                ui.heading("Brush");
                egui::Grid::new("brush_tool_grid")
                    .num_columns(3)
                    .spacing([4.0, 4.0])
                    .show(ui, |ui| {
                        if ui.selectable_label(editor.current_tool == Tool::Place, "Place").clicked() {
                            editor.select_tool(Tool::Place);
                        }
                        if ui.selectable_label(editor.current_tool == Tool::Remove, "Remove").clicked() {
                            editor.select_tool(Tool::Remove);
                        }
                        if ui.selectable_label(editor.current_tool == Tool::Paint, "Paint").clicked() {
                            editor.select_tool(Tool::Paint);
                        }
                        ui.end_row();

                        if ui.selectable_label(editor.current_tool == Tool::Eyedropper, "Pick").clicked() {
                            editor.select_tool(Tool::Eyedropper);
                        }
                        if ui.selectable_label(editor.current_tool == Tool::Fill, "Fill").clicked() {
                            editor.select_tool(Tool::Fill);
                        }
                        ui.end_row();
                    });

                ui.add_space(4.0);
                ui.heading("Shape");
                egui::Grid::new("shape_tool_grid")
                    .num_columns(3)
                    .spacing([4.0, 4.0])
                    .show(ui, |ui| {
                        if ui
                            .selectable_label(editor.current_tool == Tool::Line, "Line")
                            .on_hover_text("Drag from anchor to end (3D Bresenham line)")
                            .clicked()
                        {
                            editor.select_tool(Tool::Line);
                        }
                        if ui
                            .selectable_label(editor.current_tool == Tool::Box, "Box")
                            .on_hover_text("Drag corner to corner (filled AABB)")
                            .clicked()
                        {
                            editor.select_tool(Tool::Box);
                        }
                        if ui
                            .selectable_label(editor.current_tool == Tool::Sphere, "Sphere")
                            .on_hover_text("Drag bbox; ellipsoid fits in it")
                            .clicked()
                        {
                            editor.select_tool(Tool::Sphere);
                        }
                        ui.end_row();

                        if ui
                            .selectable_label(editor.current_tool == Tool::Cylinder, "Cylinder")
                            .on_hover_text(
                                "Drag a footprint, then pull up — the cylinder \
                                 stands along the height direction (the locked \
                                 face's normal)",
                            )
                            .clicked()
                        {
                            editor.select_tool(Tool::Cylinder);
                        }
                        ui.end_row();
                    });

                ui.add_space(4.0);
                ui.heading("Selection");
                if ui
                    .selectable_label(editor.current_tool == Tool::Select, "Box Select")
                    .on_hover_text(
                        "Drag corner-to-corner to mark an AABB region for batch \
                         operations. Esc or Ctrl+D deselects.",
                    )
                    .clicked()
                {
                    editor.select_tool(Tool::Select);
                }
                if let Some(sel) = editor.selection {
                    let (w, h, d) = sel.size();
                    ui.label(
                        egui::RichText::new(format!(
                            "Active: {}×{}×{} ({} cells)",
                            w,
                            h,
                            d,
                            sel.cell_count()
                        ))
                        .small()
                        .weak(),
                    );
                }
                let has_sel = editor.selection.is_some();
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(has_sel, egui::Button::new("Copy"))
                        .on_hover_text("Ctrl+C — copy non-air voxels into the clipboard")
                        .clicked()
                    {
                        self.state.request(UiAction::CopySelection);
                    }
                    if ui
                        .add_enabled(has_sel, egui::Button::new("Cut"))
                        .on_hover_text("Ctrl+X — copy then clear in one undoable Command")
                        .clicked()
                    {
                        self.state.request(UiAction::CutSelection);
                    }
                    let can_paste = self.has_clipboard;
                    if ui
                        .add_enabled(can_paste, egui::Button::new("Paste"))
                        .on_hover_text(
                            "Ctrl+V — paste at selection origin (or cursor cell if no \
                             selection). Ctrl+Shift+V always pastes at cursor.",
                        )
                        .clicked()
                    {
                        self.state.request(UiAction::PasteClipboard);
                    }
                });
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(has_sel, egui::Button::new("Delete"))
                        .on_hover_text("Del — clear non-air voxels inside the selection")
                        .clicked()
                    {
                        self.state.request(UiAction::DeleteSelection);
                    }
                    if ui
                        .button("Select All")
                        .on_hover_text("Ctrl+A — select the AABB of every non-air voxel")
                        .clicked()
                    {
                        self.state.request(UiAction::SelectAllSolid);
                    }
                    if ui
                        .add_enabled(has_sel, egui::Button::new("Deselect"))
                        .on_hover_text("Esc / Ctrl+D — clear the active selection")
                        .clicked()
                    {
                        // Goes through the action even though this panel
                        // holds `&mut Editor`: clearing a selection is
                        // more than `editor.selection = None` (drag/move
                        // anchors and the move ghost live on App). See
                        // `App::deselect`.
                        self.state.request(UiAction::Deselect);
                    }
                });

                ui.add_space(4.0);
                ui.heading("Sockets");
                if ui
                    .selectable_label(editor.current_tool == Tool::Socket, "Place Socket")
                    .on_hover_text(
                        "Click a voxel face (or the ground) to drop a named \
                         attachment point. Exports to glTF as an empty node \
                         (name + position + orientation).",
                    )
                    .clicked()
                {
                    editor.select_tool(Tool::Socket);
                }
                if editor.sockets.is_empty() {
                    ui.label(egui::RichText::new("No sockets yet.").small().weak());
                } else {
                    // Per-socket row: inline rename + delete + position
                    // readout. Names become glTF node names on export.
                    // Every mutation also raises `SocketsEdited`: this
                    // panel writes `editor.sockets` directly (its
                    // `&mut Editor` makes that legal), but sockets are
                    // document data no mesh rebuild notices, so without
                    // the action a rename or delete never marked the
                    // document modified — no save prompt, no autosave.
                    let mut to_delete: Option<usize> = None;
                    let mut edited = false;
                    egui::ScrollArea::vertical().max_height(120.0).show(ui, |ui| {
                        for (i, s) in editor.sockets.iter_mut().enumerate() {
                            ui.horizontal(|ui| {
                                if ui
                                    .add(
                                        egui::TextEdit::singleline(&mut s.name)
                                            .desired_width(110.0),
                                    )
                                    .on_hover_text("Name (becomes the glTF node name)")
                                    .changed()
                                {
                                    edited = true;
                                }
                                if ui
                                    .small_button("✕")
                                    .on_hover_text("Delete this socket")
                                    .clicked()
                                {
                                    to_delete = Some(i);
                                }
                                ui.label(
                                    egui::RichText::new(format!(
                                        "({:.1}, {:.1}, {:.1})",
                                        s.position[0], s.position[1], s.position[2]
                                    ))
                                    .small()
                                    .weak(),
                                );
                            });
                        }
                    });
                    if let Some(i) = to_delete {
                        editor.sockets.remove(i);
                        edited = true;
                    }
                    if ui
                        .button("Clear all sockets")
                        .on_hover_text("Remove every socket from the scene")
                        .clicked()
                    {
                        editor.sockets.clear();
                        edited = true;
                    }
                    if edited {
                        self.state.request(UiAction::SocketsEdited);
                    }
                }

                ui.separator();

                // Brush size
                ui.heading("Brush Size");
                let mut size = editor.brush_size as u32;
                ui.add(egui::Slider::new(&mut size, 1..=10).show_value(true));
                editor.brush_size = size as u8;

                ui.separator();

                // Symmetry
                ui.heading("Symmetry");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut editor.symmetry.x, "X")
                        .on_hover_text("Mirror brush across the x = 0 plane");
                    ui.checkbox(&mut editor.symmetry.y, "Y")
                        .on_hover_text("Mirror brush across the y = 0 plane");
                    ui.checkbox(&mut editor.symmetry.z, "Z")
                        .on_hover_text("Mirror brush across the z = 0 plane");
                });
                ui.label(
                    egui::RichText::new(
                        "Mirrors Place / Remove / Paint / Fill and the shape \
                         tools across enabled planes through the world origin. \
                         Eyedropper is exempt.",
                    )
                    .small()
                    .weak(),
                );

                ui.separator();

                // Color
                ui.heading("Color");
                let mut color = [
                    editor.brush_color.r as f32 / 255.0,
                    editor.brush_color.g as f32 / 255.0,
                    editor.brush_color.b as f32 / 255.0,
                ];
                if ui.color_edit_button_rgb(&mut color).changed() {
                    // Only RGB changes; keep alpha + material flags
                    // (emissive / metallic) so a color pick doesn't reset
                    // what behaves like a brush mode.
                    editor.brush_color.r = (color[0] * 255.0) as u8;
                    editor.brush_color.g = (color[1] * 255.0) as u8;
                    editor.brush_color.b = (color[2] * 255.0) as u8;
                }

                // RGB values
                ui.horizontal(|ui| {
                    ui.label("RGB:");
                    ui.label(format!("{}, {}, {}", editor.brush_color.r, editor.brush_color.g, editor.brush_color.b));
                });

                ui.separator();

                // Material flags baked into the brush's voxel template and
                // carried into GLB export as glTF materials. A brush mode,
                // like symmetry — picking a color preserves these.
                ui.heading("Material");
                ui.horizontal(|ui| {
                    let mut emissive = editor.brush_color.is_emissive();
                    if ui
                        .checkbox(&mut emissive, "Emissive")
                        .on_hover_text(
                            "Mark placed voxels as self-illuminating \
                             (exported as a glTF emissive material)",
                        )
                        .changed()
                    {
                        editor.brush_color.set_emissive(emissive);
                    }
                    let mut metallic = editor.brush_color.is_metallic();
                    if ui
                        .checkbox(&mut metallic, "Metallic")
                        .on_hover_text(
                            "Mark placed voxels as metal (exported as a \
                             glTF metallic material)",
                        )
                        .changed()
                    {
                        editor.brush_color.set_metallic(metallic);
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Tint zone");
                    let mut zone = editor.brush_color.tint_zone();
                    let before = zone;
                    let label = match zone {
                        1 => "Primary",
                        2 => "Secondary",
                        3 => "Reserved",
                        _ => "None",
                    };
                    egui::ComboBox::from_id_salt("brush_tint_zone")
                        .selected_text(label)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut zone, 0, "None");
                            ui.selectable_value(&mut zone, 1, "Primary");
                            ui.selectable_value(&mut zone, 2, "Secondary");
                            ui.selectable_value(&mut zone, 3, "Reserved");
                        });
                    if zone != before {
                        editor.brush_color.set_tint_zone(zone);
                    }
                })
                .response
                .on_hover_text(
                    "Faction recolor zone — exported per-vertex as _TINTZONE \
                     for a downstream uber-shader (does not change the editor view)",
                );

                // Show hovered voxel info
                if let Some(hit) = &editor.hovered_voxel {
                    ui.separator();
                    ui.heading("Hovered");
                    ui.label(format!("Position: ({}, {}, {})", hit.voxel_pos.0, hit.voxel_pos.1, hit.voxel_pos.2));
                    ui.label(format!("Face: ({}, {}, {})", hit.normal.0, hit.normal.1, hit.normal.2));
                }
            });
        self.state.panels.show_tools = open;
    }

    fn show_palette_panel(&mut self, ctx: &Context, editor: &mut Editor) {
        // Collected inside the window closure, applied after it — the
        // closure holds the borrow that `set_status` needs.
        let mut palette_feedback: Option<String> = None;
        egui::Window::new("Palette")
            // Right column: the left one is Statistics + Tools, and
            // Tools alone is taller than the default window, so a
            // left-column Palette started life buried under it.
            // (Float positions aren't persisted, so this constant is
            // what every session actually gets.)
            .default_pos([ctx.screen_rect().width() - 240.0, 40.0])
            .resizable(true)
            .collapsible(true)
            .open(&mut self.state.panels.show_palette)
            .show(ctx, |ui| {
                let palette = &editor.palette;
                let cols = 5;
                // Collected, then applied below: `set_palette_color`
                // takes `&mut Editor`, which can't coexist with the
                // `&editor.palette` this loop is iterating.
                let mut picked: Option<usize> = None;

                egui::Grid::new("palette_grid")
                    .spacing([4.0, 4.0])
                    .show(ui, |ui| {
                        for (i, voxel) in palette.iter().enumerate() {
                            let color = egui::Color32::from_rgb(voxel.r, voxel.g, voxel.b);
                            let is_selected = editor.brush_color.r == voxel.r
                                && editor.brush_color.g == voxel.g
                                && editor.brush_color.b == voxel.b;

                            let size = if is_selected { 24.0 } else { 20.0 };
                            let (rect, response) = ui.allocate_exact_size(
                                egui::vec2(size, size),
                                egui::Sense::click(),
                            );

                            if response.clicked() {
                                picked = Some(i);
                            }

                            ui.painter().rect_filled(rect, 2.0, color);
                            if is_selected {
                                ui.painter().rect_stroke(rect, 2.0, egui::Stroke::new(2.0, egui::Color32::WHITE));
                            }

                            if (i + 1) % cols == 0 {
                                ui.end_row();
                            }
                        }
                    });
                if let Some(i) = picked {
                    editor.set_palette_color(i);
                }

                ui.separator();

                // Quick color buttons
                ui.horizontal(|ui| {
                    if ui.button("Add").clicked() {
                        // Check if color already exists in palette
                        let color = editor.brush_color;
                        let exists = editor.palette.iter().any(|v| {
                            v.r == color.r && v.g == color.g && v.b == color.b
                        });
                        // Report both refusals. Silently doing nothing
                        // reads as a broken button — the user has no way
                        // to tell "already there" from "list is full"
                        // from "the click missed".
                        palette_feedback = Some(if exists {
                            format!(
                                "Palette already has RGB({}, {}, {})",
                                color.r, color.g, color.b
                            )
                        } else if editor.palette.len() >= MAX_PALETTE_COLORS {
                            format!(
                                "Palette is full ({} colors max) — \
                                 remove one first",
                                MAX_PALETTE_COLORS
                            )
                        } else {
                            editor.palette.push(color);
                            format!(
                                "Added RGB({}, {}, {}) to palette",
                                color.r, color.g, color.b
                            )
                        });
                    }
                });
            });
        // Set outside the window closure: `set_status` needs `&mut
        // self`, which the closure is already holding.
        if let Some(msg) = palette_feedback {
            self.set_status(msg);
        }
    }

    fn show_viewport_panel(&mut self, ctx: &Context) {
        let wireframe_supported = self.wireframe_supported;
        // Local close flag — see `show_tools_panel` for why.
        let mut open = self.state.panels.show_viewport_settings;
        egui::Window::new("Viewport Settings")
            // Below Palette, which now owns the top of the right column.
            .default_pos([ctx.screen_rect().width() - 240.0, 420.0])
            .resizable(false)
            .collapsible(true)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.heading("Display");
                ui.checkbox(&mut self.viewport.show_grid, "Show Grid");
                ui.checkbox(&mut self.viewport.show_axes, "Show Axes");
                ui.add_enabled(
                    wireframe_supported,
                    egui::Checkbox::new(
                        &mut self.viewport.wireframe_mode,
                        "Wireframe Mode",
                    ),
                )
                .on_disabled_hover_text(WIREFRAME_UNSUPPORTED);
                ui.checkbox(&mut self.viewport.show_hud, "Viewport HUD")
                    .on_hover_text(
                        "Tool & gesture readout in the bottom-left corner of the viewport",
                    );
                ui.checkbox(&mut self.viewport.show_perf_hud, "Performance HUD")
                    .on_hover_text(
                        "FPS, triangles, and re-mesh time in the bottom-right corner",
                    );

                ui.separator();

                ui.heading("Grid");
                ui.add(egui::Slider::new(&mut self.viewport.grid_size, 5..=50).text("Size"));
                ui.add(egui::Slider::new(&mut self.viewport.grid_spacing, 0.5..=5.0).text("Spacing"));

                ui.separator();

                ui.heading("Camera");
                if ui.button("Reset Camera").clicked() {
                    self.state.request(UiAction::ResetCamera);
                }

                ui.horizontal(|ui| {
                    if ui.button("Top").clicked() {
                        self.state.request(UiAction::SetCameraView(CameraView::Top));
                    }
                    if ui.button("Front").clicked() {
                        self.state.request(UiAction::SetCameraView(CameraView::Front));
                    }
                    if ui.button("Side").clicked() {
                        self.state.request(UiAction::SetCameraView(CameraView::Side));
                    }
                });

                ui.horizontal(|ui| {
                    if ui
                        .button("Frame All")
                        .on_hover_text("Fit the whole scene in view (F with no selection)")
                        .clicked()
                    {
                        self.state.request(UiAction::FrameAll);
                    }
                    if ui
                        .button("Frame Sel.")
                        .on_hover_text("Fit the selection in view (F with a selection)")
                        .clicked()
                    {
                        self.state.request(UiAction::FrameSelected);
                    }
                    if ui
                        .button("Frame Gen.")
                        .on_hover_text("Fit the most recent generation in view")
                        .clicked()
                    {
                        self.state.request(UiAction::FrameGenerated);
                    }
                });
            });
        self.state.panels.show_viewport_settings = open;
    }

    fn show_procgen_panel(&mut self, ctx: &Context) {
        // Deferred-action pattern: `.open(...)` borrows self.state.panels.show_procgen
        // and the closure borrows self.procgen, so we can't dispatch a UiAction
        // (which mutates self.state) until both are released.
        let mut generate = false;
        let procgen = &mut self.procgen;

        egui::Window::new("Procedural Generation")
            .default_pos([ctx.screen_rect().width() - 240.0, 200.0])
            .default_width(240.0)
            .resizable(true)
            .collapsible(true)
            .open(&mut self.state.panels.show_procgen)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Generator");
                    egui::ComboBox::from_id_salt("procgen_selected")
                        .selected_text(procgen.selected.label())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut procgen.selected,
                                GeneratorChoice::Terrain,
                                GeneratorChoice::Terrain.label(),
                            );
                            ui.selectable_value(
                                &mut procgen.selected,
                                GeneratorChoice::Tree,
                                GeneratorChoice::Tree.label(),
                            );
                            ui.selectable_value(
                                &mut procgen.selected,
                                GeneratorChoice::Wfc,
                                GeneratorChoice::Wfc.label(),
                            );
                        });
                });

                ui.separator();

                match procgen.selected {
                    GeneratorChoice::Terrain => {
                        terrain_params_ui(ui, &mut procgen.terrain)
                    }
                    GeneratorChoice::Tree => {
                        tree_params_ui(ui, &mut procgen.tree)
                    }
                    GeneratorChoice::Wfc => {
                        wfc_params_ui(ui, &mut procgen.wfc)
                    }
                }

                ui.separator();

                ui.horizontal(|ui| {
                    ui.checkbox(&mut procgen.preview_enabled, "Preview")
                        .on_hover_text(
                            "Show a translucent overlay of the generator's \
                             current output (debounced ~150ms)",
                        );
                    if ui
                        .button("Generate")
                        .on_hover_text("Apply generated voxels (undo-able)")
                        .clicked()
                    {
                        generate = true;
                    }
                });
            });

        if generate {
            self.state.request(UiAction::GenerateProcedural);
        }
    }

    fn show_graph_panel(&mut self, ctx: &Context) {
        // The graph as this frame found it. The panel edits it in a
        // dozen places — four deferred actions, a node drag, and every
        // widget in the inspector — and a `.vxlt` carries the result, so
        // *any* of them leaving the document looking unmodified is a
        // silent way to lose work: the unsaved-changes guard waves the
        // exit through, and the disk poll reloads over it.
        //
        // Comparing beginning to end of this one call is what makes the
        // signal trustworthy: everything that writes `self.graph` from
        // outside the panel (an agent's batch, opening a file, a reload)
        // happens between frames, so it can't be mistaken for a person
        // editing — the file paths deliberately land on a *clean*
        // document, and an agent's batch flags itself.
        let before = self.graph.clone();

        // Deferred actions: collected during the immediate-mode pass,
        // applied after the window closure releases its borrows on
        // `self.graph` and `self.state`.
        let mut run = false;
        let mut delete_id: Option<NodeId> = None;
        let mut add_kind: Option<NodeKind> = None;
        let mut auto_layout = false;
        let mut wire_action: Option<(NodeId, usize, Option<NodeId>)> = None;
        let mut wire_error: Option<String> = None;

        let graph = &mut self.graph;
        let selected = &mut self.selected_node;
        let drag_wire = &mut self.dragging_wire;
        let preview_enabled = &mut self.procgen.graph_preview_enabled;

        egui::Window::new("Pipeline Graph")
            .default_pos([240.0, 80.0])
            .default_size([960.0, 600.0])
            .min_size([520.0, 340.0])
            .resizable(true)
            .collapsible(true)
            .open(&mut self.state.panels.show_graph)
            .show(ctx, |ui| {
                // ===== Top toolbar =====
                ui.horizontal(|ui| {
                    if ui
                        .button("▶ Run Pipeline")
                        .on_hover_text("Evaluate the graph and apply (undo-able)")
                        .clicked()
                    {
                        run = true;
                    }
                    ui.checkbox(preview_enabled, "Preview").on_hover_text(
                        "Show a translucent overlay of the graph's output \
                         (debounced ~150ms)",
                    );
                    ui.separator();
                    ui.menu_button("+ Add Node", |ui| {
                        let has_output = graph
                            .nodes
                            .iter()
                            .any(|n| matches!(n.kind, NodeKind::Output { .. }));
                        for k in node_menu_options() {
                            let kind = (k.1)();
                            // Only one Output (sink) is allowed — evaluation
                            // needs a single pipeline result. Gray the entry
                            // out once one exists (#33).
                            let enabled =
                                !(has_output && matches!(kind, NodeKind::Output { .. }));
                            if ui.add_enabled(enabled, egui::Button::new(k.0)).clicked() {
                                add_kind = Some(kind);
                                ui.close_menu();
                            }
                            if k.2 {
                                ui.separator();
                            }
                        }
                    });
                    if ui.button("Auto Layout").on_hover_text("Re-grid all nodes").clicked()
                    {
                        auto_layout = true;
                    }
                    ui.separator();
                    ui.label(format!("Nodes: {}", graph.nodes.len()));
                });
                ui.separator();

                // ===== Split: canvas (left) + sidebar (right) =====
                let avail = ui.available_size();
                let (canvas_w, sidebar_w) =
                    graph_split_widths(avail.x, ui.spacing().item_spacing.x);

                ui.horizontal_top(|ui| {
                    // ---- Canvas ----
                    ui.allocate_ui_with_layout(
                        egui::vec2(canvas_w, avail.y),
                        egui::Layout::top_down(egui::Align::Min),
                        |ui| {
                            graph_canvas(
                                ui,
                                graph,
                                selected,
                                drag_wire,
                                &mut delete_id,
                                &mut wire_action,
                            );
                        },
                    );
                    // Explicit width, because the split arithmetic has to
                    // know exactly what this costs — see
                    // `graph_split_widths`.
                    ui.add(egui::Separator::default().spacing(GRAPH_DIVIDER_W));

                    // ---- Sidebar ----
                    ui.allocate_ui_with_layout(
                        egui::vec2(sidebar_w, avail.y),
                        egui::Layout::top_down(egui::Align::Min),
                        |ui| {
                            graph_sidebar(ui, graph, *selected, &mut wire_action);
                        },
                    );
                });
            });

        // ===== Apply deferred actions =====
        if let Some(id) = delete_id {
            graph.remove(id);
            if *selected == Some(id) {
                *selected = None;
            }
        }
        if let Some(kind) = add_kind {
            let id = graph.add(kind);
            *selected = Some(id);
        }
        if auto_layout {
            graph.relayout();
        }
        if let Some((target, slot, source)) = wire_action {
            if let Err(e) = graph.set_input(target, slot, source) {
                wire_error = Some(format!("{}", e));
            }
        }
        if let Some(msg) = wire_error {
            self.set_status(format!("Graph: {}", msg));
        }
        if run {
            self.state.request(UiAction::RunGraph);
        }
        if self.graph != before {
            self.state.request(UiAction::GraphEdited);
        }
    }

    fn show_help_panel(&mut self, ctx: &Context) {
        // The list runs to roughly 1200 px. On a 1080p screen the
        // window is taller than the space it has, and with
        // `resizable(false)` and no scrolling the tail — File and
        // Actions, i.e. the save/open shortcuts — was simply
        // unreachable. Cap the height against the actual screen and
        // scroll the overflow.
        let max_height = ctx.screen_rect().height() * 0.75;
        egui::Window::new("Keyboard Shortcuts")
            .default_pos([ctx.screen_rect().width() / 2.0 - 150.0, 100.0])
            .resizable(false)
            .collapsible(false)
            .open(&mut self.state.show_help)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .max_height(max_height)
                    .show(ui, |ui| {
                // One line instead of rewriting thirty entries per
                // platform: the chords below are bound to the
                // platform's command key (`primary_modifier` in
                // app/input), so the table stays written once.
                #[cfg(target_os = "macos")]
                {
                    ui.label(
                        egui::RichText::new("On macOS, use ⌘ wherever Ctrl is shown.")
                            .small()
                            .weak(),
                    );
                    ui.add_space(4.0);
                }
                egui::Grid::new("shortcuts_grid")
                    .num_columns(2)
                    .spacing([40.0, 4.0])
                    .show(ui, |ui| {
                        ui.heading("Tools");
                        ui.end_row();

                        ui.label("1");
                        ui.label("Place tool");
                        ui.end_row();

                        ui.label("2");
                        ui.label("Remove tool");
                        ui.end_row();

                        ui.label("3");
                        ui.label("Paint tool");
                        ui.end_row();

                        ui.label("4");
                        ui.label("Eyedropper");
                        ui.end_row();

                        ui.label("5");
                        ui.label("Fill tool");
                        ui.end_row();

                        ui.label("6");
                        ui.label("Line shape");
                        ui.end_row();

                        ui.label("7");
                        ui.label("Box shape");
                        ui.end_row();

                        ui.label("8");
                        ui.label("Sphere shape");
                        ui.end_row();

                        ui.label("9");
                        ui.label("Cylinder shape");
                        ui.end_row();

                        ui.label("0");
                        ui.label("Box select tool");
                        ui.end_row();

                        ui.label("Alt (hold)");
                        ui.label("Temporary eyedropper — restores on release");
                        ui.end_row();

                        ui.label("(toolbar only)");
                        ui.label("Socket tool — drop a named attach point");
                        ui.end_row();

                        ui.end_row();
                        ui.heading("Shape Tools (6–9)");
                        ui.end_row();

                        ui.label("First click + drag");
                        ui.label("Lay footprint on the locked face plane");
                        ui.end_row();

                        ui.label("Release");
                        ui.label("Enter height phase");
                        ui.end_row();

                        ui.label("Cursor up / down");
                        ui.label("Set extruded height (~8 px / voxel)");
                        ui.end_row();

                        ui.label("Second click");
                        ui.label("Commit the shape");
                        ui.end_row();

                        ui.label("Esc");
                        ui.label("Cancel shape");
                        ui.end_row();

                        ui.end_row();
                        ui.heading("Brush Drag-Paint");
                        ui.end_row();

                        ui.label("Press + drag");
                        ui.label("Paint stays on the first hit's face plane");
                        ui.end_row();

                        ui.end_row();
                        ui.heading("Edit");
                        ui.end_row();

                        ui.label("Ctrl+Z");
                        ui.label("Undo");
                        ui.end_row();

                        ui.label("Ctrl+Y");
                        ui.label("Redo");
                        ui.end_row();

                        ui.label("Ctrl+Shift+Z");
                        ui.label("Redo");
                        ui.end_row();

                        ui.end_row();
                        ui.heading("Selection");
                        ui.end_row();

                        ui.label("Drag in selection");
                        ui.label("Move (single SetVoxels Command)");
                        ui.end_row();

                        ui.label("Drag outside");
                        ui.label("Create new selection");
                        ui.end_row();

                        ui.label("Ctrl+C / Ctrl+X");
                        ui.label("Copy / Cut non-air voxels");
                        ui.end_row();

                        ui.label("Ctrl+V");
                        ui.label("Paste at selection origin (or cursor)");
                        ui.end_row();

                        ui.label("Ctrl+Shift+V");
                        ui.label("Paste at cursor cell");
                        ui.end_row();

                        ui.label("Del");
                        ui.label("Delete non-air voxels in selection");
                        ui.end_row();

                        ui.label("Ctrl+A");
                        ui.label("Select all (AABB of all solid voxels)");
                        ui.end_row();

                        ui.label("Esc / Ctrl+D");
                        ui.label("Deselect");
                        ui.end_row();

                        ui.label("Arrows");
                        ui.label("Nudge selection on X / Z (Shift × 10)");
                        ui.end_row();

                        ui.label("Ctrl + Up/Down");
                        ui.label("Nudge selection on Y axis");
                        ui.end_row();

                        ui.label("R / Shift+R");
                        ui.label("Rotate 90° around Y (CW / CCW)");
                        ui.end_row();

                        ui.label("M");
                        ui.label("Mirror across X (full axis set: Selection menu)");
                        ui.end_row();

                        ui.end_row();
                        ui.heading("Camera");
                        ui.end_row();

                        ui.label("WASD");
                        ui.label("Move camera");
                        ui.end_row();

                        ui.label("Q");
                        ui.label("Move up");
                        ui.end_row();

                        ui.label("E");
                        ui.label("Move down");
                        ui.end_row();

                        ui.label("Shift");
                        ui.label("Fly faster (×3) while moving");
                        ui.end_row();

                        ui.label("F");
                        ui.label("Frame selection (or whole scene)");
                        ui.end_row();

                        ui.label("Middle Mouse");
                        ui.label("Orbit camera");
                        ui.end_row();

                        ui.label("Right Mouse");
                        ui.label("Pan camera");
                        ui.end_row();

                        ui.label("Scroll");
                        ui.label("Zoom");
                        ui.end_row();

                        ui.label("Escape");
                        ui.label("Release cursor");
                        ui.end_row();

                        ui.end_row();
                        ui.heading("File");
                        ui.end_row();

                        ui.label("Ctrl+N");
                        ui.label("New project");
                        ui.end_row();

                        ui.label("Ctrl+O");
                        ui.label("Open project");
                        ui.end_row();

                        ui.label("Ctrl+S");
                        ui.label("Save project");
                        ui.end_row();

                        ui.label("Ctrl+Shift+S");
                        ui.label("Save as...");
                        ui.end_row();

                        ui.end_row();
                        ui.heading("Actions");
                        ui.end_row();

                        ui.label("Left Click");
                        ui.label("Apply tool");
                        ui.end_row();
                    });
                });
            });
    }

    fn show_about_dialog(&mut self, ctx: &Context) {
        egui::Window::new("About Voxelith")
            .default_pos([ctx.screen_rect().width() / 2.0 - 150.0, ctx.screen_rect().height() / 2.0 - 100.0])
            .resizable(false)
            .collapsible(false)
            .open(&mut self.state.show_about)
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.heading("Voxelith");
                    ui.add_space(8.0);
                    ui.label(format!("Version {}", env!("CARGO_PKG_VERSION")));
                    ui.add_space(16.0);
                    ui.label("Procedural-first voxel asset creation tool");
                    ui.add_space(8.0);
                    ui.label("Built with Rust, wgpu, and egui");
                    ui.add_space(16.0);
                    ui.separator();
                    ui.add_space(8.0);
                    // Read off the manifest rather than typed here: the
                    // license was stated in five places and one of them
                    // was this dialog, which is the copy nobody greps.
                    // `env!` makes it the same string `Cargo.toml`
                    // publishes, and a build with no license field
                    // fails to compile rather than shipping a blank.
                    ui.label(format!("{} License", env!("CARGO_PKG_LICENSE")));
                });
            });
    }

    fn show_status_bar(&mut self, ctx: &Context, editor: &Editor) {
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Show status message if recent (within 5 seconds)
                if let Some((msg, time)) = &self.state.status_message {
                    if time.elapsed().as_secs() < 5 {
                        ui.label(egui::RichText::new(msg).color(egui::Color32::YELLOW));
                        ui.separator();
                    } else {
                        self.state.status_message = None;
                    }
                }

                ui.label("Voxelith v0.1.0");
                ui.separator();
                // Tool name highlighted: easy to miss in the previous flat
                // style — users have ended up confused about which tool is
                // active (especially Fill / Eyedropper, which behave
                // very differently from the brush tools).
                ui.label(
                    egui::RichText::new(format!(
                        "Tool: {}",
                        editor.current_tool.name()
                    ))
                    .strong()
                    .color(egui::Color32::LIGHT_BLUE),
                );
                ui.separator();
                ui.label(format!("Brush: {}px", editor.brush_size));
                if editor.symmetry.any() {
                    ui.separator();
                    let mut axes = String::new();
                    if editor.symmetry.x { axes.push('X'); }
                    if editor.symmetry.y { axes.push('Y'); }
                    if editor.symmetry.z { axes.push('Z'); }
                    ui.label(
                        egui::RichText::new(format!("Sym: {}", axes))
                            .color(egui::Color32::LIGHT_YELLOW),
                    );
                }
                ui.separator();
                ui.label(format!(
                    "Color: RGB({}, {}, {})",
                    editor.brush_color.r, editor.brush_color.g, editor.brush_color.b
                ));
                if let Some(hit) = &editor.hovered_voxel {
                    ui.separator();
                    ui.label(format!(
                        "Cursor: ({}, {}, {})",
                        hit.voxel_pos.0, hit.voxel_pos.1, hit.voxel_pos.2
                    ));
                }
                if let Some(sel) = editor.selection {
                    let (w, h, d) = sel.size();
                    ui.separator();
                    ui.label(
                        egui::RichText::new(format!(
                            "Sel: {}×{}×{} ({} cells)",
                            w,
                            h,
                            d,
                            sel.cell_count()
                        ))
                        .color(egui::Color32::from_rgb(255, 230, 60)),
                    );
                }

                // Right-aligned viewport / preview info.
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Only claim wireframe when it's actually running:
                    // on a GPU without POLYGON_MODE_LINE the flag can
                    // still be set (from prefs written on other
                    // hardware) while the renderer draws solid.
                    if self.viewport.wireframe_mode && self.wireframe_supported {
                        ui.label("[Wireframe]");
                    }
                    if self.viewport.show_grid {
                        ui.label("[Grid]");
                    }
                    if self.viewport.show_axes {
                        ui.label("[Axes]");
                    }
                    if self.procgen.preview_enabled
                        || self.procgen.graph_preview_enabled
                    {
                        ui.label(
                            egui::RichText::new("● Preview")
                                .color(egui::Color32::LIGHT_GREEN),
                        );
                    }
                });
            });
        });
    }

    /// Set a status message to display
    pub fn set_status(&mut self, message: impl Into<String>) {
        self.state.status_message = Some((message.into(), std::time::Instant::now()));
    }

    /// Clear one-shot action flags
    pub fn clear_flags(&mut self) {
        self.state.clear_actions();
    }
}

impl Default for Ui {
    fn default() -> Self {
        Self::new()
    }
}

/// Render statistics for UI display
#[derive(Default)]
pub struct RenderStats {
    pub fps: f32,
    pub frame_time_ms: f32,
    pub triangles: usize,
    pub chunks: usize,
    pub camera_pos: (f32, f32, f32),
    /// `(milliseconds, chunk count)` of the most recent dirty-chunk
    /// re-mesh (generation + upload). `None` until the first rebuild.
    pub last_rebuild: Option<(f32, usize)>,
}

/// Preset camera views
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CameraView {
    Top,
    Front,
    Side,
}

// ---- Procgen panel parameter editors ---------------------------------
//
// Free functions so the procgen panel's borrow on `self.procgen` can
// dispatch to the right editor without involving `&mut self`. They take
// only the generator's parameter struct.

fn terrain_params_ui(ui: &mut egui::Ui, t: &mut PerlinTerrain) {
    ui.heading(GeneratorChoice::Terrain.label());
    ui.add_space(4.0);

    egui::Grid::new("terrain_params")
        .num_columns(2)
        .spacing([10.0, 4.0])
        .show(ui, |ui| {
            ui.label("Seed");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut t.seed).speed(1.0));
                if ui
                    .button("Rand")
                    .on_hover_text("Randomize seed")
                    .clicked()
                {
                    t.seed = rand::random();
                }
            });
            ui.end_row();

            ui.label("Width");
            ui.add(egui::Slider::new(&mut t.width, 8..=256));
            ui.end_row();

            ui.label("Depth");
            ui.add(egui::Slider::new(&mut t.depth, 8..=256));
            ui.end_row();

            // The two heights are independent sliders but not
            // independent values: the generator rejects min > max, and
            // with Preview on that rejection only reached a log line —
            // the overlay just vanished. Let whichever slider the user
            // is dragging push the other, so the invalid combination is
            // unreachable from the UI in the first place.
            ui.label("Min Y");
            ui.add(egui::Slider::new(&mut t.min_height, -64..=64));
            if t.max_height < t.min_height {
                t.max_height = t.min_height;
            }
            ui.end_row();

            ui.label("Max Y");
            ui.add(egui::Slider::new(&mut t.max_height, -64..=128));
            if t.min_height > t.max_height {
                t.min_height = t.max_height;
            }
            ui.end_row();

            ui.label("Frequency");
            ui.add(
                egui::Slider::new(&mut t.frequency, 0.005..=0.5)
                    .logarithmic(true),
            );
            ui.end_row();

            ui.label("Octaves");
            ui.add(egui::Slider::new(&mut t.octaves, 1..=8));
            ui.end_row();
        });

    ui.label(format!(
        "{} × {} × {}",
        t.width,
        t.depth,
        (t.max_height - t.min_height).max(0)
    ));
}

fn tree_params_ui(ui: &mut egui::Ui, t: &mut LSystemTree) {
    ui.heading(GeneratorChoice::Tree.label());
    ui.add_space(4.0);

    egui::Grid::new("tree_params")
        .num_columns(2)
        .spacing([10.0, 4.0])
        .show(ui, |ui| {
            ui.label("Seed");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut t.seed).speed(1.0));
                if ui
                    .button("Rand")
                    .on_hover_text("Randomize seed")
                    .clicked()
                {
                    t.seed = rand::random();
                }
            });
            ui.end_row();

            ui.label("Iterations");
            ui.add(egui::Slider::new(&mut t.iterations, 1..=6));
            ui.end_row();

            ui.label("Angle (°)");
            ui.add(egui::Slider::new(&mut t.angle_deg, 5.0..=60.0));
            ui.end_row();

            ui.label("Init length");
            ui.add(egui::Slider::new(&mut t.initial_length, 1.0..=12.0));
            ui.end_row();

            ui.label("Length scale");
            ui.add(egui::Slider::new(&mut t.length_scale, 0.4..=1.0));
            ui.end_row();

            ui.label("Origin");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut t.origin.0).prefix("x:"));
                ui.add(egui::DragValue::new(&mut t.origin.1).prefix("y:"));
                ui.add(egui::DragValue::new(&mut t.origin.2).prefix("z:"));
            });
            ui.end_row();

            ui.label("Trunk");
            color_button_u8(ui, &mut t.trunk_color);
            ui.end_row();

            ui.label("Leaves");
            color_button_u8(ui, &mut t.leaf_color);
            ui.end_row();
        });
}

fn wfc_params_ui(ui: &mut egui::Ui, t: &mut WfcGenerator) {
    ui.heading(GeneratorChoice::Wfc.label());
    ui.add_space(4.0);

    egui::Grid::new("wfc_params")
        .num_columns(2)
        .spacing([10.0, 4.0])
        .show(ui, |ui| {
            ui.label("Seed");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut t.seed).speed(1.0));
                if ui
                    .button("Rand")
                    .on_hover_text("Randomize seed")
                    .clicked()
                {
                    t.seed = rand::random();
                }
            });
            ui.end_row();

            ui.label("Width (tiles)");
            ui.add(egui::Slider::new(&mut t.width, 2..=24));
            ui.end_row();

            ui.label("Depth (tiles)");
            ui.add(egui::Slider::new(&mut t.depth, 2..=24));
            ui.end_row();

            ui.label("Origin");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut t.origin.0).prefix("x:"));
                ui.add(egui::DragValue::new(&mut t.origin.1).prefix("y:"));
                ui.add(egui::DragValue::new(&mut t.origin.2).prefix("z:"));
            });
            ui.end_row();

            ui.label("Tileset");
            egui::ComboBox::from_id_salt("wfc_tileset")
                .selected_text(t.tileset.label())
                .show_ui(ui, |ui| {
                    for &option in WfcTileset::ALL {
                        ui.selectable_value(&mut t.tileset, option, option.label());
                    }
                });
            ui.end_row();
        });

    let s = crate::procgen::WFC_TILE_SIZE as i32;
    ui.label(format!(
        "≈ {} × {} × {} voxels",
        t.width as i32 * s,
        s,
        t.depth as i32 * s
    ));
}

// =============================================================
// Visual graph editor: layout constants + helpers
// =============================================================

const NODE_W: f32 = 168.0;
const NODE_H: f32 = 84.0;
const NODE_HEADER_H: f32 = 22.0;
const SOCKET_R: f32 = 6.0;
const SOCKET_HIT_R: f32 = SOCKET_R + 4.0;

/// Width the divider between the graph canvas and its sidebar occupies.
/// Set explicitly on the `Separator` rather than left to the default,
/// so [`graph_split_widths`] can subtract exactly what it costs.
const GRAPH_DIVIDER_W: f32 = 12.0;

/// Sidebar width, and the floor the canvas may not shrink past.
const GRAPH_SIDEBAR_W: f32 = 280.0;
const GRAPH_SIDEBAR_MIN_W: f32 = 220.0;
const GRAPH_CANVAS_MIN_W: f32 = 200.0;

/// Split the graph window's inner width into canvas and sidebar.
///
/// The two sit in one `horizontal_top` row with a separator between
/// them, so what the row actually consumes is
/// `canvas + spacing + divider + spacing + sidebar` — and if that comes
/// out wider than what was available, egui grows the window to fit,
/// which enlarges the available width, which widens the row again. The
/// window then creeps outward every frame until it hits the screen.
/// That is exactly what happened while this reserved a flat 12 px for a
/// divider that cost 22: ten pixels a frame.
///
/// So the arithmetic is stated once, here, and pinned by a test: the
/// parts must sum to *exactly* the width handed in.
fn graph_split_widths(available: f32, item_spacing: f32) -> (f32, f32) {
    let overhead = GRAPH_DIVIDER_W + item_spacing * 2.0;
    // Shrink the sidebar before the canvas, and stop both at their
    // floors — at which point the row is wider than the window and egui
    // scrolls or clips it, rather than the window growing without end.
    // `min_size` on the window keeps this out of reach in practice.
    let sidebar = GRAPH_SIDEBAR_W
        .min(available * 0.4)
        .max(GRAPH_SIDEBAR_MIN_W)
        .min((available - overhead - GRAPH_CANVAS_MIN_W).max(GRAPH_SIDEBAR_MIN_W));
    let canvas = (available - sidebar - overhead).max(GRAPH_CANVAS_MIN_W);
    (canvas, sidebar)
}

/// Available node kinds in the "+ Add Node" menu.
/// Tuple is (label, factory, separator_after).
fn node_menu_options() -> Vec<(&'static str, fn() -> NodeKind, bool)> {
    vec![
        ("Source: Terrain", || NodeKind::Terrain(PerlinTerrain::default()), false),
        ("Source: Tree", || NodeKind::Tree(LSystemTree::default()), false),
        ("Source: WFC", || NodeKind::Wfc(WfcGenerator::default()), true),
        (
            "Translate",
            || NodeKind::Translate { input: None, dx: 0, dy: 0, dz: 0 },
            false,
        ),
        (
            "Filter",
            || NodeKind::Filter {
                input: None,
                predicate: FilterPredicate::default(),
            },
            false,
        ),
        (
            "Mask",
            || NodeKind::Mask {
                subject: None,
                mask: None,
                mode: MaskMode::default(),
            },
            false,
        ),
        (
            "Combine",
            || NodeKind::Combine {
                a: None,
                b: None,
                op: CombineOp::Union,
            },
            true,
        ),
        ("Output", || NodeKind::Output { input: None }, false),
    ]
}

/// Header tint per node kind — gives a quick visual key for source vs.
/// transform vs. sink.
fn node_header_color(kind: &NodeKind) -> egui::Color32 {
    match kind {
        NodeKind::Terrain(_) => egui::Color32::from_rgb(70, 110, 60),
        NodeKind::Tree(_) => egui::Color32::from_rgb(60, 100, 60),
        NodeKind::Wfc(_) => egui::Color32::from_rgb(100, 90, 50),
        NodeKind::Translate { .. } => egui::Color32::from_rgb(70, 80, 110),
        NodeKind::Filter { .. } => egui::Color32::from_rgb(80, 100, 110),
        NodeKind::Mask { .. } => egui::Color32::from_rgb(90, 110, 130),
        NodeKind::Combine { .. } => egui::Color32::from_rgb(110, 70, 110),
        NodeKind::Output { .. } => egui::Color32::from_rgb(120, 80, 60),
    }
}

/// One- or two-line summary shown under the header inside the node box.
fn node_summary(kind: &NodeKind) -> String {
    match kind {
        NodeKind::Terrain(t) => {
            format!("seed {} • {}×{}", t.seed, t.width, t.depth)
        }
        NodeKind::Tree(t) => {
            format!("seed {} • iter {}", t.seed, t.iterations)
        }
        NodeKind::Wfc(t) => {
            format!("seed {} • {}×{}", t.seed, t.width, t.depth)
        }
        NodeKind::Translate { dx, dy, dz, .. } => {
            format!("offset ({}, {}, {})", dx, dy, dz)
        }
        NodeKind::Filter { predicate, .. } => predicate.label(),
        NodeKind::Mask { mode, .. } => mode.label().to_string(),
        NodeKind::Combine { op, .. } => op.label().to_string(),
        NodeKind::Output { .. } => "pipeline result".to_string(),
    }
}

/// Screen-space bounding box of a node body.
fn node_screen_rect(canvas_min: egui::Pos2, node: &crate::procgen::GraphNode) -> egui::Rect {
    egui::Rect::from_min_size(
        canvas_min + egui::vec2(node.position[0], node.position[1]),
        egui::vec2(NODE_W, NODE_H),
    )
}

/// Center of an input socket in screen space. Combine nodes have
/// two inputs stacked vertically; everyone else has one centered.
fn input_socket_screen(
    canvas_min: egui::Pos2,
    node: &crate::procgen::GraphNode,
    slot: usize,
) -> egui::Pos2 {
    let body = node_screen_rect(canvas_min, node);
    match &node.kind {
        // Both 2-input kinds get vertically-stacked sockets so slot 0 and
        // slot 1 land at DISTINCT positions. Mask was missing here, so its
        // two sockets overlapped at body-center and the wire hit-test —
        // which stops at the first slot within radius — could never reach
        // slot 1 (#17).
        NodeKind::Combine { .. } | NodeKind::Mask { .. } => {
            let body_inner_top = body.min.y + NODE_HEADER_H + 14.0;
            let y = body_inner_top + slot as f32 * 22.0;
            egui::pos2(body.min.x, y)
        }
        _ => egui::pos2(body.min.x, body.center().y + 6.0),
    }
}

/// Center of a node's output socket (right edge).
fn output_socket_screen(
    canvas_min: egui::Pos2,
    node: &crate::procgen::GraphNode,
) -> egui::Pos2 {
    let body = node_screen_rect(canvas_min, node);
    egui::pos2(body.max.x, body.center().y + 6.0)
}

/// Sample a cubic Bezier at parameter `t ∈ [0, 1]`.
fn cubic_bezier_point(
    p0: egui::Pos2,
    p1: egui::Pos2,
    p2: egui::Pos2,
    p3: egui::Pos2,
    t: f32,
) -> egui::Pos2 {
    let omt = 1.0 - t;
    let omt2 = omt * omt;
    let omt3 = omt2 * omt;
    let t2 = t * t;
    let t3 = t2 * t;
    egui::pos2(
        omt3 * p0.x + 3.0 * omt2 * t * p1.x + 3.0 * omt * t2 * p2.x + t3 * p3.x,
        omt3 * p0.y + 3.0 * omt2 * t * p1.y + 3.0 * omt * t2 * p2.y + t3 * p3.y,
    )
}

/// Draw a wire from `from` (output socket) to `to` (input socket) as a
/// horizontally-bowed cubic Bezier — the standard look for node-graph
/// editors. Tessellated to a polyline so we don't depend on egui's
/// CubicBezierShape API across versions.
fn paint_wire(
    painter: &egui::Painter,
    from: egui::Pos2,
    to: egui::Pos2,
    color: egui::Color32,
) {
    let dx = (to.x - from.x).abs().max(40.0);
    let c1 = egui::pos2(from.x + dx * 0.5, from.y);
    let c2 = egui::pos2(to.x - dx * 0.5, to.y);

    const SEGMENTS: usize = 24;
    let mut pts = Vec::with_capacity(SEGMENTS + 1);
    for i in 0..=SEGMENTS {
        let t = i as f32 / SEGMENTS as f32;
        pts.push(cubic_bezier_point(from, c1, c2, to, t));
    }
    painter.add(egui::Shape::line(pts, egui::Stroke::new(2.0, color)));
}

/// Visual graph editor canvas. Renders nodes + wires, handles
/// click-select, body-drag, and socket-drag wire creation. Mutations
/// to the graph (input slot changes, deletion) are deferred via the
/// out-params so the caller can apply them outside of the borrow.
fn graph_canvas(
    ui: &mut egui::Ui,
    graph: &mut PipelineGraph,
    selected: &mut Option<NodeId>,
    drag_wire: &mut Option<NodeId>,
    delete_id: &mut Option<NodeId>,
    wire_action: &mut Option<(NodeId, usize, Option<NodeId>)>,
) {
    let avail = ui.available_size();
    let (canvas_rect, _bg) =
        ui.allocate_exact_size(avail, egui::Sense::hover());
    let painter = ui.painter_at(canvas_rect);

    // Background.
    painter.rect_filled(
        canvas_rect,
        0.0,
        egui::Color32::from_rgb(28, 28, 36),
    );

    // ===== Wires (drawn before nodes so they pass under boxes) =====
    for node in &graph.nodes {
        let in_count = PipelineGraph::input_count(&node.kind);
        for slot in 0..in_count {
            // Canonical accessor so EVERY node kind's slots are covered.
            // The old inline match omitted Filter's input and both of
            // Mask's slots (they fell through to `None`), so those wires
            // were never drawn. `get_input` is Ok for any in-range slot on
            // an existing node, so `.ok().flatten()` = the wired source.
            let Some(src_id) = graph.get_input(node.id, slot).ok().flatten() else {
                continue;
            };
            let Some(src) = graph.get(src_id) else { continue };
            let from = output_socket_screen(canvas_rect.min, src);
            let to = input_socket_screen(canvas_rect.min, node, slot);
            let highlighted = *selected == Some(node.id) || *selected == Some(src_id);
            let color = if highlighted {
                egui::Color32::from_rgb(180, 200, 255)
            } else {
                egui::Color32::from_rgb(140, 140, 160)
            };
            paint_wire(&painter, from, to, color);
        }
    }

    // ===== Live wire (while a socket-drag is active) =====
    if let Some(src_id) = *drag_wire {
        if let Some(src) = graph.get(src_id) {
            let from = output_socket_screen(canvas_rect.min, src);
            let to = ui
                .ctx()
                .input(|i| i.pointer.interact_pos())
                .unwrap_or(from);
            paint_wire(&painter, from, to, egui::Color32::YELLOW);
        }
    }

    // ===== Nodes =====
    // Two passes: first allocate all node body widgets so their drag
    // responses are registered, then draw + handle sockets. Splitting
    // keeps z-order predictable (sockets sit on top of body).
    //
    // First pass: register a click-and-drag interaction over each
    // node body so egui can route hover / click / drag events. We
    // capture the per-body response so the second pass can apply the
    // delta to the node's position without re-allocating.
    struct NodeFrame {
        body_resp: egui::Response,
        delta: egui::Vec2,
    }
    let mut frames: Vec<(NodeId, NodeFrame)> = Vec::with_capacity(graph.nodes.len());

    for node in &graph.nodes {
        let body = node_screen_rect(canvas_rect.min, node);
        let body_id = ui.id().with(("graph_node_body", node.id));
        let body_resp = ui.interact(body, body_id, egui::Sense::click_and_drag());
        let delta = if body_resp.dragged() {
            body_resp.drag_delta()
        } else {
            egui::Vec2::ZERO
        };
        frames.push((node.id, NodeFrame { body_resp, delta }));
    }

    // Apply body drags + clicks (mutates graph.position / selected).
    // A dragged node is kept inside the canvas: `ui.interact` isn't
    // clipped, so one dragged past the edge stays clickable but is
    // never drawn — invisible is lost, as far as the user is concerned.
    // Only drags clamp; re-flowing already-placed nodes on every resize
    // would squash a saved layout the moment the panel got narrow, and
    // widening it back wouldn't restore the positions.
    let drag_limit = egui::vec2(
        (canvas_rect.width() - NODE_W).max(0.0),
        (canvas_rect.height() - NODE_H).max(0.0),
    );
    for (id, frame) in &frames {
        if frame.body_resp.clicked() {
            *selected = Some(*id);
        }
        if frame.body_resp.dragged() {
            *selected = Some(*id);
            if let Some(node) = graph.get_mut(*id) {
                node.position[0] =
                    (node.position[0] + frame.delta.x).clamp(0.0, drag_limit.x);
                node.position[1] =
                    (node.position[1] + frame.delta.y).clamp(0.0, drag_limit.y);
            }
        }
    }

    // Visual + socket pass. Reads `&graph.nodes` directly — the drag
    // loop above is done with its `get_mut`, and the positions it wrote
    // are already visible here, so mid-drag frames stay smooth without
    // copying every node's parameters once per frame.
    for node in &graph.nodes {
        let body = node_screen_rect(canvas_rect.min, node);
        let is_selected = *selected == Some(node.id);

        // Body fill + outline.
        painter.rect_filled(body, 4.0, egui::Color32::from_rgb(50, 50, 60));
        let outline = if is_selected {
            egui::Stroke::new(2.0, egui::Color32::LIGHT_BLUE)
        } else {
            egui::Stroke::new(1.0, egui::Color32::from_gray(80))
        };
        painter.rect_stroke(body, 4.0, outline);

        // Header.
        let header = egui::Rect::from_min_max(
            body.min,
            egui::pos2(body.max.x, body.min.y + NODE_HEADER_H),
        );
        painter.rect_filled(header, 4.0, node_header_color(&node.kind));
        painter.text(
            header.min + egui::vec2(8.0, 3.0),
            egui::Align2::LEFT_TOP,
            format!("#{}  {}", node.id, node.kind.label()),
            egui::FontId::proportional(12.0),
            egui::Color32::WHITE,
        );

        // Delete × button (top-right corner of header).
        let close_size = 16.0;
        let close_rect = egui::Rect::from_min_size(
            egui::pos2(header.max.x - close_size - 2.0, header.min.y + 3.0),
            egui::vec2(close_size, close_size),
        );
        let close_id = ui.id().with(("graph_node_close", node.id));
        let close_resp =
            ui.interact(close_rect, close_id, egui::Sense::click());
        let close_color = if close_resp.hovered() {
            egui::Color32::from_rgb(255, 120, 120)
        } else {
            egui::Color32::from_gray(220)
        };
        painter.text(
            close_rect.center(),
            egui::Align2::CENTER_CENTER,
            "×",
            egui::FontId::proportional(14.0),
            close_color,
        );
        if close_resp.clicked() {
            *delete_id = Some(node.id);
        }

        // Summary text.
        painter.text(
            body.min + egui::vec2(8.0, NODE_HEADER_H + 6.0),
            egui::Align2::LEFT_TOP,
            node_summary(&node.kind),
            egui::FontId::proportional(11.0),
            egui::Color32::from_gray(200),
        );

        // Input sockets.
        for slot in 0..PipelineGraph::input_count(&node.kind) {
            let center = input_socket_screen(canvas_rect.min, node, slot);
            let hit_rect = egui::Rect::from_center_size(
                center,
                egui::vec2(SOCKET_HIT_R * 2.0, SOCKET_HIT_R * 2.0),
            );
            let in_id =
                ui.id().with(("graph_in_sock", node.id, slot));
            let in_resp = ui.interact(hit_rect, in_id, egui::Sense::hover());
            let hot = drag_wire.is_some() && in_resp.hovered();
            let color = if hot {
                egui::Color32::from_rgb(255, 230, 100)
            } else {
                egui::Color32::from_rgb(180, 180, 200)
            };
            painter.circle_filled(center, SOCKET_R, color);
            painter.circle_stroke(
                center,
                SOCKET_R,
                egui::Stroke::new(1.0, egui::Color32::BLACK),
            );
        }

        // Output socket.
        if PipelineGraph::has_output(&node.kind) {
            let center = output_socket_screen(canvas_rect.min, node);
            let hit_rect = egui::Rect::from_center_size(
                center,
                egui::vec2(SOCKET_HIT_R * 2.0, SOCKET_HIT_R * 2.0),
            );
            let out_id = ui.id().with(("graph_out_sock", node.id));
            let out_resp =
                ui.interact(hit_rect, out_id, egui::Sense::drag());
            painter.circle_filled(
                center,
                SOCKET_R,
                egui::Color32::from_rgb(220, 200, 100),
            );
            painter.circle_stroke(
                center,
                SOCKET_R,
                egui::Stroke::new(1.0, egui::Color32::BLACK),
            );
            if out_resp.drag_started() {
                *drag_wire = Some(node.id);
            }
            if out_resp.drag_stopped() && *drag_wire == Some(node.id) {
                // Hit-test cursor against every input socket.
                let p = ui.ctx().input(|i| i.pointer.interact_pos());
                let mut hit: Option<(NodeId, usize)> = None;
                if let Some(p) = p {
                    'outer: for target in &graph.nodes {
                        if target.id == node.id {
                            continue;
                        }
                        for slot in 0..PipelineGraph::input_count(&target.kind) {
                            let s = input_socket_screen(
                                canvas_rect.min,
                                target,
                                slot,
                            );
                            if (s - p).length() <= SOCKET_HIT_R {
                                hit = Some((target.id, slot));
                                break 'outer;
                            }
                        }
                    }
                }
                if let Some((target_id, slot)) = hit {
                    *wire_action = Some((target_id, slot, Some(node.id)));
                }
                // Released into empty space → no-op (intentional cancel).
                *drag_wire = None;
            }
        }
    }

    // If the user dragged and the cursor was released anywhere outside
    // the canvas (or pointer became unavailable), still cancel the
    // pending wire so we don't leave a stuck live wire on next frame.
    if drag_wire.is_some() && ui.ctx().input(|i| !i.pointer.any_down()) {
        *drag_wire = None;
    }

    // Empty-graph hint.
    if graph.nodes.is_empty() {
        painter.text(
            canvas_rect.center(),
            egui::Align2::CENTER_CENTER,
            "Empty pipeline.\nUse \"+ Add Node\" above.",
            egui::FontId::proportional(13.0),
            egui::Color32::from_gray(120),
        );
    }
}

/// Right-side parameter editor. Shows the selected node's params,
/// plus connection ComboBoxes (kept as a fallback to visual wiring,
/// useful for disconnecting / reading the current state).
fn graph_sidebar(
    ui: &mut egui::Ui,
    graph: &mut PipelineGraph,
    selected: Option<NodeId>,
    wire_action: &mut Option<(NodeId, usize, Option<NodeId>)>,
) {
    ui.heading("Inspector");
    ui.add_space(4.0);

    let Some(id) = selected else {
        ui.label("Click a node in the canvas to edit its parameters.");
        return;
    };

    // Snapshot of node ids for input ComboBoxes (avoids holding an
    // immutable borrow on graph.nodes while we mutate one node below).
    // Candidate sources = every node that HAS an output socket. Output
    // nodes are sinks (no output), so excluding them stops the dropdown
    // from offering a node that produces nothing (#18).
    let candidates: Vec<(NodeId, String)> = graph
        .nodes
        .iter()
        .filter(|n| PipelineGraph::has_output(&n.kind))
        .map(|n| (n.id, format!("#{}: {}", n.id, n.kind.label())))
        .collect();

    let Some(node) = graph.get_mut(id) else {
        ui.label("(node not found)");
        return;
    };

    ui.label(
        egui::RichText::new(format!("#{}  {}", node.id, node.kind.label()))
            .strong(),
    );
    ui.separator();
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| match &mut node.kind {
            NodeKind::Terrain(t) => terrain_params_ui(ui, t),
            NodeKind::Tree(t) => tree_params_ui(ui, t),
            NodeKind::Wfc(t) => wfc_params_ui(ui, t),
            NodeKind::Translate { input, dx, dy, dz } => {
                input_slot(ui, "Input", *input, id, 0, &candidates, wire_action);
                ui.horizontal(|ui| {
                    ui.label("Offset");
                    // Bounded so the panel can't casually scatter
                    // geometry thousands of cells apart — smoothed
                    // export builds a dense field over the scene's
                    // whole bounding box. This is UX guidance, not a
                    // safety limit (offsets compose across nodes, and
                    // a saved graph can carry any value); the real
                    // ceiling lives in `mesh_world_smoothed`.
                    ui.add(egui::DragValue::new(dx).prefix("x:").range(-1024..=1024));
                    ui.add(egui::DragValue::new(dy).prefix("y:").range(-1024..=1024));
                    ui.add(egui::DragValue::new(dz).prefix("z:").range(-1024..=1024));
                });
            }
            NodeKind::Filter { input, predicate } => {
                input_slot(ui, "Input", *input, id, 0, &candidates, wire_action);
                filter_predicate_ui(ui, predicate, id);
            }
            NodeKind::Mask { subject, mask, mode } => {
                input_slot(ui, "Subject", *subject, id, 0, &candidates, wire_action);
                input_slot(ui, "Mask", *mask, id, 1, &candidates, wire_action);
                ui.horizontal(|ui| {
                    ui.label("Mode");
                    egui::ComboBox::from_id_salt(("mask_mode_sb", id))
                        .selected_text(mode.label())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                mode,
                                MaskMode::AboveColumn,
                                "Above column",
                            );
                            ui.selectable_value(
                                mode,
                                MaskMode::BelowColumn,
                                "Below column",
                            );
                        });
                });
                ui.label(
                    egui::RichText::new(
                        "Keeps subject voxels based on mask's column profile. \
                         Above-column → trees above terrain; Below-column → \
                         stalactites below ceilings.",
                    )
                    .small()
                    .weak(),
                );
            }
            NodeKind::Combine { a, b, op } => {
                input_slot(ui, "Input A", *a, id, 0, &candidates, wire_action);
                input_slot(ui, "Input B", *b, id, 1, &candidates, wire_action);
                ui.horizontal(|ui| {
                    ui.label("Operation");
                    egui::ComboBox::from_id_salt(("combine_op_sb", id))
                        .selected_text(op.label())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(op, CombineOp::Union, "Union");
                            ui.selectable_value(
                                op,
                                CombineOp::Difference,
                                "Difference",
                            );
                            ui.selectable_value(
                                op,
                                CombineOp::Intersect,
                                "Intersect",
                            );
                        });
                });
            }
            NodeKind::Output { input } => {
                input_slot(ui, "Input", *input, id, 0, &candidates, wire_action);
            }
        });
}

/// Sidebar editor for a `Filter` node's predicate. Top combo switches
/// the predicate variant (resetting params to that variant's defaults
/// on change); the rows below it edit the current variant's params.
/// Variant switches discard the previous variant's params on purpose —
/// keeping a "remembered y threshold" across switches would surprise
/// the user more than help them.
fn filter_predicate_ui(
    ui: &mut egui::Ui,
    predicate: &mut FilterPredicate,
    node_id: NodeId,
) {
    // Variant selector. We compare via `matches!` rather than tag enums
    // to avoid carrying a parallel discriminator type.
    let cur_label = match predicate {
        FilterPredicate::YAbove(_) => "Y above",
        FilterPredicate::YBelow(_) => "Y below",
        FilterPredicate::MatchesColor(_) => "Color match",
        FilterPredicate::InsideBox { .. } => "Inside box",
    };
    ui.horizontal(|ui| {
        ui.label("Predicate");
        egui::ComboBox::from_id_salt(("filter_pred_kind", node_id))
            .selected_text(cur_label)
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(
                        matches!(predicate, FilterPredicate::YAbove(_)),
                        "Y above",
                    )
                    .clicked()
                    && !matches!(predicate, FilterPredicate::YAbove(_))
                {
                    *predicate = FilterPredicate::YAbove(0);
                }
                if ui
                    .selectable_label(
                        matches!(predicate, FilterPredicate::YBelow(_)),
                        "Y below",
                    )
                    .clicked()
                    && !matches!(predicate, FilterPredicate::YBelow(_))
                {
                    *predicate = FilterPredicate::YBelow(0);
                }
                if ui
                    .selectable_label(
                        matches!(predicate, FilterPredicate::MatchesColor(_)),
                        "Color match",
                    )
                    .clicked()
                    && !matches!(predicate, FilterPredicate::MatchesColor(_))
                {
                    *predicate = FilterPredicate::MatchesColor([200, 200, 200, 255]);
                }
                if ui
                    .selectable_label(
                        matches!(predicate, FilterPredicate::InsideBox { .. }),
                        "Inside box",
                    )
                    .clicked()
                    && !matches!(predicate, FilterPredicate::InsideBox { .. })
                {
                    *predicate = FilterPredicate::InsideBox {
                        min: (-8, 0, -8),
                        max: (8, 16, 8),
                    };
                }
            });
    });

    // Variant params.
    match predicate {
        FilterPredicate::YAbove(t) | FilterPredicate::YBelow(t) => {
            ui.horizontal(|ui| {
                ui.label("Threshold y");
                ui.add(egui::DragValue::new(t));
            });
        }
        FilterPredicate::MatchesColor(rgba) => {
            ui.horizontal(|ui| {
                ui.label("Color");
                let mut rgb = [rgba[0], rgba[1], rgba[2]];
                color_button_u8(ui, &mut rgb);
                rgba[0] = rgb[0];
                rgba[1] = rgb[1];
                rgba[2] = rgb[2];
                // Editor-placed voxels always have alpha 255; pin the
                // predicate's alpha to 255 too so a colour picked here
                // matches what's actually in the world.
                rgba[3] = 255;
            });
            ui.label(
                egui::RichText::new(
                    "Matches voxels with this exact RGB (alpha pinned to 255).",
                )
                .small()
                .weak(),
            );
        }
        FilterPredicate::InsideBox { min, max } => {
            ui.horizontal(|ui| {
                ui.label("Min");
                ui.add(egui::DragValue::new(&mut min.0).prefix("x:"));
                ui.add(egui::DragValue::new(&mut min.1).prefix("y:"));
                ui.add(egui::DragValue::new(&mut min.2).prefix("z:"));
            });
            ui.horizontal(|ui| {
                ui.label("Max");
                ui.add(egui::DragValue::new(&mut max.0).prefix("x:"));
                ui.add(egui::DragValue::new(&mut max.1).prefix("y:"));
                ui.add(egui::DragValue::new(&mut max.2).prefix("z:"));
            });
        }
    }
}

/// ComboBox for picking one of the graph's existing nodes as a node's
/// input slot. Reports the pick through `wire_action` (rather than
/// mutating the slot) so the caller can route it through
/// `PipelineGraph::set_input`, which rejects and rolls back cycles.
/// `target` (the node itself) is skipped in the list; Output nodes are
/// pre-excluded from `candidates` upstream. "(none)" clears the slot.
fn input_slot(
    ui: &mut egui::Ui,
    label: &str,
    current: Option<NodeId>,
    target: NodeId,
    slot: usize,
    candidates: &[(NodeId, String)],
    wire_action: &mut Option<(NodeId, usize, Option<NodeId>)>,
) {
    ui.horizontal(|ui| {
        ui.label(label);
        let current_label = match current {
            Some(id) => candidates
                .iter()
                .find(|(c, _)| *c == id)
                .map(|(_, l)| l.as_str())
                .unwrap_or("(missing)"),
            None => "(none)",
        };
        egui::ComboBox::from_id_salt(("input_slot", label, target))
            .selected_text(current_label)
            .show_ui(ui, |ui| {
                // Report the pick through `wire_action` instead of writing
                // the slot directly, so the caller routes it through
                // `set_input` (cycle-checked). A direct write here could
                // persist a cyclic graph into prefs (#18).
                if ui.selectable_label(current.is_none(), "(none)").clicked() {
                    *wire_action = Some((target, slot, None));
                }
                for (cid, clabel) in candidates {
                    if *cid == target {
                        continue;
                    }
                    if ui.selectable_label(current == Some(*cid), clabel).clicked() {
                        *wire_action = Some((target, slot, Some(*cid)));
                    }
                }
            });
    });
}

fn color_button_u8(ui: &mut egui::Ui, color: &mut [u8; 3]) {
    let mut f = [
        color[0] as f32 / 255.0,
        color[1] as f32 / 255.0,
        color[2] as f32 / 255.0,
    ];
    if ui.color_edit_button_rgb(&mut f).changed() {
        color[0] = (f[0] * 255.0).round() as u8;
        color[1] = (f[1] * 255.0).round() as u8;
        color[2] = (f[2] * 255.0).round() as u8;
    }
}

#[cfg(test)]
mod graph_layout_tests {
    use super::*;

    /// The row must never ask for more width than it was handed.
    ///
    /// Any surplus is width egui adds to the window, which comes back
    /// as more available width on the next frame, which produces more
    /// surplus — the window creeps outward until it hits the screen
    /// edge. A shortfall is merely a gap; a surplus is a runaway.
    #[test]
    fn the_graph_split_consumes_exactly_the_width_it_is_given() {
        for available in [520.0, 640.0, 960.0, 1440.0, 2560.0_f32] {
            for spacing in [0.0, 4.0, 8.0, 16.0_f32] {
                let (canvas, sidebar) = graph_split_widths(available, spacing);
                let used = canvas + spacing + GRAPH_DIVIDER_W + spacing + sidebar;
                assert!(
                    (used - available).abs() < 0.001,
                    "{available} wide at {spacing} spacing: canvas {canvas} + \
                     divider + sidebar {sidebar} = {used}, a surplus of {}",
                    used - available
                );
            }
        }
    }

    #[test]
    fn both_panes_keep_a_usable_width_when_the_window_is_small() {
        let (canvas, sidebar) = graph_split_widths(520.0, 8.0);
        assert!(canvas >= GRAPH_CANVAS_MIN_W, "canvas collapsed to {canvas}");
        assert!(sidebar >= GRAPH_SIDEBAR_MIN_W, "sidebar collapsed to {sidebar}");
    }

    #[test]
    fn procgen_settings_parse_with_fields_missing() {
        // Read in the format it actually ships in. Without the
        // struct-level `#[serde(default)]`, adding one field here
        // without its own default makes every existing `prefs.ron`
        // fail to parse — and `Prefs::load` answers a parse failure by
        // discarding the user's whole workspace.
        let s: ProcgenSettings =
            ron::from_str("(preview_enabled: true)").expect("a partial struct is still settings");
        assert!(s.preview_enabled);
        assert!(!s.graph_preview_enabled, "the rest fall back to Default");
    }
}
