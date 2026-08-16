//! User interface components using egui.

pub mod hud;
mod icons;
pub mod keymap;
mod panels;

pub use hud::HudState;
pub use panels::{
    ConfirmPrompt, ExportChoice, ExportFormat, ExportKind, ExportReport, Surface, UiAction, UiState,
};

use crate::editor::{next_socket_name, Axis, Editor, Quarter, Socket, Tool};
use crate::mcp::bridge::{Approval, DEFAULT_PORT};
use crate::procgen::{
    CombineOp, FilterPredicate, LSystemTree, MaskMode, NodeId, NodeKind, PerlinTerrain,
    PipelineGraph, WfcGenerator, WfcTileset,
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

/// Procgen workspace state: just the graph preview toggle, since
/// generator parameters live on graph nodes. Struct-level
/// `#[serde(default)]`, or a new field breaks every `prefs.ron`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ProcgenSettings {
    /// Translucent overlay of the pipeline graph's output (the Graph
    /// panel's Preview checkbox).
    pub graph_preview_enabled: bool,
}

/// Main UI manager
pub struct Ui {
    pub state: UiState,
    pub viewport: ViewportSettings,
    pub procgen: ProcgenSettings,
    /// Currently-selected node in the visual graph editor. Drives
    /// the sidebar parameter editor. Cleared automatically when the
    /// node is removed.
    pub selected_node: Option<NodeId>,
    /// Active wire drag: the node whose output socket was pressed.
    /// While set, a live wire follows the cursor; on release a hit-test
    /// against input sockets snaps or discards it.
    pub dragging_wire: Option<NodeId>,
    /// Recent-files MRU mirrored from `prefs::Prefs::recent_files`.
    /// App syncs this whenever the prefs version changes (touch_recent
    /// + initial load).
    pub recent_files: Vec<std::path::PathBuf>,
    /// Mirror of `App::clipboard.is_some()` so the Inspector can
    /// gray out the Paste button without `App::clipboard` leaking
    /// across the UI layer boundary. App syncs it before each frame.
    pub has_clipboard: bool,
    /// Mirror of `Renderer::wireframe_supported`, synced each frame.
    /// Every wireframe control gates on it, or the checkbox ticks and
    /// the status bar announces a mode nothing on screen enters.
    pub wireframe_supported: bool,

    /// Voxelization resolution along the longest axis (32 / 64 / 128)
    /// for `File ▸ Import` of a GLB. Owned by the UI so the control's
    /// state lives next to its widget.
    pub import_resolution: u32,

    /// Whether `.vox` transfers convert between MagicaVoxel's Z-up and
    /// Voxelith's Y-up, default on. Owned by the UI and read by the
    /// import and export paths at transfer time.
    pub convert_vox_axes: bool,

    /// Mirror of the in-editor MCP bridge's state, synced each frame the
    /// same way `has_clipboard` is.
    pub agent: AgentView,
}

/// What the Agent panel and its approval strip draw from — the
/// display-ready summary `App` mirrors across each frame, since the
/// bridge itself belongs on the other side of the line.
#[derive(Debug, Clone, Default)]
pub struct AgentView {
    /// The URL to hand a client, while the bridge is listening.
    pub url: Option<String>,
    /// The same URL plus the bearer token a client has to send. The URL
    /// alone is a setup line that answers 401.
    pub client_command: Option<String>,
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
        graph: &mut PipelineGraph,
        sockets: &mut Vec<Socket>,
        hud: &HudState,
    ) {
        // Top menu bar
        self.show_menu_bar(ctx, editor, graph);

        // The open project changed on disk and the reload was refused —
        // shown directly under the menu bar until it's resolved.
        if self.state.disk_conflict.is_some() {
            self.show_disk_conflict_bar(ctx);
        }

        // An agent's batch awaits approval. Same placement and reasoning
        // as the strip above: a lasting state the user has to work
        // around rather than be trapped by.
        if self.agent.pending.is_some() {
            self.show_agent_review_bar(ctx);
        }

        // Left side panel with tools
        self.show_toolbar(ctx, editor);

        // Stats panel
        if self.state.panels.show_stats {
            self.show_stats_panel(ctx, stats, editor);
        }

        // Inspector (the active tool's context panel)
        if self.state.panels.show_inspector {
            self.show_inspector_panel(ctx, editor, sockets);
        }

        // Color palette panel
        if self.state.panels.show_palette {
            self.show_palette_panel(ctx, editor);
        }

        // Viewport settings panel
        if self.state.panels.show_viewport_settings {
            self.show_viewport_panel(ctx);
        }

        // Pipeline graph panel
        if self.state.panels.show_graph {
            self.show_graph_panel(ctx, graph);
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

        // The recovery prompt, rendered last so it sits on top. An egui
        // dialog, never an `rfd::MessageDialog` — that exits the process
        // on this setup whenever it is shown.
        if self.state.show_recovery_prompt {
            self.show_recovery_prompt(ctx);
        }

        // Export… dialog — a working window, so it renders under the
        // error / report / guard dialogs that may have to interrupt it.
        if self.state.show_export_dialog {
            self.show_export_dialog(ctx);
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

    /// Somebody else wrote the open project while there were unsaved
    /// edits here. A strip rather than a modal: the writer is typically
    /// an agent, and a dialog would reopen on every batch.
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

    /// An agent's batch is on screen as translucent geometry, awaiting a
    /// yes or no. A strip, not a modal — the question is about geometry
    /// the user has to orbit around before answering.
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
        // Deferred-action pattern: `.open(...)` borrows the panel flag
        // while the closure borrows other fields, so intents collect
        // into a local and dispatch once the borrow is released.
        let mut action: Option<UiAction> = None;

        // An empty field reads as "no port" and leaves Start dead with
        // nothing saying why. Filled here rather than in the default, so
        // the one opinion about it lives in this panel.
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
                            ui.colored_label(egui::Color32::from_rgb(80, 200, 120), "listening");
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
                        // The URL alone gets a 401, since the bridge
                        // requires a token even on loopback — so the
                        // panel offers the whole setup line.
                        if let Some(command) = &agent.client_command {
                            ui.horizontal(|ui| {
                                if ui
                                    .button("Copy setup command")
                                    .on_hover_text(
                                        "Includes the access token this run generated. It \
                                         changes every time the bridge restarts",
                                    )
                                    .clicked()
                                {
                                    ui.output_mut(|out| out.copied_text = command.clone());
                                }
                                ui.label(
                                    egui::RichText::new("token required — loopback isn't a login")
                                        .small()
                                        .weak(),
                                );
                            });
                        }
                        ui.label(format!(
                            "{} batch{} applied since it started",
                            agent.applied,
                            if agent.applied == 1 { "" } else { "es" }
                        ));
                    }
                    None => {
                        ui.label(
                            "Let an agent edit this project directly, instead of \
                                  passing a file back and forth.",
                        );
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

    /// The post-export summary dialog, mirroring `show_error_dialog`'s
    /// structure. Rows for counts the format doesn't carry are skipped
    /// rather than shown empty.
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
                            egui::RichText::new(note).color(egui::Color32::from_rgb(255, 200, 80)),
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

    /// The Export… dialog: format × surface, with the pairing that
    /// doesn't exist grayed out rather than hidden. The built
    /// `ExportKind` goes through the same funnel as everything else.
    fn show_export_dialog(&mut self, ctx: &Context) {
        // Copied out and written back after the closure: the buttons
        // need `self.state` while `choice` is being edited.
        let mut choice = self.state.export_choice;
        let mut export = false;
        let mut close = false;
        egui::Window::new("Export")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.label("Format");
                ui.radio_value(&mut choice.format, ExportFormat::Glb, "glTF Binary (.glb)")
                    .on_hover_text(
                        "The game-asset path: bakes per-vertex AO, carries \
                     emissive / metallic, tint zones and sockets.",
                    );
                ui.radio_value(
                    &mut choice.format,
                    ExportFormat::Obj,
                    "Wavefront OBJ (.obj)",
                );
                ui.radio_value(&mut choice.format, ExportFormat::Vox, "MagicaVoxel (.vox)")
                    .on_hover_text(
                        "Voxel data rather than a mesh — stays editable in MagicaVoxel.",
                    );

                ui.add_space(6.0);
                ui.label("Surface");
                let no_surface = ".vox stores voxels, so there is no smoothed variant to ask for.";
                ui.add_enabled_ui(choice.format != ExportFormat::Vox, |ui| {
                    ui.radio_value(&mut choice.surface, Surface::Blocky, "Blocky")
                        .on_hover_text("Greedy mesh — the voxels as they render.")
                        .on_disabled_hover_text(no_surface);
                    ui.radio_value(
                        &mut choice.surface,
                        Surface::SmoothLight,
                        "Smoothed — light",
                    )
                    .on_hover_text(
                        "Marching Cubes over raw voxel density: voxel \
                         surfaces with rounded edges. Preserves thin \
                         features (tree branches, sparse detail).",
                    )
                    .on_disabled_hover_text(no_surface);
                    ui.radio_value(
                        &mut choice.surface,
                        Surface::SmoothHeavy,
                        "Smoothed — heavy",
                    )
                    .on_hover_text(
                        "Marching Cubes after a 3×3×3 density blur: \
                         clay-like blobs. Best for terrain / large solid \
                         masses; thin features may dissolve.",
                    )
                    .on_disabled_hover_text(no_surface);
                });

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        close = true;
                    }
                    if ui.button("Export...").clicked() {
                        export = true;
                        close = true;
                    }
                });
            });
        self.state.export_choice = choice;
        if export {
            self.state.request(UiAction::Export(choice.kind()));
        }
        if close {
            self.state.show_export_dialog = false;
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

    fn show_menu_bar(&mut self, ctx: &Context, editor: &Editor, graph: &mut PipelineGraph) {
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
                                let resp =
                                    ui.button(label).on_hover_text(path.display().to_string());
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
                    if ui.button("Export...").clicked() {
                        self.state.show_export_dialog = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Exit").clicked() {
                        self.state.request(UiAction::Exit);
                    }
                });

                ui.menu_button("Edit", |ui| {
                    let undo_text = if editor.can_undo() {
                        "Undo  Ctrl+Z"
                    } else {
                        "Undo"
                    };
                    if ui
                        .add_enabled(editor.can_undo(), egui::Button::new(undo_text))
                        .clicked()
                    {
                        self.state.request(UiAction::Undo);
                        ui.close_menu();
                    }
                    let redo_text = if editor.can_redo() {
                        "Redo  Ctrl+Y"
                    } else {
                        "Redo"
                    };
                    if ui
                        .add_enabled(editor.can_redo(), egui::Button::new(redo_text))
                        .clicked()
                    {
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
                        self.state
                            .request(UiAction::PasteClipboard { at_cursor: false });
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
                        // takes the scene and the undo history with it.
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
                    // Each Rotate submenu hosts CW / CCW / 180°. The
                    // rotated AABB extends from the same `min`, so a
                    // 4×1×2 region becomes 2×1×4 spreading toward +Z.
                    ui.menu_button("Rotate around X", |ui| {
                        if ui.add_enabled(has_sel, egui::Button::new("90°")).clicked() {
                            self.state.request(UiAction::RotateSelection {
                                axis: Axis::X,
                                quarter: Quarter::Cw,
                            });
                            ui.close_menu();
                        }
                        if ui.add_enabled(has_sel, egui::Button::new("-90°")).clicked() {
                            self.state.request(UiAction::RotateSelection {
                                axis: Axis::X,
                                quarter: Quarter::Ccw,
                            });
                            ui.close_menu();
                        }
                        if ui.add_enabled(has_sel, egui::Button::new("180°")).clicked() {
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
                        if ui.add_enabled(has_sel, egui::Button::new("180°")).clicked() {
                            self.state.request(UiAction::RotateSelection {
                                axis: Axis::Y,
                                quarter: Quarter::Half,
                            });
                            ui.close_menu();
                        }
                    });
                    ui.menu_button("Rotate around Z", |ui| {
                        if ui.add_enabled(has_sel, egui::Button::new("90°")).clicked() {
                            self.state.request(UiAction::RotateSelection {
                                axis: Axis::Z,
                                quarter: Quarter::Cw,
                            });
                            ui.close_menu();
                        }
                        if ui.add_enabled(has_sel, egui::Button::new("-90°")).clicked() {
                            self.state.request(UiAction::RotateSelection {
                                axis: Axis::Z,
                                quarter: Quarter::Ccw,
                            });
                            ui.close_menu();
                        }
                        if ui.add_enabled(has_sel, egui::Button::new("180°")).clicked() {
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
                    ui.checkbox(&mut self.state.panels.show_inspector, "Inspector");
                    ui.checkbox(&mut self.state.panels.show_palette, "Color Palette");
                    ui.checkbox(
                        &mut self.state.panels.show_viewport_settings,
                        "Viewport Settings",
                    );
                    ui.checkbox(&mut self.state.panels.show_graph, "Pipeline Graph");
                    ui.checkbox(&mut self.state.panels.show_agent, "Agent Bridge");
                    ui.separator();
                    ui.checkbox(&mut self.viewport.show_grid, "Show Grid");
                    ui.checkbox(&mut self.viewport.show_axes, "Show Axes");
                    ui.add_enabled(
                        wireframe_supported,
                        egui::Checkbox::new(&mut self.viewport.wireframe_mode, "Wireframe Mode"),
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
                    // One-node presets: each drops a source node into the
                    // graph and opens the panel. A graph with no Output
                    // gets one wired up; an existing one is never rewired.
                    let presets: [GeneratorPreset; 3] = [
                        ("Perlin Terrain", || {
                            NodeKind::Terrain(PerlinTerrain::default())
                        }),
                        ("L-System Tree", || NodeKind::Tree(LSystemTree::default())),
                        ("WFC Tile Layout", || NodeKind::Wfc(WfcGenerator::default())),
                    ];
                    for (label, make) in presets {
                        if ui
                            .button(label)
                            .on_hover_text(
                                "Add this generator as a node in the \
                                 pipeline graph and open the Graph panel",
                            )
                            .clicked()
                        {
                            let src = graph.add(make());
                            let has_output = graph
                                .nodes
                                .iter()
                                .any(|n| matches!(n.kind, NodeKind::Output { .. }));
                            if !has_output {
                                let out = graph.add(NodeKind::Output { input: None });
                                if let Err(e) = graph.set_input(out, 0, Some(src)) {
                                    // Unreachable for a fresh Output's
                                    // one input slot, but a wiring
                                    // failure must not be silent.
                                    log::warn!("Preset node wiring failed: {}", e);
                                }
                            }
                            self.selected_node = Some(src);
                            self.state.panels.show_graph = true;
                            // The graph is document data and this edit
                            // happens outside the Graph panel's own
                            // change detector, so it says so itself.
                            self.state.request(UiAction::GraphEdited);
                            ui.close_menu();
                        }
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
                    // from `Tool` itself rather than eleven copies of
                    // the key map, and the icon is painted per state.
                    let tool_button =
                        |ui: &mut egui::Ui, tool: Tool, current: Tool, note: &str| -> bool {
                            let mut tooltip = tool.name().to_string();
                            if !tool.shortcut().is_empty() {
                                tooltip.push_str(&format!(" ({})", tool.shortcut()));
                            }
                            if !note.is_empty() {
                                tooltip.push('\n');
                                tooltip.push_str(note);
                            }
                            let selected = tool == current;
                            let (rect, response) = ui
                                .allocate_exact_size(egui::vec2(36.0, 36.0), egui::Sense::click());
                            if ui.is_rect_visible(rect) {
                                let visuals = ui.style().interact_selectable(&response, selected);
                                ui.painter().rect(
                                    rect,
                                    visuals.rounding,
                                    visuals.weak_bg_fill,
                                    visuals.bg_stroke,
                                );
                                icons::paint_tool_icon(
                                    ui.painter(),
                                    rect.shrink(9.0),
                                    tool,
                                    visuals.text_color(),
                                );
                            }
                            response.on_hover_text(tooltip).clicked()
                        };

                    // One loop over the descriptor table — the same
                    // rows the help window prints, so a new tool is one
                    // `ToolSpec` entry everywhere.
                    for spec in keymap::TOOL_SPECS {
                        if spec.separator_before {
                            ui.add_space(8.0);
                            ui.separator();
                            ui.add_space(8.0);
                        }
                        if tool_button(ui, spec.tool, editor.current_tool, spec.note) {
                            editor.select_tool(spec.tool);
                        }
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
                    let (rect, _) =
                        ui.allocate_exact_size(egui::vec2(32.0, 32.0), egui::Sense::hover());
                    ui.painter().rect_filled(rect, 4.0, color);
                    ui.painter().rect_stroke(
                        rect,
                        4.0,
                        egui::Stroke::new(1.0_f32, egui::Color32::WHITE),
                    );

                    ui.add_space(8.0);

                    // Brush size indicator
                    ui.label(format!("{}", editor.brush_size));
                });
            });
    }

    fn show_stats_panel(&mut self, ctx: &Context, stats: &RenderStats, editor: &Editor) {
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
                        ui.label(format!(
                            "{} / {}",
                            editor.history.undo_count(),
                            editor.history.redo_count()
                        ));
                        ui.end_row();
                    });

                ui.separator();

                ui.label(format!(
                    "Camera: ({:.1}, {:.1}, {:.1})",
                    stats.camera_pos.0, stats.camera_pos.1, stats.camera_pos.2
                ));
            });
    }

    /// The Inspector: only the controls the active tool consumes, with
    /// `editor/tools.rs` as the authority. Sections a tool doesn't use
    /// are absent rather than grayed — the claim is "this is what it has".
    fn show_inspector_panel(
        &mut self,
        ctx: &Context,
        editor: &mut Editor,
        sockets: &mut Vec<Socket>,
    ) {
        // The close flag rides a local: `.open()` would borrow the panel
        // set for the whole window while the closure needs `self.state`.
        // Written back once both borrows are released.
        let mut open = self.state.panels.show_inspector;
        let tool = editor.current_tool;
        // One fixed window title: per-tool titles would give every tool
        // its own egui window id, and with it its own remembered
        // position — the panel would jump around as tools change.
        egui::Window::new("Inspector")
            .default_pos([60.0, 200.0])
            .resizable(true)
            .collapsible(true)
            .vscroll(true)
            .open(&mut open)
            .show(ctx, |ui| {
                // Which tool this inspects, from the same descriptor
                // row the toolbar tooltip prints.
                ui.horizontal(|ui| {
                    ui.heading(tool.name());
                    if !tool.shortcut().is_empty() {
                        ui.label(egui::RichText::new(tool.shortcut()).weak());
                    }
                });
                let note = keymap::spec_of(tool).note;
                if !note.is_empty() {
                    ui.label(egui::RichText::new(note).small().weak());
                }
                ui.separator();

                match tool {
                    Tool::Place | Tool::Paint => {
                        Self::brush_size_section(ui, editor);
                        Self::symmetry_section(ui, editor);
                        Self::color_material_section(ui, editor);
                    }
                    Tool::Remove => {
                        Self::brush_size_section(ui, editor);
                        Self::symmetry_section(ui, editor);
                    }
                    Tool::Fill => {
                        Self::symmetry_section(ui, editor);
                        Self::color_material_section(ui, editor);
                    }
                    Tool::Line | Tool::Box | Tool::Sphere | Tool::Cylinder => {
                        Self::symmetry_section(ui, editor);
                        Self::color_material_section(ui, editor);
                    }
                    Tool::Eyedropper => {
                        // Read-only: what a pick would overwrite.
                        ui.heading("Brush");
                        ui.horizontal(|ui| {
                            let color = egui::Color32::from_rgb(
                                editor.brush_color.r,
                                editor.brush_color.g,
                                editor.brush_color.b,
                            );
                            let (rect, _) = ui
                                .allocate_exact_size(egui::vec2(24.0, 24.0), egui::Sense::hover());
                            ui.painter().rect_filled(rect, 4.0, color);
                            ui.label(format!(
                                "{}, {}, {}",
                                editor.brush_color.r, editor.brush_color.g, editor.brush_color.b
                            ));
                        });
                    }
                    Tool::Select => {
                        self.selection_section(ui, editor);
                    }
                    Tool::Socket => {
                        self.sockets_section(ui, sockets);
                    }
                }

                // Show hovered voxel info
                if let Some(hit) = &editor.hovered_voxel {
                    ui.separator();
                    ui.heading("Hovered");
                    ui.label(format!(
                        "Position: ({}, {}, {})",
                        hit.voxel_pos.0, hit.voxel_pos.1, hit.voxel_pos.2
                    ));
                    ui.label(format!(
                        "Face: ({}, {}, {})",
                        hit.normal.0, hit.normal.1, hit.normal.2
                    ));
                }
            });
        self.state.panels.show_inspector = open;
    }

    /// Select's Inspector section: the live readout and the clipboard /
    /// delete verbs, mirroring the Edit menu.
    fn selection_section(&mut self, ui: &mut egui::Ui, editor: &Editor) {
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
                self.state
                    .request(UiAction::PasteClipboard { at_cursor: false });
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
                // Through the action even though this panel holds
                // `&mut Editor`: the drag anchors and the move ghost
                // live on `App`. See `App::deselect`.
                self.state.request(UiAction::Deselect);
            }
        });
    }

    /// Socket's Inspector section: the per-socket list. Placement
    /// itself is a click in the viewport; this is where the names —
    /// the part glTF consumers key on — get edited.
    fn sockets_section(&mut self, ui: &mut egui::Ui, sockets: &mut Vec<Socket>) {
        if sockets.is_empty() {
            ui.label(egui::RichText::new("No sockets yet.").small().weak());
        } else {
            // Per-socket row: rename, delete, position. Every mutation
            // also raises `SocketsEdited` — this edits document data in
            // place, and no mesh rebuild would notice it.
            let mut to_delete: Option<usize> = None;
            let mut committed: Option<usize> = None;
            let mut edited = false;
            egui::ScrollArea::vertical()
                .max_height(120.0)
                .show(ui, |ui| {
                    for (i, s) in sockets.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            let field = ui
                                .add(egui::TextEdit::singleline(&mut s.name).desired_width(110.0))
                                .on_hover_text(
                                    "Name (becomes the glTF node name). Unique per \
                                     scene: leaving the field gives a duplicate a \
                                     numeric suffix and a blank name the default.",
                                );
                            if field.changed() {
                                edited = true;
                            }
                            if field.lost_focus() {
                                committed = Some(i);
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
            // A rename resolves when the field is committed, never per
            // keystroke — typing "muzzle_left" passes through "muzzle".
            // The Inspector must stay drawn after the toolbar for this.
            if let Some(i) = committed {
                let resolved = resolve_socket_name(sockets, i);
                if sockets[i].name != resolved {
                    sockets[i].name = resolved;
                    edited = true;
                }
            }
            if let Some(i) = to_delete {
                sockets.remove(i);
                edited = true;
            }
            if ui
                .button("Clear all sockets")
                .on_hover_text("Remove every socket from the scene")
                .clicked()
            {
                sockets.clear();
                edited = true;
            }
            if edited {
                self.state.request(UiAction::SocketsEdited);
            }
        }
    }

    /// Brush radius, for the three tools that stroke cell-by-cell —
    /// the only consumers of `brush_size` (`BrushTool::apply` returns
    /// early for everything else).
    fn brush_size_section(ui: &mut egui::Ui, editor: &mut Editor) {
        ui.heading("Brush Size");
        let mut size = editor.brush_size as u32;
        ui.add(egui::Slider::new(&mut size, 1..=10).show_value(true));
        editor.brush_size = size as u8;
        ui.separator();
    }

    /// Mirror planes through the world origin. Every voxel-writing
    /// tool consumes this (the brush sphere, the fill seed, the shape
    /// sweep); Eyedropper doesn't, so its Inspector never shows it.
    fn symmetry_section(ui: &mut egui::Ui, editor: &mut Editor) {
        ui.heading("Symmetry");
        ui.horizontal(|ui| {
            ui.checkbox(&mut editor.symmetry.x, "X")
                .on_hover_text("Mirror brush across the x = 0 plane");
            ui.checkbox(&mut editor.symmetry.y, "Y")
                .on_hover_text("Mirror brush across the y = 0 plane");
            ui.checkbox(&mut editor.symmetry.z, "Z")
                .on_hover_text("Mirror brush across the z = 0 plane");
        });
        ui.separator();
    }

    /// The brush voxel template: color, and the material flags that
    /// ride with it into every placed voxel.
    fn color_material_section(ui: &mut egui::Ui, editor: &mut Editor) {
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
            ui.label(format!(
                "{}, {}, {}",
                editor.brush_color.r, editor.brush_color.g, editor.brush_color.b
            ));
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
    }

    fn show_palette_panel(&mut self, ctx: &Context, editor: &mut Editor) {
        // Collected inside the window closure, applied after it — the
        // closure holds the borrow that `set_status` needs.
        let mut palette_feedback: Option<String> = None;
        egui::Window::new("Palette")
            // Right column: the left one holds the toolbar and
            // Inspector. Float positions aren't persisted, so this
            // constant is what every session gets.
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
                            let (rect, response) = ui
                                .allocate_exact_size(egui::vec2(size, size), egui::Sense::click());

                            if response.clicked() {
                                picked = Some(i);
                            }

                            ui.painter().rect_filled(rect, 2.0, color);
                            if is_selected {
                                ui.painter().rect_stroke(
                                    rect,
                                    2.0,
                                    egui::Stroke::new(2.0_f32, egui::Color32::WHITE),
                                );
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
                        let exists = editor
                            .palette
                            .iter()
                            .any(|v| v.r == color.r && v.g == color.g && v.b == color.b);
                        // Report both refusals: in silence there is no
                        // way to tell "already there" from "list is
                        // full" from "the click missed".
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
        // Local close flag — see `show_inspector_panel` for why.
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
                    egui::Checkbox::new(&mut self.viewport.wireframe_mode, "Wireframe Mode"),
                )
                .on_disabled_hover_text(WIREFRAME_UNSUPPORTED);
                ui.checkbox(&mut self.viewport.show_hud, "Viewport HUD")
                    .on_hover_text(
                        "Tool & gesture readout in the bottom-left corner of the viewport",
                    );
                ui.checkbox(&mut self.viewport.show_perf_hud, "Performance HUD")
                    .on_hover_text("FPS, triangles, and re-mesh time in the bottom-right corner");

                ui.separator();

                ui.heading("Grid");
                ui.add(egui::Slider::new(&mut self.viewport.grid_size, 5..=50).text("Size"));
                ui.add(
                    egui::Slider::new(&mut self.viewport.grid_spacing, 0.5..=5.0).text("Spacing"),
                );

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
                        self.state
                            .request(UiAction::SetCameraView(CameraView::Front));
                    }
                    if ui.button("Side").clicked() {
                        self.state
                            .request(UiAction::SetCameraView(CameraView::Side));
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

    fn show_graph_panel(&mut self, ctx: &Context, graph: &mut PipelineGraph) {
        // The graph as this frame found it. Comparing start to end of
        // one call is what makes the signal trustworthy: everything that
        // writes the graph from outside happens between frames.
        let before = graph.clone();

        // Deferred actions: collected during the immediate-mode pass,
        // applied after the window closure releases its borrows on
        // `self.graph` and `self.state`.
        let mut run = false;
        let mut delete_id: Option<NodeId> = None;
        let mut add_kind: Option<NodeKind> = None;
        let mut auto_layout = false;
        let mut wire_action: Option<(NodeId, usize, Option<NodeId>)> = None;
        let mut wire_error: Option<String> = None;

        let graph = &mut *graph;
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
                            let enabled = !(has_output && matches!(kind, NodeKind::Output { .. }));
                            if ui.add_enabled(enabled, egui::Button::new(k.0)).clicked() {
                                add_kind = Some(kind);
                                ui.close_menu();
                            }
                            if k.2 {
                                ui.separator();
                            }
                        }
                    });
                    if ui
                        .button("Auto Layout")
                        .on_hover_text("Re-grid all nodes")
                        .clicked()
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
        if *graph != before {
            self.state.request(UiAction::GraphEdited);
        }
    }

    fn show_help_panel(&mut self, ctx: &Context) {
        // The list runs past what a 1080p screen holds, and with no
        // scrolling its tail — the save and open shortcuts — is
        // unreachable. Cap against the screen and scroll the overflow.
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
                        // One line instead of rewriting thirty entries
                        // per platform: the chords bind to the
                        // platform's command key, so the table is one copy.
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

                                // From the descriptor table — the same rows
                                // the toolbar renders, so this list can't
                                // promise a tool the toolbar doesn't have.
                                for spec in keymap::TOOL_SPECS {
                                    let shortcut = spec.tool.shortcut();
                                    if shortcut.is_empty() {
                                        ui.label("(toolbar only)");
                                    } else {
                                        ui.label(shortcut);
                                    }
                                    ui.label(spec.tool.name());
                                    ui.end_row();
                                }

                                ui.label("Alt (hold)");
                                ui.label("Temporary eyedropper — restores on release");
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

                                chord_rows(ui, keymap::HelpSection::Edit);

                                ui.end_row();
                                ui.heading("Selection");
                                ui.end_row();

                                ui.label("Drag in selection");
                                ui.label("Move (single SetVoxels Command)");
                                ui.end_row();

                                ui.label("Drag outside");
                                ui.label("Create new selection");
                                ui.end_row();

                                chord_rows(ui, keymap::HelpSection::Selection);

                                ui.label("Del");
                                ui.label("Delete non-air voxels in selection");
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

                                chord_rows(ui, keymap::HelpSection::File);

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
            .default_pos([
                ctx.screen_rect().width() / 2.0 - 150.0,
                ctx.screen_rect().height() / 2.0 - 100.0,
            ])
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
                    // Read off the manifest rather than typed here, so
                    // this is the same string `Cargo.toml` publishes and
                    // a missing license field fails the build.
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
                // The tool name is highlighted: which tool is active is
                // easy to miss in a flat style, and Fill and Eyedropper
                // behave very differently from the brush tools.
                ui.label(
                    egui::RichText::new(format!("Tool: {}", editor.current_tool.name()))
                        .strong()
                        .color(egui::Color32::LIGHT_BLUE),
                );
                ui.separator();
                ui.label(format!("Brush: {}px", editor.brush_size));
                if editor.symmetry.any() {
                    ui.separator();
                    let mut axes = String::new();
                    if editor.symmetry.x {
                        axes.push('X');
                    }
                    if editor.symmetry.y {
                        axes.push('Y');
                    }
                    if editor.symmetry.z {
                        axes.push('Z');
                    }
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
                    // Only claim wireframe when it is actually running:
                    // the flag can be set from prefs written on other
                    // hardware while this GPU draws solid.
                    if self.viewport.wireframe_mode && self.wireframe_supported {
                        ui.label("[Wireframe]");
                    }
                    if self.viewport.show_grid {
                        ui.label("[Grid]");
                    }
                    if self.viewport.show_axes {
                        ui.label("[Axes]");
                    }
                    if self.procgen.graph_preview_enabled {
                        ui.label(
                            egui::RichText::new("● Preview").color(egui::Color32::LIGHT_GREEN),
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

// ---- Generator parameter editors -------------------------------------
// Free functions over the parameter struct alone, so the graph sidebar
// can hand in a node's embedded generator directly.

fn terrain_params_ui(ui: &mut egui::Ui, t: &mut PerlinTerrain) {
    ui.heading("Perlin Terrain");
    ui.add_space(4.0);

    egui::Grid::new("terrain_params")
        .num_columns(2)
        .spacing([10.0, 4.0])
        .show(ui, |ui| {
            ui.label("Seed");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut t.seed).speed(1.0));
                if ui.button("Rand").on_hover_text("Randomize seed").clicked() {
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

            // Two sliders, but not two independent values: the
            // generator rejects min > max, and under Preview that
            // rejection only vanishes the overlay. Let each push the other.
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
            ui.add(egui::Slider::new(&mut t.frequency, 0.005..=0.5).logarithmic(true));
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
    ui.heading("L-System Tree");
    ui.add_space(4.0);

    egui::Grid::new("tree_params")
        .num_columns(2)
        .spacing([10.0, 4.0])
        .show(ui, |ui| {
            ui.label("Seed");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut t.seed).speed(1.0));
                if ui.button("Rand").on_hover_text("Randomize seed").clicked() {
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
    ui.heading("WFC Tile Layout");
    ui.add_space(4.0);

    egui::Grid::new("wfc_params")
        .num_columns(2)
        .spacing([10.0, 4.0])
        .show(ui, |ui| {
            ui.label("Seed");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut t.seed).speed(1.0));
                if ui.button("Rand").on_hover_text("Randomize seed").clicked() {
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

/// Split the graph window's inner width into canvas and sidebar. The
/// parts must sum to **exactly** the width handed in, pinned by a test:
/// over-reserving grows the window a few pixels every frame.
fn graph_split_widths(available: f32, item_spacing: f32) -> (f32, f32) {
    let overhead = GRAPH_DIVIDER_W + item_spacing * 2.0;
    // Shrink the sidebar before the canvas, and stop both at their
    // floors — past that egui clips the row rather than growing the
    // window without end.
    let sidebar = GRAPH_SIDEBAR_W
        .min(available * 0.4)
        .max(GRAPH_SIDEBAR_MIN_W)
        .min((available - overhead - GRAPH_CANVAS_MIN_W).max(GRAPH_SIDEBAR_MIN_W));
    let canvas = (available - sidebar - overhead).max(GRAPH_CANVAS_MIN_W);
    (canvas, sidebar)
}

/// The name socket `idx` keeps once its rename is committed: whitespace
/// trimmed, a blank falling back to the default sequence, and a taken
/// one growing the smallest free `_N` suffix.
fn resolve_socket_name(sockets: &[Socket], idx: usize) -> String {
    let typed = sockets[idx].name.trim();
    if typed.is_empty() {
        // Passing the whole slice is right: a blank name can never be
        // one of the `Socket_N` candidates, so this socket doesn't
        // block its own default.
        return next_socket_name(sockets);
    }
    let taken = |candidate: &str| {
        sockets
            .iter()
            .enumerate()
            .any(|(j, s)| j != idx && s.name == candidate)
    };
    if !taken(typed) {
        return typed.to_string();
    }
    let mut n = 2usize;
    loop {
        let candidate = format!("{typed}_{n}");
        if !taken(&candidate) {
            return candidate;
        }
        n += 1;
    }
}

/// One "+ Add Node" menu entry: `(label, factory, separator_after)`.
type NodeMenuOption = (&'static str, fn() -> NodeKind, bool);

/// One Generate-menu preset entry: `(label, factory)`.
type GeneratorPreset = (&'static str, fn() -> NodeKind);

/// Available node kinds in the "+ Add Node" menu.
fn node_menu_options() -> Vec<NodeMenuOption> {
    vec![
        (
            "Source: Terrain",
            || NodeKind::Terrain(PerlinTerrain::default()),
            false,
        ),
        (
            "Source: Tree",
            || NodeKind::Tree(LSystemTree::default()),
            false,
        ),
        (
            "Source: WFC",
            || NodeKind::Wfc(WfcGenerator::default()),
            true,
        ),
        (
            "Translate",
            || NodeKind::Translate {
                input: None,
                dx: 0,
                dy: 0,
                dz: 0,
            },
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
        // Both two-input kinds stack their sockets vertically so the
        // slots land at distinct positions — overlapping them puts slot
        // 1 out of reach of a hit-test that stops at the first match.
        NodeKind::Combine { .. } | NodeKind::Mask { .. } => {
            let body_inner_top = body.min.y + NODE_HEADER_H + 14.0;
            let y = body_inner_top + slot as f32 * 22.0;
            egui::pos2(body.min.x, y)
        }
        _ => egui::pos2(body.min.x, body.center().y + 6.0),
    }
}

/// Center of a node's output socket (right edge).
fn output_socket_screen(canvas_min: egui::Pos2, node: &crate::procgen::GraphNode) -> egui::Pos2 {
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

/// Draw a wire between two sockets as a horizontally-bowed cubic
/// Bezier, tessellated to a polyline rather than depending on egui's
/// `CubicBezierShape` API across versions.
fn paint_wire(painter: &egui::Painter, from: egui::Pos2, to: egui::Pos2, color: egui::Color32) {
    let dx = (to.x - from.x).abs().max(40.0);
    let c1 = egui::pos2(from.x + dx * 0.5, from.y);
    let c2 = egui::pos2(to.x - dx * 0.5, to.y);

    const SEGMENTS: usize = 24;
    let mut pts = Vec::with_capacity(SEGMENTS + 1);
    for i in 0..=SEGMENTS {
        let t = i as f32 / SEGMENTS as f32;
        pts.push(cubic_bezier_point(from, c1, c2, to, t));
    }
    painter.add(egui::Shape::line(pts, egui::Stroke::new(2.0_f32, color)));
}

/// The graph canvas: nodes, wires, click-select, body drag and
/// socket-drag wire creation. Graph mutations defer through the
/// out-params so the caller applies them outside the borrow.
fn chord_rows(ui: &mut egui::Ui, section: keymap::HelpSection) {
    for chord in keymap::CHORDS.iter().filter(|c| c.section == section) {
        ui.label(chord.chord_label);
        ui.label(chord.help);
        ui.end_row();
    }
}

fn graph_canvas(
    ui: &mut egui::Ui,
    graph: &mut PipelineGraph,
    selected: &mut Option<NodeId>,
    drag_wire: &mut Option<NodeId>,
    delete_id: &mut Option<NodeId>,
    wire_action: &mut Option<(NodeId, usize, Option<NodeId>)>,
) {
    let avail = ui.available_size();
    let (canvas_rect, _bg) = ui.allocate_exact_size(avail, egui::Sense::hover());
    let painter = ui.painter_at(canvas_rect);

    // Background.
    painter.rect_filled(canvas_rect, 0.0, egui::Color32::from_rgb(28, 28, 36));

    // ===== Wires (drawn before nodes so they pass under boxes) =====
    for node in &graph.nodes {
        let in_count = PipelineGraph::input_count(&node.kind);
        for slot in 0..in_count {
            // The canonical accessor, so every node kind's slots are
            // covered — an inline match omitted three of them and those
            // wires were simply never drawn.
            let Some(src_id) = graph.get_input(node.id, slot).ok().flatten() else {
                continue;
            };
            let Some(src) = graph.get(src_id) else {
                continue;
            };
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
            let to = ui.ctx().input(|i| i.pointer.interact_pos()).unwrap_or(from);
            paint_wire(&painter, from, to, egui::Color32::YELLOW);
        }
    }

    // Two passes: allocate every node body first so its drag response
    // is registered, then draw and handle sockets. Splitting keeps
    // z-order predictable, with sockets on top of bodies.
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

    // Apply body drags and clicks. A dragged node is clamped inside the
    // canvas — `ui.interact` isn't clipped, so one dragged past the edge
    // stays clickable but invisible. Only drags clamp, never a re-flow.
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
                node.position[0] = (node.position[0] + frame.delta.x).clamp(0.0, drag_limit.x);
                node.position[1] = (node.position[1] + frame.delta.y).clamp(0.0, drag_limit.y);
            }
        }
    }

    // Visual and socket pass, reading `&graph.nodes` directly: the drag
    // loop is done with its `get_mut` and its positions are visible
    // here, so no node's parameters need copying per frame.
    for node in &graph.nodes {
        let body = node_screen_rect(canvas_rect.min, node);
        let is_selected = *selected == Some(node.id);

        // Body fill + outline.
        painter.rect_filled(body, 4.0, egui::Color32::from_rgb(50, 50, 60));
        let outline = if is_selected {
            egui::Stroke::new(2.0_f32, egui::Color32::LIGHT_BLUE)
        } else {
            egui::Stroke::new(1.0_f32, egui::Color32::from_gray(80))
        };
        painter.rect_stroke(body, 4.0, outline);

        // Header.
        let header =
            egui::Rect::from_min_max(body.min, egui::pos2(body.max.x, body.min.y + NODE_HEADER_H));
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
        let close_resp = ui.interact(close_rect, close_id, egui::Sense::click());
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
            let in_id = ui.id().with(("graph_in_sock", node.id, slot));
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
                egui::Stroke::new(1.0_f32, egui::Color32::BLACK),
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
            let out_resp = ui.interact(hit_rect, out_id, egui::Sense::drag());
            painter.circle_filled(center, SOCKET_R, egui::Color32::from_rgb(220, 200, 100));
            painter.circle_stroke(
                center,
                SOCKET_R,
                egui::Stroke::new(1.0_f32, egui::Color32::BLACK),
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
                            let s = input_socket_screen(canvas_rect.min, target, slot);
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

    // Node ids snapshotted for the input combos, so no borrow is held
    // while one node is mutated below. Output nodes are sinks, so
    // excluding them keeps the dropdown from offering nothing.
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

    ui.label(egui::RichText::new(format!("#{}  {}", node.id, node.kind.label())).strong());
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
                    // Bounded so the panel can't scatter geometry
                    // thousands of cells apart. UX guidance, not a
                    // safety limit — the real ceiling is downstream.
                    ui.add(egui::DragValue::new(dx).prefix("x:").range(-1024..=1024));
                    ui.add(egui::DragValue::new(dy).prefix("y:").range(-1024..=1024));
                    ui.add(egui::DragValue::new(dz).prefix("z:").range(-1024..=1024));
                });
            }
            NodeKind::Filter { input, predicate } => {
                input_slot(ui, "Input", *input, id, 0, &candidates, wire_action);
                filter_predicate_ui(ui, predicate, id);
            }
            NodeKind::Mask {
                subject,
                mask,
                mode,
            } => {
                input_slot(ui, "Subject", *subject, id, 0, &candidates, wire_action);
                input_slot(ui, "Mask", *mask, id, 1, &candidates, wire_action);
                ui.horizontal(|ui| {
                    ui.label("Mode");
                    egui::ComboBox::from_id_salt(("mask_mode_sb", id))
                        .selected_text(mode.label())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(mode, MaskMode::AboveColumn, "Above column");
                            ui.selectable_value(mode, MaskMode::BelowColumn, "Below column");
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
                            ui.selectable_value(op, CombineOp::Difference, "Difference");
                            ui.selectable_value(op, CombineOp::Intersect, "Intersect");
                        });
                });
            }
            NodeKind::Output { input } => {
                input_slot(ui, "Input", *input, id, 0, &candidates, wire_action);
            }
        });
}

/// Sidebar editor for a `Filter` node's predicate: the combo switches
/// variant and the rows edit its params. A switch discards the previous
/// variant's params rather than remembering them.
fn filter_predicate_ui(ui: &mut egui::Ui, predicate: &mut FilterPredicate, node_id: NodeId) {
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
                    .selectable_label(matches!(predicate, FilterPredicate::YAbove(_)), "Y above")
                    .clicked()
                    && !matches!(predicate, FilterPredicate::YAbove(_))
                {
                    *predicate = FilterPredicate::YAbove(0);
                }
                if ui
                    .selectable_label(matches!(predicate, FilterPredicate::YBelow(_)), "Y below")
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
                egui::RichText::new("Matches voxels with this exact RGB (alpha pinned to 255).")
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

/// ComboBox for wiring an input slot. The pick is reported through
/// `wire_action` rather than written, so the caller routes it through
/// `set_input`, which rejects cycles. "(none)" clears the slot.
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
                // Report through `wire_action` rather than writing the
                // slot, so the caller routes it through the cycle check.
                // A direct write could persist a cyclic graph.
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

    /// The row must never ask for more width than it was handed: any
    /// surplus is width egui adds to the window, which comes back as
    /// more surplus. A shortfall is a gap; a surplus is a runaway.
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
        assert!(
            sidebar >= GRAPH_SIDEBAR_MIN_W,
            "sidebar collapsed to {sidebar}"
        );
    }

    #[test]
    fn procgen_settings_parse_with_fields_missing() {
        // Read in the format it ships in. Without the struct-level
        // `#[serde(default)]`, one new field breaks every `prefs.ron` —
        // and a parse failure discards the user's whole workspace.
        let s: ProcgenSettings = ron::from_str("()").expect("a partial struct is still settings");
        assert!(
            !s.graph_preview_enabled,
            "missing fields fall back to Default"
        );
    }

    #[test]
    fn procgen_settings_ignore_the_retired_single_generator_fields() {
        // An older `prefs.ron` carries the retired generator panel's
        // state. All of it must be skipped rather than refused, or the
        // user loses their whole workspace over keys nothing reads.
        let old = "(
            selected: Terrain,
            terrain: (width: 64, depth: 64, seed: 42),
            preview_enabled: true,
            graph_preview_enabled: true,
        )";
        let s: ProcgenSettings =
            ron::from_str(old).expect("retired fields must be ignored, not fatal");
        assert!(s.graph_preview_enabled, "the surviving field still reads");
    }
}

#[cfg(test)]
mod socket_name_tests {
    use super::*;

    fn socket(name: &str) -> Socket {
        Socket::new(name, [0.0; 3], [0.0, 1.0, 0.0])
    }

    #[test]
    fn a_typed_name_someone_else_holds_grows_a_suffix() {
        // Socket 1 has just been renamed onto socket 0's name.
        let sockets = [socket("muzzle"), socket("muzzle")];
        assert_eq!(resolve_socket_name(&sockets, 1), "muzzle_2");

        // The suffix climbs past names that are also taken.
        let sockets = [socket("muzzle"), socket("muzzle_2"), socket("muzzle")];
        assert_eq!(resolve_socket_name(&sockets, 2), "muzzle_3");

        // A name only this socket holds survives exactly as typed —
        // committing a field must not rewrite a legal name.
        let sockets = [socket("muzzle"), socket("barrel")];
        assert_eq!(resolve_socket_name(&sockets, 1), "barrel");

        // A socket doesn't collide with itself: re-committing an
        // untouched field is a no-op, not a suffix per focus loss.
        let sockets = [socket("muzzle")];
        assert_eq!(resolve_socket_name(&sockets, 0), "muzzle");
    }

    #[test]
    fn a_blank_name_falls_back_to_the_default_sequence() {
        let sockets = [socket("Socket_1"), socket("")];
        assert_eq!(resolve_socket_name(&sockets, 1), "Socket_2");

        // Whitespace-only reads as blank...
        let sockets = [socket("   ")];
        assert_eq!(resolve_socket_name(&sockets, 0), "Socket_1");

        // ...and surrounding whitespace is dropped rather than
        // exported into a node name nobody can see the shape of.
        let sockets = [socket(" muzzle ")];
        assert_eq!(resolve_socket_name(&sockets, 0), "muzzle");
    }
}
