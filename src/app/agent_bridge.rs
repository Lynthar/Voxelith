//! The editor's half of the agent bridge: serve an MCP client from the
//! world the user is looking at.
//!
//! Shape is `tick_preview`'s — a background task produces messages, the
//! frame loop drains them and does the mutating — with one addition: an
//! agent's tool call is waiting for an answer, so every message carries
//! a line back. The world, the selection and the undo stack never leave
//! this thread; `voxelith::mcp::bridge` owns the wire and knows nothing
//! about chunks.
//!
//! What this buys over the checkpoint-and-reload path it grew out of is
//! one undo stack. An agent's batch goes through the same
//! `CommandHistory` as a brush stroke, so Ctrl+Z walks back through
//! both, and a human can start drawing mid-build without stopping
//! anything or losing a race to save.
//!
//! Two modes, and the difference is only *when* a batch lands.
//! [`Approval::Auto`] commits it as it arrives — the human watches it
//! appear and undoes it if it was wrong, which is the faster loop for a
//! run of batches. [`Approval::Review`] puts it up as a translucent
//! preview and parks the agent's call until they answer.

use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::time::{Duration, Instant};

use voxelith::agent_ops::{self, BatchOutcome, DocumentView, OpsError};
use voxelith::core::{Voxel, World};
use voxelith::editor::{Command, VoxelChange};
use voxelith::mcp::bridge::{
    self, AgentReply, AgentRequest, Answer, Applied, Approval, BridgeCall, BridgeReceiver,
    BridgeStatus, Described, Refusal, Reviewed, RunningBridge, Sliced, Stepped, Views,
};
use voxelith::mesh::patch_to_mesh;
use voxelith::ui::AgentView;
use voxelith::view;

use super::App;

/// Alpha for the approval preview. The same 0.5 the procgen overlay
/// uses, and for the same reason: it means "not in the world yet".
const REVIEW_ALPHA: f32 = 0.5;

/// Cells the batch would clear, painted so they can be seen going.
///
/// `patch_to_mesh` skips air, so a batch that only deletes would preview
/// as nothing at all — the one preview a human can't review. Red is not
/// used anywhere else in the viewport, and a cell about to be erased has
/// no color of its own worth showing.
const CLEARED_TINT: Voxel = Voxel {
    material: 1,
    r: 220,
    g: 60,
    b: 60,
    a: 255,
    flags: 0,
    _reserved: 0,
};

/// How long an unanswered batch stays parked.
///
/// The agent's own client normally gives up long before this, and
/// [`BridgeCall::abandoned`] catches that exactly; this is the backstop
/// for the patient client whose human walked away, so the preview
/// doesn't sit on screen for the rest of the session.
const REVIEW_TIMEOUT: Duration = Duration::from_secs(300);

/// A batch waiting on a human.
pub(super) struct PendingReview {
    call: BridgeCall,
    outcome: BatchOutcome,
    /// The edit generation when this was parked. Any editing the human
    /// does meanwhile moves it, and that invalidates the batch: its
    /// `old_voxel`s are what the world held *then*, so committing it
    /// later would make undo restore a state that never existed.
    history_mark: u64,
    parked_at: Instant,
}

impl PendingReview {
    /// What the strip tells the human this batch would do.
    pub(super) fn summary(&self) -> String {
        summarize(&self.outcome.changes)
    }
}

/// Writes and clears, counted apart.
///
/// One number would do neither job: "changes 40 voxels" reads as "adds
/// 40" to the person deciding, and a batch that only deletes is exactly
/// the one they most need to understand before saying yes — especially
/// since the preview can only ever stand in for what is leaving.
fn summarize(changes: &[VoxelChange]) -> String {
    let cleared = changes
        .iter()
        .filter(|change| change.new_voxel.is_air())
        .count();
    let written = changes.len() - cleared;
    match (written, cleared) {
        (0, 0) => "changes nothing".to_string(),
        (1, 0) => "writes 1 voxel".to_string(),
        (written, 0) => format!("writes {written} voxels"),
        (0, 1) => "clears 1 voxel".to_string(),
        (0, cleared) => format!("clears {cleared} voxels"),
        (written, cleared) => format!("writes {written}, clears {cleared}"),
    }
}

/// Everything the editor keeps for the bridge, off `App` for the same
/// reason `PreviewState` is.
pub(super) struct AgentBridgeState {
    /// Calls arriving from the server. `None` when the bridge is off.
    pub receiver: Option<BridgeReceiver>,
    /// The listening server. Dropping it stops the task and closes the
    /// socket.
    pub server: Option<RunningBridge>,
    pub approval: Approval,
    pub pending: Option<PendingReview>,
    /// Batches committed since the bridge came up. The panel's evidence
    /// that something is actually happening on the other end.
    pub applied: usize,
}

impl AgentBridgeState {
    pub fn new() -> Self {
        Self {
            receiver: None,
            server: None,
            approval: Approval::default(),
            pending: None,
            applied: 0,
        }
    }

    pub fn is_running(&self) -> bool {
        self.server.is_some()
    }
}

impl App {
    /// Bind the port and start serving. No-op if the bridge is already
    /// up.
    ///
    /// Port 0 asks the OS for a free one, and the panel shows whatever
    /// it handed out.
    pub(super) fn start_agent_bridge(&mut self, port: u16) {
        if self.agent.is_running() {
            return;
        }
        let address = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
        // Bind here rather than inside the task: "that port is taken" is
        // an answer to the button the user just pressed, and it belongs
        // in front of them instead of in a log line from a background
        // task that quietly never started.
        let listener = match TcpListener::bind(address) {
            Ok(listener) => listener,
            Err(e) => {
                log::warn!("agent bridge could not bind {address}: {e}");
                self.ui
                    .set_status(format!("Agent bridge: can't listen on {address} — {e}"));
                return;
            }
        };
        let address = listener.local_addr().unwrap_or(address);

        // Minted per start, so stopping and restarting the bridge
        // invalidates whatever an old client kept. The human copies the
        // line the panel shows; nothing writes the token to disk, which
        // is also why the panel is the only place it exists.
        let token = voxelith::mcp::AccessToken::generate();
        let client_command =
            voxelith::mcp::guard::client_command(&format!("http://{address}/mcp"), &token);

        let (handle, receiver) = bridge::channel();
        let task = self.async_runtime.handle().spawn(async move {
            if let Err(e) = bridge::serve_http_bridged(handle, listener, token).await {
                // The socket is already bound by the time we get here, so
                // this is the transport itself failing rather than a port
                // clash. Nothing to recover: the receiver going quiet is
                // what the panel shows.
                log::error!("agent bridge stopped serving: {e}");
            }
        });

        let running = RunningBridge::new(address, client_command, task);
        // Also to the log, not only the panel: `--agent-port` is a
        // command-line affordance, and someone who started the editor
        // from a terminal expects the terminal to tell them how to
        // connect rather than to go hunting for a window.
        log::info!("Agent bridge client setup: {}", running.client_command());
        self.ui
            .set_status(format!("Agent bridge listening on {}", running.url()));
        self.agent.receiver = Some(receiver);
        self.agent.server = Some(running);
        self.agent.applied = 0;
    }

    /// Bring the bridge up before the first frame, for
    /// `voxelith --agent-port`. `None` leaves it off.
    ///
    /// Nothing here needs the window or the GPU — a socket and a channel
    /// are all it is — so a call that arrives before the first frame
    /// simply queues until the frame loop drains it.
    pub fn start_agent_bridge_at(&mut self, port: Option<u16>) {
        let Some(port) = port else {
            return;
        };
        self.start_agent_bridge(port);
        // Someone who started the editor this way is here to watch an
        // agent work: put the panel in front of them, so the URL and the
        // approval switch aren't behind a menu.
        self.ui.state.panels.show_agent = true;
    }

    /// Stop serving and drop the line.
    pub(super) fn stop_agent_bridge(&mut self) {
        if !self.agent.is_running() {
            return;
        }
        // Answer whatever is parked before the line goes away. An agent
        // waiting on approval has to hear that the bridge closed rather
        // than sit until its own timeout with no reason given.
        self.drop_pending_review(
            "batch_not_applied",
            "the agent bridge was switched off while this batch was waiting for approval",
            "the bridge was switched off",
        );
        self.agent.server = None;
        self.agent.receiver = None;
        self.ui.set_status("Agent bridge stopped");
    }

    /// Drain this frame's calls. Cheap when the bridge is off.
    pub(super) fn tick_agent_bridge(&mut self) {
        self.expire_pending_review();
        loop {
            let Some(call) = self
                .agent
                .receiver
                .as_mut()
                .and_then(BridgeReceiver::try_recv)
            else {
                return;
            };
            self.serve_agent_call(call);
        }
    }

    /// Commit the parked batch. The Accept button, and nothing else.
    pub(super) fn accept_agent_batch(&mut self) {
        let Some(pending) = self.agent.pending.take() else {
            return;
        };
        self.clear_review_preview();

        if self.history_mark() != pending.history_mark {
            pending.call.answer(Err(Refusal::new(
                "world_changed",
                "the project was edited while this batch waited for approval, so it no longer \
                 describes the world it was built against; nothing was applied — send it \
                 again",
            )));
            self.ui
                .set_status("Agent batch dropped — the project changed while it waited");
            return;
        }

        let applied = self.commit_agent_batch(pending.outcome, Reviewed::Accepted);
        pending.call.answer(Ok(Answer::Applied(Box::new(applied))));
    }

    /// The scene is being replaced, so whatever is parked no longer
    /// describes anything. Called from `reset_scene_session_state`,
    /// which every path that throws the world away goes through.
    pub(super) fn drop_pending_review_for_new_scene(&mut self) {
        self.drop_pending_review(
            "world_changed",
            "the editor loaded a different project while this batch waited for approval; \
             nothing was applied — describe the current world before sending it again",
            "dropped — the project was replaced",
        );
    }

    /// Decline it. The world is untouched, and the agent is told so in
    /// terms it can act on.
    pub(super) fn reject_agent_batch(&mut self) {
        self.drop_pending_review(
            "rejected",
            "the person at the editor declined this batch; nothing was applied — ask them what \
             they want different before sending it again",
            "rejected",
        );
    }

    /// The panel's per-frame snapshot, mirrored across the UI boundary
    /// so the panel reads it off `Ui` without borrowing `App` back.
    pub(super) fn agent_view(&self) -> AgentView {
        AgentView {
            url: self.agent.server.as_ref().map(RunningBridge::url),
            client_command: self
                .agent
                .server
                .as_ref()
                .map(|running| running.client_command().to_string()),
            approval: self.agent.approval,
            applied: self.agent.applied,
            pending: self.agent.pending.as_ref().map(PendingReview::summary),
        }
    }

    /// Switch between committing an agent's batches on arrival and being
    /// asked first.
    pub(super) fn set_agent_approval(&mut self, approval: Approval) {
        if self.agent.approval == approval {
            return;
        }
        self.agent.approval = approval;
        // Turning approval *off* with something already parked would
        // leave a preview on screen that nothing can answer any more:
        // the strip is gone, and auto mode has no notion of a batch
        // waiting. Land it — the user just said batches may land — and
        // the agent hears one clear answer either way.
        if approval == Approval::Auto && self.agent.pending.is_some() {
            self.accept_agent_batch();
        }
        self.ui.set_status(match approval {
            Approval::Auto => "Agent batches apply as they arrive",
            Approval::Review => "Agent batches wait for your approval",
        });
    }

    /// The witness that the world has not moved under a parked batch.
    ///
    /// This used to be the `(undo, redo)` depths, on the reasoning that
    /// every edit goes through `CommandHistory` so one of them must
    /// change. Every edit does — but the *pair* comes back to a value
    /// it already held in three ordinary ways: undo then draw, draw
    /// with the undo stack already full, and continuing a stroke. Each
    /// leaves a different world behind the same two numbers, and
    /// accepting a batch there commits `old_voxel`s describing a world
    /// that is gone, so undoing it restores a state that never existed.
    ///
    /// `CommandHistory::generation` moves on every one of those and
    /// never moves back, which makes the comparison below say what it
    /// always claimed to say.
    fn history_mark(&self) -> u64 {
        self.editor.history.generation()
    }

    /// Drop whatever is parked, telling the agent why.
    fn drop_pending_review(&mut self, code: &'static str, message: &str, status: &str) {
        let Some(pending) = self.agent.pending.take() else {
            return;
        };
        self.clear_review_preview();
        pending.call.answer(Err(Refusal::new(code, message)));
        self.ui.set_status(format!("Agent batch {status}"));
    }

    fn expire_pending_review(&mut self) {
        let Some(pending) = &self.agent.pending else {
            return;
        };
        // Checked here as well as at Accept, and this is the copy that
        // matters to the person: the moment they edit anything, the
        // parked batch is already doomed (its `old_voxel`s describe the
        // world as it was), and the three local paths that write
        // straight into the world — Generate, Run Pipeline, Import —
        // also clear the overlay it was being previewed in. Waiting for
        // a click would leave the strip asking them to approve
        // something they can no longer see, and leave the agent hanging
        // for an answer that was decided the moment they picked up the
        // brush.
        if pending.history_mark != self.editor.history.generation() {
            self.drop_pending_review(
                "world_changed",
                "the project was edited while this batch waited for approval, so it no longer \
                 describes the world it was built against; nothing was applied — describe the \
                 current world and send it again",
                "dropped — the project changed while it waited",
            );
            return;
        }
        if pending.call.abandoned() {
            self.drop_pending_review(
                "batch_not_applied",
                "the agent stopped waiting before a human answered; nothing was applied",
                "dropped — the agent stopped waiting",
            );
        } else if pending.parked_at.elapsed() >= REVIEW_TIMEOUT {
            self.drop_pending_review(
                "batch_not_applied",
                "nobody at the editor answered in time; nothing was applied — send it again \
                 when someone is watching",
                "dropped — nobody answered in time",
            );
        }
    }

    fn serve_agent_call(&mut self, call: BridgeCall) {
        // `apply_ops` is the one call that may not answer at once, since
        // under review it parks until a human decides — so it takes the
        // whole call. Running the batch here, while the request is still
        // borrowed, keeps that borrow out of the answering path.
        let batch = match &call.request {
            AgentRequest::ApplyOps(batch) => Some((
                agent_ops::run_batch(
                    agent_ops::BatchInput {
                        world: &self.document.world,
                        selection: self.editor.selection,
                        graph: &self.document.graph,
                    },
                    batch,
                ),
                batch.options.dry_run,
            )),
            _ => None,
        };
        match batch {
            Some((outcome, dry_run)) => self.finish_apply_ops(call, outcome, dry_run),
            None => {
                let reply = self.answer_agent_query(&call.request);
                call.answer(reply);
            }
        }
    }

    /// Everything that answers straight from the world as it stands.
    fn answer_agent_query(&mut self, request: &AgentRequest) -> AgentReply {
        match request {
            AgentRequest::Describe => Ok(Answer::Described(Box::new(Described {
                status: self.bridge_status(),
                description: agent_ops::describe(self.document_view()),
            }))),

            AgentRequest::Slice(request) => match agent_ops::slice(&self.document.world, request) {
                Ok(text) => Ok(Answer::Sliced(Box::new(Sliced {
                    status: self.bridge_status(),
                    slice: text.lines().map(String::from).collect(),
                }))),
                Err(e) => Err(e.into()),
            },

            AgentRequest::RenderViews { views, size } => {
                let mut rendered = Vec::with_capacity(views.len());
                for &kind in views {
                    match view::render(&self.document.world, kind, *size) {
                        Ok(view) => rendered.push(view),
                        // Same rule as everywhere else in this protocol:
                        // a size out of range is refused, not clamped.
                        Err(e) => return Err(Refusal::new("invalid_size", e.to_string())),
                    }
                }
                Ok(Answer::Views(Box::new(Views {
                    status: self.bridge_status(),
                    views: rendered,
                })))
            }

            // The history is shared, so an agent's undo can step back
            // over a human's brush stroke. That is the point rather than
            // a hazard: "undo that" should work whoever did it, and the
            // human's own Ctrl+Z is right there to put it back.
            AgentRequest::Undo => {
                let stepped = self
                    .editor
                    .history
                    .undo(&mut self.document.world, &mut self.document.graph);
                Ok(self.stepped_and_meshed(stepped))
            }
            AgentRequest::Redo => {
                let stepped = self
                    .editor
                    .history
                    .redo(&mut self.document.world, &mut self.document.graph);
                Ok(self.stepped_and_meshed(stepped))
            }

            // Only reachable if `serve_agent_call` stops routing batches
            // away from here — a bug in this file, not something an
            // agent can provoke or act on.
            AgentRequest::ApplyOps(_) => {
                log::error!("agent bridge: an ops batch reached the read-only path");
                Err(Refusal::new(
                    "internal_error",
                    "the editor mishandled this batch; nothing was applied",
                ))
            }
        }
    }

    /// Answer an undo / redo, re-meshing first when it moved.
    ///
    /// Same ordering as `commit_agent_batch`: the world changed, so the
    /// screen and the modified flag have to catch up before the answer
    /// reports on them.
    fn stepped_and_meshed(&mut self, stepped: bool) -> Answer {
        if stepped {
            self.rebuild_all_meshes();
            // The rebuild only flags the document when a chunk went
            // dirty; an entry that carried nothing but a graph
            // transition moved the document without touching one.
            self.document.bump();
        }
        Answer::Stepped(Box::new(Stepped {
            stepped,
            status: self.bridge_status(),
        }))
    }

    fn finish_apply_ops(
        &mut self,
        call: BridgeCall,
        outcome: Result<BatchOutcome, OpsError>,
        dry_run: bool,
    ) {
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(e) => return call.answer(Err(e.into())),
        };

        // A dry run changed nothing and has nothing to approve. Its
        // description has to come from the preview's world, not this
        // one: a report of the world after the batch beside a
        // description of the world before it is the one way to make
        // "what would this do?" actively misleading.
        if dry_run {
            let description = agent_ops::describe(DocumentView {
                world: &outcome.world,
                selection: outcome.selection,
                sockets: &self.document.sockets,
                // Same rule as the world: under a dry run the graph to
                // describe is the one the batch *would* leave, which is
                // the editor's own unless the batch carried a new one.
                graph: outcome.graph.as_ref().unwrap_or(&self.document.graph),
                // The history genuinely did not move, so these are the
                // editor's real depths rather than the preview's zeros.
                undo_depth: self.editor.history.undo_count(),
                redo_depth: self.editor.history.redo_count(),
            });
            return call.answer(Ok(Answer::Applied(Box::new(Applied {
                status: self.bridge_status(),
                report: outcome.report,
                description,
                review: Reviewed::Auto,
            }))));
        }

        match self.agent.approval {
            Approval::Auto => {
                let applied = self.commit_agent_batch(outcome, Reviewed::Auto);
                call.answer(Ok(Answer::Applied(Box::new(applied))));
            }
            Approval::Review => self.park_for_review(call, outcome),
        }
    }

    fn park_for_review(&mut self, call: BridgeCall, outcome: BatchOutcome) {
        if self.agent.pending.is_some() {
            // Queueing a second batch would mean approving one built
            // against a world the first one is about to change. Refuse
            // instead, and say what is holding it up.
            return call.answer(Err(Refusal::new(
                "review_pending",
                "a previous batch is still waiting for the person at the editor to approve it; \
                 nothing was applied — try again once they have answered",
            )));
        }

        self.show_review_preview(&outcome.changes);
        let pending = PendingReview {
            call,
            outcome,
            history_mark: self.history_mark(),
            parked_at: Instant::now(),
        };
        self.ui.set_status(format!(
            "Agent batch {} — waiting for you",
            pending.summary()
        ));
        self.agent.pending = Some(pending);
    }

    /// Land a batch on the user's own history and report it.
    fn commit_agent_batch(&mut self, outcome: BatchOutcome, review: Reviewed) -> Applied {
        if !outcome.changes.is_empty() {
            // Where the agent worked, so "Frame Generated" goes there.
            self.last_generated_bounds =
                super::bounds_of(outcome.changes.iter().map(|change| change.pos));
        }
        let changed = outcome.changes.len();

        // A graph the batch carried goes into the Graph panel — that is
        // what sending a graph is *for*: the agent picks the generators
        // and wires them, the human takes over at the sliders. It rides
        // inside the batch's command as a before/after transition, so
        // the one Ctrl+Z that takes the voxels back out restores the
        // graph the batch replaced too. Laid out before it's stored,
        // since an agent sends no positions and every node would
        // otherwise pile up on the origin (and redo must re-apply the
        // laid-out version, not re-pile them).
        let graph = outcome.graph.map(|mut after| {
            if after.all_at_origin() {
                after.relayout();
            }
            voxelith::editor::GraphTransition {
                before: self.document.graph.clone(),
                after,
            }
        });
        if graph.as_ref().is_some_and(|t| t.before != t.after) {
            // A batch that only carried a graph changed no voxel, so the
            // re-mesh below finds nothing dirty and the document would
            // answer "no unsaved changes" while holding a pipeline that
            // exists nowhere on disk.
            self.document.bump();
        }

        // One command for the whole batch: one entry on the same stack
        // the user's brush strokes push onto, so one Ctrl+Z takes the
        // agent's step back out.
        self.editor.history.execute_with_graph(
            Command::set_voxels_with_graph(outcome.changes, graph),
            &mut self.document.world,
            &mut self.document.graph,
        );

        // A batch that ends with no selection is clearing one, and
        // clearing a selection is more than assigning `None` — the drag
        // anchor and the move ghost belong to it too.
        match outcome.selection {
            Some(selection) => self.editor.selection = Some(selection),
            None => self.deselect(),
        }

        self.agent.applied += 1;
        self.ui
            .set_status(format!("Agent applied {changed} voxel changes"));

        // Re-mesh before answering rather than letting the frame loop
        // reach its own call further down. Two things depend on the
        // order: the geometry is on screen in the frame the agent is
        // told about it, and — since that call is also where the
        // document is flagged as modified — the `unsaved_changes` in
        // this answer is true rather than a frame stale. An agent that
        // has just written 125 voxels being told the document has no
        // unsaved changes is the editor lying about its own state. The
        // later call this frame finds no dirty chunks and returns at
        // once.
        self.rebuild_all_meshes();

        Applied {
            status: self.bridge_status(),
            report: outcome.report,
            description: agent_ops::describe(self.document_view()),
            review,
        }
    }

    fn document_view(&self) -> DocumentView<'_> {
        DocumentView {
            world: &self.document.world,
            selection: self.editor.selection,
            sockets: &self.document.sockets,
            graph: &self.document.graph,
            undo_depth: self.editor.history.undo_count(),
            redo_depth: self.editor.history.redo_count(),
        }
    }

    fn bridge_status(&self) -> BridgeStatus {
        BridgeStatus {
            path: self.project_path.as_deref().map(voxelith::mcp::display),
            voxel_count: solid_voxel_count(&self.document.world),
            undo_depth: self.editor.history.undo_count(),
            redo_depth: self.editor.history.redo_count(),
            unsaved_changes: self.document.unsaved(),
            approval: self.agent.approval,
        }
    }

    /// Put the parked batch on screen as translucent geometry.
    fn show_review_preview(&mut self, changes: &[VoxelChange]) {
        let voxels: Vec<((i32, i32, i32), Voxel)> = changes
            .iter()
            .map(|change| {
                let voxel = match change.new_voxel.is_air() {
                    true => CLEARED_TINT,
                    false => change.new_voxel,
                };
                (change.pos, voxel)
            })
            .collect();
        let mesh = patch_to_mesh(&voxels, REVIEW_ALPHA);
        // Claims the overlay slot as `AgentReview` — the procgen ticks
        // stand down while the batch is parked, so nothing repaints it.
        self.show_review_preview_mesh(&mesh);
    }

    /// Hand the overlay slot back to the procgen previews.
    ///
    /// `invalidate_preview` also resets their state machines, so
    /// whichever of them is switched on re-renders into the freed slot
    /// on the next tick instead of staying dark until a parameter moves.
    fn clear_review_preview(&mut self) {
        self.invalidate_preview();
    }
}

/// Solid voxels in the world. Per-chunk counts are maintained on write,
/// so this walks chunks rather than cells.
fn solid_voxel_count(world: &World) -> u64 {
    world
        .chunks()
        .map(|(_, chunk)| chunk.read().solid_count() as u64)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn change(pos: (i32, i32, i32), new_voxel: Voxel) -> VoxelChange {
        VoxelChange {
            pos,
            old_voxel: Voxel::AIR,
            new_voxel,
        }
    }

    /// What the approval strip says has to name deletions separately —
    /// see [`summarize`].
    #[test]
    fn the_summary_counts_writes_and_clears_apart() {
        let solid = Voxel::new(1, 10, 20, 30);
        assert_eq!(summarize(&[]), "changes nothing");
        assert_eq!(summarize(&[change((0, 0, 0), solid)]), "writes 1 voxel");
        assert_eq!(
            summarize(&[change((0, 0, 0), solid), change((1, 0, 0), solid)]),
            "writes 2 voxels"
        );
        assert_eq!(
            summarize(&[change((0, 0, 0), Voxel::AIR)]),
            "clears 1 voxel"
        );
        assert_eq!(
            summarize(&[change((0, 0, 0), solid), change((1, 0, 0), Voxel::AIR)]),
            "writes 1, clears 1"
        );
    }

    /// The preview has to show cells the batch would clear.
    /// `patch_to_mesh` skips air, so a delete-only batch would otherwise
    /// put nothing on screen and ask a human to approve an invisible
    /// change.
    #[test]
    fn cleared_cells_get_a_visible_stand_in() {
        assert!(CLEARED_TINT.is_solid(), "air would not render at all");
        assert!(
            patch_to_mesh(&[((0, 0, 0), Voxel::AIR)], REVIEW_ALPHA).is_empty(),
            "air renders as nothing — which is why the stand-in exists"
        );
        assert!(
            !patch_to_mesh(&[((0, 0, 0), CLEARED_TINT)], REVIEW_ALPHA).is_empty(),
            "the stand-in must render"
        );
    }
}
