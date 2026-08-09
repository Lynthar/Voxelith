//! The editor as the document.
//!
//! [`super::VoxelithMcp`] serves a document this process owns outright,
//! which is exactly what a headless agent wants. This serves the one
//! already open in front of a human: the world, the selection and the
//! undo stack all belong to the editor, and every tool call is a message
//! to its main thread.
//!
//! That difference is the whole point. An agent's batch and a hand-drawn
//! brush stroke land on **one** `CommandHistory`, so a human undoes an
//! agent's step with the Ctrl+Z they already use and takes over
//! mid-build without stopping anything. The checkpoint-plus-file-watch
//! path this grew out of could never do that: two processes, two worlds,
//! two undo stacks, and a `.vxlt` passed between them meant one writer
//! at a time and a human whose only move was to stop the agent first.
//!
//! The tool set is deliberately smaller than the headless server's:
//! editing and looking, no file operations. Someone is sitting at this
//! document — where it saves is theirs to decide, and `new_project`
//! would leave an agent's call parked behind the editor's
//! unsaved-changes prompt waiting for a click nobody knows to make.
//!
//! Nothing here touches the world itself. The main thread owns it, this
//! side owns the wire, and [`AgentRequest`] / [`Answer`] are the only
//! vocabulary between them — which is also why the editor never has to
//! build a `CallToolResult` and this module never has to know what a
//! chunk is.

use base64::Engine as _;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerInfo};
use rmcp::{tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler};
use serde::Serialize;
use tokio::sync::{mpsc, oneshot};

use crate::agent_ops::{ApplyReport, Description, OpsBatch, OpsError, SliceRequest};
use crate::exec::ExecError;
use crate::view;

use super::{answered, refused, RenderArgs, Rendered};

/// Loopback port the editor offers by default.
///
/// Deliberately not the headless server's 8080: someone running
/// `voxelith mcp --http` alongside the editor should not have to
/// discover the clash by having one of them fail to start.
pub const DEFAULT_PORT: u16 = 8737;

/// Largest image edge this bridge will render.
///
/// Half the headless server's ceiling, for a reason that has nothing to
/// do with the picture: these renders run on the editor's frame loop,
/// so their cost is measured in seconds the window doesn't respond.
/// The project's own measurement is ~480 ms for one 1024² view against
/// a small model — seven of those is a program that looks hung. At 512
/// a full sweep is a hitch; at the 256 default it is a frame.
pub const BRIDGE_MAX_SIZE: u32 = 512;

/// What the server asks the editor's main thread to do.
///
/// One variant per tool, and no variant that writes a file: see the
/// module note on why this set is smaller than the headless server's.
#[derive(Debug)]
pub enum AgentRequest {
    /// Boxed because an `OpsBatch` carries up to 256 ops and every other
    /// variant here is a handful of bytes.
    ApplyOps(Box<OpsBatch>),
    Describe,
    Slice(SliceRequest),
    RenderViews {
        views: Vec<view::ViewKind>,
        size: u32,
    },
    Undo,
    Redo,
}

/// How the editor treats a batch that arrives from an agent.
///
/// The default is [`Approval::Auto`]: the batch lands, the human watches
/// it land, and their Ctrl+Z is right there if it was wrong. That is a
/// better loop than gating every step, because an agent typically works
/// in a run of batches and a human who has to approve each one is doing
/// data entry rather than art direction.
///
/// [`Approval::Review`] is for when they'd rather see it first: the
/// batch goes up as a translucent preview and the agent's call waits
/// until they answer.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Approval {
    #[default]
    Auto,
    Review,
}

impl Approval {
    pub fn label(self) -> &'static str {
        match self {
            Approval::Auto => "Apply directly",
            Approval::Review => "Ask me first",
        }
    }
}

/// What became of a batch that was actually committed.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Reviewed {
    /// The editor was in [`Approval::Auto`] and committed it.
    Auto,
    /// A human looked at the preview and said yes.
    Accepted,
}

/// Where the editor's document stands, attached to every answer — an
/// agent that just applied a batch shouldn't need a second call to see
/// what it did.
#[derive(Debug, Serialize)]
pub struct BridgeStatus {
    /// The project file open in the editor, or `null` for a scene that
    /// has never been saved. Read-only here: this server has no save
    /// tool, so the path is context, not a target.
    pub path: Option<String>,
    pub voxel_count: u64,
    pub undo_depth: usize,
    pub redo_depth: usize,
    /// The human has edits of their own that aren't on disk. Worth
    /// knowing before suggesting they close anything.
    pub unsaved_changes: bool,
    /// How the editor is set to treat incoming batches right now.
    pub approval: Approval,
}

#[derive(Debug, Serialize)]
pub struct Applied {
    #[serde(flatten)]
    pub status: BridgeStatus,
    pub report: ApplyReport,
    /// The same world the report counts, always — under `dry_run` both
    /// come from the preview. Answering with a report of the world after
    /// the batch beside a description of the world before it is the one
    /// way to make "what would this do?" actively misleading.
    pub description: Description,
    pub review: Reviewed,
}

#[derive(Debug, Serialize)]
pub struct Described {
    #[serde(flatten)]
    pub status: BridgeStatus,
    pub description: Description,
}

#[derive(Debug, Serialize)]
pub struct Sliced {
    #[serde(flatten)]
    pub status: BridgeStatus,
    pub slice: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct Stepped {
    /// False when there was nothing left on the stack — not an error,
    /// just the end of the history. Note that the stack is shared, so
    /// what an agent undoes here may well be something the human drew.
    pub stepped: bool,
    #[serde(flatten)]
    pub status: BridgeStatus,
}

/// Rendered views plus the status the other answers carry.
#[derive(Debug)]
pub struct Views {
    pub status: BridgeStatus,
    pub views: Vec<view::View>,
}

/// A refused call, in the same `{code, message}` shape the CLI and the
/// headless server refuse with, so an agent that has driven one of those
/// already knows how to read this.
#[derive(Debug, Clone)]
pub struct Refusal {
    pub code: &'static str,
    pub message: String,
}

impl Refusal {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl From<OpsError> for Refusal {
    fn from(error: OpsError) -> Self {
        Self {
            code: error.code.as_str(),
            message: error.to_string(),
        }
    }
}

impl From<Refusal> for ExecError {
    fn from(refusal: Refusal) -> Self {
        ExecError::new(refusal.code, refusal.message)
    }
}

/// What the editor sends back.
pub enum Answer {
    Applied(Box<Applied>),
    Described(Box<Described>),
    Sliced(Box<Sliced>),
    Views(Box<Views>),
    Stepped(Box<Stepped>),
}

/// The editor's answer to one call.
pub type AgentReply = Result<Answer, Refusal>;

/// One call in flight: what was asked, and the line the answer goes back
/// on.
///
/// The editor may hold on to a call rather than answering it at once —
/// that is what [`Approval::Review`] is, a batch parked on screen until
/// a human decides. Dropping it unparked still unblocks the agent, with
/// "the editor went away", so a forgotten branch fails loudly instead of
/// hanging the tool call forever.
pub struct BridgeCall {
    pub request: AgentRequest,
    reply: oneshot::Sender<AgentReply>,
}

impl BridgeCall {
    pub fn answer(self, reply: AgentReply) {
        // Send fails when the agent's client gave up waiting. Nothing to
        // do about it here: whatever the editor did, it did.
        let _ = self.reply.send(reply);
    }

    /// Whether anyone is still waiting for this answer. A parked call
    /// whose agent has timed out should be cleaned off the screen rather
    /// than left asking a human to approve something nobody will hear
    /// the answer to.
    pub fn abandoned(&self) -> bool {
        self.reply.is_closed()
    }
}

/// The server's end: clone it into as many handlers as the transport
/// builds.
#[derive(Clone)]
pub struct BridgeHandle {
    calls: mpsc::UnboundedSender<BridgeCall>,
}

impl BridgeHandle {
    /// Ask the editor, and wait for its answer.
    ///
    /// Unbounded on the way in and one call per tool invocation on the
    /// way out: the queue is a handful of messages that the main thread
    /// drains every frame, and a bounded channel would only add a way
    /// for the server to block on a rendering editor.
    async fn call(&self, request: AgentRequest) -> AgentReply {
        let (reply, answer) = oneshot::channel();
        self.calls
            .send(BridgeCall { request, reply })
            .map_err(|_| editor_gone())?;
        answer.await.map_err(|_| editor_gone())?
    }
}

fn editor_gone() -> Refusal {
    Refusal::new(
        "editor_unavailable",
        "the Voxelith editor hosting this server is no longer accepting calls — \
         it may have been closed, or the agent bridge switched off",
    )
}

/// The editor's end: drained on the main thread, one frame at a time.
pub struct BridgeReceiver {
    calls: mpsc::UnboundedReceiver<BridgeCall>,
}

impl BridgeReceiver {
    /// The next queued call, or `None` when there is nothing waiting.
    ///
    /// Never blocks — this runs inside the frame loop. A closed channel
    /// reads as `None` for the same reason: the editor owns the server's
    /// lifetime, so a channel with no senders means it already stopped
    /// the thing on the other end.
    pub fn try_recv(&mut self) -> Option<BridgeCall> {
        self.calls.try_recv().ok()
    }
}

/// A fresh editor ↔ server pair.
pub fn channel() -> (BridgeHandle, BridgeReceiver) {
    let (calls, rx) = mpsc::unbounded_channel();
    (
        BridgeHandle { calls },
        BridgeReceiver { calls: rx },
    )
}

/// The server. Cloning shares the line to the editor, which the HTTP
/// transport depends on — it builds a handler per request, and every one
/// of them has to reach the same editor.
#[derive(Clone)]
pub struct BridgeMcp {
    editor: BridgeHandle,
    /// Built once and pointed at explicitly, for the reason spelled out
    /// on [`super::VoxelithMcp`]: the macro's default regenerates every
    /// tool's schema — the whole ops union included — on every request.
    tool_router: ToolRouter<BridgeMcp>,
}

/// The editor answered, but with the wrong kind of answer. Only reachable
/// through a bug in the dispatch on the other side, so it is an internal
/// error rather than something to phrase for an agent to act on.
fn mismatched(tool: &str) -> Result<CallToolResult, McpError> {
    log::error!("agent bridge: the editor answered {tool} with the wrong reply type");
    Err(McpError::internal_error(
        format!("the editor mishandled the {tool} call"),
        None,
    ))
}

#[tool_router]
impl BridgeMcp {
    pub fn new(editor: BridgeHandle) -> Self {
        Self {
            editor,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Apply a batch of edit operations to the project open in the \
                       Voxelith editor. Atomic (any failure changes nothing), sequential \
                       (each op sees the previous ones), and one undo entry for the whole \
                       batch — on the same undo stack as the human's own edits, so they \
                       can take any step back with Ctrl+Z. Set options.dry_run to preview: \
                       the report and description then describe the world the batch \
                       *would* produce, and nothing is committed. The editor may be set to \
                       ask its user before applying, in which case this call waits for \
                       their answer and can come back refused."
    )]
    async fn apply_ops(
        &self,
        Parameters(batch): Parameters<OpsBatch>,
    ) -> Result<CallToolResult, McpError> {
        match self
            .editor
            .call(AgentRequest::ApplyOps(Box::new(batch)))
            .await
        {
            Ok(Answer::Applied(applied)) => answered(applied),
            Ok(_) => mismatched("apply_ops"),
            Err(refusal) => refused(&refusal.into()),
        }
    }

    #[tool(
        description = "List the generators an op with \"op\": \"generate\" can call, each \
                       with its parameters at their default values. That listing is the \
                       parameter template: copy it, change what you want, send it back. \
                       Also returns graph_template, a working pipeline graph for the \
                       \"graph\" op — read it before writing one, it is where the node \
                       format is defined."
    )]
    fn list_generators(&self) -> Result<CallToolResult, McpError> {
        // The registry is static data, so this one needs nothing from
        // the editor and doesn't queue a call for it.
        answered(Generators {
            generators: crate::agent_ops::generator_infos(),
            graph_template: crate::agent_ops::graph_template(),
        })
    }

    #[tool(
        description = "Summarize the project open in the editor: voxel and chunk counts, \
                       bounding box, the most common colors, emissive / metallic / \
                       tint-zone tallies, sockets, selection and undo depth. Also measures \
                       the shape itself — connected components, floating parts, enclosed \
                       voxels, per-axis symmetry — which a rendered view cannot tell you \
                       reliably, and returns the pipeline graph the editor has open."
    )]
    async fn describe(&self) -> Result<CallToolResult, McpError> {
        match self.editor.call(AgentRequest::Describe).await {
            Ok(Answer::Described(described)) => answered(described),
            Ok(_) => mismatched("describe"),
            Err(refusal) => refused(&refusal.into()),
        }
    }

    #[tool(
        description = "Render one axis-aligned plane of the open project as ASCII art — \
                       the cheapest way to catch \"the door is one cell too high\". The \
                       first line states the axis ranges and row order."
    )]
    async fn slice(
        &self,
        Parameters(request): Parameters<SliceRequest>,
    ) -> Result<CallToolResult, McpError> {
        match self.editor.call(AgentRequest::Slice(request)).await {
            Ok(Answer::Sliced(sliced)) => answered(sliced),
            Ok(_) => mismatched("slice"),
            Err(refusal) => refused(&refusal.into()),
        }
    }

    #[tool(description = "Undo the last step — an agent batch or a human edit, whichever \
                          is on top of the shared history.")]
    async fn undo(&self) -> Result<CallToolResult, McpError> {
        self.step(AgentRequest::Undo, "undo").await
    }

    #[tool(description = "Redo the step undone last.")]
    async fn redo(&self) -> Result<CallToolResult, McpError> {
        self.step(AgentRequest::Redo, "redo").await
    }

    #[tool(
        description = "Look at the project as the editor is showing it: renders it as \
                       images and returns them inline. Defaults to one isometric view, \
                       which shows all three dimensions at once; ask for axis views \
                       (front, back, left, right, top, bottom, or all) when you need to \
                       check one side square-on. 256 pixels reads a silhouette and its \
                       colors fine. Each image comes with the cell bounds it covers and \
                       the cells-per-pixel, so a measurement off the picture converts back \
                       to coordinates. This is a CPU render of the voxels, not a \
                       screenshot: it ignores where the human has pointed their camera."
    )]
    async fn render_views(
        &self,
        Parameters(args): Parameters<RenderArgs>,
    ) -> Result<CallToolResult, McpError> {
        let size = args.size.unwrap_or(view::DEFAULT_SIZE);
        // Lower than the headless server's ceiling, and refused rather
        // than clamped like every other size in this protocol. The
        // difference is whose thread this runs on: the editor renders
        // in its frame loop, so the whole window stops responding for
        // as long as it takes, and seven views at 1024² is more than
        // three seconds of a program that looks hung. The same request
        // to `voxelith mcp` is fine — nobody is sitting in front of it.
        if size > BRIDGE_MAX_SIZE {
            return refused(&ExecError::new(
                "invalid_size",
                format!(
                    "size {size} would freeze the editor while it renders; \
                     the in-editor bridge tops out at {BRIDGE_MAX_SIZE} pixels — \
                     render larger images with `voxelith render` or `voxelith mcp`"
                ),
            ));
        }
        let views = match args.views.is_empty() {
            true => vec![view::ViewKind::Iso],
            false => args.views,
        };
        let rendered = match self
            .editor
            .call(AgentRequest::RenderViews { views, size })
            .await
        {
            Ok(Answer::Views(views)) => views,
            Ok(_) => return mismatched("render_views"),
            Err(refusal) => return refused(&refusal.into()),
        };

        // One text block then one image per view, rather than a summary
        // followed by a pile of pictures: the caption has to sit next to
        // the image it describes, or six views come back as six images
        // an agent has to count to identify.
        let mut blocks = Vec::with_capacity(rendered.views.len() * 2 + 1);
        blocks.push(ContentBlock::text(
            serde_json::to_string_pretty(&rendered.status).map_err(|e| {
                McpError::internal_error(format!("could not describe the document: {e}"), None)
            })?,
        ));
        for view in &rendered.views {
            let caption = serde_json::to_string_pretty(&Rendered {
                view: view.kind.as_str(),
                size: view.size,
                framing: &view.framing,
                empty: view.empty,
                truncated: view.truncated,
            })
            .map_err(|e| {
                McpError::internal_error(format!("could not describe the view: {e}"), None)
            })?;
            blocks.push(ContentBlock::text(caption));
            blocks.push(ContentBlock::image(
                base64::engine::general_purpose::STANDARD.encode(&view.png),
                "image/png",
            ));
        }
        Ok(CallToolResult::success(blocks))
    }

    /// Undo and redo differ only in which request they send, and both
    /// answer with the same shape.
    async fn step(&self, request: AgentRequest, tool: &str) -> Result<CallToolResult, McpError> {
        match self.editor.call(request).await {
            Ok(Answer::Stepped(stepped)) => answered(stepped),
            Ok(_) => mismatched(tool),
            Err(refusal) => refused(&refusal.into()),
        }
    }
}

#[derive(Serialize)]
struct Generators {
    generators: Vec<crate::agent_ops::GeneratorInfo>,
    /// A working pipeline graph to copy — the only place the graph
    /// format is spelled out, since the `graph` op's schema keeps it an
    /// opaque object rather than costing every turn nine definitions.
    graph_template: serde_json::Value,
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for BridgeMcp {
    fn get_info(&self) -> ServerInfo {
        // Named apart from the headless server on purpose: this is the
        // name a client shows its user, and the two differ in the one
        // way that matters to whoever is picking — whether there is a
        // human watching. (Spelled out rather than
        // `Implementation::from_build_env()`, which reads the *rmcp*
        // crate's build environment; see the same call on
        // `VoxelithMcp`.)
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(
                Implementation::new(
                    concat!(env!("CARGO_PKG_NAME"), "-editor"),
                    env!("CARGO_PKG_VERSION"),
                )
                .with_title("Voxelith (open editor)")
                .with_description(
                    "Edit the voxel project a human currently has open in the Voxelith \
                     editor.",
                ),
            )
            .with_instructions(
                "Voxel modeling inside a running Voxelith editor. Someone has this project \
                 open and is watching: your edits appear in their viewport as you make \
                 them and land on the same undo stack as their own, so they can step back \
                 into your work or take over at any point.\n\n\
                 The loop is: apply_ops -> render_views to see what you built -> fix it -> \
                 repeat. Call list_generators first for the parametric generators. \
                 describe and slice answer the questions a picture can't: exact counts, \
                 exact coordinates.\n\n\
                 Coordinates are integer cells, Y is up, and every region is inclusive on \
                 both ends: min [0,0,0] max [1,1,1] is 8 cells. There is no separate \
                 erase, paint or fill op — \"voxel\": \"air\" erases, write_mode \
                 \"only_solid\" repaints without changing the silhouette, and a filled box \
                 fills. Unknown fields are refused rather than ignored, so a misspelling \
                 is reported instead of silently doing nothing.\n\n\
                 There are no file tools here on purpose: saving, exporting and opening \
                 belong to the person at the keyboard. The editor may also be set to ask \
                 them before applying a batch, in which case apply_ops waits for their \
                 answer and can come back refused."
                    .to_string(),
            )
    }
}

/// Serve Streamable HTTP at `/mcp` on an already-bound socket.
///
/// The listener is bound by the caller, on the editor's own thread, so a
/// port already in use is an error a human sees in the panel where they
/// switched the bridge on — rather than a line in a log file from a task
/// that quietly never started.
///
/// The handler factory hands out clones of one server for the reason
/// spelled out on [`super::serve_http`]: under the 2026-07-28 spec each
/// POST is its own MCP session, so a fresh server per request would give
/// every call its own line to nowhere.
#[cfg(feature = "mcp-http")]
pub async fn serve_http_bridged(
    editor: BridgeHandle,
    listener: std::net::TcpListener,
) -> anyhow::Result<()> {
    use rmcp::transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpService, StreamableHttpServerConfig,
    };

    let server = BridgeMcp::new(editor);
    let service = StreamableHttpService::new(
        move || Ok(server.clone()),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );
    let router = axum::Router::new().nest_service("/mcp", service);
    listener.set_nonblocking(true)?;
    let listener = tokio::net::TcpListener::from_std(listener)?;
    axum::serve(listener, router).await?;
    Ok(())
}

/// Held by the editor for as long as the bridge is on: the address it is
/// listening on, and the task serving it.
///
/// Dropping this aborts the server task, which closes the socket. There
/// is no graceful drain because there is nothing to drain — a call in
/// flight is a call the editor is answering on its own thread, and one
/// that never gets answered is exactly what [`BridgeCall`]'s dropped
/// sender reports.
pub struct RunningBridge {
    pub address: std::net::SocketAddr,
    task: tokio::task::JoinHandle<()>,
}

impl RunningBridge {
    pub fn new(address: std::net::SocketAddr, task: tokio::task::JoinHandle<()>) -> Self {
        Self { address, task }
    }

    /// The URL to hand a client, `/mcp` and all.
    pub fn url(&self) -> String {
        format!("http://{}/mcp", self.address)
    }
}

impl Drop for RunningBridge {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch() -> OpsBatch {
        serde_json::from_str(
            r#"{"version":1,"ops":[{"op":"box","min":[0,0,0],"max":[1,1,1],"voxel":{"rgb":[1,2,3]}}]}"#,
        )
        .expect("test batch should parse")
    }

    fn status() -> BridgeStatus {
        BridgeStatus {
            path: None,
            voxel_count: 8,
            undo_depth: 1,
            redo_depth: 0,
            unsaved_changes: true,
            approval: Approval::Auto,
        }
    }

    /// The round trip the whole module exists for: a call goes out, the
    /// editor's side picks it up and answers, the waiting caller gets
    /// that answer.
    #[tokio::test]
    async fn a_call_reaches_the_editor_and_its_answer_comes_back() {
        let (handle, mut receiver) = channel();

        let editor = tokio::task::spawn_blocking(move || {
            // Stand in for the frame loop: poll until something shows up.
            loop {
                if let Some(call) = receiver.try_recv() {
                    assert!(matches!(call.request, AgentRequest::Describe));
                    call.answer(Err(Refusal::new("test", "answered")));
                    return;
                }
                std::thread::yield_now();
            }
        });

        let reply = handle.call(AgentRequest::Describe).await;
        let refusal = reply.err().expect("the editor answered with a refusal");
        assert_eq!(refusal.code, "test");
        editor.await.expect("the editor side should not panic");
    }

    /// An editor that goes away mid-call — the window closed, the bridge
    /// switched off — has to unblock whoever was waiting. The failure
    /// this rules out is a tool call that hangs until the client's own
    /// timeout, with nothing in the answer saying why.
    #[tokio::test]
    async fn a_dropped_call_unblocks_the_agent() {
        let (handle, mut receiver) = channel();

        tokio::task::spawn_blocking(move || loop {
            if let Some(call) = receiver.try_recv() {
                // Dropped without an answer.
                drop(call);
                return;
            }
            std::thread::yield_now();
        });

        let reply = handle.call(AgentRequest::ApplyOps(Box::new(batch()))).await;
        let refusal = reply.err().expect("a dropped call must refuse, not hang");
        assert_eq!(refusal.code, "editor_unavailable");
    }

    /// Same for a receiver that is gone before the call is even sent.
    #[tokio::test]
    async fn a_closed_bridge_refuses_immediately() {
        let (handle, receiver) = channel();
        drop(receiver);

        let reply = handle.call(AgentRequest::Undo).await;
        assert_eq!(
            reply.err().expect("must refuse").code,
            "editor_unavailable"
        );
    }

    /// A parked call knows when nobody is listening any more, which is
    /// what lets the editor clear an abandoned approval off the screen
    /// instead of asking a human to answer a question whose asker left.
    #[tokio::test]
    async fn a_parked_call_notices_the_agent_giving_up() {
        let (handle, mut receiver) = channel();

        let pending = tokio::spawn(async move { handle.call(AgentRequest::Redo).await });
        let call = loop {
            if let Some(call) = receiver.try_recv() {
                break call;
            }
            tokio::task::yield_now().await;
        };
        assert!(!call.abandoned(), "the agent is still waiting");

        pending.abort();
        // Give the aborted task's drop a chance to close the receiver.
        for _ in 0..100 {
            if call.abandoned() {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(call.abandoned(), "a call nobody awaits is abandoned");
    }

    /// The status block rides along with every answer, so it has to
    /// serialize flat into the answer rather than as a nested object an
    /// agent has to dig through.
    #[test]
    fn status_flattens_into_the_answer() {
        let stepped = Stepped {
            stepped: true,
            status: status(),
        };
        let json = serde_json::to_value(&stepped).expect("must serialize");
        assert_eq!(json["stepped"], serde_json::json!(true));
        assert_eq!(json["voxel_count"], serde_json::json!(8));
        assert_eq!(json["approval"], serde_json::json!("auto"));
        assert!(json["path"].is_null(), "an unsaved scene has no path");
    }

    /// An ops failure keeps its own machine-readable code on the way
    /// out, so an agent branches on the same string the CLI gives it.
    #[test]
    fn an_ops_error_keeps_its_code_through_the_bridge() {
        let error = OpsError::new(
            crate::agent_ops::ErrorCode::NoSelection,
            "nothing is selected",
        );
        let refusal: Refusal = error.into();
        assert_eq!(refusal.code, "no_selection");
        let exec: ExecError = refusal.into();
        assert_eq!(exec.code, "no_selection");
        assert!(exec.to_json().contains("\"ok\": false"));
    }
}
