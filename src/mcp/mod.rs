//! The MCP server: one [`AgentSession`] alive across calls, over stdio
//! or Streamable HTTP. **Nothing here may write to stdout** — on stdio
//! that is the protocol stream. Logs go to stderr.

use std::path::PathBuf;
use std::sync::Arc;

use base64::Engine as _;
use parking_lot::Mutex;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerInfo};
use rmcp::{tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler};
use serde::Serialize;

use crate::agent_ops::{AgentSession, ApplyReport, Description, OpsBatch, SliceRequest};
use crate::exec::{self, ExecError, ExportInfo};
use crate::io::EditorState;

/// The viewpoints a render request means: empty defaults to one
/// isometric view and duplicates collapse, since `views` is an open list
/// and every repeat would be another full CPU render.
pub(crate) fn requested_views(views: Vec<crate::view::ViewKind>) -> Vec<crate::view::ViewKind> {
    if views.is_empty() {
        return vec![crate::view::ViewKind::Iso];
    }
    let mut out: Vec<crate::view::ViewKind> = Vec::with_capacity(7);
    for view in views {
        if !out.contains(&view) {
            out.push(view);
        }
    }
    out
}
use crate::view;

pub mod bridge;
#[cfg(feature = "mcp-http")]
pub mod guard;
mod paths;

#[cfg(feature = "mcp-http")]
pub use guard::AccessToken;
pub use paths::{display, PathError, Root};

/// The document the agent is working on, and where it came from.
struct Document {
    session: AgentSession,
    /// Carried from the loaded `.vxlt` so a save preserves the artist's
    /// camera, palette and brush. Same rule `exec` follows.
    state: EditorState,
    /// Where `save_project` writes when it isn't told where. `None`
    /// until the document has been opened from, or saved to, a file.
    path: Option<PathBuf>,
}

impl Document {
    fn empty() -> Self {
        Self {
            session: AgentSession::new(),
            state: EditorState::default(),
            path: None,
        }
    }

    /// The header every tool result carries: what document this is and
    /// how big it currently is. An agent that just called `undo` wants
    /// to see the effect without a second round trip.
    fn status(&self) -> Status {
        Status {
            path: self.path.as_deref().map(paths::display),
            voxel_count: self.session.describe().voxel_count,
            undo_depth: self.session.history.undo_count(),
            redo_depth: self.session.history.redo_count(),
        }
    }
}

#[derive(Serialize)]
struct Status {
    /// `null` for a document that has never been saved.
    path: Option<String>,
    voxel_count: u64,
    undo_depth: usize,
    redo_depth: usize,
}

/// Whether a successful edit is written back to the document's file.
/// Off by default; on, the file tracks the session step by step so the
/// editor can follow it. One writer at a time — hand edits then race.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Checkpoint {
    Off,
    AfterEveryEdit,
}

/// The server. Cloning shares the document — which the HTTP transport
/// depends on, since it builds a handler per request and they all have
/// to be looking at the same model.
#[derive(Clone)]
pub struct VoxelithMcp {
    document: Arc<Mutex<Document>>,
    root: Arc<Root>,
    checkpoint: Checkpoint,
    /// Built once and pointed at explicitly below. Left to its default,
    /// `#[tool_handler]` calls `Self::tool_router()` per request, which
    /// regenerates every tool's schema — the ops union included.
    tool_router: ToolRouter<VoxelithMcp>,
}

/// What the agent sees when a call refuses: the same
/// `{"ok": false, "error": {…}}` body the CLI prints, plus `is_error` to
/// tell the client this was a failure rather than a result.
fn refused(error: &ExecError) -> Result<CallToolResult, McpError> {
    Ok(CallToolResult::error(vec![ContentBlock::text(
        error.to_json(),
    )]))
}

/// A success: `{"ok": true, …}`, pretty-printed as text — the one shape
/// every MCP client renders and every model reads without help.
fn answered<T: Serialize>(body: T) -> Result<CallToolResult, McpError> {
    #[derive(Serialize)]
    struct Envelope<T> {
        ok: bool,
        #[serde(flatten)]
        body: T,
    }
    let text = serde_json::to_string_pretty(&Envelope { ok: true, body }).map_err(|e| {
        McpError::internal_error(format!("could not serialize the answer: {e}"), None)
    })?;
    Ok(CallToolResult::success(vec![ContentBlock::text(text)]))
}

/// A path that didn't pass [`Root`], in the usual envelope.
/// `path_refused` is distinct from `input_unreadable`: the file may
/// exist and be readable, just not somewhere this server will touch.
fn path_error(error: PathError) -> ExecError {
    ExecError::new("path_refused", error.to_string())
}

#[tool_router]
impl VoxelithMcp {
    pub fn new(root: Root, checkpoint: Checkpoint) -> Self {
        Self {
            document: Arc::new(Mutex::new(Document::empty())),
            root: Arc::new(root),
            checkpoint,
            tool_router: Self::tool_router(),
        }
    }

    /// Write the document back to its file after an edit changed it;
    /// `None` when the server runs without checkpoints. A failed write
    /// reports `saved: false` with a reason but does not fail the call.
    fn checkpoint(&self, document: &Document) -> Option<CheckpointReport> {
        if self.checkpoint == Checkpoint::Off {
            return None;
        }
        let Some(path) = document.path.clone() else {
            return Some(CheckpointReport {
                saved: false,
                detail: Some(
                    "this document has no file yet — save_project once and later edits \
                     check-point themselves"
                        .to_string(),
                ),
            });
        };
        match exec::save_project(&document.session, document.state.clone(), &path) {
            Ok(_) => Some(CheckpointReport {
                saved: true,
                detail: None,
            }),
            Err(e) => {
                log::warn!("checkpoint to {} failed: {e}", display(&path));
                Some(CheckpointReport {
                    saved: false,
                    detail: Some(format!(
                        "the edit stands, but the file is now behind the session: {e}"
                    )),
                })
            }
        }
    }

    #[tool(
        description = "Discard the current document and start from an empty world. \
                       Unsaved changes are lost."
    )]
    fn new_project(&self) -> Result<CallToolResult, McpError> {
        let mut document = self.document.lock();
        *document = Document::empty();
        answered(document.status())
    }

    #[tool(
        description = "Load a .vxlt project as the current document, replacing whatever \
                       was open. Returns a summary of what was loaded."
    )]
    fn open_project(
        &self,
        Parameters(args): Parameters<PathArgs>,
    ) -> Result<CallToolResult, McpError> {
        let path = match self.root.resolve(&args.path) {
            Ok(path) => path,
            Err(e) => return refused(&path_error(e)),
        };
        let (session, state) = match exec::open_session(Some(&path)) {
            Ok(loaded) => loaded,
            Err(e) => return refused(&e),
        };
        let mut document = self.document.lock();
        *document = Document {
            session,
            state,
            path: Some(path),
        };
        answered(Opened {
            status: document.status(),
            description: document.session.describe(),
        })
    }

    #[tool(
        description = "Write the current document to a .vxlt file. With no path, saves \
                       back where it was opened from."
    )]
    fn save_project(
        &self,
        Parameters(args): Parameters<SaveArgs>,
    ) -> Result<CallToolResult, McpError> {
        let mut document = self.document.lock();
        let path = match (args.path, document.path.clone()) {
            (Some(requested), _) => match self.root.resolve(&requested) {
                Ok(path) => path,
                Err(e) => return refused(&path_error(e)),
            },
            (None, Some(current)) => current,
            (None, None) => {
                return refused(&ExecError::new(
                    "no_project_path",
                    "this document has never been saved, so there is nowhere to save it \
                     back to — pass a path",
                ))
            }
        };
        // The state is cloned rather than moved because a failed save
        // must leave the document exactly as it was, still saveable.
        if let Err(e) = exec::save_project(&document.session, document.state.clone(), &path) {
            return refused(&e);
        }
        document.path = Some(path);
        answered(document.status())
    }

    #[tool(
        description = "Apply a batch of edit operations to the current document. Atomic \
                       (any failure changes nothing), sequential (each op sees the \
                       previous ones), and one undo entry for the whole batch. Set \
                       options.dry_run to preview: the report and description then \
                       describe the world the batch *would* produce, and nothing is \
                       committed."
    )]
    fn apply_ops(
        &self,
        Parameters(batch): Parameters<OpsBatch>,
    ) -> Result<CallToolResult, McpError> {
        let mut document = self.document.lock();
        // A dry run reports on the preview, not on the session it left
        // alone, so the report and the description come from the same
        // world by construction.
        let outcome = if batch.options.dry_run {
            // The preview's history is empty and its depths are not the
            // ones `undo` would act on. The history genuinely did not
            // move, so the session's depths are the true ones.
            let depths = (
                document.session.history.undo_count(),
                document.session.history.redo_count(),
            );
            document.session.preview_ops(&batch).map(|preview| {
                let mut description = preview.session.describe();
                description.undo_depth = depths.0;
                description.redo_depth = depths.1;
                (preview.report, description)
            })
        } else {
            document
                .session
                .apply_ops(&batch)
                .map(|report| (report, document.session.describe()))
        };
        let (report, description) = match outcome {
            Ok(outcome) => outcome,
            Err(e) => return refused(&e.into()),
        };
        // A dry run changed nothing, so there is nothing to check-point:
        // writing the unchanged document back would give the human's
        // editor a reload that shows exactly what it already shows.
        let checkpoint = match batch.options.dry_run {
            true => None,
            false => self.checkpoint(&document),
        };
        answered(Applied {
            status: document.status(),
            report,
            description,
            checkpoint,
        })
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
        answered(Generators {
            generators: crate::agent_ops::generator_infos(),
            graph_template: crate::agent_ops::graph_template(),
        })
    }

    #[tool(
        description = "Summarize the current document: voxel and chunk counts, bounding \
                       box, the most common colors, emissive / metallic / tint-zone \
                       tallies, sockets, selection and undo depth. Also measures the \
                       shape itself — connected components, floating parts, enclosed \
                       voxels, per-axis symmetry — which a rendered view cannot tell you \
                       reliably, and returns the document's pipeline graph if it has one."
    )]
    fn describe(&self) -> Result<CallToolResult, McpError> {
        let document = self.document.lock();
        answered(Described {
            status: document.status(),
            description: document.session.describe(),
        })
    }

    #[tool(
        description = "Render one axis-aligned plane of the document as ASCII art — the \
                       cheapest way to catch \"the door is one cell too high\". The first \
                       line states the axis ranges and row order."
    )]
    fn slice(
        &self,
        Parameters(request): Parameters<SliceRequest>,
    ) -> Result<CallToolResult, McpError> {
        let document = self.document.lock();
        match document.session.slice(&request) {
            Ok(text) => answered(Sliced {
                slice: text.lines().map(String::from).collect(),
            }),
            Err(e) => refused(&e.into()),
        }
    }

    #[tool(description = "Undo the last applied batch.")]
    fn undo(&self) -> Result<CallToolResult, McpError> {
        let mut document = self.document.lock();
        let moved = document.session.undo();
        // Stepping the history is an edit as far as the file is
        // concerned. Leaving undo out would drift the checkpoint on the
        // one move an agent makes because the last one was wrong.
        let checkpoint = moved.then(|| self.checkpoint(&document)).flatten();
        answered(Stepped {
            stepped: moved,
            status: document.status(),
            checkpoint,
        })
    }

    #[tool(description = "Redo the batch undone last.")]
    fn redo(&self) -> Result<CallToolResult, McpError> {
        let mut document = self.document.lock();
        let moved = document.session.redo();
        let checkpoint = moved.then(|| self.checkpoint(&document)).flatten();
        answered(Stepped {
            stepped: moved,
            status: document.status(),
            checkpoint,
        })
    }

    #[tool(
        description = "Look at the current document: renders it as images and returns \
                       them inline. Defaults to one isometric view, which shows all \
                       three dimensions at once; ask for axis views (front, back, left, \
                       right, top, bottom, or all) when you need to check one side \
                       square-on. 256 pixels reads a silhouette and its colors fine — a \
                       seven-view sweep at 1024 costs a lot of tokens for detail a voxel \
                       model rarely has. Each image comes with the cell bounds it covers \
                       and the cells-per-pixel, so a measurement off the picture converts \
                       back to coordinates."
    )]
    fn render_views(
        &self,
        Parameters(args): Parameters<RenderArgs>,
    ) -> Result<CallToolResult, McpError> {
        let size = args.size.unwrap_or(view::DEFAULT_SIZE);
        let views = requested_views(args.views);
        let document = self.document.lock();

        // One text block then one image per view: the caption has to sit
        // next to the image it describes, or six views come back as six
        // pictures an agent has to count to identify.
        let mut blocks = Vec::with_capacity(views.len() * 2);
        for kind in views {
            let view = match view::render(&document.session.world, kind, size) {
                Ok(view) => view,
                Err(e) => return refused(&ExecError::new("invalid_size", e.to_string())),
            };
            let caption = serde_json::to_string_pretty(&Rendered {
                view: kind.as_str(),
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

    #[tool(
        description = "Export the current document as a mesh or voxel file. The format \
                       comes from the extension: .glb, .obj or .vox. This is the \
                       interactive File > Export, headless — for engine-ready placement \
                       (pivot, up-axis, unit scale) use the `voxelith bake` command."
    )]
    fn export(&self, Parameters(args): Parameters<PathArgs>) -> Result<CallToolResult, McpError> {
        let path = match self.root.resolve(&args.path) {
            Ok(path) => path,
            Err(e) => return refused(&path_error(e)),
        };
        let document = self.document.lock();
        match exec::export_mesh(&document.session, &path) {
            Ok(mut exported) => {
                // The canonical path is what got written; the readable
                // one is what the agent should see.
                exported.path = paths::display(&path);
                answered(Exported { exported })
            }
            Err(e) => refused(&e),
        }
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for VoxelithMcp {
    fn get_info(&self) -> ServerInfo {
        // Spelled out rather than `Implementation::from_build_env()`,
        // which reads rmcp's own build environment and would announce
        // this server as "rmcp" — the name clients show the user.
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(
                Implementation::new(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
                    .with_title("Voxelith")
                    .with_description(
                        "Build and edit voxel models, then export them as game assets.",
                    ),
            )
            .with_instructions(
                "Voxelith builds voxel models. One document stays open across calls, so \
                 undo, the selection and unsaved edits persist between tools.\n\n\
                 The loop is: apply_ops -> render_views to see what you built -> fix it \
                 -> repeat, then save_project and export. describe and slice answer the \
                 questions a picture can't: exact counts, exact coordinates.\n\n\
                 Coordinates are integer cells, Y is up, and every region is inclusive on \
                 both ends: min [0,0,0] max [1,1,1] is 8 cells. There is no separate \
                 erase, paint or fill op — \"voxel\": \"air\" erases, write_mode \
                 \"only_solid\" repaints without changing the silhouette, and a filled box \
                 fills. Unknown fields are refused rather than ignored, so a misspelling \
                 is reported instead of silently doing nothing.\n\n\
                 Every path resolves inside this server's root directory."
                    .to_string(),
            )
    }
}

// ---- tool arguments ----

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct PathArgs {
    /// Path to the file, relative to the server root or absolute inside
    /// it.
    pub path: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RenderArgs {
    /// Which viewpoints to draw. Empty means one isometric view.
    #[serde(default)]
    pub views: Vec<view::ViewKind>,
    /// Image edge in pixels, 1..=1024. Omit for 256.
    #[serde(default)]
    pub size: Option<u32>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SaveArgs {
    /// Where to write. Omit to save back where the document was opened
    /// from.
    #[serde(default)]
    pub path: Option<String>,
}

// ---- tool answers ----

#[derive(Serialize)]
struct Opened {
    #[serde(flatten)]
    status: Status,
    description: Description,
}

#[derive(Serialize)]
struct Applied {
    #[serde(flatten)]
    status: Status,
    report: ApplyReport,
    description: Description,
    #[serde(skip_serializing_if = "Option::is_none")]
    checkpoint: Option<CheckpointReport>,
}

/// What the automatic write-back did; absent unless the server runs
/// with checkpoints. `saved: false` is not a failed call — it says the
/// file on disk is no longer the session.
#[derive(Serialize)]
struct CheckpointReport {
    saved: bool,
    /// Why not, when `saved` is false.
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
}

#[derive(Serialize)]
struct Described {
    #[serde(flatten)]
    status: Status,
    description: Description,
}

#[derive(Serialize)]
struct Generators {
    generators: Vec<crate::agent_ops::GeneratorInfo>,
    /// A working pipeline graph to copy — the only place the graph
    /// format is spelled out, since the `graph` op's schema keeps it an
    /// opaque object rather than costing every turn nine definitions.
    graph_template: serde_json::Value,
}

#[derive(Serialize)]
struct Sliced {
    slice: Vec<String>,
}

#[derive(Serialize)]
struct Stepped {
    /// False when there was nothing left on the stack — not an error,
    /// just the end of the history.
    stepped: bool,
    #[serde(flatten)]
    status: Status,
    #[serde(skip_serializing_if = "Option::is_none")]
    checkpoint: Option<CheckpointReport>,
}

#[derive(Serialize)]
struct Exported {
    exported: ExportInfo,
}

/// The caption printed beside each rendered image: which view it is and
/// what it covers, so a distance measured in pixels converts back to
/// cells.
#[derive(Serialize)]
struct Rendered<'a> {
    view: &'static str,
    size: u32,
    framing: &'a view::Framing,
    /// The document held no voxels — the image is all background.
    empty: bool,
    /// Rays ran out of steps before crossing the scene, so some of the
    /// background is the walk giving up. Serialized only when it
    /// happened — a flag that is always false teaches agents to skip it.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    truncated: bool,
}

// ---- transports ----

/// Serve on stdin/stdout, the way a local client launches a server as a
/// child process.
pub async fn serve_stdio(root: Root, checkpoint: Checkpoint) -> anyhow::Result<()> {
    use rmcp::{transport::stdio, ServiceExt};

    let service = VoxelithMcp::new(root, checkpoint).serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}

/// Serve Streamable HTTP at `/mcp` on `address`, behind `token`. The
/// factory hands out clones of one server, so every request works on the
/// same document; every request goes through [`guard`] first.
#[cfg(feature = "mcp-http")]
pub async fn serve_http(
    root: Root,
    address: std::net::SocketAddr,
    checkpoint: Checkpoint,
    token: guard::AccessToken,
) -> anyhow::Result<()> {
    use rmcp::transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
    };

    let server = VoxelithMcp::new(root, checkpoint);
    let service = StreamableHttpService::new(
        move || Ok(server.clone()),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );
    let router = guard::guarded(
        axum::Router::new().nest_service("/mcp", service),
        token.clone(),
    );
    let listener = tokio::net::TcpListener::bind(address).await?;
    let url = format!("http://{address}/mcp");
    // The token covers who may call, not what the traffic crosses:
    // there is no TLS here, so off loopback the token and every voxel
    // are in clear text. Honor the address, but never silently.
    if !address.ip().is_loopback() {
        log::warn!(
            "MCP HTTP is bound to {address}, which is not a loopback address. \
             Requests are authenticated but NOT encrypted: put a TLS-terminating \
             reverse proxy in front, or bind 127.0.0.1 instead. Anything holding \
             the token can edit every project under --root."
        );
    }
    log::info!("MCP Streamable HTTP listening on {url}");
    // The token goes out as a line the user can paste rather than a
    // value they assemble a header from. stderr like every other log —
    // stdout belongs to the protocol, and the rule doesn't vary.
    log::info!("Client setup: {}", guard::client_command(&url, &token));
    axum::serve(listener, router).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Voxel, World};
    use crate::io;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("voxelith_mcp_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn server(dir: &std::path::Path) -> VoxelithMcp {
        VoxelithMcp::new(Root::new(dir).unwrap(), Checkpoint::Off)
    }

    fn checkpointing_server(dir: &std::path::Path) -> VoxelithMcp {
        VoxelithMcp::new(Root::new(dir).unwrap(), Checkpoint::AfterEveryEdit)
    }

    /// How many voxels the `.vxlt` on disk holds — what a human with
    /// that project open in the editor would be looking at.
    fn voxels_on_disk(path: &std::path::Path) -> u64 {
        let (session, _) = exec::open_session(Some(path)).expect("the file should load");
        session.describe().voxel_count
    }

    /// Tool results are JSON text; this is what an agent parses.
    fn body(result: &CallToolResult) -> serde_json::Value {
        let text = match result.content.first() {
            Some(ContentBlock::Text(text)) => text.text.clone(),
            other => panic!("expected one text block, got {other:?}"),
        };
        serde_json::from_str(&text).expect("tool results must be JSON")
    }

    const HUT: &str = r#"{"version":1,"ops":[
        {"op":"box","min":[0,0,0],"max":[6,0,6],"voxel":{"rgb":[110,110,110]}},
        {"op":"box","min":[0,1,0],"max":[6,4,6],"voxel":{"rgb":[196,148,90]},"filled":false}
    ]}"#;

    fn batch(json: &str) -> Parameters<OpsBatch> {
        Parameters(serde_json::from_str(json).expect("test batch should parse"))
    }

    #[test]
    fn the_document_survives_between_calls() {
        // The whole reason this exists next to the CLI: state is
        // resident, so a second batch builds on the first and undo
        // reaches back through both.
        let dir = scratch("resident");
        let server = server(&dir);

        let first = body(&server.apply_ops(batch(HUT)).unwrap());
        assert_eq!(first["ok"], serde_json::json!(true));
        let after_first = first["voxel_count"].as_u64().unwrap();
        assert!(after_first > 0);

        let second = body(&server
            .apply_ops(batch(
                r#"{"version":1,"ops":[{"op":"box","min":[2,5,2],"max":[4,5,4],"voxel":{"rgb":[9,9,9]}}]}"#,
            ))
            .unwrap());
        assert!(second["voxel_count"].as_u64().unwrap() > after_first);
        assert_eq!(second["undo_depth"], serde_json::json!(2));

        let undone = body(&server.undo().unwrap());
        assert_eq!(undone["stepped"], serde_json::json!(true));
        assert_eq!(undone["voxel_count"].as_u64().unwrap(), after_first);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_dry_run_describes_the_world_it_would_produce() {
        // Over MCP a preview and a description can only be asked for in
        // one call, so `apply_ops` answers with both from the same
        // world. The session would report zero voxels beside hundreds.
        let dir = scratch("dry_run");
        let server = server(&dir);
        let dry = HUT.replace(r#""ops""#, r#""options":{"dry_run":true},"ops""#);

        let result = body(&server.apply_ops(batch(&dry)).unwrap());
        assert_eq!(result["report"]["dry_run"], serde_json::json!(true));
        assert_eq!(
            result["report"]["voxel_count"],
            result["description"]["voxel_count"]
        );
        assert!(result["report"]["voxel_count"].as_u64().unwrap() > 0);
        // …and the document itself is untouched.
        assert_eq!(result["voxel_count"], serde_json::json!(0));
        assert_eq!(
            body(&server.describe().unwrap())["voxel_count"],
            serde_json::json!(0)
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_refused_batch_is_an_error_result_the_agent_can_read() {
        let dir = scratch("refused");
        let server = server(&dir);
        let result = server
            .apply_ops(batch(
                r#"{"version":1,"ops":[{"op":"rotate","axis":"y","quarters":9}]}"#,
            ))
            .unwrap();
        assert_eq!(result.is_error, Some(true), "a failed call must say so");
        let body = body(&result);
        assert_eq!(body["ok"], serde_json::json!(false));
        assert_eq!(body["error"]["code"], serde_json::json!("invalid_argument"));
        assert_eq!(body["error"]["op_index"], serde_json::json!(0));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_path_outside_the_root_is_refused_by_every_tool_that_takes_one() {
        // stdio makes this near-symbolic, but the same tool bodies serve
        // HTTP, where the caller is whoever reached the port.
        let dir = scratch("confined");
        let server = server(&dir);
        let outside = "../voxelith_mcp_confined_elsewhere.vxlt";

        for result in [
            server.open_project(Parameters(PathArgs {
                path: outside.into(),
            })),
            server.save_project(Parameters(SaveArgs {
                path: Some(outside.into()),
            })),
            server.export(Parameters(PathArgs {
                path: outside.into(),
            })),
        ] {
            let result = result.unwrap();
            assert_eq!(result.is_error, Some(true));
            assert_eq!(
                body(&result)["error"]["code"],
                serde_json::json!("path_refused")
            );
        }
        assert!(!dir.join("../voxelith_mcp_confined_elsewhere.vxlt").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_edit_save_round_trips_and_keeps_the_artists_workspace() {
        let dir = scratch("round_trip");
        let project = dir.join("scene.vxlt");
        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgb(1, 2, 3));
        let state = EditorState {
            camera_position: [12.0, 34.0, 56.0],
            palette: vec![[9, 8, 7, 255]],
            ..Default::default()
        };
        io::save_world_with_state(&world, state, Default::default(), &project).unwrap();

        let server = server(&dir);
        let opened = body(
            &server
                .open_project(Parameters(PathArgs {
                    path: "scene.vxlt".into(),
                }))
                .unwrap(),
        );
        assert_eq!(opened["voxel_count"], serde_json::json!(1));

        server.apply_ops(batch(HUT)).unwrap();
        // No path given: it saves back where it came from.
        let saved = body(
            &server
                .save_project(Parameters(SaveArgs { path: None }))
                .unwrap(),
        );
        assert_eq!(saved["ok"], serde_json::json!(true));

        let (reloaded, state, _) = io::load_world_with_state(&project).unwrap();
        assert!(reloaded.chunk_count() > 0);
        assert_eq!(
            state.camera_position,
            [12.0, 34.0, 56.0],
            "camera preserved"
        );
        assert_eq!(state.palette, vec![[9, 8, 7, 255]], "palette preserved");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn saving_a_document_that_has_no_path_says_what_to_do() {
        let dir = scratch("no_path");
        let server = server(&dir);
        server.apply_ops(batch(HUT)).unwrap();
        let result = server
            .save_project(Parameters(SaveArgs { path: None }))
            .unwrap();
        assert_eq!(result.is_error, Some(true));
        assert_eq!(
            body(&result)["error"]["code"],
            serde_json::json!("no_project_path")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn check_pointing_follows_the_session_edit_by_edit() {
        // The point of the feature: the file keeps up with the agent
        // without the agent remembering to save. Undo counts as an edit,
        // so the file goes back with the session rather than ahead.
        let dir = scratch("checkpoint");
        let project = dir.join("scene.vxlt");
        io::save_world_with_state(
            &World::new(),
            EditorState::default(),
            Default::default(),
            &project,
        )
        .unwrap();

        let server = checkpointing_server(&dir);
        server
            .open_project(Parameters(PathArgs {
                path: "scene.vxlt".into(),
            }))
            .unwrap();

        let applied = body(&server.apply_ops(batch(HUT)).unwrap());
        assert_eq!(applied["checkpoint"]["saved"], serde_json::json!(true));
        let after_apply = applied["voxel_count"].as_u64().unwrap();
        assert!(after_apply > 0);
        assert_eq!(
            voxels_on_disk(&project),
            after_apply,
            "the file should hold what the session holds, with no save call"
        );

        let undone = body(&server.undo().unwrap());
        assert_eq!(undone["checkpoint"]["saved"], serde_json::json!(true));
        assert_eq!(voxels_on_disk(&project), 0, "undo must reach the file too");

        // Nothing changed, so nothing is written and nothing is claimed.
        let spent = body(&server.undo().unwrap());
        assert_eq!(spent["stepped"], serde_json::json!(false));
        assert!(spent.get("checkpoint").is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_dry_run_writes_nothing_even_with_check_pointing_on() {
        // The world didn't change, so the file must not be rewritten —
        // otherwise "preview this" makes the human's editor reload.
        let dir = scratch("checkpoint_dry");
        let project = dir.join("scene.vxlt");
        io::save_world_with_state(
            &World::new(),
            EditorState::default(),
            Default::default(),
            &project,
        )
        .unwrap();

        let server = checkpointing_server(&dir);
        server
            .open_project(Parameters(PathArgs {
                path: "scene.vxlt".into(),
            }))
            .unwrap();
        let dry = HUT.replace(r#""ops""#, r#""options":{"dry_run":true},"ops""#);

        let result = body(&server.apply_ops(batch(&dry)).unwrap());
        assert!(result["report"]["voxel_count"].as_u64().unwrap() > 0);
        assert!(result.get("checkpoint").is_none());
        assert_eq!(voxels_on_disk(&project), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_document_with_no_file_reports_the_checkpoint_it_could_not_write() {
        // Silence here would read as "check-pointed" to an agent that
        // asked for check-pointing, and the human would sit watching a
        // file nobody is writing.
        let dir = scratch("checkpoint_no_path");
        let server = checkpointing_server(&dir);

        let applied = body(&server.apply_ops(batch(HUT)).unwrap());
        assert_eq!(
            applied["ok"],
            serde_json::json!(true),
            "the edit still stands"
        );
        assert_eq!(applied["checkpoint"]["saved"], serde_json::json!(false));
        assert!(applied["checkpoint"]["detail"]
            .as_str()
            .expect("a reason the agent can act on")
            .contains("save_project"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn without_the_flag_nothing_is_written_and_the_answer_stays_quiet() {
        let dir = scratch("checkpoint_off");
        let project = dir.join("scene.vxlt");
        io::save_world_with_state(
            &World::new(),
            EditorState::default(),
            Default::default(),
            &project,
        )
        .unwrap();

        let server = server(&dir);
        server
            .open_project(Parameters(PathArgs {
                path: "scene.vxlt".into(),
            }))
            .unwrap();
        let applied = body(&server.apply_ops(batch(HUT)).unwrap());

        assert!(
            applied.get("checkpoint").is_none(),
            "a plain server must not add a field an agent has to reason about"
        );
        assert_eq!(
            voxels_on_disk(&project),
            0,
            "the file is the agent's to save"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_render_comes_back_as_an_image_beside_the_numbers_to_read_it_with() {
        // The agent's eye. A picture alone can show that the door is too
        // high; the framing beside it is what turns that into "by one
        // cell", so the two travel together.
        let dir = scratch("render");
        let server = server(&dir);
        server.apply_ops(batch(HUT)).unwrap();

        let result = server
            .render_views(Parameters(RenderArgs {
                views: vec![view::ViewKind::Front, view::ViewKind::Top],
                size: Some(64),
            }))
            .unwrap();

        // Caption, image, caption, image — in the order asked for.
        assert_eq!(
            result.content.len(),
            4,
            "one caption and one image per view"
        );
        for (at, expected) in [(0, "front"), (2, "top")] {
            let caption: serde_json::Value = match &result.content[at] {
                ContentBlock::Text(text) => serde_json::from_str(&text.text).unwrap(),
                other => panic!("expected a caption, got {other:?}"),
            };
            assert_eq!(caption["view"], serde_json::json!(expected));
            assert_eq!(caption["size"], serde_json::json!(64));
            assert!(caption["framing"]["cells_per_pixel"].as_f64().unwrap() > 0.0);
            assert!(caption["framing"]["bounds"].is_array());

            let image = match &result.content[at + 1] {
                ContentBlock::Image(image) => image,
                other => panic!("expected an image, got {other:?}"),
            };
            assert_eq!(image.mime_type, "image/png");
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(&image.data)
                .expect("the payload must be base64");
            assert_eq!(&bytes[1..4], b"PNG", "and it must actually be a PNG");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rendering_defaults_to_one_isometric_view() {
        // The default has to be the view that shows all three dimensions
        // — an agent that just says "show me" should not get a
        // projection that hides a whole axis.
        let dir = scratch("render_default");
        let server = server(&dir);
        server.apply_ops(batch(HUT)).unwrap();

        let result = server
            .render_views(Parameters(RenderArgs {
                views: Vec::new(),
                size: None,
            }))
            .unwrap();
        assert_eq!(result.content.len(), 2);
        let caption: serde_json::Value = match &result.content[0] {
            ContentBlock::Text(text) => serde_json::from_str(&text.text).unwrap(),
            other => panic!("expected a caption, got {other:?}"),
        };
        assert_eq!(caption["view"], serde_json::json!("iso"));
        assert_eq!(caption["size"], serde_json::json!(view::DEFAULT_SIZE));
        assert_eq!(caption["empty"], serde_json::json!(false));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_empty_document_renders_but_says_it_was_empty() {
        let dir = scratch("render_empty");
        let server = server(&dir);
        let result = server
            .render_views(Parameters(RenderArgs {
                views: Vec::new(),
                size: Some(32),
            }))
            .unwrap();
        let caption: serde_json::Value = match &result.content[0] {
            ContentBlock::Text(text) => serde_json::from_str(&text.text).unwrap(),
            other => panic!("expected a caption, got {other:?}"),
        };
        assert_eq!(caption["empty"], serde_json::json!(true));
        assert!(caption["framing"]["bounds"].is_null());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_impossible_size_is_refused_with_a_readable_reason() {
        let dir = scratch("render_size");
        let server = server(&dir);
        let result = server
            .render_views(Parameters(RenderArgs {
                views: Vec::new(),
                size: Some(view::MAX_SIZE + 1),
            }))
            .unwrap();
        assert_eq!(result.is_error, Some(true));
        assert_eq!(
            body(&result)["error"]["code"],
            serde_json::json!("invalid_size")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_ops_schema_the_tools_advertise_is_generated_from_the_ops_types() {
        // The reason `agent_ops` carries JsonSchema derives at all: over
        // MCP this schema is the *only* place an agent can learn the ops
        // format. A hand-written copy would start drifting immediately.
        let schema = serde_json::to_value(schemars::schema_for!(OpsBatch)).unwrap();
        let ops = &schema["properties"]["ops"];
        assert!(ops.is_object(), "the batch must advertise its ops array");

        let text = schema.to_string();
        for op in [
            "box",
            "sphere",
            "cylinder",
            "line",
            "hollow",
            "mirror_copy",
            "generate",
        ] {
            assert!(text.contains(op), "the schema should describe the {op} op");
        }
        // The voxel form is hand-written; it must still offer both
        // shapes the deserializer accepts.
        let voxel =
            serde_json::to_value(schemars::schema_for!(crate::agent_ops::VoxelSpec)).unwrap();
        let forms = voxel["anyOf"].as_array().expect("two forms");
        assert_eq!(forms.len(), 2);
        assert_eq!(forms[0]["const"], serde_json::json!("air"));
        assert!(
            forms[1]["$ref"].is_string(),
            "the object form must reference SolidVoxel's generated schema, not copy it"
        );
    }
}
