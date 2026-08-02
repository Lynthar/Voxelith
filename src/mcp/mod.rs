//! The MCP server: the same editing primitives as a resident tool set,
//! for agents that speak the protocol instead of running shell commands.
//!
//! The difference from [`crate::exec`] is the session, not the verbs.
//! `exec` is one process per step and the `.vxlt` on disk *is* the
//! state; here one [`AgentSession`] stays alive across calls, so undo
//! history, the selection and an unsaved document all survive from one
//! tool call to the next. That is why the tools worth having are the
//! session verbs — open / save / undo / redo — rather than more ways to
//! write voxels: the ops batch already covers writing.
//!
//! Two transports over one implementation: stdio for a local agent that
//! launches this as a child process, Streamable HTTP (behind the
//! `mcp-http` feature) for one that wants a URL. Both hand tool calls to
//! the same handler, and both resolve every path through the same
//! [`Root`] — see `paths.rs` for why that rule doesn't vary by
//! transport.
//!
//! **Nothing here may write to stdout.** On the stdio transport stdout
//! *is* the protocol stream, and one stray `println!` corrupts the
//! session. Logs go to stderr.

use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerInfo,
};
use rmcp::{tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler};
use serde::Serialize;

use crate::agent_ops::{AgentSession, ApplyReport, Description, OpsBatch, SliceRequest};
use crate::exec::{self, ExecError, ExportInfo};
use crate::io::EditorState;

mod paths;

pub use paths::{display, PathError, Root};

/// The document the agent is working on, and where it came from.
struct Document {
    session: AgentSession,
    /// Carried along from the loaded `.vxlt` so a save preserves the
    /// artist's camera, palette and brush — an agent editing someone's
    /// project has no business resetting their workspace. Same rule
    /// `exec` follows.
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

/// The server. Cloning shares the document — which the HTTP transport
/// depends on, since it builds a handler per request and they all have
/// to be looking at the same model.
#[derive(Clone)]
pub struct VoxelithMcp {
    document: Arc<Mutex<Document>>,
    root: Arc<Root>,
    /// Built once and pointed at explicitly below. Left to its default,
    /// `#[tool_handler]` calls `Self::tool_router()` on every request,
    /// which regenerates all ten tools' JSON schemas — including the
    /// whole ops union — to answer a single call.
    tool_router: ToolRouter<VoxelithMcp>,
}

/// What the agent sees when a call refuses. The body is the same
/// `{"ok": false, "error": {code, message}}` the CLI prints, so an agent
/// that has driven one knows the other; `is_error` is what tells the
/// client this was a failed call rather than a result to read.
fn refused(error: &ExecError) -> Result<CallToolResult, McpError> {
    Ok(CallToolResult::error(vec![ContentBlock::text(
        error.to_json(),
    )]))
}

/// A success: `{"ok": true, …}`, pretty-printed, as text.
///
/// Text rather than a structured payload because it's the one shape
/// every MCP client renders and every model reads without help.
fn answered<T: Serialize>(body: T) -> Result<CallToolResult, McpError> {
    #[derive(Serialize)]
    struct Envelope<T> {
        ok: bool,
        #[serde(flatten)]
        body: T,
    }
    let text = serde_json::to_string_pretty(&Envelope { ok: true, body })
        .map_err(|e| McpError::internal_error(format!("could not serialize the answer: {e}"), None))?;
    Ok(CallToolResult::success(vec![ContentBlock::text(text)]))
}

/// A path that didn't pass [`Root`], as the same envelope as everything
/// else. `path_refused` is a distinct code from `input_unreadable`: the
/// file may well exist and be perfectly readable — it's just not
/// somewhere this server will touch.
fn path_error(error: PathError) -> ExecError {
    ExecError::new("path_refused", error.to_string())
}

#[tool_router]
impl VoxelithMcp {
    pub fn new(root: Root) -> Self {
        Self {
            document: Arc::new(Mutex::new(Document::empty())),
            root: Arc::new(root),
            tool_router: Self::tool_router(),
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
        // alone. Report and description come from the same world by
        // construction — handing back numbers for the world after the
        // batch beside a description of the world before it is the one
        // way to make "what would this do?" actively misleading.
        let outcome = if batch.options.dry_run {
            document
                .session
                .preview_ops(&batch)
                .map(|preview| (preview.report, preview.session.describe()))
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
        answered(Applied {
            status: document.status(),
            report,
            description,
        })
    }

    #[tool(
        description = "List the generators an op with \"op\": \"generate\" can call, each \
                       with its parameters at their default values. That listing is the \
                       parameter template: copy it, change what you want, send it back."
    )]
    fn list_generators(&self) -> Result<CallToolResult, McpError> {
        answered(Generators {
            generators: crate::agent_ops::generator_infos(),
        })
    }

    #[tool(
        description = "Summarize the current document: voxel and chunk counts, bounding \
                       box, the most common colors, emissive / metallic / tint-zone \
                       tallies, sockets, selection and undo depth."
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
        answered(Stepped {
            stepped: moved,
            status: document.status(),
        })
    }

    #[tool(description = "Redo the batch undone last.")]
    fn redo(&self) -> Result<CallToolResult, McpError> {
        let mut document = self.document.lock();
        let moved = document.session.redo();
        answered(Stepped {
            stepped: moved,
            status: document.status(),
        })
    }

    #[tool(
        description = "Export the current document as a mesh or voxel file. The format \
                       comes from the extension: .glb, .obj or .vox. This is the \
                       interactive File > Export, headless — for engine-ready placement \
                       (pivot, up-axis, unit scale) use the `voxelith bake` command."
    )]
    fn export(
        &self,
        Parameters(args): Parameters<PathArgs>,
    ) -> Result<CallToolResult, McpError> {
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
        // which reads the *rmcp* crate's build environment and so
        // announces every server built on it as "rmcp" — this is the
        // name clients show the user.
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(
                Implementation::new(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
                    .with_title("Voxelith")
                    .with_description("Build and edit voxel models, then export them as game assets."),
            )
            .with_instructions(
                "Voxelith builds voxel models. One document stays open across calls, so \
                 undo, the selection and unsaved edits persist between tools.\n\n\
                 The loop is: apply_ops -> read the report -> slice or describe to check \
                 -> repeat, then save_project and export.\n\n\
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
}

#[derive(Serialize)]
struct Exported {
    exported: ExportInfo,
}

// ---- transports ----

/// Serve on stdin/stdout, the way a local client launches a server as a
/// child process.
pub async fn serve_stdio(root: Root) -> anyhow::Result<()> {
    use rmcp::{transport::stdio, ServiceExt};

    let service = VoxelithMcp::new(root).serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}

/// Serve Streamable HTTP at `/mcp` on `address`.
///
/// The handler factory hands out clones of one server, so every request
/// works on the same document — a fresh one per request would give each
/// call its own empty world and silently lose every edit.
#[cfg(feature = "mcp-http")]
pub async fn serve_http(root: Root, address: std::net::SocketAddr) -> anyhow::Result<()> {
    use rmcp::transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpService, StreamableHttpServerConfig,
    };

    let server = VoxelithMcp::new(root);
    let service = StreamableHttpService::new(
        move || Ok(server.clone()),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );
    let router = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind(address).await?;
    log::info!("MCP Streamable HTTP listening on http://{address}/mcp");
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
        VoxelithMcp::new(Root::new(dir).unwrap())
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
        // Over MCP there is no way to ask for a preview and a
        // description together *except* in one call, so apply_ops has to
        // answer with both from the same world. Describing the session
        // instead would report zero voxels beside a report of hundreds.
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
        assert_eq!(body(&server.describe().unwrap())["voxel_count"], serde_json::json!(0));

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
            server.open_project(Parameters(PathArgs { path: outside.into() })),
            server.save_project(Parameters(SaveArgs { path: Some(outside.into()) })),
            server.export(Parameters(PathArgs { path: outside.into() })),
        ] {
            let result = result.unwrap();
            assert_eq!(result.is_error, Some(true));
            assert_eq!(body(&result)["error"]["code"], serde_json::json!("path_refused"));
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
        io::save_world_with_state(&world, state, &project).unwrap();

        let server = server(&dir);
        let opened = body(&server
            .open_project(Parameters(PathArgs { path: "scene.vxlt".into() }))
            .unwrap());
        assert_eq!(opened["voxel_count"], serde_json::json!(1));

        server.apply_ops(batch(HUT)).unwrap();
        // No path given: it saves back where it came from.
        let saved = body(&server.save_project(Parameters(SaveArgs { path: None })).unwrap());
        assert_eq!(saved["ok"], serde_json::json!(true));

        let (reloaded, state) = io::load_world_with_state(&project).unwrap();
        assert!(reloaded.chunk_count() > 0);
        assert_eq!(state.camera_position, [12.0, 34.0, 56.0], "camera preserved");
        assert_eq!(state.palette, vec![[9, 8, 7, 255]], "palette preserved");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn saving_a_document_that_has_no_path_says_what_to_do() {
        let dir = scratch("no_path");
        let server = server(&dir);
        server.apply_ops(batch(HUT)).unwrap();
        let result = server.save_project(Parameters(SaveArgs { path: None })).unwrap();
        assert_eq!(result.is_error, Some(true));
        assert_eq!(
            body(&result)["error"]["code"],
            serde_json::json!("no_project_path")
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
        for op in ["box", "sphere", "cylinder", "line", "hollow", "mirror_copy", "generate"] {
            assert!(text.contains(op), "the schema should describe the {op} op");
        }
        // The voxel form is hand-written; it must still offer both
        // shapes the deserializer accepts.
        let voxel = serde_json::to_value(schemars::schema_for!(
            crate::agent_ops::VoxelSpec
        ))
        .unwrap();
        let forms = voxel["anyOf"].as_array().expect("two forms");
        assert_eq!(forms.len(), 2);
        assert_eq!(forms[0]["const"], serde_json::json!("air"));
        assert!(
            forms[1]["$ref"].is_string(),
            "the object form must reference SolidVoxel's generated schema, not copy it"
        );
    }
}
