//! Headless agent entry point: run a batch of `agent_ops` against a
//! project file and report what happened, as JSON on stdout.
//!
//! This is the CLI face of [`crate::agent_ops`] — the thing a coding
//! agent that can run shell commands drives:
//!
//! ```text
//! voxelith exec ops.json --out hut.vxlt --describe
//! voxelith exec more.json --in hut.vxlt --out hut.vxlt --export hut.glb
//! voxelith inspect hut.vxlt --slice '{"axis":"y","index":0}'
//! ```
//!
//! Each invocation is a whole session: load (or start empty) → apply →
//! report → save / export. Nothing persists between runs except the
//! `.vxlt`, which is what keeps the CLI honest — the file *is* the
//! state, and a human can open it in the editor at any point.
//!
//! **stdout is JSON and nothing else** (logs go to stderr) so the caller
//! can parse it without stripping banners. Success is
//! `{"ok": true, …}` and exit code 0; failure is `{"ok": false, "error":
//! {"code", "message", "op_index"?}}` and exit code 1. Ops failures pass
//! their [`agent_ops::ErrorCode`](crate::agent_ops::ErrorCode) through as
//! `code`, so an agent branches on one field either way.
//!
//! What this deliberately does *not* do: pivot / up-axis / unit-scale
//! placement, Marching-Cubes smoothing, `gltfpack` compression. Those
//! are `voxelith bake`'s job — it exists to turn finished `.vxlt` files
//! into engine-ready assets, and duplicating its knobs here would give
//! the two commands two sets of export semantics to disagree over.
//! `exec --export` is the interactive File ▸ Export, headless.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::agent_ops::{AgentSession, ApplyReport, Description, OpsBatch, OpsError, SliceRequest};
use crate::editor::Socket;
use crate::io::{self, EditorState, SocketData, SocketNode};
use crate::view::{self, ViewKind};

/// One invocation's inputs. `ops`-less requests are read-only, which is
/// what `voxelith inspect` is.
#[derive(Debug, Clone, Default)]
pub struct ExecRequest {
    /// Ops batch to apply (JSON, see `agent_ops::OpsBatch`).
    pub ops: Option<PathBuf>,
    /// `.vxlt` to start from. Absent means an empty world.
    pub input: Option<PathBuf>,
    /// `.vxlt` to write afterwards.
    pub output: Option<PathBuf>,
    /// Mesh export target; the format comes from the extension
    /// (`.glb` / `.obj` / `.vox`).
    pub export: Option<PathBuf>,
    pub describe: bool,
    /// A `agent_ops::SliceRequest` as JSON.
    pub slice: Option<String>,
    /// Force `dry_run` on regardless of what the batch asked for.
    pub force_dry_run: bool,
}

/// Everything the run produced. Serialized under `{"ok": true, …}`.
#[derive(Debug, Serialize)]
pub struct ExecOutcome {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report: Option<ApplyReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<Description>,
    /// The ASCII slice, one array entry per line. Split rather than sent
    /// as one `\n`-escaped string so it stays readable in the raw
    /// output — an escaped grid is unreadable exactly when someone is
    /// staring at it to work out what went wrong.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slice: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub saved: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exported: Option<ExportInfo>,
}

#[derive(Debug, Serialize)]
pub struct ExportInfo {
    pub path: String,
    pub format: &'static str,
    pub vertices: usize,
    pub triangles: usize,
    pub bytes: u64,
    /// Lossy-export warnings — today, `.vox` palette overflow.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

/// A refused run. `code` is stable and machine-readable: either an
/// `agent_ops` error code (when the batch itself failed) or one of this
/// layer's — `ops_unreadable`, `invalid_ops_json`, `input_unreadable`,
/// `invalid_slice`, `save_failed`, `export_failed`,
/// `unsupported_export_format`, `dry_run_with_output`.
#[derive(Debug, Clone, Serialize)]
pub struct ExecError {
    pub code: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub op_index: Option<usize>,
    pub message: String,
}

impl ExecError {
    pub(crate) fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            op_index: None,
            message: message.into(),
        }
    }
}

impl From<OpsError> for ExecError {
    fn from(error: OpsError) -> Self {
        Self {
            code: error.code.as_str(),
            op_index: error.op_index,
            message: error.message,
        }
    }
}

impl std::fmt::Display for ExecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.op_index {
            Some(i) => write!(f, "op[{}] {}: {}", i, self.code, self.message),
            None => write!(f, "{}: {}", self.code, self.message),
        }
    }
}

impl std::error::Error for ExecError {}

/// `{"ok": true, …outcome}` — the success envelope. Generic so every
/// subcommand's outcome wears the same one; an agent branches on `ok`
/// without caring which command it ran.
#[derive(Serialize)]
struct OkEnvelope<'a, T> {
    ok: bool,
    #[serde(flatten)]
    outcome: &'a T,
}

/// `{"ok": false, "error": {…}}` — the failure envelope.
#[derive(Serialize)]
struct ErrEnvelope<'a> {
    ok: bool,
    error: &'a ExecError,
}

impl ExecOutcome {
    /// The exact bytes the binary prints. Lives here so `main.rs` stays
    /// presentation-free, the same split `bake::BakeOutcome` uses.
    pub fn to_json(&self) -> String {
        let envelope = OkEnvelope {
            ok: true,
            outcome: self,
        };
        serde_json::to_string_pretty(&envelope).expect("the outcome must serialize")
    }
}

impl RenderOutcome {
    /// Same envelope and the same stdout contract as `ExecOutcome` — the
    /// PNGs go to files, the report goes here.
    pub fn to_json(&self) -> String {
        let envelope = OkEnvelope {
            ok: true,
            outcome: self,
        };
        serde_json::to_string_pretty(&envelope).expect("the outcome must serialize")
    }
}

impl ExecError {
    pub fn to_json(&self) -> String {
        let envelope = ErrEnvelope {
            ok: false,
            error: self,
        };
        serde_json::to_string_pretty(&envelope).expect("the error must serialize")
    }
}

/// `{"ok": true, "generators": [...]}` — the catalog a `generate` op
/// picks from, each entry carrying its parameters at their default
/// values. That listing *is* the parameter template: copy it, change
/// what you want, send it back as `params`. Without this on the CLI the
/// only way to learn a generator's parameter names would be a table in
/// the docs, which is the drift the registry exists to avoid.
pub fn generators_json() -> String {
    #[derive(Serialize)]
    struct Catalog {
        ok: bool,
        generators: Vec<crate::agent_ops::GeneratorInfo>,
    }
    let catalog = Catalog {
        ok: true,
        generators: crate::agent_ops::generator_infos(),
    };
    serde_json::to_string_pretty(&catalog).expect("the catalog must serialize")
}

/// Load → apply → describe → save → export, in that order.
///
/// The order matters: reports describe the world as saved, and a failure
/// anywhere stops before anything is written. Ops themselves are already
/// all-or-nothing ([`AgentSession::apply_ops`]), so a refused batch can't
/// leave a half-edited file behind either.
pub fn run_exec(request: &ExecRequest) -> Result<ExecOutcome, ExecError> {
    // Parsed before anything else runs. This command *is* the session:
    // nothing survives it but the files it writes, so a typo in --slice
    // surfacing after the batch had already been applied would throw
    // away work the agent has no way to get back.
    let slice_request = match &request.slice {
        Some(text) => Some(serde_json::from_str::<SliceRequest>(text).map_err(|e| {
            ExecError::new(
                "invalid_slice",
                format!("--slice is not a valid slice request: {e}"),
            )
        })?),
        None => None,
    };

    let (mut session, state) = open_session(request.input.as_deref())?;

    // A dry run reports on a preview rather than on the session, so
    // --describe / --slice show the world the batch *would* leave. The
    // alternative is one envelope holding two contradictory pictures:
    // a report of the world after the batch beside a description of the
    // world it declined to change.
    let mut preview = None;
    let report = match &request.ops {
        Some(path) => {
            let batch = read_batch(path, request.force_dry_run)?;
            if batch.options.dry_run && (request.output.is_some() || request.export.is_some()) {
                // Silently skipping the write would be discovered three
                // steps later, when the agent wonders why its model is
                // empty.
                return Err(ExecError::new(
                    "dry_run_with_output",
                    "a dry run writes nothing, so --out / --export can't be combined with it",
                ));
            }
            Some(if batch.options.dry_run {
                let previewed = session.preview_ops(&batch)?;
                preview = Some(previewed.session);
                previewed.report
            } else {
                session.apply_ops(&batch)?
            })
        }
        None => None,
    };

    // Read-only views follow the preview when there is one; the writes
    // below never do — a dry run can't reach them at all.
    let view = preview.as_ref().unwrap_or(&session);
    let slice = match &slice_request {
        Some(parsed) => Some(view.slice(parsed)?.lines().map(String::from).collect()),
        None => None,
    };
    let description = request.describe.then(|| view.describe());

    Ok(ExecOutcome {
        report,
        description,
        slice,
        saved: match &request.output {
            Some(path) => Some(save_project(&session, state, path)?),
            None => None,
        },
        exported: match &request.export {
            Some(path) => Some(export_mesh(&session, path)?),
            None => None,
        },
    })
}

/// Start a session, from a `.vxlt` or empty.
///
/// The loaded [`EditorState`] rides along untouched so a later save
/// preserves the artist's camera, palette and brush — an agent editing
/// someone's project has no business resetting their workspace.
pub(crate) fn open_session(
    input: Option<&Path>,
) -> Result<(AgentSession, EditorState), ExecError> {
    let Some(path) = input else {
        return Ok((AgentSession::new(), EditorState::default()));
    };
    let (world, state) = io::load_world_with_state(path).map_err(|e| {
        ExecError::new(
            "input_unreadable",
            format!("could not load {}: {e}", path.display()),
        )
    })?;
    let mut session = AgentSession::new();
    session.world = world;
    session.sockets = state
        .sockets
        .iter()
        .map(|socket| Socket::new(socket.name.clone(), socket.position, socket.normal))
        .collect();
    Ok((session, state))
}

fn read_batch(path: &Path, force_dry_run: bool) -> Result<OpsBatch, ExecError> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        ExecError::new(
            "ops_unreadable",
            format!("could not read {}: {e}", path.display()),
        )
    })?;
    let mut batch: OpsBatch = serde_json::from_str(&text).map_err(|e| {
        ExecError::new(
            "invalid_ops_json",
            format!("{} is not a valid ops batch: {e}", path.display()),
        )
    })?;
    if force_dry_run {
        batch.options.dry_run = true;
    }
    Ok(batch)
}

pub(crate) fn save_project(
    session: &AgentSession,
    mut state: EditorState,
    path: &Path,
) -> Result<String, ExecError> {
    // Sockets are rebuilt from the session rather than passed through
    // from the loaded state: no v1 op touches them, but the day one
    // does, this line is already right.
    state.sockets = session
        .sockets
        .iter()
        .map(|socket| SocketData {
            name: socket.name.clone(),
            position: socket.position,
            normal: socket.normal,
        })
        .collect();
    io::save_world_with_state(&session.world, state, path).map_err(|e| {
        ExecError::new(
            "save_failed",
            format!("could not save {}: {e}", path.display()),
        )
    })?;
    Ok(path.display().to_string())
}

/// One `voxelith render` invocation.
#[derive(Debug, Clone)]
pub struct RenderRequest {
    /// `.vxlt` to draw.
    pub project: PathBuf,
    /// Which viewpoints. Empty is rejected rather than defaulted — the
    /// caller decides, and `main` supplies the default.
    pub views: Vec<ViewKind>,
    /// Image edge in pixels; see [`view::MAX_SIZE`].
    pub size: u32,
    /// Where to write. With exactly one view this is the file. With
    /// several — or none given — the images land beside the project as
    /// `<stem>-<view>.png`, since one path can't name six files.
    pub out: Option<PathBuf>,
}

/// What a render run produced, one entry per view.
#[derive(Debug, Serialize)]
pub struct RenderOutcome {
    pub views: Vec<RenderedInfo>,
}

#[derive(Debug, Serialize)]
pub struct RenderedInfo {
    pub view: &'static str,
    pub path: String,
    pub size: u32,
    pub bytes: u64,
    /// What the image covers, so a pixel can be turned back into cells.
    pub framing: view::Framing,
    /// True when the project held no voxels — the image is all
    /// background, and saying so beats letting the agent conclude its
    /// model disappeared.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub empty: bool,
}

/// Turn a `--view` spec (`"iso"`, `"front,top"`, `"all"`) into
/// viewpoints.
///
/// Here rather than in `main` so the binary never constructs an
/// `ExecError`: the stable error codes are this layer's vocabulary, and
/// a second place minting them is how two spellings of the same failure
/// get shipped.
pub fn parse_views(spec: &str) -> Result<Vec<ViewKind>, ExecError> {
    ViewKind::parse_list(spec).map_err(|message| ExecError::new("unknown_view", message))
}

/// Draw one or more views of a project and write them as PNG files.
///
/// The CLI half of [`crate::view`]. Over MCP the same renders come back
/// inline as image content; here they're files, because a shell agent
/// reads a picture by opening it.
pub fn run_render(request: &RenderRequest) -> Result<RenderOutcome, ExecError> {
    if request.views.is_empty() {
        return Err(ExecError::new(
            "no_views",
            "name at least one view: iso, front, back, left, right, top, bottom — or all",
        ));
    }
    let (session, _) = open_session(Some(&request.project))?;

    let single = request.views.len() == 1;
    let mut rendered = Vec::with_capacity(request.views.len());
    for kind in &request.views {
        let view = view::render(&session.world, *kind, request.size)
            .map_err(|e| ExecError::new("invalid_size", e.to_string()))?;
        let path = match (&request.out, single) {
            (Some(out), true) => out.clone(),
            _ => image_path(request.out.as_deref().unwrap_or(&request.project), *kind),
        };
        std::fs::write(&path, &view.png).map_err(|e| {
            ExecError::new(
                "render_failed",
                format!("could not write {}: {e}", path.display()),
            )
        })?;
        rendered.push(RenderedInfo {
            view: kind.as_str(),
            path: path.display().to_string(),
            size: view.size,
            bytes: view.png.len() as u64,
            framing: view.framing,
            empty: view.empty,
        });
    }
    Ok(RenderOutcome { views: rendered })
}

/// `<stem>-<view>.png` next to `base` — the naming that lets a sweep of
/// all seven views land in one directory without colliding.
fn image_path(base: &Path, kind: ViewKind) -> PathBuf {
    let stem = base
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "view".to_string());
    base.with_file_name(format!("{stem}-{}.png", kind.as_str()))
}

pub(crate) fn export_mesh(session: &AgentSession, path: &Path) -> Result<ExportInfo, ExecError> {
    let format = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    let failed = |e: &dyn std::fmt::Display| {
        ExecError::new(
            "export_failed",
            format!("could not export {}: {e}", path.display()),
        )
    };

    let (format, vertices, triangles, notes) = match format.as_str() {
        "glb" => {
            let sockets: Vec<SocketNode> = session
                .sockets
                .iter()
                .map(|socket| SocketNode {
                    name: socket.name.clone(),
                    translation: socket.position,
                    rotation: socket.rotation(),
                })
                .collect();
            let stats =
                io::export_glb(&session.world, &sockets, path).map_err(|e| failed(&e))?;
            ("glb", stats.vertex_count, stats.triangle_count, Vec::new())
        }
        "obj" => {
            let stats = io::export_obj(&session.world, path).map_err(|e| failed(&e))?;
            ("obj", stats.vertex_count, stats.triangle_count, Vec::new())
        }
        "vox" => {
            let mut file = std::fs::File::create(path).map_err(|e| failed(&e))?;
            // Axis conversion on, matching the editor's default: a model
            // exported from here should stand upright in MagicaVoxel.
            let overflow =
                io::export_vox(&session.world, &mut file, true).map_err(|e| failed(&e))?;
            let notes = if overflow > 0 {
                vec![format!(
                    "{overflow} color(s) did not fit the 255-slot .vox palette and were quantized to the nearest entry"
                )]
            } else {
                Vec::new()
            };
            // `.vox` is a voxel format — it has no mesh to count.
            ("vox", 0, 0, notes)
        }
        other => {
            return Err(ExecError::new(
                "unsupported_export_format",
                format!(
                    "don't know how to export {:?} (from {}); supported: .glb, .obj, .vox",
                    other,
                    path.display()
                ),
            ))
        }
    };

    Ok(ExportInfo {
        path: path.display().to_string(),
        format,
        vertices,
        triangles,
        bytes: std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
        notes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Voxel, World};

    /// Each test gets its own directory — the suite runs in parallel.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("voxelith_exec_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_ops(dir: &Path, json: &str) -> PathBuf {
        let path = dir.join("ops.json");
        std::fs::write(&path, json).unwrap();
        path
    }

    const HUT: &str = r#"{"version":1,"ops":[
        {"op":"box","min":[0,0,0],"max":[6,0,6],"voxel":{"rgb":[110,110,110]}},
        {"op":"box","min":[0,1,0],"max":[6,4,6],"voxel":{"rgb":[196,148,90]},"filled":false},
        {"op":"box","min":[2,1,0],"max":[4,3,0],"voxel":"air"}
    ]}"#;

    fn solid_voxels(world: &World) -> usize {
        world
            .chunks()
            .map(|(_, chunk)| chunk.read().solid_count() as usize)
            .sum()
    }

    #[test]
    fn one_command_builds_a_model_saves_it_and_exports_a_glb() {
        // The P1 acceptance path end to end: an agent writes ops, gets a
        // project it can reopen and an asset an engine can import.
        let dir = scratch("build_save_export");
        let project = dir.join("hut.vxlt");
        let asset = dir.join("hut.glb");

        let outcome = run_exec(&ExecRequest {
            ops: Some(write_ops(&dir, HUT)),
            output: Some(project.clone()),
            export: Some(asset.clone()),
            describe: true,
            ..Default::default()
        })
        .expect("the run should succeed");

        let report = outcome.report.as_ref().unwrap();
        assert!(!report.dry_run);
        assert_eq!(report.applied_ops, 3);
        assert!(report.changed_voxels > 0);
        assert_eq!(report.world_aabb.unwrap().max[1], 4, "walls four cells tall");

        let description = outcome.description.as_ref().unwrap();
        assert_eq!(description.voxel_count, report.voxel_count);
        assert!(description.colors.len() >= 2, "floor and walls differ");

        assert_eq!(outcome.saved.as_deref(), Some(project.display().to_string().as_str()));
        let exported = outcome.exported.as_ref().unwrap();
        assert_eq!(exported.format, "glb");
        assert!(exported.triangles > 0 && exported.bytes > 0);
        assert!(asset.exists());

        // The saved project must reopen with exactly what was reported.
        let (reloaded, _) = io::load_world_with_state(&project).unwrap();
        assert_eq!(solid_voxels(&reloaded) as u64, report.voxel_count);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn editing_someone_elses_project_leaves_their_workspace_alone() {
        // Camera, palette and brush belong to whoever was using the
        // editor. An agent adding a voxel must not reset them.
        let dir = scratch("preserves_state");
        let project = dir.join("scene.vxlt");

        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgb(1, 2, 3));
        let state = EditorState {
            camera_position: [12.0, 34.0, 56.0],
            palette: vec![[9, 8, 7, 255]],
            brush_flags: 0b11,
            brush_tint_zone: 2,
            sockets: vec![SocketData {
                name: "muzzle".into(),
                position: [1.5, 2.0, 0.5],
                normal: [0.0, 1.0, 0.0],
            }],
            ..Default::default()
        };
        io::save_world_with_state(&world, state, &project).unwrap();

        run_exec(&ExecRequest {
            ops: Some(write_ops(
                &dir,
                r#"{"version":1,"ops":[{"op":"box","min":[5,0,0],"max":[5,0,0],"voxel":{"rgb":[4,5,6]}}]}"#,
            )),
            input: Some(project.clone()),
            output: Some(project.clone()),
            ..Default::default()
        })
        .expect("the run should succeed");

        let (world, state) = io::load_world_with_state(&project).unwrap();
        assert_eq!(solid_voxels(&world), 2, "the edit landed");
        assert_eq!(state.camera_position, [12.0, 34.0, 56.0]);
        assert_eq!(state.palette, vec![[9, 8, 7, 255]]);
        assert_eq!(state.brush_flags, 0b11);
        assert_eq!(state.brush_tint_zone, 2);
        assert_eq!(state.sockets.len(), 1, "sockets survive a headless round trip");
        assert_eq!(state.sockets[0].name, "muzzle");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_dry_run_reports_without_touching_the_project() {
        let dir = scratch("dry_run");
        let project = dir.join("scene.vxlt");
        io::save_world_with_state(&World::new(), EditorState::default(), &project).unwrap();

        let outcome = run_exec(&ExecRequest {
            ops: Some(write_ops(&dir, HUT)),
            input: Some(project.clone()),
            force_dry_run: true,
            ..Default::default()
        })
        .expect("a dry run should succeed");

        let report = outcome.report.as_ref().unwrap();
        assert!(report.dry_run);
        assert!(report.changed_voxels > 0, "it still reports what would happen");

        let (world, _) = io::load_world_with_state(&project).unwrap();
        assert_eq!(solid_voxels(&world), 0, "the file must be untouched");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn asking_for_a_dry_run_and_an_output_is_refused_rather_than_half_honored() {
        let dir = scratch("dry_run_conflict");
        let error = run_exec(&ExecRequest {
            ops: Some(write_ops(&dir, HUT)),
            output: Some(dir.join("out.vxlt")),
            force_dry_run: true,
            ..Default::default()
        })
        .expect_err("the combination is contradictory");
        assert_eq!(error.code, "dry_run_with_output");
        assert!(!dir.join("out.vxlt").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_refused_batch_names_the_op_and_writes_nothing() {
        let dir = scratch("bad_op");
        let project = dir.join("out.vxlt");
        let error = run_exec(&ExecRequest {
            ops: Some(write_ops(
                &dir,
                r#"{"version":1,"ops":[
                    {"op":"box","min":[0,0,0],"max":[1,1,1],"voxel":{"rgb":[1,2,3]}},
                    {"op":"rotate","axis":"y","quarters":9}
                ]}"#,
            )),
            output: Some(project.clone()),
            ..Default::default()
        })
        .expect_err("quarters 9 is not a rotation");

        assert_eq!(error.code, "invalid_argument");
        assert_eq!(error.op_index, Some(1));
        assert!(!project.exists(), "a failed run must not leave a file behind");

        let envelope: serde_json::Value = serde_json::from_str(&error.to_json()).unwrap();
        assert_eq!(envelope["ok"], serde_json::json!(false));
        assert_eq!(envelope["error"]["op_index"], serde_json::json!(1));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn malformed_input_is_reported_by_kind() {
        let dir = scratch("malformed");
        std::fs::write(dir.join("bad.json"), "{ not json").unwrap();
        let error = run_exec(&ExecRequest {
            ops: Some(dir.join("bad.json")),
            ..Default::default()
        })
        .expect_err("that isn't JSON");
        assert_eq!(error.code, "invalid_ops_json");

        let error = run_exec(&ExecRequest {
            ops: Some(dir.join("missing.json")),
            ..Default::default()
        })
        .expect_err("there is no such file");
        assert_eq!(error.code, "ops_unreadable");

        let error = run_exec(&ExecRequest {
            input: Some(dir.join("nope.vxlt")),
            describe: true,
            ..Default::default()
        })
        .expect_err("there is no such project");
        assert_eq!(error.code, "input_unreadable");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_unknown_export_extension_is_refused_with_the_list() {
        let dir = scratch("bad_export");
        let error = run_exec(&ExecRequest {
            ops: Some(write_ops(&dir, HUT)),
            export: Some(dir.join("hut.fbx")),
            ..Default::default()
        })
        .expect_err("fbx is not an export format");
        assert_eq!(error.code, "unsupported_export_format");
        assert!(error.message.contains(".glb"), "got: {}", error.message);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn inspect_reads_a_project_without_writing_to_it() {
        let dir = scratch("inspect");
        let project = dir.join("scene.vxlt");
        let mut world = World::new();
        for x in 0..3 {
            world.set_voxel(x, 0, 0, Voxel::from_rgb(200, 0, 0));
        }
        io::save_world_with_state(&world, EditorState::default(), &project).unwrap();
        let before = std::fs::read(&project).unwrap();

        // Exactly what `voxelith inspect` builds.
        let outcome = run_exec(&ExecRequest {
            input: Some(project.clone()),
            describe: true,
            slice: Some(r#"{"axis":"y","index":0}"#.into()),
            ..Default::default()
        })
        .expect("inspect should succeed");

        assert!(outcome.report.is_none(), "nothing was applied");
        assert!(outcome.saved.is_none() && outcome.exported.is_none());
        assert_eq!(outcome.description.as_ref().unwrap().voxel_count, 3);
        let slice = outcome.slice.as_ref().unwrap();
        assert!(slice[0].contains("y=0"), "first line is the header: {}", slice[0]);
        assert_eq!(slice[1], "###", "three voxels in a row");
        assert_eq!(std::fs::read(&project).unwrap(), before, "file untouched");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_malformed_slice_request_is_caught_before_the_batch_runs() {
        // Ordering, not just the error code: the batch below would fail
        // on its own, and what comes back must still be the slice's
        // complaint. One invocation is the whole session — a --slice
        // typo discovered *after* a batch applied would throw the work
        // away with nothing written and nothing to resume from.
        let dir = scratch("bad_slice");
        let error = run_exec(&ExecRequest {
            ops: Some(write_ops(
                &dir,
                r#"{"version":1,"ops":[{"op":"rotate","axis":"y","quarters":9}]}"#,
            )),
            output: Some(dir.join("out.vxlt")),
            slice: Some("y=0".into()),
            ..Default::default()
        })
        .expect_err("the slice request must be JSON");
        assert_eq!(error.code, "invalid_slice");
        assert!(!dir.join("out.vxlt").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_dry_run_describes_the_world_the_batch_would_leave() {
        // The report and the description have to be two views of one
        // model. A dry run commits nothing, so describing the *session*
        // put "0 voxels" next to a report of 186 in the same envelope.
        let dir = scratch("dry_run_describe");
        let outcome = run_exec(&ExecRequest {
            ops: Some(write_ops(&dir, HUT)),
            describe: true,
            slice: Some(r#"{"axis":"y","index":0}"#.into()),
            force_dry_run: true,
            ..Default::default()
        })
        .expect("a dry run should succeed");

        let report = outcome.report.as_ref().unwrap();
        let description = outcome.description.as_ref().unwrap();
        assert!(report.dry_run);
        assert!(report.voxel_count > 0);
        assert_eq!(description.voxel_count, report.voxel_count);
        assert_eq!(description.world_aabb, report.world_aabb);
        // The slice shows the floor the batch would lay down, not the
        // empty world it left alone.
        assert_eq!(outcome.slice.as_ref().unwrap()[1], "#######");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_success_envelope_is_flat_and_marked_ok() {
        let dir = scratch("envelope");
        let outcome = run_exec(&ExecRequest {
            ops: Some(write_ops(&dir, HUT)),
            describe: true,
            ..Default::default()
        })
        .unwrap();

        let envelope: serde_json::Value = serde_json::from_str(&outcome.to_json()).unwrap();
        assert_eq!(envelope["ok"], serde_json::json!(true));
        assert!(envelope["report"]["changed_voxels"].is_number());
        assert!(envelope["description"]["voxel_count"].is_number());
        // Absent parts stay out of the envelope rather than showing up
        // as nulls an agent has to special-case.
        assert!(envelope.get("saved").is_none());
        assert!(envelope.get("exported").is_none());
        assert!(envelope.get("error").is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_generator_catalog_is_usable_as_sent() {
        // The round trip an agent makes: read the catalog, copy a
        // generator's params, send them straight back in a batch.
        let catalog: serde_json::Value = serde_json::from_str(&generators_json()).unwrap();
        assert_eq!(catalog["ok"], serde_json::json!(true));
        let first = &catalog["generators"][0];
        let id = first["id"].as_str().unwrap().to_string();
        assert!(first["default_params"].is_object());

        let dir = scratch("catalog");
        let ops = format!(
            r#"{{"version":1,"ops":[{{"op":"generate","generator":"{id}","params":{}}}]}}"#,
            first["default_params"]
        );
        let outcome = run_exec(&ExecRequest {
            ops: Some(write_ops(&dir, &ops)),
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("{id} rejected its own advertised params: {e}"));
        assert!(outcome.report.unwrap().changed_voxels > 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn exporting_vox_reports_palette_loss_instead_of_hiding_it() {
        let dir = scratch("vox_export");
        // 300 distinct colors don't fit .vox's 255 palette slots. Kept
        // in a compact block — .vox tops out at 256 cells per axis.
        let entries: Vec<String> = (0..300)
            .map(|i| {
                format!(
                    "[{},{},0,{{\"rgb\":[{},{},{}]}}]",
                    i % 16,
                    i / 16,
                    i % 256,
                    i / 2,
                    255 - i % 200
                )
            })
            .collect();
        let ops = format!(
            r#"{{"version":1,"ops":[{{"op":"set_voxels","voxels":[{}]}}]}}"#,
            entries.join(",")
        );
        let asset = dir.join("model.vox");
        let outcome = run_exec(&ExecRequest {
            ops: Some(write_ops(&dir, &ops)),
            export: Some(asset.clone()),
            ..Default::default()
        })
        .expect("the export should succeed");

        let exported = outcome.exported.as_ref().unwrap();
        assert_eq!(exported.format, "vox");
        assert!(asset.exists() && exported.bytes > 0);
        assert!(
            exported.notes.iter().any(|n| n.contains("palette")),
            "a lossy export must say so, got: {:?}",
            exported.notes
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
