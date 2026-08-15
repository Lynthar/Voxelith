//! Voxelith - Procedural-first voxel asset creation tool
//!
//! Entry point. With no subcommand this launches the interactive editor
//! (the `app` module). Every subcommand is headless — no window, no GPU:
//! `bake` batch-exports from a spec file ([`crate::bake`]), while `exec` /
//! `inspect` / `generators` drive the editing primitives from JSON
//! ([`crate::exec`]).

#[cfg(feature = "gui")]
mod app;

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
#[cfg(feature = "gui")]
use winit::event_loop::{ControlFlow, EventLoop};

#[derive(Parser)]
#[command(name = "voxelith", version, about = "Procedural-first voxel asset creation tool")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Start the editor with its agent bridge already listening on this
    /// loopback port, instead of waiting for the Agent panel's Start
    /// button. `0` asks the OS for a free port. Editor only — it has no
    /// meaning alongside a subcommand.
    #[cfg(feature = "gui")]
    #[arg(long, value_name = "PORT")]
    agent_port: Option<u16>,
}

#[derive(Subcommand)]
enum Commands {
    /// Batch-export `.vxlt` models to optimized `.glb` from a spec file
    /// (headless — opens no window).
    Bake {
        /// Path to the bake spec (.json).
        spec: PathBuf,
        /// Process only shard i of n, for CI fan-out, e.g. `--shard 0/4`.
        #[arg(long)]
        shard: Option<String>,
    },

    /// Apply an agent ops batch to a project (headless). Prints a JSON
    /// report on stdout.
    Exec {
        /// Ops batch to apply (.json).
        ops: PathBuf,
        /// Project to start from; omit to start from an empty world.
        #[arg(long = "in")]
        input: Option<PathBuf>,
        /// Write the resulting project here (`.vxlt`).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Also export a mesh; the format comes from the extension
        /// (`.glb` / `.obj` / `.vox`).
        #[arg(long)]
        export: Option<PathBuf>,
        /// Include a summary of the resulting model in the report.
        #[arg(long)]
        describe: bool,
        /// Include one plane as ASCII art. Takes a JSON slice request,
        /// e.g. `--slice "{\"axis\":\"y\",\"index\":0}"`.
        #[arg(long)]
        slice: Option<String>,
        /// Report what the batch would do without writing anything.
        #[arg(long = "dry-run")]
        dry_run: bool,
    },

    /// List the generators a `generate` op can call, each with its
    /// parameters at their default values (the template to copy).
    Generators,

    /// Serve the editing tools over the Model Context Protocol, holding
    /// one document open across calls.
    #[cfg(feature = "mcp")]
    Mcp {
        /// Directory every project path must resolve inside. Defaults to
        /// the current directory.
        #[arg(long)]
        root: Option<PathBuf>,
        /// Serve Streamable HTTP on this address (e.g. `127.0.0.1:8080`)
        /// instead of stdio. Needs the `mcp-http` feature.
        #[arg(long)]
        http: Option<String>,
        /// Write the document back to its file after every edit, so the
        /// editor — which reloads a project that changed on disk — shows
        /// each step. One writer at a time: don't hand-edit the same
        /// file while an agent is running.
        #[arg(long)]
        checkpoint: bool,
    },

    /// Draw a project as PNG images — the agent's eye (headless, no GPU).
    Render {
        /// Project to draw (`.vxlt`).
        project: PathBuf,
        /// Comma-separated viewpoints, or `all` for the full sweep:
        /// iso, front, back, left, right, top, bottom.
        #[arg(long, default_value = "iso")]
        view: String,
        /// Image edge in pixels (max 1024).
        #[arg(long, default_value_t = voxelith::view::DEFAULT_SIZE)]
        size: u32,
        /// Where to write. With one view this is the file; with several,
        /// the images land beside it as `<stem>-<view>.png`.
        #[arg(long)]
        out: Option<PathBuf>,
    },

    /// Grade finished projects against eval cases (headless, JSON on
    /// stdout). Exits non-zero when a case fails, so a suite can gate
    /// a run the way a test command does.
    Eval {
        /// An eval case (`.json`), or a directory of them.
        cases: PathBuf,
        /// The single result to grade. For checking one piece of work
        /// whatever its file is called.
        #[arg(long)]
        project: Option<PathBuf>,
        /// Directory holding one `<case-id>.vxlt` per case.
        #[arg(long)]
        results: Option<PathBuf>,
    },

    /// Describe a project without changing it (headless, JSON on stdout).
    Inspect {
        /// Project to read (`.vxlt`).
        project: PathBuf,
        /// Also render one plane as ASCII art; see `exec --slice`.
        #[arg(long)]
        slice: Option<String>,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Some(Commands::Bake { spec, shard }) => run_bake(&spec, shard.as_deref()),
        Some(Commands::Exec {
            ops,
            input,
            out,
            export,
            describe,
            slice,
            dry_run,
        }) => run_exec(voxelith::exec::ExecRequest {
            ops: Some(ops),
            input,
            output: out,
            export,
            describe,
            slice,
            force_dry_run: dry_run,
        }),
        Some(Commands::Generators) => println!("{}", voxelith::exec::generators_json()),
        #[cfg(feature = "mcp")]
        Some(Commands::Mcp {
            root,
            http,
            checkpoint,
        }) => run_mcp(root, http, checkpoint),
        Some(Commands::Render {
            project,
            view,
            size,
            out,
        }) => run_render(project, &view, size, out),
        Some(Commands::Eval {
            cases,
            project,
            results,
        }) => run_eval(voxelith::eval::EvalRequest {
            cases,
            project,
            results,
        }),
        Some(Commands::Inspect { project, slice }) => run_exec(voxelith::exec::ExecRequest {
            input: Some(project),
            describe: true,
            slice,
            ..Default::default()
        }),
        #[cfg(feature = "gui")]
        None => run_gui(cli.agent_port),
        #[cfg(not(feature = "gui"))]
        None => run_gui(),
    }
}

/// Headless agent step. stdout carries the JSON envelope and nothing
/// else — logs go to stderr — so the caller can parse it directly.
fn run_exec(request: voxelith::exec::ExecRequest) {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp(None)
        .init();

    match voxelith::exec::run_exec(&request) {
        Ok(outcome) => println!("{}", outcome.to_json()),
        Err(error) => {
            // The failure envelope goes to stdout too: one stream to
            // parse, whichever way the run went.
            println!("{}", error.to_json());
            std::process::exit(1);
        }
    }
}

/// Grade results against eval cases. Same stdout contract as
/// `run_exec`, plus a test-runner exit code: zero only when every case
/// passed, so a suite can gate a run without anyone parsing the report.
fn run_eval(request: voxelith::eval::EvalRequest) {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp(None)
        .init();

    match voxelith::eval::run_eval(&request) {
        Ok(report) => {
            println!("{}", report.to_json());
            if report.passed != report.total {
                std::process::exit(1);
            }
        }
        Err(error) => {
            println!("{}", error.to_json());
            std::process::exit(1);
        }
    }
}

/// Draw a project. Same stdout contract as `run_exec` — the report is
/// JSON and the images are files.
fn run_render(project: PathBuf, views: &str, size: u32, out: Option<PathBuf>) {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp(None)
        .init();

    // A bad view name and a failed render report through the same
    // envelope, so the caller parses one shape either way.
    let rendered = voxelith::exec::parse_views(views).and_then(|views| {
        voxelith::exec::run_render(&voxelith::exec::RenderRequest {
            project,
            views,
            size,
            out,
        })
    });
    match rendered {
        Ok(outcome) => println!("{}", outcome.to_json()),
        Err(error) => {
            println!("{}", error.to_json());
            std::process::exit(1);
        }
    }
}

/// Serve the MCP tool set until the client goes away.
///
/// Logs go to stderr and nothing here prints to stdout: on the stdio
/// transport stdout *is* the protocol stream.
#[cfg(feature = "mcp")]
fn run_mcp(root: Option<PathBuf>, http: Option<String>, checkpoint: bool) {
    use voxelith::mcp::Checkpoint;

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    let checkpoint = match checkpoint {
        true => Checkpoint::AfterEveryEdit,
        false => Checkpoint::Off,
    };
    let requested = root.unwrap_or_else(|| PathBuf::from("."));
    let root = match voxelith::mcp::Root::new(&requested) {
        Ok(root) => root,
        Err(e) => {
            eprintln!("can't serve from {}: {e}", requested.display());
            std::process::exit(2);
        }
    };
    log::info!("serving projects under {}", voxelith::mcp::display(root.dir()));
    if checkpoint == Checkpoint::AfterEveryEdit {
        log::info!("check-pointing the document to its file after every edit");
    }

    let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!("could not start the async runtime: {e}");
            std::process::exit(2);
        }
    };
    let served = runtime.block_on(async move {
        match http {
            Some(address) => serve_http(root, &address, checkpoint).await,
            None => voxelith::mcp::serve_stdio(root, checkpoint).await,
        }
    });
    if let Err(e) = served {
        eprintln!("mcp server stopped: {e}");
        std::process::exit(1);
    }
}

#[cfg(feature = "mcp-http")]
async fn serve_http(
    root: voxelith::mcp::Root,
    address: &str,
    checkpoint: voxelith::mcp::Checkpoint,
) -> anyhow::Result<()> {
    voxelith::mcp::serve_http(root, address.parse()?, checkpoint).await
}

/// The HTTP transport is a separate feature because it drags axum in.
/// Say which build this is rather than failing on a parse of the address.
#[cfg(all(feature = "mcp", not(feature = "mcp-http")))]
async fn serve_http(
    _root: voxelith::mcp::Root,
    _address: &str,
    _checkpoint: voxelith::mcp::Checkpoint,
) -> anyhow::Result<()> {
    anyhow::bail!(
        "this build has no HTTP transport (compiled without the `mcp-http` feature); \
         drop --http to serve on stdio, or rebuild with --features mcp-http"
    )
}

/// Headless batch export. Prints a summary and exits with a non-zero code
/// if the spec couldn't run (2) or any item failed (1), so it's CI-usable.
fn run_bake(spec: &Path, shard: Option<&str>) {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp(None)
        .init();

    match voxelith::bake::run_bake(spec, shard) {
        Ok(outcome) => {
            print!("{}", outcome.summary_string());
            if outcome.any_failed() {
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("bake error: {e}");
            std::process::exit(2);
        }
    }
}

/// Without the `gui` feature there is no editor to launch — the binary
/// is still useful (bake / exec / inspect), so say which build this is
/// rather than pretending the command was malformed.
#[cfg(not(feature = "gui"))]
fn run_gui() {
    eprintln!(
        "this build has no interactive editor (compiled without the `gui` feature); \
         run a subcommand instead — see `voxelith --help`"
    );
    std::process::exit(2);
}

/// Launch the interactive winit + egui editor (the default).
///
/// `agent_port` starts the in-editor MCP bridge as the window comes up.
/// The panel can do the same thing with a button, but an agent workflow
/// starts the editor *in order to* be edited, and having to click first
/// puts a human step in the middle of something meant to run on its own.
#[cfg(feature = "gui")]
fn run_gui(agent_port: Option<u16>) {
    // Voxelith's own logs at info; the GPU stack quieted to warnings.
    // wgpu logs device maintenance at info *per frame* — roughly seven
    // thousand lines per idle minute — which buries anything the app
    // says and makes the terminal the app was launched from useless.
    // An explicit RUST_LOG still overrides all of this.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(
        "info,wgpu_core=warn,wgpu_hal=warn,naga=warn",
    ))
    .format_timestamp(None)
    .init();

    log::info!("Starting Voxelith...");

    let event_loop = EventLoop::new().unwrap();
    // A placeholder only: `about_to_wait` re-arms the flow every turn
    // with `WaitUntil(next_frame_at)` — full rate while the user is
    // active, an idle heartbeat otherwise. `Poll` here would burn a
    // core busy-spinning between frames.
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = app::App::new();
    app.start_agent_bridge_at(agent_port);
    event_loop.run_app(&mut app).unwrap();
}
