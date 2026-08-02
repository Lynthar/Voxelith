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
        Some(Commands::Mcp { root, http }) => run_mcp(root, http),
        Some(Commands::Inspect { project, slice }) => run_exec(voxelith::exec::ExecRequest {
            input: Some(project),
            describe: true,
            slice,
            ..Default::default()
        }),
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

/// Serve the MCP tool set until the client goes away.
///
/// Logs go to stderr and nothing here prints to stdout: on the stdio
/// transport stdout *is* the protocol stream.
#[cfg(feature = "mcp")]
fn run_mcp(root: Option<PathBuf>, http: Option<String>) {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    let requested = root.unwrap_or_else(|| PathBuf::from("."));
    let root = match voxelith::mcp::Root::new(&requested) {
        Ok(root) => root,
        Err(e) => {
            eprintln!("can't serve from {}: {e}", requested.display());
            std::process::exit(2);
        }
    };
    log::info!("serving projects under {}", voxelith::mcp::display(root.dir()));

    let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!("could not start the async runtime: {e}");
            std::process::exit(2);
        }
    };
    let served = runtime.block_on(async move {
        match http {
            Some(address) => serve_http(root, &address).await,
            None => voxelith::mcp::serve_stdio(root).await,
        }
    });
    if let Err(e) = served {
        eprintln!("mcp server stopped: {e}");
        std::process::exit(1);
    }
}

#[cfg(feature = "mcp-http")]
async fn serve_http(root: voxelith::mcp::Root, address: &str) -> anyhow::Result<()> {
    voxelith::mcp::serve_http(root, address.parse()?).await
}

/// The HTTP transport is a separate feature because it drags axum in.
/// Say which build this is rather than failing on a parse of the address.
#[cfg(all(feature = "mcp", not(feature = "mcp-http")))]
async fn serve_http(_root: voxelith::mcp::Root, _address: &str) -> anyhow::Result<()> {
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
#[cfg(feature = "gui")]
fn run_gui() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    log::info!("Starting Voxelith...");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = app::App::new();
    event_loop.run_app(&mut app).unwrap();
}
