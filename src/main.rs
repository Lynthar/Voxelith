//! Voxelith - Procedural-first voxel asset creation tool
//!
//! Entry point. With no subcommand this launches the interactive editor
//! (the `app` module). The `bake` subcommand runs a headless batch export
//! — no window, no GPU — driven by a spec file (see `crate::bake` and
//! `docs/GAME_PIPELINE_ROADMAP.md` §3.5).

mod app;

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
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
    /// (headless — opens no window). See docs/GAME_PIPELINE_ROADMAP.md §3.5.
    Bake {
        /// Path to the bake spec (.json).
        spec: PathBuf,
        /// Process only shard i of n, for CI fan-out, e.g. `--shard 0/4`.
        #[arg(long)]
        shard: Option<String>,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Some(Commands::Bake { spec, shard }) => run_bake(&spec, shard.as_deref()),
        None => run_gui(),
    }
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

/// Launch the interactive winit + egui editor (the default).
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
