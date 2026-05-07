//! Voxelith - Procedural-first voxel asset creation tool
//!
//! Entry point. The actual application lives in the `app` module.

mod app;

use winit::event_loop::{ControlFlow, EventLoop};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    log::info!("Starting Voxelith...");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = app::App::new();
    event_loop.run_app(&mut app).unwrap();
}
