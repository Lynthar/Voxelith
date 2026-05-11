//! AI integration: remote 3D-generation services + mesh-to-voxel conversion.
//!
//! ## Architecture
//!
//! `App` owns one [`AiRuntime`] (a tokio multi-thread runtime running on a
//! dedicated background thread, started in `App::new`). UI gestures
//! ("Generate") spawn an async task on the runtime via the runtime's
//! [`tokio::runtime::Handle`]. The task talks to a remote provider
//! (currently a [`MockProvider`] for Phase 1; Phase 2 adds the real
//! fal.ai client) and emits [`JobEvent`]s through a `std::sync::mpsc`
//! channel back to the main thread, which drains them every frame in
//! `App::tick_ai_job` to advance an [`AiJobState`] machine.
//!
//! API keys live in the OS keychain via [`keyring_store`] (Windows
//! Credential Manager / macOS Keychain / Linux Secret Service) — never
//! in `prefs.ron`.
//!
//! ## Module map
//!
//! - [`job`]: `AiJobState` enum + `JobEvent` events
//! - [`provider`]: `AiProvider` trait + `AiRequest` / `JobHandle` types
//! - [`runtime`]: `AiRuntime` (tokio thread + handle)
//! - [`keyring_store`]: load/save/clear API key
//! - [`mock`]: a `MockProvider` for Phase 1 wiring tests
//! - [`client`]: real fal.ai HTTP client (Phase 2 — currently empty stub)
//! - [`voxelize`]: mesh → voxel conversion (Phase 3 — currently empty stub)

mod client;
mod job;
mod keyring_store;
mod mock;
mod provider;
mod runtime;
mod voxelize;

pub use client::FalHunyuanProvider;
pub use job::{AiJobState, JobEvent};
pub use keyring_store::{
    clear_api_key, has_api_key, load_api_key, save_api_key, KeyringError,
};
pub use mock::MockProvider;
pub use provider::{AiProvider, AiRequest, JobHandle};
pub use runtime::AiRuntime;
pub use voxelize::voxelize_glb;

use thiserror::Error;

/// Top-level error surfaced from the AI module.
///
/// Concrete failure modes (HTTP, keychain, voxelization) are wrapped in
/// the variants below. The string in `Failed(...)` events is rendered
/// as `Display`, so end-user-facing wording lives here.
#[derive(Debug, Error)]
pub enum AiError {
    #[error("Network: {0}")]
    Network(String),
    #[error("Provider rejected request: {0}")]
    Provider(String),
    #[error("Voxelization failed: {0}")]
    Voxelize(String),
    #[error("Cancelled")]
    Cancelled,
    #[error("Keychain: {0}")]
    Keyring(#[from] KeyringError),
}
