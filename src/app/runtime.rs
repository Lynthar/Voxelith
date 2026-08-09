//! Background tokio runtime, owned for the life of the process.
//!
//! winit owns the main thread and its event loop is synchronous, so
//! anything async has to live somewhere else. This spawns a tokio
//! multi-thread runtime on a dedicated OS thread and hands out a
//! [`tokio::runtime::Handle`]; the main thread starts work with
//! `handle.spawn(...)` and never blocks on it.
//!
//! Its one consumer today is the in-editor agent bridge, whose HTTP
//! server runs here (`app::agent_bridge`). It was originally written
//! for the AI worker and outlived it — the need is not "AI", it is
//! "winit's main thread cannot await".
//!
//! The runtime thread is intentionally never joined: it lives the whole
//! process lifetime and OS exit cleans it up. An explicit
//! `shutdown_background()` on drop would only slow the close path.

use std::sync::mpsc;
use std::thread;

use tokio::runtime::Handle;

/// Owns a tokio multi-thread runtime running on a dedicated thread.
/// Cheap to clone the handle out via [`Self::handle`].
pub struct AsyncRuntime {
    handle: Handle,
    // Kept only to document that this thread is owned for the life of
    // the process. Holding the handle without joining is behaviourally
    // identical to detaching, and in particular does NOT make a panic
    // any more visible — the panic hook prints either way. The one
    // realistic panic here (the runtime failing to build) happens
    // before the handle handshake below and is already surfaced by the
    // `recv().expect`. A spawned task's panic never reaches this thread
    // at all: tokio catches it per-task, and the caller notices through
    // whatever channel that task was feeding.
    _runtime_thread: thread::JoinHandle<()>,
}

impl AsyncRuntime {
    /// Spawn the background tokio runtime and wait for its handle to
    /// become available. Synchronous; returns once the runtime is
    /// ready to accept tasks. Panics on tokio init failure (we treat
    /// this the same as wgpu init failure — fatal).
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        let runtime_thread = thread::Builder::new()
            .name("voxelith-tokio".into())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .thread_name("voxelith-async-worker")
                    .build()
                    .expect("Failed to build tokio runtime");
                // Send the handle out before parking on `block_on`.
                // If the receiver is gone, the parent dropped us, so
                // exit immediately.
                if tx.send(runtime.handle().clone()).is_err() {
                    return;
                }
                // Park forever — the runtime stays alive as long as
                // this thread does. Tasks are spawned via the handle
                // we just sent out; they run on the worker threads.
                runtime.block_on(std::future::pending::<()>());
            })
            .expect("Failed to spawn the async runtime thread");

        let handle = rx
            .recv()
            .expect("async runtime thread terminated before sending handle");

        Self {
            handle,
            _runtime_thread: runtime_thread,
        }
    }

    /// Borrow the runtime handle. Callers clone it into worker closures
    /// via `handle.spawn(future)`.
    pub fn handle(&self) -> &Handle {
        &self.handle
    }
}

impl Default for AsyncRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_starts_and_handle_can_spawn() {
        // Sanity check that the background tokio thread came up and
        // we can run a trivial task on it.
        let rt = AsyncRuntime::new();
        let (tx, rx) = mpsc::channel();
        rt.handle().spawn(async move {
            tx.send(42).ok();
        });
        // The task is async; give it a chance to run. mpsc::recv blocks.
        let value = rx.recv().expect("worker task panicked");
        assert_eq!(value, 42);
    }
}
