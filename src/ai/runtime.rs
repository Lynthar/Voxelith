//! Background tokio runtime for AI worker tasks.
//!
//! winit owns the main thread (sync event loop). reqwest / tokio need
//! an async runtime, which can't run on the main thread without
//! blocking. We spawn a dedicated tokio multi-thread runtime in its
//! own OS thread; `App` holds a [`tokio::runtime::Handle`] cloned
//! from that runtime and uses `handle.spawn(...)` to start AI tasks
//! from the main thread.
//!
//! The runtime thread is intentionally never joined — it lives the
//! entire process lifetime, and OS process exit cleans it up. We
//! could add an explicit shutdown via `runtime.shutdown_background()`
//! on drop if a future use case needs it, but for app-lifetime AI
//! work it'd just slow down the close path.

use std::sync::mpsc;
use std::thread;

use tokio::runtime::Handle;

/// Owns a tokio multi-thread runtime running on a dedicated thread.
/// Cheap to clone the handle out via [`Self::handle`].
pub struct AiRuntime {
    handle: Handle,
    // Hold the join handle so the thread isn't detached and a panic
    // there is observable to the OS for crash reports. We never
    // actually join — the thread exits with the process.
    _runtime_thread: thread::JoinHandle<()>,
}

impl AiRuntime {
    /// Spawn the background tokio runtime and wait for its handle to
    /// become available. Synchronous; returns once the runtime is
    /// ready to accept tasks. Panics on tokio init failure (we treat
    /// this the same as wgpu init failure — fatal).
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        let runtime_thread = thread::Builder::new()
            .name("voxelith-ai-tokio".into())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .thread_name("voxelith-ai-worker")
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
            .expect("Failed to spawn AI runtime thread");

        let handle = rx
            .recv()
            .expect("AI runtime thread terminated before sending handle");

        Self {
            handle,
            _runtime_thread: runtime_thread,
        }
    }

    /// Borrow the runtime handle. `App` clones this into worker
    /// closures via `handle.spawn(future)`.
    pub fn handle(&self) -> &Handle {
        &self.handle
    }
}

impl Default for AiRuntime {
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
        let rt = AiRuntime::new();
        let (tx, rx) = mpsc::channel();
        rt.handle().spawn(async move {
            tx.send(42).ok();
        });
        // The task is async; give it a chance to run. mpsc::recv blocks.
        let value = rx.recv().expect("worker task panicked");
        assert_eq!(value, 42);
    }
}
