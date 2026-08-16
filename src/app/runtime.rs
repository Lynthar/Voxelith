//! Background tokio runtime on its own thread, owned for the life of
//! the process: winit's main thread is synchronous and cannot await.
//! Never joined — OS exit cleans it up.

use std::sync::mpsc;
use std::thread;

use tokio::runtime::Handle;

/// Owns a tokio multi-thread runtime running on a dedicated thread.
/// Cheap to clone the handle out via [`Self::handle`].
pub struct AsyncRuntime {
    handle: Handle,
    // Kept only to document that this thread is owned for the life of
    // the process — holding the handle without joining is identical to
    // detaching, and makes no panic more visible either way.
    _runtime_thread: thread::JoinHandle<()>,
}

impl AsyncRuntime {
    /// Spawn the background runtime and wait for its handle. Returns
    /// once it can accept tasks.
    ///
    /// # Panics
    /// On tokio init failure, treated as fatal like wgpu init failure.
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
