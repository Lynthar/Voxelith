//! Procgen preview state + debounced regen, arbitrated by an owner.
//!
//! Two sources can put geometry in the renderer's single preview
//! overlay slot: the pipeline graph, and an agent batch parked for
//! review. [`PreviewOwner`] records which of them the slot currently
//! belongs to, and every write or clear goes through
//! `preview_slot_set` / `preview_slot_release`, so a source can
//! neither blank another source's geometry nor leave its own behind.
//! (There used to be a third source — the single-generator panel,
//! retired 2026-08 when its generators became one-node graph presets —
//! and before the owner existed, the arbitration was the two debounced
//! sources force-resetting each other's `enabled` snapshots: a
//! generator failure could blank a healthy graph preview.)
//!
//! The graph branch keeps a debounced state machine — slider drags
//! don't trigger a regen until the graph has stayed stable for
//! `DEBOUNCE`. An agent review outranks it: while a batch is parked,
//! `tick_preview` stands down entirely (the human is being asked about
//! *that* geometry), and answering the batch releases the slot back.
//!
//! **Regen runs synchronously on the UI thread.** That's only viable
//! because `check_graph`'s ceilings keep the worst case to a few
//! milliseconds (terrain ≤ 256×256 × 8 octaves, WFC ≤ 24×24, graphs of
//! a handful of nodes) and the debounce collapses a slider drag into
//! one run. Raising any of those caps, or adding a slower generator,
//! means moving regen to a background task first — otherwise the
//! editor visibly stutters while dragging.

use std::time::{Duration, Instant};

use voxelith::mesh::{patch_to_mesh, ChunkMesh};
use voxelith::procgen::PipelineGraph;

use super::App;

/// Quiescence period before a regen runs (slider drags batch within this).
const DEBOUNCE: Duration = Duration::from_millis(150);

/// Alpha baked into preview vertex colors. 0.5 reads as "ghosted" but
/// still legible against the dark background.
const PREVIEW_ALPHA: f32 = 0.5;

/// Which source the renderer's preview overlay slot currently shows.
/// `AgentReview` is set by `agent_bridge::show_review_preview` and
/// outranks `Graph` (its tick stands down while a batch is parked).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PreviewOwner {
    None,
    Graph,
    AgentReview,
}

/// Tracks the graph preview's lifecycle. Who is actually on screen is
/// `owner`'s to say.
#[derive(Debug)]
pub(super) struct PreviewState {
    pub owner: PreviewOwner,

    // ---- Pipeline graph branch ----
    pub last_graph: PipelineGraph,
    pub graph_enabled: bool,
    pub graph_last_change: Option<Instant>,
    pub graph_needs_regen: bool,
}

impl PreviewState {
    pub fn new() -> Self {
        Self {
            owner: PreviewOwner::None,
            last_graph: PipelineGraph::default(),
            graph_enabled: false,
            graph_last_change: None,
            graph_needs_regen: false,
        }
    }
}

impl App {
    /// Put `mesh` in the overlay slot on `owner`'s behalf.
    fn preview_slot_set(&mut self, owner: PreviewOwner, mesh: &ChunkMesh) {
        if let Some(r) = &mut self.renderer {
            r.set_preview_mesh(mesh);
        }
        self.preview.owner = owner;
    }

    /// Clear the overlay slot **if** `owner` is the one holding it.
    /// A release from a source that isn't on screen changes nothing —
    /// that's the point: a generator's failure or toggle-off must not
    /// blank a graph preview the user is looking at (or vice versa).
    fn preview_slot_release(&mut self, owner: PreviewOwner) {
        if self.preview.owner != owner {
            return;
        }
        if let Some(r) = &mut self.renderer {
            r.clear_preview();
        }
        self.preview.owner = PreviewOwner::None;
    }

    /// Drive both preview state machines once per frame.
    pub(super) fn tick_preview(&mut self) {
        // A batch waiting for approval owns the overlay slot: the human
        // is being asked about *that* geometry, and a debounced generator
        // repaint would quietly replace what they are deciding on with
        // something else entirely. Answering the batch releases the slot
        // (`App::clear_review_preview`), so whichever source is switched
        // on re-renders into it on the next tick.
        if self.agent.pending.is_some() {
            return;
        }
        self.tick_graph_preview();
    }

    /// Pipeline graph preview: snapshot, debounce, regen. The change
    /// signal is "the whole graph differs from last snapshot" — covers
    /// param tweaks, node add/remove, and wire changes uniformly.
    fn tick_graph_preview(&mut self) {
        let enabled = self.ui.procgen.graph_preview_enabled;

        if !enabled {
            if self.preview.graph_enabled {
                self.preview_slot_release(PreviewOwner::Graph);
                self.preview.graph_enabled = false;
                self.preview.graph_last_change = None;
                self.preview.graph_needs_regen = false;
            }
            return;
        }

        if !self.preview.graph_enabled {
            self.preview.graph_enabled = true;
            self.preview.last_graph = self.document.graph.clone();
            self.preview.graph_last_change = Some(Instant::now());
            self.preview.graph_needs_regen = true;
        }

        // Whole-graph equality covers params + topology + positions.
        // Position-only edits also trigger a regen, which is a tiny bit
        // wasteful (output doesn't depend on layout) but the debounce
        // makes it cheap and keeps the change detector trivial.
        if self.document.graph != self.preview.last_graph {
            self.preview.last_graph = self.document.graph.clone();
            self.preview.graph_last_change = Some(Instant::now());
            self.preview.graph_needs_regen = true;
        }

        if self.preview.graph_needs_regen {
            if let Some(t) = self.preview.graph_last_change {
                if t.elapsed() >= DEBOUNCE {
                    self.regen_graph_preview();
                    self.preview.graph_needs_regen = false;
                }
            }
        }
    }

    /// Evaluate the pipeline graph and upload its output patch as the
    /// preview overlay. Graph errors (no Output node, missing inputs,
    /// cycles) and empty output both release the slot — they're
    /// in-progress states from the user's perspective, not failures
    /// worth surfacing in the status bar (the explicit "Run Pipeline"
    /// button still surfaces them).
    fn regen_graph_preview(&mut self) {
        // Checked before evaluating, for the reason `run_graph` gives —
        // and this path needs it more, because nobody clicked anything:
        // opening a project with the preview toggle on is enough to
        // reach the evaluator, and the ceilings it walks past are the
        // ones that end the process rather than the frame.
        if let Err(refusal) = voxelith::agent_ops::check_graph(&self.document.graph) {
            log::debug!("Graph preview skipped: {}", refusal.message);
            self.preview_slot_release(PreviewOwner::Graph);
            return;
        }
        let patch = match self.document.graph.evaluate() {
            Ok(p) if !p.is_empty() => p,
            Ok(_) => {
                self.preview_slot_release(PreviewOwner::Graph);
                return;
            }
            Err(e) => {
                log::debug!("Graph preview skipped: {}", e);
                self.preview_slot_release(PreviewOwner::Graph);
                return;
            }
        };

        let mesh = patch_to_mesh(&patch.voxels, PREVIEW_ALPHA);
        self.preview_slot_set(PreviewOwner::Graph, &mesh);
    }

    /// Show `mesh` as an agent-review preview. The review outranks the
    /// generator / graph branches: they stand down in `tick_preview`
    /// while the batch is parked, so nothing repaints over it.
    pub(super) fn show_review_preview_mesh(&mut self, mesh: &ChunkMesh) {
        self.preview_slot_set(PreviewOwner::AgentReview, mesh);
    }

    /// Clear the overlay unconditionally and force the graph branch to
    /// re-snapshot on the next tick. Called after geometry lands in the
    /// world (a committed generation, an accepted agent batch, an
    /// import) so the just-applied voxels don't double-render under a
    /// stale overlay — whoever owned the slot, that picture is now
    /// wrong.
    pub(super) fn invalidate_preview(&mut self) {
        if let Some(r) = &mut self.renderer {
            r.clear_preview();
        }
        self.preview.owner = PreviewOwner::None;
        self.preview.graph_last_change = None;
        self.preview.graph_needs_regen = false;
        self.preview.graph_enabled = false;
    }
}

#[cfg(test)]
mod tests {
    //! The slot-ownership rules. `App::new()` builds without a
    //! renderer; these functions touch it only through `if let Some`,
    //! so the owner bookkeeping — the part the old cross-reset hack
    //! got wrong — is exercised directly.

    use voxelith::core::Voxel;
    use voxelith::mesh::patch_to_mesh;

    use super::super::App;
    use super::PreviewOwner;

    fn some_mesh() -> voxelith::mesh::ChunkMesh {
        patch_to_mesh(&[((0, 0, 0), Voxel::from_rgb(1, 2, 3))], 0.5)
    }

    #[test]
    fn a_release_from_a_non_owner_changes_nothing() {
        // The exact failure the owner exists to prevent: one source
        // toggling off (or failing) while another's geometry is on
        // screen must not blank the slot.
        let mut app = App::new();
        app.preview_slot_set(PreviewOwner::AgentReview, &some_mesh());
        app.preview_slot_release(PreviewOwner::Graph);
        assert_eq!(app.preview.owner, PreviewOwner::AgentReview);
    }

    #[test]
    fn the_owner_releases_its_own_slot() {
        let mut app = App::new();
        app.preview_slot_set(PreviewOwner::Graph, &some_mesh());
        app.preview_slot_release(PreviewOwner::Graph);
        assert_eq!(app.preview.owner, PreviewOwner::None);
    }

    #[test]
    fn an_agent_review_preempts_and_invalidate_clears_everyone() {
        let mut app = App::new();
        app.preview_slot_set(PreviewOwner::Graph, &some_mesh());
        app.show_review_preview_mesh(&some_mesh());
        assert_eq!(app.preview.owner, PreviewOwner::AgentReview);
        // Answering the batch goes through invalidate_preview: slot
        // free, the graph branch forced to re-snapshot on its next tick.
        app.invalidate_preview();
        assert_eq!(app.preview.owner, PreviewOwner::None);
        assert!(!app.preview.graph_enabled);
    }
}
