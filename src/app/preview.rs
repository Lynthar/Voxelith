//! Procgen preview state + debounced regen.
//!
//! Two independent preview sources can be enabled: the single-generator
//! panel and the pipeline graph. Each runs its own debounced state
//! machine — slider drags don't trigger a regen until parameters have
//! stayed stable for `DEBOUNCE`. Both sources share the renderer's
//! preview overlay slot; when both are enabled and fire in the same
//! tick, the graph wins (its tick runs second). When one toggles off
//! while the other stays on, the slot is cleared and the still-active
//! source is forced through its "just-toggled-on" path on the next
//! tick so it re-renders into the freshly-cleared slot.

use std::time::{Duration, Instant};

use voxelith::mesh::patch_to_mesh;
use voxelith::procgen::{
    LSystemTree, PerlinTerrain, PipelineGraph, VoxelGenerator, WfcGenerator,
};
use voxelith::ui::GeneratorChoice;

use super::App;

/// Quiescence period before a regen runs (slider drags batch within this).
const DEBOUNCE: Duration = Duration::from_millis(150);

/// Alpha baked into preview vertex colors. 0.5 reads as "ghosted" but
/// still legible against the dark background.
const PREVIEW_ALPHA: f32 = 0.5;

/// Tracks both preview sources' lifecycle. Each source is independent:
/// turning one off doesn't disable the other, but they share the
/// renderer's overlay slot.
#[derive(Debug)]
pub(super) struct PreviewState {
    // ---- Single-generator branch ----
    pub last_terrain: PerlinTerrain,
    pub last_tree: LSystemTree,
    pub last_wfc: WfcGenerator,
    pub last_selected: GeneratorChoice,
    pub single_enabled: bool,
    pub single_last_change: Option<Instant>,
    pub single_needs_regen: bool,

    // ---- Pipeline graph branch ----
    pub last_graph: PipelineGraph,
    pub graph_enabled: bool,
    pub graph_last_change: Option<Instant>,
    pub graph_needs_regen: bool,
}

impl PreviewState {
    pub fn new() -> Self {
        Self {
            last_terrain: PerlinTerrain::default(),
            last_tree: LSystemTree::default(),
            last_wfc: WfcGenerator::default(),
            last_selected: GeneratorChoice::default(),
            single_enabled: false,
            single_last_change: None,
            single_needs_regen: false,
            last_graph: PipelineGraph::default(),
            graph_enabled: false,
            graph_last_change: None,
            graph_needs_regen: false,
        }
    }
}

impl App {
    /// Drive both preview state machines once per frame.
    pub(super) fn tick_preview(&mut self) {
        self.tick_single_preview();
        self.tick_graph_preview();
    }

    /// Single-generator panel preview: snapshot params, debounce, regen.
    fn tick_single_preview(&mut self) {
        let enabled = self.ui.procgen.preview_enabled;

        // Off (or just-toggled-off): drop our state; if the graph
        // preview is on, force it to re-render into the now-empty slot
        // by resetting its `enabled` snapshot — its tick this same
        // frame will hit the just-toggled-on path.
        if !enabled {
            if self.single_enabled() {
                if let Some(r) = &mut self.renderer {
                    r.clear_preview();
                }
                self.preview.single_enabled = false;
                self.preview.single_last_change = None;
                self.preview.single_needs_regen = false;
                if self.ui.procgen.graph_preview_enabled {
                    self.preview.graph_enabled = false;
                }
            }
            return;
        }

        // Just-toggled-on: snapshot current params and queue an initial
        // regen. Without this, the user would see no preview until they
        // wiggle a slider.
        if !self.single_enabled() {
            self.preview.single_enabled = true;
            self.preview.last_terrain = self.ui.procgen.terrain.clone();
            self.preview.last_tree = self.ui.procgen.tree.clone();
            self.preview.last_wfc = self.ui.procgen.wfc.clone();
            self.preview.last_selected = self.ui.procgen.selected;
            self.preview.single_last_change = Some(Instant::now());
            self.preview.single_needs_regen = true;
        }

        // Detect param mutation.
        let changed = self.ui.procgen.terrain != self.preview.last_terrain
            || self.ui.procgen.tree != self.preview.last_tree
            || self.ui.procgen.wfc != self.preview.last_wfc
            || self.ui.procgen.selected != self.preview.last_selected;
        if changed {
            self.preview.last_terrain = self.ui.procgen.terrain.clone();
            self.preview.last_tree = self.ui.procgen.tree.clone();
            self.preview.last_wfc = self.ui.procgen.wfc.clone();
            self.preview.last_selected = self.ui.procgen.selected;
            self.preview.single_last_change = Some(Instant::now());
            self.preview.single_needs_regen = true;
        }

        // Debounce gate.
        if self.preview.single_needs_regen {
            if let Some(t) = self.preview.single_last_change {
                if t.elapsed() >= DEBOUNCE {
                    self.regen_single_preview();
                    self.preview.single_needs_regen = false;
                }
            }
        }
    }

    /// Pipeline graph preview: same shape as single-gen, but the change
    /// signal is "the whole graph differs from last snapshot" — covers
    /// param tweaks, node add/remove, and wire changes uniformly.
    fn tick_graph_preview(&mut self) {
        let enabled = self.ui.procgen.graph_preview_enabled;

        if !enabled {
            if self.graph_enabled() {
                if let Some(r) = &mut self.renderer {
                    r.clear_preview();
                }
                self.preview.graph_enabled = false;
                self.preview.graph_last_change = None;
                self.preview.graph_needs_regen = false;
                if self.ui.procgen.preview_enabled {
                    self.preview.single_enabled = false;
                }
            }
            return;
        }

        if !self.graph_enabled() {
            self.preview.graph_enabled = true;
            self.preview.last_graph = self.ui.graph.clone();
            self.preview.graph_last_change = Some(Instant::now());
            self.preview.graph_needs_regen = true;
        }

        // Whole-graph equality covers params + topology + positions.
        // Position-only edits also trigger a regen, which is a tiny bit
        // wasteful (output doesn't depend on layout) but the debounce
        // makes it cheap and keeps the change detector trivial.
        if self.ui.graph != self.preview.last_graph {
            self.preview.last_graph = self.ui.graph.clone();
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

    fn single_enabled(&self) -> bool {
        self.preview.single_enabled
    }

    fn graph_enabled(&self) -> bool {
        self.preview.graph_enabled
    }

    /// Run the currently-selected generator and upload the resulting
    /// patch as the preview overlay. Failures and empty output clear
    /// the overlay rather than leaving stale geometry around.
    fn regen_single_preview(&mut self) {
        let result = match self.ui.procgen.selected {
            GeneratorChoice::Terrain => self.ui.procgen.terrain.generate(),
            GeneratorChoice::Tree => self.ui.procgen.tree.generate(),
            GeneratorChoice::Wfc => self.ui.procgen.wfc.generate(),
        };

        let patch = match result {
            Ok(p) if !p.is_empty() => p,
            Ok(_) => {
                if let Some(r) = &mut self.renderer {
                    r.clear_preview();
                }
                return;
            }
            Err(e) => {
                log::warn!("Preview generation failed: {}", e);
                if let Some(r) = &mut self.renderer {
                    r.clear_preview();
                }
                return;
            }
        };

        let mesh = patch_to_mesh(&patch.voxels, PREVIEW_ALPHA);
        if let Some(r) = &mut self.renderer {
            r.set_preview_mesh(&mesh);
        }
    }

    /// Evaluate the pipeline graph and upload its output patch as the
    /// preview overlay. Graph errors (no Output node, missing inputs,
    /// cycles) and empty output both clear the overlay — they're
    /// in-progress states from the user's perspective, not failures
    /// worth surfacing in the status bar (the explicit "Run Pipeline"
    /// button still surfaces them).
    fn regen_graph_preview(&mut self) {
        let patch = match self.ui.graph.evaluate() {
            Ok(p) if !p.is_empty() => p,
            Ok(_) => {
                if let Some(r) = &mut self.renderer {
                    r.clear_preview();
                }
                return;
            }
            Err(e) => {
                log::debug!("Graph preview skipped: {}", e);
                if let Some(r) = &mut self.renderer {
                    r.clear_preview();
                }
                return;
            }
        };

        let mesh = patch_to_mesh(&patch.voxels, PREVIEW_ALPHA);
        if let Some(r) = &mut self.renderer {
            r.set_preview_mesh(&mesh);
        }
    }

    /// Clear the overlay and force both preview state machines to
    /// re-snapshot on the next tick. Called after a generator (single
    /// or graph) writes into the world so the just-applied geometry
    /// doesn't double-render with the preview on top.
    pub(super) fn invalidate_preview(&mut self) {
        if let Some(r) = &mut self.renderer {
            r.clear_preview();
        }
        self.preview.single_last_change = None;
        self.preview.single_needs_regen = false;
        self.preview.single_enabled = false;
        self.preview.graph_last_change = None;
        self.preview.graph_needs_regen = false;
        self.preview.graph_enabled = false;
    }
}
