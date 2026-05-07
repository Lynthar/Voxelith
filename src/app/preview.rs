//! Procgen preview state + debounced regen.
//!
//! When the user enables the preview toggle, the app keeps a translucent
//! overlay mesh of the selected generator's current output. We don't
//! regenerate every frame — slider drags would spam the generator —
//! so we debounce: the regen runs once parameters have stayed stable
//! for `DEBOUNCE_MS`. The overlay lives on `Renderer.preview_mesh` and
//! is drawn through the transparent pipeline after opaque chunks.

use std::time::{Duration, Instant};

use voxelith::mesh::patch_to_mesh;
use voxelith::procgen::{LSystemTree, PerlinTerrain, VoxelGenerator, WfcGenerator};
use voxelith::ui::GeneratorChoice;

use super::App;

/// Quiescence period before a regen runs (slider drags batch within this).
const DEBOUNCE: Duration = Duration::from_millis(150);

/// Alpha baked into preview vertex colors. 0.5 reads as "ghosted" but
/// still legible against the dark background.
const PREVIEW_ALPHA: f32 = 0.5;

/// Tracks whatever's needed to drive the preview overlay's lifecycle.
#[derive(Debug)]
pub(super) struct PreviewState {
    /// Param snapshots from the last tick — compared against the
    /// current `ProcgenSettings` to detect mutation.
    pub last_terrain: PerlinTerrain,
    pub last_tree: LSystemTree,
    pub last_wfc: WfcGenerator,
    pub last_selected: GeneratorChoice,
    /// Was the preview toggle on last tick? Used to detect transitions.
    pub last_enabled: bool,
    /// When the user last touched a parameter. None = no pending regen.
    pub last_change: Option<Instant>,
    /// True between a parameter change and the regen that follows.
    pub needs_regen: bool,
}

impl PreviewState {
    pub fn new() -> Self {
        Self {
            last_terrain: PerlinTerrain::default(),
            last_tree: LSystemTree::default(),
            last_wfc: WfcGenerator::default(),
            last_selected: GeneratorChoice::default(),
            last_enabled: false,
            last_change: None,
            needs_regen: false,
        }
    }
}

impl App {
    /// Drive the preview lifecycle once per frame. Detects param
    /// changes, applies the debounce, regens + uploads on quiescence.
    pub(super) fn tick_preview(&mut self) {
        let enabled = self.ui.procgen.preview_enabled;

        // Off (or just-toggled-off): drop the overlay and reset state.
        if !enabled {
            if self.preview.last_enabled {
                if let Some(r) = &mut self.renderer {
                    r.clear_preview();
                }
                self.preview.last_enabled = false;
                self.preview.last_change = None;
                self.preview.needs_regen = false;
            }
            return;
        }

        // Just-toggled-on: snapshot current params and queue an initial
        // regen. Without this, the user would see no preview until they
        // wiggle a slider.
        if !self.preview.last_enabled {
            self.preview.last_enabled = true;
            self.preview.last_terrain = self.ui.procgen.terrain.clone();
            self.preview.last_tree = self.ui.procgen.tree.clone();
            self.preview.last_wfc = self.ui.procgen.wfc.clone();
            self.preview.last_selected = self.ui.procgen.selected;
            self.preview.last_change = Some(Instant::now());
            self.preview.needs_regen = true;
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
            self.preview.last_change = Some(Instant::now());
            self.preview.needs_regen = true;
        }

        // Debounce gate.
        if self.preview.needs_regen {
            if let Some(t) = self.preview.last_change {
                if t.elapsed() >= DEBOUNCE {
                    self.regen_preview();
                    self.preview.needs_regen = false;
                }
            }
        }
    }

    /// Run the currently-selected generator and upload the resulting
    /// patch as the preview overlay. Failures and empty output clear
    /// the overlay rather than leaving stale geometry around.
    fn regen_preview(&mut self) {
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

    /// Clear the overlay and force the preview state machine to
    /// re-snapshot params on the next tick. Called after a generator
    /// runs into the world so the just-applied geometry doesn't
    /// double-render with the preview on top.
    pub(super) fn invalidate_preview(&mut self) {
        if let Some(r) = &mut self.renderer {
            r.clear_preview();
        }
        self.preview.last_change = None;
        self.preview.needs_regen = false;
        // Force the on-transition path next tick if preview is still on.
        self.preview.last_enabled = false;
    }
}
