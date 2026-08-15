//! Per-frame snapshot builder for the viewport HUD.
//!
//! The HUD itself (layout, styling, formatting helpers) lives on the
//! library side in `voxelith::ui::hud`; this module is the thin glue
//! that reads the App's gesture state — which only exists in the
//! binary crate — and condenses it into a display-ready
//! [`HudState`]. Built once per frame in `render_frame`, right before
//! `Ui::show`.
//!
//! The numbers shown must come from the *same* math the commit paths
//! use, so the HUD can never disagree with what a click would
//! produce: shape height goes through `ShapeDrag::extruded_end`
//! (`commit_shape`'s source of truth), the footprint end cell is the
//! plane-locked `hovered_voxel.adjacent_pos` (same as
//! `update_brush_preview`), and the move delta mirrors
//! `update_selection_visualization`.

use voxelith::editor::Tool;
use voxelith::ui::hud::{
    delta_label, dims_label, drag_dims, plane_label, selection_label, symmetry_label,
};
use voxelith::ui::HudState;

use super::{App, EditInteraction};

impl App {
    /// Condense the current tool + gesture state into the HUD's
    /// display lines. Pure read; cheap enough to run unconditionally
    /// every frame (a handful of small `format!`s).
    pub(super) fn build_hud_state(&self) -> HudState {
        // The effective tool, so an Alt-held eyedropper is what the
        // HUD names — matching what a click would actually do.
        let tool = self.effective_tool();

        let mut phase = None;
        let mut detail = None;
        let mut hints = None;

        // `update_brush_preview` (which runs before the egui pass)
        // reconciles a gesture stranded by a mid-drag tool switch, so
        // a live gesture here always belongs to `tool`.
        match &self.interaction {
            EditInteraction::ShapeFootprint { anchor, plane } => {
                phase = Some("Footprint");
                let end = self
                    .editor
                    .hovered_voxel
                    .map(|h| h.adjacent_pos)
                    .unwrap_or(*anchor);
                detail = Some(format!(
                    "{} · plane {}",
                    dims_label(drag_dims(*anchor, end)),
                    plane_label(plane.axis, plane.sign)
                ));
                hints = Some("release: extrude height · Esc: cancel");
            }
            EditInteraction::ShapeHeight { anchor, plane, .. } => {
                phase = Some("Height");
                let end = self
                    .interaction
                    .shape_extruded_end(self.cursor_pos.1)
                    .unwrap_or(*anchor);
                detail = Some(format!(
                    "{} · plane {}",
                    dims_label(drag_dims(*anchor, end)),
                    plane_label(plane.axis, plane.sign)
                ));
                hints = Some("click: commit · Esc: cancel");
            }
            EditInteraction::SelectMove { anchor, .. } => {
                phase = Some("Moving");
                let cur = self
                    .editor
                    .hovered_voxel
                    .map(|h| Self::select_anchor_pos(&h));
                if let Some(c) = cur {
                    detail = Some(delta_label((
                        c.0 - anchor.0,
                        c.1 - anchor.1,
                        c.2 - anchor.2,
                    )));
                }
                hints = Some("release: drop");
            }
            EditInteraction::SelectDrag { anchor } => {
                phase = Some("Selecting");
                let cur = self
                    .editor
                    .hovered_voxel
                    .map(|h| Self::select_anchor_pos(&h));
                if let Some(c) = cur {
                    detail = Some(dims_label(drag_dims(*anchor, c)));
                }
                hints = Some("release: select");
            }
            EditInteraction::BrushStroke { plane: Some(p), .. } => {
                // Mid-stroke for a brush tool: surface the locked face
                // plane the drag-paint is pinned to.
                detail = Some(format!("plane {}", plane_label(p.axis, p.sign)));
            }
            EditInteraction::BrushStroke { plane: None, .. }
            | EditInteraction::Idle => {}
        }

        let symmetry = if tool_uses_symmetry(tool) {
            symmetry_label(&self.editor.symmetry)
        } else {
            None
        };

        // Select-tool-only — the status bar keeps the always-on copy
        // for other tools. Hidden mid-marquee-drag: the live size is
        // already in `detail`, and the stale pre-drag box would just
        // contradict it.
        let selection = if tool == Tool::Select
            && !matches!(self.interaction, EditInteraction::SelectDrag { .. })
        {
            self.editor.selection.map(|sel| {
                let (w, h, d) = sel.size();
                selection_label(w, h, d, sel.cell_count())
            })
        } else {
            None
        };

        HudState {
            tool: tool.name(),
            phase,
            detail,
            symmetry,
            selection,
            hints,
        }
    }
}

/// Symmetry mirrors Place / Remove / Paint / Fill writes and shape
/// commits; Eyedropper samples, Select reads, and Socket drops an
/// un-mirrored anchor — a "Sym" line for those would imply an effect
/// that won't happen.
fn tool_uses_symmetry(t: Tool) -> bool {
    !matches!(t, Tool::Eyedropper | Tool::Select | Tool::Socket)
}
