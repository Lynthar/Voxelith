//! Per-frame snapshot builder for the viewport HUD, condensing the
//! App's gesture state into a [`HudState`]. Every number comes from the
//! same math the commit paths use, so the two can't disagree.

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
            EditInteraction::BrushStroke { plane: None, .. } | EditInteraction::Idle => {}
        }

        let symmetry = if tool_uses_symmetry(tool) {
            symmetry_label(&self.editor.symmetry)
        } else {
            None
        };

        // Select only — the status bar keeps the always-on copy. Hidden
        // mid-drag, where the live size is already in `detail` and the
        // stale box would contradict it.
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

/// Symmetry mirrors the tools that write voxels. Eyedropper samples,
/// Select reads and Socket drops an un-mirrored anchor, so a "Sym" line
/// there would imply an effect that won't happen.
fn tool_uses_symmetry(t: Tool) -> bool {
    !matches!(t, Tool::Eyedropper | Tool::Select | Tool::Socket)
}
