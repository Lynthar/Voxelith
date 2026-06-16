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

use super::{App, ShapePhase};

impl App {
    /// Condense the current tool + gesture state into the HUD's
    /// display lines. Pure read; cheap enough to run unconditionally
    /// every frame (a handful of small `format!`s).
    pub(super) fn build_hud_state(&self) -> HudState {
        let tool = self.editor.current_tool;

        let mut phase = None;
        let mut detail = None;
        let mut hints = None;

        if tool.is_shape() {
            // `update_brush_preview` (which runs before the egui pass)
            // drops a shape drag stranded by a mid-drag tool switch,
            // so a live `shape_drag` here always belongs to `tool`.
            if let Some(drag) = self.shape_drag {
                let plane = plane_label(drag.plane.axis, drag.plane.sign);
                match drag.phase {
                    ShapePhase::Footprint => {
                        phase = Some("Footprint");
                        let end = self
                            .editor
                            .hovered_voxel
                            .map(|h| h.adjacent_pos)
                            .unwrap_or(drag.anchor);
                        detail = Some(format!(
                            "{} · plane {}",
                            dims_label(drag_dims(drag.anchor, end)),
                            plane
                        ));
                        hints = Some("release: extrude height · Esc: cancel");
                    }
                    ShapePhase::Height { .. } => {
                        phase = Some("Height");
                        let end = drag
                            .extruded_end(self.cursor_pos.1)
                            .unwrap_or(drag.anchor);
                        detail = Some(format!(
                            "{} · plane {}",
                            dims_label(drag_dims(drag.anchor, end)),
                            plane
                        ));
                        hints = Some("click: commit · Esc: cancel");
                    }
                }
            }
        } else if tool == Tool::Select {
            let cur = self
                .editor
                .hovered_voxel
                .map(|h| Self::select_anchor_pos(&h));
            if let Some(anchor) = self.selection_move_anchor {
                phase = Some("Moving");
                if let Some(c) = cur {
                    detail = Some(delta_label((
                        c.0 - anchor.0,
                        c.1 - anchor.1,
                        c.2 - anchor.2,
                    )));
                }
                hints = Some("release: drop");
            } else if let Some(anchor) = self.selection_drag_anchor {
                phase = Some("Selecting");
                if let Some(c) = cur {
                    detail = Some(dims_label(drag_dims(anchor, c)));
                }
                hints = Some("release: select");
            }
        } else if self.left_button_held {
            // Mid-stroke for a brush tool: surface the locked face
            // plane the drag-paint is pinned to.
            if let Some(p) = self.stroke_plane {
                detail = Some(format!("plane {}", plane_label(p.axis, p.sign)));
            }
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
        let selection = if tool == Tool::Select && self.selection_drag_anchor.is_none() {
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
