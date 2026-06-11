//! Low-interference viewport HUD: a small, click-through, translucent
//! block in the bottom-left corner of the viewport that mirrors the
//! editor state a user needs *while the cursor is in the scene* —
//! active tool, in-flight gesture phase with live numbers, locked
//! plane, symmetry, selection size, and (during modal gestures) key
//! hints.
//!
//! Placement and style follow the cross-editor conventions researched
//! for this feature (vengi's `BrushHud` is the closest prior art):
//! bottom-left block, semi-transparent dark backplate (never bare
//! alpha-blended text), display-only. The `Area` is
//! `interactable(false)` so it never consumes a press — a click
//! "through" the HUD must still paint / select (`egui_consumed` in
//! `handler.rs` would otherwise gate the editor out of it) — and
//! `Order::Background` keeps every floating egui window above it.
//!
//! The App layer owns all gesture state, so it builds the
//! display-ready [`HudState`] each frame (`App::build_hud_state`);
//! this module only formats and lays it out. The formatting helpers
//! are free functions here (rather than in the App) so they're
//! unit-testable without a window.

use egui::{Align2, Color32, Context, Id, Order, RichText};

use super::RenderStats;
use crate::editor::SymmetryAxes;

/// Display-ready HUD content, rebuilt by the App every frame.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HudState {
    /// Active tool name. Always shown.
    pub tool: &'static str,
    /// Gesture phase appended after the tool name, e.g. "Footprint",
    /// "Height", "Moving", "Selecting".
    pub phase: Option<&'static str>,
    /// Live gesture readout: shape dims + locked plane, move delta,
    /// or marquee size.
    pub detail: Option<String>,
    /// "Sym: XZ" — only set for tools symmetry affects.
    pub symmetry: Option<String>,
    /// "Sel: W×H×D (N cells)" — only set while the Select tool is
    /// active (the status bar keeps the always-on copy for other
    /// tools).
    pub selection: Option<String>,
    /// Key hints for the current modal gesture ("click: commit ·
    /// Esc: cancel"). `None` outside modal gestures.
    pub hints: Option<&'static str>,
}

/// Shared translucent backplate for both HUD corners — research
/// takeaway: in-viewport text always sits on a contrast aid, never
/// bare alpha-blended over the scene.
fn hud_frame() -> egui::Frame {
    egui::Frame::none()
        .fill(Color32::from_rgba_unmultiplied(15, 15, 22, 200))
        .rounding(egui::Rounding::same(6.0))
        .stroke(egui::Stroke::new(
            1.0,
            Color32::from_rgba_unmultiplied(255, 255, 255, 24),
        ))
        .inner_margin(egui::Margin::symmetric(10.0, 8.0))
}

/// Render the HUD block. Call after every panel has claimed its
/// screen edge so `ctx.available_rect()` is the true viewport rect.
pub(super) fn show_hud_overlay(ctx: &Context, hud: &HudState) {
    let pos = ctx.available_rect().left_bottom() + egui::vec2(12.0, -12.0);
    egui::Area::new(Id::new("viewport_hud"))
        .pivot(Align2::LEFT_BOTTOM)
        .fixed_pos(pos)
        .order(Order::Background)
        .interactable(false)
        .show(ctx, |ui| {
            hud_frame()
                .show(ui, |ui| {
                    ui.set_max_width(240.0);
                    ui.spacing_mut().item_spacing.y = 2.0;

                    let title = match hud.phase {
                        Some(p) => format!("{} — {}", hud.tool, p),
                        None => hud.tool.to_string(),
                    };
                    // Colors mirror the status bar's coding (tool =
                    // light blue, symmetry = light yellow, selection
                    // = yellow) so the two readouts visibly refer to
                    // the same state.
                    ui.label(RichText::new(title).strong().color(Color32::LIGHT_BLUE));
                    if let Some(d) = &hud.detail {
                        ui.label(RichText::new(d).color(Color32::from_gray(220)));
                    }
                    if let Some(s) = &hud.symmetry {
                        ui.label(RichText::new(s).color(Color32::LIGHT_YELLOW));
                    }
                    if let Some(s) = &hud.selection {
                        ui.label(
                            RichText::new(s).color(Color32::from_rgb(255, 230, 60)),
                        );
                    }
                    if let Some(h) = hud.hints {
                        ui.label(RichText::new(h).small().weak());
                    }
                });
        });
}

/// Render the performance readout in the bottom-right corner — the
/// always-glanceable subset of the Statistics window (which stays
/// the detailed view). Same click-through contract as the tool HUD.
/// Bottom-right per cross-editor convention; top-right stays
/// reserved for a future orientation gizmo / view cube.
pub(super) fn show_perf_overlay(ctx: &Context, stats: &RenderStats) {
    let pos = ctx.available_rect().right_bottom() + egui::vec2(-12.0, -12.0);
    egui::Area::new(Id::new("viewport_perf_hud"))
        .pivot(Align2::RIGHT_BOTTOM)
        .fixed_pos(pos)
        .order(Order::Background)
        .interactable(false)
        .show(ctx, |ui| {
            hud_frame().show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 2.0;
                ui.label(
                    RichText::new(format!(
                        "{:.0} FPS · {:.1} ms",
                        stats.fps, stats.frame_time_ms
                    ))
                    .strong()
                    .color(Color32::from_gray(230)),
                );
                ui.label(
                    RichText::new(format!(
                        "{} tris · {} chunks",
                        compact_count(stats.triangles),
                        stats.chunks
                    ))
                    .color(Color32::from_gray(200)),
                );
                if let Some((ms, chunks)) = stats.last_rebuild {
                    ui.label(
                        RichText::new(format!(
                            "rebuild {:.1} ms ({} chunks)",
                            ms, chunks
                        ))
                        .color(Color32::from_gray(200)),
                    );
                }
            });
        });
}

/// Compact count for at-a-glance HUD use: `950` → `"950"`,
/// `12_345` → `"12.3k"`, `1_234_567` → `"1.23M"`. Precision is
/// deliberately coarse — the Statistics window keeps exact numbers.
pub fn compact_count(n: usize) -> String {
    if n < 1_000 {
        n.to_string()
    } else if n < 1_000_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    }
}

/// Face label for an axis index + sign, e.g. `(1, 1)` → `"Y+"`.
/// Axis indexing matches `StrokePlane::axis` (0 = X, 1 = Y, 2 = Z).
pub fn plane_label(axis: usize, sign: i32) -> &'static str {
    match (axis, sign >= 0) {
        (0, true) => "X+",
        (0, false) => "X-",
        (1, true) => "Y+",
        (1, false) => "Y-",
        (_, true) => "Z+",
        (_, false) => "Z-",
    }
}

/// Inclusive cell-box dimensions spanned by two corner cells, in
/// `(x, y, z)` order — the same order the `Sel:` readout uses.
pub fn drag_dims(a: (i32, i32, i32), b: (i32, i32, i32)) -> (i32, i32, i32) {
    (
        (a.0 - b.0).abs() + 1,
        (a.1 - b.1).abs() + 1,
        (a.2 - b.2).abs() + 1,
    )
}

/// `"4 × 1 × 3"`.
pub fn dims_label(d: (i32, i32, i32)) -> String {
    format!("{} × {} × {}", d.0, d.1, d.2)
}

/// `"Δ +3, +0, -2"` — explicit signs so a zero axis reads as
/// "no movement on this axis" rather than a stray number.
pub fn delta_label(d: (i32, i32, i32)) -> String {
    format!("Δ {:+}, {:+}, {:+}", d.0, d.1, d.2)
}

/// `"Sym: XZ"`, or `None` when no axis is active.
pub fn symmetry_label(sym: &SymmetryAxes) -> Option<String> {
    if !sym.any() {
        return None;
    }
    let mut axes = String::new();
    if sym.x {
        axes.push('X');
    }
    if sym.y {
        axes.push('Y');
    }
    if sym.z {
        axes.push('Z');
    }
    Some(format!("Sym: {}", axes))
}

/// `"Sel: 12×5×8 (480 cells)"` — same compact format as the status
/// bar so the two readouts are recognizably the same value.
pub fn selection_label(w: i32, h: i32, d: i32, cells: usize) -> String {
    format!("Sel: {}×{}×{} ({} cells)", w, h, d, cells)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_label_covers_all_axes_and_signs() {
        assert_eq!(plane_label(0, 1), "X+");
        assert_eq!(plane_label(0, -1), "X-");
        assert_eq!(plane_label(1, 1), "Y+");
        assert_eq!(plane_label(1, -1), "Y-");
        assert_eq!(plane_label(2, 1), "Z+");
        assert_eq!(plane_label(2, -1), "Z-");
    }

    #[test]
    fn drag_dims_is_inclusive_and_corner_order_free() {
        assert_eq!(drag_dims((2, 0, 3), (5, 0, 1)), (4, 1, 3));
        assert_eq!(drag_dims((5, 0, 1), (2, 0, 3)), (4, 1, 3));
        // Same cell → a 1×1×1 box, not zero.
        assert_eq!(drag_dims((7, 7, 7), (7, 7, 7)), (1, 1, 1));
    }

    #[test]
    fn delta_label_keeps_explicit_signs() {
        assert_eq!(delta_label((3, 0, -2)), "Δ +3, +0, -2");
    }

    #[test]
    fn symmetry_label_lists_active_axes_only() {
        let none = SymmetryAxes {
            x: false,
            y: false,
            z: false,
        };
        assert_eq!(symmetry_label(&none), None);

        let xz = SymmetryAxes {
            x: true,
            y: false,
            z: true,
        };
        assert_eq!(symmetry_label(&xz).as_deref(), Some("Sym: XZ"));
    }

    #[test]
    fn selection_label_matches_status_bar_format() {
        assert_eq!(selection_label(12, 5, 8, 480), "Sel: 12×5×8 (480 cells)");
    }

    #[test]
    fn compact_count_scales_units() {
        assert_eq!(compact_count(0), "0");
        assert_eq!(compact_count(999), "999");
        assert_eq!(compact_count(1_000), "1.0k");
        assert_eq!(compact_count(12_345), "12.3k");
        assert_eq!(compact_count(1_234_567), "1.23M");
    }
}
