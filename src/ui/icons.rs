//! Hand-painted vector icons for the toolbar.
//!
//! Drawn with `egui::Painter` primitives instead of an icon font or
//! bitmap assets: no new dependency, crisp at any scale factor, and
//! the glyphs inherit whatever text color the widget state calls for
//! (hover / selected / disabled) so they never fight the theme. The
//! `match` is total over [`Tool`], so a new tool can't reach the
//! toolbar without an icon — the letter-glyph era ended when the
//! toolbar became the only place to pick one.
//!
//! Every glyph is authored in the unit square and mapped through
//! `rect`, so the caller decides the size; the toolbar hands in its
//! button rect minus padding.

use egui::{vec2, Color32, Painter, Pos2, Rect, Shape, Stroke};

use crate::editor::Tool;

/// Paint `tool`'s icon into `rect` in `color`.
pub fn paint_tool_icon(painter: &Painter, rect: Rect, tool: Tool, color: Color32) {
    let p = |x: f32, y: f32| rect.lerp_inside(vec2(x, y));
    let stroke = Stroke::new((rect.width() * 0.09).max(1.4), color);
    match tool {
        // A voxel with a small plus beside it.
        Tool::Place => {
            painter.rect_filled(Rect::from_two_pos(p(0.06, 0.34), p(0.66, 0.94)), 1.0, color);
            painter.line_segment([p(0.82, 0.06), p(0.82, 0.42)], stroke);
            painter.line_segment([p(0.64, 0.24), p(1.0, 0.24)], stroke);
        }
        // The hollow counterpart with a minus.
        Tool::Remove => {
            painter.rect_stroke(
                Rect::from_two_pos(p(0.06, 0.34), p(0.66, 0.94)),
                1.0,
                stroke,
            );
            painter.line_segment([p(0.64, 0.24), p(1.0, 0.24)], stroke);
        }
        // Paintbrush: diagonal handle, bristle wedge at the tip.
        Tool::Paint => {
            painter.line_segment([p(0.88, 0.12), p(0.5, 0.5)], stroke);
            painter.add(Shape::convex_polygon(
                vec![p(0.55, 0.32), p(0.68, 0.45), p(0.12, 0.88)],
                color,
                Stroke::NONE,
            ));
        }
        // Pipette: bulb, stem, and the sampled drop by the tip.
        Tool::Eyedropper => {
            painter.circle_filled(p(0.78, 0.22), rect.width() * 0.17, color);
            painter.line_segment([p(0.68, 0.32), p(0.22, 0.78)], stroke);
            painter.circle_filled(p(0.12, 0.92), rect.width() * 0.08, color);
        }
        // A cell filling up, with the drop that's doing it.
        Tool::Fill => {
            let cell = Rect::from_two_pos(p(0.1, 0.36), p(0.78, 0.96));
            painter.rect_stroke(cell, 1.0, stroke);
            painter.rect_filled(Rect::from_two_pos(p(0.1, 0.66), p(0.78, 0.96)), 0.0, color);
            painter.circle_filled(p(0.88, 0.16), rect.width() * 0.11, color);
        }
        // A segment with its two endpoints.
        Tool::Line => {
            painter.line_segment([p(0.14, 0.86), p(0.86, 0.14)], stroke);
            painter.circle_filled(p(0.14, 0.86), rect.width() * 0.1, color);
            painter.circle_filled(p(0.86, 0.14), rect.width() * 0.1, color);
        }
        // Wireframe cube — it makes a 3D box, so draw one: the front
        // face, the offset back face, and the four connecting edges.
        Tool::Box => {
            let front = Rect::from_two_pos(p(0.04, 0.32), p(0.68, 0.96));
            let back = Rect::from_two_pos(p(0.32, 0.04), p(0.96, 0.68));
            painter.rect_stroke(front, 0.0, stroke);
            painter.rect_stroke(back, 0.0, stroke);
            painter.line_segment([front.left_top(), back.left_top()], stroke);
            painter.line_segment([front.right_top(), back.right_top()], stroke);
            painter.line_segment([front.left_bottom(), back.left_bottom()], stroke);
            painter.line_segment([front.right_bottom(), back.right_bottom()], stroke);
        }
        // Circle with a flattened equator to say "ball, not disc".
        Tool::Sphere => {
            let c = p(0.5, 0.5);
            let r = rect.width() * 0.44;
            painter.circle_stroke(c, r, stroke);
            let equator: Vec<Pos2> = (0..=12)
                .map(|i| {
                    let t = std::f32::consts::PI * (i as f32 / 12.0);
                    Pos2::new(c.x - t.cos() * r, c.y + t.sin() * r * 0.35)
                })
                .collect();
            painter.add(Shape::line(equator, stroke));
        }
        // Side view: elliptical cap, straight walls, bottom arc.
        Tool::Cylinder => {
            let (rx, ry) = (0.36, 0.14);
            let ellipse = |cy: f32, from: f32, to: f32, n: usize| -> Vec<Pos2> {
                (0..=n)
                    .map(|i| {
                        let t = from + (to - from) * (i as f32 / n as f32);
                        p(0.5 + t.cos() * rx, cy + t.sin() * ry)
                    })
                    .collect()
            };
            use std::f32::consts::TAU;
            painter.add(Shape::closed_line(ellipse(0.2, 0.0, TAU, 16), stroke));
            painter.line_segment([p(0.5 - rx, 0.2), p(0.5 - rx, 0.8)], stroke);
            painter.line_segment([p(0.5 + rx, 0.2), p(0.5 + rx, 0.8)], stroke);
            painter.add(Shape::line(ellipse(0.8, 0.0, TAU / 2.0, 8), stroke));
        }
        // Marquee: a dashed rectangle.
        Tool::Select => {
            let r = Rect::from_two_pos(p(0.08, 0.14), p(0.92, 0.86));
            for [a, b] in [
                [r.left_top(), r.right_top()],
                [r.right_top(), r.right_bottom()],
                [r.right_bottom(), r.left_bottom()],
                [r.left_bottom(), r.left_top()],
            ] {
                painter.extend(Shape::dashed_line(
                    &[a, b],
                    stroke,
                    rect.width() * 0.14,
                    rect.width() * 0.1,
                ));
            }
        }
        // Anchor: ring, shaft, crossbar, flukes.
        Tool::Socket => {
            painter.circle_stroke(p(0.5, 0.14), rect.width() * 0.1, stroke);
            painter.line_segment([p(0.5, 0.24), p(0.5, 0.88)], stroke);
            painter.line_segment([p(0.32, 0.4), p(0.68, 0.4)], stroke);
            let flukes: Vec<Pos2> = (0..=8)
                .map(|i| {
                    // 200° → 340°: the upward-open arc under the shaft.
                    let t = (200.0 + 140.0 * (i as f32 / 8.0)).to_radians();
                    p(0.5 + t.cos() * 0.32, 0.62 - t.sin() * 0.3)
                })
                .collect();
            painter.add(Shape::line(flukes, stroke));
        }
    }
}
