//! Hand-painted toolbar icons, drawn with `egui::Painter` primitives:
//! no icon-font dependency, crisp at any scale, and colored by widget
//! state. The `match` is total over [`Tool`]; glyphs use the unit square.

use std::f32::consts::TAU;

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
        // Place's voxel with a minus: the pair differs only in + and −.
        Tool::Remove => {
            painter.rect_filled(Rect::from_two_pos(p(0.06, 0.34), p(0.66, 0.94)), 1.0, color);
            painter.line_segment([p(0.64, 0.24), p(1.0, 0.24)], stroke);
        }
        // Paint roller: drum, bent arm, grip.
        Tool::Paint => {
            painter.rect_filled(Rect::from_two_pos(p(0.08, 0.08), p(0.8, 0.3)), 1.0, color);
            painter.add(Shape::line(
                vec![
                    p(0.8, 0.19),
                    p(0.93, 0.19),
                    p(0.93, 0.45),
                    p(0.5, 0.45),
                    p(0.5, 0.6),
                ],
                stroke,
            ));
            painter.rect_filled(Rect::from_two_pos(p(0.43, 0.6), p(0.57, 0.96)), 1.0, color);
        }
        // Pipette: square bulb, collar, stem, drop. The collar and bulb
        // are what keep it from reading as the Line tool's two dots.
        Tool::Eyedropper => {
            painter.add(Shape::convex_polygon(
                vec![
                    p(0.822, 0.022),
                    p(0.978, 0.178),
                    p(0.738, 0.418),
                    p(0.582, 0.262),
                ],
                color,
                Stroke::NONE,
            ));
            painter.line_segment([p(0.5, 0.26), p(0.74, 0.5)], stroke);
            painter.line_segment([p(0.6, 0.4), p(0.17, 0.83)], stroke);
            painter.circle_filled(p(0.09, 0.93), rect.width() * 0.065, color);
        }
        // A 3×3 grid with one connected region filled: what a fill does.
        Tool::Fill => {
            let (lo, hi) = (0.06, 0.94);
            let [a, b] = [lo + (hi - lo) / 3.0, lo + (hi - lo) * 2.0 / 3.0];
            painter.rect_filled(Rect::from_two_pos(p(lo, b), p(hi, hi)), 0.0, color);
            painter.rect_filled(Rect::from_two_pos(p(b, a), p(hi, b)), 0.0, color);
            painter.rect_stroke(Rect::from_two_pos(p(lo, lo), p(hi, hi)), 0.0, stroke);
            for t in [a, b] {
                painter.line_segment([p(t, lo), p(t, hi)], stroke);
                painter.line_segment([p(lo, t), p(hi, t)], stroke);
            }
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
