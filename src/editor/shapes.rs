//! Voxel position generators for the click-and-drag shape tools.
//!
//! Each function takes two integer cell positions (anchor + drag-end)
//! and returns the set of cells the shape covers. The functions are
//! pure — no world access, no symmetry expansion. Symmetry mirroring
//! is layered on top by the caller (input.rs's shape commit path).
//!
//! All shapes are inclusive on both endpoints: a Line from `a` to `b`
//! contains `a` and `b`; a Box from `a` to `b` covers the closed AABB
//! including both corners. This matches user intuition ("I dragged
//! from here to here, I expect both ends to be voxels").

/// 3D Bresenham line. Visits one cell per step along the dominant
/// axis, with two error terms tracking when to step on the other
/// two axes. Output is ordered along the line (`a` first, `b` last)
/// — useful for animation / progressive reveal, but order doesn't
/// matter for the eventual `Command::set_voxels` since the
/// underlying world write is set-semantics.
pub fn line_voxels(a: (i32, i32, i32), b: (i32, i32, i32)) -> Vec<(i32, i32, i32)> {
    let (mut x, mut y, mut z) = a;
    let dx = (b.0 - a.0).abs();
    let dy = (b.1 - a.1).abs();
    let dz = (b.2 - a.2).abs();
    let sx = (b.0 - a.0).signum();
    let sy = (b.1 - a.1).signum();
    let sz = (b.2 - a.2).signum();

    // Pre-size to the dominant axis length + 1 (plus a small margin
    // for the worst-case path; over-allocation here is fine).
    let max_d = dx.max(dy).max(dz) as usize + 1;
    let mut out = Vec::with_capacity(max_d);
    out.push((x, y, z));

    if dx >= dy && dx >= dz {
        // X is dominant.
        let mut p1 = 2 * dy - dx;
        let mut p2 = 2 * dz - dx;
        while x != b.0 {
            x += sx;
            if p1 >= 0 {
                y += sy;
                p1 -= 2 * dx;
            }
            if p2 >= 0 {
                z += sz;
                p2 -= 2 * dx;
            }
            p1 += 2 * dy;
            p2 += 2 * dz;
            out.push((x, y, z));
        }
    } else if dy >= dx && dy >= dz {
        // Y is dominant.
        let mut p1 = 2 * dx - dy;
        let mut p2 = 2 * dz - dy;
        while y != b.1 {
            y += sy;
            if p1 >= 0 {
                x += sx;
                p1 -= 2 * dy;
            }
            if p2 >= 0 {
                z += sz;
                p2 -= 2 * dy;
            }
            p1 += 2 * dx;
            p2 += 2 * dz;
            out.push((x, y, z));
        }
    } else {
        // Z is dominant.
        let mut p1 = 2 * dy - dz;
        let mut p2 = 2 * dx - dz;
        while z != b.2 {
            z += sz;
            if p1 >= 0 {
                y += sy;
                p1 -= 2 * dz;
            }
            if p2 >= 0 {
                x += sx;
                p2 -= 2 * dz;
            }
            p1 += 2 * dy;
            p2 += 2 * dx;
            out.push((x, y, z));
        }
    }
    out
}

/// Closed axis-aligned bounding box from `a` to `b` (any pair of
/// opposite corners). Inclusive on both endpoints.
pub fn box_voxels(a: (i32, i32, i32), b: (i32, i32, i32)) -> Vec<(i32, i32, i32)> {
    let (x0, x1) = (a.0.min(b.0), a.0.max(b.0));
    let (y0, y1) = (a.1.min(b.1), a.1.max(b.1));
    let (z0, z1) = (a.2.min(b.2), a.2.max(b.2));
    let count = ((x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1)) as usize;
    let mut out = Vec::with_capacity(count);
    for z in z0..=z1 {
        for y in y0..=y1 {
            for x in x0..=x1 {
                out.push((x, y, z));
            }
        }
    }
    out
}

/// Filled ellipsoid fitting in the closed AABB `[a, b]`. Square-ish
/// drag → sphere; oblong drag → ellipsoid. Test is at cell centers
/// (offset 0.5) so a 1-cell-thick AABB still emits voxels.
pub fn sphere_voxels(a: (i32, i32, i32), b: (i32, i32, i32)) -> Vec<(i32, i32, i32)> {
    let (x0, x1) = (a.0.min(b.0), a.0.max(b.0));
    let (y0, y1) = (a.1.min(b.1), a.1.max(b.1));
    let (z0, z1) = (a.2.min(b.2), a.2.max(b.2));

    let cx = (x0 as f32 + x1 as f32 + 1.0) * 0.5;
    let cy = (y0 as f32 + y1 as f32 + 1.0) * 0.5;
    let cz = (z0 as f32 + z1 as f32 + 1.0) * 0.5;
    let rx = ((x1 - x0) as f32 + 1.0) * 0.5;
    let ry = ((y1 - y0) as f32 + 1.0) * 0.5;
    let rz = ((z1 - z0) as f32 + 1.0) * 0.5;

    let mut out = Vec::new();
    for z in z0..=z1 {
        for y in y0..=y1 {
            for x in x0..=x1 {
                let dx = (x as f32 + 0.5 - cx) / rx;
                let dy = (y as f32 + 0.5 - cy) / ry;
                let dz = (z as f32 + 0.5 - cz) / rz;
                if dx * dx + dy * dy + dz * dz <= 1.0 {
                    out.push((x, y, z));
                }
            }
        }
    }
    out
}

/// Filled cylinder fitting in the closed AABB `[a, b]`. The cylinder's
/// axis runs along whichever bbox dimension is largest (ties broken
/// in favor of Y, then X, then Z — matches the most common voxel-art
/// usage of "tall cylinder pillars"). The cross-section perpendicular
/// to the axis is an ellipse spanning the other two dimensions.
pub fn cylinder_voxels(a: (i32, i32, i32), b: (i32, i32, i32)) -> Vec<(i32, i32, i32)> {
    let (x0, x1) = (a.0.min(b.0), a.0.max(b.0));
    let (y0, y1) = (a.1.min(b.1), a.1.max(b.1));
    let (z0, z1) = (a.2.min(b.2), a.2.max(b.2));
    let dx = x1 - x0;
    let dy = y1 - y0;
    let dz = z1 - z0;

    // 0 = X, 1 = Y, 2 = Z. Tie-break order: Y, X, Z.
    let axis = if dy >= dx && dy >= dz {
        1
    } else if dx >= dz {
        0
    } else {
        2
    };

    let mut out = Vec::new();
    for z in z0..=z1 {
        for y in y0..=y1 {
            for x in x0..=x1 {
                // Always at least 1.0 so a 1-cell-thick cross-section
                // doesn't divide by zero — every cell counts as inside.
                let inside = match axis {
                    0 => {
                        let cy = (y0 as f32 + y1 as f32 + 1.0) * 0.5;
                        let cz = (z0 as f32 + z1 as f32 + 1.0) * 0.5;
                        let ry = (((y1 - y0) as f32 + 1.0) * 0.5).max(0.5);
                        let rz = (((z1 - z0) as f32 + 1.0) * 0.5).max(0.5);
                        let dy = (y as f32 + 0.5 - cy) / ry;
                        let dz = (z as f32 + 0.5 - cz) / rz;
                        dy * dy + dz * dz <= 1.0
                    }
                    1 => {
                        let cx = (x0 as f32 + x1 as f32 + 1.0) * 0.5;
                        let cz = (z0 as f32 + z1 as f32 + 1.0) * 0.5;
                        let rx = (((x1 - x0) as f32 + 1.0) * 0.5).max(0.5);
                        let rz = (((z1 - z0) as f32 + 1.0) * 0.5).max(0.5);
                        let dx = (x as f32 + 0.5 - cx) / rx;
                        let dz = (z as f32 + 0.5 - cz) / rz;
                        dx * dx + dz * dz <= 1.0
                    }
                    _ => {
                        let cx = (x0 as f32 + x1 as f32 + 1.0) * 0.5;
                        let cy = (y0 as f32 + y1 as f32 + 1.0) * 0.5;
                        let rx = (((x1 - x0) as f32 + 1.0) * 0.5).max(0.5);
                        let ry = (((y1 - y0) as f32 + 1.0) * 0.5).max(0.5);
                        let dx = (x as f32 + 0.5 - cx) / rx;
                        let dy = (y as f32 + 0.5 - cy) / ry;
                        dx * dx + dy * dy <= 1.0
                    }
                };
                if inside {
                    out.push((x, y, z));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_line_single_cell_when_endpoints_match() {
        assert_eq!(line_voxels((3, 5, 7), (3, 5, 7)), vec![(3, 5, 7)]);
    }

    #[test]
    fn test_line_horizontal_visits_every_cell() {
        let v = line_voxels((0, 0, 0), (4, 0, 0));
        assert_eq!(v, vec![(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)]);
    }

    #[test]
    fn test_line_diagonal_3d_no_gaps() {
        // 5-3-2 line: should have len = max(5,3,2)+1 = 6 cells.
        let v = line_voxels((0, 0, 0), (5, 3, 2));
        assert_eq!(v.len(), 6);
        assert_eq!(v.first(), Some(&(0, 0, 0)));
        assert_eq!(v.last(), Some(&(5, 3, 2)));
        // Each step's cell should be adjacent (chebyshev distance 1) to
        // the next — proves no gaps.
        for w in v.windows(2) {
            let d = (
                (w[1].0 - w[0].0).abs(),
                (w[1].1 - w[0].1).abs(),
                (w[1].2 - w[0].2).abs(),
            );
            assert!(
                d.0 <= 1 && d.1 <= 1 && d.2 <= 1,
                "non-adjacent step from {:?} to {:?}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_line_negative_direction() {
        let v = line_voxels((5, 5, 5), (1, 1, 1));
        assert_eq!(v.first(), Some(&(5, 5, 5)));
        assert_eq!(v.last(), Some(&(1, 1, 1)));
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_box_inclusive_corners() {
        let v = box_voxels((0, 0, 0), (1, 1, 1));
        // 2×2×2 = 8 cells.
        assert_eq!(v.len(), 8);
        let set: HashSet<_> = v.into_iter().collect();
        assert!(set.contains(&(0, 0, 0)));
        assert!(set.contains(&(1, 1, 1)));
        assert!(set.contains(&(1, 0, 0)));
        assert!(set.contains(&(0, 1, 1)));
    }

    #[test]
    fn test_box_unordered_corners() {
        // Either pair of opposite corners produces the same set.
        let a = box_voxels((0, 0, 0), (3, 2, 1));
        let b = box_voxels((3, 2, 1), (0, 0, 0));
        let set_a: HashSet<_> = a.into_iter().collect();
        let set_b: HashSet<_> = b.into_iter().collect();
        assert_eq!(set_a, set_b);
        assert_eq!(set_a.len(), 4 * 3 * 2);
    }

    #[test]
    fn test_sphere_single_cell_when_endpoints_match() {
        // 1×1×1 bbox: only one cell, must be inside.
        let v = sphere_voxels((5, 5, 5), (5, 5, 5));
        assert_eq!(v, vec![(5, 5, 5)]);
    }

    #[test]
    fn test_sphere_uniform_radius_keeps_corners_off() {
        // 5×5×5 bbox sphere: the center cell is always in; the eight
        // bbox corners should be OUT (corner distance > radius).
        let v = sphere_voxels((0, 0, 0), (4, 4, 4));
        let set: HashSet<_> = v.into_iter().collect();
        assert!(set.contains(&(2, 2, 2))); // center
        assert!(!set.contains(&(0, 0, 0))); // corner
        assert!(!set.contains(&(4, 4, 4)));
        assert!(!set.contains(&(0, 0, 4)));
    }

    #[test]
    fn test_cylinder_axis_along_largest_dimension() {
        // Y is dominant: cell (cx, ymid, cz) must be inside; cells at
        // bbox X corners with mid Y/Z should be ON the boundary or
        // just outside (they're at the extreme of the cross-section).
        let v = cylinder_voxels((0, 0, 0), (2, 10, 2));
        let set: HashSet<_> = v.into_iter().collect();
        // Center column is fully inside.
        for y in 0..=10 {
            assert!(set.contains(&(1, y, 1)), "missing center cell at y={}", y);
        }
    }

    #[test]
    fn test_cylinder_thin_dimension_does_not_panic() {
        // 1-cell-thick cross-section is the kind of degenerate case
        // that often divides by zero. Must produce a non-empty result.
        let v = cylinder_voxels((0, 0, 0), (0, 5, 0));
        assert!(!v.is_empty());
        // Every cell along the line is included.
        for y in 0..=5 {
            assert!(v.contains(&(0, y, 0)), "missing y={}", y);
        }
    }
}
