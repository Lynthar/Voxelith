//! Wireframe AABB rendering for the box-select tool.
//!
//! Renders the 12 edges of an axis-aligned box as `LineList` primitives
//! through the existing `LinePipeline` (so it shares the grid/axis
//! depth + blend rules). Unlike `GridMesh` / `AxisMesh` which build
//! once at startup, `SelectionMesh` rebuilds whenever the selection
//! AABB changes — 24 vertices is small enough that the cost is
//! negligible per frame.

use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

use super::grid::LineVertex;

/// Bright yellow with full alpha — chosen to stand out against the
/// existing grid (gray) and axes (RGB) without colliding with either.
const SELECTION_COLOR: [f32; 4] = [1.0, 0.9, 0.1, 1.0];

/// Cyan crosshair at the selection's geometric center — the reference
/// `Frame Sel.` zooms to and the plane mirror flips across.
const CENTER_COLOR: [f32; 4] = [0.2, 0.95, 1.0, 1.0];

/// Orange tripod at the `sel.min` corner — the anchor 90° rotation
/// keeps fixed (the AABB reshapes *from* this corner; see
/// `editor::transform`'s "min stays put" convention).
const ANCHOR_COLOR: [f32; 4] = [1.0, 0.5, 0.1, 1.0];

/// 12-edge wireframe mesh covering one closed-AABB selection.
///
/// The mesh extends from `min` to `max + 1` in world units so the
/// rendered box envelops the *outer faces* of the corner cells.
/// (A selection containing one cell at `(3, 3, 3)` spans the cube
/// from `(3, 3, 3)` to `(4, 4, 4)` in world space.)
pub struct SelectionMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub vertex_count: u32,
}

impl SelectionMesh {
    pub fn new(
        device: &wgpu::Device,
        min: (i32, i32, i32),
        max: (i32, i32, i32),
    ) -> Self {
        let mut vertices = build_aabb_lines(min, max);
        // Append the center crosshair + min-corner anchor markers so the
        // user can see where mirror flips (center) and where rotation
        // pins (the `sel.min` corner — see `editor::transform`).
        vertices.extend(build_markers(min, max));
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Selection Vertex Buffer"),
            contents: cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        Self {
            vertex_buffer,
            vertex_count: vertices.len() as u32,
        }
    }
}

fn build_aabb_lines(min: (i32, i32, i32), max: (i32, i32, i32)) -> Vec<LineVertex> {
    let x0 = min.0 as f32;
    let y0 = min.1 as f32;
    let z0 = min.2 as f32;
    let x1 = (max.0 + 1) as f32;
    let y1 = (max.1 + 1) as f32;
    let z1 = (max.2 + 1) as f32;
    let c = SELECTION_COLOR;
    let v = LineVertex::new;

    vec![
        // Bottom face (y = y0): 4 edges.
        v([x0, y0, z0], c), v([x1, y0, z0], c),
        v([x1, y0, z0], c), v([x1, y0, z1], c),
        v([x1, y0, z1], c), v([x0, y0, z1], c),
        v([x0, y0, z1], c), v([x0, y0, z0], c),
        // Top face (y = y1): 4 edges.
        v([x0, y1, z0], c), v([x1, y1, z0], c),
        v([x1, y1, z0], c), v([x1, y1, z1], c),
        v([x1, y1, z1], c), v([x0, y1, z1], c),
        v([x0, y1, z1], c), v([x0, y1, z0], c),
        // 4 vertical edges between the two faces.
        v([x0, y0, z0], c), v([x0, y1, z0], c),
        v([x1, y0, z0], c), v([x1, y1, z0], c),
        v([x1, y0, z1], c), v([x1, y1, z1], c),
        v([x0, y0, z1], c), v([x0, y1, z1], c),
    ]
}

/// Center crosshair + min-corner anchor tripod for a selection AABB.
///
/// - **Center crosshair** (cyan): three axis-aligned segments through
///   the box's geometric center — the mirror plane's center and where
///   `Frame Sel.` aims. Arm length scales with the smallest box extent,
///   clamped so it stays readable for both 1-cell and large selections.
/// - **Anchor tripod** (orange): three short legs from the `sel.min`
///   corner along +X/+Y/+Z — the corner a 90° rotation keeps fixed.
///
/// Returns 12 vertices (6 per marker), appended after the 24 box-edge
/// vertices. Shares the box's `LinePipeline` (depth-test on / write
/// off), so a marker tucked inside solid voxels is occluded like the
/// box edges — visible whenever its cell isn't behind geometry.
fn build_markers(min: (i32, i32, i32), max: (i32, i32, i32)) -> Vec<LineVertex> {
    let x0 = min.0 as f32;
    let y0 = min.1 as f32;
    let z0 = min.2 as f32;
    let x1 = (max.0 + 1) as f32;
    let y1 = (max.1 + 1) as f32;
    let z1 = (max.2 + 1) as f32;
    let (cx, cy, cz) = ((x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5);
    let smallest = (x1 - x0).min(y1 - y0).min(z1 - z0);
    let arm = (smallest * 0.25).clamp(0.5, 2.0);
    let leg = (smallest * 0.30).clamp(0.75, 2.5);
    let v = LineVertex::new;
    let cc = CENTER_COLOR;
    let ac = ANCHOR_COLOR;

    vec![
        // Center crosshair: one segment per axis through the center.
        v([cx - arm, cy, cz], cc), v([cx + arm, cy, cz], cc),
        v([cx, cy - arm, cz], cc), v([cx, cy + arm, cz], cc),
        v([cx, cy, cz - arm], cc), v([cx, cy, cz + arm], cc),
        // Anchor tripod: legs from the min corner along +X/+Y/+Z.
        v([x0, y0, z0], ac), v([x0 + leg, y0, z0], ac),
        v([x0, y0, z0], ac), v([x0, y0 + leg, z0], ac),
        v([x0, y0, z0], ac), v([x0, y0, z0 + leg], ac),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_has_24_vertices_for_12_edges() {
        // 12 edges × 2 vertices per LineList edge = 24 vertices.
        let v = build_aabb_lines((0, 0, 0), (3, 3, 3));
        assert_eq!(v.len(), 24);
    }

    #[test]
    fn aabb_extends_to_outer_face() {
        // A 1×1×1 selection at (3, 3, 3) should span world coords
        // (3,3,3) to (4,4,4) — outer face of the cell.
        let v = build_aabb_lines((3, 3, 3), (3, 3, 3));
        let xs: Vec<f32> = v.iter().map(|lv| lv.position[0]).collect();
        assert!(xs.contains(&3.0));
        assert!(xs.contains(&4.0));
        assert!(!xs.iter().any(|&x| x < 3.0 || x > 4.0));
    }

    #[test]
    fn markers_center_crosshair_and_min_anchor() {
        // 4×4×4 at origin → world AABB (0,0,0)..(4,4,4), center (2,2,2),
        // min corner (0,0,0).
        let m = build_markers((0, 0, 0), (3, 3, 3));
        assert_eq!(m.len(), 12, "6 crosshair + 6 tripod vertices");

        // Crosshair (first 6): every vertex lies on a center axis, i.e.
        // matches the center (2,2,2) on at least two coordinates.
        let center = [2.0_f32, 2.0, 2.0];
        for lv in &m[0..6] {
            let on_axis = (0..3)
                .filter(|&i| (lv.position[i] - center[i]).abs() < 1e-6)
                .count();
            assert!(on_axis >= 2, "crosshair vertex {:?} off-center", lv.position);
        }

        // Tripod (last 6): three legs, each starting exactly at the min
        // corner (the even vertex of each pair).
        for leg in 0..3 {
            assert_eq!(
                m[6 + leg * 2].position,
                [0.0, 0.0, 0.0],
                "tripod leg {} should start at the min corner",
                leg
            );
        }
    }

    #[test]
    fn full_selection_mesh_vertex_count() {
        // SelectionMesh::new concatenates box (24) + markers (12).
        let total = build_aabb_lines((0, 0, 0), (5, 2, 7)).len()
            + build_markers((0, 0, 0), (5, 2, 7)).len();
        assert_eq!(total, 36);
    }
}
