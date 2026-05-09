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
        let vertices = build_aabb_lines(min, max);
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
}
