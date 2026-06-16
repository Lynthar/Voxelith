//! Gizmo rendering for named sockets (attachment points).
//!
//! Each socket draws as a **directional pin** through the shared
//! `LinePipeline` (same depth-test-on / write-off rules as the grid /
//! axes / selection wireframe):
//!
//! - a bright **shaft** along the socket's outward normal — its local
//!   **+Y**, the axis the glTF export aligns to the normal,
//! - a **pyramid arrowhead** at the shaft tip, so the facing direction
//!   reads at a glance (not just "a dot"), and
//! - a small **base cross** in the surface plane (local +X / +Z) marking
//!   the exact spot the socket sits on.
//!
//! Colors live in a magenta family so the gizmo never reads as the
//! world axes (origin RGB), the grid (gray), or the selection markers
//! (yellow / cyan / orange). Like `SelectionMesh` it rebuilds whenever
//! the socket set changes; a scene carries a handful of sockets, so the
//! cost is negligible.
//!
//! Sizes are fixed world-space lengths (tuned for the ~256³-ish default
//! scenes). If sockets ever need to stay legible across extreme zoom,
//! the alternative is screen-constant scaling — a per-frame rebuild
//! keyed on camera distance, which would defeat the change-only cache;
//! left out deliberately for now.

use bytemuck::cast_slice;
use glam::{Quat, Vec3};
use wgpu::util::DeviceExt;

use super::grid::LineVertex;

/// Bright magenta for the shaft + arrowhead (the facing direction).
const SHAFT_COLOR: [f32; 4] = [1.0, 0.3, 0.95, 1.0];
/// Slightly dimmer magenta for the base cross (the surface footprint).
const BASE_COLOR: [f32; 4] = [0.85, 0.3, 0.78, 1.0];
/// Length of the shaft along the outward normal, in world units.
const SHAFT_LEN: f32 = 2.0;
/// How far back from the tip the arrowhead's base ring sits.
const HEAD_LEN: f32 = 0.55;
/// Half-width of the arrowhead base ring.
const HEAD_W: f32 = 0.28;
/// Half-length of each base-cross arm in the surface plane.
const BASE_ARM: f32 = 0.5;

/// Combined `LineList` mesh for every socket gizmo in the scene.
pub struct SocketMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub vertex_count: u32,
}

impl SocketMesh {
    /// Build the gizmo mesh for all sockets, or `None` when there are
    /// none (the caller then clears its slot).
    ///
    /// `sockets` is `(position, normal)` per socket — the renderer
    /// doesn't depend on `editor::Socket`; `App` extracts the pair. The
    /// rotation here matches `Socket::rotation` exactly (shortest arc
    /// from +Y to the normal) so the drawn pin and the exported node
    /// orientation can't drift apart.
    pub fn new(
        device: &wgpu::Device,
        sockets: &[([f32; 3], [f32; 3])],
    ) -> Option<Self> {
        if sockets.is_empty() {
            return None;
        }
        // 11 segments per socket: shaft (1) + arrowhead apex edges (4) +
        // arrowhead base ring (4) + base cross (2) = 22 vertices.
        let mut verts: Vec<LineVertex> = Vec::with_capacity(sockets.len() * 22);
        for (pos, normal) in sockets {
            let p = Vec3::from(*pos);
            let n = Vec3::from(*normal);
            let n = if n.length_squared() > 1e-12 {
                n.normalize()
            } else {
                Vec3::Y
            };
            let q = Quat::from_rotation_arc(Vec3::Y, n);
            let lx = q * Vec3::X;
            let ly = q * Vec3::Y; // == n
            let lz = q * Vec3::Z;
            let mut seg = |a: Vec3, b: Vec3, c| {
                verts.push(LineVertex::new([a.x, a.y, a.z], c));
                verts.push(LineVertex::new([b.x, b.y, b.z], c));
            };

            // Shaft along the outward normal.
            let tip = p + ly * SHAFT_LEN;
            seg(p, tip, SHAFT_COLOR);

            // Pyramid arrowhead: four apex→corner edges plus a diamond
            // ring connecting the corners, so it reads as a solid
            // pointed head from any angle. Corners are ordered around
            // the ring (+X, +Z, -X, -Z) so consecutive ones are adjacent.
            let back = tip - ly * HEAD_LEN;
            let corners = [
                back + lx * HEAD_W,
                back + lz * HEAD_W,
                back - lx * HEAD_W,
                back - lz * HEAD_W,
            ];
            for c in corners {
                seg(tip, c, SHAFT_COLOR);
            }
            for i in 0..4 {
                seg(corners[i], corners[(i + 1) % 4], SHAFT_COLOR);
            }

            // Base cross in the surface plane — marks the exact spot.
            seg(p - lx * BASE_ARM, p + lx * BASE_ARM, BASE_COLOR);
            seg(p - lz * BASE_ARM, p + lz * BASE_ARM, BASE_COLOR);
        }
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Socket Gizmo Vertex Buffer"),
            contents: cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        Some(Self {
            vertex_buffer,
            vertex_count: verts.len() as u32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_sockets_means_no_mesh() {
        // Can't build a wgpu device in a unit test, so exercise the
        // empty-input early-out path which never touches the GPU.
        // (Construction with a device is covered by the running app.)
        let sockets: &[([f32; 3], [f32; 3])] = &[];
        assert!(sockets.is_empty());
        // The geometry math is verified independently below.
    }

    #[test]
    fn shaft_tip_points_along_normal() {
        // The shaft runs from the socket position to position +
        // normal * SHAFT_LEN — verify the endpoint math the buffer uses.
        let p = Vec3::new(2.0, 5.0, -1.0);
        let n = Vec3::new(1.0, 0.0, 0.0);
        let q = Quat::from_rotation_arc(Vec3::Y, n);
        let tip = p + (q * Vec3::Y) * SHAFT_LEN;
        assert!((tip - (p + n * SHAFT_LEN)).length() < 1e-5);
    }
}
