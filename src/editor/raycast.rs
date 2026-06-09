//! Ray casting for voxel picking.
//!
//! Uses the DDA (Digital Differential Analyzer) algorithm for efficient
//! voxel traversal along a ray.

use crate::core::World;
use glam::{Mat4, Vec3, Vec4};

/// A ray in 3D space
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    /// Origin point of the ray
    pub origin: Vec3,
    /// Normalized direction of the ray
    pub direction: Vec3,
}

impl Ray {
    /// Create a new ray
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
        }
    }

    /// Create a ray from screen coordinates
    ///
    /// screen_pos: (x, y) in pixels from top-left
    /// screen_size: (width, height) in pixels
    /// view_proj_inv: inverse of view-projection matrix
    pub fn from_screen(
        screen_pos: (f32, f32),
        screen_size: (f32, f32),
        view_proj_inv: Mat4,
    ) -> Self {
        // Convert screen coordinates to NDC (-1 to 1)
        let ndc_x = (2.0 * screen_pos.0 / screen_size.0) - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_pos.1 / screen_size.1); // Flip Y

        // Create ray in clip space
        let near_point = Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let far_point = Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

        // Transform to world space
        let near_world = view_proj_inv * near_point;
        let far_world = view_proj_inv * far_point;

        // Perspective divide
        let near_world = near_world.truncate() / near_world.w;
        let far_world = far_world.truncate() / far_world.w;

        let direction = (far_world - near_world).normalize();

        Self {
            origin: near_world,
            direction,
        }
    }

    /// Get point along the ray at distance t
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

/// Result of a voxel raycast hit
#[derive(Debug, Clone, Copy)]
pub struct RaycastHit {
    /// World position of the hit voxel
    pub voxel_pos: (i32, i32, i32),
    /// Position of the adjacent empty voxel (for placing)
    pub adjacent_pos: (i32, i32, i32),
    /// Normal of the hit face (which face was hit)
    pub normal: (i32, i32, i32),
    /// Distance along the ray
    pub distance: f32,
    /// True when this hit was synthesized by `cast_with_ground_plane`
    /// because the ray missed every real voxel. Lets shape tools
    /// detect the empty-world case and substitute screen-space
    /// vertical drag for the missing Y axis (otherwise an empty-world
    /// drag is stuck flat on the plane and Sphere / Cylinder produce
    /// a disk).
    pub virtual_ground: bool,
}

/// Voxel raycaster using DDA algorithm
pub struct VoxelRaycast;

impl VoxelRaycast {
    /// Cast a ray through the voxel world and find the first solid voxel hit
    ///
    /// max_distance: Maximum distance to check (in voxel units)
    pub fn cast(ray: &Ray, world: &World, max_distance: f32) -> Option<RaycastHit> {
        // Current voxel position
        let mut x = ray.origin.x.floor() as i32;
        let mut y = ray.origin.y.floor() as i32;
        let mut z = ray.origin.z.floor() as i32;

        // Direction signs
        let step_x = if ray.direction.x > 0.0 { 1 } else { -1 };
        let step_y = if ray.direction.y > 0.0 { 1 } else { -1 };
        let step_z = if ray.direction.z > 0.0 { 1 } else { -1 };

        // How far along the ray we must move to cross a voxel boundary
        let t_delta_x = if ray.direction.x.abs() < 1e-10 {
            f32::INFINITY
        } else {
            (1.0 / ray.direction.x).abs()
        };
        let t_delta_y = if ray.direction.y.abs() < 1e-10 {
            f32::INFINITY
        } else {
            (1.0 / ray.direction.y).abs()
        };
        let t_delta_z = if ray.direction.z.abs() < 1e-10 {
            f32::INFINITY
        } else {
            (1.0 / ray.direction.z).abs()
        };

        // Distance to next boundary
        let mut t_max_x = if ray.direction.x > 0.0 {
            ((x as f32 + 1.0) - ray.origin.x) * t_delta_x
        } else if ray.direction.x < 0.0 {
            (ray.origin.x - x as f32) * t_delta_x
        } else {
            f32::INFINITY
        };

        let mut t_max_y = if ray.direction.y > 0.0 {
            ((y as f32 + 1.0) - ray.origin.y) * t_delta_y
        } else if ray.direction.y < 0.0 {
            (ray.origin.y - y as f32) * t_delta_y
        } else {
            f32::INFINITY
        };

        let mut t_max_z = if ray.direction.z > 0.0 {
            ((z as f32 + 1.0) - ray.origin.z) * t_delta_z
        } else if ray.direction.z < 0.0 {
            (ray.origin.z - z as f32) * t_delta_z
        } else {
            f32::INFINITY
        };

        // Track the last face we crossed (initialized to 0, updated during traversal)
        #[allow(unused_assignments)]
        let mut last_normal = (0, 0, 0);
        let mut distance = 0.0f32;

        // Check starting voxel
        if !world.get_voxel(x, y, z).is_air() {
            return Some(RaycastHit {
                voxel_pos: (x, y, z),
                adjacent_pos: (x, y, z), // Same position if we started inside
                normal: (0, 0, 0),
                distance: 0.0,
                virtual_ground: false,
            });
        }

        // DDA traversal
        while distance < max_distance {
            // Remember previous position for adjacent calculation
            let prev_x = x;
            let prev_y = y;
            let prev_z = z;

            // Step to next voxel boundary
            if t_max_x < t_max_y {
                if t_max_x < t_max_z {
                    x += step_x;
                    distance = t_max_x;
                    t_max_x += t_delta_x;
                    last_normal = (-step_x, 0, 0);
                } else {
                    z += step_z;
                    distance = t_max_z;
                    t_max_z += t_delta_z;
                    last_normal = (0, 0, -step_z);
                }
            } else {
                if t_max_y < t_max_z {
                    y += step_y;
                    distance = t_max_y;
                    t_max_y += t_delta_y;
                    last_normal = (0, -step_y, 0);
                } else {
                    z += step_z;
                    distance = t_max_z;
                    t_max_z += t_delta_z;
                    last_normal = (0, 0, -step_z);
                }
            }

            // Check if we hit a solid voxel
            if !world.get_voxel(x, y, z).is_air() {
                return Some(RaycastHit {
                    voxel_pos: (x, y, z),
                    adjacent_pos: (prev_x, prev_y, prev_z),
                    normal: last_normal,
                    distance,
                    virtual_ground: false,
                });
            }
        }

        None
    }

    /// Cast a ray, falling back to a virtual hit on the horizontal
    /// plane at `y = plane_y` when no solid voxel intercepts it.
    ///
    /// Used for the Place tool so the user can place voxels into a
    /// freshly-cleared (empty) world — without a fallback, raycast
    /// would miss everything and Place would have no anchor.
    ///
    /// The synthesized hit puts `voxel_pos` at `y = plane_y - 1` (a
    /// virtual sub-plane "ghost" anchor) and `adjacent_pos` on the
    /// plane itself, so a Place tool that writes at `adjacent_pos`
    /// lands directly on the plane. Other tools (Remove/Paint/Eyedrop/
    /// Fill) shouldn't call this — their semantics break on virtual
    /// hits — they should call [`Self::cast`] instead.
    ///
    /// Only fires when the camera is above the plane and looking down;
    /// looking sideways or up at the plane gives no synthetic hit
    /// (avoids the cursor "snapping" to the plane behind the user).
    pub fn cast_with_ground_plane(
        ray: &Ray,
        world: &World,
        max_distance: f32,
        plane_y: i32,
    ) -> Option<RaycastHit> {
        if let Some(hit) = Self::cast(ray, world, max_distance) {
            return Some(hit);
        }
        let plane_y_f = plane_y as f32;
        // Camera must be above the plane and the ray must head downward.
        // The 1e-6 epsilon catches near-parallel rays that would otherwise
        // intersect at huge `t` and snap the cursor far off-screen.
        if ray.origin.y <= plane_y_f || ray.direction.y >= -1e-6 {
            return None;
        }
        let t = (plane_y_f - ray.origin.y) / ray.direction.y;
        if t <= 0.0 || t > max_distance {
            return None;
        }
        let p = ray.at(t);
        let x = p.x.floor() as i32;
        let z = p.z.floor() as i32;
        Some(RaycastHit {
            voxel_pos: (x, plane_y - 1, z),
            adjacent_pos: (x, plane_y, z),
            normal: (0, 1, 0),
            distance: t,
            virtual_ground: true,
        })
    }

    /// Resolve an orbit pivot from a camera-forward `ray` (origin =
    /// camera position, direction = camera forward), Unity-style:
    ///
    /// 1. **Voxel surface** — first solid voxel the ray hits.
    /// 2. **Ground plane** — else the `y = 0` (XZ) intersection, but
    ///    only when it lies ahead (`t > 0`) and within `max_distance`.
    ///    A near-horizontal ray crosses `y = 0` only at a huge `t` (or
    ///    never), which the reach cap rejects so we don't pivot around a
    ///    point off at the horizon.
    /// 3. **`fallback`** — else the caller's current camera target. For
    ///    a forward ray the view-depth plane through the target meets
    ///    the ray exactly at the target, so this *is* the "view-depth
    ///    plane" fallback; returning the target leaves the pivot depth
    ///    unchanged and, crucially, keeps the press jump-free.
    ///
    /// The result (cases 1–2) always lies on `ray`, so moving the
    /// camera target onto it preserves the view direction — the orbit
    /// re-anchors without any visible camera jump, only `distance`
    /// changes. Returns a continuous world point, not a grid cell.
    pub fn orbit_pivot(ray: &Ray, world: &World, max_distance: f32, fallback: Vec3) -> Vec3 {
        if let Some(hit) = Self::cast(ray, world, max_distance) {
            return ray.at(hit.distance);
        }
        // Ground-plane intersection. `1e-4` rejects rays parallel
        // enough to y=0 that `t` would explode; the reach cap rejects
        // distant grazing hits. Both directions (looking down from
        // above, or up from below) are valid as long as the crossing is
        // ahead and within reach — unlike `cast_with_ground_plane`,
        // which is placement-oriented and only fires looking down.
        if ray.direction.y.abs() > 1e-4 {
            let t = -ray.origin.y / ray.direction.y;
            if t > 0.0 && t <= max_distance {
                return ray.at(t);
            }
        }
        fallback
    }

    /// Check if a voxel position is visible from camera
    pub fn is_visible(pos: (i32, i32, i32), camera_pos: Vec3, world: &World) -> bool {
        let voxel_center = Vec3::new(
            pos.0 as f32 + 0.5,
            pos.1 as f32 + 0.5,
            pos.2 as f32 + 0.5,
        );

        let ray = Ray::new(camera_pos, voxel_center - camera_pos);
        let distance = (voxel_center - camera_pos).length();

        if let Some(hit) = Self::cast(&ray, world, distance + 1.0) {
            hit.voxel_pos == pos
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Voxel;

    #[test]
    fn test_ray_at() {
        let ray = Ray::new(Vec3::ZERO, Vec3::X);
        assert!((ray.at(5.0) - Vec3::new(5.0, 0.0, 0.0)).length() < 0.001);
    }

    #[test]
    fn test_raycast_hit() {
        let mut world = World::new();
        world.set_voxel(5, 0, 0, Voxel::from_rgb(255, 0, 0));

        let ray = Ray::new(Vec3::ZERO, Vec3::X);
        let hit = VoxelRaycast::cast(&ray, &world, 100.0);

        assert!(hit.is_some());
        let hit = hit.unwrap();
        assert_eq!(hit.voxel_pos, (5, 0, 0));
        assert_eq!(hit.normal, (-1, 0, 0)); // Hit from negative X side
    }

    #[test]
    fn test_raycast_miss() {
        let world = World::new();
        let ray = Ray::new(Vec3::ZERO, Vec3::X);
        let hit = VoxelRaycast::cast(&ray, &world, 100.0);

        assert!(hit.is_none());
    }

    #[test]
    fn test_ground_plane_synthesizes_hit_when_world_empty() {
        let world = World::new();
        // Camera at (5, 10, 5), looking down toward origin (-1, -2, -1).
        let ray = Ray::new(Vec3::new(5.0, 10.0, 5.0), Vec3::new(-1.0, -2.0, -1.0));
        let hit = VoxelRaycast::cast_with_ground_plane(&ray, &world, 100.0, 0)
            .expect("ground plane fallback should fire on empty world");
        // Adjacent position lands on the plane (y = 0).
        assert_eq!(hit.adjacent_pos.1, 0);
        // Virtual voxel sits one cell below the plane.
        assert_eq!(hit.voxel_pos.1, -1);
        assert_eq!(hit.normal, (0, 1, 0));
    }

    #[test]
    fn test_ground_plane_skipped_when_ray_aims_upward() {
        let world = World::new();
        // Camera below or above plane, ray heading up — no synthesis.
        let ray = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::Y);
        assert!(VoxelRaycast::cast_with_ground_plane(&ray, &world, 100.0, 0).is_none());
    }

    #[test]
    fn test_ground_plane_skipped_when_camera_below_plane() {
        let world = World::new();
        // Origin below plane, ray heading down — no synthesis (would
        // hit plane behind / from underneath).
        let ray = Ray::new(Vec3::new(0.0, -5.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        assert!(VoxelRaycast::cast_with_ground_plane(&ray, &world, 100.0, 0).is_none());
    }

    #[test]
    fn test_real_voxel_hit_takes_precedence_over_ground_plane() {
        let mut world = World::new();
        world.set_voxel(0, 5, 0, Voxel::from_rgb(255, 0, 0));
        // Ray from above heading straight down through that voxel.
        let ray = Ray::new(Vec3::new(0.5, 10.0, 0.5), Vec3::new(0.0, -1.0, 0.0));
        let hit = VoxelRaycast::cast_with_ground_plane(&ray, &world, 100.0, 0).unwrap();
        // Real voxel hit, not the plane fallback.
        assert_eq!(hit.voxel_pos, (0, 5, 0));
    }

    // -------- orbit pivot (middle-mouse orbit re-anchor) --------

    #[test]
    fn orbit_pivot_returns_surface_point_on_voxel_hit() {
        let mut world = World::new();
        world.set_voxel(10, 0, 0, Voxel::from_rgb(1, 2, 3));
        // Forward ray straight down +X from origin; hits the voxel's
        // near (-X) face at x = 10, so the pivot sits on that face.
        let ray = Ray::new(Vec3::ZERO, Vec3::X);
        let pivot = VoxelRaycast::orbit_pivot(&ray, &world, 100.0, Vec3::splat(999.0));
        assert!(
            (pivot.x - 10.0).abs() < 1e-3,
            "pivot should land on the hit face at x=10, got {:?}",
            pivot
        );
        // Pivot lies on the ray (key property: no view-direction jump).
        assert!((pivot.y).abs() < 1e-3 && (pivot.z).abs() < 1e-3);
    }

    #[test]
    fn orbit_pivot_falls_back_to_ground_plane_when_empty() {
        let world = World::new();
        // Camera at y=10 looking down-forward; no voxels, so the pivot
        // is the y=0 crossing.
        let ray = Ray::new(Vec3::new(0.0, 10.0, 0.0), Vec3::new(1.0, -1.0, 0.0));
        let pivot = VoxelRaycast::orbit_pivot(&ray, &world, 100.0, Vec3::splat(999.0));
        assert!(
            pivot.y.abs() < 1e-3,
            "ground fallback should land on y=0, got {:?}",
            pivot
        );
        // 45° down over 10 units of height → x = 10 at the crossing.
        assert!((pivot.x - 10.0).abs() < 1e-3, "got {:?}", pivot);
    }

    #[test]
    fn orbit_pivot_keeps_target_when_ray_horizontal() {
        let world = World::new();
        // Perfectly horizontal forward ray never usefully meets y=0 →
        // fall back to the supplied target (no-jump view-depth plane).
        let fallback = Vec3::new(3.0, 4.0, 5.0);
        let ray = Ray::new(Vec3::new(0.0, 8.0, 0.0), Vec3::X);
        let pivot = VoxelRaycast::orbit_pivot(&ray, &world, 100.0, fallback);
        assert_eq!(pivot, fallback);
    }

    #[test]
    fn orbit_pivot_keeps_target_when_ground_beyond_reach() {
        let world = World::new();
        // Shallow downward ray crosses y=0 far past `max_distance`; the
        // reach cap rejects it so we don't pivot around the horizon.
        let fallback = Vec3::new(-1.0, -2.0, -3.0);
        let ray = Ray::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(1.0, -0.01, 0.0));
        let pivot = VoxelRaycast::orbit_pivot(&ray, &world, 50.0, fallback);
        assert_eq!(pivot, fallback);
    }
}
