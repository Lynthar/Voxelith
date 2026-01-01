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
                });
            }
        }

        None
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
}
