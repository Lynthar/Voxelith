//! L-System tree generator using a 3D turtle.
//!
//! The L-system rewrites `AXIOM` by `RULE_F` for `iterations` rounds,
//! then a turtle interprets the resulting string. Push/pop (`[` / `]`)
//! manages branch stack; on each push we apply a random roll seeded
//! from `self.seed` so sibling branches don't collapse into a plane.
//! `F` rasterizes a 3D Bresenham line into the patch as trunk; on
//! every `]` we drop a small leaf cluster at the branch tip.

use glam::{Mat3, Vec3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::core::Voxel;

use super::{
    GenError, GenResult, GeneratorBackend, GeneratorCategory, GeneratorMeta,
    VoxelGenerator, VoxelPatch,
};

/// Plant L-system grown by a 3D turtle.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LSystemTree {
    pub seed: u32,
    /// Rewrite rounds. Each round multiplies string length by ~5×, so
    /// values above ~6 generate hundreds of thousands of voxels.
    pub iterations: u32,
    /// Branch turn angle for `+` / `-` / `/` / `\` (degrees).
    pub angle_deg: f32,
    /// Initial segment length in voxels (top-level branches).
    pub initial_length: f32,
    /// Length multiplier applied on each `[` push (e.g. 0.8 = sub-branches
    /// 80% as long as their parent).
    pub length_scale: f32,
    /// Where the trunk emerges in world space.
    pub origin: (i32, i32, i32),
    pub trunk_color: [u8; 3],
    pub leaf_color: [u8; 3],
}

impl Default for LSystemTree {
    fn default() -> Self {
        Self {
            seed: 7,
            iterations: 4,
            angle_deg: 25.0,
            initial_length: 4.0,
            length_scale: 0.8,
            origin: (0, 0, 0),
            trunk_color: [101, 67, 33],
            leaf_color: [60, 130, 50],
        }
    }
}

const AXIOM: &str = "F";
// Classic plant rule: F -> FF+[+F-F-F]-[-F+F+F]
const RULE_F: &str = "FF+[+F-F-F]-[-F+F+F]";

const MAX_ITERATIONS: u32 = 7;

fn rewrite(axiom: &str, iterations: u32) -> String {
    let mut s = axiom.to_string();
    for _ in 0..iterations {
        let mut next = String::with_capacity(s.len() * 5);
        for ch in s.chars() {
            if ch == 'F' {
                next.push_str(RULE_F);
            } else {
                next.push(ch);
            }
        }
        s = next;
    }
    s
}

#[derive(Clone)]
struct TurtleState {
    position: Vec3,
    direction: Vec3,
    up: Vec3,
    right: Vec3,
    length: f32,
}

impl TurtleState {
    fn new(origin: Vec3, length: f32) -> Self {
        Self {
            position: origin,
            direction: Vec3::Y,
            up: Vec3::Z,
            right: Vec3::X,
            length,
        }
    }

    fn pitch(&mut self, angle: f32) {
        let rot = Mat3::from_axis_angle(self.right, angle);
        self.direction = (rot * self.direction).normalize();
        self.up = (rot * self.up).normalize();
    }

    fn yaw(&mut self, angle: f32) {
        let rot = Mat3::from_axis_angle(self.up, angle);
        self.direction = (rot * self.direction).normalize();
        self.right = (rot * self.right).normalize();
    }

    fn roll(&mut self, angle: f32) {
        let rot = Mat3::from_axis_angle(self.direction, angle);
        self.up = (rot * self.up).normalize();
        self.right = (rot * self.right).normalize();
    }
}

/// 3D Bresenham line. Calls `f` once per voxel along `a`..=`b`.
fn rasterize_line(
    a: (i32, i32, i32),
    b: (i32, i32, i32),
    mut f: impl FnMut(i32, i32, i32),
) {
    let (mut x, mut y, mut z) = a;
    let dx = (b.0 - a.0).abs();
    let dy = (b.1 - a.1).abs();
    let dz = (b.2 - a.2).abs();
    let xs = (b.0 - a.0).signum();
    let ys = (b.1 - a.1).signum();
    let zs = (b.2 - a.2).signum();

    // Drive along the axis with the largest delta. The two error terms
    // accumulate against that driver and step the other two axes when
    // their term passes zero.
    if dx >= dy && dx >= dz {
        let mut p1 = 2 * dy - dx;
        let mut p2 = 2 * dz - dx;
        for _ in 0..=dx {
            f(x, y, z);
            if p1 >= 0 {
                y += ys;
                p1 -= 2 * dx;
            }
            if p2 >= 0 {
                z += zs;
                p2 -= 2 * dx;
            }
            p1 += 2 * dy;
            p2 += 2 * dz;
            x += xs;
        }
    } else if dy >= dx && dy >= dz {
        let mut p1 = 2 * dx - dy;
        let mut p2 = 2 * dz - dy;
        for _ in 0..=dy {
            f(x, y, z);
            if p1 >= 0 {
                x += xs;
                p1 -= 2 * dy;
            }
            if p2 >= 0 {
                z += zs;
                p2 -= 2 * dy;
            }
            p1 += 2 * dx;
            p2 += 2 * dz;
            y += ys;
        }
    } else {
        let mut p1 = 2 * dy - dz;
        let mut p2 = 2 * dx - dz;
        for _ in 0..=dz {
            f(x, y, z);
            if p1 >= 0 {
                y += ys;
                p1 -= 2 * dz;
            }
            if p2 >= 0 {
                x += xs;
                p2 -= 2 * dz;
            }
            p1 += 2 * dy;
            p2 += 2 * dx;
            z += zs;
        }
    }
}

/// Place a small spherical leaf cluster centered on `c`.
fn place_leaf_cluster(
    patch: &mut VoxelPatch,
    c: (i32, i32, i32),
    color: Voxel,
    radius: i32,
) {
    let r2 = radius * radius;
    for dz in -radius..=radius {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy + dz * dz <= r2 {
                    patch.set(c.0 + dx, c.1 + dy, c.2 + dz, color);
                }
            }
        }
    }
}

fn vec3_to_voxel_pos(v: Vec3) -> (i32, i32, i32) {
    (v.x.round() as i32, v.y.round() as i32, v.z.round() as i32)
}

impl VoxelGenerator for LSystemTree {
    fn metadata(&self) -> GeneratorMeta {
        GeneratorMeta {
            id: "builtin.lsystem_tree",
            name: "L-System Tree",
            description: "Plant grown via L-system rewrite + 3D turtle",
            category: GeneratorCategory::Vegetation,
            backend: GeneratorBackend::Algorithmic,
        }
    }

    fn generate(&self) -> GenResult<VoxelPatch> {
        if self.iterations > MAX_ITERATIONS {
            return Err(GenError::InvalidParams(format!(
                "iterations must be <= {}",
                MAX_ITERATIONS
            )));
        }
        if self.initial_length <= 0.0 {
            return Err(GenError::InvalidParams(
                "initial_length must be > 0".into(),
            ));
        }

        let s = rewrite(AXIOM, self.iterations);
        let mut patch = VoxelPatch::new();
        let mut rng = StdRng::seed_from_u64(self.seed as u64);

        let trunk = Voxel::from_rgb(
            self.trunk_color[0],
            self.trunk_color[1],
            self.trunk_color[2],
        );
        let leaves = Voxel::from_rgb(
            self.leaf_color[0],
            self.leaf_color[1],
            self.leaf_color[2],
        );

        let origin = Vec3::new(
            self.origin.0 as f32,
            self.origin.1 as f32,
            self.origin.2 as f32,
        );
        let mut turtle = TurtleState::new(origin, self.initial_length);
        let angle = self.angle_deg.to_radians();
        let mut stack: Vec<TurtleState> = Vec::new();

        for ch in s.chars() {
            match ch {
                'F' => {
                    let next = turtle.position + turtle.direction * turtle.length;
                    let a = vec3_to_voxel_pos(turtle.position);
                    let b = vec3_to_voxel_pos(next);
                    rasterize_line(a, b, |x, y, z| {
                        patch.set(x, y, z, trunk);
                    });
                    turtle.position = next;
                }
                '+' => turtle.pitch(angle),
                '-' => turtle.pitch(-angle),
                '/' => turtle.yaw(angle),
                '\\' => turtle.yaw(-angle),
                '[' => {
                    stack.push(turtle.clone());
                    // Random roll per push so sibling branches don't
                    // collapse into a single plane in 3D.
                    let roll =
                        rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI);
                    turtle.roll(roll);
                    turtle.length *= self.length_scale;
                }
                ']' => {
                    let tip = vec3_to_voxel_pos(turtle.position);
                    place_leaf_cluster(&mut patch, tip, leaves, 2);
                    if let Some(saved) = stack.pop() {
                        turtle = saved;
                    }
                }
                _ => {}
            }
        }

        Ok(patch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_generates_nonempty() {
        let g = LSystemTree::default();
        let p = g.generate().unwrap();
        assert!(!p.is_empty());
    }

    #[test]
    fn test_seed_determinism() {
        let a = LSystemTree::default();
        let b = a.clone();
        let pa = a.generate().unwrap();
        let pb = b.generate().unwrap();
        assert_eq!(pa.voxels, pb.voxels);
    }

    #[test]
    fn test_seed_changes_output() {
        let a = LSystemTree {
            seed: 1,
            ..Default::default()
        };
        let b = LSystemTree {
            seed: 2,
            ..Default::default()
        };
        // Different seed = different random rolls = different geometry.
        assert_ne!(a.generate().unwrap().voxels, b.generate().unwrap().voxels);
    }

    #[test]
    fn test_invalid_params_rejected() {
        let g = LSystemTree {
            iterations: 100,
            ..Default::default()
        };
        assert!(g.generate().is_err());

        let g = LSystemTree {
            initial_length: 0.0,
            ..Default::default()
        };
        assert!(g.generate().is_err());
    }

    #[test]
    fn test_rasterize_line_endpoints() {
        // Capture every voxel on the line and check both endpoints
        // are present (Bresenham is inclusive on both ends).
        let mut points = Vec::new();
        rasterize_line((0, 0, 0), (5, 3, 2), |x, y, z| points.push((x, y, z)));
        assert_eq!(points.first(), Some(&(0, 0, 0)));
        assert_eq!(points.last(), Some(&(5, 3, 2)));
    }

    #[test]
    fn test_rasterize_line_single_point() {
        // Degenerate line (a == b) should still emit one voxel.
        let mut points = Vec::new();
        rasterize_line((3, 4, 5), (3, 4, 5), |x, y, z| points.push((x, y, z)));
        assert_eq!(points, vec![(3, 4, 5)]);
    }
}
