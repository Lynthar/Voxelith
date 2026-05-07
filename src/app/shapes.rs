//! Built-in primitive generators (sphere, pyramid).
//!
//! These produce voxels directly into the world without going through
//! the undo history — they're invoked on a freshly-cleared world from
//! the Generate menu.

use voxelith::core::Voxel;

use super::App;

impl App {
    /// Place a colored sphere centered at `center` with the given radius.
    pub(super) fn create_sphere(&mut self, center: (i32, i32, i32), radius: i32) {
        let radius_sq = (radius as f32).powi(2);
        for z in -radius..=radius {
            for y in -radius..=radius {
                for x in -radius..=radius {
                    let dist_sq = (x * x + y * y + z * z) as f32;
                    if dist_sq <= radius_sq {
                        let t = (dist_sq.sqrt() / radius as f32 * 255.0) as u8;
                        let voxel = Voxel::from_rgb(255 - t, t, 128);
                        self.world.set_voxel(
                            center.0 + x,
                            center.1 + y,
                            center.2 + z,
                            voxel,
                        );
                    }
                }
            }
        }
    }

    /// Place a colored pyramid with its base centered at `base_center`.
    pub(super) fn create_pyramid(&mut self, base_center: (i32, i32, i32), height: i32) {
        for y in 0..height {
            let size = height - y;
            for z in -size..=size {
                for x in -size..=size {
                    let t = (y as f32 / height as f32 * 255.0) as u8;
                    let voxel =
                        Voxel::from_rgb(194 - t / 2, 178 - t / 2, 128 + t / 2);
                    self.world.set_voxel(
                        base_center.0 + x,
                        base_center.1 + y,
                        base_center.2 + z,
                        voxel,
                    );
                }
            }
        }
    }
}
