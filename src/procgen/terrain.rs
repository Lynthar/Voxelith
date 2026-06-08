//! Heightmap terrain via fractal Perlin noise (FBM).

use std::time::Duration;

use noise::{NoiseFn, Perlin};

use crate::core::Voxel;

use super::{
    GenError, GenResult, GeneratorBackend, GeneratorCategory, GeneratorMeta,
    VoxelGenerator, VoxelPatch,
};

/// Heightmap terrain generator using fractal Brownian motion (FBM)
/// over Perlin noise.
///
/// Produces a `width × depth` patch centered on the world origin. The
/// height field at each `(x, z)` is the sum of `octaves` Perlin samples
/// with doubling frequency and halving amplitude per octave. Output is
/// stratified — grass on top, a dirt band, then stone below — so the
/// shape is visible immediately without lighting tweaks.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PerlinTerrain {
    /// Seed for the underlying Perlin permutation table.
    pub seed: u32,
    /// X-extent in voxels.
    pub width: u32,
    /// Z-extent in voxels.
    pub depth: u32,
    /// Lowest y to fill (inclusive). Heights below this become air.
    pub min_height: i32,
    /// Highest y a column can reach.
    pub max_height: i32,
    /// Base spatial frequency (smaller = larger features).
    pub frequency: f64,
    /// Number of FBM octaves (>= 1).
    pub octaves: u32,
}

impl Default for PerlinTerrain {
    fn default() -> Self {
        // Defaults tuned for "looks like a coherent landscape" on a
        // 64x64 footprint: shorter height range and lower frequency
        // produce gentler, more continuous slopes; fewer octaves cuts
        // the high-frequency choppiness that made every column read
        // as an isolated grass spike. Users who want jagged terrain
        // can dial these up in the panel.
        Self {
            seed: 42,
            width: 64,
            depth: 64,
            min_height: 0,
            max_height: 12,
            frequency: 0.03,
            octaves: 3,
        }
    }
}

impl VoxelGenerator for PerlinTerrain {
    fn metadata(&self) -> GeneratorMeta {
        GeneratorMeta {
            id: "builtin.perlin_terrain",
            name: "Perlin Terrain",
            description: "Heightmap terrain via fractal Perlin noise",
            category: GeneratorCategory::Terrain,
            backend: GeneratorBackend::Algorithmic,
        }
    }

    fn generate(&self) -> GenResult<VoxelPatch> {
        if self.width == 0 || self.depth == 0 {
            return Err(GenError::InvalidParams(
                "width and depth must be > 0".into(),
            ));
        }
        if self.max_height < self.min_height {
            return Err(GenError::InvalidParams(
                "max_height must be >= min_height".into(),
            ));
        }
        let octaves = self.octaves.max(1);

        let perlin = Perlin::new(self.seed);

        let half_w = (self.width / 2) as i32;
        let half_d = (self.depth / 2) as i32;
        let height_range = (self.max_height - self.min_height) as f64;

        // Rough capacity hint: half the bounding volume tends to be solid.
        let est = (self.width as usize)
            * (self.depth as usize)
            * (height_range as usize).max(1)
            / 2;
        let mut patch = VoxelPatch::with_capacity(est);

        // Color stratification (grass / dirt / stone).
        let grass = Voxel::from_rgb(76, 153, 0);
        let dirt = Voxel::from_rgb(139, 90, 43);
        let stone = Voxel::from_rgb(128, 128, 128);
        let dirt_band: i32 = 4;

        // Span exactly width × depth columns, centered on the origin.
        // Upper bound is `dim - half` (not `half`): for odd sizes the
        // half-open `-half..half` range would drop the final column/row
        // (e.g. width 255 → only 254 columns). Even sizes are unchanged
        // (dim - half == half).
        let x_end = self.width as i32 - half_w;
        let z_end = self.depth as i32 - half_d;
        for z in -half_d..z_end {
            for x in -half_w..x_end {
                // FBM: sum octaves with halving amplitude / doubling frequency.
                // Track total amplitude to normalize the output back to ~[-1, 1].
                let mut acc = 0.0_f64;
                let mut total_amp = 0.0_f64;
                let mut amp = 1.0_f64;
                let mut freq = self.frequency;

                for _ in 0..octaves {
                    let n = perlin.get([x as f64 * freq, z as f64 * freq]);
                    acc += n * amp;
                    total_amp += amp;
                    amp *= 0.5;
                    freq *= 2.0;
                }

                let h_unit = ((acc / total_amp) + 1.0) * 0.5; // [0, 1]
                let h = self.min_height + (h_unit * height_range).round() as i32;

                for y in self.min_height..=h {
                    let voxel = if y == h {
                        grass
                    } else if y > h - dirt_band {
                        dirt
                    } else {
                        stone
                    };
                    patch.set(x, y, z, voxel);
                }
            }
        }

        Ok(patch)
    }

    fn estimate_duration(&self) -> Duration {
        // Loose linear estimate; on a 64x64x4-octave it runs in < 5ms.
        let n = (self.width as u64) * (self.depth as u64) * (self.octaves.max(1) as u64);
        Duration::from_micros(n / 4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_generates_nonempty() {
        let gen = PerlinTerrain::default();
        let patch = gen.generate().unwrap();
        assert!(!patch.is_empty());
    }

    #[test]
    fn test_seed_determinism() {
        // Same seed + params -> identical output.
        let a = PerlinTerrain {
            width: 16,
            depth: 16,
            ..Default::default()
        };
        let b = a.clone();
        let pa = a.generate().unwrap();
        let pb = b.generate().unwrap();
        assert_eq!(pa.voxels.len(), pb.voxels.len());
        assert_eq!(pa.voxels, pb.voxels);
    }

    #[test]
    fn test_seed_changes_output() {
        let a = PerlinTerrain {
            seed: 1,
            width: 16,
            depth: 16,
            ..Default::default()
        };
        let b = PerlinTerrain {
            seed: 2,
            ..a.clone()
        };
        let pa = a.generate().unwrap();
        let pb = b.generate().unwrap();
        // Different seeds should not yield byte-identical patches.
        assert_ne!(pa.voxels, pb.voxels);
    }

    #[test]
    fn test_invalid_params_rejected() {
        let g = PerlinTerrain {
            width: 0,
            ..Default::default()
        };
        assert!(g.generate().is_err());

        let g = PerlinTerrain {
            min_height: 10,
            max_height: 0,
            ..Default::default()
        };
        assert!(g.generate().is_err());
    }

    #[test]
    fn test_odd_dimensions_generate_full_extent() {
        // Regression: `-half..half` dropped the last column/row for odd
        // width/depth (e.g. 255 -> 254). The footprint must span the
        // full requested width × depth. Every column gets at least one
        // voxel (min_height..=h always includes min_height), so the count
        // of distinct x / z equals width / depth.
        let g = PerlinTerrain {
            seed: 3,
            width: 9,
            depth: 7,
            min_height: 0,
            max_height: 8,
            ..Default::default()
        };
        let patch = g.generate().unwrap();
        let xs: std::collections::HashSet<i32> =
            patch.voxels.iter().map(|((x, _, _), _)| *x).collect();
        let zs: std::collections::HashSet<i32> =
            patch.voxels.iter().map(|((_, _, z), _)| *z).collect();
        assert_eq!(xs.len(), 9, "odd width should yield 9 columns, got {}", xs.len());
        assert_eq!(zs.len(), 7, "odd depth should yield 7 rows, got {}", zs.len());
        // Centered span: -half ..= dim-half-1.
        assert_eq!((*xs.iter().min().unwrap(), *xs.iter().max().unwrap()), (-4, 4));
        assert_eq!((*zs.iter().min().unwrap(), *zs.iter().max().unwrap()), (-3, 3));
    }

    #[test]
    fn test_height_bounds_respected() {
        let g = PerlinTerrain {
            seed: 7,
            width: 8,
            depth: 8,
            min_height: -5,
            max_height: 5,
            ..Default::default()
        };
        let patch = g.generate().unwrap();
        for ((_, y, _), _) in &patch.voxels {
            assert!(*y >= -5 && *y <= 5, "y={} out of [{}, {}]", y, -5, 5);
        }
    }
}
