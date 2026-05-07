//! Wave Function Collapse generator for tile-based level layouts.
//!
//! 2D collapse on a tile grid (X-Z plane). Each grid cell occupies a
//! `TILE_SIZE³` voxel block. Tiles connect at face boundaries via
//! integer connector IDs — adjacency is allowed iff some allowed tile
//! on this side has the same connector ID as some allowed tile on the
//! other side. Y is unconstrained: every tile fills the same Y range,
//! so we don't collapse vertically.
//!
//! No backtracking. If propagation reduces a cell's domain to empty
//! we substitute the fallback (`empty`) tile during output. Quality
//! degrades gracefully for over-constrained tilesets but the generator
//! always terminates — important so a UI preview can run it on every
//! parameter change without risking a hang.

use std::time::Duration;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::core::Voxel;

use super::{
    GenError, GenResult, GeneratorBackend, GeneratorCategory, GeneratorMeta,
    VoxelGenerator, VoxelPatch,
};

/// Cubic side length of each tile, in voxels.
pub const WFC_TILE_SIZE: usize = 4;
const TILE_VOLUME: usize = WFC_TILE_SIZE * WFC_TILE_SIZE * WFC_TILE_SIZE;
// Alias used inside this module so the longer public name doesn't
// clutter the implementation.
const TILE_SIZE: usize = WFC_TILE_SIZE;

/// One tile in the tileset.
#[derive(Debug, Clone)]
pub struct Tile {
    pub name: &'static str,
    /// Connector IDs in face order `[+X, -X, +Z, -Z]`. Two horizontally
    /// adjacent tiles match when one's outgoing-face connector equals
    /// the other's incoming-face connector.
    pub connectors: [u8; 4],
    /// Solid voxel mask, layout `x + y*S + z*S*S`.
    pub solid: [bool; TILE_VOLUME],
    /// Selection weight. Higher → appears more often. Floor is
    /// boosted so output isn't dominated by walls.
    pub weight: f32,
}

#[derive(Debug, Clone)]
pub struct Tileset {
    pub name: &'static str,
    pub tiles: Vec<Tile>,
}

/// Tilesets the WFC generator can dispatch to. Right now there's just
/// one, but the enum keeps the door open for a UI dropdown later
/// (e.g. dungeon vs. city vs. plumbing).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum WfcTileset {
    Dungeon,
}

impl Default for WfcTileset {
    fn default() -> Self {
        Self::Dungeon
    }
}

impl WfcTileset {
    pub fn label(self) -> &'static str {
        match self {
            Self::Dungeon => "Dungeon",
        }
    }

    pub fn build(self) -> Tileset {
        match self {
            Self::Dungeon => dungeon_tileset(),
        }
    }
}

/// 13-tile dungeon tileset: empty / floor / 2 straight walls / 4 corners /
/// 4 T-junctions / 1 cross. Connector IDs: 0 = "open" (no wall on that
/// face), 1 = "wall". Floor is weighted heaviest so output is mostly
/// open ground; T-junctions and the cross are progressively rarer to
/// keep dense intersections from dominating.
fn dungeon_tileset() -> Tileset {
    let mut tiles = Vec::with_capacity(13);

    tiles.push(Tile {
        name: "empty",
        connectors: [0, 0, 0, 0],
        solid: [false; TILE_VOLUME],
        weight: 1.5,
    });

    let mut floor = [false; TILE_VOLUME];
    for x in 0..TILE_SIZE {
        for z in 0..TILE_SIZE {
            floor[idx(x, 0, z)] = true;
        }
    }
    tiles.push(Tile {
        name: "floor",
        connectors: [0, 0, 0, 0],
        solid: floor,
        weight: 4.0,
    });

    // Straight walls.
    tiles.push(Tile {
        name: "wall_x",
        connectors: [1, 1, 0, 0],
        solid: wall_pattern(true, true, false, false),
        weight: 2.0,
    });
    tiles.push(Tile {
        name: "wall_z",
        connectors: [0, 0, 1, 1],
        solid: wall_pattern(false, false, true, true),
        weight: 2.0,
    });

    // L-shaped corners.
    tiles.push(Tile {
        name: "corner_pxpz",
        connectors: [1, 0, 1, 0],
        solid: wall_pattern(true, false, true, false),
        weight: 1.0,
    });
    tiles.push(Tile {
        name: "corner_nxpz",
        connectors: [0, 1, 1, 0],
        solid: wall_pattern(false, true, true, false),
        weight: 1.0,
    });
    tiles.push(Tile {
        name: "corner_pxnz",
        connectors: [1, 0, 0, 1],
        solid: wall_pattern(true, false, false, true),
        weight: 1.0,
    });
    tiles.push(Tile {
        name: "corner_nxnz",
        connectors: [0, 1, 0, 1],
        solid: wall_pattern(false, true, false, true),
        weight: 1.0,
    });

    // T-junctions. Each name encodes which face is the "open" arm
    // (the side that doesn't have a wall extension).
    tiles.push(Tile {
        name: "t_open_px",
        connectors: [0, 1, 1, 1],
        solid: wall_pattern(false, true, true, true),
        weight: 1.0,
    });
    tiles.push(Tile {
        name: "t_open_nx",
        connectors: [1, 0, 1, 1],
        solid: wall_pattern(true, false, true, true),
        weight: 1.0,
    });
    tiles.push(Tile {
        name: "t_open_pz",
        connectors: [1, 1, 0, 1],
        solid: wall_pattern(true, true, false, true),
        weight: 1.0,
    });
    tiles.push(Tile {
        name: "t_open_nz",
        connectors: [1, 1, 1, 0],
        solid: wall_pattern(true, true, true, false),
        weight: 1.0,
    });

    // 4-way cross. Rare so layouts don't end up with a forest of
    // intersections.
    tiles.push(Tile {
        name: "cross",
        connectors: [1, 1, 1, 1],
        solid: wall_pattern(true, true, true, true),
        weight: 0.5,
    });

    Tileset {
        name: "dungeon",
        tiles,
    }
}

#[inline]
fn idx(x: usize, y: usize, z: usize) -> usize {
    x + y * TILE_SIZE + z * TILE_SIZE * TILE_SIZE
}

/// Wall pattern with optional 1-tile-thick extensions reaching to the
/// `+X / -X / +Z / -Z` face. A central pillar (x∈{1,2}, z∈{1,2}) is
/// always present so all extensions meet cleanly. Used for the 2 straight
/// walls and the 4 corners.
fn wall_pattern(px: bool, nx: bool, pz: bool, nz: bool) -> [bool; TILE_VOLUME] {
    let mut p = [false; TILE_VOLUME];

    // Central pillar.
    for y in 0..TILE_SIZE {
        for z in 1..3 {
            for x in 1..3 {
                p[idx(x, y, z)] = true;
            }
        }
    }
    if px {
        for y in 0..TILE_SIZE {
            for z in 1..3 {
                p[idx(3, y, z)] = true;
            }
        }
    }
    if nx {
        for y in 0..TILE_SIZE {
            for z in 1..3 {
                p[idx(0, y, z)] = true;
            }
        }
    }
    if pz {
        for y in 0..TILE_SIZE {
            for x in 1..3 {
                p[idx(x, y, 3)] = true;
            }
        }
    }
    if nz {
        for y in 0..TILE_SIZE {
            for x in 1..3 {
                p[idx(x, y, 0)] = true;
            }
        }
    }
    p
}

/// One grid cell during the collapse.
#[derive(Clone)]
struct Cell {
    /// Bitset over tile indices. Up to 64 tiles (one bit each).
    allowed: u64,
    /// True once the cell has been observed (its domain reduced to a
    /// single tile). Empty domains can also flip this true to mark
    /// "done, nothing left to try".
    collapsed: bool,
}

impl Cell {
    fn count(&self) -> u32 {
        self.allowed.count_ones()
    }

    fn iter_allowed(&self) -> impl Iterator<Item = usize> + '_ {
        (0..64).filter(move |i| self.allowed & (1u64 << i) != 0)
    }
}

/// WFC parameters and the entry point implementing `VoxelGenerator`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WfcGenerator {
    pub seed: u32,
    /// Grid width in tiles. Total voxel width = `width * TILE_SIZE`.
    pub width: u32,
    /// Grid depth in tiles.
    pub depth: u32,
    /// World-space origin of the (0, 0) grid cell's min-corner.
    pub origin: (i32, i32, i32),
    pub tileset: WfcTileset,
}

impl Default for WfcGenerator {
    fn default() -> Self {
        Self {
            seed: 1,
            width: 8,
            depth: 8,
            origin: (-16, 0, -16),
            tileset: WfcTileset::Dungeon,
        }
    }
}

impl VoxelGenerator for WfcGenerator {
    fn metadata(&self) -> GeneratorMeta {
        GeneratorMeta {
            id: "builtin.wfc",
            name: "WFC Tile Layout",
            description: "Wave Function Collapse on a tile grid",
            category: GeneratorCategory::Building,
            backend: GeneratorBackend::Algorithmic,
        }
    }

    fn generate(&self) -> GenResult<VoxelPatch> {
        if self.width == 0 || self.depth == 0 {
            return Err(GenError::InvalidParams(
                "width and depth must be > 0".into(),
            ));
        }

        let tileset = self.tileset.build();
        let n_tiles = tileset.tiles.len();
        if n_tiles == 0 || n_tiles > 64 {
            return Err(GenError::InvalidParams(
                "tileset must have 1..=64 tiles".into(),
            ));
        }

        let w = self.width as usize;
        let d = self.depth as usize;
        let n_cells = w * d;
        let all_allowed: u64 =
            if n_tiles == 64 { !0 } else { (1u64 << n_tiles) - 1 };

        let mut cells: Vec<Cell> =
            vec![Cell { allowed: all_allowed, collapsed: false }; n_cells];
        let mut rng = StdRng::seed_from_u64(self.seed as u64);

        let weights: Vec<f32> =
            tileset.tiles.iter().map(|t| t.weight).collect();

        // Main collapse loop. Pick the lowest-entropy cell, observe
        // it, propagate. Bail when nothing's left to collapse.
        loop {
            let Some(target) = lowest_entropy(&cells, &mut rng) else {
                break;
            };
            collapse(&mut cells[target], &weights, &mut rng);
            propagate(&mut cells, w, d, target, &tileset.tiles);
        }

        let mut patch = VoxelPatch::new();
        let stone = Voxel::from_rgb(140, 140, 140);
        let mut failed_cells: u32 = 0;

        for cz in 0..d {
            for cx in 0..w {
                let cell_i = cz * w + cx;
                // Either the unique chosen tile or fallback to `empty`
                // (index 0) when the domain ended up empty. Empty
                // domains are an over-constrained outcome of the
                // forward-only solver — we count them so the UI can
                // surface a warning.
                let tile_i = if cells[cell_i].count() == 1 {
                    cells[cell_i].iter_allowed().next().unwrap()
                } else {
                    failed_cells += 1;
                    0
                };
                let tile = &tileset.tiles[tile_i];

                let ox = self.origin.0 + (cx as i32) * TILE_SIZE as i32;
                let oy = self.origin.1;
                let oz = self.origin.2 + (cz as i32) * TILE_SIZE as i32;

                for vy in 0..TILE_SIZE {
                    for vz in 0..TILE_SIZE {
                        for vx in 0..TILE_SIZE {
                            if tile.solid[idx(vx, vy, vz)] {
                                patch.set(
                                    ox + vx as i32,
                                    oy + vy as i32,
                                    oz + vz as i32,
                                    stone,
                                );
                            }
                        }
                    }
                }
            }
        }

        if failed_cells > 0 {
            patch.notes.push(format!(
                "WFC: {} cell(s) over-constrained, filled with empty",
                failed_cells
            ));
        }

        Ok(patch)
    }

    fn estimate_duration(&self) -> Duration {
        // Loose linear estimate; in practice an 8x8 dungeon runs in <1ms.
        let n = (self.width as u64) * (self.depth as u64);
        Duration::from_micros(n * 200)
    }
}

/// Pick the uncollapsed cell with the smallest non-empty domain. Ties
/// broken randomly so the output isn't biased toward a corner.
fn lowest_entropy(cells: &[Cell], rng: &mut StdRng) -> Option<usize> {
    let mut best_count = u32::MAX;
    let mut best: Vec<usize> = Vec::new();
    for (i, cell) in cells.iter().enumerate() {
        if cell.collapsed {
            continue;
        }
        let count = cell.count();
        if count == 0 {
            continue;
        }
        if count < best_count {
            best_count = count;
            best.clear();
            best.push(i);
        } else if count == best_count {
            best.push(i);
        }
    }
    if best.is_empty() {
        None
    } else {
        let pick = rng.gen_range(0..best.len());
        Some(best[pick])
    }
}

/// Sample one tile from the cell's domain, weighted by the tileset's
/// per-tile weights. Reduces the cell to that single tile and marks
/// it collapsed.
fn collapse(cell: &mut Cell, weights: &[f32], rng: &mut StdRng) {
    let allowed: Vec<usize> = cell.iter_allowed().collect();
    if allowed.is_empty() {
        cell.collapsed = true;
        return;
    }
    let total: f32 = allowed.iter().map(|&i| weights[i]).sum();
    let mut pick = rng.gen::<f32>() * total;
    let mut chosen = *allowed.last().unwrap();
    for &i in &allowed {
        pick -= weights[i];
        if pick <= 0.0 {
            chosen = i;
            break;
        }
    }
    cell.allowed = 1u64 << chosen;
    cell.collapsed = true;
}

/// Constraint propagation. After a cell shrinks, its neighbors' domains
/// may be reducible too (only tiles whose facing connector is matched
/// by some tile still allowed in the source cell can survive). Whenever
/// a domain shrinks we re-queue that cell, since *its* neighbors may
/// now be over-constrained in turn.
fn propagate(
    cells: &mut [Cell],
    w: usize,
    d: usize,
    start: usize,
    tiles: &[Tile],
) {
    let mut stack = vec![start];
    while let Some(idx) = stack.pop() {
        let allowed = cells[idx].allowed;
        let cx = idx % w;
        let cz = idx / w;

        // Face order: 0=+X, 1=-X, 2=+Z, 3=-Z.
        // Each entry: (dx, dz, my_face, neighbor_face).
        let dirs: [(i32, i32, usize, usize); 4] = [
            (1, 0, 0, 1),
            (-1, 0, 1, 0),
            (0, 1, 2, 3),
            (0, -1, 3, 2),
        ];
        for (dx, dz, my_face, neighbor_face) in dirs {
            let nx = cx as i32 + dx;
            let nz = cz as i32 + dz;
            if nx < 0 || nx >= w as i32 || nz < 0 || nz >= d as i32 {
                continue;
            }
            let nidx = nz as usize * w + nx as usize;

            // Connectors my cell currently exposes on `my_face`. The
            // bitset is over connector IDs (assumed to fit in u32).
            let mut my_conns: u32 = 0;
            for i in 0..tiles.len() {
                if allowed & (1u64 << i) != 0 {
                    my_conns |= 1u32 << tiles[i].connectors[my_face];
                }
            }

            // Filter the neighbor's domain to tiles whose facing
            // connector matches at least one of mine.
            let mut new_allowed: u64 = 0;
            for j in 0..tiles.len() {
                if cells[nidx].allowed & (1u64 << j) == 0 {
                    continue;
                }
                let c = tiles[j].connectors[neighbor_face];
                if my_conns & (1u32 << c) != 0 {
                    new_allowed |= 1u64 << j;
                }
            }

            if new_allowed != cells[nidx].allowed {
                cells[nidx].allowed = new_allowed;
                if !cells[nidx].collapsed {
                    stack.push(nidx);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dungeon_tileset_loads() {
        let ts = WfcTileset::Dungeon.build();
        assert_eq!(ts.tiles.len(), 13);
        assert_eq!(ts.tiles[0].name, "empty");
        assert_eq!(ts.tiles[1].name, "floor");
        assert_eq!(ts.tiles.last().unwrap().name, "cross");
    }

    #[test]
    fn test_all_tiles_distinct_connectors_or_geometry() {
        // Sanity: no two tiles should be identical (same connectors AND same
        // voxel pattern). Helps catch copy-paste mistakes when extending.
        let ts = WfcTileset::Dungeon.build();
        for i in 0..ts.tiles.len() {
            for j in (i + 1)..ts.tiles.len() {
                let a = &ts.tiles[i];
                let b = &ts.tiles[j];
                assert!(
                    a.connectors != b.connectors || a.solid != b.solid,
                    "tiles {} and {} are identical",
                    a.name, b.name
                );
            }
        }
    }

    #[test]
    fn test_default_generates_nonempty() {
        let g = WfcGenerator::default();
        let p = g.generate().unwrap();
        // Floor tiles are weighted heavily; non-empty is overwhelmingly likely.
        assert!(!p.is_empty());
    }

    #[test]
    fn test_seed_determinism() {
        let a = WfcGenerator::default();
        let b = a.clone();
        let pa = a.generate().unwrap();
        let pb = b.generate().unwrap();
        assert_eq!(pa.voxels, pb.voxels);
    }

    #[test]
    fn test_seed_changes_output() {
        let a = WfcGenerator { seed: 1, ..Default::default() };
        let b = WfcGenerator { seed: 99, ..Default::default() };
        // Different seeds should pick different tile arrangements.
        assert_ne!(
            a.generate().unwrap().voxels,
            b.generate().unwrap().voxels
        );
    }

    #[test]
    fn test_invalid_params_rejected() {
        let g = WfcGenerator { width: 0, ..Default::default() };
        assert!(g.generate().is_err());
        let g = WfcGenerator { depth: 0, ..Default::default() };
        assert!(g.generate().is_err());
    }

    #[test]
    fn test_output_within_bounds() {
        let g = WfcGenerator {
            width: 4,
            depth: 4,
            origin: (0, 0, 0),
            ..Default::default()
        };
        let p = g.generate().unwrap();
        let extent = (g.width as i32) * TILE_SIZE as i32;
        for ((x, y, z), _) in &p.voxels {
            assert!(*x >= 0 && *x < extent);
            assert!(*y >= 0 && *y < TILE_SIZE as i32);
            assert!(*z >= 0 && *z < extent);
        }
    }
}
