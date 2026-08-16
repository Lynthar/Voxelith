//! Wave Function Collapse for tile layouts: 2D collapse on an X-Z grid
//! of `TILE_SIZE³` blocks, tiles fitting by connector complement.
//! Non-backtracking — an over-constrained cell falls back to `empty`.

use std::time::Duration;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::core::Voxel;

use super::{GenError, GenResult, GeneratorCategory, GeneratorMeta, VoxelGenerator, VoxelPatch};

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
    /// Voxel data for each cell, laid out `x + y*S + z*S*S`; `AIR` means
    /// empty. Per-cell colors let one tile hold several materials.
    pub cells: [Voxel; TILE_VOLUME],
    /// Selection weight. Higher → appears more often.
    pub weight: f32,
}

#[derive(Debug, Clone)]
pub struct Tileset {
    pub name: &'static str,
    pub tiles: Vec<Tile>,
}

/// Tilesets the generator can dispatch to. A new theme goes here plus
/// [`Self::build`] and [`Self::label`]; dropdowns read [`Self::ALL`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum WfcTileset {
    /// Stone walls, floors, T-junctions, doorways. Single ground
    /// layer with walls rising the full tile height.
    #[default]
    Dungeon,
    /// Grass plots, asphalt roads with sidewalks, intersections,
    /// and small buildings rising above grass. Connector IDs
    /// `0 = grass-side`, `1 = road-side`.
    City,
}

impl WfcTileset {
    /// All tilesets, in dropdown order.
    pub const ALL: &'static [Self] = &[Self::Dungeon, Self::City];

    pub fn label(self) -> &'static str {
        match self {
            Self::Dungeon => "Dungeon",
            Self::City => "City",
        }
    }

    pub fn build(self) -> Tileset {
        match self {
            Self::Dungeon => dungeon_tileset(),
            Self::City => city_tileset(),
        }
    }
}

/// 19-tile dungeon tileset: empty, floor, two walls, four corners, four
/// T-junctions, a cross, two walls with doors and four door-mouth
/// floors. Connectors: 0 open, 1 wall, 2 door mouth, 3 door socket.
fn dungeon_tileset() -> Tileset {
    let mut tiles = Vec::with_capacity(19);
    let stone = Voxel::from_rgb(140, 140, 140);

    tiles.push(Tile {
        name: "empty",
        connectors: [0, 0, 0, 0],
        cells: [Voxel::AIR; TILE_VOLUME],
        weight: 1.5,
    });

    let mut floor = [Voxel::AIR; TILE_VOLUME];
    for x in 0..TILE_SIZE {
        for z in 0..TILE_SIZE {
            floor[idx(x, 0, z)] = stone;
        }
    }
    tiles.push(Tile {
        name: "floor",
        connectors: [0, 0, 0, 0],
        cells: floor,
        weight: 4.0,
    });

    // Straight walls.
    tiles.push(Tile {
        name: "wall_x",
        connectors: [1, 1, 0, 0],
        cells: wall_pattern(true, true, false, false, stone),
        weight: 2.0,
    });
    tiles.push(Tile {
        name: "wall_z",
        connectors: [0, 0, 1, 1],
        cells: wall_pattern(false, false, true, true, stone),
        weight: 2.0,
    });

    // L-shaped corners.
    tiles.push(Tile {
        name: "corner_pxpz",
        connectors: [1, 0, 1, 0],
        cells: wall_pattern(true, false, true, false, stone),
        weight: 1.0,
    });
    tiles.push(Tile {
        name: "corner_nxpz",
        connectors: [0, 1, 1, 0],
        cells: wall_pattern(false, true, true, false, stone),
        weight: 1.0,
    });
    tiles.push(Tile {
        name: "corner_pxnz",
        connectors: [1, 0, 0, 1],
        cells: wall_pattern(true, false, false, true, stone),
        weight: 1.0,
    });
    tiles.push(Tile {
        name: "corner_nxnz",
        connectors: [0, 1, 0, 1],
        cells: wall_pattern(false, true, false, true, stone),
        weight: 1.0,
    });

    // T-junctions. Each name encodes which face is the "open" arm
    // (the side that doesn't have a wall extension).
    tiles.push(Tile {
        name: "t_open_px",
        connectors: [0, 1, 1, 1],
        cells: wall_pattern(false, true, true, true, stone),
        weight: 1.0,
    });
    tiles.push(Tile {
        name: "t_open_nx",
        connectors: [1, 0, 1, 1],
        cells: wall_pattern(true, false, true, true, stone),
        weight: 1.0,
    });
    tiles.push(Tile {
        name: "t_open_pz",
        connectors: [1, 1, 0, 1],
        cells: wall_pattern(true, true, false, true, stone),
        weight: 1.0,
    });
    tiles.push(Tile {
        name: "t_open_nz",
        connectors: [1, 1, 1, 0],
        cells: wall_pattern(true, true, true, false, stone),
        weight: 1.0,
    });

    // 4-way cross. Rare so layouts don't end up with a forest of
    // intersections.
    tiles.push(Tile {
        name: "cross",
        connectors: [1, 1, 1, 1],
        cells: wall_pattern(true, true, true, true, stone),
        weight: 0.5,
    });

    // Doorway tiles: `wall_x` / `wall_z` geometry with a 2×2 portal
    // carved through. The mouth faces carry connector 2, which forces
    // the cells they open into to be `floor_door_*` variants.
    tiles.push(Tile {
        name: "wall_x_with_door",
        connectors: [1, 1, 2, 2],
        cells: wall_with_door_pattern_x(stone),
        weight: 0.5,
    });
    tiles.push(Tile {
        name: "wall_z_with_door",
        connectors: [2, 2, 1, 1],
        cells: wall_with_door_pattern_z(stone),
        weight: 0.5,
    });

    // Floor with a door socket: plain floor geometry exposing connector
    // 3 on one face, the complement of a wall's mouth. Low weight, so
    // these appear only where a door forces them.
    for (name, connectors) in [
        ("floor_door_px", [3u8, 0, 0, 0]),
        ("floor_door_nx", [0, 3, 0, 0]),
        ("floor_door_pz", [0, 0, 3, 0]),
        ("floor_door_nz", [0, 0, 0, 3]),
    ] {
        tiles.push(Tile {
            name,
            connectors,
            cells: floor, // [Voxel; TILE_VOLUME] is Copy
            weight: 0.4,
        });
    }

    Tileset {
        name: "dungeon",
        tiles,
    }
}

/// `wall_x` geometry with a 2×2 portal carved out of the central
/// pillar. The wall above and the jambs at x∈{0,3} stay solid, so the
/// opening reads as a door in a continuous wall.
fn wall_with_door_pattern_x(color: Voxel) -> [Voxel; TILE_VOLUME] {
    let mut p = wall_pattern(true, true, false, false, color);
    for y in 0..2 {
        for z in 1..3 {
            for x in 1..3 {
                p[idx(x, y, z)] = Voxel::AIR;
            }
        }
    }
    p
}

/// Mirror of `wall_with_door_pattern_x` for the Z-running wall. The
/// carve region is identical — the difference is only in which
/// directions the wall extends out to the tile faces.
fn wall_with_door_pattern_z(color: Voxel) -> [Voxel; TILE_VOLUME] {
    let mut p = wall_pattern(false, false, true, true, color);
    for y in 0..2 {
        for z in 1..3 {
            for x in 1..3 {
                p[idx(x, y, z)] = Voxel::AIR;
            }
        }
    }
    p
}

/// 13-tile city tileset: grass, two straight roads, four corners, four
/// T-junctions, a crossing and a building. Connector 0 is a grass side
/// and 1 a road side, so roads network into grids on their own.
fn city_tileset() -> Tileset {
    let grass = Voxel::from_rgb(76, 153, 0);
    let asphalt = Voxel::from_rgb(50, 50, 50);
    let sidewalk = Voxel::from_rgb(180, 180, 180);
    let building = Voxel::from_rgb(140, 75, 50);

    let mut tiles = Vec::with_capacity(13);

    // Pure grass: y=0 layer all green, no upper structure.
    let mut grass_only = [Voxel::AIR; TILE_VOLUME];
    for x in 0..TILE_SIZE {
        for z in 0..TILE_SIZE {
            grass_only[idx(x, 0, z)] = grass;
        }
    }
    tiles.push(Tile {
        name: "grass",
        connectors: [0, 0, 0, 0],
        cells: grass_only,
        weight: 6.0,
    });

    // Roads: straight, four corners, four T and one cross, each built by
    // `road_y0_pattern` from the flags marking which faces the asphalt
    // exits. (name, connectors, exits, weight).
    type RoadSpec = (&'static str, [u8; 4], (bool, bool, bool, bool), f32);
    #[rustfmt::skip] // hand-aligned columns: the flag grid IS the road shape
    let road_specs: &[RoadSpec] = &[
        ("road_x",            [1, 1, 0, 0], (true,  true,  false, false), 1.5),
        ("road_z",            [0, 0, 1, 1], (false, false, true,  true ), 1.5),
        ("road_corner_pxpz",  [1, 0, 1, 0], (true,  false, true,  false), 0.4),
        ("road_corner_nxpz",  [0, 1, 1, 0], (false, true,  true,  false), 0.4),
        ("road_corner_pxnz",  [1, 0, 0, 1], (true,  false, false, true ), 0.4),
        ("road_corner_nxnz",  [0, 1, 0, 1], (false, true,  false, true ), 0.4),
        ("road_t_open_px",    [0, 1, 1, 1], (false, true,  true,  true ), 0.3),
        ("road_t_open_nx",    [1, 0, 1, 1], (true,  false, true,  true ), 0.3),
        ("road_t_open_pz",    [1, 1, 0, 1], (true,  true,  false, true ), 0.3),
        ("road_t_open_nz",    [1, 1, 1, 0], (true,  true,  true,  false), 0.3),
        ("road_cross",        [1, 1, 1, 1], (true,  true,  true,  true ), 0.2),
    ];
    for &(name, conn, (px, nx, pz, nz), weight) in road_specs {
        tiles.push(Tile {
            name,
            connectors: conn,
            cells: road_y0_pattern(px, nx, pz, nz, grass, asphalt, sidewalk),
            weight,
        });
    }

    // Building: grass at y=0 under a solid 2×2 brick cube. Its weight
    // against grass keeps layouts a sparse scatter rather than a dense
    // urban core.
    let mut building_cells = grass_only;
    for y in 1..TILE_SIZE {
        for z in 1..3 {
            for x in 1..3 {
                building_cells[idx(x, y, z)] = building;
            }
        }
    }
    tiles.push(Tile {
        name: "building",
        connectors: [0, 0, 0, 0],
        cells: building_cells,
        weight: 2.0,
    });

    Tileset {
        name: "city",
        tiles,
    }
}

/// Build the y=0 layer of a road tile. Asphalt fills a 2×2 pad plus a
/// strip to each enabled face, the remaining perimeter becomes
/// sidewalk, and the rest stays grass. No flags gives pure grass.
fn road_y0_pattern(
    px: bool,
    nx: bool,
    pz: bool,
    nz: bool,
    grass: Voxel,
    asphalt: Voxel,
    sidewalk: Voxel,
) -> [Voxel; TILE_VOLUME] {
    let mut p = [Voxel::AIR; TILE_VOLUME];

    // Default y=0 fill: grass everywhere.
    for x in 0..TILE_SIZE {
        for z in 0..TILE_SIZE {
            p[idx(x, 0, z)] = grass;
        }
    }

    // No road exits → return pure grass (the function still gets
    // called for the building's base layer with all flags false).
    if !(px || nx || pz || nz) {
        return p;
    }

    // Asphalt: central 2×2 pad + 2-wide strips reaching each
    // enabled face.
    for x in 1..3 {
        for z in 1..3 {
            p[idx(x, 0, z)] = asphalt;
        }
    }
    if px {
        for z in 1..3 {
            p[idx(3, 0, z)] = asphalt;
        }
    }
    if nx {
        for z in 1..3 {
            p[idx(0, 0, z)] = asphalt;
        }
    }
    if pz {
        for x in 1..3 {
            p[idx(x, 0, 3)] = asphalt;
        }
    }
    if nz {
        for x in 1..3 {
            p[idx(x, 0, 0)] = asphalt;
        }
    }

    // Sidewalk: whatever perimeter cells didn't become asphalt, so the
    // road stays framed by walkway even on faces with no exit.
    for x in 0..TILE_SIZE {
        for z in 0..TILE_SIZE {
            let on_perimeter = x == 0 || x == TILE_SIZE - 1 || z == 0 || z == TILE_SIZE - 1;
            if on_perimeter && p[idx(x, 0, z)] == grass {
                p[idx(x, 0, z)] = sidewalk;
            }
        }
    }

    p
}

#[inline]
fn idx(x: usize, y: usize, z: usize) -> usize {
    x + y * TILE_SIZE + z * TILE_SIZE * TILE_SIZE
}

/// Wall pattern with optional extensions reaching each face. The
/// central pillar is always present so extensions meet cleanly. Solid
/// cells take `color`; the rest stay air.
fn wall_pattern(px: bool, nx: bool, pz: bool, nz: bool, color: Voxel) -> [Voxel; TILE_VOLUME] {
    let mut p = [Voxel::AIR; TILE_VOLUME];

    // Central pillar.
    for y in 0..TILE_SIZE {
        for z in 1..3 {
            for x in 1..3 {
                p[idx(x, y, z)] = color;
            }
        }
    }
    if px {
        for y in 0..TILE_SIZE {
            for z in 1..3 {
                p[idx(3, y, z)] = color;
            }
        }
    }
    if nx {
        for y in 0..TILE_SIZE {
            for z in 1..3 {
                p[idx(0, y, z)] = color;
            }
        }
    }
    if pz {
        for y in 0..TILE_SIZE {
            for x in 1..3 {
                p[idx(x, y, 3)] = color;
            }
        }
    }
    if nz {
        for y in 0..TILE_SIZE {
            for x in 1..3 {
                p[idx(x, y, 0)] = color;
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
// Every field defaults, so a partial set of parameters is legal: the
// registry merges what a caller named over these, and a `.vxlt` written
// before a field existed still loads.
#[serde(default)]
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
        // Propagation packs connectors into a `u32` bitset, so an id
        // past 31 would shift out of range. The built-in tilesets use
        // 0..=3, but nothing enforced that.
        if let Some(tile) = tileset
            .tiles
            .iter()
            .find(|t| t.connectors.iter().any(|c| *c >= 32))
        {
            return Err(GenError::InvalidParams(format!(
                "tile '{}' uses a connector id >= 32",
                tile.name
            )));
        }

        let w = self.width as usize;
        let d = self.depth as usize;
        let n_cells = w * d;
        let all_allowed: u64 = if n_tiles == 64 {
            !0
        } else {
            (1u64 << n_tiles) - 1
        };

        let mut cells: Vec<Cell> = vec![
            Cell {
                allowed: all_allowed,
                collapsed: false
            };
            n_cells
        ];
        let mut rng = StdRng::seed_from_u64(self.seed as u64);

        let weights: Vec<f32> = tileset.tiles.iter().map(|t| t.weight).collect();

        // Treat the grid boundary as connector 0, so wall ends and door
        // mouths can't open into the void. Filter the border domains
        // first, then propagate inward before the first observation.
        {
            let tiles = &tileset.tiles;
            let mut seeded: Vec<usize> = Vec::new();
            for cz in 0..d {
                for cx in 0..w {
                    let cell_i = cz * w + cx;
                    let mut allowed = cells[cell_i].allowed;
                    // (this face points off-grid, face index) per side.
                    let edges = [
                        (cx + 1 == w, 0usize), // +X
                        (cx == 0, 1),          // -X
                        (cz + 1 == d, 2),      // +Z
                        (cz == 0, 3),          // -Z
                    ];
                    for (on_edge, face) in edges {
                        if !on_edge {
                            continue;
                        }
                        // The boundary exposes connector 0; a tile may keep
                        // this cell only if its edge-facing connector is the
                        // complement of 0 (i.e. 0 itself).
                        let mut filtered = 0u64;
                        for (i, tile) in tiles.iter().enumerate().take(n_tiles) {
                            if allowed & (1u64 << i) != 0 && tile.connectors[face] == 0 {
                                filtered |= 1u64 << i;
                            }
                        }
                        allowed = filtered;
                    }
                    if allowed != cells[cell_i].allowed {
                        cells[cell_i].allowed = allowed;
                        seeded.push(cell_i);
                    }
                }
            }
            for cell_i in seeded {
                propagate(&mut cells, w, d, cell_i, tiles);
            }
        }

        // Main collapse loop. Pick the lowest-entropy cell, observe
        // it, propagate. Bail when nothing's left to collapse.
        while let Some(target) = lowest_entropy(&cells, &mut rng) {
            collapse(&mut cells[target], &weights, &mut rng);
            propagate(&mut cells, w, d, target, &tileset.tiles);
        }

        let mut patch = VoxelPatch::new();
        let mut failed_cells: u32 = 0;

        for cz in 0..d {
            for cx in 0..w {
                let cell_i = cz * w + cx;
                // The chosen tile, or `empty` when the domain ended up
                // empty — an over-constrained outcome of the
                // forward-only solver, counted so the UI can warn.
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
                            let voxel = tile.cells[idx(vx, vy, vz)];
                            if !voxel.is_air() {
                                patch.set(ox + vx as i32, oy + vy as i32, oz + vz as i32, voxel);
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

/// Pick the uncollapsed cell with the smallest non-empty domain, ties
/// broken randomly. Scans the whole grid per call, so a full generate
/// is O(cells²) — negligible at the 24×24 ceiling.
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

/// Connector compatibility is by *complement*: symmetric ids match
/// themselves, while the doorway pair is directional — a wall's mouth
/// (2) fits only a floor's socket (3), never another mouth.
fn connector_complement(id: u8) -> u8 {
    match id {
        2 => 3,
        3 => 2,
        other => other,
    }
}

/// Constraint propagation. After a cell shrinks its neighbors may be
/// reducible too, so every domain that shrinks re-queues its own cell.
fn propagate(cells: &mut [Cell], w: usize, d: usize, start: usize, tiles: &[Tile]) {
    let mut stack = vec![start];
    while let Some(idx) = stack.pop() {
        let allowed = cells[idx].allowed;
        // An empty domain is a dead end, not a constraint: propagating
        // from it would force every neighbor to zero and cascade
        // fallback across the grid. The failed cell is counted at emit.
        if allowed == 0 {
            continue;
        }
        let cx = idx % w;
        let cz = idx / w;

        // Face order: 0=+X, 1=-X, 2=+Z, 3=-Z.
        // Each entry: (dx, dz, my_face, neighbor_face).
        let dirs: [(i32, i32, usize, usize); 4] =
            [(1, 0, 0, 1), (-1, 0, 1, 0), (0, 1, 2, 3), (0, -1, 3, 2)];
        for (dx, dz, my_face, neighbor_face) in dirs {
            let nx = cx as i32 + dx;
            let nz = cz as i32 + dz;
            if nx < 0 || nx >= w as i32 || nz < 0 || nz >= d as i32 {
                continue;
            }
            let nidx = nz as usize * w + nx as usize;
            // A collapsed cell is final. With no backtracking we keep
            // its tile even if this neighbor turns out incompatible,
            // rather than zeroing the domain and cascading.
            if cells[nidx].collapsed {
                continue;
            }

            // Connectors my cell currently exposes on `my_face`. The
            // bitset is over connector IDs (assumed to fit in u32).
            let mut my_conns: u32 = 0;
            for (i, tile) in tiles.iter().enumerate() {
                if allowed & (1u64 << i) != 0 {
                    my_conns |= 1u32 << tile.connectors[my_face];
                }
            }

            // Filter the neighbor to tiles whose facing connector
            // complements one of mine, so a door mouth can't seat
            // against another mouth.
            let mut new_allowed: u64 = 0;
            for (j, tile) in tiles.iter().enumerate() {
                if cells[nidx].allowed & (1u64 << j) == 0 {
                    continue;
                }
                let c = connector_complement(tile.connectors[neighbor_face]);
                if my_conns & (1u32 << c) != 0 {
                    new_allowed |= 1u64 << j;
                }
            }

            if new_allowed != cells[nidx].allowed {
                cells[nidx].allowed = new_allowed;
                stack.push(nidx);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_tilesets_stay_within_the_connector_bitset() {
        // Propagation packs exposed connectors into a u32, so an id
        // past 31 would shift out of range. Pinned here so a future
        // tileset can't quietly break adjacency.
        for choice in [WfcTileset::Dungeon, WfcTileset::City] {
            for tile in &choice.build().tiles {
                for c in tile.connectors {
                    assert!(
                        c < 32,
                        "{} tile '{}' uses connector {}",
                        choice.label(),
                        tile.name,
                        c
                    );
                }
            }
        }
    }

    #[test]
    fn test_dungeon_tileset_loads() {
        let ts = WfcTileset::Dungeon.build();
        assert_eq!(ts.tiles.len(), 19);
        assert_eq!(ts.tiles[0].name, "empty");
        assert_eq!(ts.tiles[1].name, "floor");
        // Doorway tiles sit after the cross; the four floor-with-mouth
        // variants close out the list.
        let names: Vec<&str> = ts.tiles.iter().map(|t| t.name).collect();
        assert!(names.contains(&"wall_x_with_door"));
        assert!(names.contains(&"wall_z_with_door"));
        assert!(names.contains(&"floor_door_px"));
        assert!(names.contains(&"floor_door_nz"));
    }

    #[test]
    fn test_door_tile_carves_2x2_portal() {
        // The wall_x_with_door tile must have its central 2-wide × 2-tall
        // region empty (the portal) but the lintel above and the
        // door-jambs at x=0, x=3 must stay solid.
        let ts = WfcTileset::Dungeon.build();
        let door = ts
            .tiles
            .iter()
            .find(|t| t.name == "wall_x_with_door")
            .expect("wall_x_with_door tile missing");
        for y in 0..2 {
            for z in 1..3 {
                for x in 1..3 {
                    assert!(
                        door.cells[idx(x, y, z)].is_air(),
                        "expected portal cell ({},{},{}) to be empty",
                        x,
                        y,
                        z
                    );
                }
            }
        }
        // Lintel above the doorway is intact.
        for z in 1..3 {
            for x in 0..TILE_SIZE {
                assert!(
                    !door.cells[idx(x, 3, z)].is_air(),
                    "lintel cell ({}, 3, {}) should be solid",
                    x,
                    z
                );
            }
        }
        // Door-jambs at the extremes stay solid through the carve y range.
        for y in 0..2 {
            for z in 1..3 {
                assert!(
                    !door.cells[idx(0, y, z)].is_air(),
                    "left jamb gap at ({}, {})",
                    y,
                    z
                );
                assert!(
                    !door.cells[idx(3, y, z)].is_air(),
                    "right jamb gap at ({}, {})",
                    y,
                    z
                );
            }
        }
    }

    #[test]
    fn test_floor_door_variants_share_floor_geometry() {
        let ts = WfcTileset::Dungeon.build();
        let plain_floor = ts
            .tiles
            .iter()
            .find(|t| t.name == "floor")
            .expect("floor tile missing");
        for variant_name in [
            "floor_door_px",
            "floor_door_nx",
            "floor_door_pz",
            "floor_door_nz",
        ] {
            let v = ts
                .tiles
                .iter()
                .find(|t| t.name == variant_name)
                .unwrap_or_else(|| panic!("{} missing", variant_name));
            assert_eq!(
                v.cells, plain_floor.cells,
                "{} should share plain floor's geometry",
                variant_name
            );
            // Exactly one connector must be the door socket `3` (the
            // complement of a wall's door mouth); everything else stays
            // at `0` (open).
            let sockets = v.connectors.iter().filter(|&&c| c == 3).count();
            assert_eq!(
                sockets, 1,
                "{} should have exactly one door-socket connector",
                variant_name
            );
        }
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
                    a.connectors != b.connectors || a.cells != b.cells,
                    "tiles {} and {} are identical",
                    a.name,
                    b.name
                );
            }
        }
    }

    #[test]
    fn test_city_tileset_loads() {
        let ts = WfcTileset::City.build();
        assert_eq!(ts.tiles.len(), 13);
        let names: Vec<&str> = ts.tiles.iter().map(|t| t.name).collect();
        assert!(names.contains(&"grass"));
        assert!(names.contains(&"road_x"));
        assert!(names.contains(&"road_cross"));
        assert!(names.contains(&"building"));
    }

    #[test]
    fn test_city_road_x_has_asphalt_and_sidewalk() {
        let ts = WfcTileset::City.build();
        let road = ts.tiles.iter().find(|t| t.name == "road_x").unwrap();

        // Distinct colors for asphalt vs sidewalk vs grass — checks
        // that the multi-color tile data flows through the new
        // per-cell `Voxel` storage.
        let middle = road.cells[idx(2, 0, 1)]; // central asphalt strip
        let edge = road.cells[idx(2, 0, 0)]; // sidewalk on -Z edge
        assert!(!middle.is_air(), "road interior should be solid");
        assert!(!edge.is_air(), "sidewalk should be solid");
        assert_ne!(middle, edge, "asphalt and sidewalk should differ");

        // y=1 and above are air (no buildings on a road tile).
        for y in 1..TILE_SIZE {
            for z in 0..TILE_SIZE {
                for x in 0..TILE_SIZE {
                    assert!(
                        road.cells[idx(x, y, z)].is_air(),
                        "road_x cell ({}, {}, {}) should be empty",
                        x,
                        y,
                        z
                    );
                }
            }
        }
    }

    #[test]
    fn test_city_building_rises_above_grass_base() {
        let ts = WfcTileset::City.build();
        let b = ts.tiles.iter().find(|t| t.name == "building").unwrap();
        // Building has a 2×2 footprint at x∈{1,2}, z∈{1,2}, y∈{1..=3}.
        for y in 1..TILE_SIZE {
            for z in 1..3 {
                for x in 1..3 {
                    assert!(
                        !b.cells[idx(x, y, z)].is_air(),
                        "building cube cell ({}, {}, {}) should be solid",
                        x,
                        y,
                        z
                    );
                }
            }
        }
        // Building base (y=0) is grass everywhere.
        let g = ts.tiles.iter().find(|t| t.name == "grass").unwrap();
        for x in 0..TILE_SIZE {
            for z in 0..TILE_SIZE {
                assert_eq!(
                    b.cells[idx(x, 0, z)],
                    g.cells[idx(x, 0, z)],
                    "building base layer should match grass tile"
                );
            }
        }
    }

    #[test]
    fn test_city_default_generates_nonempty() {
        let g = WfcGenerator {
            tileset: WfcTileset::City,
            ..Default::default()
        };
        let p = g.generate().unwrap();
        assert!(!p.is_empty());
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
        let a = WfcGenerator {
            seed: 1,
            ..Default::default()
        };
        let b = WfcGenerator {
            seed: 99,
            ..Default::default()
        };
        // Different seeds should pick different tile arrangements.
        assert_ne!(a.generate().unwrap().voxels, b.generate().unwrap().voxels);
    }

    #[test]
    fn test_invalid_params_rejected() {
        let g = WfcGenerator {
            width: 0,
            ..Default::default()
        };
        assert!(g.generate().is_err());
        let g = WfcGenerator {
            depth: 0,
            ..Default::default()
        };
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

    // ---- WFC propagation / connector unit tests (#23, #32) ----------

    fn conn_tile(connectors: [u8; 4]) -> Tile {
        Tile {
            name: "t",
            connectors,
            cells: [Voxel::AIR; TILE_VOLUME],
            weight: 1.0,
        }
    }

    #[test]
    fn propagate_empty_domain_does_not_poison_neighbors() {
        // #23: a cell whose domain collapsed to empty is a dead end. It
        // must not force its neighbors to empty and cascade a single
        // contradiction across the grid.
        let tiles = [conn_tile([0, 0, 0, 0]), conn_tile([1, 1, 1, 1])];
        let mut cells = vec![
            Cell {
                allowed: 0,
                collapsed: false,
            }, // empty (dead end)
            Cell {
                allowed: 0b11,
                collapsed: false,
            }, // full domain
        ];
        propagate(&mut cells, 1, 2, 0, &tiles);
        assert_eq!(
            cells[1].allowed, 0b11,
            "empty domain cascaded into a neighbor"
        );
    }

    #[test]
    fn propagate_never_overwrites_a_collapsed_cell() {
        // #23: two adjacent collapsed cells with incompatible facing
        // connectors. Without backtracking we keep both chosen tiles
        // rather than zeroing the neighbor into a failure.
        let tiles = [conn_tile([0, 0, 0, 0]), conn_tile([1, 1, 1, 1])];
        let mut cells = vec![
            Cell {
                allowed: 0b10,
                collapsed: true,
            }, // tile 1
            Cell {
                allowed: 0b01,
                collapsed: true,
            }, // tile 0 (incompatible)
        ];
        propagate(&mut cells, 1, 2, 0, &tiles);
        assert_eq!(
            cells[1].allowed, 0b01,
            "collapsed cell's domain was overwritten"
        );
    }

    #[test]
    fn connector_complement_pairs_doorway_only() {
        // #32: symmetric IDs self-complement; the doorway pair 2/3 cross.
        assert_eq!(connector_complement(0), 0);
        assert_eq!(connector_complement(1), 1);
        assert_eq!(connector_complement(2), 3);
        assert_eq!(connector_complement(3), 2);
    }

    #[test]
    fn door_mouth_seats_only_against_floor_socket() {
        // #32: a wall's door mouth (+Z face = 2) must accept a floor
        // socket neighbor and reject another door-wall (mouth vs mouth).
        let ts = WfcTileset::Dungeon.build();
        let tiles = &ts.tiles;
        let idx_of = |name: &str| tiles.iter().position(|t| t.name == name).unwrap();
        let wall = idx_of("wall_x_with_door");
        let floor_nz = idx_of("floor_door_nz");
        let wall_bit = 1u64 << wall;
        let full = (1u64 << tiles.len()) - 1;

        // Cell 0 = the door-wall (collapsed); cell 1 is its +Z neighbor.
        let mut cells = vec![
            Cell {
                allowed: wall_bit,
                collapsed: true,
            },
            Cell {
                allowed: full,
                collapsed: false,
            },
        ];
        propagate(&mut cells, 1, 2, 0, tiles);

        assert!(
            cells[1].allowed & (1u64 << floor_nz) != 0,
            "a door mouth should accept a floor-socket neighbor"
        );
        assert!(
            cells[1].allowed & wall_bit == 0,
            "two door mouths must not seat against each other (#32)"
        );
    }

    #[test]
    fn boundary_forbids_open_connectors_on_grid_edge() {
        // On a 1×1 grid every face is a boundary, so only all-0 tiles may
        // appear. Floor lives at y=0 and walls rise above it, so a voxel
        // above the floor would prove the constraint leaked.
        for seed in 0..16 {
            let g = WfcGenerator {
                width: 1,
                depth: 1,
                origin: (0, 0, 0),
                seed,
                tileset: WfcTileset::Dungeon,
            };
            let p = g.generate().unwrap();
            for ((_, y, _), _) in &p.voxels {
                assert_eq!(
                    *y, 0,
                    "1×1 grid produced a wall/door voxel above the floor (seed {})",
                    seed
                );
            }
        }
    }
}
