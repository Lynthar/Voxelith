//! World: chunks with spatial indexing, presenting one voxel space
//! across chunk boundaries.

use super::{Chunk, ChunkPos, Voxel, CHUNK_SIZE, CHUNK_SIZE_I32};
use glam::Vec3;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// An inclusive axis-aligned box of voxel cells, `(min, max)` — the
/// shape every "bounds of some cells" answer in the codebase takes.
pub type CellAabb = ((i32, i32, i32), (i32, i32, i32));

/// A world of chunks behind `RwLock`. Unbounded — chunks are created
/// on demand wherever a write lands, so any `i32` voxel coordinate is
/// writable.
#[derive(Default)]
pub struct World {
    /// Chunks indexed by their position
    chunks: HashMap<ChunkPos, Arc<RwLock<Chunk>>>,
}

impl World {
    /// Create a new empty world
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if a chunk exists at the given position
    pub fn has_chunk(&self, pos: ChunkPos) -> bool {
        self.chunks.contains_key(&pos)
    }

    /// The chunk at `pos`, or `None` when it isn't loaded.
    ///
    /// # Safety
    /// Read through the guard. Writing bypasses `set_voxel`'s Moore
    /// dirty propagation, leaving diagonal neighbors with stale AO.
    pub(crate) fn get_chunk(&self, pos: ChunkPos) -> Option<Arc<RwLock<Chunk>>> {
        self.chunks.get(&pos).cloned()
    }

    /// Get or create chunk at position.
    pub(crate) fn get_or_create_chunk(&mut self, pos: ChunkPos) -> Arc<RwLock<Chunk>> {
        self.chunks
            .entry(pos)
            .or_insert_with(|| Arc::new(RwLock::new(Chunk::new())))
            .clone()
    }

    /// Get voxel at world position
    pub fn get_voxel(&self, x: i32, y: i32, z: i32) -> Voxel {
        let chunk_pos = ChunkPos::from_world_pos(x, y, z);
        if let Some(chunk) = self.get_chunk(chunk_pos) {
            let lx = x.rem_euclid(CHUNK_SIZE_I32) as usize;
            let ly = y.rem_euclid(CHUNK_SIZE_I32) as usize;
            let lz = z.rem_euclid(CHUNK_SIZE_I32) as usize;
            chunk.read().get(lx, ly, lz)
        } else {
            Voxel::AIR
        }
    }

    /// Set voxel at world position. Any coordinate is in range; the
    /// chunk holding it is created if it doesn't exist yet.
    pub fn set_voxel(&mut self, x: i32, y: i32, z: i32, voxel: Voxel) {
        let chunk_pos = ChunkPos::from_world_pos(x, y, z);
        let chunk = self.get_or_create_chunk(chunk_pos);

        let lx = x.rem_euclid(CHUNK_SIZE_I32) as usize;
        let ly = y.rem_euclid(CHUNK_SIZE_I32) as usize;
        let lz = z.rem_euclid(CHUNK_SIZE_I32) as usize;

        chunk.write().set(lx, ly, lz, voxel);

        // A boundary write can flip neighbors' face visibility and
        // change their AO, so mark loaded ones dirty. Missing neighbors
        // aren't created — there is nothing to re-mesh.
        self.mark_boundary_neighbors_dirty(chunk_pos, lx, ly, lz);
    }

    /// Mark every loaded neighbor a boundary write can affect — the
    /// full Moore neighborhood, since AO samples all 26 chunks. A face
    /// write reaches 1 neighbor, an edge 3, a corner 7.
    fn mark_boundary_neighbors_dirty(&self, chunk_pos: ChunkPos, lx: usize, ly: usize, lz: usize) {
        let last = CHUNK_SIZE - 1;
        // Per axis: the neighbor offsets this coordinate reaches into.
        // An interior coordinate reaches none, so for a non-boundary
        // write the product below is just `(0, 0, 0)` — skipped.
        let axis_offsets = |l: usize| -> &'static [i32] {
            if l == 0 {
                &[0, -1]
            } else if l == last {
                &[0, 1]
            } else {
                &[0]
            }
        };
        for &dx in axis_offsets(lx) {
            for &dy in axis_offsets(ly) {
                for &dz in axis_offsets(lz) {
                    if (dx, dy, dz) == (0, 0, 0) {
                        continue;
                    }
                    let neighbor_pos = chunk_pos.neighbor(dx, dy, dz);
                    if let Some(neighbor) = self.chunks.get(&neighbor_pos) {
                        neighbor.write().mark_dirty();
                    }
                }
            }
        }
    }

    /// Get all loaded chunk positions
    pub fn chunk_positions(&self) -> impl Iterator<Item = &ChunkPos> {
        self.chunks.keys()
    }

    /// Every loaded chunk, for read-only enumeration.
    ///
    /// # Safety
    /// Never write through these guards — writes belong to `set_voxel`,
    /// which owes its neighbors the dirty propagation.
    pub fn chunks(&self) -> impl Iterator<Item = (&ChunkPos, &Arc<RwLock<Chunk>>)> {
        self.chunks.iter()
    }

    /// Chunk positions in deterministic (x, y, z) order. The backing
    /// `HashMap` iterates unpredictably, so byte-reproducible exports
    /// walk this rather than `chunks()`.
    pub fn sorted_chunk_positions(&self) -> Vec<ChunkPos> {
        let mut positions: Vec<ChunkPos> = self.chunks.keys().copied().collect();
        positions.sort_unstable_by_key(|p| (p.x, p.y, p.z));
        positions
    }

    /// Get number of loaded chunks
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Copy the world, chunk contents and all. Not `impl Clone` —
    /// chunks live behind `Arc<RwLock<…>>`, so a derived clone would
    /// write through to this world's voxels. 256 KB per chunk.
    pub fn deep_clone(&self) -> World {
        World {
            chunks: self
                .chunks
                .iter()
                .map(|(pos, chunk)| (*pos, Arc::new(RwLock::new(chunk.read().clone()))))
                .collect(),
        }
    }

    /// Inclusive cell-space AABB of every solid voxel, or `None` when
    /// there are none; spans `[min, max + 1)` continuously. Walks every
    /// solid voxel, so it is for occasional events, not per frame.
    pub fn scene_aabb(&self) -> Option<CellAabb> {
        let mut bounds: Option<CellAabb> = None;
        for (chunk_pos, chunk) in self.chunks() {
            let chunk = chunk.read();
            if chunk.is_empty() {
                continue;
            }
            let (ox, oy, oz) = chunk_pos.world_origin();
            for (lp, _) in chunk.iter_solid() {
                let p = (ox + lp.x as i32, oy + lp.y as i32, oz + lp.z as i32);
                bounds = Some(match bounds {
                    Some((mn, mx)) => (
                        (mn.0.min(p.0), mn.1.min(p.1), mn.2.min(p.2)),
                        (mx.0.max(p.0), mx.1.max(p.1), mx.2.max(p.2)),
                    ),
                    None => (p, p),
                });
            }
        }
        bounds
    }

    /// Center of the solid AABB in continuous coordinates, or `None`
    /// when the world is empty. A lone voxel at `p` centers at `p + 0.5`.
    /// Used as the default orbit pivot.
    pub fn scene_center(&self) -> Option<Vec3> {
        self.scene_aabb().map(|(min, max)| {
            // AABB in continuous space is [min, max+1); center is the
            // midpoint of that interval per axis.
            Vec3::new(
                (min.0 as f32 + max.0 as f32 + 1.0) * 0.5,
                (min.1 as f32 + max.1 as f32 + 1.0) * 0.5,
                (min.2 as f32 + max.2 as f32 + 1.0) * 0.5,
            )
        })
    }

    /// Get all dirty chunks
    pub fn dirty_chunks(&self) -> Vec<ChunkPos> {
        self.chunks
            .iter()
            .filter(|(_, c)| c.read().is_dirty())
            .map(|(pos, _)| *pos)
            .collect()
    }

    /// Clear all dirty flags
    pub fn clear_dirty_flags(&mut self) {
        for chunk in self.chunks.values() {
            chunk.write().clear_dirty();
        }
    }

    /// Clear all chunks
    pub fn clear(&mut self) {
        self.chunks.clear();
    }

    /// Create a simple test world with a ground plane
    pub fn create_test_ground(&mut self, size: i32, height: i32) {
        let half = size / 2;
        for z in -half..=half {
            for y in 0..height {
                for x in -half..=half {
                    // Grass on top, dirt below
                    let voxel = if y == height - 1 {
                        Voxel::from_rgb(76, 153, 0) // Grass green
                    } else {
                        Voxel::from_rgb(139, 90, 43) // Dirt brown
                    };
                    self.set_voxel(x, y, z, voxel);
                }
            }
        }
    }

    /// Create a simple colored cube for testing
    pub fn create_test_cube(&mut self, center: (i32, i32, i32), half_size: i32) {
        let colors = [
            (255, 0, 0),   // Red
            (0, 255, 0),   // Green
            (0, 0, 255),   // Blue
            (255, 255, 0), // Yellow
            (255, 0, 255), // Magenta
            (0, 255, 255), // Cyan
        ];

        for z in -half_size..=half_size {
            for y in -half_size..=half_size {
                for x in -half_size..=half_size {
                    // Choose color based on position
                    let color_idx = ((x + y + z).unsigned_abs() as usize) % colors.len();
                    let (r, g, b) = colors[color_idx];
                    let voxel = Voxel::from_rgb(r, g, b);
                    self.set_voxel(center.0 + x, center.1 + y, center.2 + z, voxel);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_get_set() {
        let mut world = World::new();

        // Set and get voxel
        let voxel = Voxel::from_rgb(255, 0, 0);
        world.set_voxel(10, 20, 30, voxel);
        assert_eq!(world.get_voxel(10, 20, 30), voxel);

        // Unset voxel should be air
        assert!(world.get_voxel(0, 0, 0).is_air());
    }

    #[test]
    fn sorted_chunk_positions_are_ordered_regardless_of_insertion() {
        // Export determinism hinges on this: whatever order chunks were
        // created in, `sorted_chunk_positions` returns them ascending by
        // (x, y, z). Insert scrambled and assert nothing is lost.
        let mut world = World::new();
        let cells = [
            (40, 0, 40),
            (-40, 0, 0),
            (0, 40, -40),
            (40, 0, -40),
            (-40, -40, 40),
            (0, 0, 0),
        ];
        for &(x, y, z) in &cells {
            world.set_voxel(x, y, z, Voxel::from_rgb(1, 2, 3));
        }
        let sorted = world.sorted_chunk_positions();
        for w in sorted.windows(2) {
            assert!(
                (w[0].x, w[0].y, w[0].z) <= (w[1].x, w[1].y, w[1].z),
                "sorted_chunk_positions returned out-of-order positions"
            );
        }
        assert_eq!(sorted.len(), world.chunk_count());
    }

    #[test]
    fn deep_clone_copies_voxels_and_shares_nothing() {
        // The whole point of the method: writing to the copy must not
        // reach the original through the shared `Arc<RwLock<Chunk>>`.
        let mut world = World::new();
        let red = Voxel::from_rgb(255, 0, 0);
        let blue = Voxel::from_rgb(0, 0, 255);
        world.set_voxel(1, 2, 3, red);
        world.set_voxel(-40, 0, 0, red); // a second chunk

        let mut copy = world.deep_clone();
        assert_eq!(copy.chunk_count(), world.chunk_count());
        assert_eq!(copy.get_voxel(1, 2, 3), red);
        assert_eq!(copy.get_voxel(-40, 0, 0), red);

        copy.set_voxel(1, 2, 3, blue);
        copy.set_voxel(100, 0, 0, blue); // a chunk the original doesn't have
        assert_eq!(
            world.get_voxel(1, 2, 3),
            red,
            "copy wrote through to the original"
        );
        assert!(world.get_voxel(100, 0, 0).is_air());
        assert_eq!(copy.get_voxel(1, 2, 3), blue);
    }

    #[test]
    fn test_world_cross_chunk() {
        let mut world = World::new();

        // Set voxels in different chunks
        world.set_voxel(-1, 0, 0, Voxel::from_rgb(255, 0, 0));
        world.set_voxel(0, 0, 0, Voxel::from_rgb(0, 255, 0));
        world.set_voxel(32, 0, 0, Voxel::from_rgb(0, 0, 255));

        assert_eq!(world.chunk_count(), 3);
    }

    #[test]
    fn test_set_voxel_marks_neighbor_dirty() {
        let mut world = World::new();

        // Pre-create the neighbor chunk by writing into it, then clear
        // dirty flags so we can observe the next write's effect.
        world.set_voxel(32, 0, 0, Voxel::from_rgb(0, 255, 0));
        world.clear_dirty_flags();
        assert!(world.dirty_chunks().is_empty());

        // Write at the +X boundary of chunk (0,0,0). The neighbor (1,0,0)
        // must be marked dirty so its mesh re-culls boundary faces.
        world.set_voxel(31, 0, 0, Voxel::from_rgb(255, 0, 0));

        let dirty: std::collections::HashSet<_> = world.dirty_chunks().into_iter().collect();
        assert!(dirty.contains(&ChunkPos::new(0, 0, 0)));
        assert!(dirty.contains(&ChunkPos::new(1, 0, 0)));
    }

    #[test]
    fn corner_write_marks_diagonal_neighbors_dirty() {
        // AO samples the full 26-chunk neighborhood, so a corner write
        // changes AO in the diagonal neighbors — they must re-mesh too
        // or that corner keeps rendering stale shading.
        let mut world = World::new();
        // Pre-create every chunk around the (0,0,0)/(1,1,1) corner.
        for &(cx, cy, cz) in &[
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ] {
            world.set_voxel(cx * 32, cy * 32, cz * 32, Voxel::from_rgb(1, 2, 3));
        }
        world.clear_dirty_flags();

        // The (31,31,31) corner of chunk (0,0,0) touches all 7.
        world.set_voxel(31, 31, 31, Voxel::from_rgb(255, 0, 0));

        let dirty: std::collections::HashSet<_> = world.dirty_chunks().into_iter().collect();
        for &(cx, cy, cz) in &[
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ] {
            assert!(
                dirty.contains(&ChunkPos::new(cx, cy, cz)),
                "chunk ({cx},{cy},{cz}) should be dirty"
            );
        }
    }

    #[test]
    fn interior_write_marks_only_its_own_chunk() {
        // The Moore-neighborhood sweep must not turn every edit into a
        // 27-chunk re-mesh: only boundary cells reach out at all.
        let mut world = World::new();
        world.set_voxel(32, 0, 0, Voxel::from_rgb(0, 255, 0));
        world.clear_dirty_flags();

        world.set_voxel(16, 16, 16, Voxel::from_rgb(255, 0, 0));

        assert_eq!(world.dirty_chunks(), vec![ChunkPos::new(0, 0, 0)]);
    }

    #[test]
    fn writes_land_anywhere_and_unwritten_space_reads_as_air() {
        let mut world = World::new();

        world.set_voxel(0, 0, 0, Voxel::from_rgb(255, 0, 0));
        assert!(!world.get_voxel(0, 0, 0).is_air());

        // Far from the origin: the chunk is created on demand, so this
        // is a real write, not a silently dropped one.
        world.set_voxel(1000, 1000, 1000, Voxel::from_rgb(0, 255, 0));
        assert!(!world.get_voxel(1000, 1000, 1000).is_air());

        // Never-written space still reads as air rather than panicking.
        assert!(world.get_voxel(-5000, 900, 12345).is_air());
    }

    #[test]
    fn scene_center_is_none_for_empty_world() {
        let world = World::new();
        assert!(world.scene_center().is_none());
    }

    #[test]
    fn scene_center_for_single_voxel_is_cell_center() {
        // A voxel at integer position `p` occupies the AABB
        // [p, p+1)^3, whose center is at p + 0.5.
        let mut world = World::new();
        world.set_voxel(5, 7, 11, Voxel::from_rgb(255, 0, 0));
        let center = world.scene_center().expect("non-empty");
        assert!((center.x - 5.5).abs() < 1e-4);
        assert!((center.y - 7.5).abs() < 1e-4);
        assert!((center.z - 11.5).abs() < 1e-4);
    }

    #[test]
    fn scene_center_spans_aabb_of_all_solid_voxels() {
        // Two cells at (0,0,0) and (2,2,2) cover integers 0..2 per axis,
        // so the continuous AABB is [0, 3) and the center 1.5 each way.
        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgb(255, 0, 0));
        world.set_voxel(2, 2, 2, Voxel::from_rgb(0, 255, 0));
        let center = world.scene_center().expect("non-empty");
        assert!(
            (center - Vec3::new(1.5, 1.5, 1.5)).length() < 1e-4,
            "got {:?}",
            center
        );
    }

    #[test]
    fn scene_center_ignores_air_writes() {
        // Writing AIR shouldn't extend the AABB. Set one solid voxel
        // and one explicit-air write at a far-away position; center
        // should reflect only the solid voxel.
        let mut world = World::new();
        world.set_voxel(0, 0, 0, Voxel::from_rgb(255, 0, 0));
        world.set_voxel(100, 100, 100, Voxel::AIR);
        let center = world.scene_center().expect("non-empty");
        assert!(
            (center - Vec3::new(0.5, 0.5, 0.5)).length() < 1e-4,
            "AIR write extended AABB; got {:?}",
            center
        );
    }

    #[test]
    fn scene_center_handles_negative_coordinates() {
        // AABB across the origin, including negative coords. Range
        // (-2..=1) per axis → continuous [-2, 2), center (0, 0, 0).
        let mut world = World::new();
        world.set_voxel(-2, -2, -2, Voxel::from_rgb(255, 0, 0));
        world.set_voxel(1, 1, 1, Voxel::from_rgb(0, 255, 0));
        let center = world.scene_center().expect("non-empty");
        assert!((center - Vec3::ZERO).length() < 1e-4, "got {:?}", center);
    }
}
