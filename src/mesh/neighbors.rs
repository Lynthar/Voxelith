//! 26-neighbor lock + voxel query helpers for AO sampling.
//!
//! Per-vertex AO samples cells in a 2×2 footprint on the face's
//! "outside" layer. For voxels at chunk corners these samples can
//! cross **diagonal** chunk boundaries — i.e. require reads from
//! up to 3 axes' worth of neighbor chunks at once. The 6-face
//! `NeighborGuards` used by `is_face_visible` isn't enough; we need
//! the full 26 neighbors (3³ - 1).
//!
//! Lock once at meshing-start, deref through `voxel_at_local` for
//! each AO sample. Missing neighbor chunks (unloaded) → AIR, same
//! convention as the face-culling path.

use parking_lot::{RwLock, RwLockReadGuard};
use std::sync::Arc;

use crate::core::{Chunk, ChunkPos, Voxel, World, CHUNK_SIZE_I32};

/// 26 neighbor `Arc`s, indexed via [`neighbor_index`]. Caller keeps
/// this alive for the duration of any guards derived from it.
pub(crate) type NeighborArcs = [Option<Arc<RwLock<Chunk>>>; 26];

/// 26 neighbor read guards, indexed via [`neighbor_index`]. None
/// when the neighbor chunk isn't loaded.
pub(crate) type NeighborGuards<'a> = [Option<RwLockReadGuard<'a, Chunk>>; 26];

/// Map a `(dx, dy, dz)` offset (each in `{-1, 0, +1}`, not the
/// `(0, 0, 0)` self-cell) to a 0..=25 index.
#[inline]
pub(crate) fn neighbor_index(dx: i32, dy: i32, dz: i32) -> usize {
    debug_assert!((-1..=1).contains(&dx));
    debug_assert!((-1..=1).contains(&dy));
    debug_assert!((-1..=1).contains(&dz));
    debug_assert!(!(dx == 0 && dy == 0 && dz == 0));
    let raw = (dx + 1) as usize + ((dy + 1) as usize) * 3 + ((dz + 1) as usize) * 9;
    // Skip the center (raw == 13) so the 26 valid entries are dense.
    if raw > 13 {
        raw - 1
    } else {
        raw
    }
}

/// Acquire `Arc<RwLock<Chunk>>` handles for the 26 neighbors of
/// `chunk_pos`. Missing chunks → None. Cheap (just `HashMap` lookups
/// and `Arc::clone`); does not lock.
pub(crate) fn neighbor_arcs(world: &World, chunk_pos: ChunkPos) -> NeighborArcs {
    let mut out: NeighborArcs = std::array::from_fn(|_| None);
    let mut idx = 0;
    for dz in -1..=1i32 {
        for dy in -1..=1i32 {
            for dx in -1..=1i32 {
                if dx == 0 && dy == 0 && dz == 0 {
                    continue;
                }
                out[idx] = world.get_chunk(chunk_pos.neighbor(dx, dy, dz));
                idx += 1;
            }
        }
    }
    out
}

/// Read-lock all 26 neighbor chunks. Returns `None` per slot for
/// missing chunks. Caller must keep `arcs` alive for the guards'
/// borrow.
pub(crate) fn lock_neighbors<'a>(arcs: &'a NeighborArcs) -> NeighborGuards<'a> {
    std::array::from_fn(|i| arcs[i].as_ref().map(|a| a.read()))
}

/// Read voxel at chunk-local coordinate `(x, y, z)`. Coordinates
/// outside `[0, CHUNK_SIZE)` route through the corresponding
/// neighbor chunk. Missing neighbor → AIR.
///
/// Each axis can deviate by at most one chunk (the AO sampler only
/// looks one cell out), so we don't need to handle 2-or-more-chunk
/// jumps.
#[inline]
pub(crate) fn voxel_at_local(
    chunk: &Chunk,
    neighbors: &NeighborGuards,
    x: i32,
    y: i32,
    z: i32,
) -> Voxel {
    let cx = chunk_offset(x);
    let cy = chunk_offset(y);
    let cz = chunk_offset(z);
    let lx = x.rem_euclid(CHUNK_SIZE_I32) as usize;
    let ly = y.rem_euclid(CHUNK_SIZE_I32) as usize;
    let lz = z.rem_euclid(CHUNK_SIZE_I32) as usize;
    if cx == 0 && cy == 0 && cz == 0 {
        chunk.get(lx, ly, lz)
    } else {
        let idx = neighbor_index(cx, cy, cz);
        match &neighbors[idx] {
            Some(g) => g.get(lx, ly, lz),
            None => Voxel::AIR,
        }
    }
}

/// 0 if `v` is in `[0, CHUNK_SIZE)`, -1 below, +1 above. Used to
/// pick the neighbor chunk for cross-boundary samples.
#[inline]
fn chunk_offset(v: i32) -> i32 {
    if v < 0 {
        -1
    } else if v >= CHUNK_SIZE_I32 {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbor_index_covers_0_25_uniquely() {
        let mut seen = std::collections::HashSet::new();
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let idx = neighbor_index(dx, dy, dz);
                    assert!(idx < 26, "{} >= 26 for ({},{},{})", idx, dx, dy, dz);
                    assert!(seen.insert(idx), "duplicate index {} at ({},{},{})", idx, dx, dy, dz);
                }
            }
        }
        assert_eq!(seen.len(), 26);
    }
}
