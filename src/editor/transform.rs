//! Selection transforms: 90° rotations around an axis-aligned axis and
//! mirror flips. Both produce an overlap-safe `Vec<VoxelChange>` the
//! caller wraps in `Command::set_voxels` so a single Ctrl+Z reverses
//! the whole transform.
//!
//! **Anchor convention: `sel.min` stays put.** Rotation may swap the
//! AABB's W / H / D (e.g. a 4×1×2 selection rotated around Y becomes
//! 2×1×4) and the new box extends from the original `min` along the
//! swapped axes. No ±0.5 cell drift, no special cases for odd / even
//! side lengths — every transform is a pure integer remap on cell
//! indices local to the source `min`.
//!
//! Internally everything reduces to "given a source cell, produce a
//! destination cell". The shared [`build_remap_changes`] handles the
//! source-clear / destination-write bookkeeping with the same
//! overlap-safe scheme as `build_move_changes` (which is now a thin
//! delta-mapping wrapper around it): when a position is both a
//! source clear and a destination write, the world's pre-transform
//! value is recorded as `old_voxel` so undo restores exactly.

use std::collections::HashMap;

use crate::core::{Voxel, World};

use super::{Selection, VoxelChange};

/// World-aligned axis. Used for both rotation (axis of revolution) and
/// mirroring (axis along which positions are reversed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Axis {
    X,
    Y,
    Z,
}

/// Rotation amount in 90° increments. `Cw` and `Ccw` are clockwise /
/// counter-clockwise when viewed from the positive end of the axis
/// looking back toward the origin (right-hand-rule sign convention).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Quarter {
    Cw,
    Ccw,
    Half,
}

/// AABB of `sel` after a rotation around `axis`. `sel.min` is preserved;
/// the new `max` extends from `min` by the (possibly swapped) extents.
///
/// - `Axis::Y` rotations swap **W ↔ D** (X and Z dimensions)
/// - `Axis::X` rotations swap **H ↔ D** (Y and Z dimensions)
/// - `Axis::Z` rotations swap **W ↔ H** (X and Y dimensions)
/// - `Quarter::Half` (180°) leaves the size unchanged on every axis
pub fn rotated_aabb(sel: Selection, axis: Axis, quarter: Quarter) -> Selection {
    let (w, h, d) = sel.size();
    let (nw, nh, nd) = match (axis, quarter) {
        (_, Quarter::Half) => (w, h, d),
        (Axis::X, _) => (w, d, h),
        (Axis::Y, _) => (d, h, w),
        (Axis::Z, _) => (h, w, d),
    };
    Selection {
        min: sel.min,
        max: (sel.min.0 + nw - 1, sel.min.1 + nh - 1, sel.min.2 + nd - 1),
    }
}

/// Map a world-space cell inside `sel` to its rotated counterpart
/// inside the AABB returned by [`rotated_aabb`]. Pure integer math
/// — no rounding, no drift across multiple rotations.
///
/// The transforms are written on local cell indices `(lx, ly, lz)`
/// relative to `sel.min`. After the transform the result is added
/// back to `sel.min` (which equals the new selection's `min` under
/// our anchor convention) to land in world space.
pub fn rotate_pos(
    sel: Selection,
    axis: Axis,
    quarter: Quarter,
    pos: (i32, i32, i32),
) -> (i32, i32, i32) {
    let (w, h, d) = sel.size();
    let lx = pos.0 - sel.min.0;
    let ly = pos.1 - sel.min.1;
    let lz = pos.2 - sel.min.2;

    // Each arm spells out the local-coord transform for one (axis,
    // quarter) pair. Reading: Y-CW seen from +Y looks down on the X-Z
    // plane and rotates +X → +Z, so a cell at local (lx, lz) goes to
    // (lz, W-1-lx) in the rotated D×W footprint. The other arms are
    // analogous swaps; Half is just two CWs composed.
    let (nlx, nly, nlz) = match (axis, quarter) {
        (Axis::Y, Quarter::Cw) => (lz, ly, w - 1 - lx),
        (Axis::Y, Quarter::Ccw) => (d - 1 - lz, ly, lx),
        (Axis::Y, Quarter::Half) => (w - 1 - lx, ly, d - 1 - lz),
        (Axis::X, Quarter::Cw) => (lx, lz, h - 1 - ly),
        (Axis::X, Quarter::Ccw) => (lx, d - 1 - lz, ly),
        (Axis::X, Quarter::Half) => (lx, h - 1 - ly, d - 1 - lz),
        (Axis::Z, Quarter::Cw) => (ly, w - 1 - lx, lz),
        (Axis::Z, Quarter::Ccw) => (h - 1 - ly, lx, lz),
        (Axis::Z, Quarter::Half) => (w - 1 - lx, h - 1 - ly, lz),
    };

    (sel.min.0 + nlx, sel.min.1 + nly, sel.min.2 + nlz)
}

/// Map a world-space cell inside `sel` to its mirror across the
/// midplane perpendicular to `axis`. The selection's AABB is
/// unchanged.
pub fn mirror_pos(sel: Selection, axis: Axis, pos: (i32, i32, i32)) -> (i32, i32, i32) {
    let (w, h, d) = sel.size();
    let lx = pos.0 - sel.min.0;
    let ly = pos.1 - sel.min.1;
    let lz = pos.2 - sel.min.2;
    let (nlx, nly, nlz) = match axis {
        Axis::X => (w - 1 - lx, ly, lz),
        Axis::Y => (lx, h - 1 - ly, lz),
        Axis::Z => (lx, ly, d - 1 - lz),
    };
    (sel.min.0 + nlx, sel.min.1 + nly, sel.min.2 + nlz)
}

/// Build the `VoxelChange` list for a remap: each non-air voxel in
/// `sel` moves from its source position to `mapping(source)`. Source
/// cells clear to AIR; destination cells receive the moved voxel.
///
/// Overlap (where source and destination share cells) is handled the
/// same way as `build_move_changes`: when a position is both a source
/// clear and a destination write, the world's pre-transform value is
/// recorded as `old_voxel` (so undo restores the original world), and
/// the destination write wins on `new_voxel`.
///
/// The function is generic over the mapping so rotation / mirror /
/// translation all share the same overlap bookkeeping.
pub fn build_remap_changes<F>(
    world: &World,
    sel: Selection,
    mapping: F,
) -> Vec<VoxelChange>
where
    F: Fn((i32, i32, i32)) -> (i32, i32, i32),
{
    let mut by_pos: HashMap<(i32, i32, i32), (Voxel, Voxel)> = HashMap::new();

    let originals: Vec<((i32, i32, i32), Voxel)> = sel
        .iter_cells()
        .filter_map(|p| {
            let v = world.get_voxel(p.0, p.1, p.2);
            if v.is_air() {
                None
            } else {
                Some((p, v))
            }
        })
        .collect();

    // Step 1: source positions clear to AIR, with their pre-transform
    // value pinned as old_voxel.
    for &(p, old) in &originals {
        by_pos.insert(p, (old, Voxel::AIR));
    }

    // Step 2: destination positions receive the moved voxel. For new
    // (non-overlapping) destinations, read old_voxel from the world
    // now. For overlap destinations, `entry().or_insert` preserves
    // the world-old that Step 1 stored — undo restores exactly.
    for &(src, vox) in &originals {
        let dest = mapping(src);
        let world_old = world.get_voxel(dest.0, dest.1, dest.2);
        let entry = by_pos.entry(dest).or_insert((world_old, Voxel::AIR));
        entry.1 = vox;
    }

    by_pos
        .into_iter()
        .filter_map(|(pos, (old_voxel, new_voxel))| {
            if old_voxel == new_voxel {
                None
            } else {
                Some(VoxelChange { pos, old_voxel, new_voxel })
            }
        })
        .collect()
}

/// Convenience: rotate `sel`'s contents around `axis` by `quarter`,
/// returning `(new_selection_aabb, voxel_changes)`. The caller wraps
/// `voxel_changes` in `Command::set_voxels` and updates
/// `editor.selection` to `new_selection_aabb`.
pub fn rotate_selection_changes(
    world: &World,
    sel: Selection,
    axis: Axis,
    quarter: Quarter,
) -> (Selection, Vec<VoxelChange>) {
    let new_sel = rotated_aabb(sel, axis, quarter);
    let changes = build_remap_changes(world, sel, |p| rotate_pos(sel, axis, quarter, p));
    (new_sel, changes)
}

/// Convenience: mirror `sel`'s contents across the midplane
/// perpendicular to `axis`. Returns the changes; the AABB is
/// unchanged so the caller doesn't update `editor.selection`.
pub fn mirror_selection_changes(
    world: &World,
    sel: Selection,
    axis: Axis,
) -> Vec<VoxelChange> {
    build_remap_changes(world, sel, |p| mirror_pos(sel, axis, p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::Command;

    fn voxel(r: u8, g: u8, b: u8) -> Voxel {
        Voxel::from_rgb(r, g, b)
    }

    // -------- rotated_aabb --------

    #[test]
    fn rotated_aabb_y_swaps_w_and_d() {
        let s = Selection::from_corners((0, 0, 0), (3, 0, 1)); // 4×1×2
        let r = rotated_aabb(s, Axis::Y, Quarter::Cw);
        assert_eq!(r.min, (0, 0, 0));
        assert_eq!(r.size(), (2, 1, 4));
    }

    #[test]
    fn rotated_aabb_x_swaps_h_and_d() {
        let s = Selection::from_corners((0, 0, 0), (3, 4, 1)); // 4×5×2
        let r = rotated_aabb(s, Axis::X, Quarter::Cw);
        assert_eq!(r.min, (0, 0, 0));
        assert_eq!(r.size(), (4, 2, 5));
    }

    #[test]
    fn rotated_aabb_z_swaps_w_and_h() {
        let s = Selection::from_corners((0, 0, 0), (3, 4, 1)); // 4×5×2
        let r = rotated_aabb(s, Axis::Z, Quarter::Cw);
        assert_eq!(r.min, (0, 0, 0));
        assert_eq!(r.size(), (5, 4, 2));
    }

    #[test]
    fn rotated_aabb_half_preserves_size() {
        let s = Selection::from_corners((1, 2, 3), (4, 5, 6));
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            let r = rotated_aabb(s, axis, Quarter::Half);
            assert_eq!(r.min, s.min);
            assert_eq!(r.size(), s.size());
        }
    }

    #[test]
    fn rotated_aabb_preserves_min_at_arbitrary_origin() {
        // Anchor convention: regardless of where the source min sits
        // in world space, the rotated AABB starts from the same min.
        let s = Selection::from_corners((-5, 10, -3), (-2, 11, -1)); // 4×2×3
        let r = rotated_aabb(s, Axis::Y, Quarter::Cw);
        assert_eq!(r.min, (-5, 10, -3));
        assert_eq!(r.size(), (3, 2, 4));
    }

    // -------- rotate_pos --------

    #[test]
    fn rotate_y_cw_known_corners() {
        // 4×1×2 selection at origin, Y-CW.
        // Local (lx, lz) → (lz, W-1-lx), W=4.
        // Corner (0,0,0): local (0,0,0) → (0,0,3) world.
        // Corner (3,0,1): local (3,0,1) → local (1,0,0) → (1,0,0) world.
        let s = Selection::from_corners((0, 0, 0), (3, 0, 1));
        assert_eq!(rotate_pos(s, Axis::Y, Quarter::Cw, (0, 0, 0)), (0, 0, 3));
        assert_eq!(rotate_pos(s, Axis::Y, Quarter::Cw, (3, 0, 1)), (1, 0, 0));
    }

    #[test]
    fn rotate_y_cw_four_times_is_identity() {
        let s = Selection::from_corners((0, 0, 0), (3, 0, 1));
        for cell in s.iter_cells() {
            // The rotated AABB's footprint changes between steps, so
            // we have to thread the new selection through each rotation.
            let mut p = cell;
            let mut sel = s;
            for _ in 0..4 {
                p = rotate_pos(sel, Axis::Y, Quarter::Cw, p);
                sel = rotated_aabb(sel, Axis::Y, Quarter::Cw);
            }
            assert_eq!(p, cell, "4× Y-CW should be identity for {:?}", cell);
            assert_eq!(sel, s);
        }
    }

    #[test]
    fn rotate_cw_then_ccw_is_identity_all_axes() {
        let s = Selection::from_corners((0, 0, 0), (2, 3, 4));
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            for cell in s.iter_cells() {
                let cw = rotate_pos(s, axis, Quarter::Cw, cell);
                let s_cw = rotated_aabb(s, axis, Quarter::Cw);
                let back = rotate_pos(s_cw, axis, Quarter::Ccw, cw);
                assert_eq!(back, cell, "CW then CCW around {:?} should round-trip {:?}", axis, cell);
            }
        }
    }

    #[test]
    fn rotate_half_is_self_inverse_all_axes() {
        let s = Selection::from_corners((-2, -3, -4), (1, 0, 1));
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            for cell in s.iter_cells() {
                let half = rotate_pos(s, axis, Quarter::Half, cell);
                // Half rotation preserves AABB.
                let s2 = rotated_aabb(s, axis, Quarter::Half);
                assert_eq!(s2, s);
                let back = rotate_pos(s2, axis, Quarter::Half, half);
                assert_eq!(back, cell);
            }
        }
    }

    #[test]
    fn rotate_pos_stays_inside_new_aabb() {
        // Every source cell, rotated, must land inside the new AABB.
        let s = Selection::from_corners((10, 20, 30), (13, 22, 35));
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            for q in [Quarter::Cw, Quarter::Ccw, Quarter::Half] {
                let new_sel = rotated_aabb(s, axis, q);
                for cell in s.iter_cells() {
                    let p = rotate_pos(s, axis, q, cell);
                    assert!(
                        new_sel.contains(p),
                        "rotate {:?} {:?}: {:?} → {:?} not in {:?}",
                        axis, q, cell, p, new_sel
                    );
                }
            }
        }
    }

    // -------- mirror_pos --------

    #[test]
    fn mirror_x_reverses_x_only() {
        let s = Selection::from_corners((0, 0, 0), (3, 2, 1));
        assert_eq!(mirror_pos(s, Axis::X, (0, 0, 0)), (3, 0, 0));
        assert_eq!(mirror_pos(s, Axis::X, (3, 0, 0)), (0, 0, 0));
        assert_eq!(mirror_pos(s, Axis::X, (1, 1, 1)), (2, 1, 1));
    }

    #[test]
    fn mirror_twice_is_identity_all_axes() {
        let s = Selection::from_corners((-1, -2, -3), (2, 1, 0));
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            for cell in s.iter_cells() {
                let once = mirror_pos(s, axis, cell);
                let twice = mirror_pos(s, axis, once);
                assert_eq!(twice, cell);
            }
        }
    }

    // -------- build_remap_changes (overlap + undo) --------

    #[test]
    fn rotate_y_cw_overlap_writes_correct_changes() {
        // 2×1×1 region at origin, rotated Y-CW.
        // Local (lx, lz) → (lz, W-1-lx), W=2:
        //   (0,0,0) → (0,0,1)   [R moves to +Z end]
        //   (1,0,0) → (0,0,0)   [G moves to where R was]
        // (0,0,0) is a destination AND a source — overlap. Build:
        //   (0,0,0): old=R, new=G    (R cleared by Step 1, G written by Step 2)
        //   (1,0,0): old=G, new=AIR  (cleared, no dest writes here)
        //   (0,0,1): old=AIR, new=R  (fresh dest)
        let mut world = World::new();
        let r = voxel(255, 0, 0);
        let g = voxel(0, 255, 0);
        world.set_voxel(0, 0, 0, r);
        world.set_voxel(1, 0, 0, g);
        let s = Selection::from_corners((0, 0, 0), (1, 0, 0));

        let (new_sel, changes) = rotate_selection_changes(&world, s, Axis::Y, Quarter::Cw);
        // 2×1×1 → 1×1×2 along +Z, anchored at min.
        assert_eq!(new_sel, Selection::from_corners((0, 0, 0), (0, 0, 1)));

        let map: HashMap<_, _> = changes
            .iter()
            .map(|c| (c.pos, (c.old_voxel, c.new_voxel)))
            .collect();
        assert_eq!(map.get(&(0, 0, 0)), Some(&(r, g)));
        assert_eq!(map.get(&(1, 0, 0)), Some(&(g, Voxel::AIR)));
        assert_eq!(map.get(&(0, 0, 1)), Some(&(Voxel::AIR, r)));
        assert_eq!(changes.len(), 3);
    }

    #[test]
    fn rotate_y_cw_overlap_undo_round_trips() {
        // Acid test for the overlap case: apply, undo, world is byte-
        // exact what it was. Verifies build_remap_changes pinned the
        // right `old_voxel` for the (0,0,0) overlap cell.
        let mut world = World::new();
        let r = voxel(255, 0, 0);
        let g = voxel(0, 255, 0);
        world.set_voxel(0, 0, 0, r);
        world.set_voxel(1, 0, 0, g);
        let s = Selection::from_corners((0, 0, 0), (1, 0, 0));

        let (_, changes) = rotate_selection_changes(&world, s, Axis::Y, Quarter::Cw);
        let cmd = Command::set_voxels(changes);
        cmd.execute(&mut world);
        assert_eq!(world.get_voxel(0, 0, 0), g);
        assert_eq!(world.get_voxel(0, 0, 1), r);
        assert!(world.get_voxel(1, 0, 0).is_air());

        cmd.undo(&mut world);
        assert_eq!(world.get_voxel(0, 0, 0), r);
        assert_eq!(world.get_voxel(1, 0, 0), g);
        assert!(world.get_voxel(0, 0, 1).is_air());
    }

    #[test]
    fn rotate_then_undo_round_trips() {
        // The acid test: apply rotate, then undo, world must be byte-
        // exact what it was before.
        let mut world = World::new();
        world.set_voxel(0, 0, 0, voxel(255, 0, 0));
        world.set_voxel(1, 0, 0, voxel(0, 255, 0));
        world.set_voxel(2, 0, 0, voxel(0, 0, 255));
        world.set_voxel(3, 0, 0, voxel(255, 255, 0));
        let s = Selection::from_corners((0, 0, 0), (3, 0, 0));

        let (_, changes) = rotate_selection_changes(&world, s, Axis::Y, Quarter::Cw);
        let cmd = Command::set_voxels(changes);
        cmd.execute(&mut world);

        // After Y-CW: 4×1×1 → 1×1×4. Original line along +X is now
        // a line along +Z mirrored: cell at lx=0 → lz=3, lx=1 → lz=2, etc.
        assert!(world.get_voxel(1, 0, 0).is_air());
        assert!(world.get_voxel(2, 0, 0).is_air());
        assert!(world.get_voxel(3, 0, 0).is_air());

        // Undo: every original cell back.
        cmd.undo(&mut world);
        assert_eq!(world.get_voxel(0, 0, 0), voxel(255, 0, 0));
        assert_eq!(world.get_voxel(1, 0, 0), voxel(0, 255, 0));
        assert_eq!(world.get_voxel(2, 0, 0), voxel(0, 0, 255));
        assert_eq!(world.get_voxel(3, 0, 0), voxel(255, 255, 0));
        // No stray writes outside the original line.
        assert!(world.get_voxel(0, 0, 1).is_air());
        assert!(world.get_voxel(0, 0, 2).is_air());
        assert!(world.get_voxel(0, 0, 3).is_air());
    }

    #[test]
    fn rotate_preserves_voxels_inside_selection() {
        // After rotation, the multiset of non-air voxels inside the new
        // AABB equals the multiset inside the old AABB — rotation
        // conserves material, just relocates it.
        let mut world = World::new();
        let red = voxel(255, 0, 0);
        let green = voxel(0, 255, 0);
        let blue = voxel(0, 0, 255);
        world.set_voxel(0, 0, 0, red);
        world.set_voxel(1, 0, 0, green);
        world.set_voxel(0, 1, 0, blue);
        let s = Selection::from_corners((0, 0, 0), (1, 1, 0));

        let (new_sel, changes) = rotate_selection_changes(&world, s, Axis::Z, Quarter::Cw);
        Command::set_voxels(changes).execute(&mut world);

        // Vec-of-tuples (Voxel doesn't impl Hash) — sort and compare.
        let mut after: Vec<[u8; 4]> = new_sel
            .iter_cells()
            .filter_map(|c| {
                let v = world.get_voxel(c.0, c.1, c.2);
                if v.is_air() {
                    None
                } else {
                    Some([v.r, v.g, v.b, v.a])
                }
            })
            .collect();
        after.sort();
        let mut expected = vec![
            [red.r, red.g, red.b, red.a],
            [green.r, green.g, green.b, green.a],
            [blue.r, blue.g, blue.b, blue.a],
        ];
        expected.sort();
        assert_eq!(after, expected);
    }

    #[test]
    fn mirror_then_undo_round_trips() {
        let mut world = World::new();
        world.set_voxel(0, 0, 0, voxel(255, 0, 0));
        world.set_voxel(2, 0, 0, voxel(0, 0, 255));
        let s = Selection::from_corners((0, 0, 0), (2, 0, 0));

        let changes = mirror_selection_changes(&world, s, Axis::X);
        let cmd = Command::set_voxels(changes);
        cmd.execute(&mut world);

        // Mirrored across X: (0,0,0) and (2,0,0) swap.
        assert_eq!(world.get_voxel(0, 0, 0), voxel(0, 0, 255));
        assert_eq!(world.get_voxel(2, 0, 0), voxel(255, 0, 0));

        cmd.undo(&mut world);
        assert_eq!(world.get_voxel(0, 0, 0), voxel(255, 0, 0));
        assert_eq!(world.get_voxel(2, 0, 0), voxel(0, 0, 255));
    }

    #[test]
    fn mirror_palindromic_selection_is_noop() {
        // A selection whose contents are symmetric across the mirror
        // axis should produce zero changes (every cell maps to a cell
        // holding the same value).
        let mut world = World::new();
        let r = voxel(255, 0, 0);
        world.set_voxel(0, 0, 0, r);
        world.set_voxel(2, 0, 0, r);
        // (1,0,0) is air, also symmetric.
        let s = Selection::from_corners((0, 0, 0), (2, 0, 0));

        let changes = mirror_selection_changes(&world, s, Axis::X);
        // All writes are identity (r→r, air→air), filtered out.
        assert!(changes.is_empty());
    }

    #[test]
    fn rotate_into_existing_voxels_records_correct_old() {
        // Rotation lands a moved voxel onto a cell that already holds
        // a different voxel outside the selection. The destination
        // cell IS inside the new AABB (rotated_aabb extends from min)
        // so this is the "destination overwrites pre-existing" case.
        // old_voxel must reflect the world state, not AIR.
        let mut world = World::new();
        let moving = voxel(255, 0, 0);
        let pre_existing = voxel(0, 0, 255);
        // 2×1×1 selection along +X; Y-CW rotates it to a 1×1×2 line
        // along +Z. The dest cell (0,0,1) already holds something.
        world.set_voxel(0, 0, 0, moving);
        world.set_voxel(1, 0, 0, moving);
        world.set_voxel(0, 0, 1, pre_existing);
        let s = Selection::from_corners((0, 0, 0), (1, 0, 0));

        let (_, changes) = rotate_selection_changes(&world, s, Axis::Y, Quarter::Cw);
        let map: HashMap<_, _> = changes
            .iter()
            .map(|c| (c.pos, (c.old_voxel, c.new_voxel)))
            .collect();
        // Dest (0,0,1) was pre_existing → moving. Undo must restore pre_existing.
        assert_eq!(map.get(&(0, 0, 1)), Some(&(pre_existing, moving)));

        // Apply + undo, world is byte-exact what it was.
        let cmd = Command::set_voxels(changes);
        cmd.execute(&mut world);
        cmd.undo(&mut world);
        assert_eq!(world.get_voxel(0, 0, 0), moving);
        assert_eq!(world.get_voxel(1, 0, 0), moving);
        assert_eq!(world.get_voxel(0, 0, 1), pre_existing);
    }

    #[test]
    fn rotate_empty_selection_produces_no_changes() {
        let world = World::new();
        let s = Selection::from_corners((0, 0, 0), (5, 5, 5));
        let (_, changes) = rotate_selection_changes(&world, s, Axis::Y, Quarter::Cw);
        assert!(changes.is_empty());
    }
}
