//! Clipboard for box selection: copy / cut / paste of voxel groups.
//!
//! `Clipboard` holds **non-air** voxels extracted from a `Selection`,
//! storing positions relative to the selection's `min` corner. Paste
//! composites onto the destination world (Goxel `MODE_OVER` / vengi
//! `mergeVolumes` semantics) — air cells in the source aren't stored,
//! so paste doesn't punch holes in the destination.

use std::collections::HashMap;

use crate::core::{Voxel, World};

use super::{Selection, VoxelChange};

/// Voxel data extracted from a selection, stored relative to its
/// local origin so it can be pasted anywhere.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Clipboard {
    /// Non-air voxels at positions relative to the source selection's
    /// `min` corner. `(0, 0, 0)` is the cell at `selection.min`.
    pub voxels: Vec<((i32, i32, i32), Voxel)>,
    /// Footprint `(W, H, D)` of the source selection. Preserved
    /// separately so paste's auto-select-after lands on the *full*
    /// AABB of the source — not just the bounding box of its non-air
    /// cells (which would shrink the selection if the source had air
    /// gaps at its corners).
    pub size: (i32, i32, i32),
}

impl Clipboard {
    /// True when the clipboard holds no pastable voxels. A selection
    /// of pure air still produces a clipboard with non-zero `size`
    /// but `is_empty() == true`.
    pub fn is_empty(&self) -> bool {
        self.voxels.is_empty()
    }

    /// Number of non-air voxels in the clipboard.
    pub fn voxel_count(&self) -> usize {
        self.voxels.len()
    }
}

/// Extract non-air voxels from `world` that lie inside `selection`,
/// storing positions relative to `selection.min`. Air cells are
/// skipped — paste should composite over the destination, not erase
/// existing voxels in the AABB.
pub fn copy_selection_to_clipboard(world: &World, selection: Selection) -> Clipboard {
    let mut voxels = Vec::new();
    for (x, y, z) in selection.iter_cells() {
        let v = world.get_voxel(x, y, z);
        if !v.is_air() {
            voxels.push((
                (
                    x - selection.min.0,
                    y - selection.min.1,
                    z - selection.min.2,
                ),
                v,
            ));
        }
    }
    Clipboard {
        voxels,
        size: selection.size(),
    }
}

/// Build the `VoxelChange` list to paste `clipboard` so its local
/// origin lands at world-space `dest`. Identity writes (destination
/// already holds the same voxel) are dropped so an in-place
/// Copy → Paste doesn't bloat the undo history with a no-op command.
/// Caller wraps the returned changes in `Command::set_voxels`.
pub fn build_paste_changes(
    world: &World,
    clipboard: &Clipboard,
    dest: (i32, i32, i32),
) -> Vec<VoxelChange> {
    clipboard
        .voxels
        .iter()
        .filter_map(|&(rel, new_voxel)| {
            let pos = (dest.0 + rel.0, dest.1 + rel.1, dest.2 + rel.2);
            let old_voxel = world.get_voxel(pos.0, pos.1, pos.2);
            if old_voxel == new_voxel {
                None
            } else {
                Some(VoxelChange {
                    pos,
                    old_voxel,
                    new_voxel,
                })
            }
        })
        .collect()
}

/// Build the `VoxelChange` list to clear non-air voxels inside
/// `selection`. Used by both Delete and Cut. Air cells are skipped
/// — they'd produce noop changes anyway.
pub fn build_clear_changes(world: &World, selection: Selection) -> Vec<VoxelChange> {
    let mut changes = Vec::new();
    for (x, y, z) in selection.iter_cells() {
        let old_voxel = world.get_voxel(x, y, z);
        if !old_voxel.is_air() {
            changes.push(VoxelChange {
                pos: (x, y, z),
                old_voxel,
                new_voxel: Voxel::AIR,
            });
        }
    }
    changes
}

/// Build the `VoxelChange` list to translate `selection`'s non-air
/// voxels by `delta`. Source cells become air; destination cells
/// receive the moved voxel. When source and destination overlap
/// (`delta` smaller than the selection's extent on some axis), the
/// destination value wins on `new_voxel`, but `old_voxel` always
/// reflects the world's true pre-move state — so undo restores
/// exactly.
///
/// Caller wraps the result in `Command::set_voxels` for undo support
/// and updates `editor.selection = Some(selection.translated(delta))`
/// after executing.
pub fn build_move_changes(
    world: &World,
    selection: Selection,
    delta: (i32, i32, i32),
) -> Vec<VoxelChange> {
    if delta == (0, 0, 0) {
        return Vec::new();
    }

    // Aggregate by destination cell. Each touched position gets a
    // `(world_old, final_new)` pair — `final_new` may flip multiple
    // times during construction but `world_old` is recorded exactly
    // once (from the world before any change), preserving undo
    // correctness through overlapping moves.
    let mut by_pos: HashMap<(i32, i32, i32), (Voxel, Voxel)> = HashMap::new();

    // Collect source voxels first so we can iterate them twice
    // without re-reading the world.
    let originals: Vec<((i32, i32, i32), Voxel)> = selection
        .iter_cells()
        .filter_map(|(x, y, z)| {
            let v = world.get_voxel(x, y, z);
            if v.is_air() {
                None
            } else {
                Some(((x, y, z), v))
            }
        })
        .collect();

    // Step 1: source positions clear to AIR.
    for &(p, old) in &originals {
        by_pos.insert(p, (old, Voxel::AIR));
    }

    // Step 2: destination positions receive the moved voxel.
    // `entry().or_insert` preserves any pre-existing entry's
    // `old_voxel` (set by Step 1 for overlap cells); for fresh
    // destination cells we read `old_voxel` from the world now.
    for &(src, vox) in &originals {
        let dest = (src.0 + delta.0, src.1 + delta.1, src.2 + delta.2);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::Command;

    fn voxel(r: u8, g: u8, b: u8) -> Voxel {
        Voxel::from_rgb(r, g, b)
    }

    #[test]
    fn clipboard_skips_air_cells() {
        let mut world = World::new();
        // 3×1×1 region with air in the middle.
        world.set_voxel(0, 0, 0, voxel(255, 0, 0));
        world.set_voxel(2, 0, 0, voxel(0, 0, 255));

        let sel = Selection::from_corners((0, 0, 0), (2, 0, 0));
        let cb = copy_selection_to_clipboard(&world, sel);

        assert_eq!(cb.voxel_count(), 2);
        assert_eq!(cb.size, (3, 1, 1));
        // Stored relative to selection.min — both at y=z=0.
        let positions: Vec<_> = cb.voxels.iter().map(|(p, _)| *p).collect();
        assert!(positions.contains(&(0, 0, 0)));
        assert!(positions.contains(&(2, 0, 0)));
        assert!(!positions.contains(&(1, 0, 0))); // air was skipped
    }

    #[test]
    fn empty_selection_produces_empty_clipboard_with_size() {
        let world = World::new();
        let sel = Selection::from_corners((0, 0, 0), (4, 4, 4));
        let cb = copy_selection_to_clipboard(&world, sel);

        assert!(cb.is_empty());
        assert_eq!(cb.size, (5, 5, 5));
    }

    #[test]
    fn paste_offsets_relative_positions_to_dest() {
        let mut world = World::new();
        world.set_voxel(0, 0, 0, voxel(255, 0, 0));
        world.set_voxel(1, 0, 0, voxel(0, 255, 0));

        let sel = Selection::from_corners((0, 0, 0), (1, 0, 0));
        let cb = copy_selection_to_clipboard(&world, sel);

        let changes = build_paste_changes(&world, &cb, (10, 5, 0));
        let positions: Vec<_> = changes.iter().map(|c| c.pos).collect();
        assert_eq!(positions.len(), 2);
        assert!(positions.contains(&(10, 5, 0)));
        assert!(positions.contains(&(11, 5, 0)));
    }

    #[test]
    fn paste_drops_identity_writes() {
        // Pasting back at the original location with all destination
        // cells already holding the same voxels: zero changes.
        let mut world = World::new();
        world.set_voxel(0, 0, 0, voxel(255, 0, 0));
        let sel = Selection::from_corners((0, 0, 0), (0, 0, 0));
        let cb = copy_selection_to_clipboard(&world, sel);

        let changes = build_paste_changes(&world, &cb, (0, 0, 0));
        assert!(changes.is_empty());
    }

    #[test]
    fn copy_paste_round_trip_at_offset() {
        // A 2×2×1 patch copied, world cleared, pasted at +5 X — the
        // destination should hold exactly the original voxel pattern
        // shifted by (5, 0, 0).
        let mut world = World::new();
        let red = voxel(255, 0, 0);
        let blue = voxel(0, 0, 255);
        world.set_voxel(0, 0, 0, red);
        world.set_voxel(1, 0, 0, blue);
        world.set_voxel(0, 1, 0, blue);
        world.set_voxel(1, 1, 0, red);

        let sel = Selection::from_corners((0, 0, 0), (1, 1, 0));
        let cb = copy_selection_to_clipboard(&world, sel);

        // Clear the source.
        world.set_voxel(0, 0, 0, Voxel::AIR);
        world.set_voxel(1, 0, 0, Voxel::AIR);
        world.set_voxel(0, 1, 0, Voxel::AIR);
        world.set_voxel(1, 1, 0, Voxel::AIR);

        // Paste at +5 X.
        let changes = build_paste_changes(&world, &cb, (5, 0, 0));
        let cmd = Command::set_voxels(changes);
        cmd.execute(&mut world);

        assert_eq!(world.get_voxel(5, 0, 0), red);
        assert_eq!(world.get_voxel(6, 0, 0), blue);
        assert_eq!(world.get_voxel(5, 1, 0), blue);
        assert_eq!(world.get_voxel(6, 1, 0), red);
        // Source still empty.
        assert!(world.get_voxel(0, 0, 0).is_air());
    }

    #[test]
    fn clear_changes_skip_air() {
        let mut world = World::new();
        world.set_voxel(0, 0, 0, voxel(255, 0, 0));
        // (1, 0, 0) is air — should not generate a change.

        let sel = Selection::from_corners((0, 0, 0), (1, 0, 0));
        let changes = build_clear_changes(&world, sel);

        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].pos, (0, 0, 0));
        assert_eq!(changes[0].new_voxel, Voxel::AIR);
    }

    #[test]
    fn move_zero_delta_is_noop() {
        let mut world = World::new();
        world.set_voxel(0, 0, 0, voxel(255, 0, 0));
        let sel = Selection::from_corners((0, 0, 0), (0, 0, 0));
        let changes = build_move_changes(&world, sel, (0, 0, 0));
        assert!(changes.is_empty());
    }

    #[test]
    fn move_non_overlapping_clears_source_writes_dest() {
        // 2-cell selection moved far enough that source and dest
        // don't overlap. Source should clear, dest should receive.
        let mut world = World::new();
        let red = voxel(255, 0, 0);
        let green = voxel(0, 255, 0);
        world.set_voxel(0, 0, 0, red);
        world.set_voxel(1, 0, 0, green);

        let sel = Selection::from_corners((0, 0, 0), (1, 0, 0));
        let changes = build_move_changes(&world, sel, (10, 0, 0));

        // Build a position-keyed map for easy assertions.
        let map: std::collections::HashMap<_, _> = changes
            .iter()
            .map(|c| (c.pos, (c.old_voxel, c.new_voxel)))
            .collect();
        // Source cells cleared.
        assert_eq!(map.get(&(0, 0, 0)), Some(&(red, Voxel::AIR)));
        assert_eq!(map.get(&(1, 0, 0)), Some(&(green, Voxel::AIR)));
        // Destination cells written.
        assert_eq!(map.get(&(10, 0, 0)), Some(&(Voxel::AIR, red)));
        assert_eq!(map.get(&(11, 0, 0)), Some(&(Voxel::AIR, green)));
        assert_eq!(map.len(), 4);
    }

    #[test]
    fn move_overlapping_keeps_world_old_for_overlap_cells() {
        // 3-wide selection moved by +1: cells (1) and (2) are in
        // both the source and the destination. Their `old_voxel`
        // must be the world's pre-move value (so undo restores
        // exactly), not the AIR that the source-clear pass would
        // try to overwrite.
        let mut world = World::new();
        let r = voxel(255, 0, 0);
        let g = voxel(0, 255, 0);
        let b = voxel(0, 0, 255);
        world.set_voxel(0, 0, 0, r);
        world.set_voxel(1, 0, 0, g);
        world.set_voxel(2, 0, 0, b);

        let sel = Selection::from_corners((0, 0, 0), (2, 0, 0));
        let changes = build_move_changes(&world, sel, (1, 0, 0));

        let map: std::collections::HashMap<_, _> = changes
            .iter()
            .map(|c| (c.pos, (c.old_voxel, c.new_voxel)))
            .collect();

        // (0,0,0): cleared (only source, no destination overlap there).
        assert_eq!(map.get(&(0, 0, 0)), Some(&(r, Voxel::AIR)));
        // (1,0,0): both source-clear and dest-write — must remember
        // the original `g`, end up holding `r` (moved from cell 0).
        assert_eq!(map.get(&(1, 0, 0)), Some(&(g, r)));
        // (2,0,0): both source-clear and dest-write — original `b`,
        // ends up holding `g`.
        assert_eq!(map.get(&(2, 0, 0)), Some(&(b, g)));
        // (3,0,0): only dest-write — original AIR, ends up `b`.
        assert_eq!(map.get(&(3, 0, 0)), Some(&(Voxel::AIR, b)));
        assert_eq!(map.len(), 4);
    }

    #[test]
    fn move_then_undo_round_trips_through_overlap() {
        // The acid test: apply build_move_changes, then undo, the
        // world must return to its exact pre-move state.
        let mut world = World::new();
        let r = voxel(255, 0, 0);
        let g = voxel(0, 255, 0);
        let b = voxel(0, 0, 255);
        world.set_voxel(0, 0, 0, r);
        world.set_voxel(1, 0, 0, g);
        world.set_voxel(2, 0, 0, b);

        let sel = Selection::from_corners((0, 0, 0), (2, 0, 0));
        let changes = build_move_changes(&world, sel, (1, 0, 0));
        let cmd = Command::set_voxels(changes);
        cmd.execute(&mut world);

        // After move: 0 cleared, 1=r, 2=g, 3=b.
        assert!(world.get_voxel(0, 0, 0).is_air());
        assert_eq!(world.get_voxel(1, 0, 0), r);
        assert_eq!(world.get_voxel(2, 0, 0), g);
        assert_eq!(world.get_voxel(3, 0, 0), b);

        // Undo: world must be exactly as it was.
        cmd.undo(&mut world);
        assert_eq!(world.get_voxel(0, 0, 0), r);
        assert_eq!(world.get_voxel(1, 0, 0), g);
        assert_eq!(world.get_voxel(2, 0, 0), b);
        assert!(world.get_voxel(3, 0, 0).is_air());
    }

    #[test]
    fn move_drops_air_cells_in_selection() {
        // Selection has air gaps; only non-air voxels move.
        let mut world = World::new();
        let r = voxel(255, 0, 0);
        world.set_voxel(0, 0, 0, r);
        // (1,0,0) is air.
        world.set_voxel(2, 0, 0, r);

        let sel = Selection::from_corners((0, 0, 0), (2, 0, 0));
        let changes = build_move_changes(&world, sel, (10, 0, 0));

        // 2 source clears + 2 dest writes = 4 changes (no extra
        // entry for the air gap).
        assert_eq!(changes.len(), 4);
        let positions: std::collections::HashSet<_> =
            changes.iter().map(|c| c.pos).collect();
        assert!(!positions.contains(&(1, 0, 0))); // air gap untouched
        assert!(positions.contains(&(0, 0, 0)));
        assert!(positions.contains(&(2, 0, 0)));
        assert!(positions.contains(&(10, 0, 0)));
        assert!(positions.contains(&(12, 0, 0)));
    }

    #[test]
    fn cut_then_undo_restores_voxels() {
        // Cut should be a single Command — one Ctrl+Z brings back
        // every cleared voxel, not just half.
        let mut world = World::new();
        let mut history = crate::editor::CommandHistory::new(100);
        world.set_voxel(0, 0, 0, voxel(255, 0, 0));
        world.set_voxel(1, 0, 0, voxel(0, 255, 0));

        let sel = Selection::from_corners((0, 0, 0), (1, 0, 0));
        let _clipboard = copy_selection_to_clipboard(&world, sel);
        let changes = build_clear_changes(&world, sel);
        let cmd = Command::set_voxels(changes);
        history.execute(cmd, &mut world);

        // After cut: both cells empty.
        assert!(world.get_voxel(0, 0, 0).is_air());
        assert!(world.get_voxel(1, 0, 0).is_air());

        // One undo restores BOTH.
        history.undo(&mut world);
        assert_eq!(world.get_voxel(0, 0, 0), voxel(255, 0, 0));
        assert_eq!(world.get_voxel(1, 0, 0), voxel(0, 255, 0));
    }
}
