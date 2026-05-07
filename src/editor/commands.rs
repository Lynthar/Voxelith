//! Command pattern for undo/redo functionality.
//!
//! Each edit operation is encapsulated in a Command that knows how to
//! execute and reverse itself.
//!
//! Brush strokes are aggregated via [`CommandHistory::execute_merge`]:
//! within a configurable time window, consecutive `SetVoxels` commands
//! are merged into the most recent undo entry instead of being pushed
//! as separate units. The merge keeps the *earliest* `old_voxel` per
//! position and the *latest* `new_voxel`, so a single Ctrl+Z reverses
//! the whole stroke even if the user painted the same cell multiple
//! times. Merging requires `stroke_open` (set by `execute_merge`,
//! cleared by `execute` / `end_stroke` / `undo` / `redo`).

use crate::core::{Voxel, World};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// A reversible edit command
#[derive(Debug, Clone)]
pub enum Command {
    /// Set a single voxel
    SetVoxel {
        pos: (i32, i32, i32),
        old_voxel: Voxel,
        new_voxel: Voxel,
    },
    /// Set multiple voxels (batch operation)
    SetVoxels {
        changes: Vec<VoxelChange>,
    },
    /// Fill a region
    FillRegion {
        min: (i32, i32, i32),
        max: (i32, i32, i32),
        old_voxels: Vec<((i32, i32, i32), Voxel)>,
        new_voxel: Voxel,
    },
}

/// Single voxel change record
#[derive(Debug, Clone)]
pub struct VoxelChange {
    pub pos: (i32, i32, i32),
    pub old_voxel: Voxel,
    pub new_voxel: Voxel,
}

impl Command {
    /// Create a set voxel command
    pub fn set_voxel(world: &World, pos: (i32, i32, i32), new_voxel: Voxel) -> Self {
        let old_voxel = world.get_voxel(pos.0, pos.1, pos.2);
        Command::SetVoxel {
            pos,
            old_voxel,
            new_voxel,
        }
    }

    /// Create a batch voxel command
    pub fn set_voxels(changes: Vec<VoxelChange>) -> Self {
        Command::SetVoxels { changes }
    }

    /// Create a fill region command
    pub fn fill_region(world: &World, min: (i32, i32, i32), max: (i32, i32, i32), new_voxel: Voxel) -> Self {
        let mut old_voxels = Vec::new();
        for z in min.2..=max.2 {
            for y in min.1..=max.1 {
                for x in min.0..=max.0 {
                    let old = world.get_voxel(x, y, z);
                    old_voxels.push(((x, y, z), old));
                }
            }
        }
        Command::FillRegion {
            min,
            max,
            old_voxels,
            new_voxel,
        }
    }

    /// Execute the command (apply changes)
    pub fn execute(&self, world: &mut World) {
        match self {
            Command::SetVoxel { pos, new_voxel, .. } => {
                world.set_voxel(pos.0, pos.1, pos.2, *new_voxel);
            }
            Command::SetVoxels { changes } => {
                for change in changes {
                    world.set_voxel(change.pos.0, change.pos.1, change.pos.2, change.new_voxel);
                }
            }
            Command::FillRegion { min, max, new_voxel, .. } => {
                world.fill_region(*min, *max, *new_voxel);
            }
        }
    }

    /// Reverse the command (undo changes)
    pub fn undo(&self, world: &mut World) {
        match self {
            Command::SetVoxel { pos, old_voxel, .. } => {
                world.set_voxel(pos.0, pos.1, pos.2, *old_voxel);
            }
            Command::SetVoxels { changes } => {
                for change in changes {
                    world.set_voxel(change.pos.0, change.pos.1, change.pos.2, change.old_voxel);
                }
            }
            Command::FillRegion { old_voxels, .. } => {
                for (pos, old_voxel) in old_voxels {
                    world.set_voxel(pos.0, pos.1, pos.2, *old_voxel);
                }
            }
        }
    }

    /// Check if command would actually change anything
    pub fn is_noop(&self) -> bool {
        match self {
            Command::SetVoxel { old_voxel, new_voxel, .. } => old_voxel == new_voxel,
            Command::SetVoxels { changes } => {
                changes.is_empty() || changes.iter().all(|c| c.old_voxel == c.new_voxel)
            }
            Command::FillRegion { old_voxels, new_voxel, .. } => {
                old_voxels.iter().all(|(_, old)| old == new_voxel)
            }
        }
    }

    /// Try to absorb `other` into `self` in place.
    ///
    /// Only `SetVoxels` + `SetVoxels` is mergeable. For each position,
    /// the earliest `old_voxel` is preserved (so undo restores the
    /// pre-stroke state) and the latest `new_voxel` is taken (so the
    /// stroke ends in its final visible state). If the merge isn't
    /// possible the original `other` is returned unchanged in `Err`.
    pub fn try_merge_with(&mut self, other: Command) -> Result<(), Command> {
        if !matches!(
            (&*self, &other),
            (Command::SetVoxels { .. }, Command::SetVoxels { .. })
        ) {
            return Err(other);
        }

        let other_changes = match other {
            Command::SetVoxels { changes } => changes,
            _ => unreachable!(),
        };
        let self_changes = match self {
            Command::SetVoxels { changes } => changes,
            _ => unreachable!(),
        };

        // Build pos -> index into self_changes for in-place updates.
        let mut by_pos: HashMap<(i32, i32, i32), usize> =
            HashMap::with_capacity(self_changes.len() + other_changes.len());
        for (i, c) in self_changes.iter().enumerate() {
            by_pos.insert(c.pos, i);
        }
        for change in other_changes {
            if let Some(&idx) = by_pos.get(&change.pos) {
                // Preserve self_changes[idx].old_voxel; refresh new_voxel.
                self_changes[idx].new_voxel = change.new_voxel;
            } else {
                by_pos.insert(change.pos, self_changes.len());
                self_changes.push(change);
            }
        }
        Ok(())
    }
}

/// Command history for undo/redo with brush-stroke merging.
pub struct CommandHistory {
    /// Stack of executed commands (for undo)
    undo_stack: VecDeque<Command>,
    /// Stack of undone commands (for redo)
    redo_stack: VecDeque<Command>,
    /// Maximum history size
    max_size: usize,
    /// When the most recent push or merge happened. Drives the
    /// stroke-merge time window inside `execute_merge`.
    last_push_at: Option<Instant>,
    /// True between `execute_merge` (which opens a stroke) and the
    /// next `end_stroke` / `execute` / `undo` / `redo` (which closes
    /// it). Required for `execute_merge` to merge instead of push.
    stroke_open: bool,
}

impl CommandHistory {
    /// Create a new command history
    pub fn new(max_size: usize) -> Self {
        Self {
            undo_stack: VecDeque::new(),
            redo_stack: VecDeque::new(),
            max_size,
            last_push_at: None,
            stroke_open: false,
        }
    }

    /// Execute a command and push it as a fresh undo entry.
    /// Use this for one-shot operations (single click, fill, paste).
    pub fn execute(&mut self, command: Command, world: &mut World) {
        if command.is_noop() {
            return;
        }
        command.execute(world);
        self.push_new(command);
        // Single-shot: don't let the next execute_merge fold into us.
        self.stroke_open = false;
    }

    /// Execute a command, merging into the most recent undo entry if
    /// it's part of an open stroke and within `merge_window`. Falls
    /// back to a fresh push otherwise. Use this for brush-style tools.
    pub fn execute_merge(
        &mut self,
        command: Command,
        world: &mut World,
        merge_window: Duration,
    ) {
        if command.is_noop() {
            return;
        }
        command.execute(world);

        let in_window = self
            .last_push_at
            .map_or(false, |t| t.elapsed() < merge_window);

        if self.stroke_open && in_window {
            if let Some(prev) = self.undo_stack.back_mut() {
                match prev.try_merge_with(command) {
                    Ok(()) => {
                        // Successful merge: still considered new activity
                        // for redo invalidation and window refresh.
                        self.redo_stack.clear();
                        self.last_push_at = Some(Instant::now());
                        return;
                    }
                    Err(returned) => {
                        // Couldn't merge — push as a fresh entry but
                        // keep the stroke open (next call may merge).
                        self.push_new(returned);
                        self.stroke_open = true;
                        return;
                    }
                }
            }
        }

        self.push_new(command);
        self.stroke_open = true;
    }

    /// Force-finalize the current stroke. Subsequent `execute_merge`
    /// calls open a new stroke instead of folding into the previous
    /// command. Wire this to mouse-up.
    pub fn end_stroke(&mut self) {
        self.stroke_open = false;
    }

    /// Internal: push a fully-prepared command onto the undo stack,
    /// invalidate redo, trim, and stamp the activity timestamp.
    fn push_new(&mut self, command: Command) {
        self.undo_stack.push_back(command);
        self.redo_stack.clear();
        while self.undo_stack.len() > self.max_size {
            self.undo_stack.pop_front();
        }
        self.last_push_at = Some(Instant::now());
    }

    /// Undo the last command
    pub fn undo(&mut self, world: &mut World) -> bool {
        if let Some(command) = self.undo_stack.pop_back() {
            command.undo(world);
            self.redo_stack.push_back(command);
            // Any active stroke is no longer at the top of undo.
            self.stroke_open = false;
            true
        } else {
            false
        }
    }

    /// Redo the last undone command
    pub fn redo(&mut self, world: &mut World) -> bool {
        if let Some(command) = self.redo_stack.pop_back() {
            command.execute(world);
            self.undo_stack.push_back(command);
            self.stroke_open = false;
            true
        } else {
            false
        }
    }

    /// Check if undo is available
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Check if redo is available
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Get number of commands in undo history
    pub fn undo_count(&self) -> usize {
        self.undo_stack.len()
    }

    /// Get number of commands in redo history
    pub fn redo_count(&self) -> usize {
        self.redo_stack.len()
    }

    /// Clear all history
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.last_push_at = None;
        self.stroke_open = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_undo_redo() {
        let mut world = World::new();
        let mut history = CommandHistory::new(100);

        // Set a voxel
        let cmd = Command::set_voxel(&world, (0, 0, 0), Voxel::from_rgb(255, 0, 0));
        history.execute(cmd, &mut world);

        assert!(!world.get_voxel(0, 0, 0).is_air());

        // Undo
        history.undo(&mut world);
        assert!(world.get_voxel(0, 0, 0).is_air());

        // Redo
        history.redo(&mut world);
        assert!(!world.get_voxel(0, 0, 0).is_air());
    }

    #[test]
    fn test_noop_command() {
        let world = World::new();
        let cmd = Command::set_voxel(&world, (0, 0, 0), Voxel::AIR);
        assert!(cmd.is_noop());
    }

    fn voxel(r: u8) -> Voxel {
        Voxel::from_rgb(r, 0, 0)
    }

    #[test]
    fn test_try_merge_disjoint_positions() {
        let mut a = Command::SetVoxels {
            changes: vec![VoxelChange {
                pos: (0, 0, 0),
                old_voxel: Voxel::AIR,
                new_voxel: voxel(1),
            }],
        };
        let b = Command::SetVoxels {
            changes: vec![VoxelChange {
                pos: (1, 0, 0),
                old_voxel: Voxel::AIR,
                new_voxel: voxel(2),
            }],
        };
        a.try_merge_with(b).unwrap();
        if let Command::SetVoxels { changes } = &a {
            assert_eq!(changes.len(), 2);
        } else {
            panic!("a should still be SetVoxels");
        }
    }

    #[test]
    fn test_try_merge_overlapping_keeps_earliest_old() {
        // Same position painted twice. Old voxel must come from the
        // first stroke segment so a single undo restores the
        // pre-stroke state.
        let mut a = Command::SetVoxels {
            changes: vec![VoxelChange {
                pos: (0, 0, 0),
                old_voxel: Voxel::AIR,
                new_voxel: voxel(1),
            }],
        };
        let b = Command::SetVoxels {
            changes: vec![VoxelChange {
                pos: (0, 0, 0),
                old_voxel: voxel(1),
                new_voxel: voxel(2),
            }],
        };
        a.try_merge_with(b).unwrap();
        if let Command::SetVoxels { changes } = &a {
            assert_eq!(changes.len(), 1);
            assert_eq!(changes[0].old_voxel, Voxel::AIR);
            assert_eq!(changes[0].new_voxel, voxel(2));
        } else {
            panic!("a should still be SetVoxels");
        }
    }

    #[test]
    fn test_try_merge_incompatible_kinds() {
        let world = World::new();
        let mut a = Command::SetVoxel {
            pos: (0, 0, 0),
            old_voxel: Voxel::AIR,
            new_voxel: voxel(1),
        };
        let b = Command::set_voxels(vec![VoxelChange {
            pos: (1, 0, 0),
            old_voxel: Voxel::AIR,
            new_voxel: voxel(2),
        }]);
        let _ = world; // keep an unused-binding-free pattern
        // SetVoxel cannot merge with SetVoxels.
        assert!(a.try_merge_with(b).is_err());
    }

    #[test]
    fn test_execute_merge_combines_within_window() {
        let mut world = World::new();
        let mut history = CommandHistory::new(100);
        let win = Duration::from_millis(500);

        let cmd1 = Command::set_voxels(vec![VoxelChange {
            pos: (0, 0, 0),
            old_voxel: Voxel::AIR,
            new_voxel: voxel(1),
        }]);
        history.execute_merge(cmd1, &mut world, win);
        assert_eq!(history.undo_count(), 1);

        let cmd2 = Command::set_voxels(vec![VoxelChange {
            pos: (1, 0, 0),
            old_voxel: Voxel::AIR,
            new_voxel: voxel(2),
        }]);
        history.execute_merge(cmd2, &mut world, win);
        // Merged into the same undo entry.
        assert_eq!(history.undo_count(), 1);

        // Single undo restores both writes.
        history.undo(&mut world);
        assert!(world.get_voxel(0, 0, 0).is_air());
        assert!(world.get_voxel(1, 0, 0).is_air());
    }

    #[test]
    fn test_execute_merge_after_end_stroke_pushes_new() {
        let mut world = World::new();
        let mut history = CommandHistory::new(100);
        let win = Duration::from_millis(500);

        let cmd1 = Command::set_voxels(vec![VoxelChange {
            pos: (0, 0, 0),
            old_voxel: Voxel::AIR,
            new_voxel: voxel(1),
        }]);
        history.execute_merge(cmd1, &mut world, win);
        history.end_stroke();

        let cmd2 = Command::set_voxels(vec![VoxelChange {
            pos: (1, 0, 0),
            old_voxel: Voxel::AIR,
            new_voxel: voxel(2),
        }]);
        history.execute_merge(cmd2, &mut world, win);
        // Two separate strokes -> two undo entries.
        assert_eq!(history.undo_count(), 2);
    }

    #[test]
    fn test_execute_merge_zero_window_never_merges() {
        let mut world = World::new();
        let mut history = CommandHistory::new(100);
        let win = Duration::ZERO;

        let cmd1 = Command::set_voxels(vec![VoxelChange {
            pos: (0, 0, 0),
            old_voxel: Voxel::AIR,
            new_voxel: voxel(1),
        }]);
        history.execute_merge(cmd1, &mut world, win);
        let cmd2 = Command::set_voxels(vec![VoxelChange {
            pos: (1, 0, 0),
            old_voxel: Voxel::AIR,
            new_voxel: voxel(2),
        }]);
        history.execute_merge(cmd2, &mut world, win);
        assert_eq!(history.undo_count(), 2);
    }

    #[test]
    fn test_execute_after_merge_closes_stroke() {
        // A one-shot execute() in the middle should not be foldable
        // into by a later execute_merge — execute closes the stroke.
        let mut world = World::new();
        let mut history = CommandHistory::new(100);
        let win = Duration::from_millis(500);

        let cmd1 = Command::set_voxels(vec![VoxelChange {
            pos: (0, 0, 0),
            old_voxel: Voxel::AIR,
            new_voxel: voxel(1),
        }]);
        history.execute_merge(cmd1, &mut world, win);

        // One-shot fill / paste.
        let cmd2 = Command::set_voxels(vec![VoxelChange {
            pos: (5, 5, 5),
            old_voxel: Voxel::AIR,
            new_voxel: voxel(3),
        }]);
        history.execute(cmd2, &mut world);

        // Now another brush — should NOT merge into cmd2.
        let cmd3 = Command::set_voxels(vec![VoxelChange {
            pos: (1, 0, 0),
            old_voxel: Voxel::AIR,
            new_voxel: voxel(2),
        }]);
        history.execute_merge(cmd3, &mut world, win);
        assert_eq!(history.undo_count(), 3);
    }
}
