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

/// A reversible edit command.
///
/// One variant on purpose: every edit in the app — a single brush
/// click, a drag-painted stroke, a filled region, a generator patch,
/// a selection rotate — is a batch of per-voxel changes, and routing
/// all of them through one shape is what lets `try_merge_with`
/// collapse a stroke into a single undo entry. There used to be
/// `SetVoxel` and `FillRegion` variants as well; both were unused,
/// and either one would have quietly opted its caller out of stroke
/// merging (`FillRegion` also recorded a whole AABB of old voxels
/// where the changed cells alone would do).
#[derive(Debug, Clone)]
pub enum Command {
    /// Set multiple voxels (batch operation)
    SetVoxels {
        changes: Vec<VoxelChange>,
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
    /// Create a batch voxel command
    pub fn set_voxels(changes: Vec<VoxelChange>) -> Self {
        Command::SetVoxels { changes }
    }

    /// Execute the command (apply changes)
    pub fn execute(&self, world: &mut World) {
        let Command::SetVoxels { changes } = self;
        for change in changes {
            world.set_voxel(change.pos.0, change.pos.1, change.pos.2, change.new_voxel);
        }
    }

    /// Reverse the command (undo changes)
    pub fn undo(&self, world: &mut World) {
        let Command::SetVoxels { changes } = self;
        // Reverse order: if two changes ever share a position (a
        // generator patch that wrote a cell twice, or a symmetry
        // brush stroke mirroring a cell onto itself), the EARLIEST
        // record holds the true pre-command value and must be
        // restored LAST to win. Forward replay would leave a later
        // record's old_voxel. (Generator patches are now de-duped
        // upstream, so same-position olds are equal anyway — this
        // keeps undo exact for any caller regardless.)
        for change in changes.iter().rev() {
            world.set_voxel(change.pos.0, change.pos.1, change.pos.2, change.old_voxel);
        }
    }

    /// Check if command would actually change anything
    pub fn is_noop(&self) -> bool {
        let Command::SetVoxels { changes } = self;
        changes.is_empty() || changes.iter().all(|c| c.old_voxel == c.new_voxel)
    }

    /// Absorb `other` into `self` in place.
    ///
    /// For each position, the earliest `old_voxel` is preserved (so
    /// undo restores the pre-stroke state) and the latest `new_voxel`
    /// is taken (so the stroke ends in its final visible state). Any
    /// two commands merge, `Command` having a single variant; add a
    /// second one and every destructuring here stops compiling, which
    /// is the point at which a "these two don't merge" path would have
    /// to come back.
    pub fn merge_with(&mut self, other: Command) {
        let Command::SetVoxels { changes: other_changes } = other;
        let Command::SetVoxels { changes: self_changes } = self;

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
    /// Bumped by every operation that changes the world through this
    /// history. See [`CommandHistory::generation`].
    generation: u64,
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
            generation: 0,
        }
    }

    /// A counter that moves whenever this history changed the world,
    /// and never moves back.
    ///
    /// The stack depths look like they say the same thing and don't:
    /// `(undo, redo)` returns to a pair it already held in at least
    /// three ordinary ways — undo then draw (the redo stack is cleared
    /// by the new command), draw with the undo stack already at
    /// `max_size` (the push trims the oldest), and continuing a stroke
    /// (a merge folds into the entry already on top). Each of those
    /// leaves a *different world* behind the same numbers.
    ///
    /// Anything holding a change list computed against an earlier world
    /// — the editor's agent bridge parks one while a human decides —
    /// has to be able to tell that the ground moved, because its
    /// `old_voxel`s describe the world as it *was*, and committing them
    /// afterwards makes undo restore a state that never existed.
    pub fn generation(&self) -> u64 {
        self.generation
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
                prev.merge_with(command);
                // A merge still counts as new activity for redo
                // invalidation and for refreshing the merge window.
                self.redo_stack.clear();
                self.last_push_at = Some(Instant::now());
                // The world moved even though no stack did: the entry
                // on top absorbed the change.
                self.generation += 1;
                return;
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
        self.generation += 1;
    }

    /// Undo the last command
    pub fn undo(&mut self, world: &mut World) -> bool {
        if let Some(command) = self.undo_stack.pop_back() {
            command.undo(world);
            self.redo_stack.push_back(command);
            // Any active stroke is no longer at the top of undo.
            self.stroke_open = false;
            self.generation += 1;
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
            self.generation += 1;
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
        // Nothing was undone, but everything anyone held about this
        // history is now wrong — clearing accompanies throwing the
        // world away.
        self.generation += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One-cell batch — what a single brush click produces.
    fn set_one(world: &World, pos: (i32, i32, i32), new_voxel: Voxel) -> Command {
        Command::set_voxels(vec![VoxelChange {
            pos,
            old_voxel: world.get_voxel(pos.0, pos.1, pos.2),
            new_voxel,
        }])
    }

    /// The three ways `(undo_count, redo_count)` comes back to a pair
    /// it already held while the world underneath it changed. Each was
    /// a way for the agent bridge to accept a batch built against a
    /// world that is gone; the generation counter is what tells them
    /// apart.
    #[test]
    fn every_edit_moves_the_generation_even_when_the_depths_do_not() {
        let mut world = World::new();
        let solid = Voxel::from_rgb(200, 100, 50);

        // (a) undo, then draw: the new push clears the redo stack, so
        // the pair walks back to exactly where it started.
        let mut history = CommandHistory::new(100);
        for x in 0..3 {
            history.execute(set_one(&world, (x, 0, 0), solid), &mut world);
        }
        let depths = (history.undo_count(), history.redo_count());
        let mark = history.generation();
        history.undo(&mut world);
        history.execute(set_one(&world, (50, 0, 0), solid), &mut world);
        assert_eq!(depths, (history.undo_count(), history.redo_count()));
        assert_ne!(mark, history.generation());

        // (b) the undo stack is already full: the push trims the oldest
        // entry, so the depth doesn't move either.
        let mut history = CommandHistory::new(4);
        for x in 0..4 {
            history.execute(set_one(&world, (x, 1, 0), solid), &mut world);
        }
        let depths = (history.undo_count(), history.redo_count());
        let mark = history.generation();
        history.execute(set_one(&world, (60, 1, 0), solid), &mut world);
        assert_eq!(depths, (4, 0));
        assert_eq!(depths, (history.undo_count(), history.redo_count()));
        assert_ne!(mark, history.generation());

        // (c) a stroke in progress: the merge folds into the entry
        // already on top, and nothing is pushed at all.
        let mut history = CommandHistory::new(100);
        let window = Duration::from_secs(60);
        history.execute_merge(set_one(&world, (0, 2, 0), solid), &mut world, window);
        let depths = (history.undo_count(), history.redo_count());
        let mark = history.generation();
        history.execute_merge(set_one(&world, (1, 2, 0), solid), &mut world, window);
        assert_eq!(depths, (history.undo_count(), history.redo_count()));
        assert_ne!(mark, history.generation());
    }

    /// It only ever counts up, so a mark taken earlier can never be
    /// matched again by later activity — including the `clear` that
    /// accompanies throwing the world away, which lands both depths
    /// back on `(0, 0)`.
    #[test]
    fn the_generation_never_returns_to_a_value_it_has_left() {
        let mut world = World::new();
        let solid = Voxel::from_rgb(200, 100, 50);
        let mut history = CommandHistory::new(100);
        let mut seen = vec![history.generation()];

        history.execute(set_one(&world, (0, 0, 0), solid), &mut world);
        seen.push(history.generation());
        history.undo(&mut world);
        seen.push(history.generation());
        history.redo(&mut world);
        seen.push(history.generation());
        history.clear();
        seen.push(history.generation());

        assert_eq!((history.undo_count(), history.redo_count()), (0, 0));
        let mut sorted = seen.clone();
        sorted.dedup();
        assert_eq!(sorted.len(), seen.len(), "a generation repeated: {seen:?}");
        assert!(seen.windows(2).all(|w| w[0] < w[1]), "not monotonic: {seen:?}");
    }

    #[test]
    fn test_undo_redo() {
        let mut world = World::new();
        let mut history = CommandHistory::new(100);

        // Set a voxel
        let cmd = set_one(&world, (0, 0, 0), Voxel::from_rgb(255, 0, 0));
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
        let cmd = set_one(&world, (0, 0, 0), Voxel::AIR);
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
        a.merge_with(b);
        let Command::SetVoxels { changes } = &a;
        assert_eq!(changes.len(), 2);
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
        a.merge_with(b);
        let Command::SetVoxels { changes } = &a;
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].old_voxel, Voxel::AIR);
        assert_eq!(changes[0].new_voxel, voxel(2));
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
