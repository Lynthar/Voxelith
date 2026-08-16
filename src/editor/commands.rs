//! Undo/redo. Every edit is one `Command` that knows how to reverse
//! itself; brush strokes collapse into a single entry through
//! [`CommandHistory::execute_merge`].

use crate::core::{Voxel, World};
use crate::procgen::PipelineGraph;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// A pipeline-graph replacement riding with a command: the graph before
/// it and after it. The graph is document data like the voxels, so an
/// undo has to step both or it restores a state that never existed.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphTransition {
    pub before: PipelineGraph,
    pub after: PipelineGraph,
}

/// A reversible edit command. One variant on purpose: every edit is a
/// batch of per-voxel changes, and routing them all through one shape
/// is what lets a stroke collapse into a single undo entry.
#[derive(Debug, Clone)]
pub enum Command {
    /// Set multiple voxels (batch operation)
    SetVoxels {
        changes: Vec<VoxelChange>,
        /// A graph replacement riding with the batch (an agent's
        /// `graph` / `graph_edit` op). `None` for everything a human
        /// draws; boxed so the common case costs one pointer.
        graph: Option<Box<GraphTransition>>,
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
        Command::SetVoxels {
            changes,
            graph: None,
        }
    }

    /// A batch that also replaces the document's pipeline graph. Commit
    /// through [`CommandHistory::execute_with_graph`], never plain
    /// `execute` — that path has no graph to apply the rider to.
    pub fn set_voxels_with_graph(
        changes: Vec<VoxelChange>,
        graph: Option<GraphTransition>,
    ) -> Self {
        Command::SetVoxels {
            changes,
            graph: graph.map(Box::new),
        }
    }

    /// The graph transition riding with this command, if any.
    pub fn graph_rider(&self) -> Option<&GraphTransition> {
        let Command::SetVoxels { graph, .. } = self;
        graph.as_deref()
    }

    /// Execute the command (apply changes)
    pub fn execute(&self, world: &mut World) {
        let Command::SetVoxels { changes, .. } = self;
        for change in changes {
            world.set_voxel(change.pos.0, change.pos.1, change.pos.2, change.new_voxel);
        }
    }

    /// Reverse the command (undo changes)
    pub fn undo(&self, world: &mut World) {
        let Command::SetVoxels { changes, .. } = self;
        // Reverse order: when two changes share a position, the
        // earliest record holds the true pre-command value and has to
        // be restored last to win.
        for change in changes.iter().rev() {
            world.set_voxel(change.pos.0, change.pos.1, change.pos.2, change.old_voxel);
        }
    }

    /// Check if command would actually change anything
    pub fn is_noop(&self) -> bool {
        let Command::SetVoxels { changes, graph } = self;
        let voxels_noop = changes.is_empty() || changes.iter().all(|c| c.old_voxel == c.new_voxel);
        let graph_noop = graph.as_ref().is_none_or(|g| g.before == g.after);
        voxels_noop && graph_noop
    }

    /// Absorb `other` into `self`: per position the earliest
    /// `old_voxel` is kept and the latest `new_voxel` taken. The graph
    /// rider follows the same rule — earliest `before`, latest `after`.
    pub fn merge_with(&mut self, other: Command) {
        let Command::SetVoxels {
            changes: other_changes,
            graph: other_graph,
        } = other;
        let Command::SetVoxels {
            changes: self_changes,
            graph: self_graph,
        } = self;

        match (self_graph.as_mut(), other_graph) {
            (Some(mine), Some(theirs)) => mine.after = theirs.after,
            (None, Some(theirs)) => *self_graph = Some(theirs),
            (_, None) => {}
        }

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

    /// A counter that moves whenever this history changed the world and
    /// never moves back. The `(undo, redo)` depths can't stand in: that
    /// pair returns to values it has already held.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Execute a command and push it as a fresh undo entry — one-shot
    /// operations. Voxel-only: a graph rider must go through
    /// [`execute_with_graph`](Self::execute_with_graph).
    pub fn execute(&mut self, command: Command, world: &mut World) {
        debug_assert!(
            command.graph_rider().is_none(),
            "graph-carrying commands go through execute_with_graph"
        );
        if command.is_noop() {
            return;
        }
        command.execute(world);
        self.push_new(command);
        // Single-shot: don't let the next execute_merge fold into us.
        self.stroke_open = false;
    }

    /// Execute a command that may also replace the document's pipeline
    /// graph, as one undo entry. The agent commit paths use this; a
    /// plain voxel command works here too (`graph` is then untouched).
    pub fn execute_with_graph(
        &mut self,
        command: Command,
        world: &mut World,
        graph: &mut PipelineGraph,
    ) {
        if command.is_noop() {
            return;
        }
        command.execute(world);
        if let Some(t) = command.graph_rider() {
            *graph = t.after.clone();
        }
        self.push_new(command);
        self.stroke_open = false;
    }

    /// Execute a command, merging into the most recent undo entry when
    /// it belongs to an open stroke within `merge_window`. Voxel-only,
    /// like [`execute`](Self::execute).
    pub fn execute_merge(&mut self, command: Command, world: &mut World, merge_window: Duration) {
        debug_assert!(
            command.graph_rider().is_none(),
            "graph-carrying commands go through execute_with_graph"
        );
        if command.is_noop() {
            return;
        }
        command.execute(world);

        let in_window = self
            .last_push_at
            .is_some_and(|t| t.elapsed() < merge_window);

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

    /// Undo the last command. Takes the pipeline graph alongside the
    /// world because any entry may carry a [`GraphTransition`], and
    /// nothing else holds a copy of the graph it replaced.
    pub fn undo(&mut self, world: &mut World, graph: &mut PipelineGraph) -> bool {
        if let Some(command) = self.undo_stack.pop_back() {
            command.undo(world);
            if let Some(t) = command.graph_rider() {
                *graph = t.before.clone();
            }
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
    pub fn redo(&mut self, world: &mut World, graph: &mut PipelineGraph) -> bool {
        if let Some(command) = self.redo_stack.pop_back() {
            command.execute(world);
            if let Some(t) = command.graph_rider() {
                *graph = t.after.clone();
            }
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

    /// The three ways `(undo_count, redo_count)` returns to a pair it
    /// already held while the world underneath changed. The generation
    /// counter is what tells them apart.
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
        history.undo(&mut world, &mut PipelineGraph::default());
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

    /// It only counts up, so an earlier mark can never be matched by
    /// later activity — including the `clear` that lands both depths
    /// back on `(0, 0)`.
    #[test]
    fn the_generation_never_returns_to_a_value_it_has_left() {
        let mut world = World::new();
        let solid = Voxel::from_rgb(200, 100, 50);
        let mut history = CommandHistory::new(100);
        let mut seen = vec![history.generation()];

        history.execute(set_one(&world, (0, 0, 0), solid), &mut world);
        seen.push(history.generation());
        history.undo(&mut world, &mut PipelineGraph::default());
        seen.push(history.generation());
        history.redo(&mut world, &mut PipelineGraph::default());
        seen.push(history.generation());
        history.clear();
        seen.push(history.generation());

        assert_eq!((history.undo_count(), history.redo_count()), (0, 0));
        let mut sorted = seen.clone();
        sorted.dedup();
        assert_eq!(sorted.len(), seen.len(), "a generation repeated: {seen:?}");
        assert!(
            seen.windows(2).all(|w| w[0] < w[1]),
            "not monotonic: {seen:?}"
        );
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
        history.undo(&mut world, &mut PipelineGraph::default());
        assert!(world.get_voxel(0, 0, 0).is_air());

        // Redo
        history.redo(&mut world, &mut PipelineGraph::default());
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
        let mut a = Command::set_voxels(vec![VoxelChange {
            pos: (0, 0, 0),
            old_voxel: Voxel::AIR,
            new_voxel: voxel(1),
        }]);
        let b = Command::set_voxels(vec![VoxelChange {
            pos: (1, 0, 0),
            old_voxel: Voxel::AIR,
            new_voxel: voxel(2),
        }]);
        a.merge_with(b);
        let Command::SetVoxels { changes, .. } = &a;
        assert_eq!(changes.len(), 2);
    }

    #[test]
    fn test_try_merge_overlapping_keeps_earliest_old() {
        // Same position painted twice. Old voxel must come from the
        // first stroke segment so a single undo restores the
        // pre-stroke state.
        let mut a = Command::set_voxels(vec![VoxelChange {
            pos: (0, 0, 0),
            old_voxel: Voxel::AIR,
            new_voxel: voxel(1),
        }]);
        let b = Command::set_voxels(vec![VoxelChange {
            pos: (0, 0, 0),
            old_voxel: voxel(1),
            new_voxel: voxel(2),
        }]);
        a.merge_with(b);
        let Command::SetVoxels { changes, .. } = &a;
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
        history.undo(&mut world, &mut PipelineGraph::default());
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
