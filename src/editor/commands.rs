//! Command pattern for undo/redo functionality.
//!
//! Each edit operation is encapsulated in a Command that knows how to
//! execute and reverse itself.

use crate::core::{Voxel, World};

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
}

/// Command history for undo/redo
pub struct CommandHistory {
    /// Stack of executed commands (for undo)
    undo_stack: Vec<Command>,
    /// Stack of undone commands (for redo)
    redo_stack: Vec<Command>,
    /// Maximum history size
    max_size: usize,
}

impl CommandHistory {
    /// Create a new command history
    pub fn new(max_size: usize) -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_size,
        }
    }

    /// Execute a command and add to history
    pub fn execute(&mut self, command: Command, world: &mut World) {
        // Skip no-op commands
        if command.is_noop() {
            return;
        }

        // Execute the command
        command.execute(world);

        // Add to undo stack
        self.undo_stack.push(command);

        // Clear redo stack (new action invalidates redo history)
        self.redo_stack.clear();

        // Trim history if too large
        while self.undo_stack.len() > self.max_size {
            self.undo_stack.remove(0);
        }
    }

    /// Execute a command and add to history, merging with last if similar
    pub fn execute_merge(&mut self, command: Command, world: &mut World, merge_window_ms: u128) {
        // For now, just execute normally
        // TODO: Implement merging for brush strokes
        let _ = merge_window_ms;
        self.execute(command, world);
    }

    /// Undo the last command
    pub fn undo(&mut self, world: &mut World) -> bool {
        if let Some(command) = self.undo_stack.pop() {
            command.undo(world);
            self.redo_stack.push(command);
            true
        } else {
            false
        }
    }

    /// Redo the last undone command
    pub fn redo(&mut self, world: &mut World) -> bool {
        if let Some(command) = self.redo_stack.pop() {
            command.execute(world);
            self.undo_stack.push(command);
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
}
