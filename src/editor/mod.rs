//! Editor functionality: tools, commands, undo/redo.
//!
//! This module contains:
//! - Ray casting for voxel picking
//! - Tool implementations (place, remove, paint)
//! - Command pattern for undo/redo
//! - History management

mod commands;
mod raycast;
mod tools;

pub use commands::{Command, CommandHistory, VoxelChange};
pub use raycast::{Ray, RaycastHit, VoxelRaycast};
pub use tools::{eyedrop, flood_fill, BrushTool, EditorTool, Tool, ToolContext};

use crate::core::Voxel;

/// Editor state containing tools, history, and current settings
pub struct Editor {
    /// Current active tool
    pub current_tool: Tool,
    /// Command history for undo/redo
    pub history: CommandHistory,
    /// Current brush color
    pub brush_color: Voxel,
    /// Brush size (radius)
    pub brush_size: u8,
    /// Currently hovered voxel (if any)
    pub hovered_voxel: Option<RaycastHit>,
    /// Color palette
    pub palette: Vec<Voxel>,
}

impl Default for Editor {
    fn default() -> Self {
        Self::new()
    }
}

impl Editor {
    pub fn new() -> Self {
        Self {
            current_tool: Tool::Place,
            history: CommandHistory::new(100),
            brush_color: Voxel::from_rgb(200, 100, 50),
            brush_size: 1,
            hovered_voxel: None,
            palette: Self::default_palette(),
        }
    }

    /// Create default color palette
    fn default_palette() -> Vec<Voxel> {
        vec![
            // Grayscale
            Voxel::from_rgb(255, 255, 255), // White
            Voxel::from_rgb(200, 200, 200), // Light gray
            Voxel::from_rgb(150, 150, 150), // Gray
            Voxel::from_rgb(100, 100, 100), // Dark gray
            Voxel::from_rgb(50, 50, 50),    // Charcoal
            Voxel::from_rgb(0, 0, 0),       // Black
            // Primary colors
            Voxel::from_rgb(255, 0, 0),   // Red
            Voxel::from_rgb(0, 255, 0),   // Green
            Voxel::from_rgb(0, 0, 255),   // Blue
            Voxel::from_rgb(255, 255, 0), // Yellow
            Voxel::from_rgb(255, 0, 255), // Magenta
            Voxel::from_rgb(0, 255, 255), // Cyan
            // Earth tones
            Voxel::from_rgb(139, 90, 43),  // Brown
            Voxel::from_rgb(76, 153, 0),   // Grass green
            Voxel::from_rgb(194, 178, 128), // Sand
            Voxel::from_rgb(128, 128, 128), // Stone
            // Vivid colors
            Voxel::from_rgb(255, 128, 0),  // Orange
            Voxel::from_rgb(128, 0, 255),  // Purple
            Voxel::from_rgb(255, 192, 203), // Pink
            Voxel::from_rgb(0, 128, 128),  // Teal
        ]
    }

    /// Set current tool
    pub fn set_tool(&mut self, tool: Tool) {
        self.current_tool = tool;
    }

    /// Set brush color from palette index
    pub fn set_palette_color(&mut self, index: usize) {
        if index < self.palette.len() {
            self.brush_color = self.palette[index];
        }
    }

    /// Undo last action
    pub fn undo(&mut self, world: &mut crate::core::World) {
        self.history.undo(world);
    }

    /// Redo last undone action
    pub fn redo(&mut self, world: &mut crate::core::World) {
        self.history.redo(world);
    }

    /// Check if undo is available
    pub fn can_undo(&self) -> bool {
        self.history.can_undo()
    }

    /// Check if redo is available
    pub fn can_redo(&self) -> bool {
        self.history.can_redo()
    }
}
