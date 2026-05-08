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
pub use tools::{
    compute_flood_fill_changes, eyedrop, flood_fill, flood_fill_multi, BrushTool, EditorTool,
    Tool, ToolContext,
};

use crate::core::Voxel;

/// Symmetric mirroring of brush effects across world-origin planes.
///
/// Each enabled axis mirrors the brush's writes across the corresponding
/// plane through the world origin (`x = 0` / `y = 0` / `z = 0`). With
/// multiple flags on, the brush replicates across every combination —
/// 1 plane → 2-fold, 2 planes → 4-fold, 3 planes → 8-fold (octahedral)
/// symmetry.
///
/// Mirroring is cell-aligned: cell `n` reflects to cell `-n - 1` so the
/// symmetry plane lies *between* cells rather than through one. Without
/// this offset, a cell at `n = 0` would mirror to itself and the brush
/// would have no visible mirror partner there.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct SymmetryAxes {
    pub x: bool,
    pub y: bool,
    pub z: bool,
}

impl SymmetryAxes {
    pub fn any(&self) -> bool {
        self.x || self.y || self.z
    }

    /// Number of total positions a single point expands into
    /// (1, 2, 4, or 8).
    pub fn count(&self) -> usize {
        1 << (self.x as u32 + self.y as u32 + self.z as u32)
    }

    /// Expand `pos` to every mirror combination. The first element is
    /// always `pos` itself; subsequent elements come from each enabled
    /// axis flip applied in order. Result length matches `count()`.
    pub fn mirror_positions(&self, pos: (i32, i32, i32)) -> Vec<(i32, i32, i32)> {
        let mut out = Vec::with_capacity(self.count());
        out.push(pos);
        if self.x {
            for i in 0..out.len() {
                let p = out[i];
                out.push((-p.0 - 1, p.1, p.2));
            }
        }
        if self.y {
            for i in 0..out.len() {
                let p = out[i];
                out.push((p.0, -p.1 - 1, p.2));
            }
        }
        if self.z {
            for i in 0..out.len() {
                let p = out[i];
                out.push((p.0, p.1, -p.2 - 1));
            }
        }
        out
    }
}

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
    /// Tool saved before Alt key was pressed (for temporary eyedropper)
    pub tool_before_alt: Option<Tool>,
    /// Active symmetry mirroring for brush writes (Place / Remove /
    /// Paint / Fill all honor it; Eyedropper doesn't write so it's
    /// exempt). Persists across sessions via prefs.
    pub symmetry: SymmetryAxes,
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
            tool_before_alt: None,
            symmetry: SymmetryAxes::default(),
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

#[cfg(test)]
mod symmetry_tests {
    use super::*;

    #[test]
    fn test_no_axes_returns_single_position() {
        let s = SymmetryAxes::default();
        assert_eq!(s.count(), 1);
        assert_eq!(s.mirror_positions((5, 7, 11)), vec![(5, 7, 11)]);
    }

    #[test]
    fn test_x_axis_doubles_position() {
        let s = SymmetryAxes { x: true, ..Default::default() };
        assert_eq!(s.count(), 2);
        assert_eq!(
            s.mirror_positions((5, 7, 11)),
            vec![(5, 7, 11), (-6, 7, 11)]
        );
    }

    #[test]
    fn test_all_axes_octuple() {
        let s = SymmetryAxes { x: true, y: true, z: true };
        assert_eq!(s.count(), 8);
        let result = s.mirror_positions((5, 7, 11));
        assert_eq!(result.len(), 8);
        // Every sign combination of mirrored coordinates is present.
        let set: std::collections::HashSet<_> = result.into_iter().collect();
        assert!(set.contains(&(5, 7, 11)));
        assert!(set.contains(&(-6, 7, 11)));
        assert!(set.contains(&(5, -8, 11)));
        assert!(set.contains(&(5, 7, -12)));
        assert!(set.contains(&(-6, -8, 11)));
        assert!(set.contains(&(-6, 7, -12)));
        assert!(set.contains(&(5, -8, -12)));
        assert!(set.contains(&(-6, -8, -12)));
    }

    #[test]
    fn test_mirror_at_axis_boundary_offsets_correctly() {
        // Cell at x=0 must mirror to x=-1 (not to itself), and the
        // pair must be a true reflection — x=0 mirrors to x=-1 and
        // back.
        let s = SymmetryAxes { x: true, ..Default::default() };
        assert_eq!(s.mirror_positions((0, 5, 5)), vec![(0, 5, 5), (-1, 5, 5)]);
        assert_eq!(s.mirror_positions((-1, 5, 5)), vec![(-1, 5, 5), (0, 5, 5)]);
    }

    #[test]
    fn test_count_matches_axis_combinations() {
        for x in [false, true] {
            for y in [false, true] {
                for z in [false, true] {
                    let s = SymmetryAxes { x, y, z };
                    let expected = 1 << (x as u32 + y as u32 + z as u32);
                    assert_eq!(s.count(), expected);
                    assert_eq!(s.mirror_positions((1, 2, 3)).len(), expected);
                }
            }
        }
    }

    #[test]
    fn test_any_reports_true_when_any_axis_on() {
        assert!(!SymmetryAxes::default().any());
        assert!(SymmetryAxes { x: true, ..Default::default() }.any());
        assert!(SymmetryAxes { y: true, ..Default::default() }.any());
        assert!(SymmetryAxes { z: true, ..Default::default() }.any());
    }
}
