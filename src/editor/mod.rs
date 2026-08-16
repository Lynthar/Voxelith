//! Editor state and behavior: voxel picking, the tools, the undo
//! history, selections, the clipboard and named sockets.

mod clipboard;
mod commands;
mod raycast;
mod selection;
mod shapes;
mod socket;
mod tools;
mod transform;

pub use clipboard::{
    build_clear_changes, build_move_changes, build_paste_changes, copy_selection_to_clipboard,
    Clipboard,
};
pub use commands::{Command, CommandHistory, GraphTransition, VoxelChange};
pub use raycast::{Ray, RaycastHit, VoxelRaycast};
pub use selection::Selection;
pub use shapes::{box_voxels, cylinder_voxels, line_voxels, sphere_voxels};
pub use socket::{next_socket_name, Socket};
pub use tools::{
    compute_flood_fill_changes, eyedrop, flood_fill, flood_fill_multi, BrushTool, EditorTool,
    FillOutcome, Tool, ToolContext,
};
pub use transform::{
    build_remap_changes, mirror_pos, mirror_selection_changes, rotate_pos,
    rotate_selection_changes, rotated_aabb, Axis, Quarter,
};

use crate::core::Voxel;

/// Mirroring of brush writes across the world-origin planes; enabled
/// axes combine, up to 8-fold. Cell-aligned: cell `n` reflects to
/// `-n - 1`, so the plane lies between cells rather than through one.
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
    /// Active symmetry mirroring for brush writes (Place / Remove /
    /// Paint / Fill all honor it; Eyedropper doesn't write so it's
    /// exempt). Persists across sessions via prefs.
    pub symmetry: SymmetryAxes,
    /// Active box selection, if any. Neither persisted across sessions
    /// nor pushed onto the undo stack — an ephemeral marquee, as in
    /// image editors.
    pub selection: Option<Selection>,
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
            symmetry: SymmetryAxes::default(),
            selection: None,
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
            Voxel::from_rgb(139, 90, 43),   // Brown
            Voxel::from_rgb(76, 153, 0),    // Grass green
            Voxel::from_rgb(194, 178, 128), // Sand
            Voxel::from_rgb(128, 128, 128), // Stone
            // Vivid colors
            Voxel::from_rgb(255, 128, 0),   // Orange
            Voxel::from_rgb(128, 0, 255),   // Purple
            Voxel::from_rgb(255, 192, 203), // Pink
            Voxel::from_rgb(0, 128, 128),   // Teal
        ]
    }

    /// Switch to `tool` because the user asked for it. Alt's transient
    /// eyedropper never comes through here — it is derived per read, so
    /// an explicit pick can't be rolled back by an Alt release.
    pub fn select_tool(&mut self, tool: Tool) {
        self.current_tool = tool;
    }

    /// Set the brush color from a palette index, preserving the
    /// material flags: those behave like a brush mode, so picking a
    /// color shouldn't clear them.
    pub fn set_palette_color(&mut self, index: usize) {
        if let Some(c) = self.palette.get(index) {
            self.brush_color.r = c.r;
            self.brush_color.g = c.g;
            self.brush_color.b = c.b;
            self.brush_color.a = c.a;
        }
    }

    /// Undo last action
    pub fn undo(
        &mut self,
        world: &mut crate::core::World,
        graph: &mut crate::procgen::PipelineGraph,
    ) -> bool {
        self.history.undo(world, graph)
    }

    /// Redo last undone action
    pub fn redo(
        &mut self,
        world: &mut crate::core::World,
        graph: &mut crate::procgen::PipelineGraph,
    ) -> bool {
        self.history.redo(world, graph)
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
        let s = SymmetryAxes {
            x: true,
            ..Default::default()
        };
        assert_eq!(s.count(), 2);
        assert_eq!(
            s.mirror_positions((5, 7, 11)),
            vec![(5, 7, 11), (-6, 7, 11)]
        );
    }

    #[test]
    fn test_all_axes_octuple() {
        let s = SymmetryAxes {
            x: true,
            y: true,
            z: true,
        };
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
        let s = SymmetryAxes {
            x: true,
            ..Default::default()
        };
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
        assert!(SymmetryAxes {
            x: true,
            ..Default::default()
        }
        .any());
        assert!(SymmetryAxes {
            y: true,
            ..Default::default()
        }
        .any());
        assert!(SymmetryAxes {
            z: true,
            ..Default::default()
        }
        .any());
    }
}
