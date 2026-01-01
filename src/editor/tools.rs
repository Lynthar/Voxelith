//! Editor tools for voxel manipulation.
//!
//! Provides different brush types and editing modes.

use super::{Command, CommandHistory, RaycastHit, VoxelChange};
use crate::core::{Voxel, World};

/// Available editing tools
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    /// Place voxels
    Place,
    /// Remove voxels
    Remove,
    /// Paint existing voxels (change color without adding/removing)
    Paint,
    /// Pick color from existing voxel
    Eyedropper,
    /// Fill region with voxels
    Fill,
}

impl Tool {
    /// Get display name
    pub fn name(&self) -> &'static str {
        match self {
            Tool::Place => "Place",
            Tool::Remove => "Remove",
            Tool::Paint => "Paint",
            Tool::Eyedropper => "Eyedropper",
            Tool::Fill => "Fill",
        }
    }

    /// Get keyboard shortcut hint
    pub fn shortcut(&self) -> &'static str {
        match self {
            Tool::Place => "1",
            Tool::Remove => "2",
            Tool::Paint => "3",
            Tool::Eyedropper => "4 / Alt",
            Tool::Fill => "5",
        }
    }
}

/// Context passed to tools during execution
pub struct ToolContext<'a> {
    pub world: &'a mut World,
    pub history: &'a mut CommandHistory,
    pub brush_color: Voxel,
    pub brush_size: u8,
}

/// Trait for tool implementations
pub trait EditorTool {
    /// Apply the tool at the given hit location
    fn apply(&self, ctx: &mut ToolContext, hit: &RaycastHit);

    /// Get the preview positions (voxels that would be affected)
    fn preview_positions(&self, hit: &RaycastHit, brush_size: u8) -> Vec<(i32, i32, i32)>;
}

/// Brush tool for place/remove/paint operations
pub struct BrushTool {
    pub mode: Tool,
}

impl BrushTool {
    pub fn new(mode: Tool) -> Self {
        Self { mode }
    }

    /// Get affected positions for a spherical brush
    fn get_brush_positions(center: (i32, i32, i32), size: u8) -> Vec<(i32, i32, i32)> {
        let mut positions = Vec::new();
        let radius = (size as i32 - 1).max(0);
        let radius_sq = (radius as f32 + 0.5).powi(2);

        for dz in -radius..=radius {
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let dist_sq = (dx * dx + dy * dy + dz * dz) as f32;
                    if dist_sq <= radius_sq {
                        positions.push((
                            center.0 + dx,
                            center.1 + dy,
                            center.2 + dz,
                        ));
                    }
                }
            }
        }

        positions
    }
}

impl EditorTool for BrushTool {
    fn apply(&self, ctx: &mut ToolContext, hit: &RaycastHit) {
        match self.mode {
            Tool::Place => {
                // Place at adjacent position
                let positions = Self::get_brush_positions(hit.adjacent_pos, ctx.brush_size);
                let changes: Vec<VoxelChange> = positions
                    .iter()
                    .map(|&pos| VoxelChange {
                        pos,
                        old_voxel: ctx.world.get_voxel(pos.0, pos.1, pos.2),
                        new_voxel: ctx.brush_color,
                    })
                    .filter(|c| c.old_voxel != c.new_voxel)
                    .collect();

                if !changes.is_empty() {
                    let cmd = Command::set_voxels(changes);
                    ctx.history.execute(cmd, ctx.world);
                }
            }
            Tool::Remove => {
                // Remove at hit position
                let positions = Self::get_brush_positions(hit.voxel_pos, ctx.brush_size);
                let changes: Vec<VoxelChange> = positions
                    .iter()
                    .filter_map(|&pos| {
                        let old = ctx.world.get_voxel(pos.0, pos.1, pos.2);
                        if !old.is_air() {
                            Some(VoxelChange {
                                pos,
                                old_voxel: old,
                                new_voxel: Voxel::AIR,
                            })
                        } else {
                            None
                        }
                    })
                    .collect();

                if !changes.is_empty() {
                    let cmd = Command::set_voxels(changes);
                    ctx.history.execute(cmd, ctx.world);
                }
            }
            Tool::Paint => {
                // Paint at hit position (change color of existing voxels)
                let positions = Self::get_brush_positions(hit.voxel_pos, ctx.brush_size);
                let changes: Vec<VoxelChange> = positions
                    .iter()
                    .filter_map(|&pos| {
                        let old = ctx.world.get_voxel(pos.0, pos.1, pos.2);
                        if !old.is_air() && old != ctx.brush_color {
                            Some(VoxelChange {
                                pos,
                                old_voxel: old,
                                new_voxel: ctx.brush_color,
                            })
                        } else {
                            None
                        }
                    })
                    .collect();

                if !changes.is_empty() {
                    let cmd = Command::set_voxels(changes);
                    ctx.history.execute(cmd, ctx.world);
                }
            }
            Tool::Eyedropper | Tool::Fill => {
                // Eyedropper and Fill are handled separately
            }
        }
    }

    fn preview_positions(&self, hit: &RaycastHit, brush_size: u8) -> Vec<(i32, i32, i32)> {
        match self.mode {
            Tool::Place => Self::get_brush_positions(hit.adjacent_pos, brush_size),
            Tool::Remove | Tool::Paint => Self::get_brush_positions(hit.voxel_pos, brush_size),
            Tool::Eyedropper | Tool::Fill => vec![hit.voxel_pos],
        }
    }
}

/// Pick color from a voxel
pub fn eyedrop(world: &World, hit: &RaycastHit) -> Option<Voxel> {
    let voxel = world.get_voxel(hit.voxel_pos.0, hit.voxel_pos.1, hit.voxel_pos.2);
    if !voxel.is_air() {
        Some(voxel)
    } else {
        None
    }
}

/// Flood fill starting from a position
pub fn flood_fill(
    world: &mut World,
    history: &mut CommandHistory,
    start: (i32, i32, i32),
    new_voxel: Voxel,
    max_voxels: usize,
) -> usize {
    let target_voxel = world.get_voxel(start.0, start.1, start.2);

    // Don't fill if same color or target is solid and new is air
    if target_voxel == new_voxel {
        return 0;
    }

    let mut changes = Vec::new();
    let mut visited = std::collections::HashSet::new();
    let mut stack = vec![start];

    while let Some(pos) = stack.pop() {
        if visited.contains(&pos) {
            continue;
        }
        if changes.len() >= max_voxels {
            break;
        }

        let current = world.get_voxel(pos.0, pos.1, pos.2);
        if current != target_voxel {
            continue;
        }

        visited.insert(pos);
        changes.push(VoxelChange {
            pos,
            old_voxel: current,
            new_voxel,
        });

        // Add neighbors (6-connectivity)
        let neighbors = [
            (pos.0 + 1, pos.1, pos.2),
            (pos.0 - 1, pos.1, pos.2),
            (pos.0, pos.1 + 1, pos.2),
            (pos.0, pos.1 - 1, pos.2),
            (pos.0, pos.1, pos.2 + 1),
            (pos.0, pos.1, pos.2 - 1),
        ];

        for neighbor in neighbors {
            if !visited.contains(&neighbor) {
                stack.push(neighbor);
            }
        }
    }

    let count = changes.len();
    if !changes.is_empty() {
        let cmd = Command::set_voxels(changes);
        history.execute(cmd, world);
    }

    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brush_positions() {
        let positions = BrushTool::get_brush_positions((0, 0, 0), 1);
        assert_eq!(positions.len(), 1);
        assert!(positions.contains(&(0, 0, 0)));

        let positions = BrushTool::get_brush_positions((0, 0, 0), 2);
        assert!(positions.len() > 1);
    }

    #[test]
    fn test_flood_fill() {
        let mut world = World::new();
        let mut history = CommandHistory::new(100);

        // Create a small area to fill
        for x in 0..3 {
            for z in 0..3 {
                world.set_voxel(x, 0, z, Voxel::from_rgb(100, 100, 100));
            }
        }
        world.clear_dirty_flags();

        // Flood fill with new color
        let count = flood_fill(
            &mut world,
            &mut history,
            (1, 0, 1),
            Voxel::from_rgb(255, 0, 0),
            1000,
        );

        assert_eq!(count, 9);
        assert_eq!(world.get_voxel(0, 0, 0).r, 255);
    }
}
