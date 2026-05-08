//! Input handling: voxel raycast, tool application, keyboard shortcuts.

use winit::keyboard::KeyCode;

use std::collections::HashSet;

use voxelith::editor::{
    box_voxels, cylinder_voxels, eyedrop, flood_fill, flood_fill_multi, line_voxels,
    sphere_voxels, BrushTool, Command, EditorTool, Ray, Tool, ToolContext, VoxelChange,
    VoxelRaycast,
};

use super::App;

impl App {
    /// Update the editor's hovered voxel from the current cursor position.
    ///
    /// Tools that need an "anchor cell" to place new geometry (Place
    /// and the four shape tools) get a y=0 ground-plane fallback when
    /// the ray misses every voxel — that way they work in a freshly-
    /// cleared (empty) world. Tools that read existing voxels
    /// (Remove/Paint/Eyedropper/Fill) stay strict: virtual hits would
    /// give confusing previews and either no-op or, worse, explode
    /// (Fill flooding a 3D air region).
    pub(super) fn update_raycast(&mut self) {
        let Some(renderer) = &self.renderer else {
            return;
        };
        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();

        let view_proj = renderer.camera.view_projection_matrix();
        let view_proj_inv = view_proj.inverse();

        let ray = Ray::from_screen(
            self.cursor_pos,
            (size.width as f32, size.height as f32),
            view_proj_inv,
        );

        self.editor.hovered_voxel = if self.editor.current_tool.uses_ground_plane_fallback() {
            VoxelRaycast::cast_with_ground_plane(&ray, &self.world, 100.0, 0)
        } else {
            VoxelRaycast::cast(&ray, &self.world, 100.0)
        };
    }

    /// Apply the current tool at the hovered location.
    pub(super) fn apply_tool(&mut self) {
        let Some(hit) = self.editor.hovered_voxel else {
            return;
        };

        match self.editor.current_tool {
            Tool::Place | Tool::Remove | Tool::Paint => {
                let brush = BrushTool::new(self.editor.current_tool);
                let mut ctx = ToolContext {
                    world: &mut self.world,
                    history: &mut self.editor.history,
                    brush_color: self.editor.brush_color,
                    brush_size: self.editor.brush_size,
                    symmetry: self.editor.symmetry,
                };
                brush.apply(&mut ctx, &hit);
            }
            Tool::Eyedropper => {
                if let Some(color) = eyedrop(&self.world, &hit) {
                    self.editor.brush_color = color;
                }
            }
            Tool::Fill => {
                // Refuse to flood from an air cell: with Place's ground-
                // plane fallback in play the hit could in principle be a
                // virtual sub-plane voxel, and flooding from there would
                // eat the entire 3D air region around the cursor (capped
                // by `flood_fill`'s spatial limit, but still visually
                // alarming and never what the user meant).
                let v = self.world.get_voxel(
                    hit.voxel_pos.0,
                    hit.voxel_pos.1,
                    hit.voxel_pos.2,
                );
                if v.is_air() {
                    return;
                }
                let symmetry = self.editor.symmetry;
                if symmetry.any() {
                    // Combine all mirrored fills into one undo entry —
                    // a single click should be a single undo, even at
                    // 8-fold symmetry.
                    let starts = symmetry.mirror_positions(hit.voxel_pos);
                    flood_fill_multi(
                        &mut self.world,
                        &mut self.editor.history,
                        &starts,
                        self.editor.brush_color,
                        10000,
                    );
                } else {
                    flood_fill(
                        &mut self.world,
                        &mut self.editor.history,
                        hit.voxel_pos,
                        self.editor.brush_color,
                        10000,
                    );
                }
            }
            Tool::Line | Tool::Box | Tool::Sphere | Tool::Cylinder => {
                // Shape press: latch the anchor at the current cell.
                // The shape's full voxel set is computed and committed
                // in `commit_shape` on mouse-up.
                self.shape_drag_anchor = Some(hit.adjacent_pos);
            }
        }
    }

    /// Commit the in-progress shape drag (called on left-button
    /// release). Computes the shape's voxels from anchor → current
    /// hover, applies symmetry, and submits one `Command` so the
    /// shape is a single undo entry. No-op if there's no active drag
    /// or the cursor isn't over a valid cell.
    pub(super) fn commit_shape(&mut self) {
        let Some(anchor) = self.shape_drag_anchor.take() else {
            return;
        };
        let Some(hit) = self.editor.hovered_voxel else {
            return;
        };
        let tool = self.editor.current_tool;
        // `shape_end_pos` resolves the drag end: real-voxel hits use
        // their 3D adjacent_pos, virtual-ground hits substitute the
        // press-screen-vertical delta for Y so empty-world drags
        // produce real 3D shapes (Sphere → ball not disk; Cylinder
        // → tower not flat ring).
        let end = self.shape_end_pos(anchor, &hit);

        let raw = match tool {
            Tool::Line => line_voxels(anchor, end),
            Tool::Box => box_voxels(anchor, end),
            Tool::Sphere => sphere_voxels(anchor, end),
            Tool::Cylinder => cylinder_voxels(anchor, end),
            _ => return, // anchor only set for shape tools, defensive
        };

        // Apply symmetry across world-origin planes. HashSet dedupes
        // cells where mirrored shapes overlap (e.g. a Y-symmetric
        // shape spanning y=0 covers cells in both halves).
        let symmetry = self.editor.symmetry;
        let positions: Vec<(i32, i32, i32)> = if symmetry.any() {
            let mut set: HashSet<(i32, i32, i32)> = HashSet::new();
            for cell in raw {
                for m in symmetry.mirror_positions(cell) {
                    set.insert(m);
                }
            }
            set.into_iter().collect()
        } else {
            raw
        };

        let color = self.editor.brush_color;
        let changes: Vec<VoxelChange> = positions
            .into_iter()
            .map(|pos| VoxelChange {
                pos,
                old_voxel: self.world.get_voxel(pos.0, pos.1, pos.2),
                new_voxel: color,
            })
            .filter(|c| c.old_voxel != c.new_voxel)
            .collect();

        if !changes.is_empty() {
            let cmd = Command::set_voxels(changes);
            self.editor.history.execute(cmd, &mut self.world);
        }
    }

    /// Handle keyboard shortcuts (tools, undo/redo, file ops).
    pub(super) fn handle_tool_shortcut(&mut self, key: KeyCode) {
        match key {
            KeyCode::Digit1 => self.editor.current_tool = Tool::Place,
            KeyCode::Digit2 => self.editor.current_tool = Tool::Remove,
            KeyCode::Digit3 => self.editor.current_tool = Tool::Paint,
            KeyCode::Digit4 => self.editor.current_tool = Tool::Eyedropper,
            KeyCode::Digit5 => self.editor.current_tool = Tool::Fill,
            KeyCode::Digit6 => self.editor.current_tool = Tool::Line,
            KeyCode::Digit7 => self.editor.current_tool = Tool::Box,
            KeyCode::Digit8 => self.editor.current_tool = Tool::Sphere,
            KeyCode::Digit9 => self.editor.current_tool = Tool::Cylinder,
            KeyCode::KeyZ if self.modifiers.control_key() => {
                if self.modifiers.shift_key() {
                    self.editor.redo(&mut self.world);
                } else {
                    self.editor.undo(&mut self.world);
                }
            }
            KeyCode::KeyY if self.modifiers.control_key() => {
                self.editor.redo(&mut self.world);
            }
            KeyCode::KeyS if self.modifiers.control_key() => {
                if self.modifiers.shift_key() {
                    self.save_project_as();
                } else {
                    self.save_project();
                }
            }
            KeyCode::KeyO if self.modifiers.control_key() => {
                self.open_project();
            }
            KeyCode::KeyN if self.modifiers.control_key() => {
                self.new_project();
            }
            _ => {}
        }
    }
}
