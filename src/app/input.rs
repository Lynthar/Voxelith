//! Input handling: voxel raycast, tool application, keyboard shortcuts.

use winit::keyboard::KeyCode;

use voxelith::editor::{
    eyedrop, flood_fill, BrushTool, EditorTool, Ray, Tool, ToolContext, VoxelRaycast,
};

use super::App;

impl App {
    /// Update the editor's hovered voxel from the current cursor position.
    ///
    /// For the Place tool we fall back to a virtual hit on the y=0
    /// plane when the ray misses every voxel — that way the user can
    /// place voxels into a freshly-cleared (empty) world. Other tools
    /// stay strict (real voxel only): their semantics break on virtual
    /// hits (Eyedropper would sample air, Fill would explode trying to
    /// flood a 3D air region, Remove/Paint would no-op but with
    /// confusing brush previews dangling in mid-air).
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

        self.editor.hovered_voxel = if self.editor.current_tool == Tool::Place {
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
                if !v.is_air() {
                    flood_fill(
                        &mut self.world,
                        &mut self.editor.history,
                        hit.voxel_pos,
                        self.editor.brush_color,
                        10000,
                    );
                }
            }
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
