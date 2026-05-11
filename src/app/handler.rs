//! winit `ApplicationHandler` integration.
//!
//! Egui consumes events first; only unconsumed events reach the editor
//! and camera controller. The Alt key temporarily swaps the active tool
//! to `Eyedropper` (saving the prior tool in `editor.tool_before_alt`)
//! and restores it on release.

use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use voxelith::editor::Tool;

use super::App;

/// Squared pixel distance the cursor must travel from the left-press
/// point before drag-paint engages. 8 px tolerates normal click
/// tremor without blocking deliberate drags.
const DRAG_THRESHOLD_PX_SQ: f32 = 8.0 * 8.0;

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let (w, h) = self.initial_window_size();
            let window_attrs = Window::default_attributes()
                .with_title("Voxelith")
                .with_inner_size(winit::dpi::LogicalSize::new(w, h));

            let window = event_loop.create_window(window_attrs).unwrap();
            self.init(window);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // Let egui see the event first — its `consumed` flag gates the editor.
        let egui_consumed = {
            let window = self.window.as_ref().unwrap();
            let egui_state = self.egui_state.as_mut().unwrap();
            egui_state.on_window_event(window, &event).consumed
        };

        match event {
            WindowEvent::CloseRequested => {
                self.save_prefs();
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size);
                }
            }

            WindowEvent::ModifiersChanged(new_modifiers) => {
                let old_alt = self.modifiers.alt_key();
                let new_alt = new_modifiers.state().alt_key();
                self.modifiers = new_modifiers.state();

                // Alt-press: swap to eyedropper, remember prior tool.
                // Alt-release: restore.
                if new_alt && !old_alt {
                    if self.editor.current_tool != Tool::Eyedropper {
                        self.editor.tool_before_alt = Some(self.editor.current_tool);
                        self.editor.current_tool = Tool::Eyedropper;
                    }
                } else if !new_alt && old_alt {
                    if let Some(tool) = self.editor.tool_before_alt.take() {
                        self.editor.current_tool = tool;
                    }
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if !egui_consumed {
                    if let PhysicalKey::Code(key) = event.physical_key {
                        if let Some(renderer) = &mut self.renderer {
                            renderer.camera_controller.process_keyboard(key, event.state);
                        }

                        if event.state.is_pressed() {
                            self.handle_tool_shortcut(key);
                        }

                        if key == KeyCode::Escape && event.state.is_pressed() {
                            self.cursor_captured = false;
                            if let Some(window) = &self.window {
                                window.set_cursor_visible(true);
                            }
                        }
                    }
                }
            }

            WindowEvent::MouseInput { button, state, .. } => {
                if !egui_consumed {
                    if let Some(renderer) = &mut self.renderer {
                        renderer.camera_controller.process_mouse_button(
                            button,
                            state,
                            &mut renderer.camera,
                        );
                    }

                    if button == winit::event::MouseButton::Left {
                        let tool = self.editor.current_tool;
                        if state == ElementState::Pressed {
                            // Brush tools: apply on press, then drag-paint
                            // re-applies on motion. Shape tools / Select:
                            // latch the anchor on press and commit on
                            // release — the in-between motion only
                            // refreshes the translucent preview / AABB
                            // wireframe.
                            self.apply_tool();
                            self.left_button_held = true;
                            self.last_stroke_voxel =
                                self.editor.hovered_voxel.map(|h| h.voxel_pos);
                            self.stroke_start_screen_pos = Some(self.cursor_pos);
                        } else {
                            // Shape release no longer commits — it
                            // transitions to the Height phase. The
                            // shape's full voxel set is committed by
                            // a SECOND click while in Height (see
                            // `apply_tool` shape branch). This is
                            // vengi-style two-phase drag: footprint
                            // (W × D) on a locked plane, then
                            // height (H) along the plane normal,
                            // splitting screen X / Y and screen Y
                            // into two unambiguous axes.
                            if tool.is_shape() {
                                self.transition_shape_to_height();
                            } else if matches!(tool, Tool::Select) {
                                self.commit_selection();
                            } else {
                                // Brush end: finalize the merged command
                                // so the next click starts a fresh undo.
                                self.editor.history.end_stroke();
                            }
                            self.left_button_held = false;
                            self.last_stroke_voxel = None;
                            self.stroke_start_screen_pos = None;
                            // Drop the plane lock so the next stroke
                            // captures a fresh face. Hover preview
                            // immediately falls back to ray-vs-voxels.
                            self.stroke_plane = None;
                        }
                    }

                    if button == winit::event::MouseButton::Middle {
                        // Middle-drag captures the cursor for orbit; release
                        // must explicitly uncapture or `device_event` keeps
                        // consuming MouseMotion as orbit input forever.
                        let pressed = state == ElementState::Pressed;
                        self.cursor_captured = pressed;
                        if let Some(window) = &self.window {
                            window.set_cursor_visible(!pressed);
                        }
                    }
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if !egui_consumed {
                    // Compute the zoom anchor (cursor's 3D point on
                    // geometry, with a target-depth-plane fallback) BEFORE
                    // taking the mutable renderer borrow. Without zoom-to-
                    // cursor, scroll-zooming over a voxel of interest
                    // doesn't keep that voxel under the cursor — the camera
                    // dollies along the camera→target axis, the voxel
                    // drifts off-screen, and a subsequent middle-orbit
                    // pivots around `target` (which is wherever it was
                    // before, often underground or in mid-air relative to
                    // the user's actual focus). Scaling around the cursor
                    // anchor migrates `target` with the zoom so orbit
                    // naturally circles the inspected feature.
                    if let Some(anchor) = self.compute_zoom_anchor() {
                        if let Some(renderer) = &mut self.renderer {
                            renderer.camera_controller.process_scroll(
                                delta,
                                &mut renderer.camera,
                                anchor,
                            );
                        }
                    }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_pos = (position.x as f32, position.y as f32);

                if !egui_consumed {
                    self.update_raycast();

                    // Drag-paint: while left button is held, re-apply
                    // the brush whenever the hover crosses into a new
                    // voxel. Limited to brush-style tools — Eyedropper
                    // / Fill keep their click-only behavior to avoid
                    // spam (Fill especially would explode the
                    // history). A pixel dead-zone around the press
                    // point absorbs unintended micro-drags from a
                    // single click.
                    if self.left_button_held {
                        let drag_eligible = matches!(
                            self.editor.current_tool,
                            Tool::Place | Tool::Remove | Tool::Paint
                        );
                        let past_dead_zone =
                            self.stroke_start_screen_pos.map_or(false, |(sx, sy)| {
                                let dx = self.cursor_pos.0 - sx;
                                let dy = self.cursor_pos.1 - sy;
                                dx * dx + dy * dy >= DRAG_THRESHOLD_PX_SQ
                            });
                        if drag_eligible && past_dead_zone {
                            let current =
                                self.editor.hovered_voxel.map(|h| h.voxel_pos);
                            if current.is_some() && current != self.last_stroke_voxel {
                                self.apply_tool();
                                self.last_stroke_voxel = current;
                            }
                        }
                    }

                    if let Some(renderer) = &mut self.renderer {
                        renderer.camera_controller.process_mouse_motion(
                            position.x as f32,
                            position.y as f32,
                            &mut renderer.camera,
                        );
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32();
                self.last_frame = now;

                self.frame_times.push_back(dt * 1000.0);
                if self.frame_times.len() > 60 {
                    self.frame_times.pop_front();
                }

                self.tick_preview();
                self.tick_ai_job();
                self.update_brush_preview();
                self.update_selection_visualization();
                self.rebuild_all_meshes();
                self.render_frame(dt);

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        // Raw mouse motion drives smoother orbit when the cursor is captured.
        // Sign matches `CameraController::process_mouse_motion` — drag-the-scene.
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.cursor_captured {
                if let Some(renderer) = &mut self.renderer {
                    renderer.camera_controller.yaw += delta.0 as f32 * 0.003;
                    renderer.camera_controller.pitch += delta.1 as f32 * 0.003;
                    renderer.camera_controller.pitch =
                        renderer.camera_controller.pitch.clamp(-1.5, 1.5);

                    let distance = renderer.camera_controller.distance;
                    let yaw = renderer.camera_controller.yaw;
                    let pitch = renderer.camera_controller.pitch;

                    let x = distance * yaw.cos() * pitch.cos();
                    let y = distance * pitch.sin();
                    let z = distance * yaw.sin() * pitch.cos();

                    renderer.camera.position =
                        renderer.camera.target + glam::Vec3::new(x, y, z);
                }
            }
        }
    }
}
