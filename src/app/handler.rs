//! winit `ApplicationHandler` integration.
//!
//! Egui consumes events first; only unconsumed events reach the editor
//! and camera controller. The Alt key temporarily swaps the active tool
//! to `Eyedropper` (saving the prior tool in `editor.tool_before_alt`)
//! and restores it on release.

use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
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
                // Clean shutdown: drop the crash-recovery autosave so the
                // next launch doesn't mistake this for a crash.
                self.delete_autosave();
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size);
                }
            }

            WindowEvent::Focused(false) => {
                // Losing focus (alt-tab, or a modal Save/Open dialog
                // taking over) means key-release events can be delivered
                // elsewhere. Forget held keys so flight doesn't resume
                // with a phantom WASD key stuck down when focus returns.
                if let Some(renderer) = &mut self.renderer {
                    renderer.camera_controller.clear_keys();
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
                        // Command chords (Ctrl/Super + key) are editor
                        // shortcuts, not fly-camera input. Feeding the
                        // chord's letter (e.g. the 'S' in Ctrl+S) to the
                        // controller would dolly the camera while held —
                        // and if a modal Save/Open dialog swallows the
                        // key-release, the camera drifts forever. So drop
                        // the *press* while a command modifier is held,
                        // but always forward the *release* so a key pressed
                        // before the modifier (hold W, then tap Ctrl) can
                        // never get stuck "down". Sprint lives on Shift
                        // (see `CameraController::update`), which isn't a
                        // command modifier, so Shift+WASD is unaffected.
                        let command_chord =
                            self.modifiers.control_key() || self.modifiers.super_key();
                        let skip_camera = command_chord && event.state.is_pressed();
                        if !skip_camera {
                            if let Some(renderer) = &mut self.renderer {
                                renderer
                                    .camera_controller
                                    .process_keyboard(key, event.state);
                            }
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
                let pressed = state == ElementState::Pressed;

                // A *press* only acts when egui didn't take it — clicking a
                // panel must not start an orbit / pan / brush stroke. A
                // *release* always runs, even when egui consumed it: a button
                // let go over a panel (after dragging out of the viewport
                // onto one) must still tear down in-progress state, or the
                // latches stick — orbit/pan wedged on, `cursor_captured`
                // stuck (the raw-motion orbit in `device_event` ignores egui
                // entirely), a phantom selection anchor, or `left_button_held`
                // jammed true so the next tool drag-paints while the old
                // selection still tracks. That stranded release is exactly
                // the "tool states stack and can't be cancelled" bug.
                if pressed && !egui_consumed {
                    // Middle-press re-anchors the orbit pivot onto whatever
                    // the camera's forward ray hits (voxel surface, else the
                    // y=0 ground, else the current target). The hit lies on
                    // the view ray, so re-anchoring never jumps the image —
                    // only the orbit distance changes. Must precede
                    // `process_mouse_button`, whose middle-press
                    // `sync_orbit_state_from_camera` reads the new target.
                    if button == MouseButton::Middle {
                        if let Some(pivot) = self.compute_orbit_pivot() {
                            if let Some(renderer) = &mut self.renderer {
                                renderer.camera.target = pivot;
                            }
                        }
                    }
                    if let Some(renderer) = &mut self.renderer {
                        renderer.camera_controller.process_mouse_button(
                            button,
                            state,
                            &mut renderer.camera,
                        );
                    }
                    if button == MouseButton::Left {
                        // Brush tools apply on press, then drag-paint
                        // re-applies on motion. Shape / Select latch an
                        // anchor here and commit on release.
                        self.apply_tool();
                        self.left_button_held = true;
                        self.last_stroke_voxel =
                            self.editor.hovered_voxel.map(|h| h.voxel_pos);
                        self.stroke_start_screen_pos = Some(self.cursor_pos);
                    }
                    if button == MouseButton::Middle {
                        // Capture the cursor for orbit; the release branch
                        // uncaptures unconditionally.
                        self.cursor_captured = true;
                        if let Some(window) = &self.window {
                            window.set_cursor_visible(false);
                        }
                    }
                } else if !pressed {
                    // Always let the controller see the release so its
                    // middle / right pressed-flags and `last_mouse_pos` reset
                    // even when the cursor is over a panel.
                    if let Some(renderer) = &mut self.renderer {
                        renderer.camera_controller.process_mouse_button(
                            button,
                            state,
                            &mut renderer.camera,
                        );
                    }
                    if button == MouseButton::Left {
                        // Finalize an in-progress interaction only if a press
                        // actually started one in the viewport; either way,
                        // clear every latch so nothing carries into the next
                        // click. Shape release transitions to the Height
                        // phase (committed by a second click — vengi-style
                        // two-phase drag); Select commits the AABB; a brush
                        // seals its merged undo entry.
                        if self.left_button_held {
                            let tool = self.editor.current_tool;
                            if tool.is_shape() {
                                self.transition_shape_to_height();
                            } else if matches!(tool, Tool::Select) {
                                self.commit_selection();
                            } else {
                                self.editor.history.end_stroke();
                            }
                        }
                        self.left_button_held = false;
                        self.last_stroke_voxel = None;
                        self.stroke_start_screen_pos = None;
                        self.stroke_plane = None;
                        // Defensive: drop any select drag/move anchors in
                        // case a press latched one but egui swallowed the
                        // release before `commit_selection` could take it.
                        self.selection_drag_anchor = None;
                        self.selection_move_anchor = None;
                    }
                    if button == MouseButton::Middle {
                        self.cursor_captured = false;
                        if let Some(window) = &self.window {
                            window.set_cursor_visible(true);
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
                self.update_socket_visualization();
                self.rebuild_all_meshes();
                self.tick_autosave();
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
