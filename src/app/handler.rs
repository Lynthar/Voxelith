//! winit `ApplicationHandler` integration. Egui consumes events first;
//! only what it leaves reaches the editor and camera controller.

use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use voxelith::editor::Tool;

use super::{App, PendingAction};

/// Squared pixel distance the cursor must travel from the left-press
/// point before drag-paint engages. 8 px tolerates normal click
/// tremor without blocking deliberate drags.
const DRAG_THRESHOLD_PX_SQ: f32 = 8.0 * 8.0;

/// The window icon, decoded from the same artwork `build.rs` embeds as
/// the exe resource, so pinned and running taskbar items match. `None`
/// on decode failure — not worth failing startup over.
fn window_icon() -> Option<winit::window::Icon> {
    let bytes = include_bytes!("../../assets/branding/icon_64.png");
    let img = image::load_from_memory(bytes).ok()?.into_rgba8();
    let (w, h) = img.dimensions();
    winit::window::Icon::from_rgba(img.into_raw(), w, h).ok()
}

impl App {
    /// Persist prefs, drop the crash-recovery autosave, and stop the
    /// event loop. Shared by the close button and the Exit menu item so
    /// the two can't drift.
    fn shutdown(&mut self, event_loop: &ActiveEventLoop) {
        self.save_prefs();
        self.delete_autosave();
        event_loop.exit();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let saved = self.initial_window_size();
            // The prefs entry can outlive the display it was sized on,
            // so check it against the monitor first. `primary_monitor`
            // is the best guess — the window doesn't exist yet.
            let (w, h) = match event_loop.primary_monitor() {
                Some(monitor) => {
                    let logical: winit::dpi::LogicalSize<u32> =
                        monitor.size().to_logical(monitor.scale_factor());
                    super::fit_window_to_monitor(saved, (logical.width, logical.height))
                }
                None => saved,
            };
            let window_attrs = Window::default_attributes()
                .with_title("Voxelith")
                .with_window_icon(window_icon())
                .with_inner_size(winit::dpi::LogicalSize::new(w, h));

            let window = event_loop.create_window(window_attrs).unwrap();
            self.init(window);
            // Kick the first frame: the scheduler re-arms itself, but
            // the chain has to start somewhere, and macOS delivers no
            // initial paint event — the window would stay blank forever.
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // The frame scheduler: `RedrawRequested` sets `next_frame_at`,
        // and this arms the redraw when it comes due. Input wakes the
        // loop regardless, so a key press lands instantly.
        let Some(window) = &self.window else {
            return;
        };
        if Instant::now() >= self.next_frame_at {
            window.request_redraw();
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_frame_at));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // Any input (or window-geometry change) marks the user active —
        // the frame scheduler renders at full rate for a grace window
        // after this stamp and drops to the idle heartbeat otherwise.
        if matches!(
            event,
            WindowEvent::KeyboardInput { .. }
                | WindowEvent::MouseInput { .. }
                | WindowEvent::MouseWheel { .. }
                | WindowEvent::CursorMoved { .. }
                | WindowEvent::ModifiersChanged(_)
                | WindowEvent::Touch(_)
                | WindowEvent::PinchGesture { .. }
                | WindowEvent::Resized(_)
                | WindowEvent::ScaleFactorChanged { .. }
                | WindowEvent::Focused(_)
        ) {
            self.last_interaction = Instant::now();
        }

        // Let egui see the event first — its `consumed` flag gates the editor.
        let egui_consumed = {
            let window = self.window.as_ref().unwrap();
            let egui_state = self.egui_state.as_mut().unwrap();
            egui_state.on_window_event(window, &event).consumed
        };

        match event {
            WindowEvent::CloseRequested => {
                // Through the unsaved-changes guard: a clean exit deletes
                // the autosave, so closing with unsaved work would
                // destroy the only copy of it without a word.
                self.guard_then(PendingAction::Exit);
                if self.exit_requested {
                    self.shutdown(event_loop);
                }
            }

            WindowEvent::Resized(size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size);
                }
            }

            WindowEvent::Focused(false) => {
                // Losing focus means press and release events can go
                // elsewhere, so abandon every in-progress interaction.
                // The committed marquee itself is left intact.
                if let Some(renderer) = &mut self.renderer {
                    renderer.camera_controller.clear_keys();
                    renderer.camera_controller.clear_mouse_buttons();
                }
                self.cancel_interaction();
                // Modifiers refresh only on `ModifiersChanged`, which
                // goes to whoever has focus — so alt-tabbing away leaves
                // Alt claiming to be down and the eyedropper latched.
                self.modifiers = Default::default();
                if self.cursor_captured {
                    self.cursor_captured = false;
                    if let Some(window) = &self.window {
                        window.set_cursor_visible(true);
                    }
                }
            }

            WindowEvent::ModifiersChanged(new_modifiers) => {
                // Alt's temporary eyedropper needs no handling here:
                // `effective_tool` derives it from this state per read.
                self.modifiers = new_modifiers.state();
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    let pressed = event.state.is_pressed();

                    // A camera-key release must always reach the
                    // controller, even when egui consumed it, or a held
                    // WASD key latches and the camera flies forever.
                    if !pressed {
                        if let Some(renderer) = &mut self.renderer {
                            renderer
                                .camera_controller
                                .process_keyboard(key, event.state);
                        }
                    }

                    if !egui_consumed && pressed {
                        // Command chords are editor shortcuts, not fly
                        // input: the 'S' in Ctrl+S would dolly the camera
                        // while held, so the press is dropped.
                        let command_chord =
                            self.modifiers.control_key() || self.modifiers.super_key();
                        if !command_chord {
                            if let Some(renderer) = &mut self.renderer {
                                renderer
                                    .camera_controller
                                    .process_keyboard(key, event.state);
                            }
                        }

                        self.handle_tool_shortcut(key);

                        if key == KeyCode::Escape {
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

                // A press only acts when egui didn't take it; a release
                // always runs, or a button let go over a panel strands
                // the gesture and the next click resumes it.
                if pressed && !egui_consumed {
                    // Middle-press re-anchors the orbit pivot on the
                    // camera's forward hit. Must precede
                    // `process_mouse_button`, which reads the new target.
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
                        // re-applies on motion. Shape / Select enter
                        // their gesture states and finish on release.
                        self.on_left_press();
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
                        // Finish whatever the press started, dispatched on
                        // the gesture state itself — so a release over a
                        // panel tears it down the same way.
                        self.on_left_release();
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
                    // Compute the zoom anchor before taking the mutable
                    // renderer borrow. Scaling around it migrates the
                    // target, so a later orbit circles what was zoomed into.
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

                    // Drag-paint: re-apply the brush whenever the hover
                    // crosses into a new cell. Brush tools only — Fill
                    // would explode the history — with a pixel dead-zone.
                    if let super::EditInteraction::BrushStroke {
                        last_voxel,
                        start_screen: (sx, sy),
                        ..
                    } = &self.interaction
                    {
                        let (last_voxel, sx, sy) = (*last_voxel, *sx, *sy);
                        let drag_eligible = matches!(
                            self.effective_tool(),
                            Tool::Place | Tool::Remove | Tool::Paint
                        );
                        let past_dead_zone = {
                            let dx = self.cursor_pos.0 - sx;
                            let dy = self.cursor_pos.1 - sy;
                            dx * dx + dy * dy >= DRAG_THRESHOLD_PX_SQ
                        };
                        if drag_eligible && past_dead_zone {
                            let current = self.editor.hovered_voxel.map(|h| h.voxel_pos);
                            if current.is_some() && current != last_voxel {
                                self.apply_tool();
                                if let super::EditInteraction::BrushStroke { last_voxel, .. } =
                                    &mut self.interaction
                                {
                                    *last_voxel = current;
                                }
                            }
                        }
                    }

                    // Skip the windowed motion path while the cursor is
                    // captured: the raw `DeviceEvent` path drives orbit
                    // then, and running both doubles the speed.
                    if !self.cursor_captured {
                        if let Some(renderer) = &mut self.renderer {
                            renderer.camera_controller.process_mouse_motion(
                                position.x as f32,
                                position.y as f32,
                                &mut renderer.camera,
                            );
                        }
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
                // Before the mesh rebuild below, so a batch that lands
                // this frame is on screen in the same frame the agent is
                // told it landed.
                self.tick_agent_bridge();
                self.update_brush_preview();
                self.update_selection_visualization();
                self.update_socket_visualization();
                self.rebuild_all_meshes();
                self.tick_autosave();
                self.tick_disk_reload();
                self.render_frame(dt);

                // `render_frame` drains the action queue, so an Exit
                // lands here. The loop can only be stopped from an
                // `ActiveEventLoop`, which exists in this callback.
                if self.exit_requested {
                    self.shutdown(event_loop);
                    return;
                }

                // Schedule rather than request: `about_to_wait` arms the
                // redraw when this deadline passes. An unconditional
                // re-request renders a motionless scene at full rate.
                self.next_frame_at = Instant::now() + self.frame_interval();
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
                // The windowed path is gated off while captured, so this
                // is the only stream proving the user is mid-orbit —
                // unstamped, the scheduler reads a long orbit as idle.
                self.last_interaction = Instant::now();
                if let Some(renderer) = &mut self.renderer {
                    // Raw motion is the sole orbit path while captured, so
                    // the two never double-count. Through `orbit_by`, so
                    // the sensitivity math lives in one place.
                    renderer.camera_controller.orbit_by(
                        delta.0 as f32,
                        delta.1 as f32,
                        &mut renderer.camera,
                    );
                }
            }
        }
    }
}
