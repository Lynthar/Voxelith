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

/// The window icon (title bar, Alt-Tab, the running taskbar button),
/// decoded from the 64 px plated icon in assets/branding — same
/// artwork the exe resource icon is built from (build.rs), so the
/// pinned and the running taskbar item look identical. 64 px because
/// the places Windows shows *this* icon are small; feeding it the
/// 256 px master just makes the title bar blurrier. `None` on decode
/// failure: a missing icon is not worth failing startup over.
fn window_icon() -> Option<winit::window::Icon> {
    let bytes = include_bytes!("../../assets/branding/icon_64.png");
    let img = image::load_from_memory(bytes).ok()?.into_rgba8();
    let (w, h) = img.dimensions();
    winit::window::Icon::from_rgba(img.into_raw(), w, h).ok()
}

impl App {
    /// Persist prefs, drop the crash-recovery autosave (a clean exit
    /// means the next launch shouldn't offer recovery), and stop the
    /// event loop. Shared by the window's close button and the File ▸
    /// Exit menu item so the two can't drift apart.
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
            // The prefs entry can outlive the display it was sized on
            // (docked 4K → laptop panel), so check it against the
            // monitor before asking for it. `primary_monitor` is the
            // best guess available: the window doesn't exist yet, so
            // there's nothing to ask which display it landed on. None
            // (Wayland, headless) simply skips the check.
            let (w, h) = match event_loop.primary_monitor() {
                Some(monitor) => {
                    let logical: winit::dpi::LogicalSize<u32> =
                        monitor.size().to_logical(monitor.scale_factor());
                    super::fit_window_to_monitor(
                        saved,
                        (logical.width, logical.height),
                    )
                }
                None => saved,
            };
            let window_attrs = Window::default_attributes()
                .with_title("Voxelith")
                .with_window_icon(window_icon())
                .with_inner_size(winit::dpi::LogicalSize::new(w, h));

            let window = event_loop.create_window(window_attrs).unwrap();
            self.init(window);
            // Kick the first frame. `RedrawRequested` re-arms itself
            // through the scheduler, but the chain has to start
            // somewhere: Windows happens to deliver an initial WM_PAINT,
            // macOS only draws when asked — without this the window sat
            // on the compositor's blank backing store forever, and even
            // moving / resizing it never produced a first frame.
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // The frame scheduler. `RedrawRequested` sets `next_frame_at`
        // (immediately while the user is active, an idle heartbeat
        // apart otherwise — see `App::frame_interval`); this arms the
        // redraw when it comes due and parks the loop until then.
        // Input wakes the loop regardless of the deadline, so a key
        // press lands instantly however deep in an idle stretch it
        // arrives.
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
                // Route through the unsaved-changes guard: a clean exit
                // deletes the crash-recovery autosave, so closing with
                // unsaved work used to destroy the only copy of it
                // without a word. If the guard defers, it raises the
                // prompt and `exit_requested` stays false until the
                // user answers.
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
                // Losing focus (alt-tab, or a modal Save/Open dialog
                // taking over) means press/release events can be delivered
                // elsewhere. Abandon EVERY in-progress interaction so none
                // resumes latched when focus returns — mirror the mouse-
                // release + deselect cleanup: forget held keys and mouse
                // buttons, drop the brush-stroke / shape latches, release
                // the orbit cursor capture, and clear any select-drag /
                // move-drag anchor + ghost. The committed selection marquee
                // itself is left intact, exactly like a plain mouse release.
                if let Some(renderer) = &mut self.renderer {
                    renderer.camera_controller.clear_keys();
                    renderer.camera_controller.clear_mouse_buttons();
                }
                self.left_button_held = false;
                self.editor.history.end_stroke();
                self.last_stroke_voxel = None;
                self.stroke_start_screen_pos = None;
                self.stroke_plane = None;
                self.shape_drag = None;
                self.selection_drag_anchor = None;
                self.selection_move_anchor = None;
                self.move_ghost_voxels.clear();
                // Modifier state is only refreshed by ModifiersChanged,
                // which is delivered to whoever has focus — so an
                // alt-tab away leaves `modifiers` claiming Alt is still
                // down and (if Alt swapped the tool) the eyedropper
                // latched in. Reset both, mirroring the alt-release arm.
                self.modifiers = Default::default();
                if let Some(tool) = self.editor.tool_before_alt.take() {
                    self.editor.current_tool = tool;
                }
                if self.cursor_captured {
                    self.cursor_captured = false;
                    if let Some(window) = &self.window {
                        window.set_cursor_visible(true);
                    }
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
                if let PhysicalKey::Code(key) = event.physical_key {
                    let pressed = event.state.is_pressed();

                    // The camera-key RELEASE must ALWAYS reach the
                    // controller, even when egui consumed the event (a key
                    // let go while a panel has focus, or after a modal
                    // Save/Open dialog grabbed it). Otherwise a held WASD
                    // key latches "down" and the camera flies forever.
                    // Everything below stays gated on `!egui_consumed` so
                    // typing in a panel neither moves the camera nor fires
                    // tool shortcuts.
                    if !pressed {
                        if let Some(renderer) = &mut self.renderer {
                            renderer
                                .camera_controller
                                .process_keyboard(key, event.state);
                        }
                    }

                    if !egui_consumed && pressed {
                        // Command chords (Ctrl/Super + key) are editor
                        // shortcuts, not fly-camera input. Feeding the
                        // chord's letter (e.g. the 'S' in Ctrl+S) to the
                        // controller would dolly the camera while held, so
                        // drop the *press* while a command modifier is down.
                        // (The matching release is forwarded above, so a key
                        // pressed before the modifier — hold W, then tap
                        // Ctrl — never sticks.) Sprint lives on Shift, not a
                        // command modifier, so Shift+WASD is unaffected.
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
                            // Seal the stroke unconditionally, before
                            // the per-tool dispatch. That dispatch asks
                            // "what tool is active *now*", but the
                            // gesture was started by whatever tool was
                            // active at press time — switch with a
                            // number key mid-drag and the stroke would
                            // never get sealed, so the next stroke
                            // within the 200 ms merge window silently
                            // joined the previous undo entry (one
                            // Ctrl+Z, two separate strokes gone). It's
                            // a no-op when no stroke is open, and the
                            // shape / select paths commit via `execute`,
                            // which reseals anyway.
                            self.editor.history.end_stroke();
                            let tool = self.editor.current_tool;
                            if tool.is_shape() {
                                self.transition_shape_to_height();
                            } else if matches!(tool, Tool::Select) {
                                self.commit_selection();
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

                    // Skip the windowed motion path while the cursor is
                    // captured (middle-orbit): the raw `DeviceEvent` path
                    // drives orbit then, and running both double-counts it
                    // (2× orbit speed in-window). Right-button pan doesn't
                    // capture, so it still flows through here.
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

                // `render_frame` drains the UI action queue, so an Exit
                // (or an unsaved-changes prompt answered with Discard /
                // Save) lands here. The event loop can only be stopped
                // from an `ActiveEventLoop`, which exists in this
                // callback and not inside `App`.
                if self.exit_requested {
                    self.shutdown(event_loop);
                    return;
                }

                // Schedule rather than request: `about_to_wait` arms the
                // actual redraw when this deadline passes. Immediate
                // while the user is active (vsync paces the real rate),
                // a 100 ms heartbeat when idle — the unconditional
                // re-request this replaces rendered a motionless scene
                // at 144 fps for as long as the app was open.
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
                // The windowed CursorMoved path is gated off while
                // captured, so this is the only event stream proving
                // the user is mid-orbit — stamp it or the scheduler
                // would read a long smooth orbit as idleness.
                self.last_interaction = Instant::now();
                if let Some(renderer) = &mut self.renderer {
                    // Raw motion is the SOLE orbit path while captured (the
                    // windowed `CursorMoved` path is gated off then), so the
                    // two no longer double-count. Reuse `orbit_by` so the
                    // sensitivity + spherical-position math live in one
                    // place instead of a hardcoded 0.003 duplicate.
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
