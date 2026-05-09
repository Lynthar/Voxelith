//! Camera and camera controller for 3D navigation.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use winit::event::{ElementState, MouseButton, MouseScrollDelta};
use winit::keyboard::KeyCode;
use std::collections::HashSet;

/// Camera uniform data for GPU
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct CameraUniform {
    /// View-projection matrix
    pub view_proj: [[f32; 4]; 4],
    /// Camera position in world space
    pub camera_pos: [f32; 4],
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0; 4],
        }
    }
}

/// 3D camera with orbital controls
pub struct Camera {
    /// Camera position
    pub position: Vec3,
    /// Point the camera is looking at
    pub target: Vec3,
    /// Up vector
    pub up: Vec3,
    /// Aspect ratio (width / height)
    pub aspect: f32,
    /// Field of view in radians
    pub fov: f32,
    /// Near clipping plane
    pub near: f32,
    /// Far clipping plane
    pub far: f32,
}

impl Camera {
    /// Create a new camera
    pub fn new(position: Vec3, target: Vec3, aspect: f32) -> Self {
        Self {
            position,
            target,
            up: Vec3::Y,
            aspect,
            fov: 45.0_f32.to_radians(),
            near: 0.1,
            far: 1000.0,
        }
    }

    /// Build the view matrix
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    /// Build the projection matrix
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov, self.aspect, self.near, self.far)
    }

    /// Build combined view-projection matrix
    pub fn view_projection_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    /// Get camera uniform for GPU
    pub fn uniform(&self) -> CameraUniform {
        CameraUniform {
            view_proj: self.view_projection_matrix().to_cols_array_2d(),
            camera_pos: [self.position.x, self.position.y, self.position.z, 1.0],
        }
    }

    /// Get the forward direction
    pub fn forward(&self) -> Vec3 {
        (self.target - self.position).normalize()
    }

    /// Get the right direction
    pub fn right(&self) -> Vec3 {
        self.forward().cross(self.up).normalize()
    }
}

/// Camera controller for mouse/keyboard input
pub struct CameraController {
    /// Movement speed
    pub speed: f32,
    /// Mouse sensitivity for rotation
    pub sensitivity: f32,
    /// Current orbital distance from target
    pub distance: f32,
    /// Horizontal angle (yaw) in radians
    pub yaw: f32,
    /// Vertical angle (pitch) in radians
    pub pitch: f32,
    /// Currently pressed keys
    pressed_keys: HashSet<KeyCode>,
    /// Is right mouse button pressed (for panning)
    right_mouse_pressed: bool,
    /// Is middle mouse button pressed (for orbiting)
    middle_mouse_pressed: bool,
    /// Last mouse position
    last_mouse_pos: Option<(f32, f32)>,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            speed,
            sensitivity,
            distance: 40.0,
            yaw: 0.0,
            pitch: 0.5, // Look slightly down
            pressed_keys: HashSet::new(),
            right_mouse_pressed: false,
            middle_mouse_pressed: false,
            last_mouse_pos: None,
        }
    }

    /// Handle keyboard input
    pub fn process_keyboard(&mut self, key: KeyCode, state: ElementState) {
        match state {
            ElementState::Pressed => {
                self.pressed_keys.insert(key);
            }
            ElementState::Released => {
                self.pressed_keys.remove(&key);
            }
        }
    }

    /// Handle mouse button input.
    ///
    /// Takes `&mut Camera` so middle-press can sync orbit state from
    /// the camera's current position before any orbit motion runs.
    /// Without this sync, anything that wrote `target` / yaw / pitch /
    /// distance without also writing `camera.position` (Reset Camera,
    /// Set Camera View) would cause the next orbit drag to snap the
    /// camera to a stale spherical position — visible teleport.
    pub fn process_mouse_button(
        &mut self,
        button: MouseButton,
        state: ElementState,
        camera: &mut Camera,
    ) {
        let pressed = state == ElementState::Pressed;
        match button {
            MouseButton::Right => self.right_mouse_pressed = pressed,
            MouseButton::Middle => {
                self.middle_mouse_pressed = pressed;
                if pressed {
                    self.sync_orbit_state_from_camera(camera);
                }
            }
            _ => {}
        }

        if !pressed {
            self.last_mouse_pos = None;
        }
    }

    /// Re-derive `yaw` / `pitch` / `distance` from the current
    /// `camera.position - camera.target` vector. Treats the camera's
    /// actual position as the source of truth; the controller's
    /// stored angles are just a cache used to apply orbit deltas.
    ///
    /// Call this whenever the camera's position has been changed by
    /// something other than the orbit/zoom path (currently only
    /// middle-press; the other gestures — pan, WASD — already preserve
    /// the camera-target offset by translating both endpoints together).
    pub fn sync_orbit_state_from_camera(&mut self, camera: &Camera) {
        let to_camera = camera.position - camera.target;
        // Floor avoids `to_camera / 0` when camera sits exactly at
        // target (degenerate — distance becomes 0.01 instead of NaN).
        let dist = to_camera.length().max(0.01);
        self.distance = dist;
        let dir = to_camera / dist;
        self.yaw = dir.z.atan2(dir.x);
        // Clamp into the same range orbit motion uses so a first drag
        // after sync doesn't immediately hit the clamp boundary.
        self.pitch = dir.y.asin().clamp(-1.5, 1.5);
    }

    /// Handle mouse movement
    pub fn process_mouse_motion(&mut self, x: f32, y: f32, camera: &mut Camera) {
        if let Some((last_x, last_y)) = self.last_mouse_pos {
            let dx = x - last_x;
            let dy = y - last_y;

            if self.middle_mouse_pressed {
                // Orbit: drag-the-scene direction. Dragging down rolls
                // the camera up (you see more of the top), dragging
                // right swings the camera around to view the right
                // side of the scene. Inverted from the camera-relative
                // convention where dragging moves the camera itself.
                self.yaw += dx * self.sensitivity;
                self.pitch += dy * self.sensitivity;

                // Clamp pitch to avoid flipping
                self.pitch = self.pitch.clamp(-1.5, 1.5);

                self.update_camera_position(camera);
            } else if self.right_mouse_pressed {
                // Pan camera. Both position and target shift by the same
                // offset so the view direction and the camera-to-target
                // vector both stay fixed — orbit angles derived from
                // that vector remain valid, so the next orbit gesture
                // continues to rotate around the new (panned-to) target
                // without any discontinuity.
                let right = camera.right();
                let up = camera.up;
                let pan_speed = self.distance * 0.002;

                let offset = right * (-dx * pan_speed) + up * (dy * pan_speed);
                camera.position += offset;
                camera.target += offset;
            }
        }

        self.last_mouse_pos = Some((x, y));
    }

    /// Handle mouse scroll (zoom)
    pub fn process_scroll(&mut self, delta: MouseScrollDelta, camera: &mut Camera) {
        let scroll = match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
        };

        self.distance -= scroll * self.distance * 0.1;
        self.distance = self.distance.clamp(1.0, 500.0);

        self.update_camera_position(camera);
    }

    /// Write `camera.position` from the controller's current
    /// `yaw` / `pitch` / `distance` (relative to `camera.target`).
    /// Public so callers that change those fields directly (e.g.
    /// Reset Camera, Set Camera View) can apply the change immediately
    /// instead of leaving `camera.position` desynced until the next
    /// orbit drag.
    pub fn update_camera_position(&self, camera: &mut Camera) {
        let x = self.distance * self.yaw.cos() * self.pitch.cos();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.yaw.sin() * self.pitch.cos();

        camera.position = camera.target + Vec3::new(x, y, z);
    }

    /// Update camera based on keyboard input (called each frame)
    pub fn update(&mut self, camera: &mut Camera, dt: f32) {
        let mut movement = Vec3::ZERO;
        let forward = camera.forward();
        let right = camera.right();

        // WASD movement
        if self.pressed_keys.contains(&KeyCode::KeyW) {
            movement += forward;
        }
        if self.pressed_keys.contains(&KeyCode::KeyS) {
            movement -= forward;
        }
        if self.pressed_keys.contains(&KeyCode::KeyA) {
            movement -= right;
        }
        if self.pressed_keys.contains(&KeyCode::KeyD) {
            movement += right;
        }
        if self.pressed_keys.contains(&KeyCode::KeyQ) {
            movement += Vec3::Y;
        }
        if self.pressed_keys.contains(&KeyCode::KeyE) {
            movement -= Vec3::Y;
        }

        if movement != Vec3::ZERO {
            let speed = if self.pressed_keys.contains(&KeyCode::ControlLeft) {
                self.speed * 3.0
            } else {
                self.speed
            };

            let offset = movement.normalize() * speed * dt * self.distance;
            camera.position += offset;
            camera.target += offset;
        }
    }
}
