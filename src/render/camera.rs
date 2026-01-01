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

    /// Handle mouse button input
    pub fn process_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        let pressed = state == ElementState::Pressed;
        match button {
            MouseButton::Right => self.right_mouse_pressed = pressed,
            MouseButton::Middle => self.middle_mouse_pressed = pressed,
            _ => {}
        }

        if !pressed {
            self.last_mouse_pos = None;
        }
    }

    /// Handle mouse movement
    pub fn process_mouse_motion(&mut self, x: f32, y: f32, camera: &mut Camera) {
        if let Some((last_x, last_y)) = self.last_mouse_pos {
            let dx = x - last_x;
            let dy = y - last_y;

            if self.middle_mouse_pressed {
                // Orbit around target
                self.yaw -= dx * self.sensitivity;
                self.pitch -= dy * self.sensitivity;

                // Clamp pitch to avoid flipping
                self.pitch = self.pitch.clamp(-1.5, 1.5);

                self.update_camera_position(camera);
            } else if self.right_mouse_pressed {
                // Pan camera
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

    /// Update camera position based on orbital parameters
    fn update_camera_position(&self, camera: &mut Camera) {
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
        if self.pressed_keys.contains(&KeyCode::KeyQ) || self.pressed_keys.contains(&KeyCode::Space) {
            movement += Vec3::Y;
        }
        if self.pressed_keys.contains(&KeyCode::KeyE) || self.pressed_keys.contains(&KeyCode::ShiftLeft) {
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
