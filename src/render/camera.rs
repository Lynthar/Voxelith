//! Camera and camera controller for 3D navigation.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::collections::HashSet;
use winit::event::{ElementState, MouseButton, MouseScrollDelta};
use winit::keyboard::KeyCode;

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

    /// Distance from an AABB's center at which its bounding sphere just
    /// fills the view, scaled by `margin`. A sphere, so the fit holds at
    /// any angle; the tighter half-FOV is the binding constraint.
    pub fn fit_distance(&self, extent: Vec3, margin: f32) -> f32 {
        let radius = extent.length() * 0.5;
        let half_v = self.fov * 0.5;
        // Horizontal half-angle derived from the vertical FOV + aspect.
        let half_h = (half_v.tan() * self.aspect).atan();
        let half = half_v.min(half_h).max(1e-3);
        radius / half.sin() * margin
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

    /// Forget all held keys. Called on focus loss, so a key whose
    /// release went to another window can't leave the fly camera
    /// drifting when focus returns.
    pub fn clear_keys(&mut self) {
        self.pressed_keys.clear();
    }

    /// Whether camera input is live: a movement key held, or a button
    /// driving orbit or pan. The frame scheduler renders at full rate
    /// while it is, since flying is integrated per frame.
    pub fn is_navigating(&self) -> bool {
        !self.pressed_keys.is_empty() || self.right_mouse_pressed || self.middle_mouse_pressed
    }

    /// Forget any in-progress drag, so a button whose release went to
    /// another window can't leave orbit or pan latched. Clearing
    /// `last_mouse_pos` also re-seeds the next delta instead of jumping.
    pub fn clear_mouse_buttons(&mut self) {
        self.middle_mouse_pressed = false;
        self.right_mouse_pressed = false;
        self.last_mouse_pos = None;
    }

    /// Handle mouse button input. Takes `&mut Camera` so middle-press
    /// can sync orbit state from the current position first, or the next
    /// drag snaps the camera to a stale spherical pose.
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

    /// Re-derive `yaw` / `pitch` / `distance` from the camera's actual
    /// pose, which is the source of truth — the stored angles are a
    /// cache for applying orbit deltas.
    ///
    /// # Safety
    /// Every write to `camera.target` or `camera.position` that skips
    /// `update_camera_position` must be followed by a call to this.
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

    /// Apply an orbit delta in raw pixels around the current target.
    /// The single orbit implementation for both the windowed and raw
    /// motion paths — driving both per frame orbits at double speed.
    pub fn orbit_by(&mut self, dx: f32, dy: f32, camera: &mut Camera) {
        self.yaw += dx * self.sensitivity;
        self.pitch += dy * self.sensitivity;
        // Clamp pitch to avoid flipping past the poles.
        self.pitch = self.pitch.clamp(-1.5, 1.5);
        self.update_camera_position(camera);
    }

    /// Handle mouse movement
    pub fn process_mouse_motion(&mut self, x: f32, y: f32, camera: &mut Camera) {
        if let Some((last_x, last_y)) = self.last_mouse_pos {
            let dx = x - last_x;
            let dy = y - last_y;

            if self.middle_mouse_pressed {
                // Orbit in the drag-the-scene direction: dragging down
                // rolls the camera up. Inverted from the camera-relative
                // convention, and shared with raw motion via `orbit_by`.
                self.orbit_by(dx, dy, camera);
            } else if self.right_mouse_pressed {
                // Pan: position and target shift by the same offset, so
                // the camera-to-target vector is unchanged and the
                // cached orbit angles stay valid.
                let right = camera.right();
                // Screen-space up, not world up: against world `up` the
                // visible motion scales with cos(pitch), so near the
                // clamp most of a vertical drag became a dolly.
                let up = right.cross(camera.forward());
                let pan_speed = self.distance * 0.002;

                let offset = right * (-dx * pan_speed) + up * (dy * pan_speed);
                camera.position += offset;
                camera.target += offset;
            }
        }

        self.last_mouse_pos = Some((x, y));
    }

    /// Zoom to cursor: scale position and target uniformly around
    /// `anchor`, so it stays fixed on screen and the target migrates to
    /// the zoomed-in feature. The view direction is preserved.
    pub fn process_scroll(&mut self, delta: MouseScrollDelta, camera: &mut Camera, anchor: Vec3) {
        let scroll = match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
        };

        // Wheel up zooms in. Clamped so one huge event can't swallow the
        // range: a touchpad flick arrives as a single ≥100 px event,
        // which unclamped snaps straight to the minimum distance.
        let f = (1.0 - scroll * 0.1).clamp(0.2, 5.0);
        let new_distance = (self.distance * f).clamp(1.0, 500.0);
        // After clamp the actual factor may differ from `f`; use the
        // ratio so position / target scale by exactly the amount the
        // distance ended up changing.
        let actual_f = new_distance / self.distance.max(1e-6);

        camera.position = anchor + (camera.position - anchor) * actual_f;
        camera.target = anchor + (camera.target - anchor) * actual_f;
        self.distance = new_distance;
    }

    /// Write `camera.position` from the controller's current angles and
    /// distance. Public so callers that set those directly can apply the
    /// change immediately rather than leaving the pose desynced.
    pub fn update_camera_position(&self, camera: &mut Camera) {
        let x = self.distance * self.yaw.cos() * self.pitch.cos();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.yaw.sin() * self.pitch.cos();

        camera.position = camera.target + Vec3::new(x, y, z);
    }

    #[cfg(test)]
    /// Test-only constructor that builds a controller pre-synced to a
    /// camera pose. Mirrors what `Renderer::new` does but without the
    /// surrounding wgpu setup.
    fn new_synced_for_test(camera: &Camera) -> Self {
        let mut c = Self::new(0.5, 0.003);
        c.sync_orbit_state_from_camera(camera);
        c
    }

    /// Apply keyboard navigation for one frame. WASD moves on the world
    /// horizontal plane whatever the pitch, so a straight-down view
    /// doesn't collapse W/S into Q/E; Q/E stay unconditional ±Y.
    pub fn update(&mut self, camera: &mut Camera, dt: f32) {
        let mut movement = Vec3::ZERO;
        // Horizontal forward from yaw — independent of pitch, so
        // looking straight down doesn't degenerate W/S to Y motion.
        let forward_xz = Vec3::new(-self.yaw.cos(), 0.0, -self.yaw.sin());
        // Right-handed: right_xz = forward_xz × world_up.
        let right_xz = Vec3::new(self.yaw.sin(), 0.0, -self.yaw.cos());

        if self.pressed_keys.contains(&KeyCode::KeyW) {
            movement += forward_xz;
        }
        if self.pressed_keys.contains(&KeyCode::KeyS) {
            movement -= forward_xz;
        }
        if self.pressed_keys.contains(&KeyCode::KeyA) {
            movement -= right_xz;
        }
        if self.pressed_keys.contains(&KeyCode::KeyD) {
            movement += right_xz;
        }
        if self.pressed_keys.contains(&KeyCode::KeyQ) {
            movement += Vec3::Y;
        }
        if self.pressed_keys.contains(&KeyCode::KeyE) {
            movement -= Vec3::Y;
        }

        if movement != Vec3::ZERO {
            // Shift flies 3× faster. Not Ctrl: that is the command
            // modifier, and the handler drops Ctrl chords before they
            // reach the controller.
            let sprint = self.pressed_keys.contains(&KeyCode::ShiftLeft)
                || self.pressed_keys.contains(&KeyCode::ShiftRight);
            let speed = if sprint { self.speed * 3.0 } else { self.speed };

            let offset = movement.normalize() * speed * dt * self.distance;
            camera.position += offset;
            camera.target += offset;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_then_update_position_round_trips() {
        // With a controller whose defaults don't match the camera's
        // pose, `update_camera_position` teleports. After a sync the
        // round trip must leave the position unchanged.
        let mut camera = Camera::new(Vec3::new(0.0, 20.0, 40.0), Vec3::ZERO, 1.0);
        let original = camera.position;
        let mut controller = CameraController::new(0.5, 0.003);
        controller.sync_orbit_state_from_camera(&camera);
        controller.update_camera_position(&mut camera);
        assert!(
            (camera.position - original).length() < 1e-3,
            "sync→update should preserve position; got {:?} expected {:?}",
            camera.position,
            original
        );
    }

    #[test]
    fn sync_round_trips_for_arbitrary_poses() {
        // Sanity over a variety of poses: every sync→update preserves
        // the camera-target vector. Skip degenerate poses where the
        // camera is on the Y axis (yaw becomes ill-defined).
        let cases = [
            (Vec3::new(10.0, 5.0, 7.0), Vec3::ZERO),
            (Vec3::new(-15.0, 30.0, 8.0), Vec3::new(2.0, 1.0, -3.0)),
            (Vec3::new(50.0, 0.5, 0.0), Vec3::ZERO),
        ];
        for (pos, target) in cases {
            let mut camera = Camera::new(pos, target, 1.0);
            let original = camera.position;
            let mut controller = CameraController::new(0.5, 0.003);
            controller.sync_orbit_state_from_camera(&camera);
            controller.update_camera_position(&mut camera);
            assert!(
                (camera.position - original).length() < 1e-3,
                "round-trip failed for pos={:?} target={:?}",
                pos,
                target
            );
        }
    }

    #[test]
    fn post_sync_update_preserves_view_direction() {
        // Any path calling `update_camera_position` must use synced
        // state, or a stale yaw/pitch teleports the camera. After a
        // sync, changing distance must keep the view direction.
        let mut camera = Camera::new(Vec3::new(0.0, 20.0, 40.0), Vec3::ZERO, 1.0);
        let mut controller = CameraController::new_synced_for_test(&camera);
        let original_dir = (camera.position - camera.target).normalize();

        controller.distance -= 1.0 * controller.distance * 0.1;
        controller.update_camera_position(&mut camera);

        let new_dir = (camera.position - camera.target).normalize();
        assert!(
            original_dir.dot(new_dir) > 0.9999,
            "post-sync update changed view direction; \
             original={:?} new={:?}",
            original_dir,
            new_dir
        );
    }

    #[test]
    fn pre_sync_update_would_teleport_demonstrating_bug_1() {
        // Why the sync at construction is needed: the controller's
        // defaults don't match the initial camera pose, so
        // `update_camera_position` would write the wrong direction.
        let mut camera = Camera::new(Vec3::new(0.0, 20.0, 40.0), Vec3::ZERO, 1.0);
        let original_dir = (camera.position - camera.target).normalize();
        let controller = CameraController::new(0.5, 0.003); // NOT synced
        controller.update_camera_position(&mut camera);
        let new_dir = (camera.position - camera.target).normalize();
        // Without sync, the "view direction" jumps from (0, ~0.45, ~0.89)
        // to (~0.88, ~0.48, 0) — way more than rounding error.
        assert!(
            original_dir.dot(new_dir) < 0.5,
            "expected pre-sync update to produce a divergent direction \
             (this test demonstrates the bug fix is load-bearing); \
             got original={:?} new={:?}",
            original_dir,
            new_dir
        );
    }

    // -------- zoom-to-cursor (process_scroll) --------

    fn line_scroll(y: f32) -> winit::event::MouseScrollDelta {
        winit::event::MouseScrollDelta::LineDelta(0.0, y)
    }

    #[test]
    fn zoom_in_at_target_anchor_matches_legacy_scale_around_target() {
        // When the anchor IS the current target, zoom-to-cursor
        // degenerates to the original "scale around target" behavior:
        // target stays put, position approaches target by factor f.
        let mut camera = Camera::new(Vec3::new(0.0, 20.0, 40.0), Vec3::ZERO, 1.0);
        let mut controller = CameraController::new_synced_for_test(&camera);
        let anchor = camera.target;
        let original_target = camera.target;
        let original_dir = (camera.position - camera.target).normalize();
        let original_dist = controller.distance;

        controller.process_scroll(line_scroll(1.0), &mut camera, anchor);

        // f = 1 - 1*0.1 = 0.9; new_distance = 0.9 * original.
        let expected_dist = 0.9 * original_dist;
        assert!(
            (controller.distance - expected_dist).abs() < 1e-3,
            "expected distance ~{:.4}, got {:.4}",
            expected_dist,
            controller.distance
        );
        assert!(
            (camera.target - original_target).length() < 1e-4,
            "anchor=target → target should stay put; moved to {:?}",
            camera.target
        );
        let new_dir = (camera.position - camera.target).normalize();
        assert!(original_dir.dot(new_dir) > 0.9999, "direction changed");
    }

    #[test]
    fn zoom_in_keeps_anchor_fixed_relative_to_camera_direction() {
        // The defining property of zoom-to-cursor: the anchor keeps the
        // same screen direction, so `anchor - new_camera` is a positive
        // multiple of `anchor - old_camera`.
        let mut camera = Camera::new(Vec3::new(0.0, 20.0, 40.0), Vec3::ZERO, 1.0);
        let mut controller = CameraController::new_synced_for_test(&camera);
        let anchor = Vec3::new(5.0, 8.0, 5.0); // somewhere off-axis
        let old_to_anchor = anchor - camera.position;

        controller.process_scroll(line_scroll(1.0), &mut camera, anchor);

        let new_to_anchor = anchor - camera.position;
        // Both vectors point in the same direction (uniform scale
        // around anchor preserves direction from camera to anchor).
        let cos = old_to_anchor.normalize().dot(new_to_anchor.normalize());
        assert!(
            cos > 0.9999,
            "anchor screen direction changed; old={:?} new={:?} cos={}",
            old_to_anchor,
            new_to_anchor,
            cos
        );
        // And new_to_anchor is shorter (zoomed in).
        assert!(
            new_to_anchor.length() < old_to_anchor.length(),
            "zoom-in should shorten camera-to-anchor distance"
        );
    }

    #[test]
    fn zoom_preserves_camera_target_direction_so_yaw_pitch_stay_valid() {
        // A uniform scale around any anchor preserves the direction
        // `position - target`, which is what lets `process_scroll` skip
        // a sync.
        let mut camera = Camera::new(Vec3::new(0.0, 20.0, 40.0), Vec3::ZERO, 1.0);
        let mut controller = CameraController::new_synced_for_test(&camera);
        let anchor = Vec3::new(7.0, -3.0, 12.0);
        let old_dir = (camera.position - camera.target).normalize();
        let old_yaw = controller.yaw;
        let old_pitch = controller.pitch;

        controller.process_scroll(line_scroll(1.0), &mut camera, anchor);

        let new_dir = (camera.position - camera.target).normalize();
        assert!(
            old_dir.dot(new_dir) > 0.9999,
            "scaling around off-axis anchor should preserve view direction"
        );
        // process_scroll doesn't touch yaw/pitch — direction-preserving
        // scale leaves them valid.
        assert_eq!(controller.yaw, old_yaw);
        assert_eq!(controller.pitch, old_pitch);
    }

    #[test]
    fn zoom_out_grows_distance_and_pushes_camera_away_from_anchor() {
        // Symmetric to the zoom-in case: scroll-down (negative scroll)
        // should grow distance and push camera away from anchor.
        let mut camera = Camera::new(Vec3::new(0.0, 20.0, 40.0), Vec3::ZERO, 1.0);
        let mut controller = CameraController::new_synced_for_test(&camera);
        let anchor = Vec3::new(5.0, 5.0, 5.0);
        let old_dist = controller.distance;
        let old_cam_to_anchor = (anchor - camera.position).length();

        controller.process_scroll(line_scroll(-1.0), &mut camera, anchor);

        // f = 1 - (-1)*0.1 = 1.1, so distance grows.
        let expected_dist = 1.1 * old_dist;
        assert!(
            (controller.distance - expected_dist).abs() < 1e-3,
            "expected distance ~{:.4}, got {:.4}",
            expected_dist,
            controller.distance
        );
        let new_cam_to_anchor = (anchor - camera.position).length();
        assert!(
            new_cam_to_anchor > old_cam_to_anchor,
            "zoom out should move camera further from anchor"
        );
    }

    // -------- FPS-ground WASD (update) --------

    fn pressed(controller: &mut CameraController, key: KeyCode) {
        controller.pressed_keys.insert(key);
    }

    #[test]
    fn wasd_top_down_view_does_not_collapse_w_to_y() {
        // Looking straight down, `camera.forward()` is ~(0, -1, 0), so
        // driving W off it would duplicate E. W must move on the X-Z
        // plane whatever the pitch.
        let mut camera = Camera::new(Vec3::new(0.0, 50.0, 0.001), Vec3::ZERO, 1.0);
        let mut controller = CameraController::new_synced_for_test(&camera);
        let pre_pos_y = camera.position.y;
        let pre_target_y = camera.target.y;

        pressed(&mut controller, KeyCode::KeyW);
        controller.update(&mut camera, 1.0 / 60.0);

        // No vertical motion — W is purely horizontal.
        assert!(
            (camera.position.y - pre_pos_y).abs() < 1e-4,
            "W moved camera vertically by {}; expected horizontal-only",
            camera.position.y - pre_pos_y
        );
        assert!(
            (camera.target.y - pre_target_y).abs() < 1e-4,
            "W moved target vertically",
        );
        // And it actually did move on X or Z.
        let xz_delta = (camera.position.x.powi(2) + camera.position.z.powi(2)).sqrt();
        assert!(xz_delta > 1e-3, "W produced no horizontal motion");
    }

    #[test]
    fn wasd_horizontal_only_at_arbitrary_pitch() {
        // For any non-degenerate camera pose, WASD shouldn't change Y.
        let cases = [
            (Vec3::new(0.0, 5.0, 30.0), Vec3::ZERO),
            (Vec3::new(20.0, 30.0, 20.0), Vec3::ZERO),
            (Vec3::new(-10.0, 50.0, 5.0), Vec3::new(0.0, 5.0, 0.0)),
        ];
        for (pos, target) in cases {
            for key in [KeyCode::KeyW, KeyCode::KeyS, KeyCode::KeyA, KeyCode::KeyD] {
                let mut camera = Camera::new(pos, target, 1.0);
                let mut controller = CameraController::new_synced_for_test(&camera);
                let pre_y = camera.position.y;

                pressed(&mut controller, key);
                controller.update(&mut camera, 1.0 / 60.0);

                assert!(
                    (camera.position.y - pre_y).abs() < 1e-3,
                    "{:?} changed Y for pos={:?} target={:?}: {} -> {}",
                    key,
                    pos,
                    target,
                    pre_y,
                    camera.position.y
                );
            }
        }
    }

    #[test]
    fn qe_still_moves_y_unconditionally() {
        // Q goes up, E goes down — independent of camera pose.
        let mut camera = Camera::new(Vec3::new(0.0, 20.0, 40.0), Vec3::ZERO, 1.0);
        let mut controller = CameraController::new_synced_for_test(&camera);
        let pre_y = camera.position.y;

        pressed(&mut controller, KeyCode::KeyQ);
        controller.update(&mut camera, 1.0 / 60.0);
        assert!(camera.position.y > pre_y, "Q should move +Y");

        controller.pressed_keys.remove(&KeyCode::KeyQ);
        let mid_y = camera.position.y;
        pressed(&mut controller, KeyCode::KeyE);
        controller.update(&mut camera, 1.0 / 60.0);
        assert!(camera.position.y < mid_y, "E should move -Y");
    }

    #[test]
    fn w_at_yaw_pi_over_two_moves_negative_z() {
        // Camera at (0, 20, 40) looking at origin → yaw = π/2.
        // forward_xz formula = (-cos(π/2), 0, -sin(π/2)) = (0, 0, -1).
        // Pressing W should translate camera along -Z.
        let mut camera = Camera::new(Vec3::new(0.0, 20.0, 40.0), Vec3::ZERO, 1.0);
        let mut controller = CameraController::new_synced_for_test(&camera);
        let pre_x = camera.position.x;
        let pre_z = camera.position.z;

        pressed(&mut controller, KeyCode::KeyW);
        controller.update(&mut camera, 1.0 / 60.0);

        assert!(
            (camera.position.x - pre_x).abs() < 1e-3,
            "W shouldn't change X for yaw=π/2"
        );
        assert!(
            camera.position.z < pre_z,
            "W should decrease Z for yaw=π/2; was {}, is {}",
            pre_z,
            camera.position.z
        );
    }

    #[test]
    fn zoom_clamped_at_minimum_distance_doesnt_move_camera() {
        // When already at min distance (1.0), further zoom-in should
        // be a no-op — no NaN, no jitter, no target shift.
        let mut camera = Camera::new(Vec3::new(0.0, 0.5, 0.5), Vec3::ZERO, 1.0);
        // |camera - target| ≈ 0.707, but distance clamps to 1.0 at
        // process_scroll boundary. Force-set controller to 1.0 to
        // simulate "already at min".
        let mut controller = CameraController::new_synced_for_test(&camera);
        controller.distance = 1.0;
        let original_pos = camera.position;
        let original_target = camera.target;

        controller.process_scroll(line_scroll(1.0), &mut camera, Vec3::new(2.0, 2.0, 2.0));

        // distance stays at 1.0; pos/target unchanged (actual_f = 1).
        assert!((controller.distance - 1.0).abs() < 1e-6);
        assert!((camera.position - original_pos).length() < 1e-4);
        assert!((camera.target - original_target).length() < 1e-4);
    }

    // -------- fit-distance framing --------

    #[test]
    fn fit_distance_subtends_half_fov_at_square_aspect() {
        // At aspect 1 the vertical and horizontal FOV are equal, so a
        // bounding sphere of radius r placed at the fit distance (margin
        // 1) subtends exactly the half-FOV: asin(r / d) == fov / 2.
        let cam = Camera::new(Vec3::ZERO, Vec3::ZERO, 1.0);
        let extent = Vec3::splat(10.0);
        let r = extent.length() * 0.5;
        let d = cam.fit_distance(extent, 1.0);
        assert!(
            ((r / d).asin() - cam.fov * 0.5).abs() < 1e-4,
            "asin(r/d)={} expected half-fov={}",
            (r / d).asin(),
            cam.fov * 0.5
        );
    }

    #[test]
    fn fit_distance_scales_linearly_with_size() {
        let cam = Camera::new(Vec3::ZERO, Vec3::ZERO, 1.6);
        let d1 = cam.fit_distance(Vec3::splat(4.0), 1.1);
        let d2 = cam.fit_distance(Vec3::splat(8.0), 1.1);
        assert!((d2 - 2.0 * d1).abs() < 1e-3, "d1={} d2={}", d1, d2);
    }

    #[test]
    fn fit_distance_margin_scales_distance() {
        let cam = Camera::new(Vec3::ZERO, Vec3::ZERO, 1.0);
        let tight = cam.fit_distance(Vec3::splat(10.0), 1.0);
        let padded = cam.fit_distance(Vec3::splat(10.0), 1.25);
        assert!((padded - tight * 1.25).abs() < 1e-3);
    }

    #[test]
    fn fit_distance_portrait_pulls_back_more_than_landscape() {
        // A narrow (portrait) viewport is horizontally tighter, so it
        // must sit further back than a wide one to fit the same box.
        let landscape = Camera::new(Vec3::ZERO, Vec3::ZERO, 1.78);
        let portrait = Camera::new(Vec3::ZERO, Vec3::ZERO, 0.56);
        let e = Vec3::splat(10.0);
        assert!(portrait.fit_distance(e, 1.0) > landscape.fit_distance(e, 1.0));
    }

    // -------- right-drag pan --------

    /// Drive one right-drag from `from` to `to` and return how far the
    /// camera moved.
    fn pan_offset(camera: &mut Camera, from: (f32, f32), to: (f32, f32)) -> Vec3 {
        let mut controller = CameraController::new_synced_for_test(camera);
        controller.right_mouse_pressed = true;
        controller.last_mouse_pos = Some(from);
        let before = camera.position;
        controller.process_mouse_motion(to.0, to.1, camera);
        camera.position - before
    }

    #[test]
    fn vertical_pan_stays_effective_at_the_pitch_clamp() {
        // Why pan uses screen-space up: against world up the visible
        // part of a vertical drag scales with cos(pitch), which is
        // ~0.07 at the clamp — the ordinary top-down editing pose.
        let mut camera = Camera::new(Vec3::new(0.0, 40.0, 3.0), Vec3::ZERO, 1.0);
        let forward = camera.forward();
        let offset = pan_offset(&mut camera, (100.0, 100.0), (100.0, 140.0));

        assert!(offset.length() > 1e-3, "drag produced no motion");
        assert!(
            offset.dot(forward).abs() / offset.length() < 1e-3,
            "pan must not dolly along the view axis; offset={:?}",
            offset
        );
    }

    #[test]
    fn pan_moves_position_and_target_together() {
        // Pan translates the whole rig, which is what keeps the cached
        // yaw / pitch valid without a re-sync.
        let mut camera = Camera::new(Vec3::new(0.0, 20.0, 40.0), Vec3::ZERO, 1.0);
        let before = camera.position - camera.target;
        pan_offset(&mut camera, (0.0, 0.0), (25.0, -10.0));
        let after = camera.position - camera.target;
        assert!(
            (after - before).length() < 1e-4,
            "camera-to-target vector changed: {:?} -> {:?}",
            before,
            after
        );
    }

    #[test]
    fn huge_pixel_scroll_stays_a_forward_zoom() {
        // A touchpad flick arrives as one big `PixelDelta`. The scale
        // factor is clamped, so the step stays a sane zoom instead of
        // collapsing to (or past) zero.
        let mut camera = Camera::new(Vec3::new(0.0, 20.0, 40.0), Vec3::ZERO, 1.0);
        let mut controller = CameraController::new_synced_for_test(&camera);
        let before = controller.distance;
        let anchor = camera.target;

        let delta = winit::event::MouseScrollDelta::PixelDelta(winit::dpi::PhysicalPosition::new(
            0.0, 400.0,
        ));
        controller.process_scroll(delta, &mut camera, anchor);

        assert!(
            controller.distance >= 0.2 * before - 1e-3,
            "one event zoomed further than the clamp allows: {} -> {}",
            before,
            controller.distance
        );
        assert!(
            (camera.position - anchor).dot(Vec3::new(0.0, 20.0, 40.0)) > 0.0,
            "camera crossed the anchor; position={:?}",
            camera.position
        );
    }
}
