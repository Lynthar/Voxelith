//! Named attachment points: a world position plus an outward normal,
//! carrying no geometry and exported to glTF as empty nodes. Document
//! data that persists in `.vxlt`, but not on the undo stack.

use glam::{Quat, Vec3};

/// A named attachment point: a world position plus an outward unit
/// normal. The glTF export derives its node rotation from `normal` at
/// write time, so the convention can change without migrating files.
#[derive(Debug, Clone, PartialEq)]
pub struct Socket {
    /// Display and export name, unique within a scene because glTF
    /// nodes are keyed by name downstream.
    ///
    /// # Safety
    /// Uniqueness is enforced only at the two doors that write one; a
    /// third door has to carry the rule itself.
    pub name: String,
    /// World-space position — the center of the face the socket was
    /// dropped on, so it carries sub-cell `.5` offsets.
    pub position: [f32; 3],
    /// Outward unit normal of the face the socket sits on — one of six
    /// axis directions in practice, stored as a general vector so free
    /// orientation would need no format change.
    pub normal: [f32; 3],
}

impl Socket {
    pub fn new(name: impl Into<String>, position: [f32; 3], normal: [f32; 3]) -> Self {
        Self {
            name: name.into(),
            position,
            normal,
        }
    }

    /// glTF node rotation as a unit quaternion `[x, y, z, w]`: the
    /// shortest arc taking local +Y onto `normal`, so an attached prop
    /// grows out of the surface. `+Y` is the identity.
    pub fn rotation(&self) -> [f32; 4] {
        let n = Vec3::from(self.normal);
        let n = if n.length_squared() > 1e-12 {
            n.normalize()
        } else {
            Vec3::Y
        };
        let q = Quat::from_rotation_arc(Vec3::Y, n);
        [q.x, q.y, q.z, q.w]
    }
}

/// The smallest `Socket_N` name not already in `existing`. A counter
/// would leave gaps after deletes and `len() + 1` could collide with a
/// survivor, so this scans for the first free slot.
pub fn next_socket_name(existing: &[Socket]) -> String {
    let mut n = 1usize;
    loop {
        let candidate = format!("Socket_{n}");
        if !existing.iter().any(|s| s.name == candidate) {
            return candidate;
        }
        n += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_name_picks_first_free_slot() {
        assert_eq!(next_socket_name(&[]), "Socket_1");

        let one = vec![Socket::new("Socket_1", [0.0; 3], [0.0, 1.0, 0.0])];
        assert_eq!(next_socket_name(&one), "Socket_2");

        // After deleting Socket_2, the gap is reused (not Socket_4).
        let gap = vec![
            Socket::new("Socket_1", [0.0; 3], [0.0, 1.0, 0.0]),
            Socket::new("Socket_3", [0.0; 3], [0.0, 1.0, 0.0]),
        ];
        assert_eq!(next_socket_name(&gap), "Socket_2");

        // A user-renamed socket doesn't block the auto sequence.
        let renamed = vec![Socket::new("muzzle", [0.0; 3], [0.0, 1.0, 0.0])];
        assert_eq!(next_socket_name(&renamed), "Socket_1");
    }

    #[test]
    fn rotation_for_up_normal_is_identity() {
        let s = Socket::new("s", [0.5, 1.0, 0.5], [0.0, 1.0, 0.0]);
        let [x, y, z, w] = s.rotation();
        // Identity quaternion is (0, 0, 0, 1).
        assert!(x.abs() < 1e-6 && y.abs() < 1e-6 && z.abs() < 1e-6);
        assert!((w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rotation_maps_plus_y_onto_normal() {
        // For each axis-aligned face normal, applying the exported
        // rotation to local +Y must reproduce that normal (the whole
        // point of the convention).
        let normals = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        for n in normals {
            let s = Socket::new("s", [0.0; 3], n);
            let [qx, qy, qz, qw] = s.rotation();
            let q = Quat::from_xyzw(qx, qy, qz, qw);
            let out = q * Vec3::Y;
            let want = Vec3::from(n);
            assert!(
                (out - want).length() < 1e-5,
                "normal {n:?}: rotation maps +Y to {out:?}, expected {want:?}"
            );
        }
    }
}
