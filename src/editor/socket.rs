//! Named socket / attachment points for game-asset export.
//!
//! A socket marks a named anchor in the voxel scene — a world-space
//! position plus an outward orientation (the face normal it was placed
//! on). Sockets carry no geometry; they export to glTF as **empty
//! nodes** (`name` + `translation` + `rotation`, no `mesh`), the
//! standard way to ship attachment points — weapon mounts, banner /
//! emblem slots, FX anchors — that downstream engines hang separate
//! parts onto. See `docs/GAME_PIPELINE_ROADMAP.md` §3.1.4.
//!
//! Sockets are *document* data: they persist in `.vxlt` (embedded in
//! `io::EditorState`) and round-trip through autosave. Like the box
//! selection, though, they are **not** part of the undo history —
//! placement / rename / delete are managed directly (the `Socket` tool
//! drops one per click; the Tools panel renames and deletes them).

use glam::{Quat, Vec3};

/// A named attachment point: a world-space position plus an outward
/// orientation (unit normal).
///
/// Both fields are the source of truth; the glTF export derives the
/// node rotation from `normal` at write time (see [`Socket::rotation`]),
/// so the on-disk orientation convention can evolve without migrating
/// saved files.
#[derive(Debug, Clone, PartialEq)]
pub struct Socket {
    /// Display + export name. Unique within a scene (see
    /// [`next_socket_name`]) because glTF nodes are keyed by name
    /// downstream.
    pub name: String,
    /// World-space position — the center of the face the socket was
    /// dropped on, so it carries sub-cell `.5` offsets.
    pub position: [f32; 3],
    /// Outward unit normal of the face the socket sits on. One of the
    /// six axis directions for face / ground placement, but stored as a
    /// general vector so a future free-orientation editor needs no
    /// format change.
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

    /// glTF node rotation as a unit quaternion `[x, y, z, w]` (the glTF
    /// component order, with `w` the scalar — see the glTF 2.0 spec
    /// §3.5.2).
    ///
    /// It's the shortest-arc rotation taking local **+Y** onto the
    /// socket's outward `normal`, so the attached prop's "up" axis
    /// points out of the surface. A ground socket (normal `+Y`) is the
    /// identity rotation. `glam::Quat::from_rotation_arc` chooses a
    /// stable orthogonal axis for the antiparallel (`normal = -Y`)
    /// case, so a downward-facing socket still gets a valid 180°
    /// rotation rather than NaNs.
    ///
    /// (This is the producer half; the engine-side consumption
    /// convention is documented in `docs/ENGINE_CONTRACT.md` §sockets.)
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

/// Pick the smallest `Socket_N` (N ≥ 1) name not already present in
/// `existing`.
///
/// A monotonic counter would leave gaps after deletes and a plain
/// `len() + 1` could collide with a survivor, so we scan for the first
/// free slot instead. Names must stay unique because they become glTF
/// node names downstream. `existing` is tiny (a handful of sockets), so
/// the linear scan is irrelevant cost.
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
