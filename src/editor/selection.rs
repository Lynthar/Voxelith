//! Box selection: an axis-aligned region of cells the user has marked
//! for batch operations (copy / cut / paste / delete / move).
//!
//! `Selection` is a closed AABB inclusive on both corners — a 1×1×1
//! selection is a single cell. It deliberately does *not* track the
//! actual voxel contents inside the box: copy / cut read the world at
//! command-build time so they always see the latest state, and a
//! selection survives unrelated edits without going stale.

/// Axis-aligned closed selection box in world cell coordinates.
/// Inclusive on both corners.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Selection {
    pub min: (i32, i32, i32),
    pub max: (i32, i32, i32),
}

impl Selection {
    /// Build a selection from any two opposite corners. Either order
    /// of `(anchor, end)` produces the same box — corners are sorted.
    pub fn from_corners(a: (i32, i32, i32), b: (i32, i32, i32)) -> Self {
        Self {
            min: (a.0.min(b.0), a.1.min(b.1), a.2.min(b.2)),
            max: (a.0.max(b.0), a.1.max(b.1), a.2.max(b.2)),
        }
    }

    /// True if `pos` lies inside the closed AABB.
    pub fn contains(&self, pos: (i32, i32, i32)) -> bool {
        pos.0 >= self.min.0
            && pos.0 <= self.max.0
            && pos.1 >= self.min.1
            && pos.1 <= self.max.1
            && pos.2 >= self.min.2
            && pos.2 <= self.max.2
    }

    /// Width / height / depth in cells (X / Y / Z), each at least 1.
    pub fn size(&self) -> (i32, i32, i32) {
        (
            self.max.0 - self.min.0 + 1,
            self.max.1 - self.min.1 + 1,
            self.max.2 - self.min.2 + 1,
        )
    }

    /// Total cell count (volume of the AABB, including air).
    pub fn cell_count(&self) -> usize {
        let (w, h, d) = self.size();
        (w as usize) * (h as usize) * (d as usize)
    }

    /// Shift the selection by `delta` so the box keeps its size.
    pub fn translated(&self, delta: (i32, i32, i32)) -> Self {
        Self {
            min: (
                self.min.0 + delta.0,
                self.min.1 + delta.1,
                self.min.2 + delta.2,
            ),
            max: (
                self.max.0 + delta.0,
                self.max.1 + delta.1,
                self.max.2 + delta.2,
            ),
        }
    }

    /// Iterate every cell in the box. Order is `z` outermost, `y`
    /// middle, `x` innermost — matches `box_voxels` so callers that
    /// extract voxels from the selection get a stable order.
    pub fn iter_cells(&self) -> impl Iterator<Item = (i32, i32, i32)> + '_ {
        let (x0, x1) = (self.min.0, self.max.0);
        let (y0, y1) = (self.min.1, self.max.1);
        let (z0, z1) = (self.min.2, self.max.2);
        (z0..=z1).flat_map(move |z| {
            (y0..=y1).flat_map(move |y| (x0..=x1).map(move |x| (x, y, z)))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_corners_unordered() {
        let a = Selection::from_corners((3, 5, 7), (1, 2, 4));
        let b = Selection::from_corners((1, 2, 4), (3, 5, 7));
        assert_eq!(a, b);
        assert_eq!(a.min, (1, 2, 4));
        assert_eq!(a.max, (3, 5, 7));
    }

    #[test]
    fn single_cell_selection() {
        let s = Selection::from_corners((5, 5, 5), (5, 5, 5));
        assert_eq!(s.size(), (1, 1, 1));
        assert_eq!(s.cell_count(), 1);
        assert!(s.contains((5, 5, 5)));
        assert!(!s.contains((6, 5, 5)));
    }

    #[test]
    fn contains_inclusive_at_corners() {
        let s = Selection::from_corners((0, 0, 0), (3, 4, 5));
        assert!(s.contains((0, 0, 0)));
        assert!(s.contains((3, 4, 5)));
        assert!(!s.contains((-1, 0, 0)));
        assert!(!s.contains((4, 0, 0)));
        assert!(!s.contains((0, 0, 6)));
    }

    #[test]
    fn size_and_cell_count() {
        let s = Selection::from_corners((0, 0, 0), (1, 2, 3));
        assert_eq!(s.size(), (2, 3, 4));
        assert_eq!(s.cell_count(), 24);
    }

    #[test]
    fn translate_preserves_size() {
        let s = Selection::from_corners((0, 0, 0), (3, 3, 3));
        let t = s.translated((10, -5, 2));
        assert_eq!(t.min, (10, -5, 2));
        assert_eq!(t.max, (13, -2, 5));
        assert_eq!(t.size(), s.size());
    }

    #[test]
    fn negative_coordinates() {
        let s = Selection::from_corners((-5, -10, -3), (-2, -1, -1));
        assert_eq!(s.size(), (4, 10, 3));
        assert!(s.contains((-3, -5, -2)));
        assert!(!s.contains((0, 0, 0)));
    }

    #[test]
    fn iter_cells_visits_every_cell_once() {
        let s = Selection::from_corners((0, 0, 0), (1, 1, 1));
        let cells: Vec<_> = s.iter_cells().collect();
        assert_eq!(cells.len(), 8);
        let set: std::collections::HashSet<_> = cells.into_iter().collect();
        assert_eq!(set.len(), 8);
        for c in &[
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ] {
            assert!(set.contains(c));
        }
    }
}
