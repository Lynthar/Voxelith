//! Input handling: voxel raycast, tool application, keyboard shortcuts.

use winit::keyboard::KeyCode;

use std::collections::HashSet;

use voxelith::editor::{
    box_voxels, build_clear_changes, build_move_changes, build_paste_changes,
    copy_selection_to_clipboard, cylinder_voxels, eyedrop, flood_fill, flood_fill_multi,
    line_voxels, mirror_selection_changes, rotate_selection_changes, sphere_voxels, Axis,
    BrushTool, Command, EditorTool, FillOutcome, Quarter, Ray, RaycastHit, Selection, Tool,
    ToolContext, VoxelChange, VoxelRaycast,
};

use super::{build_stroke_plane, App, EditInteraction, StrokePlane};

/// How far the hover ray travels looking for a hit, capping DDA work
/// per cursor move and how far a voxel can be placed from. Sized under
/// the fog distance, so anything clearly visible is still clickable.
const RAYCAST_MAX_DIST: f32 = 500.0;

impl App {
    /// The 3D anchor for a zoom-to-cursor scroll: the raycast hit, else
    /// the cursor ray against the view-depth plane through the target.
    /// `None` only when the renderer isn't initialized.
    pub(super) fn compute_zoom_anchor(&self) -> Option<glam::Vec3> {
        let renderer = self.renderer.as_ref()?;
        let window = self.window.as_ref()?;
        let size = window.inner_size();
        let view_proj_inv = renderer.camera.view_projection_matrix().inverse();
        let ray = Ray::from_screen(
            self.cursor_pos,
            (size.width as f32, size.height as f32),
            view_proj_inv,
        );

        // Real-geometry hit takes priority — this is the use case the
        // user described: "zoom in to inspect this voxel". Use the same
        // RAYCAST_MAX_DIST as editor picking so the reach is consistent.
        if let Some(hit) = VoxelRaycast::cast(&ray, &self.document.world, RAYCAST_MAX_DIST) {
            return Some(ray.at(hit.distance));
        }

        // Project the cursor ray onto the plane through `camera.target`,
        // keeping the anchor at the orbit pivot's view depth so the
        // shift is lateral rather than an unintended dolly.
        let camera = &renderer.camera;
        let view_dir = (camera.target - camera.position).normalize();
        let denom = ray.direction.dot(view_dir);
        if denom.abs() > 1e-6 {
            let t = (camera.target - ray.origin).dot(view_dir) / denom;
            if t > 0.0 {
                return Some(ray.at(t));
            }
        }
        // Degenerate (ray parallel to view plane or pointing the wrong
        // way). Falling back to `target` makes process_scroll behave
        // exactly like the pre-zoom-to-cursor "scale around target".
        Some(camera.target)
    }

    /// The orbit pivot for a middle-mouse press: cast along the camera's
    /// *forward*, not the cursor, so the hit lies on the view ray and
    /// re-anchoring changes only the distance, never the image.
    pub(super) fn compute_orbit_pivot(&self) -> Option<glam::Vec3> {
        let renderer = self.renderer.as_ref()?;
        let camera = &renderer.camera;
        let ray = Ray::new(camera.position, camera.forward());
        Some(VoxelRaycast::orbit_pivot(
            &ray,
            &self.document.world,
            RAYCAST_MAX_DIST,
            camera.target,
        ))
    }

    /// Frame an inclusive cell AABB: target the box center, then pull
    /// back along the *current* view direction. Only target and distance
    /// change, so framing never snaps to a new orientation.
    pub(super) fn frame_camera_on_aabb(&mut self, min: (i32, i32, i32), max: (i32, i32, i32)) {
        let Some(renderer) = &mut self.renderer else {
            return;
        };
        // Cells occupy [n, n+1), so the box spans min .. max+1 in world.
        let wmin = glam::Vec3::new(min.0 as f32, min.1 as f32, min.2 as f32);
        let wmax = glam::Vec3::new(max.0 as f32 + 1.0, max.1 as f32 + 1.0, max.2 as f32 + 1.0);
        let center = (wmin + wmax) * 0.5;
        let extent = wmax - wmin;

        let camera = &mut renderer.camera;
        // Preserve the current orbit direction; fall back to a 3/4 view
        // when the camera sits on the target (direction undefined).
        let mut dir = camera.position - camera.target;
        if !dir.is_finite() || dir.length() < 1e-4 {
            dir = glam::Vec3::new(1.0, 0.8, 1.0);
        }
        let dir = dir.normalize();
        let dist = camera.fit_distance(extent, 1.15).clamp(2.0, 500.0);
        camera.target = center;
        camera.position = center + dir * dist;
        renderer
            .camera_controller
            .sync_orbit_state_from_camera(camera);
    }

    /// Frame the whole scene (AABB of every non-air voxel).
    pub(super) fn frame_all(&mut self) {
        match self.document.world.scene_aabb() {
            Some((min, max)) => {
                self.frame_camera_on_aabb(min, max);
                self.ui.set_status("Framed scene");
            }
            None => self.ui.set_status("World is empty — nothing to frame"),
        }
    }

    /// Frame the active selection's AABB.
    pub(super) fn frame_selected(&mut self) {
        match self.editor.selection {
            Some(sel) => {
                self.frame_camera_on_aabb(sel.min, sel.max);
                self.ui.set_status("Framed selection");
            }
            None => self
                .ui
                .set_status("No selection — drag with the Select tool first"),
        }
    }

    /// Frame the footprint of the most recent generation (procgen /
    /// graph / AI), if any.
    pub(super) fn frame_generated(&mut self) {
        match self.last_generated_bounds {
            Some((min, max)) => {
                self.frame_camera_on_aabb(min, max);
                self.ui.set_status("Framed last generation");
            }
            None => self.ui.set_status("Nothing generated yet to frame"),
        }
    }

    /// Update the hovered voxel from the cursor. Anchor tools get the
    /// ground-plane fallback so they work in an empty world; tools that
    /// read voxels stay strict. A gesture's locked plane wins over both.
    pub(super) fn update_raycast(&mut self) {
        // A latched gesture locks the cursor to its face plane. Height
        // phase doesn't read `hovered_voxel`, but routing through the
        // lock keeps a stray move from confusing the preview cache key.
        if let Some(plane) = self.interaction.locked_plane() {
            self.editor.hovered_voxel = self.cast_ray_to_plane(&plane);
            return;
        }

        let Some(renderer) = &self.renderer else {
            return;
        };
        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();

        let view_proj = renderer.camera.view_projection_matrix();
        let view_proj_inv = view_proj.inverse();

        let ray = Ray::from_screen(
            self.cursor_pos,
            (size.width as f32, size.height as f32),
            view_proj_inv,
        );

        self.editor.hovered_voxel = if self.effective_tool().uses_ground_plane_fallback() {
            VoxelRaycast::cast_with_ground_plane(&ray, &self.document.world, RAYCAST_MAX_DIST, 0)
        } else {
            VoxelRaycast::cast(&ray, &self.document.world, RAYCAST_MAX_DIST)
        };
    }

    /// Synthesize a `RaycastHit` from a ray-vs-plane intersection, to
    /// keep drag-paint on the locked face. `None` when the ray is
    /// parallel to the plane or crosses it behind the camera.
    fn cast_ray_to_plane(&self, plane: &StrokePlane) -> Option<RaycastHit> {
        let renderer = self.renderer.as_ref()?;
        let window = self.window.as_ref()?;
        let size = window.inner_size();
        let view_proj_inv = renderer.camera.view_projection_matrix().inverse();
        let ray = Ray::from_screen(
            self.cursor_pos,
            (size.width as f32, size.height as f32),
            view_proj_inv,
        );

        let dir_arr = ray.direction.to_array();
        let origin_arr = ray.origin.to_array();
        let dir_axis = dir_arr[plane.axis];
        if dir_axis.abs() < 1e-6 {
            return None;
        }
        let t = (plane.plane_coord - origin_arr[plane.axis]) / dir_axis;
        if t <= 0.0 {
            return None;
        }
        // Cap the reach as the picking cast does: a ray nearly parallel
        // to the plane produces an enormous `t`, putting the footprint's
        // far corner millions of cells away.
        if t > RAYCAST_MAX_DIST {
            return None;
        }
        let p_arr = ray.at(t).to_array();
        let other1 = (plane.axis + 1) % 3;
        let other2 = (plane.axis + 2) % 3;
        let mut ap = [0i32; 3];
        ap[plane.axis] = plane.anchor_along_axis;
        ap[other1] = p_arr[other1].floor() as i32;
        ap[other2] = p_arr[other2].floor() as i32;
        let mut vp = ap;
        vp[plane.axis] -= plane.sign;
        let mut normal = [0i32; 3];
        normal[plane.axis] = plane.sign;
        Some(RaycastHit {
            voxel_pos: (vp[0], vp[1], vp[2]),
            adjacent_pos: (ap[0], ap[1], ap[2]),
            normal: (normal[0], normal[1], normal[2]),
            distance: t,
            virtual_ground: false,
        })
    }

    /// A left press in the viewport: arm the press-hold state, then let
    /// `apply_tool` refine it into a shape footprint or selection drag.
    /// A press over empty sky still arms drag-paint.
    pub(super) fn on_left_press(&mut self) {
        if !self.interaction.is_active() {
            self.interaction = EditInteraction::BrushStroke {
                plane: None,
                last_voxel: self.editor.hovered_voxel.map(|h| h.voxel_pos),
                start_screen: self.cursor_pos,
            };
        }
        self.apply_tool();
    }

    /// The left button came up: seal the stroke's undo entry, then
    /// finish the gesture. Runs even when egui consumed the release.
    /// `ShapeHeight` is left alone — its button isn't held.
    pub(super) fn on_left_release(&mut self) {
        // Seal unconditionally, before the per-state dispatch — a no-op
        // when no stroke is open. Otherwise a tool switch mid-drag
        // merges the next stroke into the previous undo entry.
        self.editor.history.end_stroke();
        match std::mem::take(&mut self.interaction) {
            EditInteraction::ShapeFootprint { anchor, plane } => {
                self.shape_footprint_released(anchor, plane);
            }
            keep @ EditInteraction::ShapeHeight { .. } => {
                self.interaction = keep;
            }
            EditInteraction::SelectDrag { anchor } => {
                self.commit_select_drag(anchor);
            }
            EditInteraction::SelectMove { anchor, .. } => {
                self.commit_select_move(anchor);
            }
            EditInteraction::BrushStroke { .. } | EditInteraction::Idle => {}
        }
    }

    /// Apply the current tool at the hovered location.
    pub(super) fn apply_tool(&mut self) {
        // The Height-phase commit needs no hovered voxel: it extrudes by
        // screen-Y against the locked plane. Behind the hover guard it
        // was swallowed exactly when the ray left that plane.
        if self.effective_tool().is_shape()
            && matches!(self.interaction, EditInteraction::ShapeHeight { .. })
        {
            self.commit_shape();
            return;
        }

        let Some(hit) = self.editor.hovered_voxel else {
            return;
        };

        match self.effective_tool() {
            Tool::Place | Tool::Remove | Tool::Paint => {
                // Lock the stroke to the first hit's face plane, so
                // drag-paint stays on one face instead of stacking
                // toward the camera. The lock dies on release.
                if let EditInteraction::BrushStroke {
                    plane: plane @ None,
                    ..
                } = &mut self.interaction
                {
                    *plane = build_stroke_plane(&hit);
                }
                let brush = BrushTool::new(self.effective_tool());
                let mut ctx = ToolContext {
                    world: &mut self.document.world,
                    history: &mut self.editor.history,
                    brush_color: self.editor.brush_color,
                    brush_size: self.editor.brush_size,
                    symmetry: self.editor.symmetry,
                };
                brush.apply(&mut ctx, &hit);
            }
            Tool::Eyedropper => {
                if let Some(color) = eyedrop(&self.document.world, &hit) {
                    self.editor.brush_color = color;
                }
            }
            Tool::Fill => {
                // Refuse to flood from an air cell: a virtual hit would
                // eat the whole air region around the cursor, bounded
                // only by the spatial cap.
                let v = self.document.world.get_voxel(
                    hit.voxel_pos.0,
                    hit.voxel_pos.1,
                    hit.voxel_pos.2,
                );
                if v.is_air() {
                    return;
                }
                let symmetry = self.editor.symmetry;
                let outcome = if symmetry.any() {
                    // Combine all mirrored fills into one undo entry —
                    // a single click should be a single undo, even at
                    // 8-fold symmetry.
                    let starts = symmetry.mirror_positions(hit.voxel_pos);
                    flood_fill_multi(
                        &mut self.document.world,
                        &mut self.editor.history,
                        &starts,
                        self.editor.brush_color,
                        10000,
                    )
                } else {
                    flood_fill(
                        &mut self.document.world,
                        &mut self.editor.history,
                        hit.voxel_pos,
                        self.editor.brush_color,
                        10000,
                    )
                };
                self.report_fill(outcome);
            }
            Tool::Line | Tool::Box | Tool::Sphere | Tool::Cylinder => {
                // Two-phase: the first press enters Footprint, locking
                // the plane and anchoring; the second commits from
                // Height. A press during Footprint can't happen.
                match self.interaction {
                    EditInteraction::ShapeHeight { .. } => {
                        self.commit_shape();
                    }
                    EditInteraction::ShapeFootprint { .. } => {
                        // Defensive: ignore.
                    }
                    _ => {
                        if let Some(plane) = build_stroke_plane(&hit) {
                            self.interaction = EditInteraction::ShapeFootprint {
                                anchor: hit.adjacent_pos,
                                plane,
                            };
                        } else {
                            self.ui.set_status(
                                "Shape tool: face normal not axis-aligned, ignoring click",
                            );
                        }
                    }
                }
            }
            Tool::Select => {
                // Inside an existing selection starts a move; anywhere
                // else starts a fresh drag. `select_anchor_pos` keeps
                // an empty-world drag from sinking one cell under.
                let cell = Self::select_anchor_pos(&hit);
                if let Some(sel) = self.editor.selection {
                    if sel.contains(cell) {
                        let ghost = self.move_ghost_snapshot(sel);
                        self.interaction = EditInteraction::SelectMove {
                            anchor: cell,
                            ghost,
                        };
                        return;
                    }
                }
                self.interaction = EditInteraction::SelectDrag { anchor: cell };
            }
            Tool::Socket => {
                // Drop a socket at the clicked face's center, oriented
                // along its normal. Single click, not undoable; the
                // handler excludes Socket from drags, so no duplicates.
                let (nx, ny, nz) = hit.normal;
                if nx == 0 && ny == 0 && nz == 0 {
                    // Degenerate normal (ray started inside a voxel) —
                    // a socket with no orientation can't export a
                    // meaningful rotation.
                    self.ui
                        .set_status("Socket: aim at a voxel face or the ground to place");
                    return;
                }
                // Face center = hit voxel center + half a cell along the
                // outward normal. For a virtual-ground hit (voxel_pos
                // (x, -1, z), normal +Y) this lands on the y=0 plane.
                let (vx, vy, vz) = hit.voxel_pos;
                let position = [
                    vx as f32 + 0.5 + nx as f32 * 0.5,
                    vy as f32 + 0.5 + ny as f32 * 0.5,
                    vz as f32 + 0.5 + nz as f32 * 0.5,
                ];
                let normal = [nx as f32, ny as f32, nz as f32];
                let name = voxelith::editor::next_socket_name(&self.document.sockets);
                self.document.sockets.push(voxelith::editor::Socket::new(
                    name.clone(),
                    position,
                    normal,
                ));
                // Sockets are document data no mesh rebuild notices —
                // placing one has to raise the unsaved flags itself, or
                // "place a socket, quit" lost it without a prompt.
                self.document.bump();
                self.ui.set_status(format!(
                    "Placed {} at ({:.1}, {:.1}, {:.1})",
                    name, position[0], position[1], position[2]
                ));
            }
        }
    }

    /// Largest selection AABB the dense sweeps will walk. Copy, cut,
    /// delete, move, rotate and mirror all iterate every cell including
    /// air. The marquee may be any size; only the sweeps are bounded.
    pub(super) const MAX_SELECTION_SWEEP_CELLS: i64 = 16_777_216;

    /// Cells in a selection's AABB, in `i64` — the `i32` arithmetic in
    /// `Selection::size` wraps for exactly the boxes this guards.
    fn selection_sweep_cells(sel: &Selection) -> i64 {
        let extent = |a: i32, b: i32| (b as i64 - a as i64) + 1;
        extent(sel.min.0, sel.max.0) * extent(sel.min.1, sel.max.1) * extent(sel.min.2, sel.max.2)
    }

    /// True (with a status message) when `sel` is too big for a dense
    /// sweep; the caller returns without starting one.
    fn refuse_oversized_sweep(&mut self, sel: Selection, what: &str) -> bool {
        let cells = Self::selection_sweep_cells(&sel);
        if cells > Self::MAX_SELECTION_SWEEP_CELLS {
            self.ui.set_status(format!(
                "{what} would sweep {cells} cells; the largest supported \
                 box is {} (256³) — shrink the selection",
                Self::MAX_SELECTION_SWEEP_CELLS
            ));
            true
        } else {
            false
        }
    }

    /// Finish a `SelectMove` on release: translate the voxels by
    /// `current - anchor` as one undoable command, then update the
    /// AABB. The ghost snapshot went with the state.
    fn commit_select_move(&mut self, anchor: (i32, i32, i32)) {
        match (self.editor.selection, self.editor.hovered_voxel) {
            (Some(_sel), Some(hit)) => {
                let cur = Self::select_anchor_pos(&hit);
                let delta = (cur.0 - anchor.0, cur.1 - anchor.1, cur.2 - anchor.2);
                if delta != (0, 0, 0) {
                    self.move_selection(delta);
                }
            }
            // Released with the ray off the world. Say so: the ghost
            // snaps back, and in silence that reads as "the move didn't
            // take" with no clue why.
            (Some(_), None) => {
                self.ui
                    .set_status("Move canceled (cursor off-world on release)");
            }
            _ => {}
        }
    }

    /// Finish a `SelectDrag` on release: build a `Selection` from the
    /// anchor to the current cell. The marquee is ephemeral and never
    /// reaches the undo history.
    fn commit_select_drag(&mut self, anchor: (i32, i32, i32)) {
        let Some(hit) = self.editor.hovered_voxel else {
            self.ui
                .set_status("Selection canceled (cursor off-world on release)");
            return;
        };
        let end = Self::select_anchor_pos(&hit);
        self.editor.selection = Some(Selection::from_corners(anchor, end));
    }

    /// Translate the selection's non-air voxels by `delta` as one
    /// command, and move the AABB with them. Overlap handling lives in
    /// `build_move_changes`.
    pub(super) fn move_selection(&mut self, delta: (i32, i32, i32)) {
        if delta == (0, 0, 0) {
            return;
        }
        let Some(sel) = self.editor.selection else {
            return;
        };
        if self.refuse_oversized_sweep(sel, "Move") {
            return;
        }
        let changes = build_move_changes(&self.document.world, sel, delta);
        if !changes.is_empty() {
            let cmd = Command::set_voxels(changes);
            self.editor.history.execute(cmd, &mut self.document.world);
        }
        // Even an empty selection (all air) bumps its AABB so the
        // user can keyboard-nudge a marquee around empty space.
        self.editor.selection = Some(sel.translated(delta));
    }

    /// A `ShapeFootprint` release moves to Height phase: the current
    /// plane-locked hit becomes the footprint corner and its screen-Y
    /// the baseline. A release off-world cancels instead.
    fn shape_footprint_released(&mut self, anchor: (i32, i32, i32), plane: StrokePlane) {
        let Some(hit) = self.editor.hovered_voxel else {
            self.ui
                .set_status("Shape canceled (cursor off-plane on release)");
            return;
        };
        self.interaction = EditInteraction::ShapeHeight {
            anchor,
            plane,
            end_on_plane: hit.adjacent_pos,
            release_screen_y: self.cursor_pos.1,
        };
        self.ui
            .set_status("Drag vertically to set height, click to commit (Esc cancels)");
    }

    /// Rotate the selection's contents around `axis`. The AABB may swap
    /// extents but its `min` stays put, and the whole rotation is one
    /// command.
    pub(super) fn rotate_selection(&mut self, axis: Axis, quarter: Quarter) {
        let Some(sel) = self.editor.selection else {
            self.ui
                .set_status("No selection — drag with the Select tool first");
            return;
        };
        if self.refuse_oversized_sweep(sel, "Rotate") {
            return;
        }
        let (new_sel, changes) = rotate_selection_changes(&self.document.world, sel, axis, quarter);
        let count = changes.len();
        if !changes.is_empty() {
            let cmd = Command::set_voxels(changes);
            self.editor.history.execute(cmd, &mut self.document.world);
        }
        // Bump the selection AABB even when empty so a user rotating
        // an air-only marquee still sees the box reorient.
        self.editor.selection = Some(new_sel);
        let label = match (axis, quarter) {
            (Axis::X, Quarter::Cw) => "Rotate X 90°",
            (Axis::X, Quarter::Ccw) => "Rotate X -90°",
            (Axis::X, Quarter::Half) => "Rotate X 180°",
            (Axis::Y, Quarter::Cw) => "Rotate Y 90°",
            (Axis::Y, Quarter::Ccw) => "Rotate Y -90°",
            (Axis::Y, Quarter::Half) => "Rotate Y 180°",
            (Axis::Z, Quarter::Cw) => "Rotate Z 90°",
            (Axis::Z, Quarter::Ccw) => "Rotate Z -90°",
            (Axis::Z, Quarter::Half) => "Rotate Z 180°",
        };
        if count == 0 {
            self.ui.set_status(format!("{} (selection empty)", label));
        } else {
            self.ui.set_status(format!("{} ({} cells)", label, count));
        }
    }

    /// Report what a Fill click did. Without it, a click that hit a cap
    /// — or changed nothing because the region already held the brush
    /// color — is indistinguishable from one that missed.
    fn report_fill(&mut self, outcome: FillOutcome) {
        let msg = if outcome.truncated {
            // Deliberately ahead of the zero case: a fill that stopped
            // at a cap has *more region out there*, which "already that
            // color" would flatly contradict.
            format!(
                "Filled {} voxels (stopped at the fill limit)",
                outcome.written
            )
        } else if outcome.written == 0 {
            "Filled 0 voxels (already that color)".to_string()
        } else {
            format!("Filled {} voxels", outcome.written)
        };
        self.ui.set_status(msg);
    }

    /// Mirror the active selection's contents across the midplane
    /// perpendicular to `axis`. The AABB is unchanged. Single
    /// `Command::set_voxels` so one Ctrl+Z reverses the flip.
    pub(super) fn mirror_selection(&mut self, axis: Axis) {
        let Some(sel) = self.editor.selection else {
            self.ui
                .set_status("No selection — drag with the Select tool first");
            return;
        };
        if self.refuse_oversized_sweep(sel, "Mirror") {
            return;
        }
        let changes = mirror_selection_changes(&self.document.world, sel, axis);
        let count = changes.len();
        if !changes.is_empty() {
            let cmd = Command::set_voxels(changes);
            self.editor.history.execute(cmd, &mut self.document.world);
        }
        let label = match axis {
            Axis::X => "Flip X",
            Axis::Y => "Flip Y",
            Axis::Z => "Flip Z",
        };
        if count == 0 {
            self.ui
                .set_status(format!("{} (no change — selection is symmetric)", label));
        } else {
            self.ui.set_status(format!("{} ({} cells)", label, count));
        }
    }

    /// Step the selection by `delta` in response to an arrow-key
    /// press. No-op if there's no selection or a mouse drag is in
    /// progress (so the user can't fight a drag with the keyboard).
    fn step_selection(&mut self, delta: (i32, i32, i32)) {
        if matches!(
            self.interaction,
            EditInteraction::SelectDrag { .. } | EditInteraction::SelectMove { .. }
        ) {
            return;
        }
        if self.editor.selection.is_none() {
            return;
        }
        self.move_selection(delta);
    }

    /// Commit the in-progress shape gesture on the second click; a
    /// no-op when none is active. A footprint-only commit treats height
    /// as 0, giving a shape one cell thick along the plane normal.
    pub(super) fn commit_shape(&mut self) {
        let tool = self.effective_tool();
        let cursor_y = self.cursor_pos.1;

        let (anchor, end, plane_axis) = match self.interaction {
            EditInteraction::ShapeFootprint { anchor, plane } => {
                // Defensive: a second-click commit always comes from
                // Height phase. From Footprint, fall back to the
                // current plane-locked cell — the gesture ends anyway.
                let Some(hit) = self.editor.hovered_voxel else {
                    self.interaction = EditInteraction::Idle;
                    return;
                };
                (anchor, hit.adjacent_pos, plane.axis)
            }
            EditInteraction::ShapeHeight { anchor, plane, .. } => {
                let end = self
                    .interaction
                    .shape_extruded_end(cursor_y)
                    .expect("Height phase");
                (anchor, end, plane.axis)
            }
            _ => return,
        };
        self.interaction = EditInteraction::Idle;

        // Budget check before the enumeration it bounds: a glancing drag
        // can legally describe hundreds of millions of cells. A shape
        // that outgrew its preview can still commit under this cap.
        let cost = super::shape_cell_cost(tool, anchor, end)
            .saturating_mul(super::symmetry_factor(self.editor.symmetry));
        if cost > super::MAX_SHAPE_COMMIT_CELLS {
            self.ui.set_status(format!(
                "Shape spans {cost} cells; the largest commit is {} — \
                 build it in smaller pieces",
                super::MAX_SHAPE_COMMIT_CELLS
            ));
            return;
        }
        let raw = match tool {
            Tool::Line => line_voxels(anchor, end),
            Tool::Box => box_voxels(anchor, end),
            Tool::Sphere => sphere_voxels(anchor, end),
            Tool::Cylinder => cylinder_voxels(anchor, end, Some(plane_axis)),
            _ => return, // shape state only entered by shape tools, defensive
        };

        // Apply symmetry across world-origin planes. HashSet dedupes
        // cells where mirrored shapes overlap (e.g. a Y-symmetric
        // shape spanning y=0 covers cells in both halves).
        let symmetry = self.editor.symmetry;
        let positions: Vec<(i32, i32, i32)> = if symmetry.any() {
            let mut set: HashSet<(i32, i32, i32)> = HashSet::new();
            for cell in raw {
                for m in symmetry.mirror_positions(cell) {
                    set.insert(m);
                }
            }
            set.into_iter().collect()
        } else {
            raw
        };

        let color = self.editor.brush_color;
        let changes: Vec<VoxelChange> = positions
            .into_iter()
            .map(|pos| VoxelChange {
                pos,
                old_voxel: self.document.world.get_voxel(pos.0, pos.1, pos.2),
                new_voxel: color,
            })
            .filter(|c| c.old_voxel != c.new_voxel)
            .collect();

        if !changes.is_empty() {
            let cmd = Command::set_voxels(changes);
            self.editor.history.execute(cmd, &mut self.document.world);
        }
    }

    /// Capture the active selection's non-air voxels into the
    /// clipboard. No-op (with a status hint) if there's no selection.
    pub(super) fn copy_selection(&mut self) {
        let Some(sel) = self.editor.selection else {
            self.ui
                .set_status("No selection — drag with the Select tool first");
            return;
        };
        if self.refuse_oversized_sweep(sel, "Copy") {
            return;
        }
        let clipboard = copy_selection_to_clipboard(&self.document.world, sel);
        let count = clipboard.voxel_count();
        self.clipboard = Some(clipboard);
        if count == 0 {
            self.ui.set_status("Selection contains no solid voxels");
        } else {
            self.ui.set_status(format!("Copied {} voxels", count));
        }
    }

    /// Cut: snapshot the selection into the clipboard, then clear it in
    /// a **single** command — pushing copy and delete separately would
    /// make one Ctrl+Z restore half the cut.
    pub(super) fn cut_selection(&mut self) {
        let Some(sel) = self.editor.selection else {
            self.ui
                .set_status("No selection — drag with the Select tool first");
            return;
        };
        if self.refuse_oversized_sweep(sel, "Cut") {
            return;
        }
        let clipboard = copy_selection_to_clipboard(&self.document.world, sel);
        let count = clipboard.voxel_count();
        self.clipboard = Some(clipboard);

        let changes = build_clear_changes(&self.document.world, sel);
        if !changes.is_empty() {
            let cmd = Command::set_voxels(changes);
            self.editor.history.execute(cmd, &mut self.document.world);
        }

        if count == 0 {
            self.ui
                .set_status("Selection had no solid voxels — clipboard empty");
        } else {
            self.ui.set_status(format!("Cut {} voxels", count));
        }
    }

    /// Delete: clear non-air cells inside the selection without
    /// touching the clipboard.
    pub(super) fn delete_selection(&mut self) {
        let Some(sel) = self.editor.selection else {
            self.ui
                .set_status("No selection — drag with the Select tool first");
            return;
        };
        if self.refuse_oversized_sweep(sel, "Delete") {
            return;
        }
        let changes = build_clear_changes(&self.document.world, sel);
        let count = changes.len();
        if !changes.is_empty() {
            let cmd = Command::set_voxels(changes);
            self.editor.history.execute(cmd, &mut self.document.world);
        }
        if count == 0 {
            self.ui
                .set_status("Selection had no solid voxels to delete");
        } else {
            self.ui.set_status(format!("Deleted {} voxels", count));
        }
    }

    /// Paste at the selection's origin, or at the hovered cell when
    /// `prefer_cursor` is set or nothing is selected. The destination is
    /// auto-selected afterwards, so a second paste chains from it.
    pub(super) fn paste_clipboard(&mut self, prefer_cursor: bool) {
        let Some(clipboard) = self.clipboard.as_ref() else {
            self.ui
                .set_status("Clipboard is empty — Copy / Cut a selection first");
            return;
        };
        if clipboard.is_empty() {
            self.ui.set_status("Clipboard is empty");
            return;
        }

        let cursor_dest = self
            .editor
            .hovered_voxel
            .map(|h| Self::select_anchor_pos(&h));
        let dest = if prefer_cursor {
            cursor_dest
        } else {
            self.editor.selection.map(|s| s.min).or(cursor_dest)
        };

        let Some(dest) = dest else {
            self.ui
                .set_status("Move the cursor over the world to paste");
            return;
        };

        let changes = build_paste_changes(&self.document.world, clipboard, dest);
        let count = changes.len();
        if !changes.is_empty() {
            let cmd = Command::set_voxels(changes);
            self.editor.history.execute(cmd, &mut self.document.world);
        }

        // Auto-select the destination AABB so the user can chain
        // Paste→drag→Paste without re-marqueeing.
        let (sw, sh, sd) = clipboard.size;
        self.editor.selection = Some(Selection {
            min: dest,
            max: (dest.0 + sw - 1, dest.1 + sh - 1, dest.2 + sd - 1),
        });

        if count == 0 {
            self.ui
                .set_status("Pasted (no changes — destination already matched)");
        } else {
            self.ui.set_status(format!("Pasted {} voxels", count));
        }
    }

    /// Select the AABB of every non-air voxel — the same bounds Frame
    /// All and the exporters use. Says so when the world is empty.
    pub(super) fn select_all_solid(&mut self) {
        match self.document.world.scene_aabb() {
            Some((min, max)) => {
                self.editor.selection = Some(Selection { min, max });
                let (w, h, d) = (max.0 - min.0 + 1, max.1 - min.1 + 1, max.2 - min.2 + 1);
                self.ui
                    .set_status(format!("Selected all: {}×{}×{}", w, h, d));
            }
            None => {
                // Through `deselect`, not by assigning `None`: a
                // selection is also the drag anchor and the move ghost,
                // and clearing the field alone leaves those on screen.
                self.deselect();
                self.ui.set_status("World is empty — nothing to select");
            }
        }
    }

    /// Clear all selection state: the marquee plus any select or move
    /// gesture, whose ghost dies with it. Shared by the UI action, Esc
    /// and Ctrl+D so the three can't drift; other gestures are untouched.
    pub(super) fn deselect(&mut self) {
        if matches!(
            self.interaction,
            EditInteraction::SelectDrag { .. } | EditInteraction::SelectMove { .. }
        ) {
            self.interaction = EditInteraction::Idle;
        }
        self.editor.selection = None;
    }

    /// The tool a click acts as right now: Eyedropper while Alt is held,
    /// the selected tool otherwise. A derived read, not a mode switch,
    /// so a stuck swap is unrepresentable.
    pub(super) fn effective_tool(&self) -> Tool {
        if self.modifiers.alt_key() {
            Tool::Eyedropper
        } else {
            self.editor.current_tool
        }
    }

    /// The platform's command modifier: ⌘ on macOS, Ctrl elsewhere.
    /// Every chord keys off this — checking `control_key()` alone left
    /// ⌘S dead on macOS.
    fn primary_modifier(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            self.modifiers.super_key()
        }
        #[cfg(not(target_os = "macos"))]
        {
            self.modifiers.control_key()
        }
    }

    /// Any command modifier at all — the classification `handler.rs`
    /// uses to keep chords out of the fly camera. The bare-letter
    /// shortcuts guard on this, or ⌘R falls through to rotate.
    fn command_chord(&self) -> bool {
        self.modifiers.control_key() || self.modifiers.super_key()
    }

    /// Handle keyboard shortcuts (tools, undo/redo, file ops,
    /// selection).
    pub(super) fn handle_tool_shortcut(&mut self, key: KeyCode) {
        // Command chords dispatch from the descriptor table, the same
        // rows the help window prints. Everything below it is
        // hand-written: those bindings depend on state a row can't hold.
        if self.primary_modifier() {
            if let Some(spec) = voxelith::ui::keymap::find_chord(key, self.modifiers.shift_key()) {
                self.ui.state.request((spec.make)());
                return;
            }
        }
        match key {
            KeyCode::Digit1 => self.editor.select_tool(Tool::Place),
            KeyCode::Digit2 => self.editor.select_tool(Tool::Remove),
            KeyCode::Digit3 => self.editor.select_tool(Tool::Paint),
            KeyCode::Digit4 => self.editor.select_tool(Tool::Eyedropper),
            KeyCode::Digit5 => self.editor.select_tool(Tool::Fill),
            KeyCode::Digit6 => self.editor.select_tool(Tool::Line),
            KeyCode::Digit7 => self.editor.select_tool(Tool::Box),
            KeyCode::Digit8 => self.editor.select_tool(Tool::Sphere),
            KeyCode::Digit9 => self.editor.select_tool(Tool::Cylinder),
            KeyCode::Digit0 => self.editor.select_tool(Tool::Select),
            // Esc cancels the in-flight gesture first and only
            // deselects when there is none — doing both at once threw
            // away a marquee the user had set up before the shape.
            KeyCode::Escape => {
                if matches!(
                    self.interaction,
                    EditInteraction::ShapeFootprint { .. } | EditInteraction::ShapeHeight { .. }
                ) {
                    self.cancel_interaction();
                    self.ui.set_status("Shape canceled");
                } else {
                    self.deselect();
                }
            }
            KeyCode::Delete => {
                self.delete_selection();
            }
            // Rotate or mirror the selection; the full axis set lives in
            // the Selection menu. Guarded against every command
            // modifier, so a stray ⌘R can't silently transform geometry.
            KeyCode::KeyR if !self.command_chord() => {
                if self.modifiers.shift_key() {
                    self.rotate_selection(Axis::Y, Quarter::Ccw);
                } else {
                    self.rotate_selection(Axis::Y, Quarter::Cw);
                }
            }
            KeyCode::KeyM if !self.command_chord() => {
                self.mirror_selection(Axis::X);
            }
            // Arrow-key nudge: ←→ on X, ↑↓ on Z, the command modifier
            // promoting ↑↓ to Y since four arrows can't cover six
            // directions. Skipped while a drag is mid-flight.
            KeyCode::ArrowLeft => {
                let step = if self.modifiers.shift_key() { 10 } else { 1 };
                self.step_selection((-step, 0, 0));
            }
            KeyCode::ArrowRight => {
                let step = if self.modifiers.shift_key() { 10 } else { 1 };
                self.step_selection((step, 0, 0));
            }
            KeyCode::ArrowUp => {
                let step = if self.modifiers.shift_key() { 10 } else { 1 };
                if self.primary_modifier() {
                    self.step_selection((0, step, 0));
                } else {
                    self.step_selection((0, 0, -step));
                }
            }
            KeyCode::ArrowDown => {
                let step = if self.modifiers.shift_key() { 10 } else { 1 };
                if self.primary_modifier() {
                    self.step_selection((0, -step, 0));
                } else {
                    self.step_selection((0, 0, step));
                }
            }
            // F frames the selection's AABB, else the whole scene:
            // target and distance move while the viewing angle stays.
            // The recovery hatch after flying off the model.
            KeyCode::KeyF => {
                if self.editor.selection.is_some() {
                    self.frame_selected();
                } else {
                    self.frame_all();
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    //! The gesture transition table, pinned cell by cell. `App::new()`
    //! builds without a window or GPU, so these drive the real
    //! transitions rather than a test-only reimplementation.

    use voxelith::core::Voxel;
    use voxelith::editor::{RaycastHit, Selection, Tool};
    use winit::keyboard::{KeyCode, ModifiersState};

    use super::super::{App, EditInteraction};

    /// A hit on the top face of the voxel at `(x, y, z)`.
    fn top_hit(x: i32, y: i32, z: i32) -> RaycastHit {
        RaycastHit {
            voxel_pos: (x, y, z),
            adjacent_pos: (x, y + 1, z),
            normal: (0, 1, 0),
            distance: 5.0,
            virtual_ground: false,
        }
    }

    /// A hit whose normal is not axis-aligned (ray started inside a
    /// voxel) — the input a shape press must refuse.
    fn degenerate_hit() -> RaycastHit {
        RaycastHit {
            voxel_pos: (0, 0, 0),
            adjacent_pos: (0, 0, 0),
            normal: (0, 0, 0),
            distance: 0.0,
            virtual_ground: false,
        }
    }

    fn app_with_tool(tool: Tool) -> App {
        let mut app = App::new();
        app.editor.select_tool(tool);
        app
    }

    #[test]
    fn a_brush_press_arms_a_stroke_and_locks_the_plane() {
        let mut app = app_with_tool(Tool::Place);
        app.document
            .world
            .set_voxel(0, 0, 0, Voxel::from_rgb(1, 2, 3));
        app.editor.hovered_voxel = Some(top_hit(0, 0, 0));
        app.on_left_press();
        // The press painted, and the stroke latched onto the hit face.
        assert!(!app.document.world.get_voxel(0, 1, 0).is_air());
        match &app.interaction {
            EditInteraction::BrushStroke { plane: Some(p), .. } => {
                assert_eq!((p.axis, p.sign), (1, 1), "top face = +Y plane");
            }
            other => panic!("expected a plane-locked BrushStroke, got {other:?}"),
        }
        app.on_left_release();
        assert!(matches!(app.interaction, EditInteraction::Idle));
    }

    #[test]
    fn a_press_over_empty_sky_still_arms_the_stroke_without_a_plane() {
        let mut app = app_with_tool(Tool::Place);
        app.editor.hovered_voxel = None;
        app.on_left_press();
        // Dragging into the world later must still drag-paint, so the
        // hold is armed; the plane waits for the first in-world apply.
        assert!(matches!(
            app.interaction,
            EditInteraction::BrushStroke { plane: None, .. }
        ));
    }

    #[test]
    fn a_click_tool_press_is_a_plain_hold() {
        let mut app = app_with_tool(Tool::Eyedropper);
        app.document
            .world
            .set_voxel(0, 0, 0, Voxel::from_rgb(9, 8, 7));
        app.editor.hovered_voxel = Some(top_hit(0, 0, 0));
        app.on_left_press();
        // Eyedropper sampled, but holds no plane and starts no gesture
        // beyond "the button is down".
        assert_eq!(app.editor.brush_color.color()[0], 9);
        assert!(matches!(
            app.interaction,
            EditInteraction::BrushStroke { plane: None, .. }
        ));
    }

    #[test]
    fn a_shape_press_enters_footprint_and_release_extrudes() {
        let mut app = app_with_tool(Tool::Box);
        app.editor.hovered_voxel = Some(top_hit(0, 0, 0));
        app.on_left_press();
        assert!(matches!(
            app.interaction,
            EditInteraction::ShapeFootprint {
                anchor: (0, 1, 0),
                ..
            }
        ));
        // Release with a plane hit → Height phase, footprint corner
        // locked.
        app.editor.hovered_voxel = Some(top_hit(2, 0, 2));
        app.on_left_release();
        assert!(matches!(
            app.interaction,
            EditInteraction::ShapeHeight {
                anchor: (0, 1, 0),
                end_on_plane: (2, 1, 2),
                ..
            }
        ));
    }

    #[test]
    fn a_footprint_release_off_world_cancels() {
        let mut app = app_with_tool(Tool::Box);
        app.editor.hovered_voxel = Some(top_hit(0, 0, 0));
        app.on_left_press();
        app.editor.hovered_voxel = None;
        app.on_left_release();
        assert!(matches!(app.interaction, EditInteraction::Idle));
    }

    #[test]
    fn the_second_click_commits_the_shape_as_one_undo_entry() {
        let mut app = app_with_tool(Tool::Box);
        app.editor.hovered_voxel = Some(top_hit(0, 0, 0));
        app.on_left_press();
        app.editor.hovered_voxel = Some(top_hit(1, 0, 1));
        app.on_left_release();
        // Second click, cursor unmoved since release → height 0, a
        // one-cell-thick 2×2 slab.
        app.on_left_press();
        assert!(matches!(app.interaction, EditInteraction::Idle));
        assert_eq!(app.editor.history.undo_count(), 1);
        assert!(!app.document.world.get_voxel(0, 1, 0).is_air());
        assert!(!app.document.world.get_voxel(1, 1, 1).is_air());
    }

    #[test]
    fn a_ghost_release_leaves_the_height_phase_pending() {
        let mut app = app_with_tool(Tool::Box);
        app.editor.hovered_voxel = Some(top_hit(0, 0, 0));
        app.on_left_press();
        app.on_left_release();
        assert!(matches!(
            app.interaction,
            EditInteraction::ShapeHeight { .. }
        ));
        // A release with no matching press in the viewport (egui ate
        // the press) must not kill the pending extrusion.
        app.on_left_release();
        assert!(matches!(
            app.interaction,
            EditInteraction::ShapeHeight { .. }
        ));
    }

    #[test]
    fn a_shape_press_on_a_degenerate_normal_stays_a_plain_hold() {
        let mut app = app_with_tool(Tool::Sphere);
        app.editor.hovered_voxel = Some(degenerate_hit());
        app.on_left_press();
        assert!(matches!(
            app.interaction,
            EditInteraction::BrushStroke { plane: None, .. }
        ));
    }

    #[test]
    fn a_select_drag_commits_the_marquee_on_release() {
        let mut app = app_with_tool(Tool::Select);
        app.editor.hovered_voxel = Some(top_hit(1, 0, 1));
        app.on_left_press();
        assert!(matches!(
            app.interaction,
            EditInteraction::SelectDrag { anchor: (1, 0, 1) }
        ));
        app.editor.hovered_voxel = Some(top_hit(3, 0, 4));
        app.on_left_release();
        assert!(matches!(app.interaction, EditInteraction::Idle));
        let sel = app.editor.selection.expect("marquee committed");
        assert_eq!((sel.min, sel.max), ((1, 0, 1), (3, 0, 4)));
    }

    #[test]
    fn a_press_inside_the_selection_moves_its_voxels() {
        let mut app = app_with_tool(Tool::Select);
        app.document
            .world
            .set_voxel(0, 0, 0, Voxel::from_rgb(5, 5, 5));
        app.editor.selection = Some(Selection {
            min: (0, 0, 0),
            max: (0, 0, 0),
        });
        app.editor.hovered_voxel = Some(top_hit(0, 0, 0));
        app.on_left_press();
        match &app.interaction {
            EditInteraction::SelectMove { anchor, ghost } => {
                assert_eq!(*anchor, (0, 0, 0));
                assert_eq!(ghost.len(), 1, "ghost snapshots the one solid voxel");
            }
            other => panic!("expected SelectMove, got {other:?}"),
        }
        // Drop two cells over on X: the voxel moves as one undo entry.
        app.editor.hovered_voxel = Some(top_hit(2, 0, 0));
        app.on_left_release();
        assert!(matches!(app.interaction, EditInteraction::Idle));
        assert!(app.document.world.get_voxel(0, 0, 0).is_air());
        assert!(!app.document.world.get_voxel(2, 0, 0).is_air());
        assert_eq!(app.editor.history.undo_count(), 1);
        assert_eq!(app.editor.selection.unwrap().min, (2, 0, 0));
    }

    #[test]
    fn escape_cancels_the_shape_but_keeps_the_marquee() {
        let mut app = app_with_tool(Tool::Box);
        app.editor.selection = Some(Selection {
            min: (0, 0, 0),
            max: (1, 1, 1),
        });
        app.editor.hovered_voxel = Some(top_hit(0, 0, 0));
        app.on_left_press();
        app.handle_tool_shortcut(KeyCode::Escape);
        assert!(matches!(app.interaction, EditInteraction::Idle));
        assert!(
            app.editor.selection.is_some(),
            "Esc mid-shape spares the marquee"
        );
        // A second Esc, with no gesture in flight, deselects.
        app.handle_tool_shortcut(KeyCode::Escape);
        assert!(app.editor.selection.is_none());
    }

    #[test]
    fn cancel_returns_any_gesture_to_idle() {
        // The focus-loss / scene-reset verb, over every state.
        for tool in [Tool::Place, Tool::Box, Tool::Select] {
            let mut app = app_with_tool(tool);
            app.editor.hovered_voxel = Some(top_hit(0, 0, 0));
            app.on_left_press();
            app.cancel_interaction();
            assert!(
                matches!(app.interaction, EditInteraction::Idle),
                "{tool:?} gesture must cancel to Idle"
            );
        }
    }

    #[test]
    fn switching_tools_reconciles_shape_and_select_gestures() {
        // A shape gesture dies when the tool leaves the shape family…
        let mut app = app_with_tool(Tool::Box);
        app.editor.hovered_voxel = Some(top_hit(0, 0, 0));
        app.on_left_press();
        app.editor.select_tool(Tool::Place);
        app.update_brush_preview();
        assert!(matches!(app.interaction, EditInteraction::Idle));

        // …but survives a switch within it (Box → Sphere mid-drag).
        let mut app = app_with_tool(Tool::Box);
        app.editor.hovered_voxel = Some(top_hit(0, 0, 0));
        app.on_left_press();
        app.editor.select_tool(Tool::Sphere);
        app.update_brush_preview();
        assert!(matches!(
            app.interaction,
            EditInteraction::ShapeFootprint { .. }
        ));

        // A select drag dies with the Select tool.
        let mut app = app_with_tool(Tool::Select);
        app.editor.hovered_voxel = Some(top_hit(0, 0, 0));
        app.on_left_press();
        app.editor.select_tool(Tool::Place);
        app.update_brush_preview();
        assert!(matches!(app.interaction, EditInteraction::Idle));
    }

    #[test]
    fn arrow_nudges_are_ignored_mid_select_gesture() {
        let mut app = app_with_tool(Tool::Select);
        app.document
            .world
            .set_voxel(0, 0, 0, Voxel::from_rgb(5, 5, 5));
        app.editor.selection = Some(Selection {
            min: (0, 0, 0),
            max: (0, 0, 0),
        });
        app.editor.hovered_voxel = Some(top_hit(0, 0, 0));
        app.on_left_press(); // SelectMove in flight
        app.step_selection((1, 0, 0));
        assert!(
            !app.document.world.get_voxel(0, 0, 0).is_air(),
            "nudge must not fight the drag"
        );
    }

    #[test]
    fn alt_makes_the_eyedropper_effective_without_touching_the_selection() {
        let mut app = app_with_tool(Tool::Place);
        app.modifiers = ModifiersState::ALT;
        assert_eq!(app.effective_tool(), Tool::Eyedropper);
        assert_eq!(
            app.editor.current_tool,
            Tool::Place,
            "persisted tool untouched"
        );
        // With Alt down, a press samples instead of painting.
        app.document
            .world
            .set_voxel(0, 0, 0, Voxel::from_rgb(42, 1, 1));
        app.editor.hovered_voxel = Some(top_hit(0, 0, 0));
        app.on_left_press();
        assert_eq!(app.editor.brush_color.color()[0], 42);
        assert!(
            app.document.world.get_voxel(0, 1, 0).is_air(),
            "nothing painted"
        );
        // Alt up (or the modifiers reset on focus loss): back to Place.
        app.modifiers = ModifiersState::empty();
        assert_eq!(app.effective_tool(), Tool::Place);
    }
}
