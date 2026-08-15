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

use super::{
    build_stroke_plane, App, PendingAction, ShapeDrag, ShapePhase, StrokePlane,
};

/// Maximum distance (in voxel units) the editor's mouse-hover ray
/// will travel through the world looking for a hit. Caps DDA work
/// per cursor move; also implicitly limits how far the user can
/// place / remove voxels from. Sized to comfortably exceed the
/// camera's typical zoom-out distance for 256³-ish scenes — fog
/// (in `voxel.wgsl`) goes to 800, so 500 lets you still click
/// anything you can clearly see.
const RAYCAST_MAX_DIST: f32 = 500.0;

impl App {
    /// Compute the 3D world anchor for a zoom-to-cursor scroll. Tries
    /// to raycast against world geometry first; if the cursor isn't
    /// over anything solid, falls back to the cursor ray's intersection
    /// with the plane through `camera.target` perpendicular to the
    /// view direction (= same view-depth as the orbit pivot — keeps
    /// the lateral shift sensible when zooming into empty space).
    ///
    /// Returns `None` only when prerequisites (renderer / window)
    /// aren't initialized; callers should fall back to whatever
    /// "no anchor" behavior makes sense (typically just no zoom).
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
        if let Some(hit) = VoxelRaycast::cast(&ray, &self.world, RAYCAST_MAX_DIST) {
            return Some(ray.at(hit.distance));
        }

        // Fallback: project cursor ray onto the plane through
        // `camera.target` perpendicular to view direction. Keeps the
        // anchor at the same view-depth as the current orbit pivot so
        // the resulting target shift is purely lateral, not "into the
        // distance" (which would feel like an unintended dolly).
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

    /// Compute the orbit pivot for a middle-mouse press, Unity-style:
    /// cast a ray straight out of the camera (screen center = camera
    /// forward) and pivot around whatever it hits — a voxel surface,
    /// else the `y = 0` ground plane, else the current target (see
    /// [`VoxelRaycast::orbit_pivot`] for the fallback rationale).
    ///
    /// Casting along the *forward* direction (not the cursor) is
    /// deliberate: the hit lies on the view ray, so re-anchoring
    /// `camera.target` onto it leaves the view direction untouched —
    /// the press only changes the orbit distance, never jumps the
    /// image. `handler.rs` writes the result to `camera.target` before
    /// `process_mouse_button`'s `sync_orbit_state_from_camera` runs, so
    /// the orbit drag immediately rotates around the new pivot.
    ///
    /// Returns `None` only when the renderer isn't initialized yet.
    pub(super) fn compute_orbit_pivot(&self) -> Option<glam::Vec3> {
        let renderer = self.renderer.as_ref()?;
        let camera = &renderer.camera;
        let ray = Ray::new(camera.position, camera.forward());
        Some(VoxelRaycast::orbit_pivot(
            &ray,
            &self.world,
            RAYCAST_MAX_DIST,
            camera.target,
        ))
    }

    /// Move the camera to frame an inclusive cell-coord AABB: put the
    /// orbit target on the box center, then pull back along the *current*
    /// view direction to the fit distance. Framing keeps the user's
    /// viewing angle — only target + distance change (Blender / Unity
    /// "frame" convention), so it never disorients by snapping to a new
    /// orientation.
    ///
    /// Distance is clamped to the orbit zoom range `[2, 500]` so a
    /// following scroll behaves; a scene larger than that hits the cap
    /// (consistent with the reach/fog tuning noted in `CLAUDE.md`).
    pub(super) fn frame_camera_on_aabb(
        &mut self,
        min: (i32, i32, i32),
        max: (i32, i32, i32),
    ) {
        let Some(renderer) = &mut self.renderer else {
            return;
        };
        // Cells occupy [n, n+1), so the box spans min .. max+1 in world.
        let wmin = glam::Vec3::new(min.0 as f32, min.1 as f32, min.2 as f32);
        let wmax =
            glam::Vec3::new(max.0 as f32 + 1.0, max.1 as f32 + 1.0, max.2 as f32 + 1.0);
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
        match self.world.scene_aabb() {
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

    /// Update the editor's hovered voxel from the current cursor position.
    ///
    /// Tools that need an "anchor cell" to place new geometry (Place
    /// and the four shape tools) get a y=0 ground-plane fallback when
    /// the ray misses every voxel — that way they work in a freshly-
    /// cleared (empty) world. Tools that read existing voxels
    /// (Remove/Paint/Eyedropper/Fill) stay strict: virtual hits would
    /// give confusing previews and either no-op or, worse, explode
    /// (Fill flooding a 3D air region).
    ///
    /// **Plane-locked drag-paint takes precedence**: when
    /// `stroke_plane` is set (Place / Remove / Paint left-pressed),
    /// the cursor casts ray-vs-plane against the locked face. This
    /// keeps the stroke on one face instead of stacking along the
    /// view direction as new voxels occlude the ray-vs-voxels hit.
    pub(super) fn update_raycast(&mut self) {
        if let Some(plane) = self.stroke_plane {
            self.editor.hovered_voxel = self.cast_ray_to_plane(&plane);
            return;
        }
        // Shape drag (Footprint or Height phase) also locks the
        // plane — Footprint needs ray-vs-plane to compute the other
        // corner; Height doesn't actually use `hovered_voxel`, but
        // routing through plane lock means a stray cursor move
        // doesn't briefly reveal a "real-world" hit and confuse the
        // preview cache key.
        if let Some(drag) = self.shape_drag {
            self.editor.hovered_voxel = self.cast_ray_to_plane(&drag.plane);
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

        self.editor.hovered_voxel = if self.editor.current_tool.uses_ground_plane_fallback() {
            VoxelRaycast::cast_with_ground_plane(&ray, &self.world, RAYCAST_MAX_DIST, 0)
        } else {
            VoxelRaycast::cast(&ray, &self.world, RAYCAST_MAX_DIST)
        };
    }

    /// Synthesize a `RaycastHit` from a ray-vs-plane intersection
    /// against `plane`. Used during drag-paint to keep the stroke
    /// on the locked face. Returns `None` if the ray is parallel to
    /// the plane or the intersection lies behind the camera (cursor
    /// pointing the wrong way).
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
        // Cap the reach exactly like the voxel-picking cast: a ray nearly
        // parallel to the plane produces an enormous `t`, placing the
        // footprint's far corner millions of cells away — `box_voxels`
        // would then try to build billions of cells (freeze / capacity-
        // overflow panic). Beyond editor reach = no hit, same as picking.
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

    /// Apply the current tool at the hovered location.
    pub(super) fn apply_tool(&mut self) {
        // The shape tools' Height-phase commit is the one action here
        // that doesn't need a hovered voxel: it extrudes by the cursor's
        // screen-Y against the already-locked plane (`extruded_end`).
        // Behind the hover guard below it was silently swallowed exactly
        // when the ray left that plane — building a wall from a low
        // angle, the moment the cursor crossed the plane's screen
        // horizon the commit click did nothing, with no message, while
        // the preview still showed the full extrusion.
        if self.editor.current_tool.is_shape()
            && matches!(
                self.shape_drag,
                Some(ShapeDrag {
                    phase: ShapePhase::Height { .. },
                    ..
                })
            )
        {
            self.commit_shape();
            return;
        }

        let Some(hit) = self.editor.hovered_voxel else {
            return;
        };

        match self.editor.current_tool {
            Tool::Place | Tool::Remove | Tool::Paint => {
                // Lock the stroke to the first hit's face plane.
                // Subsequent CursorMoved events (drag-paint) will
                // ray-vs-plane against this lock instead of the
                // voxel world — so paint stays on one face instead
                // of stacking toward the camera. The lock is
                // released in `handler.rs` on left-up.
                if self.stroke_plane.is_none() {
                    self.stroke_plane = build_stroke_plane(&hit);
                }
                let brush = BrushTool::new(self.editor.current_tool);
                let mut ctx = ToolContext {
                    world: &mut self.world,
                    history: &mut self.editor.history,
                    brush_color: self.editor.brush_color,
                    brush_size: self.editor.brush_size,
                    symmetry: self.editor.symmetry,
                };
                brush.apply(&mut ctx, &hit);
            }
            Tool::Eyedropper => {
                if let Some(color) = eyedrop(&self.world, &hit) {
                    self.editor.brush_color = color;
                }
            }
            Tool::Fill => {
                // Refuse to flood from an air cell: with Place's ground-
                // plane fallback in play the hit could in principle be a
                // virtual sub-plane voxel, and flooding from there would
                // eat the entire 3D air region around the cursor (capped
                // by `flood_fill`'s spatial limit, but still visually
                // alarming and never what the user meant).
                let v = self.world.get_voxel(
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
                        &mut self.world,
                        &mut self.editor.history,
                        &starts,
                        self.editor.brush_color,
                        10000,
                    )
                } else {
                    flood_fill(
                        &mut self.world,
                        &mut self.editor.history,
                        hit.voxel_pos,
                        self.editor.brush_color,
                        10000,
                    )
                };
                self.report_fill(outcome);
            }
            Tool::Line | Tool::Box | Tool::Sphere | Tool::Cylinder => {
                // Shape press is two-phase:
                //   - First press (drag is None): enter Footprint —
                //     lock the plane from the hit's face, anchor at
                //     `adjacent_pos`. Subsequent CursorMoved walks
                //     ray-vs-plane to find the W×D corner.
                //   - Second press (drag is in Height phase): commit
                //     the extruded shape and clear the drag.
                //   - Press while still in Footprint shouldn't happen
                //     (the second press only fires after release
                //     transitions us to Height); ignore defensively.
                match self.shape_drag {
                    None => {
                        if let Some(plane) = build_stroke_plane(&hit) {
                            self.shape_drag = Some(ShapeDrag {
                                anchor: hit.adjacent_pos,
                                plane,
                                phase: ShapePhase::Footprint,
                            });
                        } else {
                            self.ui.set_status(
                                "Shape tool: face normal not axis-aligned, ignoring click",
                            );
                        }
                    }
                    Some(ShapeDrag {
                        phase: ShapePhase::Height { .. },
                        ..
                    }) => {
                        self.commit_shape();
                    }
                    Some(ShapeDrag {
                        phase: ShapePhase::Footprint,
                        ..
                    }) => {
                        // Defensive: ignore.
                    }
                }
            }
            Tool::Select => {
                // Selection press splits two ways:
                //   - Inside an existing selection → move mode.
                //   - Anywhere else → start a fresh selection drag.
                // `select_anchor_pos` picks the hit voxel for real
                // hits and the plane cell for virtual-ground hits,
                // so empty-world drags don't sink one cell
                // underground.
                let cell = Self::select_anchor_pos(&hit);
                if let Some(sel) = self.editor.selection {
                    if sel.contains(cell) {
                        self.selection_move_anchor = Some(cell);
                        self.begin_move_ghost(sel);
                        return;
                    }
                }
                self.selection_drag_anchor = Some(cell);
            }
            Tool::Socket => {
                // Drop a named attachment point at the center of the
                // clicked face, oriented along its outward normal.
                // Single click — no drag, no release-commit, not
                // undoable (managed via the Tools panel, like the
                // selection). `drag_eligible` in `handler.rs` excludes
                // Socket, so a held-drag never spams duplicates.
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
                let name = voxelith::editor::next_socket_name(&self.editor.sockets);
                self.editor
                    .sockets
                    .push(voxelith::editor::Socket::new(name.clone(), position, normal));
                // Sockets are document data no mesh rebuild notices —
                // placing one has to raise the unsaved flags itself, or
                // "place a socket, quit" lost it without a prompt.
                self.mark_document_modified();
                self.ui.set_status(format!(
                    "Placed {} at ({:.1}, {:.1}, {:.1})",
                    name, position[0], position[1], position[2]
                ));
            }
        }
    }

    /// Largest selection AABB (in cells, air included) the dense sweep
    /// operations will walk — 256³, a one-to-two-second worst case.
    /// Copy / cut / delete / move / rotate / mirror all iterate every
    /// cell of the box including air, so two voxels a million cells
    /// apart plus Ctrl+A used to turn any of them into an hours-long
    /// freeze. (Worlds like that are routine on the agent side, whose
    /// coordinate ceiling is ±1,048,576.) The marquee itself may be any
    /// size; only the sweeps are bounded.
    pub(super) const MAX_SELECTION_SWEEP_CELLS: i64 = 16_777_216;

    /// Cells in a selection's AABB, in `i64` — the `i32` arithmetic in
    /// `Selection::size` wraps for exactly the boxes this guards.
    fn selection_sweep_cells(sel: &Selection) -> i64 {
        let extent = |a: i32, b: i32| (b as i64 - a as i64) + 1;
        extent(sel.min.0, sel.max.0)
            * extent(sel.min.1, sel.max.1)
            * extent(sel.min.2, sel.max.2)
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

    /// Commit the in-progress selection drag on left-button release.
    /// Two paths:
    /// - **Move drag** (`selection_move_anchor` set): translate the
    ///   selection's voxels by `current - anchor` as a single
    ///   `SetVoxels` Command, then update the AABB.
    /// - **New-selection drag** (`selection_drag_anchor` set): build
    ///   a `Selection` from the press anchor → current hover cell
    ///   and store it on the editor.
    ///
    /// Selection state itself is *not* pushed onto the undo history
    /// — the marquee is ephemeral, like in image editors. Move's
    /// voxel writes *are* undoable through their `SetVoxels`.
    pub(super) fn commit_selection(&mut self) {
        // A release always ends any in-flight move drag — drop the
        // ghost snapshot so a large moved region doesn't linger in
        // memory (the renderer slot itself clears once the anchor is
        // gone).
        self.move_ghost_voxels.clear();

        // Move mode wins if both anchors happen to be set (defensive
        // — they shouldn't both be set at once).
        if let Some(move_anchor) = self.selection_move_anchor.take() {
            // Cancel any new-selection anchor that snuck in.
            self.selection_drag_anchor = None;
            match (self.editor.selection, self.editor.hovered_voxel) {
                (Some(_sel), Some(hit)) => {
                    let cur = Self::select_anchor_pos(&hit);
                    let delta = (
                        cur.0 - move_anchor.0,
                        cur.1 - move_anchor.1,
                        cur.2 - move_anchor.2,
                    );
                    if delta != (0, 0, 0) {
                        self.move_selection(delta);
                    }
                }
                // Released with the ray off the world entirely (cursor
                // above the horizon, past the raycast reach, or nearly
                // parallel to the ground plane). Say so: the ghost
                // snaps back to the original spot, and silence made
                // that read as "the move didn't take" with no clue why.
                // Matches the footprint-off-plane message shape.
                (Some(_), None) => {
                    self.ui
                        .set_status("Move canceled (cursor off-world on release)");
                }
                _ => {}
            }
            return;
        }

        let Some(anchor) = self.selection_drag_anchor.take() else {
            return;
        };
        let Some(hit) = self.editor.hovered_voxel else {
            self.ui
                .set_status("Selection canceled (cursor off-world on release)");
            return;
        };
        let end = Self::select_anchor_pos(&hit);
        self.editor.selection = Some(Selection::from_corners(anchor, end));
    }

    /// Translate the active selection's non-air voxels by `delta` as
    /// a single `SetVoxels` Command (so one Ctrl+Z undoes the whole
    /// move). Updates `editor.selection` to the translated AABB.
    /// Overlap handling lives in `build_move_changes`.
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
        let changes = build_move_changes(&self.world, sel, delta);
        if !changes.is_empty() {
            let cmd = Command::set_voxels(changes);
            self.editor.history.execute(cmd, &mut self.world);
        }
        // Even an empty selection (all air) bumps its AABB so the
        // user can keyboard-nudge a marquee around empty space.
        self.editor.selection = Some(sel.translated(delta));
    }

    /// Transition an in-progress shape drag from Footprint to
    /// Height phase on left-button release. The cursor's current
    /// plane-locked hit becomes the locked footprint corner, and
    /// its screen-Y becomes the baseline that future cursor moves
    /// measure against to set extruded height.
    ///
    /// If the cursor is off-world at release (no plane hit), cancel
    /// the drag — committing a shape with no second corner would
    /// produce a single-cell at the anchor, which is almost never
    /// what the user wants.
    pub(super) fn transition_shape_to_height(&mut self) {
        let Some(drag) = self.shape_drag else {
            return;
        };
        if !matches!(drag.phase, ShapePhase::Footprint) {
            return;
        }
        let Some(hit) = self.editor.hovered_voxel else {
            self.shape_drag = None;
            self.ui
                .set_status("Shape canceled (cursor off-plane on release)");
            return;
        };
        self.shape_drag = Some(ShapeDrag {
            anchor: drag.anchor,
            plane: drag.plane,
            phase: ShapePhase::Height {
                end_on_plane: hit.adjacent_pos,
                release_screen_y: self.cursor_pos.1,
            },
        });
        self.ui
            .set_status("Drag vertically to set height, click to commit (Esc cancels)");
    }

    /// Rotate the active selection's contents around `axis` by
    /// `quarter` (90° / -90° / 180°). The selection's AABB may
    /// change footprint (Y-rotation swaps W ↔ D, etc.) but its
    /// `min` corner stays put — see `editor::transform` for the
    /// anchor convention. Result is one `Command::set_voxels` so
    /// Ctrl+Z reverses the entire rotation.
    pub(super) fn rotate_selection(&mut self, axis: Axis, quarter: Quarter) {
        let Some(sel) = self.editor.selection else {
            self.ui
                .set_status("No selection — drag with the Select tool first");
            return;
        };
        if self.refuse_oversized_sweep(sel, "Rotate") {
            return;
        }
        let (new_sel, changes) =
            rotate_selection_changes(&self.world, sel, axis, quarter);
        let count = changes.len();
        if !changes.is_empty() {
            let cmd = Command::set_voxels(changes);
            self.editor.history.execute(cmd, &mut self.world);
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

    /// Report what a Fill click did, the way delete / cut / rotate
    /// already report. Fill was the one destructive tool that said
    /// nothing at all, so a click that hit a cap — or that changed
    /// nothing because the region was already the brush color — was
    /// indistinguishable from a click that missed.
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
        let changes = mirror_selection_changes(&self.world, sel, axis);
        let count = changes.len();
        if !changes.is_empty() {
            let cmd = Command::set_voxels(changes);
            self.editor.history.execute(cmd, &mut self.world);
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
        if self.selection_drag_anchor.is_some() || self.selection_move_anchor.is_some() {
            return;
        }
        if self.editor.selection.is_none() {
            return;
        }
        self.move_selection(delta);
    }

    /// Commit the in-progress shape drag. Called on the second
    /// click (after the user has dragged a footprint, released, and
    /// then optionally moved the cursor vertically to set height).
    /// Reads anchor + plane + phase from `shape_drag` and clears it.
    /// No-op if there's no active drag.
    ///
    /// Footprint-only commit (no Height phase reached) treats height
    /// as 0 — the shape is one cell thick along the plane normal,
    /// matching the Goxel `planar=on` single-click flow.
    pub(super) fn commit_shape(&mut self) {
        let Some(drag) = self.shape_drag.take() else {
            return;
        };
        let tool = self.editor.current_tool;
        let cursor_y = self.cursor_pos.1;

        let (anchor, end) = match drag.phase {
            ShapePhase::Footprint => {
                // Defensive — second-click commit should always come
                // from Height phase. If we somehow get here from
                // Footprint, fall back to the cursor's current
                // plane-locked cell.
                let Some(hit) = self.editor.hovered_voxel else {
                    return;
                };
                (drag.anchor, hit.adjacent_pos)
            }
            ShapePhase::Height { .. } => {
                let end = drag.extruded_end(cursor_y).expect("Height phase");
                (drag.anchor, end)
            }
        };

        // Budget check BEFORE the enumeration it bounds — a glancing
        // drag can legally describe hundreds of millions of cells, and
        // materializing them freezes the frame loop or exhausts memory.
        // Same accounting as the preview; the preview's smaller budget
        // only limits per-cursor-step work, so a shape that outgrew its
        // preview can still commit under this cap.
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
            Tool::Cylinder => cylinder_voxels(anchor, end, Some(drag.plane.axis)),
            _ => return, // anchor only set for shape tools, defensive
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
                old_voxel: self.world.get_voxel(pos.0, pos.1, pos.2),
                new_voxel: color,
            })
            .filter(|c| c.old_voxel != c.new_voxel)
            .collect();

        if !changes.is_empty() {
            let cmd = Command::set_voxels(changes);
            self.editor.history.execute(cmd, &mut self.world);
        }
    }

    /// Capture the active selection's non-air voxels into the
    /// clipboard. No-op (with a status hint) if there's no selection.
    pub(super) fn copy_selection(&mut self) {
        let Some(sel) = self.editor.selection else {
            self.ui.set_status("No selection — drag with the Select tool first");
            return;
        };
        if self.refuse_oversized_sweep(sel, "Copy") {
            return;
        }
        let clipboard = copy_selection_to_clipboard(&self.world, sel);
        let count = clipboard.voxel_count();
        self.clipboard = Some(clipboard);
        if count == 0 {
            self.ui.set_status("Selection contains no solid voxels");
        } else {
            self.ui.set_status(format!("Copied {} voxels", count));
        }
    }

    /// Cut: snapshot the selection into the clipboard, then clear
    /// every non-air cell inside the selection in a **single**
    /// `Command::set_voxels`. Critical that it's one Command — if we
    /// pushed Copy + Delete separately, Ctrl+Z would only restore
    /// half the cut, which is the textbook reverse-intuitive bug.
    pub(super) fn cut_selection(&mut self) {
        let Some(sel) = self.editor.selection else {
            self.ui.set_status("No selection — drag with the Select tool first");
            return;
        };
        if self.refuse_oversized_sweep(sel, "Cut") {
            return;
        }
        let clipboard = copy_selection_to_clipboard(&self.world, sel);
        let count = clipboard.voxel_count();
        self.clipboard = Some(clipboard);

        let changes = build_clear_changes(&self.world, sel);
        if !changes.is_empty() {
            let cmd = Command::set_voxels(changes);
            self.editor.history.execute(cmd, &mut self.world);
        }

        if count == 0 {
            self.ui.set_status("Selection had no solid voxels — clipboard empty");
        } else {
            self.ui.set_status(format!("Cut {} voxels", count));
        }
    }

    /// Delete: clear non-air cells inside the selection without
    /// touching the clipboard.
    pub(super) fn delete_selection(&mut self) {
        let Some(sel) = self.editor.selection else {
            self.ui.set_status("No selection — drag with the Select tool first");
            return;
        };
        if self.refuse_oversized_sweep(sel, "Delete") {
            return;
        }
        let changes = build_clear_changes(&self.world, sel);
        let count = changes.len();
        if !changes.is_empty() {
            let cmd = Command::set_voxels(changes);
            self.editor.history.execute(cmd, &mut self.world);
        }
        if count == 0 {
            self.ui.set_status("Selection had no solid voxels to delete");
        } else {
            self.ui.set_status(format!("Deleted {} voxels", count));
        }
    }

    /// Paste the clipboard at:
    /// - **selection origin** when `prefer_cursor == false` and a
    ///   selection exists (Ctrl+V — typical "paste back where the
    ///   selection is");
    /// - **hovered cell** otherwise (Ctrl+V with no selection, OR
    ///   Ctrl+Shift+V regardless of selection — vengi-style "paste
    ///   to cursor").
    ///
    /// After pasting, auto-select the destination AABB so a
    /// subsequent Paste (or M3 drag-move) chains naturally without
    /// re-marqueeing — abuses vengi's `autoSelectSolidVoxels` trick.
    pub(super) fn paste_clipboard(&mut self, prefer_cursor: bool) {
        let Some(clipboard) = self.clipboard.as_ref() else {
            self.ui.set_status("Clipboard is empty — Copy / Cut a selection first");
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
            self.ui.set_status("Move the cursor over the world to paste");
            return;
        };

        let changes = build_paste_changes(&self.world, clipboard, dest);
        let count = changes.len();
        if !changes.is_empty() {
            let cmd = Command::set_voxels(changes);
            self.editor.history.execute(cmd, &mut self.world);
        }

        // Auto-select the destination AABB so the user can chain
        // Paste→drag→Paste without re-marqueeing.
        let (sw, sh, sd) = clipboard.size;
        self.editor.selection = Some(Selection {
            min: dest,
            max: (dest.0 + sw - 1, dest.1 + sh - 1, dest.2 + sd - 1),
        });

        if count == 0 {
            self.ui.set_status("Pasted (no changes — destination already matched)");
        } else {
            self.ui.set_status(format!("Pasted {} voxels", count));
        }
    }

    /// Set the selection to the AABB of every non-air voxel in the
    /// world (`World::scene_aabb`, the same bounds Frame All and the
    /// exporters use). Surfaces "world is empty" if there's nothing
    /// to select.
    pub(super) fn select_all_solid(&mut self) {
        match self.world.scene_aabb() {
            Some((min, max)) => {
                self.editor.selection = Some(Selection { min, max });
                let (w, h, d) = (max.0 - min.0 + 1, max.1 - min.1 + 1, max.2 - min.2 + 1);
                self.ui.set_status(format!("Selected all: {}×{}×{}", w, h, d));
            }
            None => {
                // Through `deselect`, not by assigning `None`: a
                // selection is also the drag anchor, the move anchor and
                // the translucent move ghost, and clearing only the
                // field leaves those on screen.
                self.deselect();
                self.ui.set_status("World is empty — nothing to select");
            }
        }
    }

    /// Clear all box-selection state: the marquee, an in-progress
    /// select-drag or move-drag anchor, and the translucent move ghost.
    /// Shared by the `Deselect` UI action, Esc, and Ctrl+D so the three
    /// entry points can't drift — Esc / Ctrl+D used to omit the move
    /// anchor + ghost, stranding a ghost after a cancelled move.
    pub(super) fn deselect(&mut self) {
        self.selection_drag_anchor = None;
        self.selection_move_anchor = None;
        self.move_ghost_voxels.clear();
        self.editor.selection = None;
    }

    /// The platform's command modifier: ⌘ on macOS, Ctrl elsewhere.
    /// Every chord shortcut (save / undo / copy …) keys off this.
    /// Checking `control_key()` alone left ⌘S dead on macOS — the
    /// event layer had already classified the press as a command chord
    /// and kept it from the fly camera, so the key appeared to do
    /// nothing at all.
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

    /// Any command modifier at all — the same classification
    /// `handler.rs` uses to keep chords out of the fly camera. The
    /// bare-letter shortcuts (R / M) are guarded on this rather than on
    /// Ctrl alone: guarding only Ctrl left ⌘R / ⌘M on macOS falling
    /// through to rotate / mirror the selection.
    fn command_chord(&self) -> bool {
        self.modifiers.control_key() || self.modifiers.super_key()
    }

    /// Handle keyboard shortcuts (tools, undo/redo, file ops,
    /// selection).
    pub(super) fn handle_tool_shortcut(&mut self, key: KeyCode) {
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
            KeyCode::KeyZ if self.primary_modifier() => {
                let stepped = if self.modifiers.shift_key() {
                    self.editor.redo(&mut self.world, &mut self.ui.graph)
                } else {
                    self.editor.undo(&mut self.world, &mut self.ui.graph)
                };
                if stepped {
                    // Voxel entries are flagged by the mesh rebuild; a
                    // graph-only transition reaches no chunk, so the
                    // step has to say so itself.
                    self.mark_document_modified();
                }
            }
            KeyCode::KeyY if self.primary_modifier() => {
                if self.editor.redo(&mut self.world, &mut self.ui.graph) {
                    self.mark_document_modified();
                }
            }
            KeyCode::KeyS if self.primary_modifier() => {
                if self.modifiers.shift_key() {
                    self.save_project_as();
                } else {
                    self.save_project();
                }
            }
            // Both go through the unsaved-changes guard, same as the
            // File menu — the guard lives on `App` precisely because
            // these two reach the file ops without passing the UiAction
            // queue.
            KeyCode::KeyO if self.primary_modifier() => {
                self.guard_then(PendingAction::OpenPicker);
            }
            KeyCode::KeyN if self.primary_modifier() => {
                self.guard_then(PendingAction::NewProject);
            }
            // Esc: cancel the modal interaction first — an in-progress
            // shape drag — and only deselect when there is none. Doing
            // both at once meant bailing out of a shape also silently
            // threw away the marquee the user had set up before it.
            // Deselect follows the Photoshop / image-editor convention;
            // Ctrl+D matches it for users coming from PS / vengi.
            KeyCode::Escape => {
                if self.shape_drag.is_some() {
                    self.shape_drag = None;
                    self.ui.set_status("Shape canceled");
                } else {
                    self.deselect();
                }
            }
            KeyCode::KeyD if self.primary_modifier() => {
                self.deselect();
            }
            // Selection clipboard ops. Ctrl+Shift+V forces "paste
            // at cursor" (vengi-style two-channel paste); plain
            // Ctrl+V uses the selection's origin if one exists.
            KeyCode::KeyC if self.primary_modifier() => {
                self.copy_selection();
            }
            KeyCode::KeyX if self.primary_modifier() => {
                self.cut_selection();
            }
            KeyCode::KeyV if self.primary_modifier() => {
                let prefer_cursor = self.modifiers.shift_key();
                self.paste_clipboard(prefer_cursor);
            }
            KeyCode::Delete => {
                self.delete_selection();
            }
            // Ctrl+A / ⌘A = select-all-solid: AABB of every non-air
            // voxel in the world. Standard image-editor convention.
            KeyCode::KeyA if self.primary_modifier() => {
                self.select_all_solid();
            }
            // Rotate / mirror the active selection (no-op with a status
            // hint if there's none). R spins around Y — the common
            // "turn it around" — and Shift+R reverses; M flips
            // left-right across X. The full axis × angle set lives in
            // the Selection menu. Guarded against every command
            // modifier so a stray Ctrl+R / ⌘R / ⌘M can't silently
            // transform geometry.
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
            // Arrow-key selection nudge. ←→ = X axis, ↑↓ = Z axis
            // (matches "screen up = away from camera" for the
            // default camera). Ctrl+↑↓ (⌘ on macOS) promotes to the Y
            // axis since four arrows can't cover six 3D directions;
            // Shift multiplies the step by 10 for fast travel.
            //
            // Skipped (via `step_selection` guards) when there's no
            // selection or a mouse drag is mid-flight, so the user
            // can't fight a drag with the keyboard.
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
            // F frames the view: the selection's AABB if one exists,
            // else the whole scene. Beyond a bare recenter it also fits
            // the camera *distance* to the box (frame-selected /
            // frame-all) while keeping the current viewing angle — only
            // target + distance move, the orientation doesn't snap.
            // Recovery hatch for WASD-flying / panning off the model.
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
