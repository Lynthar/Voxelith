//! Input handling: voxel raycast, tool application, keyboard shortcuts.

use winit::keyboard::KeyCode;

use std::collections::HashSet;

use voxelith::editor::{
    box_voxels, build_clear_changes, build_move_changes, build_paste_changes,
    copy_selection_to_clipboard, cylinder_voxels, eyedrop, flood_fill, flood_fill_multi,
    line_voxels, sphere_voxels, BrushTool, Command, EditorTool, Ray, RaycastHit, Selection,
    Tool, ToolContext, VoxelChange, VoxelRaycast,
};

use super::{build_stroke_plane, App, ShapeDrag, ShapePhase, StrokePlane};

/// Maximum distance (in voxel units) the editor's mouse-hover ray
/// will travel through the world looking for a hit. Caps DDA work
/// per cursor move; also implicitly limits how far the user can
/// place / remove voxels from. Sized to comfortably exceed the
/// camera's typical zoom-out distance for 256³-ish scenes — fog
/// (in `voxel.wgsl`) goes to 800, so 500 lets you still click
/// anything you can clearly see.
const RAYCAST_MAX_DIST: f32 = 500.0;

impl App {
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
                if symmetry.any() {
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
                    );
                } else {
                    flood_fill(
                        &mut self.world,
                        &mut self.editor.history,
                        hit.voxel_pos,
                        self.editor.brush_color,
                        10000,
                    );
                }
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
                        return;
                    }
                }
                self.selection_drag_anchor = Some(cell);
            }
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
        // Move mode wins if both anchors happen to be set (defensive
        // — they shouldn't both be set at once).
        if let Some(move_anchor) = self.selection_move_anchor.take() {
            // Cancel any new-selection anchor that snuck in.
            self.selection_drag_anchor = None;
            if let (Some(_sel), Some(hit)) =
                (self.editor.selection, self.editor.hovered_voxel)
            {
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
            return;
        }

        let Some(anchor) = self.selection_drag_anchor.take() else {
            return;
        };
        let Some(hit) = self.editor.hovered_voxel else {
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

        let raw = match tool {
            Tool::Line => line_voxels(anchor, end),
            Tool::Box => box_voxels(anchor, end),
            Tool::Sphere => sphere_voxels(anchor, end),
            Tool::Cylinder => cylinder_voxels(anchor, end),
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
    /// world. Walks loaded chunks, skipping empty ones via
    /// `Chunk::is_empty`. Surfaces "world is empty" if there's
    /// nothing to select.
    pub(super) fn select_all_solid(&mut self) {
        let mut bounds: Option<((i32, i32, i32), (i32, i32, i32))> = None;
        for (chunk_pos, chunk) in self.world.chunks() {
            let chunk = chunk.read();
            if chunk.is_empty() {
                continue;
            }
            let (ox, oy, oz) = chunk_pos.world_origin();
            for (lp, _) in chunk.iter_solid() {
                let p = (
                    ox + lp.x as i32,
                    oy + lp.y as i32,
                    oz + lp.z as i32,
                );
                bounds = Some(match bounds {
                    Some((mn, mx)) => (
                        (mn.0.min(p.0), mn.1.min(p.1), mn.2.min(p.2)),
                        (mx.0.max(p.0), mx.1.max(p.1), mx.2.max(p.2)),
                    ),
                    None => (p, p),
                });
            }
        }
        match bounds {
            Some((min, max)) => {
                self.editor.selection = Some(Selection { min, max });
                let (w, h, d) = (max.0 - min.0 + 1, max.1 - min.1 + 1, max.2 - min.2 + 1);
                self.ui.set_status(format!("Selected all: {}×{}×{}", w, h, d));
            }
            None => {
                self.editor.selection = None;
                self.ui.set_status("World is empty — nothing to select");
            }
        }
    }

    /// Handle keyboard shortcuts (tools, undo/redo, file ops,
    /// selection).
    pub(super) fn handle_tool_shortcut(&mut self, key: KeyCode) {
        match key {
            KeyCode::Digit1 => self.editor.current_tool = Tool::Place,
            KeyCode::Digit2 => self.editor.current_tool = Tool::Remove,
            KeyCode::Digit3 => self.editor.current_tool = Tool::Paint,
            KeyCode::Digit4 => self.editor.current_tool = Tool::Eyedropper,
            KeyCode::Digit5 => self.editor.current_tool = Tool::Fill,
            KeyCode::Digit6 => self.editor.current_tool = Tool::Line,
            KeyCode::Digit7 => self.editor.current_tool = Tool::Box,
            KeyCode::Digit8 => self.editor.current_tool = Tool::Sphere,
            KeyCode::Digit9 => self.editor.current_tool = Tool::Cylinder,
            KeyCode::Digit0 => self.editor.current_tool = Tool::Select,
            KeyCode::KeyZ if self.modifiers.control_key() => {
                if self.modifiers.shift_key() {
                    self.editor.redo(&mut self.world);
                } else {
                    self.editor.undo(&mut self.world);
                }
            }
            KeyCode::KeyY if self.modifiers.control_key() => {
                self.editor.redo(&mut self.world);
            }
            KeyCode::KeyS if self.modifiers.control_key() => {
                if self.modifiers.shift_key() {
                    self.save_project_as();
                } else {
                    self.save_project();
                }
            }
            KeyCode::KeyO if self.modifiers.control_key() => {
                self.open_project();
            }
            KeyCode::KeyN if self.modifiers.control_key() => {
                self.new_project();
            }
            // Esc deselects (Photoshop / image-editor convention).
            // Ctrl+D matches the same convention for users coming
            // from PS / vengi (`Ctrl+D` = select none). Both also
            // abort an in-progress Select drag so the user can bail
            // mid-gesture without committing a stray AABB.
            KeyCode::Escape => {
                self.selection_drag_anchor = None;
                self.editor.selection = None;
                if self.shape_drag.is_some() {
                    self.shape_drag = None;
                    self.ui.set_status("Shape canceled");
                }
            }
            KeyCode::KeyD if self.modifiers.control_key() => {
                self.selection_drag_anchor = None;
                self.editor.selection = None;
            }
            // Selection clipboard ops. Ctrl+Shift+V forces "paste
            // at cursor" (vengi-style two-channel paste); plain
            // Ctrl+V uses the selection's origin if one exists.
            KeyCode::KeyC if self.modifiers.control_key() => {
                self.copy_selection();
            }
            KeyCode::KeyX if self.modifiers.control_key() => {
                self.cut_selection();
            }
            KeyCode::KeyV if self.modifiers.control_key() => {
                let prefer_cursor = self.modifiers.shift_key();
                self.paste_clipboard(prefer_cursor);
            }
            KeyCode::Delete => {
                self.delete_selection();
            }
            // Ctrl+A = select-all-solid: AABB of every non-air
            // voxel in the world. Standard image-editor convention.
            KeyCode::KeyA if self.modifiers.control_key() => {
                self.select_all_solid();
            }
            // Arrow-key selection nudge. ←→ = X axis, ↑↓ = Z axis
            // (matches "screen up = away from camera" for the
            // default camera). Ctrl+↑↓ promotes to the Y axis since
            // four arrows can't cover six 3D directions; Shift
            // multiplies the step by 10 for fast travel.
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
                if self.modifiers.control_key() {
                    self.step_selection((0, step, 0));
                } else {
                    self.step_selection((0, 0, -step));
                }
            }
            KeyCode::ArrowDown => {
                let step = if self.modifiers.shift_key() { 10 } else { 1 };
                if self.modifiers.control_key() {
                    self.step_selection((0, -step, 0));
                } else {
                    self.step_selection((0, 0, step));
                }
            }
            _ => {}
        }
    }
}
