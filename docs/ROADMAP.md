# Voxelith Roadmap

The prioritized, dependency-aware plan from where the editor is today to the
long-term vision. This sits between the other two docs:

- [`PROGRESS.md`](./PROGRESS.md) — **what's built** (authoritative current state).
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — **the far vision** (Phase 1–7, not pruned).
- **`ROADMAP.md`** (this file) — **the plan** to get from one to the other.

When an item ships: check it off here and fold a one-line summary into
`PROGRESS.md`. Keep this file the *plan*, not a second status log.

## How to read this

- `- [ ]` todo · `- [x]` done.
- Status tag per item:
  - **done** — shipped (also checked).
  - **surface** — the backend / data already exists; the work is mostly UI exposure. **Cheapest wins — harvest first.**
  - **partial** — some of it exists; the box lists what's left.
  - **new** — build from scratch.
- **needs:** hard dependency (do that first). **enables:** what this unblocks.
- Effort is a rough **S / M / L** (hours / a day / multi-day).
- Priorities P0–P5 are *thematic buckets*. The actual order to build is the
  linear list at the bottom (["Recommended build sequence"](#recommended-build-sequence)),
  which crosses buckets per the trust-first / harvest-cheap-wins logic.

---

## P0 — Don't break the illusion (correctness & trust)

> Mostly cleared. Finish 0.2 + 0.3 while the importer/AI code is fresh.

### 0.1 High-priority review fixes — **done**
- [x] AI result schema parses `model_glb` / `model_urls.glb` (not the dead `model_mesh`) — `src/ai/client.rs`
- [x] Render residue: clear `chunk_meshes` on every scene replace — `App::replace_scene` (`src/app/ui_actions.rs`)
- [x] Importer robustness: bounded allocations in .vxlt / .vox — `io::read_exact_vec` / `io::skip_bytes`
- [x] Keyboard ↔ camera crosstalk: drop Ctrl/Super presses to the fly-cam, clear keys on focus loss, sprint moved to Shift — `src/app/handler.rs`, `src/render/camera.rs`
- [x] Remote cancel: best-effort `PUT cancel_url` from the poll loop — `fal_cancel` (`src/ai/client.rs`)

### 0.2 Quality gate — **partial**
- [x] AI JSON fixture tests pin the schema — `src/ai/client.rs` tests
- [x] Procgen odd/even dimension test — `src/procgen/terrain.rs`
- [x] Importer bound tests (`read_exact_vec` / `skip_bytes`, "claims 4 GiB, has 4 bytes") — `src/io/mod.rs` tests
- [x] .vxlt save→open roundtrip test (editor-state + negative/multi-chunk voxels) — `test_roundtrip_preserves_editor_state_and_cross_chunk_voxels`
- [x] .vox import golden test — already covered: v150 export→import roundtrip + hand-built v200 fixtures (`v200_ntrn_translation_offsets_single_model`, `v200_multi_model_with_separate_translations`, `v200_skips_unknown_chunks_safely`)
- [x] CI: build + test gate on windows-latest — `.github/workflows/ci.yml`
- [ ] **new · S** Clippy/format cleanup pass → flip CI clippy to `-D warnings` and make `cargo fmt --check` a gate (both informational today; codebase has ~6 pre-existing lints + a deliberate narrow manual format)

### 0.3 Errors: "it failed" → "why + recovery action" — **done**
- [x] Import failure: specific reason (wrong file / unsupported version / too large / truncated / corrupt-chunk) + recovery action in an in-app egui dialog, concise status one-liner — `describe_vox_import_error` (`src/app/file_ops.rs`)
- [x] Open failure: same treatment for .vxlt (`describe_project_open_error`)
- [x] Save / export failure: in-app write-error dialog (permission / disk / path + try-different-location); save status says "your work is NOT saved" — `App::show_write_error`
- [x] AI failure: messages already carry the stage (Submit / Fetch result / Parsing result / …); added a 401/403 "check your API key" hint — `src/ai/client.rs`
- note: full per-field forensic detail (exact chunk offset / length field) is deferred to **2.1 pre-import inspection**, its natural home; the dialogs avoid overflowing the single-line status bar
- note: these dialogs (and the crash-recovery prompt) are **in-app egui windows, not `rfd::MessageDialog`** — calling the latter exits the process on the dev's winit+wgpu+Windows setup, which would crash exactly on a file-op failure (`rfd::FileDialog` is unaffected; see CLAUDE.md)

---

## P1 — Editing experience

### 1.1 Selection / move polish — *the mainline* — **done**
- [x] AABB + live status readout `Sel: W×H×D (N cells)`
- [x] Move-drag wireframe ghost (`selection.translated`)
- [x] Paste auto-selects the destination AABB
- [x] Center + pivot markers on the selection wireframe (cyan center crosshair + orange `min`-corner anchor — `render/selection.rs`)
- [x] Voxel-content ghost during move-drag (not just the wireframe) — translucent snapshot of the picked-up voxels on its own `move_ghost_mesh` overlay slot, re-translated by the live drag delta each frame (`App::update_selection_visualization` / `begin_move_ghost`)
- [x] Selection follows moved content (parity with paste auto-select) — already satisfied by `move_selection` (`src/app/input.rs`, shared by drag-commit + arrow-nudge): it ends with `editor.selection = sel.translated(delta)`, so the marquee lands on the moved voxels
- [x] Frame-selected camera (`F` with a selection, or the "Frame Sel." button)
- [x] *(extra)* Rotate / mirror the selection from the keyboard (`R` / `Shift+R` / `M`); the menu items show the keys

### 1.2 Low-interference viewport HUD — **done**
- [x] Edge HUD: bottom-left translucent **click-through** block (vengi-BrushHud-style) showing active tool, gesture phase + live numbers (shape footprint/height dims, move delta, marquee size), locked plane, symmetry, selection size (Select tool only — the status bar keeps the always-on copy), and modal key hints — `src/ui/hud.rs` + `src/app/hud.rs`; toggle in View menu + Viewport Settings, persisted. Brush *size* deliberately left out: the hover preview already visualizes the radius in-scene and the number stays in the status bar

### 1.3 Keybindings & camera presets — **scope-split**
- [ ] **new · S** Camera navigation presets (Blender / MagicaVoxel / Goxel) — *do this half first*
- [ ] **new · L** *(deferred)* Fully configurable keymap + conflict detection + searchable key help

### 1.4 Brush / paint detail
- [ ] **new · M** surface-only paint mode
- [ ] **new · M** replace-color tool
- [ ] **new · M** paint-only-selected (writes masked to the selection) · needs: 1.1
- [ ] **new · S** recent colors
- [ ] **new · S** palette slot naming
- [ ] **new · S** region-fill preview · brush preview slot already exists

### 1.5 Undo history panel — **new**
- [ ] **new · M** History list ("Paint stroke / Move selection / Generate terrain"); view-only first, time-travel later · needs: human labels on `Command`

---

## P2 — Files & export

### 2.1 Pre-import inspection — **new**
- [ ] **new · M** Opening .vox / .vxlt first shows model count, dims, voxel count, palette, warnings — confirm before committing · needs: 2.4 (shared "peek without loading" path), 0.3

### 2.2 Export presets — **new (data exists)**
- [ ] **new · M** Godot / Unity / Blender / Web / MagicaVoxel presets (format + axis + scale + smoothing baked into one click) · needs: 2.3 data

### 2.3 Export report — **surface**
- [ ] **surface · S** Report dialog: tris, materials, file size, greedy vs MC, any lost material info · data already in `GlbStats` (`src/io/gltf.rs`) / `ObjStats` (`src/io/obj.rs`)

### 2.4 Autosave + crash recovery — **done**
- [x] Timed autosave (60 s) to `…/voxelith/autosave.vxlt` while dirty + non-empty — `App::tick_autosave`; dirty detected via the single `rebuild_all_meshes` chokepoint (dirty chunks ⟺ voxels changed)
- [x] Crash detection = delete-on-clean-exit (CloseRequested + UiAction::Exit); presence at launch ⟹ unclean shutdown
- [x] Recover-on-launch native Yes/No prompt — `App::try_recover_or_initial_scene` → `recover_from_autosave` (loads with `project_path = None` so Save prompts; a corrupt/truncated autosave is discarded → default scene, never bricks startup) · enables: 2.1, 2.5
- note: filesystem/renderer flow isn't unit-tested (needs a running app); data path is covered by the .vxlt roundtrip tests. Manual smoke: edit → wait 60 s → kill the process → relaunch → expect the recover prompt

### 2.5 .vxlt version migration — **new**
- [ ] **new · M** Explicit schema version + migration path + clear "file from a newer/older build" error · needs: 2.4 (shared load path), 0.2 roundtrip test

---

## P3 — Procgen & graph

### 3.1 Controllable preview — **partial**
- [x] Debounced regen (150 ms) + param change-detection + stale flag — `src/app/preview.rs`
- [ ] **surface · S** Show estimated time, voxel count, debounce / stale status · `estimate_duration()` already exists on generators
- [ ] **new · M** Cancel for large generators · needs: generator run off the main thread

### 3.2 Commit semantics — **new**
- [ ] **new · M** Overwrite / Add / New-layer / Into-selection on apply · needs: 1.1 (into-selection); layers aren't a concept yet (scope check before committing)

### 3.3 Node diagnostics on the canvas — **surface**
- [ ] **surface · S** Paint cycle / missing-input / no-Output / empty-output **on the node itself** · `GraphError` (`src/procgen/graph.rs`) already detects all of these; today they only reach the status bar
- [ ] **new · S** Oversized-output warning

### 3.4 Node caching across evaluations — **deferred**
- [ ] **new · M** *(after 3.5)* Memoize unchanged upstream by (params + upstream hash) · note: `evaluate()` already memoizes *within* one run — this is the *cross-run* cache

### 3.5 Graph templates / examples — **new (do before 3.4)**
- [ ] **new · M** Built-in presets: "terrain + trees above surface", "dungeon room", "city block" · proves the graph is worth optimizing → justifies 3.4

---

## P4 — AI experience

> Treat 4.1 / 4.3 / 4.4 as **one feature**, not three items — staging only feels
> right with a GLB cache (free re-voxelize) and auto-select underneath it.

### 4.1 Staging instead of dump-into-world — **new (flagship)**
- [ ] **new · L** AI result lands in a staging area: preview, move, scale, accept / discard · needs: 4.3 cache, transparent overlay (exists) · replaces the current straight-to-world `apply_ai_patch`

### 4.2 Cost / time expectations — **new**
- [ ] **new · S** Show expected cost + ETA before submit (matters for a paid remote API)

### 4.3 Prompt history + GLB cache — **new**
- [ ] **new · M** Prompt MRU + favorites + seed · uses existing prefs
- [ ] **new · M** Cache the raw GLB; re-voxelize without re-charging · enables: 4.1 (try resolutions for free)

### 4.4 Auto-select the AI result — **new**
- [ ] **new · S** Select the result AABB on apply so it's immediately move/copy/delete-able · needs: 1.1, 4.1

### 4.5 Remote cancel as a UX contract — **done; surface remaining**
- [x] `PUT cancel_url` on cancel — `fal_cancel`
- [ ] **surface · S** Reflect "remote cancelled" explicitly in the AI panel status

---

## P5 — Viewport & feedback

### 5.1 Framing — **done**
- [x] Top / Front / Side presets — `SetCameraView` (the `recenter_camera_on_scene` helper is still used on world replacement)
- [x] Fit-distance framing: frame-all / frame-selected / frame-generated — fits the camera *distance* to the AABB along the current view direction (`Camera::fit_distance`, `App::frame_camera_on_aabb`). `F` frames the selection or, with none, the whole scene; buttons in the viewport menu. `last_generated_bounds` tracks the most recent procgen/graph/AI footprint for "frame generated"

### 5.2 Viewport settings — **new**
- [ ] **new · M** Grid / axis / fog / clip distance / background / light intensity panel · note: keep clip-distance & fog co-tuned (see CLAUDE.md "reach ↔ fog")

### 5.3 Measure tool — **new (cheap, rides on 1.1)**
- [ ] **new · S** Distance / voxel count / bbox readout · needs: raycast + selection AABB (both exist)

### 5.4 Turntable / screenshot — **deferred**
- [ ] **new · M** *(after editing + file trust)* Screenshot export + turntable preview

### 5.5 Perf HUD — **done**
- [x] Bottom-right click-through overlay (same `Area` contract as the 1.2 HUD, default **off**): FPS + frame ms (CPU frame interval off the `frame_times` ring — true GPU timestamp queries stay out of scope), compact tris + chunk count, and the new mesh-rebuild sample (`rebuild 4.2 ms (3 chunks)`, timed in `rebuild_all_meshes` over generation + upload) — `ui/hud.rs::show_perf_overlay`, toggles beside the viewport HUD, persisted

---

## Dependency graph (load-bearing edges)

```
0.1 (done) ──> 0.3 error messages
0.2 roundtrip ──> 2.5 versioning
2.4 autosave ──> 2.1 pre-import peek, 2.5 versioning
1.1 selection ──> 1.4 paint-only-selected, 3.2 into-selection, 4.4 auto-select, 5.1 frame-selected
5.1 fit-distance ──> 1.1 frame-selected
2.3 stats (exist) ──> 2.2 export presets
GraphError (exist) ──> 3.3 node diagnostics
3.5 templates ──> 3.4 node cache
4.3 GLB cache ──> 4.1 staging ──> 4.4 auto-select
```

## Recommended build sequence

1. **Finish P0** — 0.2 (.vxlt roundtrip + CI) and 0.3 (error messages). Cheap, and the importer/AI code is still warm.
2. **2.4 autosave + crash recovery** (carry 2.5 versioning with it). The trust floor for an editor; the user flagged it above most new features, and it reuses existing serialization.
3. **The spine** — 1.1 selection polish → 1.2 HUD → 5.1 fit-distance / 5.3 measure (rides along). Builds on the most mature subsystem (box-select + clipboard).
4. **AI bundle** — 4.3 GLB cache → 4.1 staging → 4.4 auto-select → 4.2 cost. Highest differentiation; ship as one coherent feature.
5. **Graph** — 3.3 diagnostics + 3.1 preview numbers (both *surface*) → 3.5 templates → 3.4 cache.
6. **Harvest *surface* wins opportunistically** anytime there's slack: 2.3 export report, 5.5 perf HUD, 4.5 cancel-status.

## Maintenance

This file is the plan; `PROGRESS.md` is the done-state. Ship an item → check it
here, move a one-line summary to `PROGRESS.md`. Re-tag items (**new** →
**partial** → **done**) as they evolve, and prune dependency edges that have
been satisfied so the graph stays honest.
