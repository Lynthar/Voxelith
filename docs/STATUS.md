# Voxelith — Status

The single source of truth for **what's actually built** plus a concise map of **what isn't yet**. Replaces the former `PROGRESS.md` / `ROADMAP.md` / `ARCHITECTURE.md` triad — completed work is recorded as current state (not as planning/vision history); the long design rationale and the original phase essays live in git history.

For a user-facing intro and the full keyboard map, see [`README.md`](../README.md).

---

## Snapshot

| | |
|---|---|
| **Tests** | 298 (`cargo test`) — 288 prior + 10 new for the bake tool & export transform |
| **Build** | `cargo build --release` clean on Windows + Vulkan |
| **Entry** | `src/main.rs` → GUI (`src/app/`, winit `ApplicationHandler`) or headless `voxelith bake <spec.json>` (`src/bake.rs`) |
| **Stack** | Rust · wgpu 22 · egui 0.29 · winit 0.30 · rayon · noise · reqwest/tokio (AI). Full list in `Cargo.toml` |
| **Storage** | flat-array 32³ chunk store (no octree yet) |

---

## Implemented (current state)

### Editor
- **5 brush tools** (Place / Remove / Paint / Eyedropper / Fill) + **4 shape tools** (Line / Box / Sphere / Cylinder) with vengi-style two-phase drag (footprint on a locked face-plane → height phase → commit; whole shape = one undo).
- Brush drag-paint with first-hit **plane lock** + 8 px dead-zone; **stroke-merged undo** (consecutive `SetVoxels` within 200 ms collapse); hover preview; independent **X / Y / Z symmetry** (1–8-fold, cell-aligned).
- **Box-select + clipboard** (`Tool::Select`, `0`): AABB marquee with live readout; inside-drag **move** as a single overlap-safe `SetVoxels` Command + translucent ghost; `Ctrl+C/X/V`, `Ctrl+Shift+V` paste-at-cursor, paste auto-selects destination, `Ctrl+A`, `Del`, `Esc`/`Ctrl+D`; arrow-key nudge (`Shift`×10, `Ctrl+↑↓` for Y).
- **Selection transforms**: rotate / mirror (each an undoable `SetVoxels`), with cyan center + orange min-corner markers on the wireframe.
- DDA voxel raycast picking with `y=0` ground-plane fallback for anchor tools; capped `flood_fill`; Alt transient eyedropper; color palette with custom additions; per-brush **emissive / metallic** material toggles + a **tint-zone** picker (faction recolor zone: none / primary / secondary / reserved) — written to the placed voxel's `flags` / `_reserved`; picking a color preserves them; GLB export honors materials as glTF `materials[]` and zones as a per-vertex `_TINTZONE` attribute.
- **Named sockets** (`Tool::Socket`): click a voxel face (or the ground) to drop a named attachment point — position = face center, orientation = face normal. In-viewport magenta directional-pin gizmo (shaft + arrowhead along the `+Y→normal` facing the export bakes), Tools-panel rename/delete/clear list. Persist in `.vxlt`, export to glTF as empty nodes. Not on the undo stack (managed like the selection).

### Core
- **32³ chunks**, **8-byte voxel** = `material:u16 + RGBA + flags(bit0 emissive / bit1 metallic) + _reserved`; `Pod`/`Zeroable` for direct GPU upload.
- `World` = chunk hashmap (`Arc<RwLock<Chunk>>`), optional bounds.
- Two-layer dirty tracking with cross-chunk boundary propagation.

### Mesh
- **`GreedyMesher`** (default render + OBJ/GLB): per-voxel-RGBA face merging (Lysenko) + **per-vertex AO** (0fps 12-sample); greedy key = `(rgba << 8) | ao` with diagonal-flip. Winding reversed from ABCD walk → **CCW-from-outside** (wgpu/glTF standard).
- **`NaiveMesher`** — reference / fallback (shared quad helper, seam-consistent).
- **`mesh_world_smoothed`** (Marching Cubes, **export-only**): `light` (rounded cubes, keeps thin features) / `heavy` (3×3×3 blur, clay) + per-triangle winding correction.
- Cross-chunk face culling; rayon-parallel re-mesh (sequential GPU upload).

### Render
- wgpu pipelines: opaque + optional wireframe (feature-gated) + transparent; two overlay slots (procgen preview α0.5, brush preview α0.75).
- Orbital camera: WASD fly, scroll zoom-to-cursor, RMB pan, **MMB orbit re-anchored under cursor**; **fit-distance framing** (`F`, frame all/selected/generated); orbit angles re-derive each press (no first-drag teleport).
- Grid + axes + selection wireframe; ambient + directional light + distance fog.

### Procgen
- `VoxelGenerator` trait → `GenResult<VoxelPatch>` — generators emit **patches** (not direct world writes) → undoable via `CommandHistory`, previewable via `patch_to_mesh`.
- Three generators: **`PerlinTerrain`** (FBM heightmap), **`LSystemTree`** (3D turtle), **`WfcGenerator`** (2D WFC, **Dungeon** 19-tile + **City** 13-tile, forward-only with empty/grass fallback).
- Two UIs: single-generator panel + **visual node-graph editor** (`Translate` / `Filter` / `Mask` / `Combine` → `Output`, cycle-prevention + auto-layout). Both debounced 150 ms preview; commit routes through `Command::set_voxels`.

### I/O
- **`.vxlt`** — native gzip format (magic `VXLT` v1), embeds `EditorState` (camera / brush / palette / sockets; `#[serde(default)]` so pre-socket files still load).
- **`.vox`** — MagicaVoxel import (v150 + v200 scene-graph flatten) / export (v150, 254-color, palette-overflow report).
- **`.obj`** — export (greedy + MC light/heavy), per-chunk groups, vertex-color extension (per-vertex AO baked into RGB).
- **`.glb`** — glTF 2.0 binary export (greedy + MC light/heavy): `POSITION / NORMAL / COLOR_0` (per-vertex AO baked into RGB) `/ _TINTZONE` + `TEXCOORD_0.x` (per-vertex faction tint zone — the custom attr plus a UV mirror Unity glTFast can read), u32 indices; geometry is split into **per-material-group primitives with glTF `materials[]`** — plain (explicit non-metallic, since the glTF default is metallic), emissive (white `emissiveFactor`), and metallic (`metallicFactor` 1). **Named sockets** export as **empty nodes** (`name` + `translation` + `rotation`, no mesh; `+Y→normal` quaternion) — even for a geometry-free scene. Imports directly into Unity / Unreal / Godot / Blender. The engine-side consumption contract (every attribute / material / node field, color space, per-engine support) is specified in [`docs/GAME_PIPELINE_ROADMAP.md`](GAME_PIPELINE_ROADMAP.md) §3.2; a Unity URP reference shader is still TODO.
- Post-export report dialog (format / geometry source / triangle-vertex-chunk counts / file size / lost-color notes).
- **Headless batch export** — `voxelith bake <spec.json> [--shard i/n]` (`src/bake.rs` + clap in `main.rs`): batch `.vxlt`→`.glb` from a declarative `{ defaults, items[] }` spec with per-asset **pivot / up-axis / unit-scale** (a lossless root-node transform — `io::export_glb_with_transform`), optional **`gltfpack` meshopt compression** (`optimize: "meshopt"`, graceful skip if not installed), `srcDir`/`outDir` bulk expansion, `--shard` for CI fan-out, and a per-item JSON report next to each output. CPU-only (no window/GPU). Identity transform ⇒ byte-identical to the interactive export. See [`GAME_PIPELINE_ROADMAP.md`](GAME_PIPELINE_ROADMAP.md) §3.4–3.5.

### AI generation
- `src/ai/` — tokio background runtime, OS-keychain API key (`keyring`), `AiJobState` machine, egui AI panel, `MockProvider` for free end-to-end testing.
- **`FalHunyuanProvider`** (fal.ai `hunyuan3d-v3` text-to-3D): queue API + 2 s polling + 5 min cap + cooperative & remote cancel; key never leaks into errors.
- **`voxelize_glb`**: scene-graph walk + per-triangle adaptive sampling + 3-axis parity interior fill; lands as undoable `Command::set_voxels`. Prompt MRU + result auto-select/frame done.

### Prefs & resilience
- `prefs.ron` (window / panels / viewport / procgen / graph / brush / recent-files); `#[serde(default)]` forward-compat; scale-factor-aware (logical px).
- Timed **autosave** (60 s, atomic write) + **crash recovery** (delete-on-clean-exit → recover prompt at next launch; corrupt autosave falls back to default, never bricks startup).

### UI
- egui: menu / toolbar / status bar + Stats / Tools / Palette / Viewport / Help / About / Procgen / Graph / AI panels; in-app error & recovery dialogs.
- **Viewport HUD** (bottom-left, click-through: tool / gesture+numbers / locked plane / symmetry / selection size) + **Perf HUD** (bottom-right, default off: FPS+ms / tris / chunks / last rebuild).

---

## Not yet built

Concise forward map (the unbuilt parts of the former roadmap + vision), grouped and roughly priority-ordered within each area.

**Editing** — configurable keymap + conflict detection + key-help; camera nav presets (Blender/Maya/Goxel); surface-only paint; replace-color tool; paint-only-selected; recent colors; palette-slot naming; undo-history panel.

**Files & export** — pre-import inspection (peek dims/palette/warnings before commit — the headless bake's per-item JSON report partly covers this for `.glb`); `.vxlt` version migration; `.gltf` text variant; `.vox` v200 export. (Export presets are now subsumed by `voxelith bake` named `defaults` blocks; a GUI hook to launch a bake from the editor is the remaining nicety.)

**Game asset pipeline** (see [`docs/GAME_PIPELINE_ROADMAP.md`](GAME_PIPELINE_ROADMAP.md)) — §3.1 data export (AO / emissive-metallic / tint-zone / sockets) **done**; §3.2 `TEXCOORD_0` zone mirror **done**, consumption contract specified in roadmap §3.2, **Unity URP reference shader shipped** (`docs/reference/VoxelithUberURP.shader`); §3.4 **post-export optimization done** (the `voxelith bake` tool shells out to `gltfpack -cc -noq`) and §3.5 **batch/headless export done** (`voxelith bake`). **Remaining:** (a) §3.3 a better smooth mesher (Surface Nets / Dual Contouring) — lowest priority; (b) the §3.2 **GATE** — verifying the `TEXCOORD_0.x` zone survives Unity glTFast's UV pruning end-to-end (needs a running Unity 6 + glTFast; procedure in roadmap §3.2); (c) optional native meshopt (§3.4 plan B) to drop the external `gltfpack` dependency.

**Procgen & graph** — WFC backtracking (currently forward-only); more tilesets (Castle/Pipes/sci-fi); on-canvas node diagnostics; preview time/count; cancel for large gens; commit semantics (overwrite/add/layer/into-selection); graph templates; cross-run node cache; **shape grammar** (not started).

**Rendering & perf** — real-time MC render preview; SSAO + soft shadows; viewport settings panel (grid/fog/clip/bg/light); measure tool; turntable/screenshot; **PBR materials** (per-voxel `material_id` + palette material table + metallic-roughness glTF → metal/wood/emissive distinguish downstream); octree/SVO compression; GPU/multithread procgen.

**AI** — staging area (preview/move/accept before commit) + GLB cache (free re-voxelize) + cost/ETA before submit + provider dropdown + image-to-3D UI. **Local inference** (Candle/ONNX) deferred — no viable Rust path for TRELLIS / Hunyuan3D as of 2026-05 (mesh→voxel through a remote API remains the route).

**Platform & ecosystem** — WASM/WebGPU build; scripting (Lua/Rhai); plugin API; tileset/material externalization to `.ron`; asset library.

**Tooling / CI** — flip clippy (`-- -D warnings`) and `cargo fmt --check` from informational to hard gates after a cleanup pass (codebase carries a handful of pre-existing lints + a deliberate narrow manual format).

---

## Design decisions & invariants worth knowing

Load-bearing gotchas for anyone touching the code:

- **Generators emit patches, not direct world writes** — decouples them from `World` locking, makes generation undoable, lets the same patch render as a preview.
- **WFC is non-backtracking on purpose** — the preview re-evaluates after every change, so termination beats perfectly-constrained output; over-constrained cells fall back to empty/grass + surface a `note`.
- **GLB/OBJ winding is reversed from the natural ABCD walk** → CCW-from-outside (verified by `test_winding_*`); don't change without re-running those.
- **Marching Cubes is export-only** — never used at render time.
- **AI patch coupling is one-directional**: `ai → procgen` (`JobEvent::Done` carries `Option<VoxelPatch>`); never the inverse.
- **API key lives in the OS keychain**, never `prefs.ron`.
- **Errors / recovery use in-app egui dialogs, never `rfd::MessageDialog`** — the native dialog exits the process on the dev's winit+wgpu+Windows setup (`rfd::FileDialog` is unaffected).

---

## Known limitations

- MC export dissolves thin / 1-cell features at `blur=heavy` (use `light`; fundamental MC limit, not a bug).
- No undo for procgen preview (ephemeral by design); active selection not persisted across restarts (by design, like image editors).

---

## Onboarding

1. `cargo run --release` — verify it launches and the cube + ground show.
2. `cargo test` — should be 288 passing.
3. `git log --oneline` — see the recent direction and last-committed work.
