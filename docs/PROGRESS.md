# Voxelith Progress

This file tracks **what's actually implemented** in the codebase, against
the long-term vision in [`ARCHITECTURE.md`](./ARCHITECTURE.md). When
they disagree, this file is authoritative — `ARCHITECTURE.md` is a
design document that hasn't been pruned as features land.

For the prioritized plan of upcoming work (checkboxes, dependencies,
recommended build order) see [`ROADMAP.md`](./ROADMAP.md) — this file is
the done-state, ROADMAP is the plan.

For coding-agent guidance (commands, invariants, conventions) see
[`CLAUDE.md`](../CLAUDE.md). For user-facing intro see
[`README.md`](../README.md).

---

## Snapshot

| | |
|---|---|
| **Tests** | 252 passing (`cargo test`) |
| **Build** | `cargo build --release` clean on Windows + Vulkan |
| **Binary entry** | `src/main.rs` (~20 lines) → `src/app/` (App + winit `ApplicationHandler`) |

(Run `git log --oneline -1` for the current head — pinning a hash here ages poorly.)

---

## Implemented

### Editor

- 5 brush tools: Place / Remove / Paint / Eyedropper / Fill
- 4 shape tools: Line (3D Bresenham) / Box (filled AABB) / Sphere (ellipsoid fitting drag bbox) / Cylinder (axis = bbox's longest dimension). **vengi-style two-phase drag**: first click + drag lays a footprint W × D on a locked face plane (ray-vs-plane projection — no 透视歧义, no "flat shape" bug), release transitions to height phase, cursor's vertical screen movement defines extruded H along the plane normal (~8 px / voxel), second click commits. Esc cancels. The full shape is one undo entry; symmetry expands shape voxels with mirrors before commit
- **Brush drag-paint plane lock**: Place / Remove / Paint hold-and-drag stays on the first hit's face plane (vengi-style) — paint doesn't stack toward the camera as new voxels would otherwise occlude the next ray-vs-voxels hit. Works in empty world too (locks to y=0 ground plane)
- **Box select + clipboard** (`Tool::Select`, shortcut `0`): drag corner-to-corner to mark an AABB region; status-bar live readout `Sel: W×H×D (N cells)`; bright-yellow line-pipeline wireframe occluded correctly by intervening voxels. **Inside-selection drag = move** (single `SetVoxels` Command — overlap-safe via `build_move_changes`, `Ctrl+Z` reverses the entire move including the overlap region). **Outside-selection drag = create new selection**. Arrow keys nudge X / Z (`Shift` × 10), `Ctrl+↑↓` nudges Y axis. **Clipboard**: `Ctrl+C/X/V`, `Ctrl+Shift+V` for paste-at-cursor (vengi-style two-channel paste), `Del` to clear non-air voxels in selection, `Ctrl+A` selects the AABB of every non-air voxel in the world, `Esc` / `Ctrl+D` deselect. Cut is a single Command (not Copy + Delete) so one undo restores everything. Paste auto-selects the destination AABB so Paste→drag→Paste chains naturally (vengi `autoSelectSolidVoxels` trick). Selection state itself is *not* on the undo stack — ephemeral marquee, like image editors
- Spherical brush (radius 1–10)
- Drag-paint with 8 px dead-zone (click ≠ accidental streak)
- Stroke merging undo: consecutive `SetVoxels` within 200 ms collapse into one undo entry; `end_stroke()` on mouse-up
- Brush hover preview: translucent overlay showing where the next click will land (Place → adjacent cell, Remove/Paint/Fill → hovered cell). Mirror copies appear when symmetry is on
- Symmetry brush: independent X / Y / Z toggles mirror Place / Remove / Paint / Fill across the corresponding world-origin planes (1, 2, 4, or 8-fold). Cell-aligned reflection (`n → -n - 1`) so the plane lies between mirrored cells, not through one. Symmetric Fill collapses all mirror seeds into a single undo entry
- DDA-based voxel raycast for picking, with a y=0 ground-plane fallback for tools that need an "anchor cell" (Place + all 4 shape tools) — empty-world building works (the cursor's preview snaps to the plane); strict tools (Remove/Paint/Eyedropper/Fill) stay real-voxel-only so brush hints don't dangle in mid-air
- `flood_fill` capped by both voxel count and chebyshev distance from start; Fill tool guards air to avoid flooding empty regions
- Alt-key transient eyedropper
- Color palette panel with custom additions
- Standard shortcuts: `1-5` brush tools, `6-9` shape tools (Line/Box/Sphere/Cylinder), `0` box-select tool, `Ctrl+Z/Y` undo/redo, `Ctrl+C/X/V/A/D` selection clipboard ops, `Del` delete selection, arrow keys nudge selection, `Ctrl+S/O/N` file ops, `WASD/Q/E` camera, middle-mouse orbit, right-mouse pan, scroll zoom. Orbit angles re-derive from the camera's actual position+target on every middle-press, so Reset Camera / Set Camera View / pan / WASD never cause "first orbit drag teleports" desync

### Core

- 32³ chunks, 8-byte voxel layout (material + RGBA + flags)
- `Voxel` is `Pod`/`Zeroable` for direct GPU upload
- `World`: chunk hashmap with `Arc<RwLock<Chunk>>`, optional bounds
- Two-layer dirty tracking with cross-chunk boundary propagation (writes on chunk-edge cells mark loaded face-neighbors dirty so their meshes re-cull)

### Mesh

- `GreedyMesher` (default in `App`): merges adjacent same-color same-direction faces into larger quads via Lysenko's classic algorithm generalized to per-voxel RGBA. A 32×32 single-color flat plane collapses from 1024 quads to 1; mixed scenes typically see 3–10× fewer triangles than naive. Cross-chunk merging is intentionally not done — chunks emit independently, seams are geometrically invisible
- `NaiveMesher` (kept as reference / fallback): one quad per visible face. Same winding logic as Greedy via the shared `face_quad_vertices_sized` helper, so adjacent merged + unmerged quads at chunk seams stay consistent
- **Per-vertex Ambient Occlusion** (Minecraft / 0fps style): every emitted face vertex carries an AO factor in `[0, 1]` baked at meshing time. `mesh::ao::compute_face_ao` samples 12 cells per face (4 corners × 3 neighbors). Fragment shader applies `ambient_min(0.5) + 0.5 * ao` to brightness so dark corners stay legible. Greedy mask key extended to `u64 = (packed_rgba << 8) | packed_ao` — only cells with matching color **and** all 4 corner AO values merge, with diagonal-flip (`ao[0]+ao[2] > ao[1]+ao[3]`) on each quad to align the triangle fold with the dark corner pair. AO sampling at chunk corners needs all 26 neighbor chunks; `mesh::neighbors` does the lock + cross-boundary voxel routing. Procgen preview leaves AO at 1.0 (no occlusion) so the translucent overlay doesn't get double-darkened. **Triangle winding** in `ChunkMesh::push_quad` is reversed from the natural ABCD walk to land on CCW-from-outside (matching wgpu / glTF standard) — see `test_winding_cross_parallel_to_face_normal` for the hard verification
- `mesh_world_smoothed(world, blur: bool)` (Marching Cubes, **export-time only** — never used at render time): samples a density field at voxel centers (1.0 solid / 0.0 air), optionally box-blurs it 3×3×3, then runs the standard Lorensen-Cline / Paul Bourke MC tables. Per-vertex colors blend solid voxels touching each MC edge; per-vertex normals are gradient-derived for smooth shading. The `blur` flag picks the smoothing strength: `false` ("light") preserves thicker features at the cost of less organic curvature; `true` ("heavy") gives clay-like blobs but dissolves thin / sparse detail. Available via four OBJ / GLB "smoothed, light" / "smoothed, heavy" export menu entries. **Per-triangle winding correction**: the standard Lorensen-Cline TRI_TABLE has minor mixed-winding for some isolated-corner configurations; emit-time we compare each triangle's cross product against `v0`'s outward-facing vertex normal (`density_gradient` returns `-grad`, i.e., toward air) and swap `v1`/`v2` if anti-parallel. Catches the export-only winding inconsistency that would show as inside-out facets in Blender / Unity
- Proper cross-chunk face culling (both meshers acquire read locks on the chunk + 6 neighbors)
- `patch_to_mesh`: convert a sparse `(pos, voxel)` list into a renderable mesh with internal face culling and a baked alpha — used for both procgen and brush hover overlays
- Parallel re-mesh of dirty chunks via `rayon::par_iter` (uploads stay sequential since `wgpu::Device` isn't trivially shareable)

### Render

- wgpu pipeline: opaque triangle pipeline, optional wireframe (feature-gated on `POLYGON_MODE_LINE`), transparent pipeline (alpha-blend, depth-test on, depth-write off)
- Two overlay slots on `Renderer`: `preview_mesh` (procgen, alpha 0.5) and `brush_preview_mesh` (brush hover, alpha 0.75)
- Grid + axes line meshes
- Orbital camera with WASD movement, mouse orbit/pan, scroll zoom
- Simple lighting (ambient + directional) + distance fog in `voxel.wgsl`

### Procgen

`VoxelGenerator` trait → `GenResult<VoxelPatch>`. Patches carry voxel writes + optional `notes: Vec<String>` for non-fatal warnings (e.g. WFC "N cells over-constrained").

Three concrete generators:

- **`PerlinTerrain`** — FBM heightmap with stratified grass/dirt/stone output
- **`LSystemTree`** — 3D turtle interpreting a rewritten plant L-system (`F → FF+[+F-F-F]-[-F+F+F]`), random per-push roll for non-planar branches, leaf clusters at branch tips
- **`WfcGenerator`** — 2D Wave Function Collapse over a 4³-voxel tile grid. Two built-in tilesets selectable from the panel + graph node sidebar:
  - **`Dungeon`** (19 tiles): empty / floor / 2 walls / 4 corners / 4 T / cross / 2 walls-with-door / 4 floor-with-door-mouth. Connectors: `0` = open, `1` = wall, `2` = doorway mouth (forces doors onto walkable floor)
  - **`City`** (13 tiles): grass / road_x / road_z / 4 corners / 4 T / cross / building. Connectors: `0` = grass-side, `1` = road-side. Roads form grid networks framed by sidewalks; buildings rise as 2×2×3 brick cubes from a grass base
  - No backtracking — over-constrained cells fall back to the tileset's empty / grass tile so it always terminates. Tile geometry uses per-cell `Voxel` data (not just bool), so multi-color tiles like `road_x` can have asphalt + sidewalk in one tile

Two UIs for procgen, both with debounced 150 ms preview overlays:

- **Single-generator panel** — pick one generator from a `GeneratorChoice` combo, edit params, click "Generate" to commit. Preview toggle drives a translucent overlay of the selected generator's current output.
- **Pipeline graph editor** — visual DAG: draggable node boxes, output→input bezier wires, drag-create wires with cycle prevention (`set_input` tentatively wires + topo-sorts, reverts on `Cycle`), per-node parameter sidebar, `Auto Layout` button. Sources (`Terrain`/`Tree`/`Wfc`) → `Translate` / `Filter` (per-voxel predicate: `YAbove` / `YBelow` / `MatchesColor` / `InsideBox`) / `Mask` (column-projected: `AboveColumn` / `BelowColumn`, e.g. "trees only above terrain surface") / `Combine` (Union/Difference/Intersect) → `Output`. Preview toggle in the top toolbar reuses the same debounce machinery — change detection is whole-graph `PartialEq` so param tweaks, node add/remove, and wire changes all trigger one regen path.

Both previews are independent state machines but share the renderer's overlay slot — when both are on, the graph's tick runs second and wins the slot. Toggling one off forces the other through its "just-toggled-on" path so it re-renders into the freshly-cleared slot.

Both UIs route their output through `CommandHistory::execute(Command::set_voxels(...))` so generation is undo-able.

### I/O

- `.vxlt` — native gzip-compressed format with magic `VXLT`, version 1, embeds `EditorState` (camera, brush, palette)
- `.vox` — MagicaVoxel import/export, 256³ size cap, 254-slot palette. **Reads both v150 (0.97/0.98) and v200 (0.99.7+) files** — v200 multi-model files are flattened into the single `World` voxel grid by walking the `nTRN/nGRP/nSHP` scene graph and applying cumulative transform (translation + rotation around model center) to each `nSHP`'s models. Material / layer / camera / render-object chunks are read and discarded. Writes always v150 (universally readable; our `World` has no layers / materials / scene graph to serialize anyway)
- Export reports `palette_overflow` count when more than 254 distinct colors → user sees "(N colors quantized)" in the status bar
- `.obj` — Wavefront OBJ export only (no import). Three variants in the menu:
  - **Standard**: re-meshes via greedy and writes per-chunk `g chunk_X_Y_Z` groups with the `v x y z r g b` vertex-color extension. CCW winding, Y-up axis. Typically 3–10× smaller than naive output.
  - **Smoothed, light**: Marching Cubes on raw 0/1 density (no blur). "Voxel surfaces with rounded edges" — preserves thin features (tree branches, sparse detail) at the cost of less organic curvature.
  - **Smoothed, heavy**: 3×3×3 box blur on density before MC. Clay-like blobs — best for terrain / large solid masses. Thin / isolated features dilute below the 0.5 isolevel and dissolve.
- `.glb` — glTF 2.0 Binary export only. Single-file format with embedded JSON scene + binary vertex/index buffers; imports directly into Unity / Unreal / Godot / Blender. Same three variants as OBJ (standard greedy / smoothed light / smoothed heavy). Vertex attributes: POSITION (vec3 f32), NORMAL (vec3 f32), COLOR_0 (vec4 f32). Indices are u32 so large worlds aren't capped at 64k vertices. Empty world produces a valid scene with no meshes (no BIN chunk).

### Prefs

`%APPDATA%/voxelith/prefs.ron` (or platform equivalent via `dirs::config_dir`):

- Window size (logical pixels, scale-factor aware)
- Panel visibility toggles (stats / tools / palette / viewport / procgen / graph)
- `ViewportSettings` (grid, axes, wireframe, grid size/spacing)
- `ProcgenSettings` (selected generator + each generator's params + preview toggle)
- `PipelineGraph` (full graph state including node positions)
- Editor brush state (color, size, current tool, custom palette, symmetry axes)
- Recent-files MRU (cap 10, dedup, surfaced via `File → Open Recent`)

`#[serde(default)]` everywhere → older prefs files with missing fields still load. Saved on `WindowEvent::CloseRequested` and `UiAction::Exit`. Hard-crash exits would lose changes (acceptable trade-off for now).

### UI

egui-based. Panels: menu bar, side toolbar, status bar (with highlighted current tool + preview indicator), Stats, Tools, Palette, Viewport Settings, Help, About, Procedural Generation, Pipeline Graph.

---

## Status against `ARCHITECTURE.md` phases

`ARCHITECTURE.md` lists Phase 1–7. Translating to actual state:

| Phase | ARCHITECTURE intent | Actual state |
|---|---|---|
| **1 — MVP core** | chunk store, wgpu, brush, egui, save/load, generator trait | ✅ Done — chunk store is flat-array (no octree yet), `VoxelGenerator` trait wired through |
| **2 — Procgen basics** | Perlin/Simplex terrain, basic WFC, parameter UI, mesh→voxel framework | ✅ Done — `PerlinTerrain`, WFC with two tilesets (19-tile Dungeon + 13-tile City), `procgen panel`, `patch_to_mesh` |
| **3 — Advanced procgen** | full WFC + backtracking, L-system, shape grammar, node graph editor | 🟡 Partial — L-system done, visual node graph done; **WFC has no backtracking** (forward-only with empty fallback), **shape grammar not started** |
| **(extra)** | box select + clipboard / move (MagicaVoxel/Goxel/vengi parity) | ✅ Done — `Selection` AABB, `Clipboard`, single-Command move with overlap-safe `build_move_changes`, paste auto-select-destination |
| **4 — AI alpha** | generator registry + orchestrator, remote API, text-to-voxel, model adapters, UI | 🟡 Phase 1–3 of 4 done (single fal.ai provider end-to-end). Phase 4 polish next. See "AI integration — status" below |
| **5 — AI beta** | local inference, model manager, hybrid pipeline, AI nodes, variation | ❌ Not started — likely skipped: 2026-05 research found no viable Rust local-inference path for TRELLIS / Hunyuan3D (Candle / ort don't have these models, ONNX export of the full diffusion + VAE pipeline is non-trivial). Local inference re-evaluated when the Rust ML ecosystem catches up |
| **6 — Optimization** | greedy meshing, multi-thread/GPU procgen, more export formats, asset library | 🟡 Partial — re-mesh is rayon-parallel; greedy mesher landed (default render + OBJ + GLB); OBJ + GLB export landed (each with greedy + smoothed Marching Cubes variants); marching cubes is export-only for now |
| **7 — Extension** | WASM, scripting, plugin API, custom AI | ❌ Not started |

---

## Architectural decisions worth knowing

(Detail in `CLAUDE.md`'s "Cross-file invariants" section.)

- **Generators emit patches, not direct world writes.** Decouples them from `World` locking, makes generation undo-able through `CommandHistory`, and lets the same patch be rendered as a translucent preview via `patch_to_mesh` without ever touching the world.
- **Graph editor and single-generator panel coexist** with separate prefs (`prefs.graph` vs `prefs.procgen`). Same generators can be reached either way.
- **WFC is non-backtracking on purpose.** Real backtracking can hang on bad tilesets; the preview ticker calls `evaluate()` after every parameter change, so termination is more important than perfectly-constrained output. Over-constrained cells become empty and surface as a `note` in the status bar.
- **Two transparent overlays share one pipeline.** `transparent_pipeline` (alpha-blend + depth-test on / depth-write off) is reused by procgen preview *and* brush hover preview — they're independent mesh slots on `Renderer` (`preview_mesh`, `brush_preview_mesh`).
- **Drag-paint has a dead-zone.** 8 px from the press point. Without it a click with the slightest hand-tremor would paint a streak.
- **Prefs writes are scale-factor-aware.** `inner_size()` returns physical pixels; we save logical pixels (divided by `scale_factor`) so the window doesn't grow by `scale_factor` on each restart on high-DPI displays.

---

## Next-step menu

Not a roadmap, just an ordered shortlist. Pick whichever fits the next session's available time.

### Short-term polish (~1 session each)

- **More WFC tilesets** — `Castle`, `Pipes`, sci-fi corridor, etc. Infrastructure (`WfcTileset` enum + dropdown + per-cell color) is in place; just authoring work for new themes
- **`.gltf` text variant** — current export is binary `.glb` only. Adding `.gltf` (JSON + sidecar `.bin`) is a straightforward fork of `gltf.rs` for tooling that prefers the text form. ~80 LOC
- **Configurable raycast / fog distance via `ViewportSettings`** — currently `RAYCAST_MAX_DIST = 500` (in `app/input.rs`) and shader fog `200..800` are hardcoded. Putting them on `ViewportSettings` sliders lets users adapt to scene scale (small detail work vs. 1000³ procgen). ~50 LOC

### Performance / rendering (mid-effort)

- **Real-time MC render mode** — current MC is export-only (with winding correction landed). Adding a "smoothed render preview" toggle would let users see the smoothed output in the viewport before exporting. Reuses existing MC + winding-fix code; needs renderer plumbing to swap meshers per-chunk. ~200 LOC
- **WFC backtracking** — current WFC is forward-only and falls back to the empty / grass tile when over-constrained. Real backtracking (with attempt-budget cap so it terminates on impossible tilesets) would let user-authored complex tilesets succeed where they currently bail. ~200 LOC
- **SSAO / soft shadows** — already have per-vertex AO; SSAO would add a screen-space "soft" layer on top, plus directional shadows from the sun light. ~300+ LOC and needs a `geometry_buffer` pass

### Long-term / differentiation (multi-session)

- **AI integration** — `ARCHITECTURE.md`'s Phase 4–5 headline. Start with text-to-voxel through a remote API (OpenAI / Replicate, Shap-E adapter). Decisions to make: which provider, point-cloud vs. mesh intermediate, voxelization config (resolution, color mapping). ~1000+ LOC, also depends on external service availability and API budgeting
- **Tileset data externalization** — move WFC tilesets out of code into `.ron` files so users (or generators) can author custom tilesets. Prerequisite for a real dungeon-design workflow. ~300 LOC
- **PBR materials** — extend `Voxel` with `material_id`, add palette-level material table (metallic / roughness / emission / IOR), upgrade `voxel.wgsl` to a PBR shader, write metallic-roughness PBR data into glTF `MATL`-equivalent materials. Lets Voxelith output drop into Unity/Unreal/Godot with proper material distinction (metal weapon vs. wood box vs. neon sign). ~500-1000 LOC
- **`.vox` v200 export** — currently only v150 (which all MagicaVoxel versions read). v200 export would write a single-model scene graph + materials. Low value (v150 already universal), only useful if a tool specifically refuses v150. ~100 LOC

### AI integration — status (2026-05-11)

Route **A** picked (single fal.ai provider in MVP, Tripo direct API
deferred — see "Decision: route A vs route B" below).

| Phase | Status | What it built |
|---|---|---|
| **1 — Foundation** | ✅ | `src/ai/` module (`mod` / `job` / `provider` / `runtime` / `keyring_store` / `mock` / `client` / `voxelize`); tokio background runtime; OS-keychain API key storage via `keyring` crate; `AiJobState` state machine; AI panel in egui (prompt textbox, password-masked API key entry, resolution combo 32/64/128, Generate/Cancel, progress bar, terminal-state rendering); `MockProvider` for end-to-end UI testing without spending credits |
| **2 — Real client** | ✅ | `FalHunyuanProvider` in `src/ai/client.rs` against `https://queue.fal.run/fal-ai/hunyuan3d-v3/text-to-3d`; queue API with 2 s polling + 5 min cap; cooperative cancel between every stage; transient 5xx tolerance; API key never appears in error messages; response body truncated to 200 chars (UTF-8 safe) |
| **3 — Voxelization** | ✅ | `voxelize_glb(bytes, resolution) -> VoxelPatch` in `src/ai/voxelize.rs`; gltf crate for parsing, image crate for PBR texture decode; scene-graph walk with cumulative transforms; per-triangle adaptive grid sampling; color priority COLOR_0 → texture-at-UV → baseColorFactor → gray; 3-axis parity scan with majority vote for interior fill; AABB re-anchored to (0, 0, 0); `JobEvent::Done` gained `patch: Option<VoxelPatch>`; `App::tick_ai_job` lands the patch via `Command::set_voxels` (undoable via Ctrl+Z) |
| **4 — Polish** | ⏳ Next | Result placement (center on world / re-center camera on result / auto-select destination AABB so user can immediately Move/Copy), recent-prompts MRU through prefs, status-bar polish, possibly: provider dropdown (Mock for free testing), image-to-3D upload UI (would also use the same `fal-ai/hunyuan3d-v3/image-to-3d` endpoint — `AiRequest::image` field is already wired through to provider, just no UI yet) |

**Architecture lock** (don't re-litigate these next session unless the constraints change):

- **`AiProvider` trait** is the extension point. Each impl owns its own HTTP client, request shaping, polling, result download. Communicates back via `mpsc::Sender<JobEvent>` + cooperative `Arc<AtomicBool>` cancel.
- **State machine on main thread; worker on tokio**. `App::tick_ai_job` per frame drains `mpsc::Receiver<JobEvent>` into `AiJobState` transitions; UI mirrors `AiJobState` for rendering. Pattern mirrors `app::preview::tick_preview`.
- **Patch coupling**: `JobEvent::Done` carries `Option<VoxelPatch>`. `procgen::VoxelPatch` is imported in `ai/job.rs`, which is an ai → procgen direction dependency. Inverse coupling (procgen → ai) is avoided.
- **Keychain not prefs.ron**: API key never serialized to disk in user-readable form. `keyring` service `"voxelith"`, username field encodes provider (currently only `"fal_ai"`).
- **Voxelize on `spawn_blocking`** in the worker, not on the tokio worker thread itself — CPU-bound 100–2000 ms ops would block other async tasks otherwise.

**Decision: route A vs route B** (recorded 2026-05-11)

User picked **A**: single fal.ai provider in MVP. Alternative B was "add Tripo direct API as second provider in MVP" (Tripo v3 has native voxel/LEGO style which is closer aesthetic match for Voxelith). Reasoning to defer Tripo:
- Phase 2 LOC ~doubles with second provider (separate HTTP client, auth, key, polling)
- UI grows (provider dropdown + per-provider params)
- One-provider MVP can be validated end-to-end faster
- Adding Tripo later is an additive `impl AiProvider` (~150–200 LOC), no refactor

**fal.ai endpoint corrections vs the original research notes**

The 2026-05 notes below recommended `Tripo v2.5 (image-to-3D)` as the MVP target. By 2026-05-11 fal.ai had added a direct **`fal-ai/hunyuan3d-v3/text-to-3d`** endpoint ($0.16/gen, ~10 s, PBR), which is what's actually implemented — text input is much simpler UX than image upload for the MVP. The research notes are retained below for historical context (and as input to a future "Tripo voxel-style provider" PR).

### AI integration — research notes (2026-05)

Snapshot of the 3D-AI landscape researched for the Phase 4 starter
session. Use this to skip repeating the survey when picking the work
back up; sources cited at the bottom in case the landscape moves.

**TL;DR** — there's no production-ready "text → voxel grid" API in
2026-05; every viable path goes mesh → voxel. Recommended MVP:
**fal.ai + Tripo v2.5 (image-to-3D, $0.20–0.40/model, 10–30s) → GLB
→ `voxquant_core` mesh-to-voxel + reuse `editor::flood_fill` for
interior fill → emit a `procgen::VoxelPatch` so it lands as a
single undoable `Command::set_voxels`**. Total ~900 LOC.

**Top candidates** (all output mesh unless noted):

| Tier | Option | Cost / VRAM | Latency | PBR | Notes |
|---|---|---|---|---|---|
| API (recommended) | **Tripo v2.5 via fal.ai** | $0.20–0.40 / model | 10–30 s | yes | One SDK shape; can swap to TRELLIS / Hunyuan on the same host |
| API | Meshy | ~$0.30 | 40–60 s | yes | Best docs, two-stage preview→refine |
| API | Rodin (Hyper3D) | $99/mo+ | 60–180 s | yes (10B params) | Highest quality, expensive |
| Local | **TRELLIS.2-4B** (MS, MIT, 2025-12) | 24 GB GPU | 3 s @ 512³ on H100; ~2 min on RTX 4090 | yes | Internal repr is sparse 1024³ voxels but default export is mesh; raw voxel needs source hack |
| Local | Hunyuan3D-2.5 (Tencent, 2025-06) | 6 GB GPU + 16 GB for textures | tens of seconds | yes | Most accessible local PBR pipeline |
| Local | TripoSR | 8 GB | <1 s @ A100 | no | Cheap demo; quality below v2.5 |

**Native voxel generation status** — XCube (NVIDIA, CVPR 2024) is
the only model targeting voxel grids natively but went inactive
after 2024 and is sized for 100 m × 100 m outdoor scenes (10 cm
voxels), not asset-scale. TRELLIS.2's O-Voxel is closer but
requires hacking the Sparse VAE pre-decode. **Don't wait for these
to mature** — go through GLB.

**Rust crates ready to drop in**:

- `voxquant_core` (2026-03, parallel, native glTF input) — surface
  voxelization. Pair with our existing `editor::flood_fill`
  (refactored to operate on a temporary `HashMap` instead of
  writing to `World` directly) for solid interior fill.
- `dda-voxelize-rs` (MIERUNE) — fallback if `voxquant_core` is
  ever yanked.
- `reqwest` + `tokio` — already in `Cargo.toml` behind the `ai`
  feature flag.
- `keyring` — store fal API key in OS keychain (don't write to
  `prefs.ron`).

**Engineering breakdown** (~900 LOC total):

| Module | LOC | Notes |
|---|---|---|
| `src/ai/client.rs` | ~250 | fal.ai HTTP polling, auth, retry/backoff |
| `src/ai/voxelize.rs` | ~150 | `voxquant_core` invocation + flood-fill interior |
| `src/ai/job.rs` | ~150 | `enum AiJob { Idle, Pending, Polling{progress}, Done(VoxelPatch), Failed(String) }` — mirrors `app::preview::PreviewState` lifecycle |
| `src/ui/ai_panel.rs` | ~200 | prompt input + progress bar + Cancel + Retry, wired to `Ui::ai_panel` |
| `src/app/ai_actions.rs` | ~100 | `tick_ai_job` parallel to `tick_preview`, lands `Done(patch)` via `Command::set_voxels` |
| `src/prefs.rs` extension | ~50 | keyring API key + recent-prompts MRU |

**Key UX decisions** (already worked through):

- Async + polling — never block main thread; reuse the
  `app::preview::tick_*` debounce-state-machine pattern
- Generate to a **staging area** (offset coordinates), let the
  user preview before committing — don't smash live `World`
- Cancel via `tokio::sync::oneshot` or `AtomicBool`
- 429 / timeout / network failure → exponential backoff 3×, then
  surface to status bar via `ui.set_status`

**Sources** (capture date 2026-05):

- [Tripo on fal.ai](https://fal.ai/models/tripo3d/tripo/v2.5/image-to-3d/api), [Tripo billing](https://platform.tripo3d.ai/docs/billing)
- [Meshy API quickstart](https://docs.meshy.ai/en/api/quick-start)
- [Microsoft TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
- [Tencent Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)
- [NVIDIA XCube](https://research.nvidia.com/labs/toronto-ai/xcube/) — flagged inactive
- [voxquant_core crate](https://crates.io/crates/voxquant_core)
- [3D AI price comparison 2026](https://www.3daistudio.com/blog/best-3d-model-generation-apis-2026)

---

### Known limitations not on the menu

- **MC export with mixed shapes**: large smoothed meshes work fine, but very thin / 1-cell-wide features (e.g. tree branches) dissolve below the 0.5 isolevel when blur=heavy. Workaround: use blur=light. Fundamental MC limitation, not a code bug
- **No undo for procgen preview**: preview overlays don't go through `CommandHistory`. By design — preview is ephemeral
- **Selection state not persisted**: closing the editor drops the active selection. By design — same as image editors

---

## Onboarding a new session

When picking this back up after time away:

1. `cargo run --release` — verify it still launches and the cube + ground show
2. Skim `CLAUDE.md` "Cross-file invariants worth knowing" — the gotchas accumulate fast in a 6 k-line codebase
3. Run `cargo test` — should be 189 passing
4. Pick from the "Next-step menu" above, or reopen `ARCHITECTURE.md` if the long-term direction needs revisiting
5. Open `git log --oneline` to see what was last committed and what the recent direction was
