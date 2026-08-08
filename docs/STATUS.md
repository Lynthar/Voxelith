# Voxelith — Status

The single source of truth for **what's actually built** plus a concise map of **what isn't yet**. Replaces the former `PROGRESS.md` / `ROADMAP.md` / `ARCHITECTURE.md` triad — completed work is recorded as current state (not as planning/vision history); the long design rationale and the original phase essays live in git history.

For a user-facing intro and the full keyboard map, see [`README.md`](../README.md).

---

## Snapshot

| | |
|---|---|
| **Tests** | 530 (`cargo test`) · 440 headless (`cargo test --no-default-features`) |
| **Build** | `cargo build --release` clean on Windows + Vulkan. Features: `default = ["gui", "mcp"]`; `--no-default-features` builds the headless half with no winit / wgpu / egui / rfd in the tree; `mcp-http` adds the Streamable HTTP transport (and axum), and `gui` turns it on because the editor hosts an MCP server of its own. CI builds four configurations: default, headless, headless + `mcp`, headless + `mcp-http`. |
| **Entry** | `src/main.rs` → GUI (`src/app/`, winit `ApplicationHandler`, `--agent-port <P>` to open its agent bridge at launch) or headless `voxelith bake <spec.json>` (`src/bake.rs`) / `exec` / `inspect` / `generators` (`src/exec.rs`) / `mcp` (`src/mcp/`) |
| **Stack** | Rust · wgpu 22 · egui 0.29 · winit 0.30 · rayon · noise · reqwest/tokio (AI) · rmcp 3.1 + schemars (MCP). Full list in `Cargo.toml` |
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
- `World` = chunk hashmap (`Arc<RwLock<Chunk>>`), unbounded — chunks are created on demand wherever a write lands, so every `i32` coordinate is writable.
- Per-chunk dirty tracking, propagated across chunk boundaries over the **full Moore neighborhood** (a corner write dirties up to 7 neighbors) — per-vertex AO samples all 26 surrounding chunks, so face-neighbors alone leave diagonals with stale AO.

### Mesh
- **`GreedyMesher`** (default render + OBJ/GLB): per-voxel-RGBA face merging (Lysenko) + **per-vertex AO** (0fps 12-sample); greedy key = `(tint_zone << 40) | (rgba << 8) | ao` with diagonal-flip. Winding reversed from ABCD walk → **CCW-from-outside** (wgpu/glTF standard).
- **`NaiveMesher`** — reference / fallback (shared quad helper, seam-consistent).
- **`mesh_world_smoothed`** (Marching Cubes, **export-only**): `light` (rounded cubes, keeps thin features) / `heavy` (3×3×3 blur, clay) + per-triangle winding correction; vertices shared per cube edge; refuses (`SmoothMeshError::SceneTooLarge`) rather than allocating a density field bigger than `MAX_DENSITY_CELLS`, since the field is dense over the scene's whole bounding box.
- Cross-chunk face culling; rayon-parallel re-mesh (sequential GPU upload).

### Render
- wgpu pipelines: opaque + optional wireframe (feature-gated) + transparent; **3 transparent overlay slots** (procgen preview α0.5, brush preview α0.75, selection-move ghost) + **2 wireframe overlay slots** (selection box, socket gizmos).
- Orbital camera: WASD fly, scroll zoom-to-cursor, RMB pan, **MMB orbit re-anchored under cursor**; **fit-distance framing** (`F`, frame all/selected/generated); orbit angles re-derive each press (no first-drag teleport).
- Grid + axes + selection wireframe; ambient + directional light + distance fog.

### Procgen
- `VoxelGenerator` trait → `GenResult<VoxelPatch>` — generators emit **patches** (not direct world writes) → undoable via `CommandHistory`, previewable via `patch_to_mesh`.
- Three generators: **`PerlinTerrain`** (FBM heightmap), **`LSystemTree`** (3D turtle), **`WfcGenerator`** (2D WFC, **Dungeon** 19-tile + **City** 13-tile, forward-only with empty/grass fallback).
- Two UIs: single-generator panel + **visual node-graph editor** (`Translate` / `Filter` / `Mask` / `Combine` → `Output`, cycle-prevention + auto-layout). Both debounced 150 ms preview; commit routes through `Command::set_voxels`.

### I/O
- **`.vxlt`** — native gzip format (magic `VXLT` v1), embeds `EditorState` (camera / brush / palette / sockets / **pipeline graph**; `#[serde(default)]` so pre-socket and pre-graph files still load). The graph is document data, not workspace state — it says how the model was made, so it travels with the project rather than with the machine, which is also the only way a graph an agent built headless can reach the human who opens the file. A project written headless has no camera of its own, so `EditorState::default()` carries the editor's own starting pose (`io::DEFAULT_CAMERA_POSITION`, which `Renderer::new` reads too) — all-zeros would be a degenerate look-at and a blank viewport. On load a pose that still isn't usable is refused and the scene framed instead.
- **`.vox`** — MagicaVoxel import (v150 + v200 scene-graph flatten) / export (v150, 254-color, palette-overflow report), with a File ▸ Import toggle for the Z-up↔Y-up conversion (default on, applied as one global rotation after flattening). Import is hardened for untrusted files: chunk bodies are read through a length-limited reader and drained (unknown trailing fields from a newer MagicaVoxel can't desync the stream), the scene graph is walked as a real DAG under depth/visit/chunk budgets, the official 256-color default palette is used when `RGBA` is absent, and palette alpha is normalized opaque.
- **`.obj`** — export (greedy + MC light/heavy), per-chunk groups, vertex-color extension (per-vertex AO baked into RGB).
- **`.glb`** — glTF 2.0 binary export (greedy + MC light/heavy): `POSITION / NORMAL / COLOR_0` (per-vertex AO baked into RGB) `/ _TINTZONE` + `TEXCOORD_0.x` (per-vertex faction tint zone — the custom attr plus a UV mirror Unity glTFast can read), u32 indices; geometry is split into **per-material-group primitives with glTF `materials[]`** — plain (explicit non-metallic, since the glTF default is metallic), emissive (white `emissiveFactor`), and metallic (`metallicFactor` 1). **Named sockets** export as **empty nodes** (`name` + `translation` + `rotation`, no mesh; `+Y→normal` quaternion) — even for a geometry-free scene. Imports directly into Unity / Unreal / Godot / Blender. The engine-side consumption contract — every attribute, its color space, and what a stock URP project must do to read it — is the reference shader at [`docs/reference/VoxelithUberURP.shader`](reference/VoxelithUberURP.shader); the producer half is documented where each attribute is written, in `src/io/gltf.rs`.
- Post-export report dialog (format / geometry source / triangle-vertex-chunk counts / file size / lost-color notes).
- **Headless batch export** — `voxelith bake <spec.json> [--shard i/n]` (`src/bake.rs` + clap in `main.rs`): batch `.vxlt`→`.glb` from a declarative `{ defaults, items[] }` spec with per-asset **pivot / up-axis / unit-scale** (a lossless root-node transform — `io::export_glb_with_transform`), optional **`gltfpack` meshopt compression** (`optimize: "meshopt"`, graceful skip if not installed), `srcDir`/`outDir` bulk expansion, `--shard` for CI fan-out, and a per-item JSON report next to each output. CPU-only (no window/GPU). Identity transform ⇒ byte-identical to the interactive export.

### AI generation
- `src/ai/` — tokio background runtime, OS-keychain API key (`keyring`), `AiJobState` machine, egui AI panel, plus a `MockProvider` offline stub (defined for test / offline wiring; not currently constructed by any code path).
- **`FalHunyuanProvider`** (fal.ai `hunyuan3d-v3` text-to-3D): queue API + 2 s polling + 5 min cap + cooperative & remote cancel; key never leaks into errors. Remote input is bounded — streamed bodies with hard byte ceilings (GLB 256 MiB / JSON 4 MiB), https-only download URL, per-request timeouts, non-transient 4xx fails fast instead of retrying to the cap, and Cancel races the in-flight request rather than waiting it out.
- **`voxelize_glb`**: scene-graph walk + per-triangle adaptive sampling + 3-axis parity interior fill; lands as undoable `Command::set_voxels`. Prompt MRU + result auto-select/frame done. Parses with `Gltf::from_slice` + `import_buffers` so only the base-color textures a material actually samples are decoded, each under an explicit `image::Limits` (a downloaded GLB can otherwise declare a decompression-bomb texture).

### Agent ops
- `src/agent_ops/` — a JSON edit protocol so an external agent can drive the modeling primitives directly: `AgentSession` (world + history + selection + sockets + pipeline graph) takes an `OpsBatch` and returns an `ApplyReport`. **14 ops**: `box` / `sphere` / `cylinder` / `line` (each with `filled` and a `write_mode` of `replace` / `only_air` / `only_solid` — `box` + `only_solid` *is* the paint tool, and `"voxel": "air"` *is* the eraser), `hollow`, `set_voxels`, `generate`, `graph` / `graph_edit`, `select` / `deselect`, `rotate` / `mirror` / `mirror_copy`.
- **The node graph is an agent-facing primitive** (`graph`, `graph_edit`): an agent composes generators and transforms into a pipeline instead of placing voxels, and the graph is stored **with the project**, so the human who opens it afterwards finds it in the Graph panel with the sliders live. Its output goes through the same write path a `generate` op does, so it inherits the cell budget, the coordinate ceiling and `write_mode`; its source nodes are held to the same size ceilings the generator registry enforces, since a node carries an already-built generator that would otherwise walk past them. Graphs are capped at 64 nodes (evaluation is a recursive descent) and 8 sources (evaluation memoizes a patch per node). `graph_edit` carries up to 64 typed edits — `add_node` / `remove_node` / `set_params` / `connect` / `disconnect` / `clear` — which run against a copy, so a batch that fails part-way leaves the graph exactly as it was. The wire format is the storage format: one flat object per node, `{"id": 1, "kind": "translate", "input": 0, "dy": 8}`, with a source node's `kind` being its generator id. The op's schema keeps the graph an opaque object on purpose and `list_generators` hands out a working `graph_template` instead — spelling the format out in the schema would cost every conversation about nine more type definitions than it is worth.
- **Structural measurements** (`describe`) — connected components (6-connectivity), floating parts, fully-enclosed voxels (the interior `hollow` would remove) and per-axis mirror symmetry. Deterministic answers to questions a rendered view can't settle: a one-cell gap between two halves is the most common way a model looks finished and isn't. Measurements, never verdicts — two components is wrong for a sword and right for a pair of boots. Skipped above two million voxels, which reports `null` rather than stalling the editor's main thread.
- **Atomic, sequential, one undo entry per batch.** Ops run against a `World::deep_clone` so each sees the results of the ones before it; any failure discards the copy and the real world is untouched; success commits as a single `Command::SetVoxels`. `dry_run` runs the whole pipeline and reports what *would* happen, with a report identical to the real one — and `preview_ops` hands the resulting world back inside a `Preview`, so `describe` / `slice` can inspect a result nobody has committed (a dry run that described the *session* would answer "what would this do?" with a picture of the world it declined to change).
- **Generator registry** — the three built-in generators, each advertising its `Default` serialized as the parameter template (no hand-written JSON Schema to drift); partial params merge over the defaults and unknown keys are named as errors (the strict check lives here, not on the shared param structs, whose `serde(default)` forward compatibility `prefs.ron` / `.vxlt` depend on).
- **Feedback** — `describe()` (counts, AABB, color histogram, material/tint-zone tallies, sockets, selection, undo depth) and `slice()` (one plane as ASCII art, solid or per-color with a legend).
- **Rendered views** (`src/view.rs`) — the agent's eye: CPU ray casting over the voxel grid, no GPU and no window, so it runs wherever an agent does. Seven orthographic viewpoints (six axes + isometric), lambert against a key light that follows the camera, plus per-face ambient occlusion; emissive voxels come through unshaded. Parallel projection on purpose — a wall that looks straight *is* straight, and equal cells stay equal size — and every image reports the cell bounds it covers and its cells-per-pixel, so a distance measured on the picture converts back to coordinates. `voxelith render` writes PNG files; over MCP `render_views` returns them inline as image content. Sizes outside 1..=1024 are refused rather than clamped.
- **Limits are explicit errors, never silent no-ops**: ops/batch, `set_voxels` entries, per-op region cells, per-batch cells, coordinate range, new chunks per batch. Every failure carries `op_index` + a stable `code` + a message that says what to do instead.
- **CLI** (`src/exec.rs`) — `voxelith exec <ops.json> [--in p.vxlt] [--out p.vxlt] [--export f.glb|f.obj|f.vox] [--describe] [--slice <json>] [--dry-run]`, plus read-only `inspect <p.vxlt>` and `generators` (the catalog, each entry carrying its parameters at their default values — that listing *is* the params template). stdout is JSON and nothing else (logs go to stderr): `{"ok":true,…}` / exit 0, or `{"ok":false,"error":{code,op_index?,message}}` / exit 1. A loaded project's camera / palette / brush / sockets ride through a headless edit untouched. The ops schema is documented on the types in `src/agent_ops/schema.rs`; `voxelith generators` prints the generator catalog with its parameter templates.
- **Headless feature split** — the `gui` feature (default on) gates winit / wgpu / egui / egui-wgpu / egui-winit / rfd / pollster, plus the `render`, `ui` and `prefs` modules (`prefs` is entirely editor workspace state) and `Vertex::layout` (the mesh layer's one GPU-typed function). `--no-default-features` leaves the library + `bake` / `exec` / `inspect` / `generators` with **zero** GUI crates in the dependency tree; without the feature the no-subcommand launch says so and exits 2 instead of pretending the command was malformed. CI builds and tests both configurations.
- **MCP server** (`src/mcp/`, `voxelith mcp`) — the same primitives as a resident tool set, for agents that speak the protocol rather than running commands. **11 tools**: `new_project` / `open_project` / `save_project`, `apply_ops`, `list_generators`, `describe`, `slice`, `render_views`, `undo` / `redo`, `export`. The difference from the CLI is the session, not the verbs: one document stays open across calls, so undo history, the selection and unsaved edits survive from one tool call to the next. `apply_ops` answers with the report *and* a description of the same world — under `dry_run` both come from the preview, since over a protocol there is no other way to ask for one. Tool argument schemas are generated from the `agent_ops` types (`schemars`), because over MCP the schema is where an agent learns the ops format. The pipeline graph is the one deliberate exception: its schema would be nine more type definitions carried in context every turn, so the op takes an opaque object and `list_generators` hands out a working graph to copy instead — the same "the template is the documentation" trick the generator registry uses.
- **Two transports, one handler** — stdio (the `mcp` feature, default on) for a client that launches the server as a child process; Streamable HTTP at `/mcp` (the `mcp-http` feature, off by default because it drags axum in) for one that wants a URL. Streamable HTTP is stateless at the protocol layer under the 2026-07-28 spec — a fresh MCP session per request — so the handlers it builds per request are clones sharing one `Arc<Mutex<Document>>`. That sharing is what makes a conversation over HTTP possible at all; a fresh document per request would silently drop every edit.
- **Every path resolves inside one root** (`--root`, default the working directory), canonicalized first so `..` and symlinks are resolved before the containment test rather than pattern-matched away. Over stdio this buys little — the client launched the process — but the same tool bodies serve HTTP, and a rule that changes with the transport is one an agent's working recipe trips over.
- **A human can watch it work** — `voxelith mcp --checkpoint` writes the document back to its own file after every edit (undo included; a dry run changes nothing, so it writes nothing), and the editor polls the project it has open and reloads a version it didn't write. Keep the same `.vxlt` open in the GUI and the agent's steps appear as they land. The answer says what the write did — `saved: false` with a reason, never silence — because a checkpoint that quietly stopped landing looks exactly like an agent that stopped working. A reload restores the world, its sockets and its graph: camera, brush, palette and tool stay where the user left them, or every batch would yank the view back to wherever it pointed at the last save. The graph is on the restore side of that line because it is document data — an agent that rewrote the recipe means for the human to see the new one.
- **Single writer while that lasts** — the editor refuses to reload over unsaved local edits: the user's copy wins, and a strip under the menu bar says which file moved and offers both ways out (Reload, behind a confirm since it discards unsaved work / Keep mine, which dismisses it until the next write). It's a strip and not a dialog because the writer is an agent working a batch at a time — a modal would reopen on every step; and it's persistent state rather than a status line because the refusal holds for every later write too, so a message that scrolls away leaves the feature looking broken. Two people editing one file is detected, never merged. This is the file-passing path; the in-editor bridge below removes the limit outright.

### In-editor agent bridge
- **A second MCP server, serving the project the editor has open** (`src/mcp/bridge.rs` + `src/app/agent_bridge.rs`). Start it from the Agent panel or launch with `voxelith --agent-port 8737`, then point a client at the URL it shows (`claude mcp add --transport http voxelith http://127.0.0.1:8737/mcp`). Loopback only.
- **One undo stack.** An agent's batch is committed through the same `CommandHistory` as a brush stroke, so Ctrl+Z walks back through both and a human can take over mid-build. There is no file in the middle, so no single-writer race and no reload: the agent and the person are editing the same world. This is what `agent_ops::run_batch` exists for — it runs a batch against a world it doesn't own and hands back a change list, so the editor commits with its own history rather than keeping a second one beside the user's.
- **7 tools, not 11**: `apply_ops`, `describe`, `slice`, `render_views`, `list_generators`, `undo`, `redo`. No file operations — someone is sitting at this document, and where it saves is theirs to decide.
- **Two ways to treat an incoming batch.** *Apply directly* (default): it lands as it arrives, the human watches it happen, Ctrl+Z is right there. *Ask me first*: it goes up as translucent geometry and the agent's call waits until they apply or discard it — cells the batch would clear are painted red, since geometry that is leaving has nothing else to show. A batch waiting for approval is invalidated by any edit underneath it (its `old_voxel`s describe the world as it was), and answering says so in terms the agent can act on.
- Calls are drained in the frame loop, so the world never leaves the main thread; the wire never learns what a chunk is. A call whose editor goes away is answered, not left to hang.

### Prefs & resilience
- `prefs.ron` (window / panels / viewport / procgen / brush / recent-files / last export + import directory); `#[serde(default)]` forward-compat; the pipeline graph moved out of here into the project file, and a graph left in an older prefs file is carried into the session once, with a note, rather than dropped; scale-factor-aware (logical px). The recent-files MRU holds `.vxlt` projects only — its one consumer is Open Recent, which feeds every entry to the project loader; exports and `.vox` imports seed the corresponding dialog directory instead.
- Timed **autosave** (60 s, atomic write) + **crash recovery** (delete-on-clean-exit → recover prompt at next launch; corrupt autosave falls back to default, never bricks startup).
- **Auto-reload** — the open project's modification time is polled every 500 ms, and a version the editor didn't write is loaded in place (world + sockets + pipeline graph, i.e. the document; the camera and workspace stay put). Unsaved local edits veto it and raise the disk-conflict strip instead (Reload / Keep mine — see Agent ops above). Built for watching an agent work, but it fires for any writer: a `voxelith exec --out` run, a `git checkout`.
- **Unsaved-changes guard** on every path that discards the scene (New / Open / Open Recent / Import / Generate\* / window close / File ▸ Exit, keyboard shortcuts included): in-app Save / Don't Save / Cancel prompt, where Save only proceeds if the write actually landed. Clear All has its own confirm dialog (it wipes the undo history, so it can't be undone).

### UI
- egui: menu / toolbar / status bar + Stats / Tools / Palette / Viewport / Help / About / Procgen / Graph / AI / Agent panels; in-app dialogs for errors, the export report, crash recovery, destructive-action confirmation, and the unsaved-changes guard. Wireframe toggles gray out on GPUs without `POLYGON_MODE_LINE` instead of silently doing nothing.
- Every workspace panel has a title-bar close button, and all of them (including AI) survive a restart — visibility lives in one `prefs::PanelVisibility` that `UiState` holds directly, so load and save are whole-struct assignments. Tools scrolls internally (its content is taller than the default window). Float *positions* are not persisted; the `default_pos` constants are what every session gets. A restored window size is checked against the primary monitor and shrunk only if it no longer fits (`fit_window_to_monitor`) — a size saved on a 4K dock used to come back off-screen on a laptop panel.
- **Viewport HUD** (bottom-left, click-through: tool / gesture+numbers / locked plane / symmetry / selection size) + **Perf HUD** (bottom-right, default off: FPS+ms / tris / chunks / last rebuild).

---

## Not yet built

Concise forward map (the unbuilt parts of the former roadmap + vision), grouped and roughly priority-ordered within each area.

**Editing** — configurable keymap + conflict detection + key-help; camera nav presets (Blender/Maya/Goxel); surface-only paint; replace-color tool; paint-only-selected; recent colors; palette-slot naming; undo-history panel.

**Files & export** — pre-import inspection (peek dims/palette/warnings before commit — the headless bake's per-item JSON report partly covers this for `.glb`); `.vxlt` version migration; `.gltf` text variant; `.vox` v200 export. (Export presets are now subsumed by `voxelith bake` named `defaults` blocks; a GUI hook to launch a bake from the editor is the remaining nicety.)

**Game asset pipeline** — data export (AO / emissive-metallic / tint-zone / sockets) **done**; the `TEXCOORD_0` zone mirror **done**, with the Unity URP reference shader shipped (`docs/reference/VoxelithUberURP.shader`) as the consumption contract; post-export optimization **done** (`voxelith bake` shells out to `gltfpack -cc -noq`) and batch/headless export **done** (`voxelith bake`). **Remaining:** (a) a better smooth mesher (Surface Nets / Dual Contouring) — lowest priority; (b) a **GATE** nobody can close from a unit test — importing a baked `.glb` into a stock Unity 6 + glTFast project and confirming the `TEXCOORD_0.x` zone still reaches the shader (glTFast documents no support for custom vertex attributes, which is why the zone is mirrored into a UV set at all, and says nothing either way about keeping a UV set no material samples); (c) optional native meshopt to drop the external `gltfpack` dependency.

**Procgen & graph** — WFC backtracking (currently forward-only); more tilesets (Castle/Pipes/sci-fi); on-canvas node diagnostics; preview time/count; cancel for large gens; commit semantics (overwrite/add/layer/into-selection); graph templates; cross-run node cache; **shape grammar** (not started).

**Rendering & perf** — real-time MC render preview; SSAO + soft shadows; viewport settings panel (grid/fog/clip/bg/light); measure tool; turntable/screenshot; **PBR materials** (per-voxel `material_id` + palette material table + metallic-roughness glTF → metal/wood/emissive distinguish downstream); octree/SVO compression; GPU/multithread procgen.

**AI** — staging area (preview/move/accept before commit) + GLB cache (free re-voxelize) + cost/ETA before submit + provider dropdown + image-to-3D UI. **Local inference** (Candle/ONNX) deferred — no viable Rust path for TRELLIS / Hunyuan3D as of 2026-05 (mesh→voxel through a remote API remains the route).

**Agent integration** — the ops layer, the headless CLI, the MCP server over both transports, checkpoint-save + editor auto-reload, the CPU raycast views that are the agent's eyes, the in-editor bridge (one undo history shared with hand editing, which retires the single-writer limit), node-graph composition and the structural measurements are all done. **Remaining:** modeling-guidance prompts / skills, an example library, and an eval set — making an agent *good* at this rather than merely able to; the structural measurements are the eval set's judging criteria, so that half is already built. Deferred inside the protocol itself: socket ops, clipboard ops, a symmetry modifier on draw ops, per-op undo, multi-document sessions.

**Platform & ecosystem** — WASM/WebGPU build; scripting (Lua/Rhai) — largely subsumed by the agent ops layer above, which is the programmable surface; plugin API; tileset/material externalization to `.ron`; asset library.

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
- **File pickers go through `App::file_dialog`**, which attaches the main window as parent — an ownerless picker on Windows can open behind the app, and because it runs a modal loop the app stops rendering, which reads as a hang.
- **`unsaved_changes` vs `autosave_pending` are deliberately separate** — autosave must not clear the former, or "edit → autosave → close" would skip the guard and then delete the autosave that held the only copy.
- **Every voxel in the world is opaque (α = 255)** — the greedy mesher's zero-key "no visible face" sentinel and the flood fill's region test both depend on it; ingest paths (`.vox`, AI voxelize) normalize alpha.
- **A voxel is a cell `[p, p+1)`, not a point** — mirroring an axis maps `p ↦ size-1-p`; the point formula (`-p`) shifts even-sized models one cell (`io::vox::rotate_cell`). The agent layer's `mirror_copy` is the same rule stated as a seam: cell `p` reflects across the seam at `plane` to `2·plane − 1 − p`, and `plane: 0` reproduces the editor's `-p-1` symmetry exactly.
- **The agent ops schema denies unknown fields; the storage formats ignore them** — opposite rules, both deliberate. `prefs.ron` / `.vxlt` must read what an older build wrote; an ops batch is something a language model just invented, and a hallucinated field that silently does nothing is the one outcome an agent can't recover from.

---

## Known limitations

- MC export dissolves thin / 1-cell features at `blur=heavy` (use `light`; fundamental MC limit, not a bug).
- No undo for procgen preview (ephemeral by design); active selection not persisted across restarts (by design, like image editors).

---

## Onboarding

1. `cargo run --release` — verify it launches and the cube + ground show.
2. `cargo test` — should be 498 passing.
3. `git log --oneline` — see the recent direction and last-committed work.
