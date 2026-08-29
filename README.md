<div align="center">

<img src="assets/branding/voxelith-banner.svg" alt="Voxelith — a runestone tablet with a V of glowing voxels" width="100%">

[![license](https://img.shields.io/github/license/Lynthar/Voxelith)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/Lynthar/Voxelith/ci.yml?branch=main&label=CI)](https://github.com/Lynthar/Voxelith/actions/workflows/ci.yml)
[![audit](https://img.shields.io/github/actions/workflow/status/Lynthar/Voxelith/audit.yml?branch=main&label=audit)](https://github.com/Lynthar/Voxelith/actions/workflows/audit.yml)

</div>

Procedural-first voxel asset creation — a wgpu/egui editor plus a headless CLI and MCP server that agents can drive

English | [简体中文](README.zh-CN.md)

A voxel editor with a GPU viewport, and the same codebase with no window at all.
Run it without a subcommand and you get the editor; give it one and it bakes a
batch of models to glTF, applies a JSON list of operations, renders turnaround
PNGs on the CPU, or scores a finished model against a set of assertions.

I built the headless mode for one workflow: an agent builds something, then
checks its own work. That's why the operation vocabulary, the MCP server and the
eval suite were designed in from the start, not added afterwards.

<img src="docs/media/editor.png" alt="The Voxelith editor with a lighthouse diorama open in the viewport" width="100%">

<sub>The editor — wgpu viewport, egui panels. This lighthouse is one ops batch:
48 operations applied with <code>voxelith exec</code>, then opened here.</sub>

<img src="docs/media/render.png" alt="The same model drawn by voxelith render" width="100%">

<sub>The same file through <code>voxelith render</code>, which draws on the CPU
with no GPU involved. Emissive voxels reach the image unshaded, so the lantern
lights up here and not in the viewport.</sub>

## Install

**There are no prebuilt binaries yet.** Building from source is the only way in,
and needs Rust 1.88 or newer.

```bash
git clone https://github.com/Lynthar/Voxelith.git
cd Voxelith
cargo run --release
```

For the headless mode only, without the windowing and GPU dependencies:

```bash
cargo build --release --no-default-features
cargo build --release --no-default-features --features mcp
```

On macOS, building an app bundle is what gets you a real Dock icon — winit can't
set one for a bare `cargo run`:

```bash
packaging/macos/bundle.sh
```

CI covers Windows and macOS. Linux isn't in the matrix.

## Usage

The editor has five brushes — place, remove, paint, eyedropper, fill — four
shapes, box select, and three generators (Perlin terrain, an L-system tree, wave
function collapse) whose parameters live in the project's pipeline graph.

```bash
voxelith                                   # the editor
voxelith --agent-port 8737                 # editor plus a loopback MCP bridge
```

Seven subcommands run without a window:

```bash
voxelith bake spec.json --shard 0/4
voxelith exec ops.json --in in.vxlt --out hut.vxlt --export hut.glb --dry-run
voxelith render hut.vxlt --view all --size 512 --out hut.png
voxelith inspect hut.vxlt --slice '{"axis":"y","index":1}'
voxelith eval evals/cases --results run-2026-08-08/
voxelith generators
voxelith mcp --root ./models --http 127.0.0.1:8080 --token …
```

`exec` takes fourteen operations — boxes, spheres, cylinders, lines, hollowing,
selection, mirroring, generator graphs — and the same vocabulary is exposed over
MCP as eleven tools. When the editor hosts that server itself, an agent's edits
land in your undo stack while you watch. `docs/reference/bake-spec.example.json`
is a working bake spec to copy from; `evals/` holds the cases, each a task
description plus properties the result has to satisfy.

The MCP bridge binds to loopback and requires a token (`VOXELITH_MCP_TOKEN`, or
one generated and printed at startup); `mcp --http` opens a port, so treat that
token like any local API credential.

## Limitations

- **Nothing has been released yet** — no tags, no binaries, not on crates.io.
- **No layers, no multiple objects, no scene tree.** There's one world, which
  means complex assets can't be split into parts. MagicaVoxel,
  [Goxel](https://github.com/guillaumechereau/goxel) and
  [vengi](https://github.com/vengi-voxel/vengi) all have those; Voxelith has the
  headless mode instead, and it reads `.vox` in both versions, so models can move
  between those tools and Voxelith.
- **`.vox` export writes version 150 only.** Reading accepts 150 and 200, but a
  version 200 scene graph gets flattened on the way in.
- **Linux has no automated coverage.** Whether the GUI builds and runs there has
  never been verified.
- **Evals judge assembly, not appearance.** They check component counts,
  enclosure and dimensions; whether it looks right is your call.
- **One writer at a time.** Concurrent edits to one file are detected and
  refused, never merged.

## License

Mozilla Public License 2.0 — see [LICENSE](LICENSE). Copyright (c) 2026 Lynthar.

This Source Code Form is subject to the terms of the Mozilla Public License, v.
2.0. If a copy of the MPL was not distributed with this file, You can obtain one
at <https://mozilla.org/MPL/2.0/>.
