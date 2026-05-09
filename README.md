```
██╗   ██╗ ██████╗ ██╗  ██╗███████╗██╗     ██╗████████╗██╗  ██╗
██║   ██║██╔═══██╗╚██╗██╔╝██╔════╝██║     ██║╚══██╔══╝██║  ██║
██║   ██║██║   ██║ ╚███╔╝ █████╗  ██║     ██║   ██║   ███████║
╚██╗ ██╔╝██║   ██║ ██╔██╗ ██╔══╝  ██║     ██║   ██║   ██╔══██║
 ╚████╔╝ ╚██████╔╝██╔╝ ██╗███████╗███████╗██║   ██║   ██║  ██║
  ╚═══╝   ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝   ╚═╝   ╚═╝  ╚═╝
```

<div align="center">

**Procedural-first Voxel Asset Creation Tool**

[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![wgpu](https://img.shields.io/badge/wgpu-22.0-blue.svg)](https://wgpu.rs/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[中文文档](README_CN.md)

</div>

---

## Overview

**Voxelith** is a modern voxel editor built with Rust, featuring GPU-accelerated rendering via wgpu and a clean egui interface. Designed as a procedural-first tool for both manual editing and programmatic generation.

## Features

| Feature | Description |
|---------|-------------|
| 🎨 **Editing** | 5 brush tools (Place / Remove / Paint / Eyedropper / Fill) + 4 shape tools (Line / Box / Sphere / Cylinder) with click-anchor / drag / release. Drag-paint with stroke-merged undo, brush hover preview, X / Y / Z symmetry mirroring |
| ▭ **Box select** | `0` to enter Select. Drag corners to mark an AABB; drag inside to move (single undoable Command, overlap-safe); arrow keys nudge X / Z, `Ctrl+↑↓` Y, `Shift` × 10. `Ctrl+C/X/V`, `Ctrl+Shift+V` paste-at-cursor, `Del`, `Ctrl+A` select-all-solid, `Esc` / `Ctrl+D` deselect. Paste auto-selects the destination AABB so Paste→drag→Paste chains |
| 🌱 **Procedural generation** | Perlin terrain, L-system trees, WFC tilesets (Dungeon + City) — pick one in the procgen panel or compose with Translate / Filter / Mask / Combine nodes in the visual graph editor |
| ✨ **Live preview** | Debounced translucent overlay shows generator output before you commit |
| 📁 **File I/O** | Native `.vxlt` (gzip + state), MagicaVoxel `.vox` (import + export with palette quantization), Wavefront `.obj` and glTF Binary `.glb` export. OBJ / GLB also have Marching Cubes "smoothed" variants (light: rounded cubes / heavy: clay-like) for organic exports |
| 💾 **Persistent state** | Window layout, panel toggles, generator params, recent files all survive restarts |
| 🖥️ **Viewport** | Orbit / pan / zoom camera (with auto-resync on every orbit), grid, axes, optional wireframe |
| 💡 **Per-vertex AO** | Minecraft-style ambient occlusion baked into the greedy mesh — corners and crevices darken, open faces stay bright. Adds visible block-by-block depth without runtime cost |

## Quick Start

```bash
git clone https://github.com/Lynthar/Voxelith.git
cd Voxelith
cargo run --release
```

## Keyboard Shortcuts

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| `1-5` | Brush tools | `Ctrl+Z` | Undo |
| `6-9` | Shape tools | `Ctrl+Y` | Redo |
| `0` | Box select | `Ctrl+C/X/V` | Copy / Cut / Paste |
| `WASD` | Move camera | `Ctrl+Shift+V` | Paste at cursor |
| `Q` / `E` | Camera up / down | `Del` | Delete selection |
| `Middle Mouse` | Orbit | `Ctrl+A` | Select all solid |
| `Right Mouse` | Pan | `Esc / Ctrl+D` | Deselect |
| `Scroll` | Zoom | `Arrows / Ctrl+↑↓` | Nudge selection |
| `Ctrl+S/O/N` | File ops | `Alt` (hold) | Eyedropper |

## Tech Stack

- 🦀 **Rust** - Systems language
- 🎮 **wgpu** - GPU rendering
- 🖼️ **egui** - Immediate mode UI
- 🗜️ **flate2** - Compression

## Architecture

```
┌──────────────────────────────────────────────┐
│ UI (egui panels + visual node graph editor) │
├──────────────────────────────────────────────┤
│ Editor (tools, commands, raycast, undo)     │
├──────────────────────────────────────────────┤
│ Procgen (terrain / tree / WFC + DAG eval)   │
├──────────────────────────────────────────────┤
│ Core (voxel, chunk, world) │ Mesh           │
│ Render (wgpu)              │ IO    Prefs    │
└──────────────────────────────────────────────┘
```

See [`docs/PROGRESS.md`](docs/PROGRESS.md) for implementation status and the next-step menu, and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for long-term design vision.

## License

MIT License © 2024
