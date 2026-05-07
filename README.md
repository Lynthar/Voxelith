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
| 🎨 **Editing** | Place / Remove / Paint / Eyedropper / Fill, drag-paint with stroke-merged undo, brush hover preview |
| 🌱 **Procedural generation** | Perlin terrain, L-system trees, WFC dungeon — pick one in the procgen panel or compose them in the visual node graph editor |
| ✨ **Live preview** | Debounced translucent overlay shows generator output before you commit |
| 📁 **File I/O** | Native `.vxlt` and MagicaVoxel `.vox` (lossy-color quantization is reported) |
| 💾 **Persistent state** | Window layout, panel toggles, generator params, recent files all survive restarts |
| 🖥️ **Viewport** | Orbit / pan / zoom camera, grid, axes, optional wireframe |

## Quick Start

```bash
git clone https://github.com/Lynthar/Voxelith.git
cd Voxelith
cargo run --release
```

## Keyboard Shortcuts

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| `1-5` | Select tool | `Ctrl+Z` | Undo |
| `WASD` | Move camera | `Ctrl+Y` | Redo |
| `Scroll` | Zoom | `Ctrl+S` | Save |
| `Middle Mouse` | Orbit | `Ctrl+O` | Open |

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
