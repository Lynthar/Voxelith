<div align="center">

<img src="assets/branding/voxelith-banner.svg" alt="Voxelith — 刻着发光体素 V 的符文碑" width="100%">

**程序化优先的体素资产创作工具**

[![Rust](https://img.shields.io/badge/Rust-1.88+-orange.svg)](https://www.rust-lang.org/)
[![wgpu](https://img.shields.io/badge/wgpu-22.0-blue.svg)](https://wgpu.rs/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

[English](README.md)

<br>

<img src="docs/media/agent-castle-keep.png" width="30%" alt="Agent 通过 ops 协议建造的城堡主楼"> <img src="docs/media/agent-terrain.png" width="30%" alt="程序化节点图生成的分层地形"> <img src="docs/media/agent-arched-bridge.png" width="30%" alt="拱桥 —— footprint 读数区分拱与墙">

<sub>直接出自工具本身：agent 经 ops 协议建造，<code>voxelith render</code> 出图，<code>voxelith eval</code> 判分。</sub>

</div>

---

## 概述

**Voxelith** 是一款使用 Rust 构建的现代体素编辑器，采用 wgpu 实现 GPU 加速渲染，配合简洁的 egui 界面。专为手动编辑和程序化生成设计。

## 功能特性

| 功能 | 说明 |
|------|------|
| 🎨 **编辑** | 5 个笔刷工具(放置/删除/绘制/取色/填充)+ 4 个形状工具(直线/盒/球/圆柱),两段式手势:拖出底面、松开、上提定高度、再点一下提交(Esc 取消)。支持拖动笔刷、笔画级 undo、实时悬停预览、X/Y/Z 对称镜像 |
| ▭ **盒选** | `0` 切到 Select 工具。拖角创建 AABB,选区内拖动 = 整团搬运(单一可撤销 Command,正确处理重叠);方向键平移 X/Z(`Ctrl+↑↓` 走 Y 轴,`Shift` × 10)。`R`/`Shift+R` 旋转、`M` 镜像,各算一步 undo。`Ctrl+C/X/V`、`Ctrl+Shift+V` 粘到光标、`Del` 删除、`Ctrl+A` 选所有非空、`Esc`/`Ctrl+D` 取消。粘贴后自动选中目标 AABB,可链式 Paste→拖→Paste |
| ⚓ **挂点(Socket)** | 在任意体素面上放置命名挂点(位置 + 朝外法线),随工程保存,导出 GLB 时变成 glTF 空节点 —— 武器挂载、特效锚点、旗帜插槽,交给引擎挂接部件 |
| 📥 **网格导入** | 把 `.glb` 体素化进场景,可选 32³ / 64³ / 128³:表面采样加奇偶扫描填充内部,颜色取自材质的 baseColor 因子与贴图。叠加到现有内容上,算一步可撤销编辑,Ctrl+Z 就能退回 |
| 🔌 **Agent 桥** | 编辑器自己托管一个 MCP server,agent 直接编辑你开着的工程 —— 它的批次进的是**你的**撤销栈,一批一次 Ctrl+Z,你随时可以接管。它还能直接给你一张节点图而不是一堆体素,结果依然可调参。也可以让它先问:批次以半透明几何上屏,由你决定应用还是丢弃。没人看着的时候还有无头版(命令行和独立 server) |
| 🏷️ **游戏资产材质** | 每笔刷的自发光 / 金属标记,外加 4 档阵营 **tint zone**,导出 GLB 时分别落成 glTF materials 与逐顶点 `_TINTZONE` 属性,供下游换色 shader 使用 |
| 🌱 **程序化生成** | Perlin 地形、L-System 树、WFC 多套 tileset(Dungeon + City)—— 从 Generate 菜单一键加为现成的图节点,再在可视化节点图里用 Translate / Filter / Mask / Combine 组合。节点图随工程保存,agent 也能直接产出一张图交给你继续调 |
| ✨ **实时预览** | 防抖半透明叠加,生成结果落世界前可见 |
| 📁 **文件支持** | 原生 `.vxlt`(gzip+状态)、MagicaVoxel `.vox` 导入(v150 + v200 多模型场景图)/导出(v150),Wavefront `.obj` 和 glTF Binary `.glb` 导出。OBJ/GLB 还有 Marching Cubes "smoothed" 变体(light: 圆角方块 / heavy: 黏土感)支持有机模型导出 |
| 💾 **状态持久化** | 窗口布局、面板状态、笔刷与调色板、最近文件跨重启保留;生成器参数住在节点图里,随工程走而不是随机器走 |
| 🖥️ **视口控制** | 轨道相机(每次开始 orbit 自动从相机当前状态同步)、网格、坐标轴、线框模式 |
| 💡 **逐顶点 AO** | Minecraft 风格的环境光遮蔽烘焙到 greedy mesh — 角落和凹陷自动变暗,开阔面保持明亮。视觉立体感显著提升,运行时零成本 |

## 快速开始

```bash
git clone https://github.com/Lynthar/Voxelith.git
cd Voxelith
cargo run --release

# 无头批量导出:spec 里列出的每个 .vxlt → .glb,
# 逐资产指定 pivot / 上轴 / 缩放。不开窗口、不用 GPU。
cargo run --release -- bake assets/spec.json

# 用 JSON 编辑协议从命令行(或 AI agent)驱动建模原语:
# 提交一批操作,读报告,看切片。
cargo run --release -- exec ops.json --out hut.vxlt --describe
cargo run --release -- inspect hut.vxlt --slice '{"axis":"y","index":1}'
cargo run --release -- render hut.vxlt --view all   # 看一眼:CPU 光线投射出图,不用 GPU
cargo run --release -- generators        # generate op 能调用哪些生成器

# 或者把同一套原语挂到 Model Context Protocol 上,一份文档跨调用常驻
# (默认 stdio;--http 需要 mcp-http feature)
# 加 --checkpoint 后每次编辑都写回工程文件,编辑器会自动重载 ——
# 开着那个 .vxlt 就能看着 agent 一步步建模
cargo run --release -- mcp --root ./models --checkpoint

# 或者干脆不经过文件:编辑器自己就托管一个 server,agent 直接编辑你
# 开着的工程,批次进的是你自己的撤销栈。把客户端指向它打印的 URL
# (仅回环地址)
cargo run --release -- --agent-port 8737
```

以上子命令全部无头,`--agent-port` 那条则是编辑器本身。
`cargo build --no-default-features` 构建的正是无头这一半——库加 CLI,
依赖树里没有 winit / wgpu / egui——供没有 GPU 的容器或 CI 使用。想在
里面保留 `mcp` 子命令要加 `--features mcp`:它跟着自己的 feature 走,
纯 `--no-default-features` 的构建里没有这个子命令。参数见 `voxelith exec --help`,生成器目录见
`voxelith generators`,ops 协议本身的说明写在 `src/agent_ops/schema.rs`
的类型文档上。若要让 AI agent 来驱动这个仓库,该读的是
`.claude/skills/voxelith-modeling/`:三条驱动路径怎么选、完整 op 词表,
以及那些能让第一次尝试就不出错的建模技法。

## 快捷键

macOS 上,下表中的 `Ctrl` 一律换成 ⌘。

| 按键 | 功能 | 按键 | 功能 |
|------|------|------|------|
| `1-5` | 笔刷工具 | `Ctrl+Z` | 撤销 |
| `6-9` | 形状工具 | `Ctrl+Y` / `Ctrl+Shift+Z` | 重做 |
| `0` | 盒选工具 | `Ctrl+C/X/V` | 复制 / 剪切 / 粘贴 |
| `WASD` | 移动相机 | `Ctrl+Shift+V` | 粘到光标 |
| `Q` / `E` | 相机上 / 下 | `Del` | 删除选区 |
| `鼠标中键` | 轨道旋转 | `Ctrl+A` | 全选所有非空体素 |
| `鼠标右键` | 平移 | `Esc / Ctrl+D` | 取消选区 |
| `滚轮` | 缩放 | `方向键 / Ctrl+↑↓` | 微调选区 |
| `F` | 框选区(无选区则框全场景) | `R` / `Shift+R` | 选区绕 Y 轴旋转 ±90° |
| `Ctrl+S/O/N` | 保存 / 打开 / 新建 | `M` | 选区沿 X 轴镜像 |
| `Ctrl+Shift+S` | 另存为 | `Alt`(按住) | 取色 |

## 技术栈

- 🦀 **Rust** - 系统级语言
- 🎮 **wgpu** - GPU 渲染
- 🖼️ **egui** - 即时模式 UI
- 🗜️ **flate2** - 压缩算法

## 架构设计

```
┌──────────────────────────────────────────────┐
│ UI(egui 面板 + 可视化节点图编辑器)         │
├──────────────────────────────────────────────┤
│ 编辑器(工具、命令、射线拾取、undo)         │
├──────────────────────────────────────────────┤
│ 程序化生成(地形 / 树 / WFC + DAG 求值)     │
├──────────────────────────────────────────────┤
│ 核心(体素、区块、世界)  │ 网格生成        │
│ 渲染(wgpu)              │ 文件 I/O 偏好    │
└──────────────────────────────────────────────┘
```

当前实现状态、剩余计划与设计不变量见 [`docs/STATUS.md`](docs/STATUS.md)。

## 许可证

Apache License 2.0 © 2024-2026 Lynthar
