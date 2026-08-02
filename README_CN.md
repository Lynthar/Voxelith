```
██╗   ██╗ ██████╗ ██╗  ██╗███████╗██╗     ██╗████████╗██╗  ██╗
██║   ██║██╔═══██╗╚██╗██╔╝██╔════╝██║     ██║╚══██╔══╝██║  ██║
██║   ██║██║   ██║ ╚███╔╝ █████╗  ██║     ██║   ██║   ███████║
╚██╗ ██╔╝██║   ██║ ██╔██╗ ██╔══╝  ██║     ██║   ██║   ██╔══██║
 ╚████╔╝ ╚██████╔╝██╔╝ ██╗███████╗███████╗██║   ██║   ██║  ██║
  ╚═══╝   ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝   ╚═╝   ╚═╝  ╚═╝
```

<div align="center">

**程序化优先的体素资产创作工具**

[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![wgpu](https://img.shields.io/badge/wgpu-22.0-blue.svg)](https://wgpu.rs/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[English](README.md)

</div>

---

## 概述

**Voxelith** 是一款使用 Rust 构建的现代体素编辑器，采用 wgpu 实现 GPU 加速渲染，配合简洁的 egui 界面。专为手动编辑和程序化生成设计。

## 功能特性

| 功能 | 说明 |
|------|------|
| 🎨 **编辑** | 5 个笔刷工具(放置/删除/绘制/取色/填充)+ 4 个形状工具(直线/盒/球/圆柱),按下-拖拽-释放手势。支持拖动笔刷、笔画级 undo、实时悬停预览、X/Y/Z 对称镜像 |
| ▭ **盒选** | `0` 切到 Select 工具。拖角创建 AABB,选区内拖动 = 整团搬运(单一可撤销 Command,正确处理重叠);方向键平移 X/Z(`Ctrl+↑↓` 走 Y 轴,`Shift` × 10)。`R`/`Shift+R` 旋转、`M` 镜像,各算一步 undo。`Ctrl+C/X/V`、`Ctrl+Shift+V` 粘到光标、`Del` 删除、`Ctrl+A` 选所有非空、`Esc`/`Ctrl+D` 取消。粘贴后自动选中目标 AABB,可链式 Paste→拖→Paste |
| ⚓ **挂点(Socket)** | 在任意体素面上放置命名挂点(位置 + 朝外法线),随工程保存,导出 GLB 时变成 glTF 空节点 —— 武器挂载、特效锚点、旗帜插槽,交给引擎挂接部件 |
| 🤖 **AI 生成** | 接 fal.ai Hunyuan3D 文生 3D:输入提示词,返回的网格自动体素化进场景,算一步可撤销编辑。跑在后台运行时(编辑器不卡),中途可取消;API key 存 OS 钥匙串,不落配置文件 |
| 🏷️ **游戏资产材质** | 每笔刷的自发光 / 金属标记,外加 4 档阵营 **tint zone**,导出 GLB 时分别落成 glTF materials 与逐顶点 `_TINTZONE` 属性,供下游换色 shader 使用 |
| 🌱 **程序化生成** | Perlin 地形、L-System 树、WFC 多套 tileset(Dungeon + City)—— 单生成器面板,或在可视化节点图里用 Translate / Filter / Mask / Combine 组合 |
| ✨ **实时预览** | 防抖半透明叠加,生成结果落世界前可见 |
| 📁 **文件支持** | 原生 `.vxlt`(gzip+状态)、MagicaVoxel `.vox` 导入(v150 + v200 多模型场景图)/导出(v150),Wavefront `.obj` 和 glTF Binary `.glb` 导出。OBJ/GLB 还有 Marching Cubes "smoothed" 变体(light: 圆角方块 / heavy: 黏土感)支持有机模型导出 |
| 💾 **状态持久化** | 窗口布局、面板状态、生成器参数、最近文件跨重启保留 |
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
cargo run --release -- generators        # generate op 能调用哪些生成器
```

以上子命令全部无头。`cargo build --no-default-features` 构建的正是这
一半——库加 CLI,依赖树里没有 winit / wgpu / egui——供没有 GPU 的容器
或 CI 使用。参数见 `voxelith exec --help`,生成器目录见
`voxelith generators`,ops 协议本身的说明写在 `src/agent_ops/schema.rs`
的类型文档上。

## 快捷键

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

MIT License © 2024
