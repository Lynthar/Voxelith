# Voxelith - 体素风格游戏素材程序化生成工具

## 1. 现有工具分析

### 1.1 主流体素编辑器

| 工具 | 优点 | 缺点 | 程序化能力 |
|------|------|------|-----------|
| **MagicaVoxel** | 免费、渲染优秀、支持2048³体素 | 无程序化生成、闭源 | ❌ 无 |
| **Goxel** | 开源、跨平台、图层支持 | 功能较少、无动画 | ❌ 无 |
| **Qubicle** | 专业级、无限矩阵、多格式导出 | 付费、无程序化 | ❌ 无 |
| **VoxEdit** | 免费、动画支持 | 仅限Sandbox生态 | ❌ 无 |
| **Avoyd** | 256k体素、路径追踪 | 付费、学习曲线陡 | ❌ 无 |
| **vengi** | 开源、多格式、动画支持 | 社区较小 | ⚠️ 有限 |

### 1.2 程序化生成工具

| 工具/算法 | 适用场景 | 特点 |
|-----------|----------|------|
| **Wave Function Collapse** | 城市、建筑、地牢 | 基于约束的模块拼接 |
| **Perlin/Simplex Noise** | 地形、洞穴、纹理 | 连续梯度噪声 |
| **L-System** | 植被、树木、分形结构 | 递归语法规则 |
| **Marching Cubes** | 平滑体素到网格 | 等值面提取 |
| **Cellular Automata** | 洞穴、有机形态 | 邻域规则演化 |

### 1.3 市场空白

**关键发现**: 目前没有一款工具同时具备：
1. ✅ 强大的手动体素编辑能力
2. ✅ 内置程序化生成系统
3. ✅ 可视化规则编辑器
4. ✅ 模块化资产组合
5. ✅ 实时预览与导出

**Voxelith 的定位**: 填补这一空白，成为第一个**程序化优先**的体素创作工具。

---

## 2. 核心功能设计

### 2.1 功能模块

```
┌─────────────────────────────────────────────────────────────────┐
│                         Voxelith                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  手动编辑器  │  │ 程序化引擎  │  │  资产管理   │              │
│  │  - 绘制     │  │  - WFC      │  │  - 模板库   │              │
│  │  - 选择     │  │  - Noise    │  │  - 组合件   │              │
│  │  - 变换     │  │  - L-System │  │  - 规则集   │              │
│  │  - 图层     │  │  - Grammar  │  │  - 导入导出 │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  预览系统   │  │  规则编辑器  │  │  导出管线   │              │
│  │  - 实时渲染 │  │  - 可视化   │  │  - .vox     │              │
│  │  - 多视图   │  │  - 节点图   │  │  - .gltf    │              │
│  │  - 光照     │  │  - 约束定义 │  │  - .obj     │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 程序化生成能力

#### 地形生成
- 多层噪声叠加 (FBM/分形布朗运动)
- 生物群系分布
- 洞穴雕刻 (3D Perlin + 阈值)
- 水体与河流

#### 建筑生成
- 形状语法 (Shape Grammar)
- WFC 模块拼接
- 参数化窗户/门/屋顶
- 风格模板 (中世纪/现代/科幻)

#### 场景布置
- 散布系统 (植被、岩石、道具)
- 密度图控制
- 群落规则

#### 角色/物品
- 对称生成器
- 部件组合 (头/身/手/武器)
- 颜色变体生成

---

## 3. 整体架构

### 3.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              用户界面层                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │   3D视口     │ │  属性面板    │ │  节点编辑器  │ │   资产浏览   │   │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │
├─────────────────────────────────────────────────────────────────────────┤
│                              应用逻辑层                                  │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                       命令/撤销系统 (Command Pattern)              │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │  编辑器控制  │ │  生成器管理  │ │  资产系统    │ │   项目管理   │   │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │
├─────────────────────────────────────────────────────────────────────────┤
│                              核心引擎层                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │  体素存储    │ │  程序化生成  │ │  网格生成    │ │   渲染引擎   │   │
│  │  - 稀疏八叉树│ │  - WFC       │ │  - Greedy    │ │   - wgpu     │   │
│  │  - RLE压缩   │ │  - Noise     │ │  - Marching  │ │   - 光照     │   │
│  │  - 分块加载  │ │  - L-System  │ │  - Naive     │ │   - AO       │   │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │
├─────────────────────────────────────────────────────────────────────────┤
│                              平台抽象层                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │  窗口系统    │ │   输入处理   │ │   文件系统   │ │   并行计算   │   │
│  │  (winit)     │ │              │ │              │ │   (rayon)    │   │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 模块职责

| 模块 | 职责 | 关键技术 |
|------|------|----------|
| **VoxelCore** | 体素数据的存储、访问、修改 | 稀疏八叉树、RLE、分块 |
| **ProcGen** | 程序化生成算法实现 | WFC、Noise、L-System |
| **Mesher** | 体素到三角形网格转换 | Greedy Meshing、Marching Cubes |
| **Renderer** | GPU渲染、光照、材质 | wgpu、PBR、SSAO |
| **Editor** | 用户交互、工具、编辑操作 | ECS、Command Pattern |
| **NodeGraph** | 可视化规则编辑 | 节点图评估器 |
| **AssetMgr** | 资产加载、保存、格式转换 | VOX、GLTF、自定义格式 |
| **UI** | 用户界面组件 | egui |

---

## 4. 技术栈选择

### 4.1 推荐方案: Rust + WebGPU

```
┌─────────────────────────────────────────────────────────┐
│                    技术栈组成                            │
├─────────────────────────────────────────────────────────┤
│  语言:      Rust                                        │
│  图形API:   wgpu (WebGPU抽象层)                         │
│  UI框架:    egui                                        │
│  窗口:      winit                                       │
│  数学库:    glam / nalgebra                            │
│  序列化:    serde + bincode/ron                        │
│  并行:      rayon                                       │
│  噪声:      noise-rs / fastnoise-lite                  │
│  ECS:       bevy_ecs (可选)                            │
├─────────────────────────────────────────────────────────┤
│  目标平台:                                              │
│  - Desktop: Windows, macOS, Linux                      │
│  - Web: WASM + WebGPU (渐进式支持)                      │
└─────────────────────────────────────────────────────────┘
```

### 4.2 为什么选择 Rust + wgpu？

| 考量因素 | Rust + wgpu | C++ + OpenGL | TypeScript + Three.js |
|----------|-------------|--------------|----------------------|
| **性能** | ⭐⭐⭐⭐⭐ 原生 | ⭐⭐⭐⭐⭐ 原生 | ⭐⭐⭐ 受限于JS |
| **安全性** | ⭐⭐⭐⭐⭐ 内存安全 | ⭐⭐ 手动管理 | ⭐⭐⭐⭐ GC |
| **跨平台** | ⭐⭐⭐⭐⭐ 含Web | ⭐⭐⭐ 需移植 | ⭐⭐⭐⭐⭐ 天然Web |
| **开发效率** | ⭐⭐⭐⭐ 现代工具链 | ⭐⭐ 复杂 | ⭐⭐⭐⭐⭐ 快速 |
| **社区/生态** | ⭐⭐⭐⭐ 活跃增长 | ⭐⭐⭐⭐⭐ 成熟 | ⭐⭐⭐⭐ 大量库 |
| **未来趋势** | ⭐⭐⭐⭐⭐ WebGPU | ⭐⭐ 逐渐淘汰 | ⭐⭐⭐⭐ 依赖WebGPU |

### 4.3 备选方案

**方案B: Godot + GDExtension (Rust)**
- 优点: 成熟的编辑器框架、快速原型
- 缺点: 定制受限、额外抽象层

**方案C: Tauri + WebGPU (Rust后端 + Web前端)**
- 优点: 现代UI开发体验、热重载
- 缺点: 前后端通信开销、调试复杂

---

## 5. 程序化生成算法详解

### 5.1 Wave Function Collapse (WFC)

最适合: **建筑、地牢、城市街道**

```
┌──────────────────────────────────────────────┐
│              WFC 工作流程                     │
├──────────────────────────────────────────────┤
│  1. 定义模块 (Modules/Tiles)                 │
│     - 每个模块是一个小型体素块               │
│     - 定义每面的连接器 (Connectors)          │
│                                              │
│  2. 定义约束 (Constraints)                   │
│     - 哪些连接器可以相邻                     │
│     - 频率权重                               │
│                                              │
│  3. 坍缩过程 (Collapse)                      │
│     - 找到熵最低的槽位                       │
│     - 随机选择一个有效模块                   │
│     - 传播约束到邻居                         │
│     - 重复直到所有槽位确定                   │
│                                              │
│  4. 回溯 (Backtracking)                      │
│     - 遇到矛盾时回退                         │
└──────────────────────────────────────────────┘
```

### 5.2 噪声地形生成

```rust
// 伪代码示例
fn generate_terrain(x: i32, z: i32) -> Vec<Voxel> {
    let mut voxels = Vec::new();

    // 基础高度 (多层噪声叠加)
    let height =
        perlin(x * 0.01, z * 0.01) * 64.0 +     // 大尺度山脉
        perlin(x * 0.05, z * 0.05) * 16.0 +     // 中尺度丘陵
        perlin(x * 0.1, z * 0.1) * 4.0;          // 小尺度细节

    // 生物群系决定 (温度/湿度)
    let temp = perlin(x * 0.005, z * 0.005);
    let humidity = perlin(x * 0.008 + 1000.0, z * 0.008);
    let biome = get_biome(temp, humidity);

    // 洞穴雕刻 (3D噪声)
    for y in 0..height as i32 {
        let cave_density = perlin_3d(x * 0.05, y * 0.05, z * 0.05);
        if cave_density > 0.6 { continue; } // 挖空

        voxels.push(get_block_for_biome(biome, y, height));
    }

    voxels
}
```

### 5.3 L-System 树木生成

```
规则定义:
  Axiom:   F
  Rules:   F -> FF+[+F-F-F]-[-F+F+F]

  F = 向前生长
  + = 右转25°
  - = 左转25°
  [ = 保存状态
  ] = 恢复状态
```

### 5.4 形状语法 (Shape Grammar)

适用于: **建筑立面、城市规划**

```
Building -> Foundation Floor* Roof
Floor    -> Wall Window Wall Door Wall
Wall     -> Brick | Stone | Wood
Window   -> frame(Glass) | empty
Roof     -> Flat | Gabled | Domed
```

---

## 6. 数据结构设计

### 6.1 体素存储

```rust
// 稀疏八叉树 + 分块
pub struct VoxelWorld {
    chunks: HashMap<ChunkPos, Chunk>,
    chunk_size: usize, // 通常 32 或 64
}

pub struct Chunk {
    // 使用 SVO (Sparse Voxel Octree) 压缩存储
    octree: SparseOctree<VoxelData>,
    // 或使用 RLE 压缩
    rle_data: Vec<RleRun<VoxelData>>,
    // 脏标记用于增量网格更新
    dirty: bool,
}

pub struct VoxelData {
    pub material: u16,      // 材质ID
    pub color: [u8; 4],     // RGBA
    pub flags: u8,          // 特殊标记
}
```

### 6.2 程序化规则

```rust
// 可视化节点图的抽象
pub enum GenNode {
    Noise {
        noise_type: NoiseType,
        frequency: f32,
        octaves: u32,
    },
    WFC {
        tileset: TilesetId,
        dimensions: [u32; 3],
    },
    Combine {
        op: CombineOp, // Add, Subtract, Intersect
        inputs: Vec<NodeId>,
    },
    Transform {
        offset: [i32; 3],
        rotation: u8,
        scale: f32,
    },
    Output,
}

pub struct GeneratorGraph {
    nodes: Vec<GenNode>,
    edges: Vec<(NodeId, NodeId)>,
}
```

---

## 7. 项目结构

```
voxelith/
├── Cargo.toml
├── src/
│   ├── main.rs                 # 入口
│   ├── lib.rs                  # 库导出
│   ├── core/
│   │   ├── mod.rs
│   │   ├── voxel.rs            # 体素数据结构
│   │   ├── chunk.rs            # 分块管理
│   │   ├── octree.rs           # 八叉树实现
│   │   └── world.rs            # 世界管理
│   ├── procgen/
│   │   ├── mod.rs
│   │   ├── wfc/
│   │   │   ├── mod.rs
│   │   │   ├── solver.rs       # WFC求解器
│   │   │   └── tileset.rs      # 模块定义
│   │   ├── noise/
│   │   │   ├── mod.rs
│   │   │   ├── terrain.rs      # 地形生成
│   │   │   └── caves.rs        # 洞穴生成
│   │   ├── lsystem/
│   │   │   ├── mod.rs
│   │   │   └── tree.rs         # 树木生成
│   │   └── grammar/
│   │       ├── mod.rs
│   │       └── building.rs     # 建筑语法
│   ├── mesh/
│   │   ├── mod.rs
│   │   ├── greedy.rs           # 贪心网格化
│   │   └── marching.rs         # Marching Cubes
│   ├── render/
│   │   ├── mod.rs
│   │   ├── pipeline.rs         # 渲染管线
│   │   ├── camera.rs           # 相机控制
│   │   └── shaders/
│   │       ├── voxel.wgsl
│   │       └── lighting.wgsl
│   ├── editor/
│   │   ├── mod.rs
│   │   ├── tools/              # 编辑工具
│   │   ├── commands.rs         # 命令系统
│   │   └── history.rs          # 撤销/重做
│   ├── ui/
│   │   ├── mod.rs
│   │   ├── viewport.rs         # 3D视口
│   │   ├── properties.rs       # 属性面板
│   │   ├── nodegraph.rs        # 节点编辑器
│   │   └── assets.rs           # 资产浏览器
│   └── io/
│       ├── mod.rs
│       ├── vox.rs              # MagicaVoxel格式
│       ├── gltf.rs             # GLTF导出
│       └── project.rs          # 项目文件
├── assets/
│   ├── tilesets/               # WFC模块集
│   ├── templates/              # 生成模板
│   └── shaders/                # 着色器
└── docs/
    └── ARCHITECTURE.md         # 本文档
```

---

## 8. 开发路线图

### Phase 1: 核心基础 (MVP)
- [ ] 基础体素数据结构 (Chunk + Octree)
- [ ] wgpu 渲染管线
- [ ] 简单体素绘制工具
- [ ] 基础 UI 框架 (egui)
- [ ] 项目保存/加载

### Phase 2: 程序化生成
- [ ] Perlin/Simplex 噪声地形
- [ ] 基础 WFC 实现
- [ ] 可视化参数调节

### Phase 3: 高级生成
- [ ] 完整 WFC + 回溯
- [ ] L-System 植被
- [ ] 形状语法建筑
- [ ] 节点图规则编辑器

### Phase 4: 优化与生态
- [ ] Greedy Meshing 优化
- [ ] 多线程生成
- [ ] 更多导出格式
- [ ] 模板库与资产市场

### Phase 5: 扩展
- [ ] WASM/WebGPU 浏览器版本
- [ ] 脚本系统 (Lua/Rhai)
- [ ] 插件 API

---

## 9. 参考资源

### 算法
- [Wave Function Collapse - mxgmn](https://github.com/mxgmn/WaveFunctionCollapse)
- [Infinite City with WFC - marian42](https://marian42.de/article/wfc/)
- [Red Blob Games - Terrain from Noise](https://www.redblobgames.com/maps/terrain-from-noise/)
- [Boris the Brave - WFC Tips](https://www.boristhebrave.com/2020/02/08/wave-function-collapse-tips-and-tricks/)

### 体素技术
- [wgpu 官方文档](https://wgpu.rs/)
- [Rezcraft - Rust Voxel Engine](https://github.com/Shapur1234/Rezcraft)
- [vengi - 开源体素工具](https://github.com/vengi-voxel/vengi)

### 工具参考
- [MagicaVoxel](https://ephtracy.github.io/)
- [Goxel](https://github.com/guillaumechereau/goxel)
- [Avoyd](https://www.avoyd.com/)
