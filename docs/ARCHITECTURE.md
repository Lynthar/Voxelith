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

### 1.3 AI 3D 生成技术现状

| 技术/模型 | 类型 | 输出格式 | 特点 |
|-----------|------|----------|------|
| **Shap-E** (OpenAI) | Text/Image-to-3D | NeRF/Mesh | 快速生成、质量中等 |
| **Point-E** (OpenAI) | Text-to-3D | 点云 | 开源、可本地运行 |
| **XCube** (NVIDIA) | 3D Diffusion | 稀疏体素 | 1024³分辨率、大规模场景 |
| **Meta 3D Gen** | Text-to-3D | Mesh+PBR | 30秒生成、纹理完整 |
| **3D-UDDPM** | Diffusion | 体素 | 局部场景细节 |
| **MarioGPT** | LLM-based PCG | 2D关卡 | 自然语言控制 |

### 1.4 市场空白

**关键发现**: 目前没有一款工具同时具备：
1. ✅ 强大的手动体素编辑能力
2. ✅ 内置程序化生成系统
3. ✅ 可视化规则编辑器
4. ✅ 模块化资产组合
5. ✅ 实时预览与导出
6. ✅ **AI 辅助生成能力**
7. ✅ **传统算法 + AI 混合工作流**

**Voxelith 的定位**: 填补这一空白，成为第一个**程序化优先 + AI 增强**的体素创作工具。

---

## 2. 核心功能设计

### 2.1 功能模块

```
┌───────────────────────────────────────────────────────────────────────────┐
│                              Voxelith                                     │
├───────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │  手动编辑器  │  │ 程序化引擎  │  │  AI 生成    │  │  资产管理   │      │
│  │  - 绘制     │  │  - WFC      │  │  - 文本生成 │  │  - 模板库   │      │
│  │  - 选择     │  │  - Noise    │  │  - 图像生成 │  │  - 组合件   │      │
│  │  - 变换     │  │  - L-System │  │  - 草图生成 │  │  - 规则集   │      │
│  │  - 图层     │  │  - Grammar  │  │  - 变体生成 │  │  - 导入导出 │      │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │  预览系统   │  │  规则编辑器  │  │  AI 细化    │  │  导出管线   │      │
│  │  - 实时渲染 │  │  - 可视化   │  │  - 细节补充 │  │  - .vox     │      │
│  │  - 多视图   │  │  - 节点图   │  │  - 风格迁移 │  │  - .gltf    │      │
│  │  - 光照     │  │  - 约束定义 │  │  - 智能修复 │  │  - .obj     │      │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
└───────────────────────────────────────────────────────────────────────────┘
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
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                   用户界面层                                      │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐     │
│  │   3D视口   │ │  属性面板  │ │ 节点编辑器 │ │  AI助手    │ │  资产浏览  │     │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘ └────────────┘     │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                   应用逻辑层                                      │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                        命令/撤销系统 (Command Pattern)                      │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐     │
│  │ 编辑器控制 │ │ 生成器管理 │ │ AI编排器   │ │  资产系统  │ │  项目管理  │     │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘ └────────────┘     │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                   核心引擎层                                      │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐     │
│  │  体素存储  │ │ 程序化生成 │ │  AI生成    │ │  网格生成  │ │  渲染引擎  │     │
│  │ -稀疏八叉树│ │  - WFC     │ │ -模型推理  │ │  - Greedy  │ │  - wgpu    │     │
│  │ - RLE压缩  │ │  - Noise   │ │ -格式转换  │ │  -Marching │ │  - 光照    │     │
│  │ - 分块加载 │ │  - L-System│ │ -后处理    │ │  - Naive   │ │  - AO      │     │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘ └────────────┘     │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                 AI 集成抽象层                                     │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐     │
│  │ 生成器接口 │ │ 本地推理   │ │ 远程API    │ │ 格式转换   │ │  模型管理  │     │
│  │(Generator) │ │ (ONNX/     │ │ (REST/     │ │(点云/NeRF/ │ │ (下载/缓存 │     │
│  │            │ │  Candle)   │ │  gRPC)     │ │ Mesh→Voxel)│ │  /版本)    │     │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘ └────────────┘     │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                   平台抽象层                                      │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐     │
│  │  窗口系统  │ │  输入处理  │ │  文件系统  │ │  并行计算  │ │  GPU计算   │     │
│  │  (winit)   │ │            │ │            │ │  (rayon)   │ │  (wgpu)    │     │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘ └────────────┘     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 模块职责

| 模块 | 职责 | 关键技术 |
|------|------|----------|
| **VoxelCore** | 体素数据的存储、访问、修改 | 稀疏八叉树、RLE、分块 |
| **ProcGen** | 程序化生成算法实现 | WFC、Noise、L-System |
| **AIGen** | AI模型推理与格式转换 | ONNX、Candle、体素化 |
| **AIOrchestrator** | AI工作流编排与混合生成 | 管线组合、缓存策略 |
| **Mesher** | 体素到三角形网格转换 | Greedy Meshing、Marching Cubes |
| **Renderer** | GPU渲染、光照、材质 | wgpu、PBR、SSAO |
| **Editor** | 用户交互、工具、编辑操作 | ECS、Command Pattern |
| **NodeGraph** | 可视化规则编辑 | 节点图评估器 |
| **AssetMgr** | 资产加载、保存、格式转换 | VOX、GLTF、自定义格式 |
| **ModelMgr** | AI模型下载、缓存、版本管理 | HuggingFace Hub、本地存储 |
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

## 6. AI 生成架构设计

### 6.1 设计原则

为了让 Voxelith 能够灵活适应快速发展的 AI 3D 生成技术，架构设计遵循以下原则：

1. **抽象接口优先**: 统一的生成器接口，传统算法与 AI 模型共用
2. **本地/远程透明**: 支持本地模型推理和远程 API 调用，对上层透明
3. **格式转换管线**: 点云、NeRF、Mesh 等多种输出统一转换为体素
4. **混合工作流**: AI 生成粗稿 + 传统算法细化 + 手动编辑
5. **渐进式集成**: 核心功能不依赖 AI，AI 作为增强模块

### 6.2 生成器抽象接口

```rust
/// 统一的生成器接口 - 传统算法和AI模型都实现此trait
#[async_trait]
pub trait VoxelGenerator: Send + Sync {
    /// 生成器元数据
    fn metadata(&self) -> GeneratorMeta;

    /// 生成体素数据
    async fn generate(&self, input: GenInput, ctx: &GenContext) -> Result<VoxelData>;

    /// 是否支持增量/局部生成
    fn supports_incremental(&self) -> bool { false }

    /// 预估生成时间 (用于UI显示进度)
    fn estimate_duration(&self, input: &GenInput) -> Duration;
}

/// 生成器输入类型
pub enum GenInput {
    // 传统算法输入
    Noise { seed: u64, params: NoiseParams },
    WFC { tileset: TilesetId, constraints: WfcConstraints },

    // AI 模型输入
    Text { prompt: String, negative_prompt: Option<String> },
    Image { image: ImageData, depth_hint: Option<ImageData> },
    Sketch { strokes: Vec<Stroke3D>, style: StyleRef },
    Voxel { base: VoxelData, instruction: String },  // 基于现有体素的修改

    // 混合输入
    Hybrid { components: Vec<GenInput> },
}

/// 生成器元数据
pub struct GeneratorMeta {
    pub id: String,
    pub name: String,
    pub category: GeneratorCategory,
    pub backend: GeneratorBackend,
    pub capabilities: Capabilities,
}

pub enum GeneratorCategory {
    Terrain,
    Building,
    Character,
    Prop,
    Vegetation,
    General,
}

pub enum GeneratorBackend {
    Algorithmic,       // WFC, Noise, L-System 等
    LocalModel,        // 本地 ONNX/Candle 模型
    RemoteAPI,         // OpenAI, Replicate 等
    Hybrid,            // 混合
}
```

### 6.3 AI 集成层架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          AI 编排器 (Orchestrator)                        │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                      工作流引擎 (Pipeline)                           ││
│  │   [Text] → [AI生成] → [格式转换] → [后处理] → [体素化] → [输出]       ││
│  └─────────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────────┤
│                          生成器注册表 (Registry)                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │   WFC生成器  │ │  噪声生成器  │ │  Shap-E适配  │ │  自定义模型  │   │
│  │(Algorithmic) │ │(Algorithmic) │ │ (LocalModel) │ │ (RemoteAPI)  │   │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │
├─────────────────────────────────────────────────────────────────────────┤
│                          推理后端 (Inference Backend)                    │
│  ┌──────────────────────────────┐ ┌──────────────────────────────────┐ │
│  │       本地推理引擎            │ │         远程API客户端             │ │
│  │  ┌────────┐ ┌────────┐       │ │  ┌────────┐ ┌────────┐          │ │
│  │  │ ONNX   │ │ Candle │       │ │  │OpenAI  │ │Replicate│         │ │
│  │  │Runtime │ │(Rust ML)│      │ │  │  API   │ │  API   │          │ │
│  │  └────────┘ └────────┘       │ │  └────────┘ └────────┘          │ │
│  │  ┌────────┐ ┌────────┐       │ │  ┌────────┐ ┌────────┐          │ │
│  │  │ llama  │ │  burn  │       │ │  │自建服务│ │HuggingFace│       │ │
│  │  │ .cpp   │ │        │       │ │  │        │ │Inference│         │ │
│  │  └────────┘ └────────┘       │ │  └────────┘ └────────┘          │ │
│  └──────────────────────────────┘ └──────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────┤
│                          格式转换管线 (Converter Pipeline)               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │  点云→体素   │ │  NeRF→体素   │ │  Mesh→体素   │ │  SDF→体素    │   │
│  │(Voxelization)│ │ (Sampling)   │ │(Rasterization│ │ (Threshold)  │   │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │
├─────────────────────────────────────────────────────────────────────────┤
│                          模型管理器 (Model Manager)                      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │  模型下载    │ │  版本管理    │ │  缓存策略    │ │  许可证检查  │   │
│  │(HuggingFace) │ │              │ │  (LRU)       │ │              │   │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.4 格式转换: 从 AI 输出到体素

```rust
/// 3D 数据格式转换器
pub trait ToVoxel {
    fn to_voxel(&self, config: VoxelizeConfig) -> Result<VoxelData>;
}

/// 点云转体素 (如 Point-E 输出)
impl ToVoxel for PointCloud {
    fn to_voxel(&self, config: VoxelizeConfig) -> Result<VoxelData> {
        // 1. 确定包围盒和分辨率
        // 2. 为每个点分配到最近的体素格
        // 3. 可选: 使用 Poisson 重建填充内部
        // 4. 颜色/材质映射
    }
}

/// 三角网格转体素 (如 Shap-E, Meta 3D Gen 输出)
impl ToVoxel for TriMesh {
    fn to_voxel(&self, config: VoxelizeConfig) -> Result<VoxelData> {
        // 1. 光栅化: 遍历三角形，标记相交的体素
        // 2. 填充: 使用扫描线或flood fill填充内部
        // 3. 颜色采样: 从纹理/顶点色映射到体素
    }
}

/// NeRF/SDF 隐式表示转体素
impl ToVoxel for ImplicitField {
    fn to_voxel(&self, config: VoxelizeConfig) -> Result<VoxelData> {
        // 1. 在3D网格上均匀采样
        // 2. 查询每点的密度/SDF值
        // 3. 阈值化确定占用
        // 4. Marching Cubes 可选平滑
    }
}

pub struct VoxelizeConfig {
    pub resolution: [u32; 3],       // 目标分辨率
    pub fill_interior: bool,        // 是否填充内部
    pub color_mode: ColorMode,      // 颜色处理方式
    pub simplify: bool,             // 是否简化 (减少体素数)
}
```

### 6.5 混合生成工作流

```
┌─────────────────────────────────────────────────────────────────┐
│                    混合生成管线示例                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  用户输入: "一个中世纪城堡，有四个塔楼"                           │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐                                                │
│  │  AI 粗稿    │  ← Text-to-3D 生成基础形状                      │
│  │  (Shap-E)   │    输出: 低分辨率 mesh                         │
│  └─────────────┘                                                │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐                                                │
│  │  体素化     │  ← Mesh → Voxel 转换                           │
│  │  64³分辨率  │    保持基础结构                                 │
│  └─────────────┘                                                │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐                                                │
│  │  WFC 细化   │  ← 使用"中世纪城堡"模块集                       │
│  │  (传统算法) │    在AI粗稿约束下填充细节                       │
│  └─────────────┘                                                │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐                                                │
│  │  噪声纹理   │  ← Perlin噪声添加石材纹理变化                   │
│  │  (传统算法) │                                                │
│  └─────────────┘                                                │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐                                                │
│  │  手动调整   │  ← 用户编辑器微调                               │
│  │  (编辑器)   │                                                │
│  └─────────────┘                                                │
│       │                                                         │
│       ▼                                                         │
│     输出: 高质量体素城堡模型                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.6 AI 相关技术选型

| 组件 | 推荐方案 | 备选方案 | 理由 |
|------|----------|----------|------|
| **本地推理** | Candle (Rust) | ONNX Runtime | 纯Rust、GPU支持、与wgpu兼容 |
| **模型格式** | ONNX + Safetensors | PyTorch | 跨框架、安全加载 |
| **远程API** | reqwest + tokio | ureq | 异步支持、连接池 |
| **LLM集成** | llama.cpp bindings | Ollama API | 本地运行、低延迟 |
| **模型下载** | hf-hub (Rust) | 手动下载 | HuggingFace官方支持 |

### 6.7 节点图中的 AI 节点

```rust
/// 扩展 GenNode 枚举以支持 AI 节点
pub enum GenNode {
    // ... 原有的 Noise, WFC 等 ...

    /// AI 文本生成节点
    AITextToVoxel {
        model: ModelRef,
        prompt: String,
        negative_prompt: Option<String>,
        resolution: u32,
        seed: Option<u64>,
    },

    /// AI 图像生成节点
    AIImageToVoxel {
        model: ModelRef,
        image_input: NodeId,  // 连接到图像输入节点
        depth_estimation: bool,
    },

    /// AI 变体生成节点
    AIVariation {
        model: ModelRef,
        base_voxel: NodeId,   // 连接到现有体素
        variation_strength: f32,
    },

    /// AI 风格迁移节点
    AIStyleTransfer {
        content: NodeId,
        style_ref: StyleRef,  // 风格参考 (图像或预设)
    },

    /// AI 补全/扩展节点
    AIInpaint {
        base_voxel: NodeId,
        mask: MaskData,       // 需要补全的区域
        context_prompt: Option<String>,
    },
}
```

---

## 7. 数据结构设计

### 7.1 体素存储

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

### 7.2 程序化规则

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

    // AI 生成节点 (见 6.7 节)
    AITextToVoxel { /* ... */ },
    AIImageToVoxel { /* ... */ },
    AIVariation { /* ... */ },
}

pub struct GeneratorGraph {
    nodes: Vec<GenNode>,
    edges: Vec<(NodeId, NodeId)>,
}
```

---

## 8. 项目结构

```
voxelith/
├── Cargo.toml
├── src/
│   ├── main.rs                 # 入口
│   ├── lib.rs                  # 库导出
│   │
│   ├── core/                   # 核心数据结构
│   │   ├── mod.rs
│   │   ├── voxel.rs            # 体素数据结构
│   │   ├── chunk.rs            # 分块管理
│   │   ├── octree.rs           # 八叉树实现
│   │   └── world.rs            # 世界管理
│   │
│   ├── procgen/                # 传统程序化生成
│   │   ├── mod.rs
│   │   ├── generator.rs        # 生成器trait定义 (统一接口)
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
│   │
│   ├── ai/                     # AI 生成模块 (可选feature)
│   │   ├── mod.rs
│   │   ├── orchestrator.rs     # AI工作流编排
│   │   ├── registry.rs         # 生成器注册表
│   │   ├── inference/          # 推理后端
│   │   │   ├── mod.rs
│   │   │   ├── local.rs        # 本地推理 (Candle/ONNX)
│   │   │   └── remote.rs       # 远程API调用
│   │   ├── models/             # 模型适配器
│   │   │   ├── mod.rs
│   │   │   ├── shap_e.rs       # Shap-E 适配
│   │   │   ├── point_e.rs      # Point-E 适配
│   │   │   └── custom.rs       # 自定义模型支持
│   │   ├── convert/            # 格式转换
│   │   │   ├── mod.rs
│   │   │   ├── pointcloud.rs   # 点云→体素
│   │   │   ├── mesh.rs         # Mesh→体素
│   │   │   ├── nerf.rs         # NeRF→体素
│   │   │   └── sdf.rs          # SDF→体素
│   │   └── model_manager.rs    # 模型下载/缓存管理
│   │
│   ├── mesh/                   # 网格生成
│   │   ├── mod.rs
│   │   ├── greedy.rs           # 贪心网格化
│   │   └── marching.rs         # Marching Cubes
│   │
│   ├── render/                 # 渲染引擎
│   │   ├── mod.rs
│   │   ├── pipeline.rs         # 渲染管线
│   │   ├── camera.rs           # 相机控制
│   │   └── shaders/
│   │       ├── voxel.wgsl
│   │       └── lighting.wgsl
│   │
│   ├── editor/                 # 编辑器逻辑
│   │   ├── mod.rs
│   │   ├── tools/              # 编辑工具
│   │   ├── commands.rs         # 命令系统
│   │   └── history.rs          # 撤销/重做
│   │
│   ├── ui/                     # 用户界面
│   │   ├── mod.rs
│   │   ├── viewport.rs         # 3D视口
│   │   ├── properties.rs       # 属性面板
│   │   ├── nodegraph.rs        # 节点编辑器
│   │   ├── ai_panel.rs         # AI助手面板
│   │   └── assets.rs           # 资产浏览器
│   │
│   └── io/                     # 输入输出
│       ├── mod.rs
│       ├── vox.rs              # MagicaVoxel格式
│       ├── gltf.rs             # GLTF导出
│       └── project.rs          # 项目文件
│
├── assets/
│   ├── tilesets/               # WFC模块集
│   ├── templates/              # 生成模板
│   ├── models/                 # AI模型缓存 (gitignore)
│   └── shaders/                # 着色器
│
└── docs/
    └── ARCHITECTURE.md         # 本文档
```

---

## 9. 开发路线图

### Phase 1: 核心基础 (MVP)
- [ ] 基础体素数据结构 (Chunk + Octree)
- [ ] wgpu 渲染管线
- [ ] 简单体素绘制工具
- [ ] 基础 UI 框架 (egui)
- [ ] 项目保存/加载
- [ ] **统一生成器接口设计 (为AI预留)**

### Phase 2: 程序化生成
- [ ] Perlin/Simplex 噪声地形
- [ ] 基础 WFC 实现
- [ ] 可视化参数调节
- [ ] **格式转换基础框架 (Mesh→Voxel)**

### Phase 3: 高级生成
- [ ] 完整 WFC + 回溯
- [ ] L-System 植被
- [ ] 形状语法建筑
- [ ] 节点图规则编辑器

### Phase 4: AI 集成 (Alpha)
- [ ] 生成器注册表与编排器
- [ ] 远程 API 集成 (OpenAI/Replicate)
- [ ] 基础 Text-to-Voxel 工作流
- [ ] AI 助手面板 UI
- [ ] 点云/Mesh 到体素转换

### Phase 5: AI 增强 (Beta)
- [ ] 本地模型推理 (Candle/ONNX)
- [ ] 模型管理器 (下载/缓存)
- [ ] 混合生成管线 (AI + 传统算法)
- [ ] AI 节点集成到节点图编辑器
- [ ] 变体生成与风格迁移

### Phase 6: 优化与生态
- [ ] Greedy Meshing 优化
- [ ] 多线程/GPU生成
- [ ] 更多导出格式
- [ ] 模板库与资产市场

### Phase 7: 扩展
- [ ] WASM/WebGPU 浏览器版本
- [ ] 脚本系统 (Lua/Rhai)
- [ ] 插件 API
- [ ] 自定义 AI 模型支持

---

## 10. 参考资源

### 程序化生成算法
- [Wave Function Collapse - mxgmn](https://github.com/mxgmn/WaveFunctionCollapse)
- [Infinite City with WFC - marian42](https://marian42.de/article/wfc/)
- [Red Blob Games - Terrain from Noise](https://www.redblobgames.com/maps/terrain-from-noise/)
- [Boris the Brave - WFC Tips](https://www.boristhebrave.com/2020/02/08/wave-function-collapse-tips-and-tricks/)

### 体素技术
- [wgpu 官方文档](https://wgpu.rs/)
- [Rezcraft - Rust Voxel Engine](https://github.com/Shapur1234/Rezcraft)
- [vengi - 开源体素工具](https://github.com/vengi-voxel/vengi)

### AI 3D 生成
- [Shap-E - OpenAI](https://github.com/openai/shap-e) - Text/Image to 3D
- [Point-E - OpenAI](https://github.com/openai/point-e) - Text to Point Cloud
- [XCube - NVIDIA](https://research.nvidia.com/labs/toronto-ai/publication/2024_cvpr_xcube/) - 大规模稀疏体素生成
- [Awesome 3D Diffusion](https://github.com/cwchenwang/awesome-3d-diffusion) - 3D Diffusion 论文合集
- [Text-to-3D 综述](https://onlinelibrary.wiley.com/doi/10.1111/cgf.15061) - Computer Graphics Forum

### Rust ML/AI 生态
- [Candle](https://github.com/huggingface/candle) - HuggingFace Rust ML框架
- [Burn](https://github.com/burn-rs/burn) - Rust深度学习框架
- [ONNX Runtime Rust](https://github.com/pykeio/ort) - ONNX推理

### 工具参考
- [MagicaVoxel](https://ephtracy.github.io/)
- [Goxel](https://github.com/guillaumechereau/goxel)
- [Avoyd](https://www.avoyd.com/)
