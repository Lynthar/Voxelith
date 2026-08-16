//! The interactive viewport: wgpu device setup, pipelines, camera and
//! mesh upload. GPU and window only — the headless views live in
//! `crate::view`.

mod camera;
mod gpu_mesh;
mod grid;
mod pipeline;
mod selection;
mod socket;

pub use camera::{Camera, CameraController, CameraUniform};
pub use gpu_mesh::GpuMesh;
pub use grid::{AxisMesh, GridMesh, LinePipeline, LineVertex};
pub use pipeline::RenderPipeline;
pub use selection::SelectionMesh;
pub use socket::SocketMesh;

use crate::core::ChunkPos;
use crate::mesh::ChunkMesh;
use std::collections::HashMap;
use std::sync::Arc;

/// Main renderer state
pub struct Renderer {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub pipeline: RenderPipeline,
    pub line_pipeline: LinePipeline,
    pub camera: Camera,
    pub camera_controller: CameraController,
    pub chunk_meshes: HashMap<ChunkPos, GpuMesh>,
    pub depth_texture: wgpu::TextureView,
    pub grid_mesh: GridMesh,
    pub axis_mesh: AxisMesh,
    /// Translucent overlay mesh from the procgen preview, drawn with
    /// `pipeline.transparent_pipeline` after opaque chunks. `None`
    /// when preview is disabled or the generator output is empty.
    pub preview_mesh: Option<GpuMesh>,
    /// Translucent overlay showing the cells a click would land on,
    /// updated as the cursor moves.
    pub brush_preview_mesh: Option<GpuMesh>,
    /// Wireframe AABB for the active selection, or the live preview
    /// during a Select drag. Drawn through the same `LinePipeline` as
    /// the grid and axes; `None` when neither is present.
    pub selection_mesh: Option<SelectionMesh>,
    /// Translucent ghost of the picked-up voxels while a selection is
    /// being dragged. `None` unless a move drag is in progress; owned
    /// by `App::update_selection_visualization`.
    pub move_ghost_mesh: Option<GpuMesh>,
    /// Gizmo lines for the named sockets, through the same
    /// `LinePipeline` as the selection wireframe. `None` when the scene
    /// has none; rebuilt when the socket set changes.
    pub socket_mesh: Option<SocketMesh>,
    /// Whether wireframe mode is supported
    pub wireframe_supported: bool,
}

impl Renderer {
    /// Create a new renderer for the given window
    pub async fn new(window: Arc<winit::window::Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();

        // Create wgpu instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Create surface
        let surface = instance.create_surface(window.clone())?;

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to find suitable GPU adapter"))?;

        log::info!("Using GPU: {}", adapter.get_info().name);

        // Check if wireframe mode is supported
        let adapter_features = adapter.features();
        let wireframe_supported = adapter_features.contains(wgpu::Features::POLYGON_MODE_LINE);

        // Request device with optional wireframe support
        let required_features = if wireframe_supported {
            wgpu::Features::POLYGON_MODE_LINE
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Voxelith Device"),
                    required_features,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await?;

        if wireframe_supported {
            log::info!("Wireframe mode supported");
        } else {
            log::info!("Wireframe mode not supported on this GPU");
        }

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Create render pipeline with optional wireframe support
        let pipeline =
            RenderPipeline::new_with_features(&device, surface_format, required_features);

        // Create line pipeline (uses same camera bind group layout)
        let line_pipeline =
            LinePipeline::new(&device, surface_format, &pipeline.camera_bind_group_layout);

        // The same constant a headless-built project is saved with, so
        // a scene an agent made opens on the view a new one starts at.
        let camera = Camera::new(
            glam::Vec3::from_array(crate::io::DEFAULT_CAMERA_POSITION),
            glam::Vec3::ZERO,
            size.width as f32 / size.height as f32,
        );
        // Sync the controller's cached orbit state from the camera's
        // actual pose: its defaults don't match the position set above,
        // so the first path reading them would teleport the camera.
        let mut camera_controller = CameraController::new(0.5, 0.003);
        camera_controller.sync_orbit_state_from_camera(&camera);

        // Create depth texture
        let depth_texture = Self::create_depth_texture(&device, &config);

        // Create grid and axis meshes
        let grid_mesh = GridMesh::new(&device, 20, 1.0);
        let axis_mesh = AxisMesh::new(&device, 10.0);

        Ok(Self {
            device,
            queue,
            surface,
            config,
            pipeline,
            line_pipeline,
            camera,
            camera_controller,
            chunk_meshes: HashMap::new(),
            depth_texture,
            grid_mesh,
            axis_mesh,
            preview_mesh: None,
            brush_preview_mesh: None,
            selection_mesh: None,
            move_ghost_mesh: None,
            socket_mesh: None,
            wireframe_supported,
        })
    }

    /// Create depth texture for depth testing
    fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> wgpu::TextureView {
        let size = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            // RENDER_ATTACHMENT only: nothing samples this, and a
            // `TEXTURE_BINDING` a texture never needs can cost the
            // driver's compression paths for it.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Handle window resize
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture = Self::create_depth_texture(&self.device, &self.config);
            self.camera.aspect = new_size.width as f32 / new_size.height as f32;
        }
    }

    /// Update grid mesh with new settings
    pub fn update_grid(&mut self, size: i32, spacing: f32) {
        self.grid_mesh = GridMesh::new(&self.device, size, spacing);
    }

    /// Upload a chunk mesh to the GPU
    pub fn upload_mesh(&mut self, mesh: &ChunkMesh) {
        if mesh.is_empty() {
            self.chunk_meshes.remove(&mesh.chunk_pos);
            return;
        }

        let gpu_mesh = GpuMesh::new(&self.device, mesh);
        self.chunk_meshes.insert(mesh.chunk_pos, gpu_mesh);
    }

    /// Replace the procgen preview overlay. Empty mesh -> clear.
    pub fn set_preview_mesh(&mut self, mesh: &ChunkMesh) {
        if mesh.is_empty() {
            self.preview_mesh = None;
        } else {
            self.preview_mesh = Some(GpuMesh::new(&self.device, mesh));
        }
    }

    /// Clear the procgen preview overlay.
    pub fn clear_preview(&mut self) {
        self.preview_mesh = None;
    }

    /// Draw the preview overlay (if any) using the transparent pipeline.
    /// Must be invoked after opaque geometry in the same render pass so
    /// the depth buffer it reads is already populated.
    pub fn draw_preview<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if let Some(preview) = &self.preview_mesh {
            render_pass.set_pipeline(&self.pipeline.transparent_pipeline);
            render_pass.set_bind_group(0, &self.pipeline.camera_bind_group, &[]);
            preview.draw(render_pass);
        }
    }

    /// Replace the brush hover overlay. Empty mesh -> clear.
    pub fn set_brush_preview_mesh(&mut self, mesh: &ChunkMesh) {
        if mesh.is_empty() {
            self.brush_preview_mesh = None;
        } else {
            self.brush_preview_mesh = Some(GpuMesh::new(&self.device, mesh));
        }
    }

    /// Clear the brush hover overlay.
    pub fn clear_brush_preview(&mut self) {
        self.brush_preview_mesh = None;
    }

    /// Draw the brush hover overlay. Same depth/blend rules as
    /// `draw_preview` — call after opaque geometry.
    pub fn draw_brush_preview<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if let Some(preview) = &self.brush_preview_mesh {
            render_pass.set_pipeline(&self.pipeline.transparent_pipeline);
            render_pass.set_bind_group(0, &self.pipeline.camera_bind_group, &[]);
            preview.draw(render_pass);
        }
    }

    /// Replace the move-drag voxel ghost overlay. Empty mesh -> clear.
    pub fn set_move_ghost_mesh(&mut self, mesh: &ChunkMesh) {
        if mesh.is_empty() {
            self.move_ghost_mesh = None;
        } else {
            self.move_ghost_mesh = Some(GpuMesh::new(&self.device, mesh));
        }
    }

    /// Clear the move-drag voxel ghost overlay.
    pub fn clear_move_ghost(&mut self) {
        self.move_ghost_mesh = None;
    }

    /// Draw the move-drag voxel ghost. Same depth/blend rules as
    /// `draw_brush_preview` — call after opaque geometry so the depth
    /// buffer already gates it.
    pub fn draw_move_ghost<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if let Some(ghost) = &self.move_ghost_mesh {
            render_pass.set_pipeline(&self.pipeline.transparent_pipeline);
            render_pass.set_bind_group(0, &self.pipeline.camera_bind_group, &[]);
            ghost.draw(render_pass);
        }
    }

    /// Replace the selection wireframe with one covering the closed
    /// AABB `[min, max]` in cells. The mesh expands to `max + 1` so it
    /// envelops the outer face of the corner cells.
    pub fn set_selection_mesh(&mut self, min: (i32, i32, i32), max: (i32, i32, i32)) {
        self.selection_mesh = Some(SelectionMesh::new(&self.device, min, max));
    }

    /// Clear the box-selection wireframe.
    pub fn clear_selection(&mut self) {
        self.selection_mesh = None;
    }

    /// Draw the box-selection wireframe (if any) using the line
    /// pipeline. Call after grid/axes so it draws on top.
    pub fn draw_selection<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if let Some(sel) = &self.selection_mesh {
            render_pass.set_pipeline(&self.line_pipeline.render_pipeline);
            render_pass.set_bind_group(0, &self.pipeline.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, sel.vertex_buffer.slice(..));
            render_pass.draw(0..sel.vertex_count, 0..1);
        }
    }

    /// Replace the socket gizmo overlay from a list of `(position,
    /// normal)` pairs. An empty list clears the slot.
    pub fn set_socket_mesh(&mut self, sockets: &[([f32; 3], [f32; 3])]) {
        self.socket_mesh = SocketMesh::new(&self.device, sockets);
    }

    /// Clear the socket gizmo overlay.
    pub fn clear_socket(&mut self) {
        self.socket_mesh = None;
    }

    /// Draw the socket gizmos (if any) through the line pipeline.
    /// Same depth rules as the selection wireframe — call alongside it.
    pub fn draw_socket<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if let Some(sockets) = &self.socket_mesh {
            render_pass.set_pipeline(&self.line_pipeline.render_pipeline);
            render_pass.set_bind_group(0, &self.pipeline.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, sockets.vertex_buffer.slice(..));
            render_pass.draw(0..sockets.vertex_count, 0..1);
        }
    }

    /// Draw grid in render pass
    pub fn draw_grid<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.line_pipeline.render_pipeline);
        render_pass.set_bind_group(0, &self.pipeline.camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.grid_mesh.vertex_buffer.slice(..));
        render_pass.draw(0..self.grid_mesh.vertex_count, 0..1);
    }

    /// Draw axes in render pass
    pub fn draw_axes<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.line_pipeline.render_pipeline);
        render_pass.set_bind_group(0, &self.pipeline.camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.axis_mesh.vertex_buffer.slice(..));
        render_pass.draw(0..self.axis_mesh.vertex_count, 0..1);
    }

    // There is deliberately no `Renderer::render()`: `App::render` owns
    // the pass order. A convenience method here could only draw the
    // chunk subset, then present a half-drawn frame.

    /// Get total triangle count
    pub fn total_triangles(&self) -> usize {
        self.chunk_meshes.values().map(|m| m.index_count / 3).sum()
    }
}
