//! Rendering system using wgpu.
//!
//! This module handles all GPU-related operations:
//! - Device initialization
//! - Shader compilation
//! - Render pipeline management
//! - Mesh rendering

mod camera;
mod pipeline;
mod gpu_mesh;
mod grid;

pub use camera::{Camera, CameraController, CameraUniform};
pub use pipeline::RenderPipeline;
pub use gpu_mesh::GpuMesh;
pub use grid::{AxisMesh, GridMesh, LinePipeline, LineVertex};

use crate::mesh::ChunkMesh;
use crate::core::ChunkPos;
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

        // Request device
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Voxelith Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await?;

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

        // Create render pipeline
        let pipeline = RenderPipeline::new(&device, surface_format);

        // Create line pipeline (uses same camera bind group layout)
        let line_pipeline = LinePipeline::new(&device, surface_format, &pipeline.camera_bind_group_layout);

        // Create camera
        let camera = Camera::new(
            glam::Vec3::new(0.0, 20.0, 40.0),
            glam::Vec3::ZERO,
            size.width as f32 / size.height as f32,
        );
        let camera_controller = CameraController::new(0.5, 0.003);

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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
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

    /// Remove a chunk mesh
    pub fn remove_mesh(&mut self, chunk_pos: ChunkPos) {
        self.chunk_meshes.remove(&chunk_pos);
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

    /// Render a frame
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Update camera uniform
        self.pipeline.update_camera(&self.queue, &self.camera);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.15,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.pipeline.render_pipeline);
            render_pass.set_bind_group(0, &self.pipeline.camera_bind_group, &[]);

            // Render all chunk meshes
            for mesh in self.chunk_meshes.values() {
                mesh.draw(&mut render_pass);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    /// Get total triangle count
    pub fn total_triangles(&self) -> usize {
        self.chunk_meshes.values().map(|m| m.index_count / 3).sum()
    }
}
