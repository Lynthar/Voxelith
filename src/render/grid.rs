//! Grid and axis rendering using line primitives.
//!
//! Renders a ground grid and coordinate axes for visual reference.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Line vertex format (position + color)
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct LineVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

impl LineVertex {
    pub fn new(position: [f32; 3], color: [f32; 4]) -> Self {
        Self { position, color }
    }

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

/// Grid mesh for rendering
pub struct GridMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub vertex_count: u32,
}

/// Line vertices for a square grid on the XZ plane, spanning
/// `±size * spacing`. The extent must match the placement range or the
/// outer lines never cross. Split out to be testable without a GPU.
fn grid_vertices(size: i32, spacing: f32) -> Vec<LineVertex> {
    let mut vertices = Vec::new();
    let half = size as f32 * spacing;
    let grid_color = [0.3, 0.3, 0.3, 0.6];
    let origin_color = [0.5, 0.5, 0.5, 0.8];

    // Grid lines along X axis
    for i in -size..=size {
        let z = i as f32 * spacing;
        let color = if i == 0 { origin_color } else { grid_color };
        vertices.push(LineVertex::new([-half, 0.0, z], color));
        vertices.push(LineVertex::new([half, 0.0, z], color));
    }

    // Grid lines along Z axis
    for i in -size..=size {
        let x = i as f32 * spacing;
        let color = if i == 0 { origin_color } else { grid_color };
        vertices.push(LineVertex::new([x, 0.0, -half], color));
        vertices.push(LineVertex::new([x, 0.0, half], color));
    }

    vertices
}

impl GridMesh {
    /// Create a grid mesh on the XZ plane at y=0
    pub fn new(device: &wgpu::Device, size: i32, spacing: f32) -> Self {
        let vertices = grid_vertices(size, spacing);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            vertex_buffer,
            vertex_count: vertices.len() as u32,
        }
    }
}

/// Coordinate axes mesh
pub struct AxisMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub vertex_count: u32,
}

impl AxisMesh {
    /// Create coordinate axes at origin
    pub fn new(device: &wgpu::Device, length: f32) -> Self {
        let vertices = vec![
            // X axis (red)
            LineVertex::new([0.0, 0.0, 0.0], [1.0, 0.2, 0.2, 1.0]),
            LineVertex::new([length, 0.0, 0.0], [1.0, 0.2, 0.2, 1.0]),
            // Y axis (green)
            LineVertex::new([0.0, 0.0, 0.0], [0.2, 1.0, 0.2, 1.0]),
            LineVertex::new([0.0, length, 0.0], [0.2, 1.0, 0.2, 1.0]),
            // Z axis (blue)
            LineVertex::new([0.0, 0.0, 0.0], [0.2, 0.2, 1.0, 1.0]),
            LineVertex::new([0.0, 0.0, length], [0.2, 0.2, 1.0, 1.0]),
        ];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Axis Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            vertex_buffer,
            vertex_count: vertices.len() as u32,
        }
    }
}

/// Line rendering pipeline
pub struct LinePipeline {
    pub render_pipeline: wgpu::RenderPipeline,
}

impl LinePipeline {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Line Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/line.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Line Pipeline Layout"),
            bind_group_layouts: &[camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Line Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[LineVertex::layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false, // Don't write depth for lines
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Self { render_pipeline }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lines must be placed and drawn over the same range, or the outer
    /// ring degenerates into non-crossing stripes.
    #[test]
    fn grid_lines_span_the_full_extent() {
        let size = 20;
        let spacing = 1.0;
        let extent = size as f32 * spacing;
        let verts = grid_vertices(size, spacing);

        // 2 directions × (2·size + 1) lines × 2 endpoints.
        assert_eq!(verts.len(), 2 * 2 * (2 * size as usize + 1));

        let (mut min_x, mut max_x) = (f32::MAX, f32::MIN);
        let (mut min_z, mut max_z) = (f32::MAX, f32::MIN);
        for v in &verts {
            min_x = min_x.min(v.position[0]);
            max_x = max_x.max(v.position[0]);
            min_z = min_z.min(v.position[2]);
            max_z = max_z.max(v.position[2]);
            assert_eq!(v.position[1], 0.0, "grid must be flat at y = 0");
        }
        assert_eq!((min_x, max_x), (-extent, extent));
        assert_eq!((min_z, max_z), (-extent, extent));
    }

    /// Every individual line must run the grid's full width, so lines
    /// in the two directions intersect everywhere rather than only in
    /// the middle.
    #[test]
    fn every_line_crosses_the_whole_grid() {
        let size = 20;
        let spacing = 1.0;
        let width = 2.0 * size as f32 * spacing;
        for pair in grid_vertices(size, spacing).chunks_exact(2) {
            let (a, b) = (pair[0].position, pair[1].position);
            let len = (b[0] - a[0]).abs().max((b[2] - a[2]).abs());
            assert_eq!(len, width, "line {:?}..{:?} is short", a, b);
        }
    }
}
