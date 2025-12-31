//! Voxelith - Procedural-first voxel asset creation tool
//!
//! Main application entry point.

use std::sync::Arc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, ModifiersState, PhysicalKey},
    window::{Window, WindowId},
};

use voxelith::{
    core::{ChunkPos, World},
    editor::{
        eyedrop, flood_fill, BrushTool, Editor, EditorTool, Ray, Tool, ToolContext, VoxelRaycast,
    },
    mesh::{Mesher, NaiveMesher},
    render::Renderer,
    ui::{RenderStats, Ui},
};

/// Main application state
struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    egui_state: Option<egui_winit::State>,
    egui_renderer: Option<egui_wgpu::Renderer>,

    world: World,
    mesher: NaiveMesher,
    editor: Editor,
    ui: Ui,

    last_frame: Instant,
    frame_times: Vec<f32>,

    cursor_captured: bool,
    cursor_pos: (f32, f32),
    modifiers: ModifiersState,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            egui_state: None,
            egui_renderer: None,
            world: World::new(),
            mesher: NaiveMesher::new(),
            editor: Editor::new(),
            ui: Ui::new(),
            last_frame: Instant::now(),
            frame_times: Vec::with_capacity(60),
            cursor_captured: false,
            cursor_pos: (0.0, 0.0),
            modifiers: ModifiersState::empty(),
        }
    }

    /// Initialize the application with a window
    fn init(&mut self, window: Window) {
        let window = Arc::new(window);
        self.window = Some(window.clone());

        // Initialize renderer
        let renderer = pollster::block_on(Renderer::new(window.clone()))
            .expect("Failed to create renderer");

        // Initialize egui
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx,
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        let egui_renderer = egui_wgpu::Renderer::new(
            &renderer.device,
            renderer.config.format,
            Some(wgpu::TextureFormat::Depth32Float),
            1,
            false,
        );

        self.renderer = Some(renderer);
        self.egui_state = Some(egui_state);
        self.egui_renderer = Some(egui_renderer);

        // Create initial test content
        self.create_initial_scene();
    }

    /// Create initial test scene
    fn create_initial_scene(&mut self) {
        // Create a colorful test cube
        self.world.create_test_cube((0, 8, 0), 4);

        // Create a ground plane
        self.world.create_test_ground(20, 2);

        // Generate meshes for all chunks
        self.rebuild_all_meshes();
    }

    /// Rebuild meshes for all dirty chunks
    fn rebuild_all_meshes(&mut self) {
        if let Some(renderer) = &mut self.renderer {
            let dirty_chunks: Vec<ChunkPos> = self.world.dirty_chunks();

            for chunk_pos in dirty_chunks {
                if let Some(chunk) = self.world.get_chunk(chunk_pos) {
                    let mesh = self.mesher.generate(&chunk.read(), chunk_pos);
                    renderer.upload_mesh(&mesh);
                }
            }

            self.world.clear_dirty_flags();
        }
    }

    /// Calculate render stats
    fn calculate_stats(&self) -> RenderStats {
        let avg_frame_time = if self.frame_times.is_empty() {
            16.67
        } else {
            self.frame_times.iter().sum::<f32>() / self.frame_times.len() as f32
        };

        let renderer = self.renderer.as_ref().unwrap();
        let camera_pos = renderer.camera.position;

        RenderStats {
            fps: 1000.0 / avg_frame_time,
            frame_time_ms: avg_frame_time,
            triangles: renderer.total_triangles(),
            chunks: self.world.chunk_count(),
            camera_pos: (camera_pos.x, camera_pos.y, camera_pos.z),
        }
    }

    /// Handle UI actions
    fn handle_ui_actions(&mut self) {
        if self.ui.state.exit_requested {
            std::process::exit(0);
        }

        if self.ui.state.undo_requested {
            self.editor.undo(&mut self.world);
        }

        if self.ui.state.redo_requested {
            self.editor.redo(&mut self.world);
        }

        if self.ui.state.generate_test_cube {
            self.world.clear();
            self.editor.history.clear();
            self.world.create_test_cube((0, 8, 0), 4);
            self.rebuild_all_meshes();
        }

        if self.ui.state.generate_ground {
            self.world.clear();
            self.editor.history.clear();
            self.world.create_test_ground(20, 2);
            self.rebuild_all_meshes();
        }

        self.ui.clear_flags();
    }

    /// Update raycast for hovered voxel
    fn update_raycast(&mut self) {
        if let Some(renderer) = &self.renderer {
            let window = self.window.as_ref().unwrap();
            let size = window.inner_size();

            let view_proj = renderer.camera.view_projection_matrix();
            let view_proj_inv = view_proj.inverse();

            let ray = Ray::from_screen(
                self.cursor_pos,
                (size.width as f32, size.height as f32),
                view_proj_inv,
            );

            self.editor.hovered_voxel = VoxelRaycast::cast(&ray, &self.world, 100.0);
        }
    }

    /// Apply the current tool at the hovered location
    fn apply_tool(&mut self) {
        if let Some(hit) = self.editor.hovered_voxel {
            match self.editor.current_tool {
                Tool::Place | Tool::Remove | Tool::Paint => {
                    let brush = BrushTool::new(self.editor.current_tool);
                    let mut ctx = ToolContext {
                        world: &mut self.world,
                        history: &mut self.editor.history,
                        brush_color: self.editor.brush_color,
                        brush_size: self.editor.brush_size,
                    };
                    brush.apply(&mut ctx, &hit);
                }
                Tool::Eyedropper => {
                    if let Some(color) = eyedrop(&self.world, &hit) {
                        self.editor.brush_color = color;
                    }
                }
                Tool::Fill => {
                    flood_fill(
                        &mut self.world,
                        &mut self.editor.history,
                        hit.voxel_pos,
                        self.editor.brush_color,
                        10000, // Max voxels to fill
                    );
                }
            }
        }
    }

    /// Handle keyboard shortcuts for tools
    fn handle_tool_shortcut(&mut self, key: KeyCode) {
        match key {
            KeyCode::Digit1 => self.editor.current_tool = Tool::Place,
            KeyCode::Digit2 => self.editor.current_tool = Tool::Remove,
            KeyCode::Digit3 => self.editor.current_tool = Tool::Paint,
            KeyCode::Digit4 => self.editor.current_tool = Tool::Eyedropper,
            KeyCode::Digit5 => self.editor.current_tool = Tool::Fill,
            KeyCode::KeyZ if self.modifiers.control_key() => {
                if self.modifiers.shift_key() {
                    self.editor.redo(&mut self.world);
                } else {
                    self.editor.undo(&mut self.world);
                }
            }
            KeyCode::KeyY if self.modifiers.control_key() => {
                self.editor.redo(&mut self.world);
            }
            _ => {}
        }
    }

    /// Render a frame
    fn render_frame(&mut self, dt: f32) {
        let window = self.window.as_ref().unwrap().clone();
        let egui_state = self.egui_state.as_mut().unwrap();

        // Start egui frame
        let raw_input = egui_state.take_egui_input(&window);
        let egui_ctx = egui_state.egui_ctx().clone();
        egui_ctx.begin_pass(raw_input);

        // Render UI
        let stats = self.calculate_stats();
        self.ui.show(&egui_ctx, &stats, &mut self.editor);

        // End egui frame
        let full_output = egui_ctx.end_pass();

        // Handle UI actions after egui pass
        self.handle_ui_actions();

        // Process egui platform output
        let egui_state = self.egui_state.as_mut().unwrap();
        egui_state.handle_platform_output(&window, full_output.platform_output);

        // Now do the actual rendering
        let renderer = self.renderer.as_mut().unwrap();
        let egui_renderer = self.egui_renderer.as_mut().unwrap();

        // Update camera
        renderer.camera_controller.update(&mut renderer.camera, dt);

        // Get surface texture
        let output = match renderer.surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                let size = window.inner_size();
                renderer.resize(size);
                return;
            }
            Err(e) => {
                log::error!("Surface error: {:?}", e);
                return;
            }
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Update camera uniform
        renderer.pipeline.update_camera(&renderer.queue, &renderer.camera);

        let mut encoder = renderer
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Main render pass
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
                    view: &renderer.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&renderer.pipeline.render_pipeline);
            render_pass.set_bind_group(0, &renderer.pipeline.camera_bind_group, &[]);

            for mesh in renderer.chunk_meshes.values() {
                mesh.draw(&mut render_pass);
            }
        }

        // Egui render pass
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [renderer.config.width, renderer.config.height],
            pixels_per_point: window.scale_factor() as f32,
        };

        let paint_jobs = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

        for (id, delta) in full_output.textures_delta.set {
            egui_renderer.update_texture(&renderer.device, &renderer.queue, id, &delta);
        }

        egui_renderer.update_buffers(
            &renderer.device,
            &renderer.queue,
            &mut encoder,
            &paint_jobs,
            &screen_descriptor,
        );

        // Egui render pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // SAFETY: The render_pass reference is valid for the duration of this block.
            // egui-wgpu 0.29 requires 'static but we ensure the pass lives long enough.
            let render_pass_static: &mut wgpu::RenderPass<'static> =
                unsafe { std::mem::transmute(&mut render_pass) };
            egui_renderer.render(render_pass_static, &paint_jobs, &screen_descriptor);
        }

        // Free textures
        for id in full_output.textures_delta.free {
            egui_renderer.free_texture(&id);
        }

        renderer.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attrs = Window::default_attributes()
                .with_title("Voxelith")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

            let window = event_loop.create_window(window_attrs).unwrap();
            self.init(window);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // Handle egui events first
        let egui_consumed = {
            let window = self.window.as_ref().unwrap();
            let egui_state = self.egui_state.as_mut().unwrap();
            egui_state.on_window_event(window, &event).consumed
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size);
                }
            }

            WindowEvent::ModifiersChanged(new_modifiers) => {
                self.modifiers = new_modifiers.state();
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if !egui_consumed {
                    if let PhysicalKey::Code(key) = event.physical_key {
                        if let Some(renderer) = &mut self.renderer {
                            renderer.camera_controller.process_keyboard(key, event.state);
                        }

                        // Tool shortcuts (only on press)
                        if event.state.is_pressed() {
                            self.handle_tool_shortcut(key);
                        }

                        // Escape to release cursor
                        if key == KeyCode::Escape && event.state.is_pressed() {
                            self.cursor_captured = false;
                            if let Some(window) = &self.window {
                                window.set_cursor_visible(true);
                            }
                        }
                    }
                }
            }

            WindowEvent::MouseInput { button, state, .. } => {
                if !egui_consumed {
                    if let Some(renderer) = &mut self.renderer {
                        renderer.camera_controller.process_mouse_button(button, state);
                    }

                    // Left click to apply tool
                    if button == winit::event::MouseButton::Left && state == ElementState::Pressed {
                        self.apply_tool();
                    }

                    // Middle click to capture cursor for camera control
                    if button == winit::event::MouseButton::Middle && state.is_pressed() {
                        self.cursor_captured = true;
                        if let Some(window) = &self.window {
                            window.set_cursor_visible(false);
                        }
                    }
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if !egui_consumed {
                    if let Some(renderer) = &mut self.renderer {
                        renderer.camera_controller.process_scroll(delta, &mut renderer.camera);
                    }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_pos = (position.x as f32, position.y as f32);

                if !egui_consumed {
                    // Update raycast for hovered voxel
                    self.update_raycast();

                    if self.cursor_captured {
                        if let Some(renderer) = &mut self.renderer {
                            renderer.camera_controller.process_mouse_motion(
                                position.x as f32,
                                position.y as f32,
                                &mut renderer.camera,
                            );
                        }
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                // Calculate delta time
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32();
                self.last_frame = now;

                // Track frame times for FPS display
                self.frame_times.push(dt * 1000.0);
                if self.frame_times.len() > 60 {
                    self.frame_times.remove(0);
                }

                // Rebuild any dirty meshes
                self.rebuild_all_meshes();

                // Render the frame
                self.render_frame(dt);

                // Request next frame
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        // Handle raw mouse motion for smoother camera control
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.cursor_captured {
                if let Some(renderer) = &mut self.renderer {
                    // Use delta directly for smoother motion
                    renderer.camera_controller.yaw -= delta.0 as f32 * 0.003;
                    renderer.camera_controller.pitch -= delta.1 as f32 * 0.003;
                    renderer.camera_controller.pitch = renderer.camera_controller.pitch.clamp(-1.5, 1.5);

                    // Update camera position
                    let distance = renderer.camera_controller.distance;
                    let yaw = renderer.camera_controller.yaw;
                    let pitch = renderer.camera_controller.pitch;

                    let x = distance * yaw.cos() * pitch.cos();
                    let y = distance * pitch.sin();
                    let z = distance * yaw.sin() * pitch.cos();

                    renderer.camera.position = renderer.camera.target + glam::Vec3::new(x, y, z);
                }
            }
        }
    }
}

fn main() {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    log::info!("Starting Voxelith...");

    // Create event loop and run
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
