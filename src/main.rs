//! Voxelith - Procedural-first voxel asset creation tool
//!
//! Main application entry point.

use std::path::PathBuf;
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
    io,
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

    /// Current project file path (None = unsaved)
    project_path: Option<PathBuf>,
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
            project_path: None,
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

        if self.ui.state.clear_all_requested {
            self.world.clear();
            self.editor.history.clear();
            if let Some(renderer) = &mut self.renderer {
                renderer.chunk_meshes.clear();
            }
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

        if self.ui.state.generate_sphere {
            self.world.clear();
            self.editor.history.clear();
            self.create_sphere((0, 10, 0), 6);
            self.rebuild_all_meshes();
        }

        if self.ui.state.generate_pyramid {
            self.world.clear();
            self.editor.history.clear();
            self.create_pyramid((0, 0, 0), 10);
            self.rebuild_all_meshes();
        }

        if self.ui.state.reset_camera_requested {
            if let Some(renderer) = &mut self.renderer {
                renderer.camera.target = glam::Vec3::ZERO;
                renderer.camera_controller.distance = 40.0;
                renderer.camera_controller.yaw = 0.0;
                renderer.camera_controller.pitch = 0.5;
            }
        }

        if let Some(view) = self.ui.state.camera_view {
            if let Some(renderer) = &mut self.renderer {
                use voxelith::ui::CameraView;
                match view {
                    CameraView::Top => {
                        renderer.camera_controller.pitch = 1.5;
                        renderer.camera_controller.yaw = 0.0;
                    }
                    CameraView::Front => {
                        renderer.camera_controller.pitch = 0.0;
                        renderer.camera_controller.yaw = 0.0;
                    }
                    CameraView::Side => {
                        renderer.camera_controller.pitch = 0.0;
                        renderer.camera_controller.yaw = std::f32::consts::FRAC_PI_2;
                    }
                }
            }
        }

        // File operations
        if self.ui.state.new_project_requested {
            self.new_project();
        }

        if self.ui.state.save_project_requested {
            self.save_project();
        }

        if self.ui.state.save_as_requested {
            self.save_project_as();
        }

        if self.ui.state.open_project_requested {
            self.open_project();
        }

        if self.ui.state.import_vox_requested {
            self.import_vox();
        }

        if self.ui.state.export_vox_requested {
            self.export_vox();
        }

        self.ui.clear_flags();
    }

    /// Create a new empty project
    fn new_project(&mut self) {
        self.world.clear();
        self.editor.history.clear();
        self.project_path = None;
        if let Some(renderer) = &mut self.renderer {
            renderer.chunk_meshes.clear();
        }
        self.ui.set_status("New project created");
    }

    /// Save the current project
    fn save_project(&mut self) {
        if let Some(path) = &self.project_path.clone() {
            self.do_save_project(path.clone());
        } else {
            self.save_project_as();
        }
    }

    /// Save the project to a new file
    fn save_project_as(&mut self) {
        let dialog = rfd::FileDialog::new()
            .add_filter("Voxelith Project", &["vxlt"])
            .set_title("Save Project As");

        if let Some(path) = dialog.save_file() {
            self.do_save_project(path);
        }
    }

    /// Actually save the project to a path
    fn do_save_project(&mut self, path: PathBuf) {
        match io::save_world(&self.world, &path) {
            Ok(_) => {
                self.project_path = Some(path.clone());
                let filename = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("project");
                self.ui.set_status(format!("Saved: {}", filename));
            }
            Err(e) => {
                log::error!("Failed to save project: {}", e);
                self.ui.set_status(format!("Save failed: {}", e));
            }
        }
    }

    /// Open a project file
    fn open_project(&mut self) {
        let dialog = rfd::FileDialog::new()
            .add_filter("Voxelith Project", &["vxlt"])
            .add_filter("All Files", &["*"])
            .set_title("Open Project");

        if let Some(path) = dialog.pick_file() {
            match io::load_world(&path) {
                Ok(world) => {
                    self.world = world;
                    self.editor.history.clear();
                    self.project_path = Some(path.clone());
                    if let Some(renderer) = &mut self.renderer {
                        renderer.chunk_meshes.clear();
                    }
                    self.rebuild_all_meshes();
                    let filename = path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("project");
                    self.ui.set_status(format!("Opened: {}", filename));
                }
                Err(e) => {
                    log::error!("Failed to open project: {}", e);
                    self.ui.set_status(format!("Open failed: {}", e));
                }
            }
        }
    }

    /// Import a VOX file
    fn import_vox(&mut self) {
        let dialog = rfd::FileDialog::new()
            .add_filter("MagicaVoxel", &["vox"])
            .set_title("Import MagicaVoxel File");

        if let Some(path) = dialog.pick_file() {
            match std::fs::File::open(&path) {
                Ok(mut file) => {
                    match io::import_vox(&mut file) {
                        Ok(world) => {
                            self.world = world;
                            self.editor.history.clear();
                            if let Some(renderer) = &mut self.renderer {
                                renderer.chunk_meshes.clear();
                            }
                            self.rebuild_all_meshes();
                            let filename = path.file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("file");
                            self.ui.set_status(format!("Imported: {}", filename));
                        }
                        Err(e) => {
                            log::error!("Failed to import VOX: {}", e);
                            self.ui.set_status(format!("Import failed: {}", e));
                        }
                    }
                }
                Err(e) => {
                    log::error!("Failed to open file: {}", e);
                    self.ui.set_status(format!("Open failed: {}", e));
                }
            }
        }
    }

    /// Export to VOX format
    fn export_vox(&mut self) {
        let dialog = rfd::FileDialog::new()
            .add_filter("MagicaVoxel", &["vox"])
            .set_title("Export as MagicaVoxel");

        if let Some(path) = dialog.save_file() {
            match std::fs::File::create(&path) {
                Ok(mut file) => {
                    match io::export_vox(&self.world, &mut file) {
                        Ok(_) => {
                            let filename = path.file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("file");
                            self.ui.set_status(format!("Exported: {}", filename));
                        }
                        Err(e) => {
                            log::error!("Failed to export VOX: {}", e);
                            self.ui.set_status(format!("Export failed: {}", e));
                        }
                    }
                }
                Err(e) => {
                    log::error!("Failed to create file: {}", e);
                    self.ui.set_status(format!("Create file failed: {}", e));
                }
            }
        }
    }

    /// Create a voxel sphere
    fn create_sphere(&mut self, center: (i32, i32, i32), radius: i32) {
        let radius_sq = (radius as f32).powi(2);
        for z in -radius..=radius {
            for y in -radius..=radius {
                for x in -radius..=radius {
                    let dist_sq = (x * x + y * y + z * z) as f32;
                    if dist_sq <= radius_sq {
                        // Color based on distance from center
                        let t = (dist_sq.sqrt() / radius as f32 * 255.0) as u8;
                        let voxel = voxelith::core::Voxel::from_rgb(255 - t, t, 128);
                        self.world.set_voxel(
                            center.0 + x,
                            center.1 + y,
                            center.2 + z,
                            voxel,
                        );
                    }
                }
            }
        }
    }

    /// Create a voxel pyramid
    fn create_pyramid(&mut self, base_center: (i32, i32, i32), height: i32) {
        for y in 0..height {
            let size = height - y;
            for z in -size..=size {
                for x in -size..=size {
                    // Color gradient based on height
                    let t = (y as f32 / height as f32 * 255.0) as u8;
                    let voxel = voxelith::core::Voxel::from_rgb(194 - t / 2, 178 - t / 2, 128 + t / 2);
                    self.world.set_voxel(
                        base_center.0 + x,
                        base_center.1 + y,
                        base_center.2 + z,
                        voxel,
                    );
                }
            }
        }
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
            KeyCode::KeyS if self.modifiers.control_key() => {
                if self.modifiers.shift_key() {
                    self.save_project_as();
                } else {
                    self.save_project();
                }
            }
            KeyCode::KeyO if self.modifiers.control_key() => {
                self.open_project();
            }
            KeyCode::KeyN if self.modifiers.control_key() => {
                self.new_project();
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

        // Get viewport settings before borrowing renderer
        let show_grid = self.ui.viewport.show_grid;
        let show_axes = self.ui.viewport.show_axes;

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

            // Draw grid first (behind everything)
            if show_grid {
                renderer.draw_grid(&mut render_pass);
            }

            // Draw axes
            if show_axes {
                renderer.draw_axes(&mut render_pass);
            }

            // Draw voxel meshes
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
