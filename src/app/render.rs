//! Per-frame render pipeline.
//!
//! The flow is: drive the egui pass → drain UI actions → grid/axes/voxel
//! main pass → egui overlay pass → submit. Wireframe replaces the voxel
//! pipeline when enabled (and supported by the GPU).

use super::App;

impl App {
    /// Render a single frame.
    pub(super) fn render_frame(&mut self, dt: f32) {
        let window = self.window.as_ref().unwrap().clone();
        let egui_state = self.egui_state.as_mut().unwrap();

        // egui frame
        let raw_input = egui_state.take_egui_input(&window);
        let egui_ctx = egui_state.egui_ctx().clone();
        egui_ctx.begin_pass(raw_input);

        let stats = self.calculate_stats();
        // Mirror clipboard presence into Ui so Tools-panel buttons can
        // gray out Paste when there's nothing to paste. Cheap (bool
        // copy) and avoids leaking App::clipboard across the UI
        // boundary.
        self.ui.has_clipboard = self.clipboard.is_some();
        // Same pattern for AI panel: mirror state owned by App so the
        // panel reads them off `Ui` without needing a borrow back.
        self.ui.ai_job = self.ai_job.clone();
        self.ui.ai_has_key = self.ai_has_key;
        self.ui.show(&egui_ctx, &stats, &mut self.editor);

        let full_output = egui_ctx.end_pass();

        // Drain UI actions before touching wgpu state
        self.handle_ui_actions();

        let egui_state = self.egui_state.as_mut().unwrap();
        egui_state.handle_platform_output(&window, full_output.platform_output);

        // Snapshot viewport settings before borrowing renderer mutably
        let show_grid = self.ui.viewport.show_grid;
        let show_axes = self.ui.viewport.show_axes;
        let grid_size = self.ui.viewport.grid_size;
        let grid_spacing = self.ui.viewport.grid_spacing;
        let wireframe_mode = self.ui.viewport.wireframe_mode;

        let renderer = self.renderer.as_mut().unwrap();

        // Refresh grid mesh if settings changed
        if grid_size != self.last_grid_size
            || (grid_spacing - self.last_grid_spacing).abs() > 0.01
        {
            renderer.update_grid(grid_size, grid_spacing);
            self.last_grid_size = grid_size;
            self.last_grid_spacing = grid_spacing;
        }
        let egui_renderer = self.egui_renderer.as_mut().unwrap();

        // Update camera (WASD movement etc.)
        renderer.camera_controller.update(&mut renderer.camera, dt);

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

        renderer
            .pipeline
            .update_camera(&renderer.queue, &renderer.camera);

        let mut encoder = renderer
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Main pass: grid → axes → voxels
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

            if show_grid {
                renderer.draw_grid(&mut render_pass);
            }
            if show_axes {
                renderer.draw_axes(&mut render_pass);
            }

            let use_wireframe =
                wireframe_mode && renderer.pipeline.wireframe_pipeline.is_some();
            if use_wireframe {
                render_pass
                    .set_pipeline(renderer.pipeline.wireframe_pipeline.as_ref().unwrap());
            } else {
                render_pass.set_pipeline(&renderer.pipeline.render_pipeline);
            }
            render_pass.set_bind_group(0, &renderer.pipeline.camera_bind_group, &[]);

            for mesh in renderer.chunk_meshes.values() {
                mesh.draw(&mut render_pass);
            }

            // Box-selection wireframe (yellow AABB). Drawn after
            // opaque chunks but before the translucent overlays so
            // brush hover hints stay readable on top of the selection.
            // Uses `LinePipeline` (depth-test on, depth-write off) —
            // the wireframe is correctly occluded by intervening
            // voxels, matching how Goxel renders its selection.
            renderer.draw_selection(&mut render_pass);

            // Procgen preview overlay (alpha-blended). Drawn after
            // opaque chunks so the depth buffer already correctly
            // gates it; the transparent pipeline reads but does not
            // write depth so multiple translucent fragments composite.
            renderer.draw_preview(&mut render_pass);

            // Brush hover overlay — shows where the next click will
            // land. Same transparent-pipeline rules.
            renderer.draw_brush_preview(&mut render_pass);

            // Move-drag voxel ghost — the selection's content trailing
            // the cursor while it's relocated. Same transparent rules;
            // during a move drag the brush hover slot above is empty
            // (Select tool), so the two never fight for the frame.
            renderer.draw_move_ghost(&mut render_pass);
        }

        // egui overlay pass
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [renderer.config.width, renderer.config.height],
            pixels_per_point: window.scale_factor() as f32,
        };

        let paint_jobs =
            egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

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

        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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

            egui_renderer.render(
                &mut render_pass.forget_lifetime(),
                &paint_jobs,
                &screen_descriptor,
            );
        }

        for id in full_output.textures_delta.free {
            egui_renderer.free_texture(&id);
        }

        renderer.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}
