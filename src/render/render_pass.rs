use super::*;

impl State {
    /// Updates the GPU buffer cache for chunks that have changed
    pub(super) fn update_chunk_buffers(&mut self) {
        // Collect chunk positions that need buffer updates
        let chunks_to_update: Vec<((i32, i32), u32, bool, bool, bool)> = self.world.chunks.iter()
            .filter_map(|(&pos, chunk)| {
                let cached = self.chunk_buffers.get(&pos);
                let needs_update = match cached {
                    Some(cb) => cb.mesh_version != chunk.mesh_version,
                    None => true,
                };
                if needs_update && !chunk.vertices.is_empty() {
                    Some((pos, chunk.mesh_version, !chunk.vertices.is_empty(), !chunk.water_vertices.is_empty(), !chunk.transparent_vertices.is_empty()))
                } else if needs_update && chunk.vertices.is_empty() {
                    // Remove empty chunks from cache
                    Some((pos, chunk.mesh_version, false, false, false))
                } else {
                    None
                }
            })
            .collect();

        // Update buffers for chunks that need it
        for (pos, mesh_version, has_geometry, has_water, has_transparent) in chunks_to_update {
            if !has_geometry {
                // Remove from cache if chunk has no geometry
                self.chunk_buffers.remove(&pos);
                continue;
            }

            let chunk = match self.world.chunks.get(&pos) {
                Some(c) => c,
                None => continue,
            };

            let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Chunk Vertex Buffer"),
                contents: bytemuck::cast_slice(&chunk.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

            let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Chunk Index Buffer"),
                contents: bytemuck::cast_slice(&chunk.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            let (water_vertex_buffer, water_index_buffer, water_index_count) = if has_water {
                let wvb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Water Vertex Buffer"),
                    contents: bytemuck::cast_slice(&chunk.water_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let wib = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Water Index Buffer"),
                    contents: bytemuck::cast_slice(&chunk.water_indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
                (Some(wvb), Some(wib), chunk.water_indices.len() as u32)
            } else {
                (None, None, 0)
            };

            let (transparent_vertex_buffer, transparent_index_buffer, transparent_index_count) = if has_transparent {
                let tvb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Transparent Vertex Buffer"),
                    contents: bytemuck::cast_slice(&chunk.transparent_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let tib = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Transparent Index Buffer"),
                    contents: bytemuck::cast_slice(&chunk.transparent_indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
                (Some(tvb), Some(tib), chunk.transparent_indices.len() as u32)
            } else {
                (None, None, 0)
            };

            self.chunk_buffers.insert(pos, ChunkBuffers {
                vertex_buffer,
                index_buffer,
                index_count: chunk.indices.len() as u32,
                water_vertex_buffer,
                water_index_buffer,
                water_index_count,
                transparent_vertex_buffer,
                transparent_index_buffer,
                transparent_index_count,
                mesh_version,
            });
        }

        // Remove cached buffers for chunks that no longer exist
        let existing_chunks: std::collections::HashSet<(i32, i32)> =
            self.world.chunks.keys().cloned().collect();
        self.chunk_buffers.retain(|pos, _| existing_chunks.contains(pos));
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.rebuild_hud_vertices();
        self.rebuild_item_cube_vertices();

        let output = self.surface.get_current_texture()?;
        let swap_chain_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Always render world to scene_texture for motion blur post-processing.
        // Motion blur then writes to swap_chain (or post_process_texture if underwater).
        let world_render_target = &self.scene_texture_view;

        // --- SHADOW MAP PASS: Render depth from sun's perspective ---
        {
            let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shadow Map Pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_map_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            shadow_pass.set_pipeline(&self.shadow_map_pipeline);
            shadow_pass.set_bind_group(0, &self.shadow_map_camera_bind_group, &[]);
            shadow_pass.set_bind_group(1, &self.texture_atlas.bind_group, &[]);

            for (_, buffers) in self.chunk_buffers.iter() {
                shadow_pass.set_vertex_buffer(0, buffers.vertex_buffer.slice(..));
                shadow_pass.set_index_buffer(buffers.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                shadow_pass.draw_indexed(0..buffers.index_count, 0, 0..1);
            }
        }

        // --- PASS 0: RENDER SKY (background behind everything) ---
        if !self.frozen_stone_ceiling {
            let mut sky_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Sky Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: world_render_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            sky_pass.set_pipeline(&self.sky_pipeline);
            sky_pass.set_bind_group(0, &self.sky_bind_group, &[]);
            sky_pass.draw(0..3, 0..1); // Fullscreen triangle
        }

        // --- PASS 1: RENDER WORLD (Opaque chunks & enemies) ---
        {
            let clear_color = if self.frozen_stone_ceiling {
                wgpu::LoadOp::Clear(wgpu::Color { r: 0.00, g: 0.01, b: 0.02, a: 1.0 })
            } else {
                wgpu::LoadOp::Load // Sky already rendered
            };

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("World Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: world_render_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: clear_color,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.texture_atlas.bind_group, &[]);
            render_pass.set_bind_group(2, &self.fog_bind_group, &[]);
            render_pass.set_bind_group(3, &self.sun_bind_group, &[]);

            // Render chunks using cached GPU buffers with frustum culling
            for (&(cx, cz), buffers) in self.chunk_buffers.iter() {
                // Calculate chunk bounding box
                let min = Vector3::new(
                    (cx * CHUNK_SIZE as i32) as f32,
                    0.0,
                    (cz * CHUNK_SIZE as i32) as f32,
                );
                let max = Vector3::new(
                    min.x + CHUNK_SIZE as f32,
                    CHUNK_HEIGHT as f32,
                    min.z + CHUNK_SIZE as f32,
                );

                // Skip chunks outside the view frustum
                if !self.frustum.is_box_visible(min, max) {
                    continue;
                }

                render_pass.set_vertex_buffer(0, buffers.vertex_buffer.slice(..));
                render_pass.set_index_buffer(buffers.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..buffers.index_count, 0, 0..1);
            }

            // Render enemies
            for enemy in &self.enemy_manager.enemies {
                if !enemy.alive { continue; }
                let vertices = create_enemy_vertices(enemy);
                if vertices.is_empty() { continue; }

                let num_cubes = vertices.len() / 24;
                let indices = generate_enemy_indices(num_cubes);

                let vertex_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Enemy Vertex Buffer"),
                            contents: bytemuck::cast_slice(&vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                let index_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Enemy Index Buffer"),
                            contents: bytemuck::cast_slice(&indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });

                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
            }

            // Dropped items (mini-blocks, or flat panels for tuft/vine types)
            // (Old shadow pipeline removed — lighting now handled by dynamic sun)

            // Render dropped items (mini-blocks, or flat panels for tuft/vine types)
            for item in &self.dropped_item_manager.items {
                let (vertices, item_indices): (Vec<Vertex>, Vec<u16>) =
                    if item.block_type.is_flat_item() {
                        create_flat_item_vertices(
                            item.position,
                            item.block_type,
                            item.get_size(),
                            1.0,
                        )
                    } else {
                        (
                            create_scaled_cube_vertices(
                                item.position,
                                item.block_type,
                                item.get_size(),
                                1.0,
                            ),
                            CUBE_INDICES.to_vec(),
                        )
                    };

                let vertex_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Dropped Item Vertex Buffer"),
                            contents: bytemuck::cast_slice(&vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                let index_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Dropped Item Index Buffer"),
                            contents: bytemuck::cast_slice(&item_indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });

                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..item_indices.len() as u32, 0, 0..1);
            }

            // Render particles
            for particle in &self.particle_manager.particles {
                let vertices = create_particle_vertices(
                    particle.position,
                    particle.color,
                    particle.size,
                    particle.get_alpha(),
                );

                let vertex_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Particle Vertex Buffer"),
                            contents: bytemuck::cast_slice(&vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                let index_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Particle Index Buffer"),
                            contents: bytemuck::cast_slice(CUBE_INDICES),
                            usage: wgpu::BufferUsages::INDEX,
                        });

                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..CUBE_INDICES.len() as u32, 0, 0..1);
            }

            // Render birds
            for bird in &self.bird_manager.birds {
                let vertices = create_bird_vertices(bird);
                if vertices.is_empty() {
                    continue;
                }

                // Calculate number of cubes (each cube has 24 vertices)
                let num_cubes = vertices.len() / 24;
                let indices = generate_bird_indices(num_cubes);

                let vertex_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Bird Vertex Buffer"),
                            contents: bytemuck::cast_slice(&vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                let index_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Bird Index Buffer"),
                            contents: bytemuck::cast_slice(&indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });

                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
            }

            // Render fish
            for fish in &self.fish_manager.fish {
                let vertices = create_fish_vertices(fish);
                if vertices.is_empty() {
                    continue;
                }

                // Calculate number of cubes (each cube has 24 vertices)
                let num_cubes = vertices.len() / 24;
                let indices = generate_fish_indices(num_cubes);

                let vertex_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Fish Vertex Buffer"),
                            contents: bytemuck::cast_slice(&vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                let index_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Fish Index Buffer"),
                            contents: bytemuck::cast_slice(&indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });

                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
            }
        }

        // Copy depth buffer to the copy texture for water shader to sample
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: &self.depth_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &self.depth_copy_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
        );

        // --- PASS 2: RENDER CLOUDS (before water so they appear behind transparent water) ---
        if self.cloud_index_count > 0 && !self.frozen_stone_ceiling {
            let mut cloud_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Cloud Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: world_render_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Keep existing color
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Keep existing depth
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            cloud_pass.set_pipeline(&self.cloud_pipeline);
            cloud_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            cloud_pass.set_bind_group(1, &self.fog_bind_group, &[]);
            cloud_pass.set_bind_group(2, &self.cloud_drift_bind_group, &[]);
            cloud_pass.set_vertex_buffer(0, self.cloud_vertex_buffer.slice(..));
            cloud_pass.set_index_buffer(self.cloud_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            cloud_pass.draw_indexed(0..self.cloud_index_count, 0, 0..1);
        }

        // --- PASS 3: RENDER WATER (Transparent) ---
        {
            let mut water_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Water Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: world_render_target, // Target same buffer as world
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Keep existing color
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Keep existing depth
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            water_pass.set_pipeline(&self.water_pipeline);
            water_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            water_pass.set_bind_group(1, &self.water_bind_group, &[]);
            water_pass.set_bind_group(2, &self.fog_bind_group, &[]);
            water_pass.set_bind_group(3, &self.sun_bind_group, &[]);

            // Collect water chunks, sort back-to-front for correct alpha blending:
            // far water renders first, then near water blends on top.
            let cam_x = self.camera.position.x;
            let cam_z = self.camera.position.z;
            let half_chunk = CHUNK_SIZE as f32 * 0.5;

            let mut water_chunks: Vec<(&(i32, i32), &ChunkBuffers)> = self.chunk_buffers.iter()
                .filter(|(_, buffers)| {
                    buffers.water_vertex_buffer.is_some()
                        && buffers.water_index_buffer.is_some()
                        && buffers.water_index_count > 0
                })
                .collect();

            water_chunks.sort_by(|a, b| {
                let (ax, az) = *a.0;
                let (bx, bz) = *b.0;
                let da = (ax as f32 * CHUNK_SIZE as f32 + half_chunk - cam_x).powi(2)
                       + (az as f32 * CHUNK_SIZE as f32 + half_chunk - cam_z).powi(2);
                let db = (bx as f32 * CHUNK_SIZE as f32 + half_chunk - cam_x).powi(2)
                       + (bz as f32 * CHUNK_SIZE as f32 + half_chunk - cam_z).powi(2);
                db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
            });

            for (&(cx, cz), buffers) in &water_chunks {
                let min = Vector3::new(
                    (cx * CHUNK_SIZE as i32) as f32,
                    0.0,
                    (cz * CHUNK_SIZE as i32) as f32,
                );
                let max = Vector3::new(
                    min.x + CHUNK_SIZE as f32,
                    CHUNK_HEIGHT as f32,
                    min.z + CHUNK_SIZE as f32,
                );

                if !self.frustum.is_box_visible(min, max) {
                    continue;
                }

                let wvb = buffers.water_vertex_buffer.as_ref().unwrap();
                let wib = buffers.water_index_buffer.as_ref().unwrap();
                water_pass.set_vertex_buffer(0, wvb.slice(..));
                water_pass.set_index_buffer(wib.slice(..), wgpu::IndexFormat::Uint16);
                water_pass.draw_indexed(0..buffers.water_index_count, 0, 0..1);
            }
        }

        // --- PASS 4: RENDER SEMI-TRANSPARENT BLOCKS (Ice) ---
        {
            let mut transparent_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Transparent Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: world_render_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Keep existing color
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Keep existing depth
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            transparent_pass.set_pipeline(&self.transparent_pipeline);
            transparent_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            transparent_pass.set_bind_group(1, &self.texture_atlas.bind_group, &[]);
            transparent_pass.set_bind_group(2, &self.fog_bind_group, &[]);
            transparent_pass.set_bind_group(3, &self.sun_bind_group, &[]);

            // Render semi-transparent blocks using cached GPU buffers with frustum culling
            for (&(cx, cz), buffers) in self.chunk_buffers.iter() {
                if let (Some(tvb), Some(tib)) = (&buffers.transparent_vertex_buffer, &buffers.transparent_index_buffer) {
                    if buffers.transparent_index_count > 0 {
                        // Calculate chunk bounding box
                        let min = Vector3::new(
                            (cx * CHUNK_SIZE as i32) as f32,
                            0.0,
                            (cz * CHUNK_SIZE as i32) as f32,
                        );
                        let max = Vector3::new(
                            min.x + CHUNK_SIZE as f32,
                            CHUNK_HEIGHT as f32,
                            min.z + CHUNK_SIZE as f32,
                        );

                        // Skip chunks outside the view frustum
                        if !self.frustum.is_box_visible(min, max) {
                            continue;
                        }

                        transparent_pass.set_vertex_buffer(0, tvb.slice(..));
                        transparent_pass.set_index_buffer(tib.slice(..), wgpu::IndexFormat::Uint16);
                        transparent_pass.draw_indexed(0..buffers.transparent_index_count, 0, 0..1);
                    }
                }
            }
        }

        // --- PASS 5: OVERLAYS (Breaking, Outlines, Debug) ---
        // Render breaking overlay if actively breaking a block
        if let Some(ref breaking_state) = self.breaking_state {
            let (bx, by, bz) = breaking_state.block_pos;
            let destroy_stage = breaking_state.get_destroy_stage();

            // Generate overlay vertices for visible faces
            let (overlay_vertices, overlay_indices): (Vec<Vertex>, Vec<u16>) = self.create_breaking_overlay_vertices(bx, by, bz, destroy_stage);

            if !overlay_vertices.is_empty() {
                let mut breaking_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Breaking Overlay Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: world_render_target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Breaking Overlay Vertex Buffer"),
                    contents: bytemuck::cast_slice(&overlay_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });

                let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Breaking Overlay Index Buffer"),
                    contents: bytemuck::cast_slice(&overlay_indices),
                    usage: wgpu::BufferUsages::INDEX,
                });

                breaking_pass.set_pipeline(&self.breaking_pipeline);
                breaking_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                breaking_pass.set_bind_group(1, &self.texture_atlas.bind_group, &[]);
                breaking_pass.set_bind_group(2, &self.fog_bind_group, &[]);
                breaking_pass.set_bind_group(3, &self.sun_bind_group, &[]);
                breaking_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                breaking_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                breaking_pass.draw_indexed(0..overlay_indices.len() as u32, 0, 0..1);
            }
        }

        // Render block outline if targeting a block - uses depth buffer to hide occluded edges
        if let Some((x, y, z)) = self.targeted_block {
            let mut outline_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Outline Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: world_render_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Keep existing depth values
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            let outline_vertices = create_block_outline(x, y, z);
            let outline_vertex_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Outline Vertex Buffer"),
                        contents: bytemuck::cast_slice(&outline_vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    });

            outline_pass.set_pipeline(&self.outline_pipeline);
            outline_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            outline_pass.set_vertex_buffer(0, outline_vertex_buffer.slice(..));
            outline_pass.draw(0..outline_vertices.len() as u32, 0..1);
        }

        // Render chunk outlines if debug mode is enabled
        if self.show_chunk_outlines {
            let mut chunk_outline_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Chunk Outline Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: world_render_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            let chunk_outline_vertices = self.build_chunk_outline_vertices();
            if !chunk_outline_vertices.is_empty() {
                let chunk_outline_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Chunk Outline Vertex Buffer"),
                            contents: bytemuck::cast_slice(&chunk_outline_vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                chunk_outline_pass.set_pipeline(&self.chunk_outline_pipeline);
                chunk_outline_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                chunk_outline_pass.set_vertex_buffer(0, chunk_outline_buffer.slice(..));
                chunk_outline_pass.draw(0..chunk_outline_vertices.len() as u32, 0..1);
            }
        }

        // Render enemy collision hitbox outlines if debug mode is enabled
        if self.show_enemy_hitboxes {
            let hitbox_verts = create_enemy_collision_outlines(&self.enemy_manager.enemies);
            if !hitbox_verts.is_empty() {
                let mut hitbox_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Enemy Hitbox Outline Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: world_render_target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                let hitbox_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Enemy Hitbox Vertex Buffer"),
                            contents: bytemuck::cast_slice(&hitbox_verts),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                hitbox_pass.set_pipeline(&self.chunk_outline_pipeline);
                hitbox_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                hitbox_pass.set_vertex_buffer(0, hitbox_buffer.slice(..));
                hitbox_pass.draw(0..hitbox_verts.len() as u32, 0..1);
            }
        }

        // --- BLOOM PASSES (emissive render → blur H → blur V → additive composite) ---
        // Pass B1: Render only emissive blocks into full-res emissive texture (with depth test)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Emissive Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_emissive_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,  // reuse depth from main render
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.bloom_emissive_pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, &self.texture_atlas.bind_group, &[]);
            pass.set_bind_group(2, &self.fog_bind_group, &[]);
            pass.set_bind_group(3, &self.sun_bind_group, &[]);

            // Re-render chunk geometry — fs_emissive discards non-emissive fragments
            for (&(cx, cz), buffers) in self.chunk_buffers.iter() {
                let min = Vector3::new(
                    (cx * CHUNK_SIZE as i32) as f32, 0.0,
                    (cz * CHUNK_SIZE as i32) as f32,
                );
                let max = Vector3::new(
                    min.x + CHUNK_SIZE as f32, CHUNK_HEIGHT as f32,
                    min.z + CHUNK_SIZE as f32,
                );
                if !self.frustum.is_box_visible(min, max) {
                    continue;
                }
                pass.set_vertex_buffer(0, buffers.vertex_buffer.slice(..));
                pass.set_index_buffer(buffers.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                pass.draw_indexed(0..buffers.index_count, 0, 0..1);
            }
        }

        // Pass B2: Downsample emissive_texture (full-res) → bloom_texture_a (quarter-res)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Downsample Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_texture_a_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.bloom_downsample_pipeline);
            pass.set_bind_group(0, &self.bloom_emissive_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass B3: Horizontal blur bloom_texture_a → bloom_texture_b (both quarter-res)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Blur H Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_texture_b_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.bloom_blur_h_pipeline);
            pass.set_bind_group(0, &self.bloom_a_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass B3: Vertical blur bloom_texture_b → bloom_texture_a
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Blur V Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_texture_a_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.bloom_blur_v_pipeline);
            pass.set_bind_group(0, &self.bloom_b_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass B4: Additive composite bloom_texture_a onto scene_texture
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bloom Composite Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.scene_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // preserve existing scene content
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.bloom_composite_pipeline);
            pass.set_bind_group(0, &self.bloom_a_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // --- PASS 6: POST-PROCESSING (Motion Blur + Underwater + Damage) ---
        // Texture routing:
        //   Neither:           blur scene_texture -> swap_chain
        //   Underwater only:   blur scene_texture -> post_process, underwater post_process -> swap_chain
        //   Damage only:       blur scene_texture -> post_process, damage post_process -> swap_chain
        //   Both:              blur scene_texture -> post_process, underwater post_process -> scene_texture,
        //                      damage scene_texture -> swap_chain
        let damage_active = self.damage_flash_intensity > 0.01;
        let needs_post = self.camera_underwater || damage_active;
        {
            let blur_target = if needs_post {
                &self.post_process_texture_view
            } else {
                &swap_chain_view
            };

            let mut blur_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Motion Blur Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: blur_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            blur_pass.set_pipeline(&self.motion_blur_pipeline);
            blur_pass.set_bind_group(0, &self.motion_blur_bind_group, &[]);
            blur_pass.draw(0..3, 0..1);
        }

        // Underwater: post_process_texture -> (scene_texture if damage follows, else swap_chain)
        if self.camera_underwater {
            let underwater_target = if damage_active {
                &self.scene_texture_view
            } else {
                &swap_chain_view
            };

            let mut underwater_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Underwater Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: underwater_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            underwater_pass.set_pipeline(&self.underwater_pipeline);
            underwater_pass.set_bind_group(0, &self.underwater_bind_group, &[]);
            underwater_pass.draw(0..3, 0..1);
        }

        // Damage flash: reads from post_process (no underwater) or scene_texture (after underwater) -> swap_chain
        if damage_active {
            let damage_bind_group = if self.camera_underwater {
                &self.damage_bind_group_alt
            } else {
                &self.damage_bind_group
            };

            let mut damage_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Damage Flash Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &swap_chain_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            damage_pass.set_pipeline(&self.damage_pipeline);
            damage_pass.set_bind_group(0, damage_bind_group, &[]);
            damage_pass.draw(0..3, 0..1);
        }

        if self.paused {
            // ── PAUSED: blur backdrop + modal overlay ─────────────────────────

            // Pass A: Gaussian-blur + darken scene_texture → swap_chain
            {
                let mut blur_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Pause Blur Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &swap_chain_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load:  wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                blur_pass.set_pipeline(&self.pause_blur_pipeline);
                blur_pass.set_bind_group(0, &self.pause_blur_bind_group, &[]);
                blur_pass.draw(0..3, 0..1);
            }

            // Build modal geometry on the CPU
            let sw = self.size.width  as f32;
            let sh = self.size.height as f32;

            // Sand panel vertices
            let sand_verts = self.pause_modal.build_sand_vertices(sw, sh);
            self.queue.write_buffer(
                &self.modal_sand_vertex_buffer,
                0,
                bytemuck::cast_slice(&sand_verts),
            );

            // UI overlay vertices (border, bevel, buttons, title)
            let ui_verts = self.pause_modal.build_ui_vertices(sw, sh);
            self.modal_ui_vertex_count = ui_verts.len() as u32;
            self.queue.write_buffer(
                &self.modal_ui_vertex_buffer,
                0,
                bytemuck::cast_slice(&ui_verts),
            );

            // Pass B: Sand-textured modal panel
            {
                let mut sand_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Modal Sand Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &swap_chain_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load:  wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                sand_pass.set_pipeline(&self.modal_sand_pipeline);
                sand_pass.set_bind_group(0, &self.modal_sand_bind_group, &[]);
                sand_pass.set_vertex_buffer(0, self.modal_sand_vertex_buffer.slice(..));
                sand_pass.draw(0..sand_verts.len() as u32, 0..1);
            }

            // Pass C: Solid-color modal UI (border, bevel, buttons, text)
            if self.modal_ui_vertex_count > 0 {
                let mut ui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Modal UI Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &swap_chain_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load:  wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                ui_pass.set_pipeline(&self.ui_pipeline);
                ui_pass.set_bind_group(0, &self.ui_bind_group, &[]);
                ui_pass.set_vertex_buffer(0, self.modal_ui_vertex_buffer.slice(..));
                ui_pass.draw(0..self.modal_ui_vertex_count, 0..1);
            }

        } else if self.hud_enabled {
            // ── NORMAL: crosshair + HUD ───────────────────────────────────────

            // If the crafting UI is open, render its modal sand background first
            if self.crafting_ui_open {
                let sw = self.size.width  as f32;
                let sh = self.size.height as f32;
                let sand_verts = self.crafting_modal.build_sand_vertices(sw, sh);
                self.queue.write_buffer(
                    &self.modal_sand_vertex_buffer,
                    0,
                    bytemuck::cast_slice(&sand_verts),
                );
                let mut sand_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Crafting Sand Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &swap_chain_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load:  wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                sand_pass.set_pipeline(&self.modal_sand_pipeline);
                sand_pass.set_bind_group(0, &self.modal_sand_bind_group, &[]);
                sand_pass.set_vertex_buffer(0, self.modal_sand_vertex_buffer.slice(..));
                sand_pass.draw(0..sand_verts.len() as u32, 0..1);
            }

            // --- PASS 7: UI (Crosshair & HUD) ---
            // Always render UI directly to Swap Chain so it stays sharp and on top
            {
                let mut ui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("UI Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &swap_chain_view, // Always Screen
                        resolve_target: None,
                        ops: wgpu::Operations {
                            // If underwater, we just drew the background in Pass 4, so Load.
                            // If NOT underwater, the World pass drew to this view in Pass 1, so Load.
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                // 1. Crosshair + HUD chrome (slot backgrounds, borders, name labels, fps)
                ui_pass.set_pipeline(&self.ui_pipeline);
                ui_pass.set_bind_group(0, &self.ui_bind_group, &[]);
                ui_pass.set_vertex_buffer(0, self.crosshair_vertex_buffer.slice(..));
                ui_pass.draw(0..self.crosshair_vertex_count, 0..1);
                ui_pass.set_vertex_buffer(0, self.hud_vertex_buffer.slice(..));
                ui_pass.draw(0..self.hud_vertex_count, 0..1);

                // 2. Item cubes (drawn on top of slot backgrounds)
                if self.item_cube_vertex_count > 0 {
                    ui_pass.set_pipeline(&self.item_cube_pipeline);
                    ui_pass.set_bind_group(0, &self.item_cube_bind_group, &[]);
                    ui_pass.set_vertex_buffer(0, self.item_cube_vertex_buffer.slice(..));
                    ui_pass.draw(0..self.item_cube_vertex_count, 0..1);
                }

                // 3. Count text overlay (drawn on top of cubes)
                if self.hud_text_vertex_count > 0 {
                    ui_pass.set_pipeline(&self.ui_pipeline);
                    ui_pass.set_bind_group(0, &self.ui_bind_group, &[]);
                    ui_pass.set_vertex_buffer(0, self.hud_text_vertex_buffer.slice(..));
                    ui_pass.draw(0..self.hud_text_vertex_count, 0..1);
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
