use super::*;

impl State {
    pub(super) fn respawn(&mut self) {
        self.player.health = self.player.max_health;
        self.camera.position = self.spawn_point;
        self.player.position = self.spawn_point;
        self.camera_controller.velocity = Vector3::new(0.0, 0.0, 0.0);
        self.camera_controller.on_ground = false;
        self.camera_controller.last_fall_velocity = 0.0;
    }

    /// Saves all modified chunks and enemies to disk (call before exiting)
    pub fn save_world(&mut self) {
        self.world.save_all_modified_chunks();
        self.enemy_manager.save_to_disk();
        crate::config::PlayerSave {
            x: self.camera.position.x,
            y: self.camera.position.y,
            z: self.camera.position.z,
            yaw: self.camera.yaw.0,
            pitch: self.camera.pitch.0,
            show_chunk_outlines: self.show_chunk_outlines,
            noclip_mode: self.noclip_mode,
            show_enemy_hitboxes: self.show_enemy_hitboxes,
        }
        .save();
    }

    fn complete_block_break(&mut self, x: i32, y: i32, z: i32, block_type: BlockType) {
        // Spawn dropped items (4 mini-blocks)
        let block_pos = Point3::new(x as f32, y as f32, z as f32);
        self.dropped_item_manager.spawn_drops(block_pos, block_type);

        // Spawn break particles
        self.particle_manager.spawn_block_break(block_pos, block_type);

        // Remove the block from the world
        self.world.set_block_world(x, y, z, BlockType::Air);

        // Clear water level if it was water
        if block_type == BlockType::Water {
            self.world.set_water_level_world(x, y, z, 0);
        }

        // Schedule water updates for adjacent water blocks (they may flow into the gap)
        let directions: [(i32, i32, i32); 6] = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)];
        for &(dx, dy, dz) in &directions {
            let nx = x + dx;
            let ny = y + dy;
            let nz = z + dz;
            if self.world.get_block_world(nx, ny, nz) == BlockType::Water {
                self.water_simulation.schedule_update(nx, ny, nz);
            }
        }

        println!("Broke block: {:?}", block_type);

        // Break any cross-model block sitting on top (e.g. grass tufts)
        let above = self.world.get_block_world(x, y + 1, z);
        if above.is_cross_model() {
            self.world.set_block_world(x, y + 1, z, BlockType::Air);
        }
    }

    /// Creates vertices for the breaking overlay on visible faces of a block
    pub(super) fn create_breaking_overlay_vertices(&self, x: i32, y: i32, z: i32, destroy_stage: u32) -> (Vec<Vertex>, Vec<u16>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let world_pos = Vector3::new(x as f32, y as f32, z as f32);
        let block_type = self.world.get_block_world(x, y, z);

        // Small offset to render in front of block faces (prevents z-fighting)
        let offset = 0.001;

        let face_directions: [(i32, i32, i32); 6] = [
            (0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0),
        ];

        // Get UVs for the destroy texture
        let tex_index = TEX_DESTROY_BASE + destroy_stage;
        let uvs = get_face_uvs(tex_index);

        for (face_idx, &(dx, dy, dz)) in face_directions.iter().enumerate() {
            let neighbor = self.world.get_block_world(x + dx, y + dy, z + dz);

            // Only render overlay on visible faces (where neighbor is transparent)
            if neighbor.is_transparent() {
                // Offset position slightly toward camera based on face normal
                let offset_pos = Vector3::new(
                    world_pos.x + dx as f32 * offset,
                    world_pos.y + dy as f32 * offset,
                    world_pos.z + dz as f32 * offset,
                );

                let face_verts = create_face_vertices(offset_pos, block_type, face_idx, [1.0; 4], tex_index, uvs, [1.0; 4]);

                let base_index = vertices.len() as u16;
                vertices.extend_from_slice(&face_verts);
                indices.extend_from_slice(&[
                    base_index, base_index + 1, base_index + 2,
                    base_index + 2, base_index + 3, base_index,
                ]);
            }
        }

        (vertices, indices)
    }

    fn update_block_breaking(&mut self, dt: f32) {
        if !self.left_mouse_held || !self.mouse_captured {
            // Not holding left mouse or not in game
            self.breaking_state = None;
            return;
        }

        let direction = self.camera.get_direction();

        // Check if cursor is aimed at an enemy (prioritize over block breaking)
        let enemy_hit_idx = self.enemy_manager.raycast_enemy(
            self.camera.position, direction, self.player.reach_distance,
        );

        if let Some(idx) = enemy_hit_idx {
            // Cursor is on an enemy - cancel any block breaking
            self.breaking_state = None;

            // Attempt melee hit if cooldown is ready
            if self.hit_cooldown <= 0.0 {
                // Calculate damage: 1.0 base + block durability if holding an item
                let damage = match self.player.inventory.get_selected_item() {
                    Some(item) => 1.0 + item.block_type.get_durability(),
                    None => 1.0,
                };

                // Calculate knockback direction (horizontal, from player toward enemy)
                let enemy_pos = self.enemy_manager.enemies[idx].position;
                let to_enemy = Vector3::new(
                    enemy_pos.x - self.camera.position.x,
                    0.0,
                    enemy_pos.z - self.camera.position.z,
                );
                let knockback_dir = if to_enemy.magnitude() > 0.001 {
                    to_enemy.normalize()
                } else {
                    Vector3::new(direction.x, 0.0, direction.z).normalize()
                };

                self.enemy_manager.enemies[idx].hit(damage, knockback_dir);
                self.hit_cooldown = 0.4;
                self.hit_indicator_timer = 0.2;
            }
            return;
        }

        // No enemy targeted - check for block breaking
        let target = self.player.raycast_block(direction, &self.world);

        // If no block targeted either, attempt a melee swing into the air
        if target.is_none() && self.hit_cooldown <= 0.0 {
            // Show the hit indicator even when swinging at nothing
            // (no damage applied, just visual feedback of the swing)
        }

        match (&mut self.breaking_state, target) {
            (Some(state), Some((x, y, z, _))) => {
                if state.block_pos == (x, y, z) {
                    // Still targeting same block - increment progress
                    let durability = state.block_type.get_durability();
                    if durability > 0.0 {
                        state.progress += dt / durability;

                        if state.progress >= 1.0 {
                            // Block broken!
                            let block_type = state.block_type;
                            let pos = state.block_pos;
                            self.breaking_state = None;
                            self.complete_block_break(pos.0, pos.1, pos.2, block_type);
                        }
                    }
                } else {
                    // Targeting different block - reset
                    let block_type = self.world.get_block_world(x, y, z);
                    if block_type.is_breakable() {
                        self.breaking_state = Some(BreakingState::new((x, y, z), block_type));
                    } else {
                        self.breaking_state = None;
                    }
                }
            }
            (None, Some((x, y, z, _))) => {
                // Start breaking new block
                let block_type = self.world.get_block_world(x, y, z);
                if block_type.is_breakable() {
                    self.breaking_state = Some(BreakingState::new((x, y, z), block_type));
                }
            }
            (_, None) => {
                // Not targeting any block
                self.breaking_state = None;
            }
        }
    }

    pub(super) fn handle_block_place(&mut self) {
        if let Some(selected) = self.player.inventory.get_selected_item() {
            // Only allow placement if player has at least 1.0 of the block
            if selected.count < 1.0 {
                return;
            }

            let block_type = selected.block_type;
            let direction = self.camera.get_direction();

            if let Some((x, y, z, normal)) = self.player.raycast_block(direction, &self.world) {
                let place_x = x + normal.x;
                let place_y = y + normal.y;
                let place_z = z + normal.z;

                if self.world.get_block_world(place_x, place_y, place_z) == BlockType::Air {
                    // Check if placement would overlap with player's collision capsule
                    let player_pos = self.camera.position;
                    let player_radius = 0.25;
                    let player_height = 1.6;
                    let player_min_x = player_pos.x - player_radius;
                    let player_max_x = player_pos.x + player_radius;
                    let player_min_y = player_pos.y - player_height;
                    let player_max_y = player_pos.y;
                    let player_min_z = player_pos.z - player_radius;
                    let player_max_z = player_pos.z + player_radius;

                    let block_min_x = place_x as f32;
                    let block_max_x = place_x as f32 + 1.0;
                    let block_min_y = place_y as f32;
                    let block_max_y = place_y as f32 + 1.0;
                    let block_min_z = place_z as f32;
                    let block_max_z = place_z as f32 + 1.0;

                    let overlaps = player_max_x > block_min_x && player_min_x < block_max_x
                        && player_max_y > block_min_y && player_min_y < block_max_y
                        && player_max_z > block_min_z && player_min_z < block_max_z;

                    if overlaps {
                        return; // Don't place block inside player
                    }

                    self.world
                        .set_block_world(place_x, place_y, place_z, block_type);

                    // If placing water, set it as a source block and schedule flow
                    if block_type == BlockType::Water {
                        self.world.set_water_level_world(place_x, place_y, place_z, crate::chunk::WATER_LEVEL_SOURCE);
                        self.water_simulation.schedule_update(place_x, place_y, place_z);
                    }

                    self.player
                        .inventory
                        .remove_item(self.player.inventory.selected_slot, 1.0);
                    println!("Placed block: {:?}", block_type);
                }
            }
        }
    }

    pub fn update(&mut self) {
        let now = Instant::now();
        // Cap dt to 50 ms (20 fps minimum) so a long first frame or stutter
        // doesn't let gravity teleport the player through the terrain.
        let dt = (now - self.last_frame).as_secs_f32().min(0.05);
        self.last_frame = now;

        // ── Walking sound fade ────────────────────────────────────────────────
        {
            const START_WALKING_SOUND_FADE_SPEED: f32 = 4.0;
            const STOP_WALKING_SOUND_FADE_SPEED: f32 = 15.0; // 20.0 is almost instantaneous

            let horiz_speed = {
                let vx = self.camera_controller.velocity.x;
                let vz = self.camera_controller.velocity.z;
                (vx * vx + vz * vz).sqrt()
            };
            let is_walking = !self.paused
                && self.camera_controller.on_ground
                && horiz_speed > 0.5;
            let target = if is_walking { 1.0f32 } else { 0.0 };
            let fade_speed = if is_walking { START_WALKING_SOUND_FADE_SPEED } else { STOP_WALKING_SOUND_FADE_SPEED };
            self.walk_volume += (target - self.walk_volume) * (fade_speed * dt).min(1.0);
            self.walk_volume = self.walk_volume.clamp(0.0, 1.0);
            if let Some(sink) = &self.walk_sink {
                sink.set_volume(self.walk_volume);
            }
        }

        // Update Underwater shader time
        let total_time = (now - self.start_time).as_secs_f32();
        self.queue.write_buffer(
            &self.underwater_uniform_buffer,
            0,
            bytemuck::cast_slice(&[total_time]),
        );

        // Update damage flash uniforms
        self.queue.write_buffer(
            &self.damage_uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.damage_flash_intensity, total_time]),
        );

        // Update water wave animation time (only the time component, rest stays constant)
        self.queue.write_buffer(
            &self.water_time_buffer,
            0,
            bytemuck::cast_slice(&[total_time]),
        );

        // ── Sun / sky uniform update ─────────────────────────────────────────
        {
            let cycle = self.sky_config.day_cycle_secs;
            // time_of_day: 0.0 = sunrise, 0.25 = noon, 0.5 = sunset, 0.75 = midnight
            let time_of_day = (total_time % cycle) / cycle;

            // Sun angle: full 360-degree rotation over the cycle
            // At time_of_day=0.0 sun is at horizon (rising east), 0.25 = zenith, 0.5 = horizon (setting west)
            let sun_angle = time_of_day * std::f32::consts::TAU;

            // Sun direction: rotates in the Y-Z plane (rises from east, sets in west)
            // X component adds slight tilt for more interesting shadows
            let sun_y = sun_angle.cos();  // 1 at noon, -1 at midnight
            let sun_x = 0.3 * sun_angle.sin(); // slight east-west offset
            let sun_z = sun_angle.sin();  // main east-west axis

            let sun_len = (sun_x * sun_x + sun_y * sun_y + sun_z * sun_z).sqrt();
            let sun_dir = [sun_x / sun_len, sun_y / sun_len, sun_z / sun_len, 0.0];

            // Sun intensity: 1.0 when above horizon, fades to 0 below
            // smoothstep from -0.1 to 0.2 on sun_y
            let sun_elevation = sun_dir[1];
            let sun_intensity = ((sun_elevation + 0.1) / 0.3).clamp(0.0, 1.0);
            let sun_intensity = sun_intensity * sun_intensity * (3.0 - 2.0 * sun_intensity); // smoothstep

            // ── Compute sun's orthographic view-projection for shadow map ────
            let shadow_range = self.sky_config.shadow_map_range;
            let shadow_depth = self.sky_config.shadow_map_depth;
            let half = shadow_range * 0.5;

            // Sun "eye" position: offset from player along sun direction by full depth
            // so the entire shadow volume is in front of the near plane
            let sun_eye = cgmath::Point3::new(
                self.camera.position.x + sun_dir[0] * shadow_depth,
                self.camera.position.y + sun_dir[1] * shadow_depth,
                self.camera.position.z + sun_dir[2] * shadow_depth,
            );
            let sun_target = cgmath::Point3::new(
                self.camera.position.x,
                self.camera.position.y,
                self.camera.position.z,
            );

            // Choose up vector that isn't parallel to sun direction
            let sun_dir_v = Vector3::new(sun_dir[0], sun_dir[1], sun_dir[2]);
            let up_candidate = if sun_dir_v.y.abs() > 0.99 {
                Vector3::new(0.0, 0.0, 1.0)
            } else {
                Vector3::new(0.0, 1.0, 0.0)
            };

            let sun_view = cgmath::Matrix4::look_at_rh(sun_eye, sun_target, up_candidate);
            let sun_proj = cgmath::ortho(-half, half, -half, half, 0.1, shadow_depth * 2.0);
            // Correction matrix: remap z from OpenGL [-1,1] to wgpu [0,1]
            #[rustfmt::skip]
            let opengl_to_wgpu = cgmath::Matrix4::new(
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.5, 0.0,
                0.0, 0.0, 0.5, 1.0,
            );
            let sun_view_proj_mat = opengl_to_wgpu * sun_proj * sun_view;
            let sun_view_proj: [[f32; 4]; 4] = sun_view_proj_mat.into();

            // Write sun camera for shadow map render pass
            self.queue.write_buffer(
                &self.shadow_map_camera_buffer,
                0,
                bytemuck::cast_slice(&sun_view_proj),
            );

            // Write sun uniform for block shaders
            let sun_uniform = crate::config::SunUniform {
                sun_view_proj,
                sun_dir,
                sun_color: [
                    self.sky_config.sun_color_r * self.sky_config.sun_brightness,
                    self.sky_config.sun_color_g * self.sky_config.sun_brightness,
                    self.sky_config.sun_color_b * self.sky_config.sun_brightness,
                    1.0,
                ],
                params: [sun_intensity, self.sky_config.night_ambient, self.sky_config.shadow_strength, self.sky_config.shadow_bias],
                params2: [self.sky_config.shadow_softness, self.sky_config.shadow_normal_bias, 0.0, 0.0],
            };
            self.queue.write_buffer(
                &self.sun_buffer,
                0,
                bytemuck::cast_slice(&[sun_uniform]),
            );

            // Write sky uniform for sky shader
            // Layout: inv_view_proj (64B) + camera_pos (16B) + sun_dir (16B) + sun_color (16B)
            //       + params (16B) + sky_params (16B) + sky_params2 (16B)
            //       + zenith_day (16B) + horizon_day (16B) + zenith_night (16B) + sunset_color (16B)
            //       = 12 * 16 = 192 bytes
            let view_proj = self.projection.calc_matrix() * self.camera.get_view_matrix();
            let inv_view_proj: [[f32; 4]; 4] = cgmath::Matrix4::from(
                view_proj.invert().unwrap_or(cgmath::Matrix4::from_scale(1.0))
            ).into();

            let sun_radius_rad = self.sky_config.sun_radius_deg.to_radians();

            let sky_data: [f32; 56] = [
                // inv_view_proj (16 floats)
                inv_view_proj[0][0], inv_view_proj[0][1], inv_view_proj[0][2], inv_view_proj[0][3],
                inv_view_proj[1][0], inv_view_proj[1][1], inv_view_proj[1][2], inv_view_proj[1][3],
                inv_view_proj[2][0], inv_view_proj[2][1], inv_view_proj[2][2], inv_view_proj[2][3],
                inv_view_proj[3][0], inv_view_proj[3][1], inv_view_proj[3][2], inv_view_proj[3][3],
                // camera_pos (4 floats)
                self.camera.position.x, self.camera.position.y, self.camera.position.z, 1.0,
                // sun_dir (4 floats)
                sun_dir[0], sun_dir[1], sun_dir[2], 0.0,
                // sun_color (4 floats)
                sun_uniform.sun_color[0], sun_uniform.sun_color[1], sun_uniform.sun_color[2], 1.0,
                // params (4 floats)
                sun_intensity, self.sky_config.night_ambient, self.sky_config.shadow_strength, time_of_day,
                // sky_params (4 floats)
                sun_radius_rad, self.sky_config.sun_glow_falloff, self.sky_config.star_density, self.sky_config.star_brightness,
                // sky_params2 (4 floats)
                self.sky_config.star_twinkle_speed, 0.0, 0.0, total_time,
                // zenith_day (4 floats)
                self.sky_config.sky_zenith_day_r, self.sky_config.sky_zenith_day_g, self.sky_config.sky_zenith_day_b, 0.0,
                // horizon_day (4 floats)
                self.sky_config.sky_horizon_day_r, self.sky_config.sky_horizon_day_g, self.sky_config.sky_horizon_day_b, 0.0,
                // zenith_night (4 floats)
                self.sky_config.sky_zenith_night_r, self.sky_config.sky_zenith_night_g, self.sky_config.sky_zenith_night_b, 0.0,
                // sunset_color (4 floats)
                self.sky_config.sunset_color_r, self.sky_config.sunset_color_g, self.sky_config.sunset_color_b, 0.0,
            ];
            self.queue.write_buffer(
                &self.sky_uniform_buffer,
                0,
                bytemuck::cast_slice(&sky_data),
            );
        }

        // Update FPS counter
        self.fps_frame_count += 1;
        self.fps_timer += dt;
        if self.fps_timer >= 0.5 {
            self.fps = self.fps_frame_count as f32 / self.fps_timer;
            self.fps_frame_count = 0;
            self.fps_timer = 0.0;
        }

        // Clear hover when paused; it will be recomputed below when unpaused.
        self.hovered_dropped_item = None;

        // ── Simulation (skipped while paused) ────────────────────────────────
        if !self.paused {

        // Update camera
        self.camera_underwater = self.camera_controller.update_camera(&mut self.camera, dt, &self.world, self.noclip_mode);
        self.player.position = self.camera.position;

        // Update motion blur uniform from camera rotation velocity
        {
            use crate::camera::{MOTION_BLUR_AMOUNT};
            let yaw_vel = self.camera_controller.yaw_velocity;
            let pitch_vel = self.camera_controller.pitch_velocity;
            // Map rotation velocity to screen-space blur direction
            // yaw -> horizontal, pitch -> vertical; scale by dt-like factor for perceptual sizing
            let blur_scale = 0.015; // converts rad/s to UV-space offset
            let blur_x = yaw_vel * blur_scale;
            let blur_y = pitch_vel * blur_scale;
            let magnitude = (blur_x * blur_x + blur_y * blur_y).sqrt();
            // Normalize direction, strength is magnitude * user amount
            let (dir_x, dir_y) = if magnitude > 0.0001 {
                (blur_x / magnitude, blur_y / magnitude)
            } else {
                (0.0, 0.0)
            };
            let strength = (magnitude * MOTION_BLUR_AMOUNT).min(0.05); // cap to avoid extreme blur
            let uniform_data: [f32; 4] = [dir_x, dir_y, strength, 0.0];
            self.queue.write_buffer(
                &self.motion_blur_uniform_buffer,
                0,
                bytemuck::cast_slice(&uniform_data),
            );
        }

        // Fall damage: velocity threshold of -15.0 (roughly 4+ block fall)
        // Damage scales with how far beyond the threshold
        let fall_vel = self.camera_controller.last_fall_velocity;
        if fall_vel < -15.0 {
            let damage = (fall_vel.abs() - 15.0) * 2.5;
            self.player.take_damage(damage);
            self.damage_flash_intensity = 1.0;
            if !self.player.is_alive() {
                self.respawn();
            }
        }
        self.camera_controller.last_fall_velocity = 0.0;

        // Update world
        self.world
            .update_chunks((self.camera.position.x, self.camera.position.z));
        self.world.rebuild_dirty_chunks(self.smooth_lighting);

        // Update water simulation
        self.water_simulation.update(&mut self.world, dt);

        // Update enemies and spawn death particles
        let death_events = self.enemy_manager.update(dt, self.player.position, &self.world);
        for (pos, color) in death_events {
            self.particle_manager.spawn_enemy_death(pos, color);
        }

        // Update birds
        self.bird_manager.update(dt, self.player.position, &self.world);

        // Update fish
        self.fish_manager.update(dt, self.player.position, &self.world);

        // Update clouds: advances drift and loads/unloads chunks as needed
        self.cloud_manager.update(self.player.position, dt, self.world.get_render_distance());

        // Re-upload combined geometry to GPU only when the chunk set changed (rare)
        if self.cloud_manager.geometry_rebuilt() {
            let (cloud_vertices, cloud_indices) = self.cloud_manager.geometry();
            if !cloud_vertices.is_empty() {
                use wgpu::util::DeviceExt;
                let vertex_data = bytemuck::cast_slice(cloud_vertices);
                let index_data  = bytemuck::cast_slice(cloud_indices);

                if vertex_data.len() as u64 > self.cloud_vertex_buffer.size() {
                    self.cloud_vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Cloud Vertex Buffer"),
                        contents: vertex_data,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    });
                } else {
                    self.queue.write_buffer(&self.cloud_vertex_buffer, 0, vertex_data);
                }

                if index_data.len() as u64 > self.cloud_index_buffer.size() {
                    self.cloud_index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Cloud Index Buffer"),
                        contents: index_data,
                        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    });
                } else {
                    self.queue.write_buffer(&self.cloud_index_buffer, 0, index_data);
                }
            }
            self.cloud_index_count = self.cloud_manager.index_count();
        }

        // Always update drift uniform (16 bytes) — this is the only per-frame cloud GPU write
        let (drift_x, drift_z) = self.cloud_manager.get_drift();
        self.queue.write_buffer(
            &self.cloud_drift_buffer,
            0,
            bytemuck::bytes_of(&[drift_x, drift_z, 0.0f32, 0.0f32]),
        );

        // Advance dropped-item physics (no auto-collect; player must right-click)
        self.dropped_item_manager.update(dt, &self.world);

        // Raycast to find which dropped item (if any) the crosshair is over
        let look_dir = self.camera.get_direction();
        self.hovered_dropped_item = self.dropped_item_manager
            .raycast_item(self.camera.position, look_dir, 5.0);

        // Update particles
        self.particle_manager.update(dt);

        // Check for enemy damage
        let damage = self.enemy_manager.check_player_damage(self.player.position);
        if damage > 0.0 {
            self.player.take_damage(damage * dt);
            self.damage_flash_intensity = 1.0;
            if !self.player.is_alive() {
                self.respawn();
            }
        }

        // Decay damage flash
        if self.damage_flash_intensity > 0.0 {
            self.damage_flash_intensity = (self.damage_flash_intensity - dt * 3.0).max(0.0);
        }

        // Update targeted block (for outline rendering)
        let direction = self.camera.get_direction();
        self.targeted_block = self.player.raycast_block(direction, &self.world)
            .map(|(x, y, z, _)| (x, y, z));

        // Update crafting table hover hint
        self.hovered_crafting_table = match self.targeted_block {
            Some((x, y, z)) => self.world.get_block_world(x, y, z) == BlockType::CraftingTable,
            None => false,
        };

        // Update block breaking / melee combat
        self.update_block_breaking(dt);

        // Decay hit indicator and cooldown timers
        if self.hit_cooldown > 0.0 {
            self.hit_cooldown = (self.hit_cooldown - dt).max(0.0);
        }
        if self.hit_indicator_timer > 0.0 {
            self.hit_indicator_timer = (self.hit_indicator_timer - dt).max(0.0);
        }

        // Rebuild crosshair with hit indicator state
        {
            let aspect = self.size.width as f32 / self.size.height as f32;
            let hit_active = self.hit_indicator_timer > 0.0;
            let crosshair_verts = Self::build_crosshair_vertices(aspect, hit_active);
            self.crosshair_vertex_count = crosshair_verts.len() as u32;
            self.queue.write_buffer(
                &self.crosshair_vertex_buffer,
                0,
                bytemuck::cast_slice(&crosshair_verts),
            );
        }

        } // end if !self.paused

        // Update camera uniform and frustum (always, so the paused scene renders)
        self.camera_uniform
            .update_view_proj(&self.camera, &self.projection);
        self.frustum = Frustum::from_view_proj(
            &(self.projection.calc_matrix() * self.camera.get_view_matrix())
        );
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // Update chunk GPU buffer cache
        self.update_chunk_buffers();
    }
}
