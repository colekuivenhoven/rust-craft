use super::*;

impl State {
    pub(super) fn build_crosshair_vertices(aspect_ratio: f32, hit_active: bool) -> Vec<UiVertex> {
        let crosshair_size = 0.03;
        let crosshair_thickness = 0.01;
        let crosshair_color = [1.0, 1.0, 1.0, 0.5];

        // Correct X coordinates for aspect ratio to maintain 1:1 ratio
        let h_size = crosshair_size / aspect_ratio;
        let h_thick = crosshair_thickness / aspect_ratio;

        let mut verts = vec![
            // Horizontal bar (X scaled for aspect ratio)
            UiVertex { position: [-h_size, -crosshair_thickness], color: crosshair_color },
            UiVertex { position: [h_size, -crosshair_thickness], color: crosshair_color },
            UiVertex { position: [h_size, crosshair_thickness], color: crosshair_color },
            UiVertex { position: [-h_size, -crosshair_thickness], color: crosshair_color },
            UiVertex { position: [h_size, crosshair_thickness], color: crosshair_color },
            UiVertex { position: [-h_size, crosshair_thickness], color: crosshair_color },
            // Vertical bar (X scaled for aspect ratio)
            UiVertex { position: [-h_thick, -crosshair_size], color: crosshair_color },
            UiVertex { position: [h_thick, -crosshair_size], color: crosshair_color },
            UiVertex { position: [h_thick, crosshair_size], color: crosshair_color },
            UiVertex { position: [-h_thick, -crosshair_size], color: crosshair_color },
            UiVertex { position: [h_thick, crosshair_size], color: crosshair_color },
            UiVertex { position: [-h_thick, crosshair_size], color: crosshair_color },
        ];

        // Red hit indicator circle
        if hit_active {
            let hit_color = [1.0, 0.2, 0.2, 0.9];
            let radius = 0.10;
            let thickness = 0.008;
            let segments = 24;

            for i in 0..segments {
                let angle0 = (i as f32 / segments as f32) * std::f32::consts::TAU;
                let angle1 = ((i + 1) as f32 / segments as f32) * std::f32::consts::TAU;

                let inner_r = radius - thickness * 0.5;
                let outer_r = radius + thickness * 0.5;

                // Inner and outer points for this segment (aspect-corrected X)
                let ix0 = inner_r * angle0.cos() / aspect_ratio;
                let iy0 = inner_r * angle0.sin();
                let ox0 = outer_r * angle0.cos() / aspect_ratio;
                let oy0 = outer_r * angle0.sin();
                let ix1 = inner_r * angle1.cos() / aspect_ratio;
                let iy1 = inner_r * angle1.sin();
                let ox1 = outer_r * angle1.cos() / aspect_ratio;
                let oy1 = outer_r * angle1.sin();

                // Two triangles forming the quad for this segment
                verts.push(UiVertex { position: [ix0, iy0], color: hit_color });
                verts.push(UiVertex { position: [ox0, oy0], color: hit_color });
                verts.push(UiVertex { position: [ox1, oy1], color: hit_color });
                verts.push(UiVertex { position: [ix0, iy0], color: hit_color });
                verts.push(UiVertex { position: [ox1, oy1], color: hit_color });
                verts.push(UiVertex { position: [ix1, iy1], color: hit_color });
            }
        }

        verts
    }

    /// Project a world-space point to screen pixel coordinates.
    /// Returns None if the point is behind the camera.
    fn world_to_screen(&self, world_pos: Point3<f32>) -> Option<(f32, f32)> {
        let view = self.camera.get_view_matrix();
        let proj = self.projection.calc_matrix();
        let vp = proj * view;

        // Transform to clip space
        let p = vp * cgmath::Vector4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);
        if p.w <= 0.0 {
            return None; // Behind camera
        }

        // Perspective divide to NDC
        let ndc_x = p.x / p.w;
        let ndc_y = p.y / p.w;

        // NDC to pixel coordinates
        let screen_w = self.size.width as f32;
        let screen_h = self.size.height as f32;
        let px = (ndc_x + 1.0) * 0.5 * screen_w;
        let py = (1.0 - ndc_y) * 0.5 * screen_h;

        Some((px, py))
    }

    pub(super) fn rebuild_hud_vertices(&mut self) {
        let screen_w = self.size.width as f32;
        let screen_h = self.size.height as f32;

        let slots = self.player.inventory.size.max(1);
        let selected = self.player.inventory.selected_slot.min(slots - 1);

        // Layout in pixels
        let slot_size = 80.0;
        let slot_gap = 8.0;
        let margin_bottom = 32.0;
        let total_w = (slots as f32) * slot_size + (slots as f32 - 1.0) * slot_gap;
        let start_x = (screen_w - total_w) * 0.5;
        let start_y = screen_h - margin_bottom - slot_size;

        let mut verts: Vec<UiVertex> = Vec::with_capacity(4096);
        // Count text is rendered after item cubes so it appears on top
        let mut text_verts: Vec<UiVertex> = Vec::with_capacity(512);

        // === FPS Counter (top-left) ===
        let fps_text = format!("{} FPS", self.fps as u32);
        let fps_color = [1.0, 1.0, 1.0, 0.9];
        let fps_bg_color = [0.0, 0.0, 0.0, 0.5];
        let fps_x = 10.0;
        let fps_y = 10.0;
        let fps_scale = 2.0;
        let fps_char_w = 6.0 * fps_scale; // 5 pixels + 1 spacing
        let fps_char_h = 7.0 * fps_scale;
        let fps_text_width = fps_text.len() as f32 * fps_char_w;
        // Background
        bitmap_font::push_rect_px(
            &mut verts,
            fps_x - 4.0,
            fps_y - 4.0,
            fps_text_width + 8.0,
            fps_char_h + 8.0,
            fps_bg_color,
            screen_w,
            screen_h,
        );
        bitmap_font::draw_text_quads(
            &mut verts,
            &fps_text,
            fps_x,
            fps_y,
            fps_scale,
            fps_scale,
            fps_color,
            screen_w,
            screen_h,
        );

        // === Chunk Outline Toggle Indicator (below FPS) ===
        let (debug_text, debug_color, debug_bg_color) = if self.show_chunk_outlines {
            ("F1 - CHUNK OUTLINE: ON", [0.5, 1.0, 0.5, 1.0], [0.0, 0.2, 0.0, 0.6])  // Light green when on
        } else {
            ("F1 - CHUNK OUTLINE: OFF", [1.0, 1.0, 1.0, 0.9], [0.0, 0.0, 0.0, 0.5])  // White when off
        };
        let debug_y = fps_y + fps_char_h + 8.0;
        let debug_text_width = debug_text.len() as f32 * fps_char_w;
        // Background
        bitmap_font::push_rect_px(
            &mut verts,
            fps_x - 4.0,
            debug_y - 4.0,
            debug_text_width + 8.0,
            fps_char_h + 8.0,
            debug_bg_color,
            screen_w,
            screen_h,
        );
        bitmap_font::draw_text_quads(
            &mut verts,
            debug_text,
            fps_x,
            debug_y,
            fps_scale,
            fps_scale,
            debug_color,
            screen_w,
            screen_h,
        );

        // === Noclip Toggle Indicator (below Chunk Outline) ===
        let (noclip_text, noclip_color, noclip_bg_color) = if self.noclip_mode {
            ("F2 - NOCLIP: ON", [0.5, 1.0, 0.5, 1.0], [0.0, 0.2, 0.0, 0.6])  // Light green when on
        } else {
            ("F2 - NOCLIP: OFF", [1.0, 1.0, 1.0, 0.9], [0.0, 0.0, 0.0, 0.5])  // White when off
        };
        let noclip_y = debug_y + fps_char_h + 8.0;
        let noclip_text_width = noclip_text.len() as f32 * fps_char_w;
        // Background
        bitmap_font::push_rect_px(
            &mut verts,
            fps_x - 4.0,
            noclip_y - 4.0,
            noclip_text_width + 8.0,
            fps_char_h + 8.0,
            noclip_bg_color,
            screen_w,
            screen_h,
        );
        bitmap_font::draw_text_quads(
            &mut verts,
            noclip_text,
            fps_x,
            noclip_y,
            fps_scale,
            fps_scale,
            noclip_color,
            screen_w,
            screen_h,
        );

        // === Enemy Hitbox Toggle Indicator (below Noclip) ===
        let (hitbox_text, hitbox_color, hitbox_bg_color) = if self.show_enemy_hitboxes {
            ("F3 - ENEMY HITBOX: ON", [0.5, 1.0, 0.5, 1.0], [0.0, 0.2, 0.0, 0.6])
        } else {
            ("F3 - ENEMY HITBOX: OFF", [1.0, 1.0, 1.0, 0.9], [0.0, 0.0, 0.0, 0.5])
        };
        let hitbox_y = noclip_y + fps_char_h + 8.0;
        let hitbox_text_width = hitbox_text.len() as f32 * fps_char_w;
        bitmap_font::push_rect_px(
            &mut verts,
            fps_x - 4.0,
            hitbox_y - 4.0,
            hitbox_text_width + 8.0,
            fps_char_h + 8.0,
            hitbox_bg_color,
            screen_w,
            screen_h,
        );
        bitmap_font::draw_text_quads(
            &mut verts,
            hitbox_text,
            fps_x,
            hitbox_y,
            fps_scale,
            fps_scale,
            hitbox_color,
            screen_w,
            screen_h,
        );

        // === Smooth Lighting Toggle Indicator (below Enemy Hitbox) ===
        let (smooth_text, smooth_color, smooth_bg_color) = if self.smooth_lighting {
            ("F4 - SMOOTH LIGHTING: ON", [0.5, 1.0, 0.5, 1.0], [0.0, 0.2, 0.0, 0.6])
        } else {
            ("F4 - SMOOTH LIGHTING: OFF", [1.0, 1.0, 1.0, 0.9], [0.0, 0.0, 0.0, 0.5])
        };
        let smooth_y = hitbox_y + fps_char_h + 8.0;
        let smooth_text_width = smooth_text.len() as f32 * fps_char_w;
        bitmap_font::push_rect_px(
            &mut verts,
            fps_x - 4.0,
            smooth_y - 4.0,
            smooth_text_width + 8.0,
            fps_char_h + 8.0,
            smooth_bg_color,
            screen_w,
            screen_h,
        );
        bitmap_font::draw_text_quads(
            &mut verts,
            smooth_text,
            fps_x,
            smooth_y,
            fps_scale,
            fps_scale,
            smooth_color,
            screen_w,
            screen_h,
        );

        let (hud_text, hud_color, hud_bg_color) = if self.hud_enabled {
            ("F5 - HUD Enabled: ON", [0.5, 1.0, 0.5, 1.0], [0.0, 0.2, 0.0, 0.6])
        } else {
            ("F5 - HUD Enabled: OFF", [1.0, 1.0, 1.0, 0.9], [0.0, 0.0, 0.0, 0.5])
        };
        let hud_toggle_y = smooth_y + fps_char_h + 8.0;
        let hud_text_width = hud_text.len() as f32 * fps_char_w;
        bitmap_font::push_rect_px(
            &mut verts,
            fps_x - 4.0,
            hud_toggle_y - 4.0,
            hud_text_width + 8.0,
            fps_char_h + 8.0,
            hud_bg_color,
            screen_w,
            screen_h,
        );
        bitmap_font::draw_text_quads(
            &mut verts,
            hud_text,
            fps_x,
            hud_toggle_y,
            fps_scale,
            fps_scale,
            hud_color,
            screen_w,
            screen_h,
        );

        // === Debug Axes (top-right) ===
        self.build_debug_axes(&mut verts, screen_w, screen_h);

        // === Player Position (below compass) ===
        let pos_text = format!(
            "X:{:.2}, Y:{:.2}, Z:{:.2}",
            self.camera.position.x,
            self.camera.position.y,
            self.camera.position.z
        );
        let pos_color = [1.0, 1.0, 1.0, 0.9];
        let pos_bg_color = [0.0, 0.0, 0.0, 0.5];
        let pos_scale = 2.0;
        let pos_char_w = 6.0 * pos_scale;
        let pos_char_h = 7.0 * pos_scale;
        let pos_text_width = pos_text.len() as f32 * pos_char_w;
        let pos_x = screen_w - pos_text_width - 9.0; // Right-aligned with padding
        let pos_y = 125.0; // Below compass

        // Background
        bitmap_font::push_rect_px(
            &mut verts,
            pos_x - 4.0,
            pos_y - 4.0,
            pos_text_width + 8.0,
            pos_char_h + 8.0,
            pos_bg_color,
            screen_w,
            screen_h,
        );
        bitmap_font::draw_text_quads(
            &mut verts,
            &pos_text,
            pos_x,
            pos_y,
            pos_scale,
            pos_scale,
            pos_color,
            screen_w,
            screen_h,
        );

        // === Health Bar ===
        let health_bar_w = 300.0;
        let health_bar_h = 30.0;
        let health_bar_border = 2.0;
        let health_bar_padding = 6.0; // Right padding for text
        //let health_bar_x = start_x;
        let health_bar_x = (screen_w - health_bar_w) / 2.0;
        let health_bar_y = 24.0;
        let health_pct = self.player.health / self.player.max_health;

        // Dark background
        bitmap_font::push_rect_px(
            &mut verts,
            health_bar_x,
            health_bar_y,
            health_bar_w,
            health_bar_h,
            [0.0, 0.0, 0.0, 0.25],
            screen_w,
            screen_h,
        );

        // White border (4 thin quads)
        let border_color = [1.0, 1.0, 1.0, 0.85];
        // top
        bitmap_font::push_rect_px(&mut verts, health_bar_x, health_bar_y, health_bar_w, health_bar_border, border_color, screen_w, screen_h);
        // bottom
        bitmap_font::push_rect_px(&mut verts, health_bar_x, health_bar_y + health_bar_h - health_bar_border, health_bar_w, health_bar_border, border_color, screen_w, screen_h);
        // left
        bitmap_font::push_rect_px(&mut verts, health_bar_x, health_bar_y, health_bar_border, health_bar_h, border_color, screen_w, screen_h);
        // right
        bitmap_font::push_rect_px(&mut verts, health_bar_x + health_bar_w - health_bar_border, health_bar_y, health_bar_border, health_bar_h, border_color, screen_w, screen_h);

        // Red health fill (inside borders)
        let fill_x = health_bar_x + health_bar_border;
        let fill_y = health_bar_y + health_bar_border;
        let fill_max_w = health_bar_w - health_bar_border * 2.0;
        let fill_h = health_bar_h - health_bar_border * 2.0;
        let fill_w = fill_max_w * health_pct;
        if fill_w > 0.0 {
            bitmap_font::push_rect_px(
                &mut verts,
                fill_x,
                fill_y,
                fill_w,
                fill_h,
                [0.8, 0.1, 0.1, 0.9],
                screen_w,
                screen_h,
            );
        }

        // Percentage text (anchored right inside the bar)
        let health_text = format!("{}%", (health_pct * 100.0).round() as u32);
        let health_text_scale = 2.0;
        let health_char_w = 6.0 * health_text_scale;
        let health_char_h = 7.0 * health_text_scale;
        let health_text_w = health_text.len() as f32 * health_char_w;
        let health_text_x = health_bar_x + health_bar_w - health_bar_border - health_bar_padding - health_text_w;
        let health_text_y = health_bar_y + (health_bar_h - health_char_h) * 0.5;
        bitmap_font::draw_text_quads(
            &mut verts,
            &health_text,
            health_text_x,
            health_text_y,
            health_text_scale,
            health_text_scale,
            [1.0, 1.0, 1.0, 0.95],
            screen_w,
            screen_h,
        );

        //let fill = [1.0, 1.0, 1.0, 0.10]; // white
        let fill = [0.0, 0.0, 0.0, 0.3]; // dark
        let outline = [1.0, 1.0, 1.0, 0.85];
        //let outline_selected = [1.0, 1.0, 1.0, 1.0];  // white
        //let outline_selected = [0.14, 0.64, 0.98, 1.0]; // blue
        let outline_selected = [1.0, 0.6, 0.0, 1.0]; // yellow-orange

        // Slot content layout
        let slot_padding = 6.0;
        let name_scale = 2.0;
        let name_char_w = 6.0 * name_scale;
        let name_char_h = 7.0 * name_scale;
        let label_gap = 6.0;

        for i in 0..slots {
            let x = start_x + i as f32 * (slot_size + slot_gap);
            let y = start_y;

            // Fill quad
            bitmap_font::push_rect_px(
                &mut verts,
                x,
                y,
                slot_size,
                slot_size,
                fill,
                screen_w,
                screen_h,
            );

            // Outline as 4 thin quads (thicker for selected)
            let thick = if i == selected { 6.0 } else { 3.0 };
            let c = if i == selected { outline_selected } else { outline };
            // top
            bitmap_font::push_rect_px(&mut verts, x, y, slot_size, thick, c, screen_w, screen_h);
            // bottom
            bitmap_font::push_rect_px(
                &mut verts,
                x,
                y + slot_size - thick,
                slot_size,
                thick,
                c,
                screen_w,
                screen_h,
            );
            // left
            bitmap_font::push_rect_px(&mut verts, x, y, thick, slot_size, c, screen_w, screen_h);
            // right
            bitmap_font::push_rect_px(
                &mut verts,
                x + slot_size - thick,
                y,
                thick,
                slot_size,
                c,
                screen_w,
                screen_h,
            );

            // Slot content
            if let Some(stack) = self.player.inventory.get_slot(i) {

                // Block name
                if i == selected {
                    let name = stack.block_type.display_name();
                    let name_color = [1.0, 1.0, 1.0, 0.95];
                    let name_w = (name.chars().count() as f32) * name_char_w;
                    let name_x = x + (slot_size - name_w) * 0.5;
                    let name_y = (y - name_char_h - label_gap).max(0.0);

                    // Background
                    bitmap_font::push_rect_px(
                        &mut verts,
                        name_x - 4.0,
                        name_y - 4.0,
                        name_w + 8.0,
                        name_char_h + 8.0,
                        [0.0, 0.0, 0.0, 0.45],
                        screen_w,
                        screen_h,
                    );

                    bitmap_font::draw_text_quads(
                        &mut verts,
                        name,
                        name_x,
                        name_y,
                        name_scale,
                        name_scale,
                        name_color,
                        screen_w,
                        screen_h,
                    );
                }

                // Block count
                if stack.count > 0.0 {
                    let count_text = format!("{:.2}", stack.count);
                    let count_color = [1.0, 1.0, 1.0, 0.95];
                    let count_scale = 2.0;
                    let count_w = (count_text.chars().count() as f32) * 6.0 * count_scale;
                    let count_x = x + (slot_size - count_w) * 0.5;
                    let count_y = y + slot_size + (3.0 * count_scale);
                    let count_background_padding = 4.0;

                    // Background
                    bitmap_font::push_rect_px(
                        &mut text_verts,
                        count_x - count_background_padding,
                        count_y - count_background_padding,
                        (count_text.len() as f32) * 6.0 * count_scale + (count_background_padding * 2.0),
                        7.0 * count_scale + (count_background_padding * 2.0),
                        [0.0, 0.0, 0.0, 0.55],
                        screen_w,
                        screen_h,
                    );

                    bitmap_font::draw_text_quads(
                        &mut text_verts,
                        &count_text,
                        count_x,
                        count_y,
                        count_scale,
                        count_scale,
                        count_color,
                        screen_w,
                        screen_h,
                    );
                }
            }
        }

        // === Enemy Floating Health Bars ===
        for enemy in &self.enemy_manager.enemies {
            if !enemy.alive { continue; }
            // Only show health bar if enemy has taken damage
            if enemy.health >= enemy.max_health && enemy.death_timer < 0.0 { continue; }

            // Project position above enemy head
            let bar_world_pos = Point3::new(
                enemy.position.x,
                enemy.position.y + enemy.height + 0.6,
                enemy.position.z,
            );
            if let Some((sx, sy)) = self.world_to_screen(bar_world_pos) {
                // Skip if off-screen
                if sx < -100.0 || sx > screen_w + 100.0 || sy < -50.0 || sy > screen_h + 50.0 {
                    continue;
                }

                // Scale bar size based on distance
                let dist = {
                    let dx = enemy.position.x - self.camera.position.x;
                    let dy = enemy.position.y - self.camera.position.y;
                    let dz = enemy.position.z - self.camera.position.z;
                    (dx * dx + dy * dy + dz * dz).sqrt()
                };
                let scale = (24.0 / dist.max(2.0)).clamp(0.3, 4.5);

                let bar_w = 60.0 * scale;
                let bar_h = 12.0 * scale;
                let border = 1.0 * scale;
                let bar_x = sx - bar_w * 0.5;
                let bar_y = sy - bar_h * 0.5;

                // Dark background
                bitmap_font::push_rect_px(
                    &mut verts, bar_x, bar_y, bar_w, bar_h,
                    [0.0, 0.0, 0.0, 0.5], screen_w, screen_h,
                );

                // White border
                let bc = [1.0, 1.0, 1.0, 0.7];
                bitmap_font::push_rect_px(&mut verts, bar_x, bar_y, bar_w, border, bc, screen_w, screen_h);
                bitmap_font::push_rect_px(&mut verts, bar_x, bar_y + bar_h - border, bar_w, border, bc, screen_w, screen_h);
                bitmap_font::push_rect_px(&mut verts, bar_x, bar_y, border, bar_h, bc, screen_w, screen_h);
                bitmap_font::push_rect_px(&mut verts, bar_x + bar_w - border, bar_y, border, bar_h, bc, screen_w, screen_h);

                // Red health fill
                let health_pct = (enemy.health / enemy.max_health).clamp(0.0, 1.0);
                let fill_x = bar_x + border;
                let fill_y = bar_y + border;
                let fill_max_w = bar_w - border * 2.0;
                let fill_h = bar_h - border * 2.0;
                let fill_w = fill_max_w * health_pct;
                if fill_w > 0.0 {
                    bitmap_font::push_rect_px(
                        &mut verts, fill_x, fill_y, fill_w, fill_h,
                        [0.8, 0.1, 0.1, 0.9], screen_w, screen_h,
                    );
                }

                // Health percentage text (only if bar is large enough)
                if scale > 0.5 {
                    let text_scale = 1.0 * scale;
                    let pct_text = format!("{}%", (health_pct * 100.0).round() as u32);
                    let char_w = 6.0 * text_scale;
                    let char_h = 7.0 * text_scale;
                    let text_w = pct_text.len() as f32 * char_w;
                    let text_x = sx - text_w * 0.5;
                    let text_y = bar_y + (bar_h - char_h) * 0.5;
                    bitmap_font::draw_text_quads(
                        &mut verts, &pct_text, text_x, text_y,
                        text_scale, text_scale,
                        [1.0, 1.0, 1.0, 0.95], screen_w, screen_h,
                    );
                }
            }
        }

        // ── Dropped-item pickup hint ──────────────────────────────────────────
        if let Some(idx) = self.hovered_dropped_item {
            if let Some(item) = self.dropped_item_manager.items.get(idx) {
                let name = item.block_type.name();
                let label = format!("Pickup {} (Right-click)", name);
                let scale = 2.0f32;
                let char_w = 6.0 * scale; // 5px glyph + 1px spacing
                let char_h = 7.0 * scale;
                let text_w = label.len() as f32 * char_w;
                let pad = 10.0f32;

                // Position: 24 px to the right of screen centre, vertically centred
                let tx = screen_w * 0.5 + 48.0;
                let ty = screen_h * 0.5 - char_h * 0.5;

                // Dark background
                modal::push_rect_px(
                    &mut verts,
                    tx - pad, ty - pad,
                    tx + text_w + pad, ty + char_h + pad,
                    [0.0, 0.0, 0.0, 0.4],
                    screen_w, screen_h,
                );
                // White text
                bitmap_font::draw_text_quads(
                    &mut verts, &label,
                    tx, ty, scale, scale,
                    [1.0, 1.0, 1.0, 1.0],
                    screen_w, screen_h,
                );
            }
        }

        // ── Crafting Table hover hint ─────────────────────────────────────────
        if self.hovered_crafting_table && !self.crafting_ui_open && self.hovered_dropped_item.is_none() {
            let label = "Open Crafting Table (Right-click)";
            let scale = 2.0f32;
            let char_w = 6.0 * scale;
            let char_h = 7.0 * scale;
            let text_w = label.len() as f32 * char_w;
            let pad = 10.0f32;
            let tx = screen_w * 0.5 + 48.0;
            let ty = screen_h * 0.5 - char_h * 0.5;
            modal::push_rect_px(&mut verts, tx - pad, ty - pad, tx + text_w + pad, ty + char_h + pad,
                [0.0, 0.0, 0.0, 0.4], screen_w, screen_h);
            bitmap_font::draw_text_quads(&mut verts, label, tx, ty, scale, scale,
                [1.0, 1.0, 1.0, 1.0], screen_w, screen_h);
        }

        // ── Crafting Table UI panel ────────────────────────────────────────
        if self.crafting_ui_open {
            let lo = crafting_layout(
                self.crafting_modal.panel_x, self.crafting_modal.panel_y,
                self.crafting_modal.panel_w, self.crafting_modal.panel_h,
            );
            let ct_slot = lo.ct_slot;
            let ct_gap  = lo.ct_gap;

            // ── Modal chrome (border, bevel, title) from the reusable Modal ──
            let chrome = self.crafting_modal.build_ui_vertices(screen_w, screen_h);
            verts.extend_from_slice(&chrome);

            // ── 3×3 Crafting Grid ──
            let qty_scale = 1.5f32;
            let qty_cw    = 6.0 * qty_scale;
            let qty_ch    = 7.0 * qty_scale;
            for row in 0..3usize {
                for col in 0..3usize {
                    let sx = lo.row1_x + col as f32 * (ct_slot + ct_gap);
                    let sy = lo.row1_y + row as f32 * (ct_slot + ct_gap);
                    modal::push_rect_px(&mut verts, sx, sy, sx + ct_slot, sy + ct_slot,
                        [0.08, 0.07, 0.04, 1.0], screen_w, screen_h);
                    let hovered = self.crafting_hovered_grid == Some((row, col));
                    let border_col = if hovered { [1.0, 0.8, 0.1, 1.0] } else { [0.6, 0.55, 0.38, 0.9] };
                    let bw = if hovered { 3.0f32 } else { 2.0f32 };
                    modal::push_rect_px(&mut verts, sx - bw, sy - bw, sx + ct_slot + bw, sy + bw, border_col, screen_w, screen_h);
                    modal::push_rect_px(&mut verts, sx - bw, sy + ct_slot, sx + ct_slot + bw, sy + ct_slot + bw, border_col, screen_w, screen_h);
                    modal::push_rect_px(&mut verts, sx - bw, sy, sx, sy + ct_slot, border_col, screen_w, screen_h);
                    modal::push_rect_px(&mut verts, sx + ct_slot, sy, sx + ct_slot + bw, sy + ct_slot, border_col, screen_w, screen_h);
                    if let Some((_, qty)) = self.crafting_grid.slots[row][col] {
                        let qty_text = format!("{:.2}", qty);
                        let qty_x = sx + ct_slot - qty_text.len() as f32 * qty_cw - 2.0;
                        let qty_y = sy + ct_slot - qty_ch - 2.0;
                        bitmap_font::draw_text_quads(&mut verts, &qty_text, qty_x, qty_y,
                            qty_scale, qty_scale, [1.0, 1.0, 0.6, 1.0], screen_w, screen_h);
                    }
                }
            }

            // ── Arrow ──
            bitmap_font::draw_text_quads(&mut verts, "→", lo.arrow_x, lo.arrow_y,
                lo.arrow_scale, lo.arrow_scale, [0.0, 0.0, 0.0, 0.55], screen_w, screen_h);

            // ── Output slot ──
            let out_has_item = self.crafting_output.is_some();
            let out_bg = if out_has_item { [0.1, 0.14, 0.06, 1.0] } else { [0.06, 0.05, 0.03, 1.0] };
            modal::push_rect_px(&mut verts, lo.out_x, lo.out_y, lo.out_x + ct_slot, lo.out_y + ct_slot, out_bg, screen_w, screen_h);
            let out_border = if self.crafting_hovered_output && out_has_item {
                [0.5, 1.0, 0.3, 1.0]
            } else if out_has_item {
                [0.4, 0.8, 0.2, 0.9]
            } else {
                [0.4, 0.38, 0.25, 0.8]
            };
            let bw = 2.0f32;
            modal::push_rect_px(&mut verts, lo.out_x - bw, lo.out_y - bw, lo.out_x + ct_slot + bw, lo.out_y + bw, out_border, screen_w, screen_h);
            modal::push_rect_px(&mut verts, lo.out_x - bw, lo.out_y + ct_slot, lo.out_x + ct_slot + bw, lo.out_y + ct_slot + bw, out_border, screen_w, screen_h);
            modal::push_rect_px(&mut verts, lo.out_x - bw, lo.out_y, lo.out_x, lo.out_y + ct_slot, out_border, screen_w, screen_h);
            modal::push_rect_px(&mut verts, lo.out_x + ct_slot, lo.out_y, lo.out_x + ct_slot + bw, lo.out_y + ct_slot, out_border, screen_w, screen_h);
            if let Some((_, qty)) = self.crafting_output {
                let qty_text = format!("{:.2}", qty);
                let qty_x = lo.out_x + ct_slot - qty_text.len() as f32 * qty_cw - 2.0;
                let qty_y = lo.out_y + ct_slot - qty_ch - 2.0;
                bitmap_font::draw_text_quads(&mut verts, &qty_text, qty_x, qty_y,
                    qty_scale, qty_scale, [0.6, 1.0, 0.4, 1.0], screen_w, screen_h);
            }

            // ── Inventory row (Row 2) ──
            for i in 0..9usize {
                let sx = lo.row2_x + i as f32 * (ct_slot + ct_gap);
                let sy = lo.row2_y;
                let inv_slot = self.player.inventory.get_slot(i);
                let has_item = inv_slot.is_some();
                modal::push_rect_px(&mut verts, sx, sy, sx + ct_slot, sy + ct_slot,
                    [0.08, 0.07, 0.04, 1.0], screen_w, screen_h);
                let hovered = self.crafting_hovered_inv == Some(i);
                let border_col = if hovered && has_item { [1.0, 0.8, 0.1, 1.0] } else { [0.6, 0.55, 0.38, 0.9] };
                let bw = if hovered && has_item { 3.0f32 } else { 2.0f32 };
                modal::push_rect_px(&mut verts, sx - bw, sy - bw, sx + ct_slot + bw, sy + bw, border_col, screen_w, screen_h);
                modal::push_rect_px(&mut verts, sx - bw, sy + ct_slot, sx + ct_slot + bw, sy + ct_slot + bw, border_col, screen_w, screen_h);
                modal::push_rect_px(&mut verts, sx - bw, sy, sx, sy + ct_slot, border_col, screen_w, screen_h);
                modal::push_rect_px(&mut verts, sx + ct_slot, sy, sx + ct_slot + bw, sy + ct_slot, border_col, screen_w, screen_h);
                if let Some(slot) = inv_slot {
                    let qty_text = format!("{:.2}", slot.count);
                    let qty_x = sx + ct_slot - qty_text.len() as f32 * qty_cw - 2.0;
                    let qty_y = sy + ct_slot - qty_ch - 2.0;
                    bitmap_font::draw_text_quads(&mut verts, &qty_text, qty_x, qty_y,
                        qty_scale, qty_scale, [1.0, 1.0, 0.6, 1.0], screen_w, screen_h);
                }
            }

            // ── Inventory slot tooltip (name + qty near cursor) ──
            // Pushed into text_verts so it renders AFTER item cubes and stays on top.
            if let Some(i) = self.crafting_hovered_inv {
                if let Some(stack) = self.player.inventory.get_slot(i) {
                    let qty = self.crafting_inv_qty[i];
                    let tooltip = format!("{} ({:.2} / {:.2})", stack.block_type.name(), qty, stack.count);
                    let tt_scale = 2.0f32;
                    let tt_cw = 6.0 * tt_scale;
                    let tt_ch = 7.0 * tt_scale;
                    let tt_w = tooltip.len() as f32 * tt_cw;
                    let pad = 6.0f32;
                    let (cx, cy) = self.cursor_pos_px;
                    let tt_x = cx + 16.0;
                    let tt_y = (cy - tt_ch - 4.0).max(0.0);
                    modal::push_rect_px(&mut text_verts, tt_x - pad, tt_y - pad, tt_x + tt_w + pad, tt_y + tt_ch + pad,
                        [0.0, 0.0, 0.0, 0.7], screen_w, screen_h);
                    bitmap_font::draw_text_quads(&mut text_verts, &tooltip, tt_x, tt_y, tt_scale, tt_scale,
                        [1.0, 1.0, 1.0, 1.0], screen_w, screen_h);
                }
            }

            // ── Held item: ghost slot at cursor ──
            if let Some((_, qty)) = self.crafting_held {
                let (cx, cy) = self.cursor_pos_px;
                let hx = cx - ct_slot * 0.5;
                let hy = cy - ct_slot * 0.5;
                modal::push_rect_px(&mut verts, hx, hy, hx + ct_slot, hy + ct_slot,
                    [0.2, 0.18, 0.10, 0.7], screen_w, screen_h);
                let qty_text = format!("{:.2}", qty);
                let qty_x = hx + ct_slot - qty_text.len() as f32 * qty_cw - 2.0;
                let qty_y = hy + ct_slot - qty_ch - 2.0;
                bitmap_font::draw_text_quads(&mut verts, &qty_text, qty_x, qty_y,
                    qty_scale, qty_scale, [1.0, 1.0, 0.6, 1.0], screen_w, screen_h);
            }

        }

        self.hud_vertex_count = verts.len() as u32;
        self.queue
            .write_buffer(&self.hud_vertex_buffer, 0, bytemuck::cast_slice(&verts));

        self.hud_text_vertex_count = text_verts.len() as u32;
        self.queue
            .write_buffer(&self.hud_text_vertex_buffer, 0, bytemuck::cast_slice(&text_verts));
    }

    pub(super) fn rebuild_item_cube_vertices(&mut self) {
        let screen_w = self.size.width as f32;
        let screen_h = self.size.height as f32;

        let slots = self.player.inventory.size.max(1);
        let slot_size = 80.0_f32;
        let slot_gap = 8.0_f32;
        let margin_bottom = 32.0_f32;
        let total_w = (slots as f32) * slot_size + (slots as f32 - 1.0) * slot_gap;
        let start_x = (screen_w - total_w) * 0.5;
        let start_y = screen_h - margin_bottom - slot_size;

        let mut verts: Vec<ItemCubeVertex> = Vec::with_capacity(200);

        for i in 0..slots {
            if let Some(stack) = self.player.inventory.get_slot(i) {
                let x = start_x + i as f32 * (slot_size + slot_gap);
                let cx = x + slot_size * 0.5;
                let cy = start_y + slot_size * 0.5;
                let cube_size = slot_size * 0.65; // cube size relative to slot
                push_item_cube(&mut verts, stack.block_type, cx, cy, cube_size, screen_w, screen_h);
            }
        }

        // ── Crafting UI item cubes ────────────────────────────────────────
        if self.crafting_ui_open {
            let lo = crafting_layout(
                self.crafting_modal.panel_x, self.crafting_modal.panel_y,
                self.crafting_modal.panel_w, self.crafting_modal.panel_h,
            );
            let cube_size = lo.ct_slot * 0.65;

            // 3×3 grid slots
            for row in 0..3usize {
                for col in 0..3usize {
                    if let Some((bt, _)) = self.crafting_grid.slots[row][col] {
                        let cx = lo.row1_x + col as f32 * (lo.ct_slot + lo.ct_gap) + lo.ct_slot * 0.5;
                        let cy = lo.row1_y + row as f32 * (lo.ct_slot + lo.ct_gap) + lo.ct_slot * 0.5;
                        push_item_cube(&mut verts, bt, cx, cy, cube_size, screen_w, screen_h);
                    }
                }
            }

            // Output slot
            if let Some((bt, _)) = self.crafting_output {
                push_item_cube(&mut verts, bt,
                    lo.out_x + lo.ct_slot * 0.5, lo.out_y + lo.ct_slot * 0.5,
                    cube_size, screen_w, screen_h);
            }

            // Inventory row (slots 0-8)
            for i in 0..9usize {
                if let Some(stack) = self.player.inventory.get_slot(i) {
                    let cx = lo.row2_x + i as f32 * (lo.ct_slot + lo.ct_gap) + lo.ct_slot * 0.5;
                    let cy = lo.row2_y + lo.ct_slot * 0.5;
                    push_item_cube(&mut verts, stack.block_type, cx, cy, cube_size, screen_w, screen_h);
                }
            }

            // Held item (follows cursor)
            if let Some((bt, _)) = self.crafting_held {
                let (cx, cy) = self.cursor_pos_px;
                push_item_cube(&mut verts, bt, cx, cy, cube_size, screen_w, screen_h);
            }
        }

        self.item_cube_vertex_count = verts.len() as u32;
        self.queue
            .write_buffer(&self.item_cube_vertex_buffer, 0, bytemuck::cast_slice(&verts));
    }
}

// Push the three visible isometric faces of a block cube icon into 'verts'
pub(super) fn push_item_cube(
    verts: &mut Vec<ItemCubeVertex>,
    block_type: BlockType,
    cx: f32,
    cy: f32,
    cube_size: f32,
    screen_w: f32,
    screen_h: f32,
) {
    let face_tex = block_type.get_face_textures(false);
    let bc = block_type.get_color();

    // Horizontal scale: cube_size maps to 40 isometric units wide
    let kx = cube_size / 40.0;
    // Vertical scale: 1.1× taller than the mathematically-correct square so the cube doesn't look squat (the hex waist creates a "short" illusion).
    let ky = kx * 1.1;

    // ── 7 key 2D hex points (pixel space, y increases downward) ──────────
    // Derived from 2:1 isometric with a vertical stretch applied.
    // Width = 40·kx,  Height = 40·ky,  visual center at (cx, cy).
    let tbl = [cx,           cy - 20.0 * ky]; // top vertex
    let tfl = [cx - 20.0*kx, cy - 10.0 * ky]; // upper-left
    let tbr = [cx + 20.0*kx, cy - 10.0 * ky]; // upper-right
    let tfr = [cx,           cy             ]; // hex center (inner divider)
    let fl  = [cx - 20.0*kx, cy + 10.0 * ky]; // lower-left
    let br  = [cx + 20.0*kx, cy + 10.0 * ky]; // lower-right
    let fr  = [cx,           cy + 20.0 * ky]; // bottom vertex

    // Pixel-to-clip-space conversion (y flipped: screen y-down → clip y-up)
    let p2c = |p: [f32; 2]| -> [f32; 2] {
        [2.0 * p[0] / screen_w - 1.0, 1.0 - 2.0 * p[1] / screen_h]
    };

    // Atlas UV for a given tile index
    let tile_uv = |tile: u32| -> (f32, f32, f32, f32) {
        let col = (tile % 16) as f32;
        let row = (tile / 16) as f32;
        let u0 = col / 16.0;
        let v0 = row / 16.0;
        (u0, v0, u0 + 1.0 / 16.0, v0 + 1.0 / 16.0)
    };

    // Build one ItemCubeVertex
    let make_v = |pos: [f32; 2], uv: [f32; 2], shade: f32, tile: u32| -> ItemCubeVertex {
        let use_texture = if tile != TEX_NONE { 1.0_f32 } else { 0.0_f32 };
        let color = if tile != TEX_NONE {
            [shade, shade, shade, 1.0]
        } else {
            [bc[0] * shade, bc[1] * shade, bc[2] * shade, 1.0]
        };
        ItemCubeVertex {
            position: p2c(pos),
            uv,
            color,
            use_texture,
            _pad: 0.0,
        }
    };

    // Push a quad (4 corners CCW) as two triangles: (0,1,2) and (0,2,3)
    let mut push_quad = |v0: ItemCubeVertex, v1: ItemCubeVertex, v2: ItemCubeVertex, v3: ItemCubeVertex| {
        verts.push(v0);
        verts.push(v1);
        verts.push(v2);
        verts.push(v0);
        verts.push(v2);
        verts.push(v3);
    };

    // ── TOP FACE (shade 1.0, uses face_tex.top) ───────────────────────────
    // Corners: tfl, tbl, tbr, tfr
    // UV (looking down from above — x right, z toward viewer):
    //   tfl (front-left)  → (u0, v1)
    //   tbl (back-left)   → (u0, v0)
    //   tbr (back-right)  → (u1, v0)
    //   tfr (front-right) → (u1, v1)
    {
        let ti = face_tex.top;
        let (u0, v0, u1, v1) = tile_uv(ti);
        let shade = 1.0;
        push_quad(
            make_v(tfl, [u0, v1], shade, ti),
            make_v(tbl, [u0, v0], shade, ti),
            make_v(tbr, [u1, v0], shade, ti),
            make_v(tfr, [u1, v1], shade, ti),
        );
    }

    // ── LEFT FACE — front-left (Z+ face, shade 0.75, uses face_tex.sides) ─
    // Corners: tfl, tfr, fr, fl
    // UV (looking at Z+ face from outside — x right, y up):
    //   tfl → (u0, v0)  tfr → (u1, v0)  fr → (u1, v1)  fl → (u0, v1)
    {
        let ti = face_tex.sides;
        let (u0, v0, u1, v1) = tile_uv(ti);
        let shade = 0.75;
        push_quad(
            make_v(tfl, [u0, v0], shade, ti),
            make_v(tfr, [u1, v0], shade, ti),
            make_v(fr,  [u1, v1], shade, ti),
            make_v(fl,  [u0, v1], shade, ti),
        );
    }

    // ── RIGHT FACE (X+ face, shade 0.60, uses face_tex.sides) ────────────
    // Corners: tfr, tbr, br, fr
    // UV (looking at X+ face from outside — z-axis "left", y up):
    //   tfr → (u0, v0)  tbr → (u1, v0)  br → (u1, v1)  fr → (u0, v1)
    {
        let ti = face_tex.sides;
        let (u0, v0, u1, v1) = tile_uv(ti);
        let shade = 0.60;
        push_quad(
            make_v(tfr, [u0, v0], shade, ti),
            make_v(tbr, [u1, v0], shade, ti),
            make_v(br,  [u1, v1], shade, ti),
            make_v(fr,  [u0, v1], shade, ti),
        );
    }
}
