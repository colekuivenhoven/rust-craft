use super::*;

impl State {
    /// Generate line vertices for chunk boundary outlines
    pub(super) fn build_chunk_outline_vertices(&self) -> Vec<LineVertex> {
        let mut vertices = Vec::new();

        for (&(cx, cz), _chunk) in &self.world.chunks {
            // World coordinates of chunk boundaries
            let x0 = (cx * CHUNK_SIZE as i32) as f32;
            let x1 = x0 + CHUNK_SIZE as f32;
            let z0 = (cz * CHUNK_SIZE as i32) as f32;
            let z1 = z0 + CHUNK_SIZE as f32;
            let y0 = 0.0f32;
            let y1 = CHUNK_HEIGHT as f32;

            // Vertical edges (4 corners)
            vertices.extend_from_slice(&[
                LineVertex { position: [x0, y0, z0] }, LineVertex { position: [x0, y1, z0] },
                LineVertex { position: [x1, y0, z0] }, LineVertex { position: [x1, y1, z0] },
                LineVertex { position: [x0, y0, z1] }, LineVertex { position: [x0, y1, z1] },
                LineVertex { position: [x1, y0, z1] }, LineVertex { position: [x1, y1, z1] },
            ]);

            // Bottom edges
            vertices.extend_from_slice(&[
                LineVertex { position: [x0, y0, z0] }, LineVertex { position: [x1, y0, z0] },
                LineVertex { position: [x1, y0, z0] }, LineVertex { position: [x1, y0, z1] },
                LineVertex { position: [x1, y0, z1] }, LineVertex { position: [x0, y0, z1] },
                LineVertex { position: [x0, y0, z1] }, LineVertex { position: [x0, y0, z0] },
            ]);

            // Top edges
            vertices.extend_from_slice(&[
                LineVertex { position: [x0, y1, z0] }, LineVertex { position: [x1, y1, z0] },
                LineVertex { position: [x1, y1, z0] }, LineVertex { position: [x1, y1, z1] },
                LineVertex { position: [x1, y1, z1] }, LineVertex { position: [x0, y1, z1] },
                LineVertex { position: [x0, y1, z1] }, LineVertex { position: [x0, y1, z0] },
            ]);
        }

        vertices
    }

    pub(super) fn build_debug_axes(&self, verts: &mut Vec<UiVertex>, screen_w: f32, screen_h: f32) {
        // Debug axes gizmo in top-right corner
        // Colors: X = Red, Y = Green, Z = Blue (R-G-B matches X-Y-Z alphabetically)
        let margin = 60.0;
        let center_x = screen_w - margin;
        let center_y = margin;
        let axis_length = 40.0;
        let line_thickness = 5.0;

        // Get camera rotation (yaw and pitch)
        let yaw = self.camera.yaw.0;
        let pitch = self.camera.pitch.0;

        // Create rotation matrix (view rotation)
        // We need to transform world axes by the inverse of the camera's view rotation
        // to show how world axes appear from the camera's perspective
        let cos_yaw = yaw.cos();
        let sin_yaw = yaw.sin();
        let cos_pitch = pitch.cos();
        let sin_pitch = pitch.sin();

        // View rotation matrix (rotates world into view space)
        // First rotate around Y (yaw), then around X (pitch)
        // The result shows world axes as seen from camera
        let rotate_world_to_view = |world_axis: Vector3<f32>| -> (f32, f32) {
            // Apply yaw rotation (around Y axis)
            let x1 = world_axis.x * cos_yaw + world_axis.z * sin_yaw;
            let y1 = world_axis.y;
            let z1 = -world_axis.x * sin_yaw + world_axis.z * cos_yaw;

            // Apply pitch rotation (around X axis)
            let x2 = x1;
            let y2 = y1 * cos_pitch - z1 * sin_pitch;
            let _z2 = y1 * sin_pitch + z1 * cos_pitch;

            // Project to 2D (x2 is right, y2 is up in screen space)
            (x2, -y2) // Negate y because screen Y goes down
        };

        // World axis directions
        let x_axis = Vector3::new(1.0, 0.0, 0.0);
        let y_axis = Vector3::new(0.0, 1.0, 0.0);
        let z_axis = Vector3::new(0.0, 0.0, 1.0);

        // Transform to screen space
        let (x_screen_x, x_screen_y) = rotate_world_to_view(x_axis);
        let (y_screen_x, y_screen_y) = rotate_world_to_view(y_axis);
        let (z_screen_x, z_screen_y) = rotate_world_to_view(z_axis);

        // Helper to draw a line as a quad from center to endpoint
        let draw_axis_line = |verts: &mut Vec<UiVertex>, dx: f32, dy: f32, color: [f32; 4]| {
            let end_x = center_x + dx * axis_length;
            let end_y = center_y + dy * axis_length;

            // Calculate perpendicular for line thickness
            let len = (dx * dx + dy * dy).sqrt().max(0.001);
            let perp_x = -dy / len * line_thickness * 0.5;
            let perp_y = dx / len * line_thickness * 0.5;

            // Create quad vertices (two triangles)
            let p0 = (center_x + perp_x, center_y + perp_y);
            let p1 = (center_x - perp_x, center_y - perp_y);
            let p2 = (end_x - perp_x, end_y - perp_y);
            let p3 = (end_x + perp_x, end_y + perp_y);

            // Convert to clip space
            let to_clip = |px: f32, py: f32| -> [f32; 2] {
                [
                    (px / screen_w) * 2.0 - 1.0,
                    1.0 - (py / screen_h) * 2.0,
                ]
            };

            let v0 = UiVertex { position: to_clip(p0.0, p0.1), color };
            let v1 = UiVertex { position: to_clip(p1.0, p1.1), color };
            let v2 = UiVertex { position: to_clip(p2.0, p2.1), color };
            let v3 = UiVertex { position: to_clip(p3.0, p3.1), color };

            // Two triangles
            verts.push(v0);
            verts.push(v1);
            verts.push(v2);
            verts.push(v0);
            verts.push(v2);
            verts.push(v3);
        };

        // Draw background
        let bg_color = [0.0, 0.0, 0.0, 0.5];
        let bg_radius = axis_length + 15.0;
        bitmap_font::push_rect_px(
            verts,
            center_x - bg_radius,
            center_y - bg_radius,
            bg_radius * 2.0,
            bg_radius * 2.0,
            bg_color,
            screen_w,
            screen_h,
        );

        // Draw axes (R-G-B for X-Y-Z)
        let red = [1.0, 0.2, 0.2, 1.0];
        let green = [0.2, 1.0, 0.2, 1.0];
        let blue = [0.3, 0.5, 1.0, 1.0];

        draw_axis_line(verts, x_screen_x, x_screen_y, red);   // X axis - Red
        draw_axis_line(verts, y_screen_x, y_screen_y, green); // Y axis - Green
        draw_axis_line(verts, z_screen_x, z_screen_y, blue);  // Z axis - Blue

        // Draw axis labels
        let label_offset = axis_length + 8.0;
        let label_scale = 1.5;

        // X label
        let x_label_x = center_x + x_screen_x * label_offset - 4.0;
        let x_label_y = center_y + x_screen_y * label_offset - 5.0;
        bitmap_font::draw_text_quads(verts, "X", x_label_x, x_label_y, label_scale, label_scale, red, screen_w, screen_h);

        // Y label
        let y_label_x = center_x + y_screen_x * label_offset - 4.0;
        let y_label_y = center_y + y_screen_y * label_offset - 5.0;
        bitmap_font::draw_text_quads(verts, "Y", y_label_x, y_label_y, label_scale, label_scale, green, screen_w, screen_h);

        // Z label
        let z_label_x = center_x + z_screen_x * label_offset - 4.0;
        let z_label_y = center_y + z_screen_y * label_offset - 5.0;
        bitmap_font::draw_text_quads(verts, "Z", z_label_x, z_label_y, label_scale, label_scale, blue, screen_w, screen_h);
    }
}
