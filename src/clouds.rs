use cgmath::Point3;
use noise::{NoiseFn, Perlin};
use crate::block::Vertex;
use crate::config::CloudConfig;

/// Chunk size (must match chunk.rs)
const CHUNK_SIZE: f32 = 32.0;

pub struct CloudManager {
    noise: Perlin,
    vertices: Vec<Vertex>,
    indices: Vec<u16>,
    render_distance: i32,
    plane_size: f32,
    config: CloudConfig,
    time_offset: f64,
}

impl CloudManager {
    pub fn new(render_distance: i32, config: CloudConfig) -> Self {
        let plane_size = render_distance as f32 * CHUNK_SIZE * 3.0;
        Self {
            noise: Perlin::new(12345), // Fixed seed for consistent clouds
            vertices: Vec::new(),
            indices: Vec::new(),
            render_distance,
            plane_size,
            config,
            time_offset: 0.0,
        }
    }

    /// Update cloud plane to center on player position
    pub fn update(&mut self, player_pos: Point3<f32>, _render_distance: i32) {
        self.vertices.clear();
        self.indices.clear();

        // Increment time offset for cloud animation
        self.time_offset += self.config.noise_offset_change_speed;

        // Calculates plane bounds centered on player
        let half_size = self.plane_size / 3.0;
        let start_x = player_pos.x - half_size;
        let start_z = player_pos.z - half_size;

        // Calculate how many pixels fit in the plane
        let pixels_per_side = (self.plane_size / self.config.pixel_size) as i32;

        // Apply time-based noise offset for cloud movement
        let noise_offset_x = self.time_offset;
        let noise_offset_z = self.time_offset * 0.7; // Use different speed for Z to create more natural drift

        // Generate pixelated cloud plane using 2D noise
        for px in 0..pixels_per_side {
            for pz in 0..pixels_per_side {
                let world_x = start_x + px as f32 * self.config.pixel_size;
                let world_z = start_z + pz as f32 * self.config.pixel_size;

                // Sample 2D noise using absolute world coordinates with offset
                let noise_val = self.noise.get([
                    (world_x as f64 + noise_offset_x) * self.config.noise_scale,
                    (world_z as f64 + noise_offset_z) * self.config.noise_scale,
                ]);

                // Convert noise from [-1, 1] to [0, 1]
                let noise_normalized = (noise_val + 1.0) * 0.5;

                // Only generate cloud pixels above threshold
                if noise_normalized > self.config.threshold {
                    // Use noise value to determine alpha (more noise = more opaque)
                    let alpha = (noise_normalized * 0.7 + 0.3) as f32; // Range: 0.3 to 1.0

                    self.add_cloud_pixel(world_x, world_z, alpha);
                }
            }
        }
    }

    /// Add a single cloud pixel as a double-sided flat quad
    fn add_cloud_pixel(&mut self, x: f32, z: f32, alpha: f32) {
        let size = self.config.pixel_size;
        let y = self.config.height;
        let color = [1.0, 1.0, 1.0]; // White clouds
        let light = 1.0; // Fully lit

        // Bottom face (visible from below) - normal pointing down
        let base_idx = self.vertices.len() as u16;

        self.vertices.push(Vertex {
            position: [x, y, z],
            color,
            normal: [0.0, -1.0, 0.0],
            light_level: light,
            alpha,
            uv: [0.0, 0.0],
            tex_index: 255,
            ao: 1.0,
        });

        self.vertices.push(Vertex {
            position: [x + size, y, z],
            color,
            normal: [0.0, -1.0, 0.0],
            light_level: light,
            alpha,
            uv: [0.0, 0.0],
            tex_index: 255,
            ao: 1.0,
        });

        self.vertices.push(Vertex {
            position: [x + size, y, z + size],
            color,
            normal: [0.0, -1.0, 0.0],
            light_level: light,
            alpha,
            uv: [0.0, 0.0],
            tex_index: 255,
            ao: 1.0,
        });

        self.vertices.push(Vertex {
            position: [x, y, z + size],
            color,
            normal: [0.0, -1.0, 0.0],
            light_level: light,
            alpha,
            uv: [0.0, 0.0],
            tex_index: 255,
            ao: 1.0,
        });

        // Two triangles for bottom face
        self.indices.extend_from_slice(&[
            base_idx, base_idx + 1, base_idx + 2,
            base_idx + 2, base_idx + 3, base_idx,
        ]);

        // Top face (visible from above) - normal pointing up
        let base_idx_top = self.vertices.len() as u16;

        self.vertices.push(Vertex {
            position: [x, y, z],
            color,
            normal: [0.0, 1.0, 0.0],
            light_level: light,
            alpha,
            uv: [0.0, 0.0],
            tex_index: 255,
            ao: 1.0,
        });

        self.vertices.push(Vertex {
            position: [x + size, y, z],
            color,
            normal: [0.0, 1.0, 0.0],
            light_level: light,
            alpha,
            uv: [0.0, 0.0],
            tex_index: 255,
            ao: 1.0,
        });

        self.vertices.push(Vertex {
            position: [x + size, y, z + size],
            color,
            normal: [0.0, 1.0, 0.0],
            light_level: light,
            alpha,
            uv: [0.0, 0.0],
            tex_index: 255,
            ao: 1.0,
        });

        self.vertices.push(Vertex {
            position: [x, y, z + size],
            color,
            normal: [0.0, 1.0, 0.0],
            light_level: light,
            alpha,
            uv: [0.0, 0.0],
            tex_index: 255,
            ao: 1.0,
        });

        // Two triangles for top face (reversed winding for correct facing)
        self.indices.extend_from_slice(&[
            base_idx_top, base_idx_top + 3, base_idx_top + 2,
            base_idx_top + 2, base_idx_top + 1, base_idx_top,
        ]);
    }

    /// Get cloud plane vertices and indices for rendering
    pub fn get_geometry(&self) -> (&[Vertex], &[u16]) {
        (&self.vertices, &self.indices)
    }
}
