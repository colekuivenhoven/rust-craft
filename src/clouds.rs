use std::collections::HashMap;
use cgmath::Point3;
use noise::{NoiseFn, Perlin};
use crate::block::Vertex;
use crate::config::CloudConfig;

/// Number of cloud pixels per cloud chunk side.
const CLOUD_CHUNK_PIXELS: i32 = 32;

/// A single cloud chunk: pre-built geometry in cloud-space (no drift baked in).
struct CloudChunk {
    vertices: Vec<Vertex>,
    indices:  Vec<u32>,
}

pub struct CloudManager {
    noise:    Perlin,
    chunks:   HashMap<(i32, i32), CloudChunk>,
    config:   CloudConfig,
    drift_x:  f64,  // Accumulated X drift in world units
    drift_z:  f64,  // Accumulated Z drift in world units
    render_distance: i32,

    // Cached combined geometry — rebuilt only when chunks load/unload, not every frame.
    combined_vertices: Vec<Vertex>,
    combined_indices:  Vec<u32>,
    /// True when combined geometry was rebuilt this frame — signals renderer to re-upload.
    geometry_rebuilt: bool,
}

impl CloudManager {
    pub fn new(render_distance: i32, config: CloudConfig) -> Self {
        Self {
            noise: Perlin::new(12345),
            chunks: HashMap::new(),
            config,
            drift_x: 0.0,
            drift_z: 0.0,
            render_distance,
            combined_vertices: Vec::new(),
            combined_indices:  Vec::new(),
            geometry_rebuilt: true, // Force initial GPU upload on first frame
        }
    }

    /// Advance drift, load/unload cloud chunks around the player.
    /// Sets `geometry_rebuilt` if the chunk set changed this frame.
    pub fn update(&mut self, player_pos: Point3<f32>, dt: f32, render_distance: i32) {
        self.render_distance = render_distance;
        self.geometry_rebuilt = false;

        // Advance drift in world units per second
        self.drift_x += dt as f64 * self.config.noise_offset_change_speed;
        self.drift_z += dt as f64 * self.config.noise_offset_change_speed * 0.3;

        let chunk_world_size = self.chunk_world_size();

        // Player position in cloud-space (subtract drift so the noise grid is stable)
        let cloud_px = (player_pos.x as f64 - self.drift_x) / chunk_world_size;
        let cloud_pz = (player_pos.z as f64 - self.drift_z) / chunk_world_size;
        let pcx = cloud_px.floor() as i32;
        let pcz = cloud_pz.floor() as i32;

        let terrain_radius = self.render_distance as f64 * 32.0;
        let crd = (terrain_radius / chunk_world_size).ceil() as i32 + 1;

        // Load missing chunks
        for cx in (pcx - crd)..=(pcx + crd) {
            for cz in (pcz - crd)..=(pcz + crd) {
                if !self.chunks.contains_key(&(cx, cz)) {
                    let chunk = self.build_chunk(cx, cz);
                    self.chunks.insert((cx, cz), chunk);
                    self.geometry_rebuilt = true;
                }
            }
        }

        // Unload distant chunks
        let before = self.chunks.len();
        let unload_dist = crd + 2;
        self.chunks.retain(|&(cx, cz), _| {
            (cx - pcx).abs() <= unload_dist && (cz - pcz).abs() <= unload_dist
        });
        if self.chunks.len() != before {
            self.geometry_rebuilt = true;
        }

        // Rebuild combined CPU buffer only when the chunk set changed
        if self.geometry_rebuilt {
            self.rebuild_combined();
        }
    }

    fn chunk_world_size(&self) -> f64 {
        CLOUD_CHUNK_PIXELS as f64 * self.config.pixel_size as f64
    }

    /// Build geometry for one cloud chunk. Vertex positions are in cloud-space with no drift.
    fn build_chunk(&self, cx: i32, cz: i32) -> CloudChunk {
        let mut vertices: Vec<Vertex> = Vec::new();
        let mut indices:  Vec<u32>   = Vec::new();

        let chunk_world_size = self.chunk_world_size();
        let start_x = cx as f64 * chunk_world_size;
        let start_z = cz as f64 * chunk_world_size;
        let ps = self.config.pixel_size as f64;

        for px in 0..CLOUD_CHUNK_PIXELS {
            for pz in 0..CLOUD_CHUNK_PIXELS {
                let nx = start_x + px as f64 * ps;
                let nz = start_z + pz as f64 * ps;

                let noise_val = self.noise.get([
                    nx * self.config.noise_scale,
                    nz * self.config.noise_scale,
                ]);
                let noise_normalized = (noise_val + 1.0) * 0.5;

                if noise_normalized > self.config.threshold {
                    add_cloud_pixel(
                        &mut vertices,
                        &mut indices,
                        nx as f32,
                        nz as f32,
                        self.config.pixel_size,
                        self.config.height,
                    );
                }
            }
        }

        CloudChunk { vertices, indices }
    }

    /// Rebuild the flat combined vertex/index buffers from all loaded chunks.
    fn rebuild_combined(&mut self) {
        self.combined_vertices.clear();
        self.combined_indices.clear();

        for chunk in self.chunks.values() {
            if chunk.vertices.is_empty() {
                continue;
            }
            let base = self.combined_vertices.len() as u32;
            self.combined_vertices.extend_from_slice(&chunk.vertices);
            for &idx in &chunk.indices {
                self.combined_indices.push(base + idx);
            }
        }
    }

    /// Returns the cached combined geometry. Always valid after `update()`.
    pub fn geometry(&self) -> (&[Vertex], &[u32]) {
        (&self.combined_vertices, &self.combined_indices)
    }

    /// True if the combined geometry was rebuilt this frame — renderer should re-upload to GPU.
    pub fn geometry_rebuilt(&self) -> bool {
        self.geometry_rebuilt
    }

    /// Current drift in world units. The vertex shader adds this to vertex X/Z positions.
    pub fn get_drift(&self) -> (f32, f32) {
        (self.drift_x as f32, self.drift_z as f32)
    }

    pub fn index_count(&self) -> u32 {
        self.combined_indices.len() as u32
    }
}

/// Emit a single two-sided cloud pixel quad into the mesh buffers.
fn add_cloud_pixel(
    vertices: &mut Vec<Vertex>,
    indices:  &mut Vec<u32>,
    x: f32,
    z: f32,
    pixel_size: f32,
    y: f32,
) {
    let s = pixel_size;
    let color = [1.0f32, 1.0, 1.0];
    let light = 1.0f32;
    let alpha = 0.5f32;

    // Bottom face (normal pointing down, visible from below)
    let b = vertices.len() as u32;
    for &(px, pz) in &[(x, z), (x + s, z), (x + s, z + s), (x, z + s)] {
        vertices.push(Vertex {
            position: [px, y, pz],
            color,
            normal: [0.0, -1.0, 0.0],
            light_level: light,
            alpha,
            uv: [0.0, 0.0],
            tex_index: 255,
            ao: 1.0,
        });
    }
    indices.extend_from_slice(&[b, b + 1, b + 2, b + 2, b + 3, b]);

    // Top face (normal pointing up, visible from above)
    let b = vertices.len() as u32;
    for &(px, pz) in &[(x, z), (x + s, z), (x + s, z + s), (x, z + s)] {
        vertices.push(Vertex {
            position: [px, y, pz],
            color,
            normal: [0.0, 1.0, 0.0],
            light_level: light,
            alpha,
            uv: [0.0, 0.0],
            tex_index: 255,
            ao: 1.0,
        });
    }
    indices.extend_from_slice(&[b, b + 3, b + 2, b + 2, b + 1, b]);
}
