use crate::chunk::{Chunk, ChunkNeighbors, CHUNK_SIZE, CHUNK_HEIGHT};
use crate::chunk_loader::ChunkLoader;
use crate::chunk_storage;
use crate::block::BlockType;
use crate::lighting;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

//* —— Rebuild/Recal Variables —————————————————————————————————————————————————————————————————————
pub const MAX_LIGHT_RECALCS: usize = 2;
pub const MAX_MESH_REBUILDS: usize = 4;

pub struct World {
    pub chunks: HashMap<(i32, i32), Chunk>,
    render_distance: i32,
    chunk_loader: ChunkLoader,
    last_center: (i32, i32),
    master_seed: u32,
    terrain_cfg: Arc<crate::config::TerrainConfig>,
}

impl World {
    pub fn new(render_distance: i32, master_seed: u32, terrain_cfg: Arc<crate::config::TerrainConfig>) -> Self {
        let mut world = Self {
            chunks: HashMap::new(),
            render_distance,
            chunk_loader: ChunkLoader::new(master_seed, terrain_cfg.clone()),
            last_center: (0, 0),
            master_seed,
            terrain_cfg,
        };
        // Generate initial chunks synchronously for immediate spawn
        world.generate_initial_chunks(0, 0);
        world
    }

    pub fn get_render_distance(&self) -> i32 {
        self.render_distance
    }

    /// Generate initial chunks synchronously (only used at startup for immediate spawn)
    fn generate_initial_chunks(&mut self, center_x: i32, center_z: i32) {
        // Only generate a small area synchronously for immediate playability
        let initial_radius = 2.min(self.render_distance);

        let positions: Vec<(i32, i32)> = (-initial_radius..=initial_radius)
            .flat_map(|x| {
                (-initial_radius..=initial_radius).map(move |z| (center_x + x, center_z + z))
            })
            .collect();

        // Generate chunks in parallel using rayon
        let master_seed = self.master_seed;
        let terrain_cfg = self.terrain_cfg.clone();
        let generated_chunks: Vec<((i32, i32), Chunk)> = positions
            .par_iter()
            .map(|&(cx, cz)| {
                let mut chunk = if let Some(loaded) = chunk_storage::load_chunk(cx, cz) {
                    loaded
                } else {
                    Chunk::new(cx, cz, master_seed, &terrain_cfg)
                };
                lighting::calculate_chunk_lighting(&mut chunk);
                ((cx, cz), chunk)
            })
            .collect();

        let new_chunk_positions: Vec<(i32, i32)> = generated_chunks
            .iter()
            .map(|(pos, _)| *pos)
            .collect();

        for (pos, chunk) in generated_chunks {
            self.chunks.insert(pos, chunk);
        }

        // Mark neighbors dirty (cardinals + diagonals, so corner smooth-lighting samples refresh)
        for (cx, cz) in &new_chunk_positions {
            for (dx, dz) in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)] {
                let neighbor_pos = (cx + dx, cz + dz);
                if let Some(neighbor) = self.chunks.get_mut(&neighbor_pos) {
                    neighbor.dirty = true;
                }
            }
        }

        self.propagate_cross_chunk_lighting(&new_chunk_positions);
        self.last_center = (center_x, center_z);
    }

    /// Queue missing chunks to be loaded in the background
    fn queue_missing_chunks(&mut self, center_x: i32, center_z: i32) {
        for dx in -self.render_distance..=self.render_distance {
            for dz in -self.render_distance..=self.render_distance {
                let pos = (center_x + dx, center_z + dz);

                // Skip if already loaded or already pending
                if self.chunks.contains_key(&pos) || self.chunk_loader.is_pending(&pos) {
                    continue;
                }

                // Calculate priority based on distance (squared)
                let priority = (dx * dx + dz * dz) as f32;
                self.chunk_loader.request_chunk(pos, priority);
            }
        }
    }

    /// Process completed chunks from the background loader
    fn receive_loaded_chunks(&mut self) -> Vec<(i32, i32)> {
        let results = self.chunk_loader.receive_chunks();
        let mut new_positions = Vec::new();

        for result in results {
            let pos = result.position;

            // Don't insert if it's now outside render distance
            let (cx, cz) = self.last_center;
            if (pos.0 - cx).abs() > self.render_distance || (pos.1 - cz).abs() > self.render_distance {
                continue;
            }

            self.chunks.insert(pos, result.chunk);
            new_positions.push(pos);
        }

        // Mark neighbors dirty for new chunks (cardinals + diagonals, so corner smooth-lighting samples refresh)
        for (cx, cz) in &new_positions {
            for (dx, dz) in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)] {
                let neighbor_pos = (cx + dx, cz + dz);
                if let Some(neighbor) = self.chunks.get_mut(&neighbor_pos) {
                    neighbor.dirty = true;
                }
            }
        }

        new_positions
    }

    pub fn update_chunks(&mut self, camera_pos: (f32, f32)) {
        let chunk_x = (camera_pos.0 / CHUNK_SIZE as f32).floor() as i32;
        let chunk_z = (camera_pos.1 / CHUNK_SIZE as f32).floor() as i32;
        self.last_center = (chunk_x, chunk_z);

        // Unload chunks outside render distance (queue saves in background)
        let chunks_to_unload: Vec<(i32, i32)> = self.chunks
            .iter()
            .filter(|(&(cx, cz), _)| {
                (cx - chunk_x).abs() > self.render_distance
                    || (cz - chunk_z).abs() > self.render_distance
            })
            .map(|(&pos, _)| pos)
            .collect();

        for pos in chunks_to_unload {
            if let Some(chunk) = self.chunks.remove(&pos) {
                if chunk.modified {
                    // Queue save in background thread
                    self.chunk_loader.queue_save(pos, chunk.blocks, chunk.modified);
                }
            }
        }

        // Queue missing chunks to load in background
        self.queue_missing_chunks(chunk_x, chunk_z);

        // Receive completed chunks (limited per frame)
        let new_positions = self.receive_loaded_chunks();

        // Lightweight light propagation for each new chunk
        // Only blends light with immediate neighbors, not all chunks
        for pos in new_positions {
            self.propagate_light_for_chunk(pos);
        }
    }

    pub fn get_block_world(&self, x: i32, y: i32, z: i32) -> BlockType {
        if y < 0 || y >= crate::chunk::CHUNK_HEIGHT as i32 {
            return BlockType::Air;
        }

        let chunk_x = x.div_euclid(CHUNK_SIZE as i32);
        let chunk_z = z.div_euclid(CHUNK_SIZE as i32);
        let local_x = x.rem_euclid(CHUNK_SIZE as i32) as usize;
        let local_z = z.rem_euclid(CHUNK_SIZE as i32) as usize;

        if let Some(chunk) = self.chunks.get(&(chunk_x, chunk_z)) {
            chunk.get_block(local_x, y as usize, local_z)
        } else {
            BlockType::Air
        }
    }

    pub fn set_block_world(&mut self, x: i32, y: i32, z: i32, block_type: BlockType) {
        if y < 0 || y >= crate::chunk::CHUNK_HEIGHT as i32 {
            return;
        }

        let chunk_x = x.div_euclid(CHUNK_SIZE as i32);
        let chunk_z = z.div_euclid(CHUNK_SIZE as i32);
        let local_x = x.rem_euclid(CHUNK_SIZE as i32) as usize;
        let local_z = z.rem_euclid(CHUNK_SIZE as i32) as usize;

        // 1. Update the block in the specific chunk
        // set_block() calls on_block_removed/on_block_placed for incremental lighting
        // and marks dirty=true for mesh rebuild. No need for full light recalc.
        if let Some(chunk) = self.chunks.get_mut(&(chunk_x, chunk_z)) {
            chunk.set_block(local_x, y as usize, local_z, block_type);
        }

        // 2. Mark immediate neighbor chunks as dirty (mesh rebuild only) if block
        // is within 2 blocks of the chunk edge. The incremental lighting update
        // already handled most light changes; neighbors just need mesh updates
        // for correct face culling at boundaries.
        let edge_distance = 2;

        let min_dx = if local_x < edge_distance { -1 } else { 0 };
        let max_dx = if local_x >= CHUNK_SIZE - edge_distance { 1 } else { 0 };

        let min_dz = if local_z < edge_distance { -1 } else { 0 };
        let max_dz = if local_z >= CHUNK_SIZE - edge_distance { 1 } else { 0 };

        for dx in min_dx..=max_dx {
            for dz in min_dz..=max_dz {
                if dx == 0 && dz == 0 {
                    continue;
                }

                if let Some(neighbor) = self.chunks.get_mut(&(chunk_x + dx, chunk_z + dz)) {
                    neighbor.dirty = true;
                }
            }
        }

        // 3. Lightweight cross-chunk light propagation for the modified chunk only
        self.propagate_light_for_chunk((chunk_x, chunk_z));
    }

    pub fn rebuild_dirty_chunks(&mut self, smooth_lighting: bool) {
        // Recalculate lighting only for chunks explicitly marked light_dirty
        // (e.g., from new chunk loading, NOT from single block changes which use
        // incremental updates via on_block_removed/on_block_placed)
        let light_dirty_positions: Vec<(i32, i32)> = self.chunks.iter()
            .filter(|(_, c)| c.light_dirty)
            .map(|(&pos, _)| pos)
            .collect();

        // Limit full light recalcs per frame to avoid stuttering
        for pos in light_dirty_positions.iter().take(MAX_LIGHT_RECALCS) {
            if let Some(chunk) = self.chunks.get_mut(pos) {
                lighting::calculate_chunk_lighting(chunk);
            }
        }

        // Cross-chunk propagation scoped to only dirty chunks and their neighbors
        if !light_dirty_positions.is_empty() {
            let recalced: Vec<(i32, i32)> = light_dirty_positions.iter()
                .take(MAX_LIGHT_RECALCS).cloned().collect();
            self.propagate_cross_chunk_lighting(&recalced);
        }

        // Rebuild meshes for dirty chunks (limit per frame to avoid stuttering)
        let dirty_chunk_positions: Vec<(i32, i32)> = self.chunks.iter()
            .filter(|(_, c)| c.dirty)
            .map(|(&pos, _)| pos)
            .take(MAX_MESH_REBUILDS)
            .collect();

        for pos in dirty_chunk_positions {
            let (cx, cz) = pos;

            let mesh_data = {
                if let Some(center) = self.chunks.get(&pos) {
                    let neighbors = ChunkNeighbors {
                        center,
                        left:        self.chunks.get(&(cx - 1, cz    )),
                        right:       self.chunks.get(&(cx + 1, cz    )),
                        front:       self.chunks.get(&(cx,     cz + 1)),
                        back:        self.chunks.get(&(cx,     cz - 1)),
                        front_left:  self.chunks.get(&(cx - 1, cz + 1)),
                        front_right: self.chunks.get(&(cx + 1, cz + 1)),
                        back_left:   self.chunks.get(&(cx - 1, cz - 1)),
                        back_right:  self.chunks.get(&(cx + 1, cz - 1)),
                    };
                    Some(Chunk::generate_mesh(&neighbors, smooth_lighting))
                } else {
                    None
                }
            };

            if let Some((vertices, indices, water_vertices, water_indices, transparent_vertices, transparent_indices)) = mesh_data {
                if let Some(chunk) = self.chunks.get_mut(&pos) {
                    chunk.vertices = vertices;
                    chunk.indices = indices;
                    chunk.water_vertices = water_vertices;
                    chunk.water_indices = water_indices;
                    chunk.transparent_vertices = transparent_vertices;
                    chunk.transparent_indices = transparent_indices;
                    chunk.dirty = false;
                    chunk.mesh_version = chunk.mesh_version.wrapping_add(1);
                }
            }
        }
    }

    /// Lightweight light propagation for a single newly-loaded chunk.
    /// Only propagates between this chunk and its 4 immediate neighbors.
    /// Much faster than full propagation: O(4 * height * size) vs O(15 * N * height * size)
    fn propagate_light_for_chunk(&mut self, pos: (i32, i32)) {
        let (cx, cz) = pos;

        // Do a few passes to let light spread across the boundary
        for _ in 0..3 {
            // Propagate from new chunk to neighbors
            self.propagate_edge_to_neighbor(cx, cz, 1, 0);  // to right
            self.propagate_edge_to_neighbor(cx, cz, -1, 0); // to left
            self.propagate_edge_to_neighbor(cx, cz, 0, 1);  // to front
            self.propagate_edge_to_neighbor(cx, cz, 0, -1); // to back

            // Propagate from neighbors back to new chunk
            self.propagate_edge_to_neighbor(cx + 1, cz, -1, 0);
            self.propagate_edge_to_neighbor(cx - 1, cz, 1, 0);
            self.propagate_edge_to_neighbor(cx, cz + 1, 0, -1);
            self.propagate_edge_to_neighbor(cx, cz - 1, 0, 1);
        }
    }

    /// Propagate light from one chunk's edge to its neighbor in the given direction.
    fn propagate_edge_to_neighbor(&mut self, cx: i32, cz: i32, dx: i32, dz: i32) {
        let neighbor_pos = (cx + dx, cz + dz);

        if !self.chunks.contains_key(&neighbor_pos) {
            return;
        }

        // Collect edge lights from source chunk
        let edge_lights: Vec<(usize, usize, u8)> = {
            if let Some(chunk) = self.chunks.get(&(cx, cz)) {
                let mut lights = Vec::new();
                for y in 0..CHUNK_HEIGHT {
                    if dx == 1 {
                        // Right edge: x = CHUNK_SIZE - 1
                        for z in 0..CHUNK_SIZE {
                            let light = chunk.light_levels[CHUNK_SIZE - 1][y][z];
                            if light > 1 {
                                lights.push((y, z, light - 1));
                            }
                        }
                    } else if dx == -1 {
                        // Left edge: x = 0
                        for z in 0..CHUNK_SIZE {
                            let light = chunk.light_levels[0][y][z];
                            if light > 1 {
                                lights.push((y, z, light - 1));
                            }
                        }
                    } else if dz == 1 {
                        // Front edge: z = CHUNK_SIZE - 1
                        for x in 0..CHUNK_SIZE {
                            let light = chunk.light_levels[x][y][CHUNK_SIZE - 1];
                            if light > 1 {
                                lights.push((x, y, light - 1));
                            }
                        }
                    } else if dz == -1 {
                        // Back edge: z = 0
                        for x in 0..CHUNK_SIZE {
                            let light = chunk.light_levels[x][y][0];
                            if light > 1 {
                                lights.push((x, y, light - 1));
                            }
                        }
                    }
                }
                lights
            } else {
                Vec::new()
            }
        };

        // Apply lights to neighbor
        if let Some(neighbor) = self.chunks.get_mut(&neighbor_pos) {
            for (a, b, light) in edge_lights {
                let changed = if dx == 1 {
                    lighting::seed_light_and_propagate(neighbor, 0, a, b, light)
                } else if dx == -1 {
                    lighting::seed_light_and_propagate(neighbor, CHUNK_SIZE - 1, a, b, light)
                } else if dz == 1 {
                    lighting::seed_light_and_propagate(neighbor, a, b, 0, light)
                } else {
                    lighting::seed_light_and_propagate(neighbor, a, b, CHUNK_SIZE - 1, light)
                };
                if changed {
                    neighbor.dirty = true;
                }
            }
        }
    }

    /// Propagate light across chunk boundaries for the given dirty chunks and their neighbors.
    fn propagate_cross_chunk_lighting(&mut self, dirty_positions: &[(i32, i32)]) {
        // Build the set of chunks to process: dirty chunks + their immediate neighbors
        let mut positions_to_process: Vec<(i32, i32)> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for &(cx, cz) in dirty_positions {
            for dx in -1..=1 {
                for dz in -1..=1 {
                    let pos = (cx + dx, cz + dz);
                    if self.chunks.contains_key(&pos) && seen.insert(pos) {
                        positions_to_process.push(pos);
                    }
                }
            }
        }

        for _ in 0..3 {
            let mut any_changes = false;

            // For each chunk in the scoped set, propagate its edge light into neighbors
            for (cx, cz) in positions_to_process.clone() {

                // Propagate to right neighbor (cx+1): our x=CHUNK_SIZE-1 -> their x=0
                if self.chunks.contains_key(&(cx + 1, cz)) {
                    let edge_lights: Vec<(usize, usize, u8)> = {
                        if let Some(chunk) = self.chunks.get(&(cx, cz)) {
                            let mut lights = Vec::new();
                            for y in 0..CHUNK_HEIGHT {
                                for z in 0..CHUNK_SIZE {
                                    let light = chunk.light_levels[CHUNK_SIZE - 1][y][z];
                                    if light > 1 {
                                        lights.push((y, z, light - 1));
                                    }
                                }
                            }
                            lights
                        } else {
                            Vec::new()
                        }
                    };

                    if let Some(neighbor) = self.chunks.get_mut(&(cx + 1, cz)) {
                        for (y, z, light) in edge_lights {
                            if lighting::seed_light_and_propagate(neighbor, 0, y, z, light) {
                                any_changes = true;
                                neighbor.dirty = true; // Rebuild mesh if light changed
                            }
                        }
                    }
                }

                // Propagate to left neighbor (cx-1): our x=0 -> their x=CHUNK_SIZE-1
                if self.chunks.contains_key(&(cx - 1, cz)) {
                    let edge_lights: Vec<(usize, usize, u8)> = {
                        if let Some(chunk) = self.chunks.get(&(cx, cz)) {
                            let mut lights = Vec::new();
                            for y in 0..CHUNK_HEIGHT {
                                for z in 0..CHUNK_SIZE {
                                    let light = chunk.light_levels[0][y][z];
                                    if light > 1 {
                                        lights.push((y, z, light - 1));
                                    }
                                }
                            }
                            lights
                        } else {
                            Vec::new()
                        }
                    };

                    if let Some(neighbor) = self.chunks.get_mut(&(cx - 1, cz)) {
                        for (y, z, light) in edge_lights {
                            if lighting::seed_light_and_propagate(neighbor, CHUNK_SIZE - 1, y, z, light) {
                                any_changes = true;
                                neighbor.dirty = true; // Rebuild mesh if light changed
                            }
                        }
                    }
                }

                // Propagate to front neighbor (cz+1): our z=CHUNK_SIZE-1 -> their z=0
                if self.chunks.contains_key(&(cx, cz + 1)) {
                    let edge_lights: Vec<(usize, usize, u8)> = {
                        if let Some(chunk) = self.chunks.get(&(cx, cz)) {
                            let mut lights = Vec::new();
                            for y in 0..CHUNK_HEIGHT {
                                for x in 0..CHUNK_SIZE {
                                    let light = chunk.light_levels[x][y][CHUNK_SIZE - 1];
                                    if light > 1 {
                                        lights.push((x, y, light - 1));
                                    }
                                }
                            }
                            lights
                        } else {
                            Vec::new()
                        }
                    };

                    if let Some(neighbor) = self.chunks.get_mut(&(cx, cz + 1)) {
                        for (x, y, light) in edge_lights {
                            if lighting::seed_light_and_propagate(neighbor, x, y, 0, light) {
                                any_changes = true;
                                neighbor.dirty = true; // Rebuild mesh if light changed
                            }
                        }
                    }
                }

                // Propagate to back neighbor (cz-1): our z=0 -> their z=CHUNK_SIZE-1
                if self.chunks.contains_key(&(cx, cz - 1)) {
                    let edge_lights: Vec<(usize, usize, u8)> = {
                        if let Some(chunk) = self.chunks.get(&(cx, cz)) {
                            let mut lights = Vec::new();
                            for y in 0..CHUNK_HEIGHT {
                                for x in 0..CHUNK_SIZE {
                                    let light = chunk.light_levels[x][y][0];
                                    if light > 1 {
                                        lights.push((x, y, light - 1));
                                    }
                                }
                            }
                            lights
                        } else {
                            Vec::new()
                        }
                    };

                    if let Some(neighbor) = self.chunks.get_mut(&(cx, cz - 1)) {
                        for (x, y, light) in edge_lights {
                            if lighting::seed_light_and_propagate(neighbor, x, y, CHUNK_SIZE - 1, light) {
                                any_changes = true;
                                neighbor.dirty = true; // Rebuild mesh if light changed
                            }
                        }
                    }
                }
            }

            if !any_changes {
                break;
            }
        }
    }

    /// Saves all modified chunks to disk (call on game exit)
    pub fn save_all_modified_chunks(&mut self) {
        // First, wait for any pending background saves to complete
        self.chunk_loader.shutdown();

        // Then save all currently loaded modified chunks synchronously
        let mut saved_count = 0;
        for (pos, chunk) in &self.chunks {
            if chunk.modified {
                if let Err(e) = chunk_storage::save_chunk(chunk) {
                    eprintln!("Failed to save chunk {:?}: {}", pos, e);
                } else {
                    saved_count += 1;
                }
            }
        }
        if saved_count > 0 {
            log::info!("Saved {} modified chunks", saved_count);
        }
    }
}