use crate::chunk::{Chunk, ChunkNeighbors, CHUNK_SIZE, CHUNK_HEIGHT};
use crate::chunk_storage;
use crate::block::BlockType;
use crate::lighting;
use rayon::prelude::*;
use std::collections::HashMap;

pub struct World {
    pub chunks: HashMap<(i32, i32), Chunk>,
    render_distance: i32,
}

impl World {
    pub fn new(render_distance: i32) -> Self {
        let mut world = Self {
            chunks: HashMap::new(),
            render_distance,
        };
        world.generate_chunks(0, 0);
        world
    }

    pub fn generate_chunks(&mut self, center_x: i32, center_z: i32) {
        // Collect positions that need new chunks
        let positions_needed: Vec<(i32, i32)> = (-self.render_distance..=self.render_distance)
            .flat_map(|x| {
                (-self.render_distance..=self.render_distance).map(move |z| (center_x + x, center_z + z))
            })
            .filter(|pos| !self.chunks.contains_key(pos))
            .collect();

        if positions_needed.is_empty() {
            return;
        }

        // First, try to load saved chunks
        let mut loaded_chunks: Vec<((i32, i32), Chunk)> = Vec::new();
        let mut positions_to_generate: Vec<(i32, i32)> = Vec::new();

        for &(cx, cz) in &positions_needed {
            if let Some(mut chunk) = chunk_storage::load_chunk(cx, cz) {
                lighting::calculate_chunk_lighting(&mut chunk);
                loaded_chunks.push(((cx, cz), chunk));
            } else {
                positions_to_generate.push((cx, cz));
            }
        }

        // Generate new chunks in parallel using rayon
        let generated_chunks: Vec<((i32, i32), Chunk)> = positions_to_generate
            .par_iter()
            .map(|&(cx, cz)| {
                let mut chunk = Chunk::new(cx, cz);
                lighting::calculate_chunk_lighting(&mut chunk);
                ((cx, cz), chunk)
            })
            .collect();

        // Combine loaded and generated chunks
        let all_new_chunks: Vec<((i32, i32), Chunk)> = loaded_chunks
            .into_iter()
            .chain(generated_chunks.into_iter())
            .collect();

        // Insert chunks into the world (must be sequential due to HashMap)
        let new_chunk_positions: Vec<(i32, i32)> = all_new_chunks
            .iter()
            .map(|(pos, _)| *pos)
            .collect();

        for (pos, chunk) in all_new_chunks {
            self.chunks.insert(pos, chunk);
        }

        // Mark neighbors dirty so they re-mesh (stitch) with the new chunks
        for (cx, cz) in &new_chunk_positions {
            for (dx, dz) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let neighbor_pos = (cx + dx, cz + dz);
                if let Some(neighbor) = self.chunks.get_mut(&neighbor_pos) {
                    neighbor.dirty = true;
                }
            }
        }

        // Propagate light between the new chunks and the existing world
        // - This ensures sunlight or caves flow correctly across the new boundaries
        self.propagate_cross_chunk_lighting(&new_chunk_positions);
    }

    pub fn update_chunks(&mut self, camera_pos: (f32, f32)) {
        let chunk_x = (camera_pos.0 / CHUNK_SIZE as f32).floor() as i32;
        let chunk_z = (camera_pos.1 / CHUNK_SIZE as f32).floor() as i32;

        // Save modified chunks before unloading
        let chunks_to_unload: Vec<(i32, i32)> = self.chunks
            .iter()
            .filter(|(&(cx, cz), _)| {
                (cx - chunk_x).abs() > self.render_distance
                    || (cz - chunk_z).abs() > self.render_distance
            })
            .map(|(&pos, _)| pos)
            .collect();

        for pos in chunks_to_unload {
            if let Some(chunk) = self.chunks.get(&pos) {
                if chunk.modified {
                    if let Err(e) = chunk_storage::save_chunk(chunk) {
                        eprintln!("Failed to save chunk {:?}: {}", pos, e);
                    }
                }
            }
            self.chunks.remove(&pos);
        }

        self.generate_chunks(chunk_x, chunk_z);
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

        // Light can propagate up to 15 blocks, so changes near edges affect neighbors.
        let light_propagation_distance = 15;

        // 1. Update the block in the specific chunk
        if let Some(chunk) = self.chunks.get_mut(&(chunk_x, chunk_z)) {
            chunk.set_block(local_x, y as usize, local_z, block_type);
            // Always mark light_dirty to ensure full recalculation
            chunk.light_dirty = true;
        }

        // 2. Check all neighbors (including diagonals) within range.
        // - We determine the range of chunk offsets (-1, 0, or 1) based on proximity to edges.
        
        let min_dx = if local_x < light_propagation_distance { -1 } else { 0 };
        let max_dx = if local_x >= CHUNK_SIZE - light_propagation_distance { 1 } else { 0 };
        
        let min_dz = if local_z < light_propagation_distance { -1 } else { 0 };
        let max_dz = if local_z >= CHUNK_SIZE - light_propagation_distance { 1 } else { 0 };

        for dx in min_dx..=max_dx {
            for dz in min_dz..=max_dz {
                // Skip the center chunk (dx=0, dz=0) as it was handled above
                if dx == 0 && dz == 0 {
                    continue;
                }

                if let Some(neighbor) = self.chunks.get_mut(&(chunk_x + dx, chunk_z + dz)) {
                    neighbor.dirty = true;       // Needs mesh rebuild
                    neighbor.light_dirty = true; // Needs light recalc
                }
            }
        }
    }

    pub fn rebuild_dirty_chunks(&mut self) {
        // First, recalculate lighting for any chunks that need it
        let light_dirty_positions: Vec<(i32, i32)> = self.chunks.iter()
            .filter(|(_, c)| c.light_dirty)
            .map(|(&pos, _)| pos)
            .collect();

        for pos in light_dirty_positions.iter() {
            if let Some(chunk) = self.chunks.get_mut(pos) {
                lighting::calculate_chunk_lighting(chunk);
            }
        }

        // Cross-chunk light propagation: propagate light from chunk edges into neighbors
        // - This is needed because calculate_chunk_lighting only works within a single chunk
        if !light_dirty_positions.is_empty() {
            self.propagate_cross_chunk_lighting(&light_dirty_positions);
        }

        // Then rebuild meshes for dirty chunks
        let dirty_chunk_positions: Vec<(i32, i32)> = self.chunks.iter()
            .filter(|(_, c)| c.dirty)
            .map(|(&pos, _)| pos)
            .collect();

        for pos in dirty_chunk_positions {
            let (cx, cz) = pos;

            // Generate mesh data immutably (read-only access to self.chunks)
            let mesh_data = {
                if let Some(center) = self.chunks.get(&pos) {
                    let neighbors = ChunkNeighbors {
                        center,
                        left: self.chunks.get(&(cx - 1, cz)),
                        right: self.chunks.get(&(cx + 1, cz)),
                        front: self.chunks.get(&(cx, cz + 1)),
                        back: self.chunks.get(&(cx, cz - 1)),
                    };
                    Some(Chunk::generate_mesh(&neighbors))
                } else {
                    None
                }
            };

            // Apply mesh data mutably (write access to self.chunks)
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

    /// Propagate light across chunk boundaries.
    fn propagate_cross_chunk_lighting(&mut self, _dirty_positions: &[(i32, i32)]) {
        for _ in 0..15 {
            let mut any_changes = false;

            // Get all chunk positions (we need to check all, not just dirty ones)
            let all_positions: Vec<(i32, i32)> = self.chunks.keys().cloned().collect();

            // For each chunk, propagate its edge light into neighbors
            for (cx, cz) in all_positions {

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
    pub fn save_all_modified_chunks(&self) {
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