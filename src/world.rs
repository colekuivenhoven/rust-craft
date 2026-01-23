use crate::chunk::{Chunk, ChunkNeighbors, CHUNK_SIZE};
use crate::block::BlockType;
use crate::lighting;
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
        for x in -self.render_distance..=self.render_distance {
            for z in -self.render_distance..=self.render_distance {
                let chunk_pos = (center_x + x, center_z + z);
                if !self.chunks.contains_key(&chunk_pos) {
                    let mut chunk = Chunk::new(chunk_pos.0, chunk_pos.1);
                    lighting::calculate_chunk_lighting(&mut chunk);
                    self.chunks.insert(chunk_pos, chunk);
                }
            }
        }
    }

    pub fn update_chunks(&mut self, camera_pos: (f32, f32)) {
        let chunk_x = (camera_pos.0 / CHUNK_SIZE as f32).floor() as i32;
        let chunk_z = (camera_pos.1 / CHUNK_SIZE as f32).floor() as i32;

        self.chunks.retain(|&(cx, cz), _| {
            (cx - chunk_x).abs() <= self.render_distance
                && (cz - chunk_z).abs() <= self.render_distance
        });

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

        // 1. Update the block in the specific chunk
        if let Some(chunk) = self.chunks.get_mut(&(chunk_x, chunk_z)) {
            chunk.set_block(local_x, y as usize, local_z, block_type);
            
            // 2. IMPORTANT: Check if we are on a chunk edge.
            // If we modified a block at the edge, the *neighboring* chunk 
            // might have a hidden face that now needs to be revealed.
            
            if local_x == 0 {
                // Modified left edge -> Dirty Left Neighbor
                if let Some(neighbor) = self.chunks.get_mut(&(chunk_x - 1, chunk_z)) {
                    neighbor.dirty = true;
                }
            } else if local_x == CHUNK_SIZE - 1 {
                // Modified right edge -> Dirty Right Neighbor
                if let Some(neighbor) = self.chunks.get_mut(&(chunk_x + 1, chunk_z)) {
                    neighbor.dirty = true;
                }
            }

            if local_z == 0 {
                // Modified back edge -> Dirty Back Neighbor
                if let Some(neighbor) = self.chunks.get_mut(&(chunk_x, chunk_z - 1)) {
                    neighbor.dirty = true;
                }
            } else if local_z == CHUNK_SIZE - 1 {
                // Modified front edge -> Dirty Front Neighbor
                if let Some(neighbor) = self.chunks.get_mut(&(chunk_x, chunk_z + 1)) {
                    neighbor.dirty = true;
                }
            }
        }
    }

    pub fn rebuild_dirty_chunks(&mut self) {
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
            if let Some((vertices, indices, water_vertices, water_indices)) = mesh_data {
                if let Some(chunk) = self.chunks.get_mut(&pos) {
                    chunk.vertices = vertices;
                    chunk.indices = indices;
                    chunk.water_vertices = water_vertices;
                    chunk.water_indices = water_indices;
                    chunk.dirty = false;
                }
            }
        }
    }
}