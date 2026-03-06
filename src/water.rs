use crate::block::BlockType;
use crate::chunk::{WATER_LEVEL_SOURCE, WATER_LEVEL_MAX_FLOW, WATER_LEVEL_MIN_FLOW};
use crate::world::World;
use std::collections::VecDeque;

pub struct WaterSimulation {
    pending_updates: VecDeque<(i32, i32, i32)>,
    update_interval: f32,
    elapsed: f32,
}

impl WaterSimulation {
    pub fn new(update_interval: f32) -> Self {
        Self {
            pending_updates: VecDeque::new(),
            update_interval,
            elapsed: 0.0,
        }
    }

    pub fn update(&mut self, world: &mut World, dt: f32) {
        self.elapsed += dt;

        if self.elapsed >= self.update_interval {
            self.elapsed = 0.0;
            self.process_water_physics(world);
        }
    }

    pub fn schedule_update(&mut self, x: i32, y: i32, z: i32) {
        self.pending_updates.push_back((x, y, z));
    }

    fn process_water_physics(&mut self, world: &mut World) {
        let batch_size = 200.min(self.pending_updates.len());
        let mut new_water: Vec<(i32, i32, i32, u8)> = Vec::new();

        for _ in 0..batch_size {
            if let Some((x, y, z)) = self.pending_updates.pop_front() {
                let block = world.get_block_world(x, y, z);
                if block != BlockType::Water {
                    continue;
                }

                let level = world.get_water_level_world(x, y, z);
                if level == 0 {
                    continue;
                }

                // Water flows down first (full source block)
                let below = world.get_block_world(x, y - 1, z);
                if below == BlockType::Air {
                    new_water.push((x, y - 1, z, WATER_LEVEL_SOURCE));
                    continue;
                }

                // Horizontal spreading: level decreases by 1
                if level >= WATER_LEVEL_MIN_FLOW + 1 || level == WATER_LEVEL_SOURCE {
                    let flow_level = if level == WATER_LEVEL_SOURCE {
                        WATER_LEVEL_MAX_FLOW
                    } else {
                        level - 1
                    };

                    let directions = [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)];
                    for &(dx, dz) in &directions {
                        let nx = x + dx;
                        let nz = z + dz;
                        let neighbor = world.get_block_world(nx, y, nz);

                        if neighbor == BlockType::Air {
                            new_water.push((nx, y, nz, flow_level));
                        } else if neighbor == BlockType::Water {
                            // If neighbor has a lower level, update it
                            let neighbor_level = world.get_water_level_world(nx, y, nz);
                            if neighbor_level < flow_level {
                                new_water.push((nx, y, nz, flow_level));
                            }
                        }
                    }
                }
            }
        }

        // Apply new water blocks
        for (x, y, z, level) in new_water {
            let existing = world.get_block_world(x, y, z);
            let existing_level = if existing == BlockType::Water {
                world.get_water_level_world(x, y, z)
            } else {
                0
            };

            // Only place/update if we're increasing the level
            if level > existing_level {
                if existing != BlockType::Water {
                    world.set_block_world(x, y, z, BlockType::Water);
                }
                world.set_water_level_world(x, y, z, level);
                self.schedule_update(x, y, z);
            }
        }
    }

    pub fn add_water_source(&mut self, x: i32, y: i32, z: i32, world: &mut World) {
        if world.get_block_world(x, y, z) == BlockType::Air {
            world.set_block_world(x, y, z, BlockType::Water);
            world.set_water_level_world(x, y, z, WATER_LEVEL_SOURCE);
            self.schedule_update(x, y, z);
        }
    }
}
