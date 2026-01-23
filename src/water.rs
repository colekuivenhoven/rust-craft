use crate::block::BlockType;
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
        let batch_size = 100.min(self.pending_updates.len());
        let mut new_water_blocks = Vec::new();

        for _ in 0..batch_size {
            if let Some((x, y, z)) = self.pending_updates.pop_front() {
                let block = world.get_block_world(x, y, z);

                if block == BlockType::Water {
                    // Water flows down
                    let below = world.get_block_world(x, y - 1, z);
                    if below == BlockType::Air {
                        new_water_blocks.push((x, y - 1, z));
                        continue;
                    }

                    // Water spreads horizontally
                    let directions = [(1, 0), (-1, 0), (0, 1), (0, -1)];
                    for &(dx, dz) in &directions {
                        let nx = x + dx;
                        let nz = z + dz;
                        let neighbor = world.get_block_world(nx, y, nz);

                        if neighbor == BlockType::Air {
                            // Simple spreading logic - water spreads to adjacent air blocks
                            if rand::random::<f32>() > 0.7 {
                                new_water_blocks.push((nx, y, nz));
                            }
                        }
                    }
                }
            }
        }

        // Apply new water blocks
        for (x, y, z) in new_water_blocks {
            world.set_block_world(x, y, z, BlockType::Water);
            self.schedule_update(x, y, z);
        }
    }

    pub fn add_water_source(&mut self, x: i32, y: i32, z: i32, world: &mut World) {
        if world.get_block_world(x, y, z) == BlockType::Air {
            world.set_block_world(x, y, z, BlockType::Water);
            self.schedule_update(x, y, z);
        }
    }
}
