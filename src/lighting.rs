use crate::block::BlockType;
use crate::chunk::{Chunk, CHUNK_HEIGHT, CHUNK_SIZE};
use std::collections::VecDeque;

pub const MAX_LIGHT_LEVEL: u8 = 15;
pub const SUNLIGHT_LEVEL: u8 = 15;

const DIRECTIONS: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

#[derive(Clone, Copy)]
struct LightNode {
    x: i32,
    y: i32,
    z: i32,
}

#[derive(Clone, Copy)]
struct LightRemovalNode {
    x: i32,
    y: i32,
    z: i32,
    light_level: u8,
}

/// Calculate all light levels for a chunk from scratch.
/// Call this when a chunk is first generated.
pub fn calculate_chunk_lighting(chunk: &mut Chunk) {
    // Clear existing light
    for x in 0..CHUNK_SIZE {
        for y in 0..CHUNK_HEIGHT {
            for z in 0..CHUNK_SIZE {
                chunk.light_levels[x][y][z] = 0;
            }
        }
    }

    let mut light_queue: VecDeque<LightNode> = VecDeque::new();

    // Phase 1: Sunlight propagation (top-down)
    for x in 0..CHUNK_SIZE {
        for z in 0..CHUNK_SIZE {
            let mut sunlight = SUNLIGHT_LEVEL;
            for y in (0..CHUNK_HEIGHT).rev() {
                let block = chunk.blocks[x][y][z];

                if block.is_solid() && !block.is_transparent() {
                    sunlight = 0;
                } else if block.is_transparent() && block != BlockType::Air {
                    sunlight = sunlight.saturating_sub(1);
                }

                if sunlight > 0 {
                    chunk.light_levels[x][y][z] = sunlight;
                    light_queue.push_back(LightNode {
                        x: x as i32,
                        y: y as i32,
                        z: z as i32,
                    });
                }
            }
        }
    }

    // Phase 2: Block light sources (GlowStone, etc.)
    for x in 0..CHUNK_SIZE {
        for y in 0..CHUNK_HEIGHT {
            for z in 0..CHUNK_SIZE {
                let emission = chunk.blocks[x][y][z].get_light_emission();
                if emission > 0 {
                    chunk.light_levels[x][y][z] = emission;
                    light_queue.push_back(LightNode {
                        x: x as i32,
                        y: y as i32,
                        z: z as i32,
                    });
                }
            }
        }
    }

    // Phase 3: BFS light propagation
    propagate_light(chunk, &mut light_queue);

    chunk.light_dirty = false;
}

/// Called when a block is removed (broken) at position (x, y, z).
pub fn on_block_removed(chunk: &mut Chunk, x: usize, y: usize, z: usize) {
    let ix = x as i32;
    let iy = y as i32;
    let iz = z as i32;

    let mut light_queue: VecDeque<LightNode> = VecDeque::new();

    // Check if this column receives sunlight from above
    let mut receives_sunlight = true;
    for check_y in (y + 1)..CHUNK_HEIGHT {
        let block = chunk.blocks[x][check_y][z];
        if block.is_solid() && !block.is_transparent() {
            receives_sunlight = false;
            break;
        }
    }

    if receives_sunlight {
        // Calculate sunlight level at this position
        let mut sunlight = SUNLIGHT_LEVEL;
        for check_y in ((y + 1)..CHUNK_HEIGHT).rev() {
            let block = chunk.blocks[x][check_y][z];
            if block.is_transparent() && block != BlockType::Air {
                sunlight = sunlight.saturating_sub(1);
            }
        }

        // Fill sunlight down
        for fill_y in (0..=y).rev() {
            let block = chunk.blocks[x][fill_y][z];
            if block.is_solid() && !block.is_transparent() {
                break;
            }
            if block.is_transparent() && block != BlockType::Air {
                sunlight = sunlight.saturating_sub(1);
            }

            if sunlight > chunk.light_levels[x][fill_y][z] {
                chunk.light_levels[x][fill_y][z] = sunlight;
                light_queue.push_back(LightNode {
                    x: ix,
                    y: fill_y as i32,
                    z: iz,
                });
            }
        }
    }

    // Check neighbors for light that should flood in
    let mut max_neighbor_light: u8 = 0;
    for &(dx, dy, dz) in &DIRECTIONS {
        let nx = ix + dx;
        let ny = iy + dy;
        let nz = iz + dz;

        if let Some(light) = get_light_checked(chunk, nx, ny, nz) {
            max_neighbor_light = max_neighbor_light.max(light);
        }
    }

    if max_neighbor_light > 1 {
        let new_light = max_neighbor_light - 1;
        if new_light > chunk.light_levels[x][y][z] {
            chunk.light_levels[x][y][z] = new_light;
            light_queue.push_back(LightNode {
                x: ix,
                y: iy,
                z: iz,
            });
        }
    }

    propagate_light(chunk, &mut light_queue);
}

/// Called when a block is placed at position (x, y, z).
pub fn on_block_placed(chunk: &mut Chunk, x: usize, y: usize, z: usize) {
    let block = chunk.blocks[x][y][z];
    let emission = block.get_light_emission();

    // If the placed block emits light, propagate it
    if emission > 0 {
        chunk.light_levels[x][y][z] = emission;
        let mut light_queue = VecDeque::new();
        light_queue.push_back(LightNode {
            x: x as i32,
            y: y as i32,
            z: z as i32,
        });
        propagate_light(chunk, &mut light_queue);
        return;
    }

    // If the block is solid and blocks light, remove light
    if block.is_solid() && !block.is_transparent() {
        let old_light = chunk.light_levels[x][y][z];
        chunk.light_levels[x][y][z] = 0;

        if old_light > 0 {
            remove_light_optimized(chunk, x, y, z, old_light);
        }

        // Check if we blocked sunlight for blocks below
        let mut was_sunlit = true;
        for check_y in (y + 1)..CHUNK_HEIGHT {
            let check_block = chunk.blocks[x][check_y][z];
            if check_block.is_solid() && !check_block.is_transparent() {
                was_sunlit = false;
                break;
            }
        }

        if was_sunlit {
            for below_y in (0..y).rev() {
                let below_block = chunk.blocks[x][below_y][z];
                if below_block.is_solid() && !below_block.is_transparent() {
                    break;
                }
                let below_light = chunk.light_levels[x][below_y][z];
                if below_light > 0 {
                    chunk.light_levels[x][below_y][z] = 0;
                    remove_light_optimized(chunk, x, below_y, z, below_light);
                }
            }
        }
    }
}

/// Optimized light removal that doesn't scan the entire chunk.
/// Uses BFS to find affected areas and collects relight sources from the boundary.
fn remove_light_optimized(chunk: &mut Chunk, x: usize, y: usize, z: usize, light_level: u8) {
    let mut removal_queue: VecDeque<LightRemovalNode> = VecDeque::new();
    let mut relight_queue: VecDeque<LightNode> = VecDeque::new();

    removal_queue.push_back(LightRemovalNode {
        x: x as i32,
        y: y as i32,
        z: z as i32,
        light_level,
    });

    // BFS to remove light and find boundary light sources
    while let Some(node) = removal_queue.pop_front() {
        for &(dx, dy, dz) in &DIRECTIONS {
            let nx = node.x + dx;
            let ny = node.y + dy;
            let nz = node.z + dz;

            if !in_bounds(nx, ny, nz) {
                continue;
            }

            let nux = nx as usize;
            let nuy = ny as usize;
            let nuz = nz as usize;

            let neighbor_light = chunk.light_levels[nux][nuy][nuz];
            let neighbor_block = chunk.blocks[nux][nuy][nuz];

            // Skip solid blocks - they don't propagate light
            if neighbor_block.is_solid() && !neighbor_block.is_transparent() {
                continue;
            }

            if neighbor_light > 0 && neighbor_light < node.light_level {
                // This was lit by the removed light source
                chunk.light_levels[nux][nuy][nuz] = 0;
                removal_queue.push_back(LightRemovalNode {
                    x: nx,
                    y: ny,
                    z: nz,
                    light_level: neighbor_light,
                });
            } else if neighbor_light >= node.light_level {
                // This has light from another source - needs to re-propagate
                relight_queue.push_back(LightNode {
                    x: nx,
                    y: ny,
                    z: nz,
                });
            }
        }
    }

    // Re-propagate from boundary sources
    propagate_light(chunk, &mut relight_queue);
}

/// Propagate light outward from all nodes in the queue using BFS.
fn propagate_light(chunk: &mut Chunk, queue: &mut VecDeque<LightNode>) {
    while let Some(node) = queue.pop_front() {
        let current_light = match get_light_checked(chunk, node.x, node.y, node.z) {
            Some(l) => l,
            None => continue,
        };

        if current_light <= 1 {
            continue;
        }

        let new_light = current_light - 1;

        for &(dx, dy, dz) in &DIRECTIONS {
            let nx = node.x + dx;
            let ny = node.y + dy;
            let nz = node.z + dz;

            if !in_bounds(nx, ny, nz) {
                continue;
            }

            let nux = nx as usize;
            let nuy = ny as usize;
            let nuz = nz as usize;

            let neighbor_block = chunk.blocks[nux][nuy][nuz];

            // Light doesn't pass through solid non-transparent blocks
            if neighbor_block.is_solid() && !neighbor_block.is_transparent() {
                continue;
            }

            if new_light > chunk.light_levels[nux][nuy][nuz] {
                chunk.light_levels[nux][nuy][nuz] = new_light;
                queue.push_back(LightNode {
                    x: nx,
                    y: ny,
                    z: nz,
                });
            }
        }
    }
}

#[inline]
fn in_bounds(x: i32, y: i32, z: i32) -> bool {
    x >= 0
        && x < CHUNK_SIZE as i32
        && y >= 0
        && y < CHUNK_HEIGHT as i32
        && z >= 0
        && z < CHUNK_SIZE as i32
}

#[inline]
fn get_light_checked(chunk: &Chunk, x: i32, y: i32, z: i32) -> Option<u8> {
    if in_bounds(x, y, z) {
        Some(chunk.light_levels[x as usize][y as usize][z as usize])
    } else {
        None
    }
}
