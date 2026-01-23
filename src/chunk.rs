use crate::block::{BlockType, Vertex, create_face_vertices};
use crate::lighting;
use cgmath::Vector3;
use noise::{NoiseFn, Perlin};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

pub const CHUNK_SIZE: usize = 32;
pub const CHUNK_HEIGHT: usize = 64;

// Tree generation configuration
pub const TREE_MIN_HEIGHT: usize = 4;      // Minimum trunk height (current average)
pub const TREE_MAX_HEIGHT: usize = 10;     // Maximum trunk height (tall trees)
pub const TREE_BRANCH_CHANCE: f64 = 0.4;   // Chance for trunk to branch near top (0.0 - 1.0)
pub const TREE_TALL_CHANCE: f64 = 0.3;     // Chance for a tree to be tall variant (0.0 - 1.0)

pub struct Chunk {
    pub blocks: [[[BlockType; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE],
    pub light_levels: [[[u8; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE],
    pub position: (i32, i32),
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
    pub water_vertices: Vec<Vertex>,       // Separate water vertices for transparency pass
    pub water_indices: Vec<u16>,           // Separate water indices for transparency pass
    pub dirty: bool,
    pub light_dirty: bool,
}

/// Helper struct to provide safe access to neighboring chunks during mesh generation
pub struct ChunkNeighbors<'a> {
    pub center: &'a Chunk,
    pub left: Option<&'a Chunk>,   // (x-1, z)
    pub right: Option<&'a Chunk>,  // (x+1, z)
    pub front: Option<&'a Chunk>,  // (x, z+1)
    pub back: Option<&'a Chunk>,   // (x, z-1)
}

impl<'a> ChunkNeighbors<'a> {
    pub fn get_block(&self, x: i32, y: i32, z: i32) -> BlockType {
        if y < 0 || y >= CHUNK_HEIGHT as i32 {
            return BlockType::Air;
        }

        let (target_chunk, lx, lz) = self.resolve_coordinates(x, z);

        match target_chunk {
            Some(chunk) => chunk.blocks[lx][y as usize][lz],
            None => BlockType::Boundary, // Virtual block: transparent for solids, opaque for water
        }
    }

    pub fn get_light(&self, x: i32, y: i32, z: i32) -> u8 {
        if y < 0 { return 0; }
        if y >= CHUNK_HEIGHT as i32 { return 15; }

        let (target_chunk, lx, lz) = self.resolve_coordinates(x, z);

        match target_chunk {
            Some(chunk) => chunk.light_levels[lx][y as usize][lz],
            None => 15, // Assume full brightness for unloaded neighbors
        }
    }

    fn resolve_coordinates(&self, x: i32, z: i32) -> (Option<&'a Chunk>, usize, usize) {
        if x < 0 {
            (self.left, (x + CHUNK_SIZE as i32) as usize, z as usize)
        } else if x >= CHUNK_SIZE as i32 {
            (self.right, (x - CHUNK_SIZE as i32) as usize, z as usize)
        } else if z < 0 {
            (self.back, x as usize, (z + CHUNK_SIZE as i32) as usize)
        } else if z >= CHUNK_SIZE as i32 {
            (self.front, x as usize, (z - CHUNK_SIZE as i32) as usize)
        } else {
            (Some(self.center), x as usize, z as usize)
        }
    }
}

impl Chunk {
    pub fn new(chunk_x: i32, chunk_z: i32) -> Self {
        let mut chunk = Self {
            blocks: [[[BlockType::Air; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE],
            light_levels: [[[0u8; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE],
            position: (chunk_x, chunk_z),
            vertices: Vec::new(),
            indices: Vec::new(),
            water_vertices: Vec::new(),
            water_indices: Vec::new(),
            dirty: true,
            light_dirty: true,
        };
        chunk.generate_terrain();
        chunk
    }

    fn generate_terrain(&mut self) {
        let perlin = Perlin::new(42);
        let biome_perlin = Perlin::new(44);     // very low frequency: ocean/land mask
        let mountain_perlin = Perlin::new(45);  // low frequency: mountain mask
        let detail_perlin = Perlin::new(46);    // higher frequency: small variation

        // Cave noise generators
        let cave_perlin_1 = Perlin::new(100);   // Primary spaghetti cave noise
        let cave_perlin_2 = Perlin::new(101);   // Secondary spaghetti cave noise (perpendicular)
        let cheese_perlin = Perlin::new(102);   // Large cavern "cheese" caves
        let cave_mask_perlin = Perlin::new(103); // Controls cave density in different areas

        let world_offset_x = self.position.0 * CHUNK_SIZE as i32;
        let world_offset_z = self.position.1 * CHUNK_SIZE as i32;
        let sea_level: isize = 28;

        fn clamp01(v: f64) -> f64 {
            v.max(0.0).min(1.0)
        }

        // Smoothstep for soft biome transitions.
        fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
            let t = clamp01((x - edge0) / (edge1 - edge0));
            t * t * (3.0 - 2.0 * t)
        }

        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let world_x = (world_offset_x + x as i32) as f64;
                let world_z = (world_offset_z + z as i32) as f64;

                // --- Large-scale biome masks ---
                // Ocean mask: 0 = deep ocean, 1 = inland.
                let biome_scale = 0.0035;
                let biome_raw = (biome_perlin.get([world_x * biome_scale, world_z * biome_scale]) + 1.0) * 0.5;
                let inland = smoothstep(0.35, 0.55, biome_raw); // Push more area into ocean while keeping smooth shores.

                // Mountain mask: 0 = no mountains, 1 = mountains.
                let mountain_scale = 0.006;
                let mountain_raw = (mountain_perlin.get([world_x * mountain_scale, world_z * mountain_scale]) + 1.0) * 0.5;
                let mountains = smoothstep(0.60, 0.80, mountain_raw) * inland; // Only allow strong mountains well inland.

                // --- Base terrain (plains) + local detail ---
                let base_scale = 0.02;
                let base_n = (perlin.get([world_x * base_scale, world_z * base_scale]) + 1.0) * 0.5;
                let base_height = 18.0 + base_n * 18.0; // ~18..36

                let detail_scale = 0.06;
                let detail = detail_perlin.get([world_x * detail_scale, world_z * detail_scale]) * 2.5;

                // --- Ocean shaping ---
                let ocean_drop = (1.0 - inland) * 16.0; // Pull terrain down near coasts/ocean.

                // --- Mountain shaping ---
                let ridge_scale = 0.018; // Add extra height in mountainous areas. Some ruggedness comes from slightly higher frequency noise.
                let ridge = (perlin.get([world_x * ridge_scale, world_z * ridge_scale]).abs()) * 12.0;
                let mountain_lift = mountains * (22.0 + ridge);

                let height_f = base_height + detail + mountain_lift - ocean_drop;
                let height_i = height_f.round() as isize;

                let height = height_i
                    .clamp(0, (CHUNK_HEIGHT as isize) - 1) as usize;

                // Surface palette
                let sea = sea_level as usize;
                for y in 0..=height {
                    self.blocks[x][y][z] = if y == height && y > sea {
                        BlockType::Grass
                    } 
                    else if y > height.saturating_sub(3) && y > sea {
                        BlockType::Dirt
                    } 
                    else if y > (sea.saturating_sub(3)) {
                        BlockType::Stone // Slightly more stone above sea floor.
                    } 
                    else {
                        BlockType::Sand // Sand near/below sea level (beaches/sea floor).
                    };
                }

                // Fill oceans/lakes.
                for y in 0..(sea_level as usize) {
                    if self.blocks[x][y][z] == BlockType::Air {
                        self.blocks[x][y][z] = BlockType::Water;
                    }
                }
            }
        }

        // === Cave Generation ===
        // Carve caves using 3D noise after terrain is generated
        // This creates both "spaghetti" tunnels and "cheese" caverns

        let cave_floor = 3;  // Don't carve below this (bedrock layer)
        let base_surface_depth = 4;  // Base minimum depth below surface
        let surface_opening_perlin = Perlin::new(104);  // Controls where caves can reach surface

        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let world_x = (world_offset_x + x as i32) as f64;
                let world_z = (world_offset_z + z as i32) as f64;

                // Find surface height at this column
                let mut surface_y = 0;
                for y in (0..CHUNK_HEIGHT).rev() {
                    if self.blocks[x][y][z] != BlockType::Air && self.blocks[x][y][z] != BlockType::Water {
                        surface_y = y;
                        break;
                    }
                }

                // Regional cave density mask - some areas have more caves than others
                let cave_density = (cave_mask_perlin.get([world_x * 0.008, world_z * 0.008]) + 1.0) * 0.5;

                // Surface opening chance - occasional spots where caves can reach surface
                // Uses low frequency noise so openings cluster into natural-looking sinkholes
                let surface_opening_noise = surface_opening_perlin.get([world_x * 0.02, world_z * 0.02]);
                let allow_surface_opening = surface_opening_noise > 0.7;  // ~15% of terrain can have openings
                let min_surface_depth = if allow_surface_opening { 0 } else { base_surface_depth };

                for y in cave_floor..CHUNK_HEIGHT {
                    // Skip air and water blocks
                    let block = self.blocks[x][y][z];
                    if block == BlockType::Air || block == BlockType::Water {
                        continue;
                    }

                    // Don't carve too close to surface (unless this is an opening zone)
                    let depth_below_surface = surface_y.saturating_sub(y);
                    if depth_below_surface < min_surface_depth {
                        continue;
                    }

                    let world_y = y as f64;

                    // === Spaghetti Caves ===
                    // Two perpendicular noise fields that create winding tunnels where they intersect
                    let spaghetti_scale = 0.04;
                    let spaghetti_1 = cave_perlin_1.get([
                        world_x * spaghetti_scale,
                        world_y * spaghetti_scale * 0.5,  // Stretch vertically for horizontal tunnels
                        world_z * spaghetti_scale
                    ]);
                    let spaghetti_2 = cave_perlin_2.get([
                        world_x * spaghetti_scale,
                        world_y * spaghetti_scale * 0.5,
                        world_z * spaghetti_scale
                    ]);

                    // Tunnel forms where both noise values are near zero (narrow band)
                    let tunnel_threshold = 0.12 + (1.0 - cave_density) * 0.08;  // Varies with density
                    let is_spaghetti = spaghetti_1.abs() < tunnel_threshold && spaghetti_2.abs() < tunnel_threshold;

                    // === Cheese Caves (larger caverns) ===
                    let cheese_scale = 0.025;
                    let cheese_value = cheese_perlin.get([
                        world_x * cheese_scale,
                        world_y * cheese_scale * 0.6,  // Slightly squashed vertically
                        world_z * cheese_scale
                    ]);

                    // Cheese caves only appear at lower elevations and where density allows
                    let cheese_y_factor = (1.0 - (world_y / 40.0).min(1.0)).max(0.0);  // Stronger at lower Y
                    let cheese_threshold = 0.55 + (1.0 - cave_density) * 0.2 - cheese_y_factor * 0.15;
                    let is_cheese = cheese_value > cheese_threshold && cave_density > 0.3;

                    // Carve the cave
                    if is_spaghetti || is_cheese {
                        // Don't carve through sand underwater (prevents ocean flooding caves)
                        if block == BlockType::Sand && y < sea_level as usize {
                            continue;
                        }
                        self.blocks[x][y][z] = BlockType::Air;
                    }
                }
            }
        }

        // === Tree Generation ===
        // (Now separate loop after caves are carved)
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let world_x = (world_offset_x + x as i32) as f64;
                let world_z = (world_offset_z + z as i32) as f64;

                // Recalculate biome values for tree placement
                let biome_scale = 0.0035;
                let biome_raw = (biome_perlin.get([world_x * biome_scale, world_z * biome_scale]) + 1.0) * 0.5;
                let inland = smoothstep(0.35, 0.55, biome_raw);

                // Find grass surface for tree placement
                // Trees should only spawn on Grass blocks (original terrain surface, not cave ceilings)
                let mut height = 0;
                let mut found_grass = false;
                for y in (0..CHUNK_HEIGHT).rev() {
                    if self.blocks[x][y][z] == BlockType::Grass {
                        height = y;
                        found_grass = true;
                        break;
                    }
                }

                // Skip if no grass found (underwater, cave, etc.)
                if !found_grass {
                    continue;
                }

                let sea = sea_level as usize;

                // Tree generation with variable height and occasional branching
                let max_tree_space = TREE_MAX_HEIGHT + 4; // trunk + leaves

                // Avoid trees in/near oceans (require inlandness) and keep them above sea level.
                if height > sea && height < CHUNK_HEIGHT - max_tree_space && x > 3 && x < CHUNK_SIZE - 3 && z > 3 && z < CHUNK_SIZE - 3 {
                    // Reduce trees near coasts.
                    if inland > 0.55 {
                        let tree_noise = perlin.get([world_x * 0.1, world_z * 0.1]);
                        if tree_noise > 0.7 {
                            // Check if there's already a tree trunk within 2 blocks (prevents adjacent trees)
                            let mut too_close_to_tree = false;
                            'check: for check_x in x.saturating_sub(2)..=(x + 2).min(CHUNK_SIZE - 1) {
                                for check_z in z.saturating_sub(2)..=(z + 2).min(CHUNK_SIZE - 1) {
                                    // Skip checking the current position
                                    if check_x == x && check_z == z {
                                        continue;
                                    }
                                    // Check for wood at trunk level (height + 1)
                                    if height + 1 < CHUNK_HEIGHT && self.blocks[check_x][height + 1][check_z] == BlockType::Wood {
                                        too_close_to_tree = true;
                                        break 'check;
                                    }
                                }
                            }

                            if too_close_to_tree {
                                continue; // Skip this tree, too close to another
                            }

                            // Use deterministic RNG based on world position for consistent tree generation
                            let tree_seed = ((world_x as i64).wrapping_mul(73856093) ^ (world_z as i64).wrapping_mul(19349663)) as u64;
                            let mut rng = StdRng::seed_from_u64(tree_seed);

                            // Determine tree height - either normal or tall variant
                            let is_tall = rng.gen::<f64>() < TREE_TALL_CHANCE;
                            let trunk_height = if is_tall {
                                rng.gen_range(TREE_MAX_HEIGHT - 2..=TREE_MAX_HEIGHT)
                            } else {
                                rng.gen_range(TREE_MIN_HEIGHT..=TREE_MIN_HEIGHT + 2)
                            };

                            // Generate main trunk (1 block wide)
                            for trunk_y in height + 1..=height + trunk_height {
                                if trunk_y < CHUNK_HEIGHT {
                                    self.blocks[x][trunk_y][z] = BlockType::Wood;
                                }
                            }

                            // Determine where branching starts (last 2-3 blocks of trunk)
                            let branch_start_y = height + trunk_height.saturating_sub(2);

                            // Occasionally add branches near the top of the trunk
                            let should_branch = rng.gen::<f64>() < TREE_BRANCH_CHANCE;
                            if should_branch && trunk_height >= 5 {
                                // Add 1-2 branch blocks adjacent to trunk near the top
                                let num_branches = rng.gen_range(1..=3);
                                for _ in 0..num_branches {
                                    let branch_y = rng.gen_range(branch_start_y..=height + trunk_height);
                                    if branch_y < CHUNK_HEIGHT {
                                        // Pick a random adjacent direction for the branch
                                        let directions: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                                        let (dx, dz) = directions[rng.gen_range(0..4)];
                                        let bx = (x as i32 + dx) as usize;
                                        let bz = (z as i32 + dz) as usize;
                                        if bx < CHUNK_SIZE && bz < CHUNK_SIZE {
                                            self.blocks[bx][branch_y][bz] = BlockType::Wood;
                                        }
                                    }
                                }
                            }

                            // Generate leaves - size scales with tree height
                            let leaf_radius = if trunk_height >= 7 { 3 } else { 2 };
                            let leaf_start_y = height + trunk_height - 1; // Leaves start a bit below top of trunk
                            let leaf_height = if trunk_height >= 7 { 5 } else { 4 };

                            for lx in x.saturating_sub(leaf_radius)..=(x + leaf_radius).min(CHUNK_SIZE - 1) {
                                for lz in z.saturating_sub(leaf_radius)..=(z + leaf_radius).min(CHUNK_SIZE - 1) {
                                    for ly in leaf_start_y..leaf_start_y + leaf_height {
                                        if ly < CHUNK_HEIGHT {
                                            // Create a more natural canopy shape
                                            let dx = (lx as i32 - x as i32).abs();
                                            let dz = (lz as i32 - z as i32).abs();
                                            let dy = ly - leaf_start_y;

                                            // Skip corners for rounder shape, especially at top and bottom
                                            let is_corner = dx == leaf_radius as i32 && dz == leaf_radius as i32;
                                            let skip_corner = is_corner && (dy == 0 || dy >= leaf_height - 1);

                                            // Taper the top of the canopy
                                            let at_top = dy >= leaf_height - 1;
                                            let too_far_at_top = at_top && (dx > 1 || dz > 1);

                                            if !skip_corner && !too_far_at_top && self.blocks[lx][ly][lz] == BlockType::Air {
                                                self.blocks[lx][ly][lz] = BlockType::Leaves;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let glow_perlin = Perlin::new(43); 
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let world_x = (world_offset_x + x as i32) as f64;
                let world_z = (world_offset_z + z as i32) as f64;
                for y in 5..25 {
                    if self.blocks[x][y][z] == BlockType::Stone {
                        let noise_val = glow_perlin.get([world_x * 0.15, y as f64 * 0.15, world_z * 0.15]);
                        if noise_val > 0.85 {
                            self.blocks[x][y][z] = BlockType::GlowStone;
                        }
                    }
                }
            }
        }
    }

    pub fn get_block(&self, x: usize, y: usize, z: usize) -> BlockType {
        if x >= CHUNK_SIZE || y >= CHUNK_HEIGHT || z >= CHUNK_SIZE {
            BlockType::Air
        } else {
            self.blocks[x][y][z]
        }
    }

    pub fn set_block(&mut self, x: usize, y: usize, z: usize, block_type: BlockType) {
        if x < CHUNK_SIZE && y < CHUNK_HEIGHT && z < CHUNK_SIZE {
            let old_block = self.blocks[x][y][z];
            self.blocks[x][y][z] = block_type;
            self.dirty = true;

            if old_block != BlockType::Air && block_type == BlockType::Air {
                lighting::on_block_removed(self, x, y, z);
            } else if block_type != BlockType::Air {
                lighting::on_block_placed(self, x, y, z);
            }
        }
    }

    pub fn get_light(&self, x: usize, y: usize, z: usize) -> u8 {
        if x >= CHUNK_SIZE || y >= CHUNK_HEIGHT || z >= CHUNK_SIZE {
            0
        } else {
            self.light_levels[x][y][z]
        }
    }

    pub fn generate_mesh(neighbors: &ChunkNeighbors) -> (Vec<Vertex>, Vec<u16>, Vec<Vertex>, Vec<u16>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut water_vertices = Vec::new();
        let mut water_indices = Vec::new();
        let chunk = neighbors.center;

        let world_offset_x = chunk.position.0 * CHUNK_SIZE as i32;
        let world_offset_z = chunk.position.1 * CHUNK_SIZE as i32;

        let face_directions: [(i32, i32, i32); 6] = [
            (0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0),
        ];

        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_HEIGHT {
                for z in 0..CHUNK_SIZE {
                    let block = chunk.blocks[x][y][z];
                    if block == BlockType::Air {
                        continue;
                    }

                    let world_pos = Vector3::new(
                        (world_offset_x + x as i32) as f32,
                        y as f32,
                        (world_offset_z + z as i32) as f32,
                    );

                    let is_water = block == BlockType::Water;

                    for (face_idx, &(dx, dy, dz)) in face_directions.iter().enumerate() {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        let nz = z as i32 + dz;

                        // Check neighbor block via the neighbor struct
                        let neighbor_block = neighbors.get_block(nx, ny, nz);

                        // Check if we should draw this face
                        // For water, use special transparency check that treats Boundary as opaque
                        let should_draw = if is_water {
                            neighbor_block.is_transparent_for_water() && neighbor_block != block
                        } else {
                            neighbor_block.is_transparent() && neighbor_block != block
                        };

                        if should_draw {
                            let light = neighbors.get_light(nx, ny, nz);
                            let light_normalized = light as f32 / 15.0;

                            if is_water {
                                // Water uses separate mesh, alpha calculated in shader based on depth
                                let face_verts = create_face_vertices(world_pos, block, face_idx, light_normalized);

                                let base_index = water_vertices.len() as u16;
                                water_vertices.extend_from_slice(&face_verts);
                                water_indices.extend_from_slice(&[
                                    base_index, base_index + 1, base_index + 2,
                                    base_index + 2, base_index + 3, base_index,
                                ]);
                            } else {
                                let face_verts = create_face_vertices(world_pos, block, face_idx, light_normalized);

                                let base_index = vertices.len() as u16;
                                vertices.extend_from_slice(&face_verts);
                                indices.extend_from_slice(&[
                                    base_index, base_index + 1, base_index + 2,
                                    base_index + 2, base_index + 3, base_index,
                                ]);
                            }
                        }
                    }
                }
            }
        }
        (vertices, indices, water_vertices, water_indices)
    }
}