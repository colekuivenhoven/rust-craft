use crate::block::{BlockType, Vertex, create_face_vertices, create_water_face_vertices, AO_OFFSETS, calculate_ao};
use crate::lighting;
use crate::texture::{get_face_uvs, TEX_NONE};
use cgmath::Vector3;
use noise::{NoiseFn, Perlin};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

pub const CHUNK_SIZE: usize = 16;
pub const CHUNK_HEIGHT: usize = 128;

// Tree generation configuration
pub const TREE_MIN_HEIGHT: usize = 4;      // Minimum trunk height (current average)
pub const TREE_MAX_HEIGHT: usize = 12;     // Maximum trunk height (tall trees)
pub const TREE_BRANCH_CHANCE: f64 = 0.6;   // Chance for trunk to branch near top (0.0 - 1.0)
pub const TREE_TALL_CHANCE: f64 = 0.3;     // Chance for a tree to be tall variant (0.0 - 1.0)

pub struct Chunk {
    pub blocks: Box<[[[BlockType; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]>,
    pub light_levels: Box<[[[u8; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]>,
    pub position: (i32, i32),
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
    pub water_vertices: Vec<Vertex>,       // Separate water vertices for transparency pass
    pub water_indices: Vec<u16>,           // Separate water indices for transparency pass
    pub transparent_vertices: Vec<Vertex>, // Semi-transparent blocks (ice) rendered after opaque
    pub transparent_indices: Vec<u16>,     // Semi-transparent block indices
    pub dirty: bool,
    pub light_dirty: bool,
    pub mesh_version: u32,                 // Incremented when mesh is rebuilt, used for GPU buffer caching
    pub modified: bool,                    // True if chunk has player modifications (needs saving)
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

        // Handle diagonal out-of-bounds (corner neighbors crossing chunk boundaries)
        if lx >= CHUNK_SIZE || lz >= CHUNK_SIZE {
            return BlockType::Boundary;
        }

        match target_chunk {
            Some(chunk) => chunk.blocks[lx][y as usize][lz],
            None => BlockType::Boundary, // Virtual block: transparent for solids, opaque for water
        }
    }

    pub fn get_light(&self, x: i32, y: i32, z: i32) -> u8 {
        if y < 0 { return 0; }
        if y >= CHUNK_HEIGHT as i32 { return 15; }

        let (target_chunk, lx, lz) = self.resolve_coordinates(x, z);

        // Handle diagonal out-of-bounds (corner neighbors crossing chunk boundaries)
        if lx >= CHUNK_SIZE || lz >= CHUNK_SIZE {
            return 15;
        }

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
        // Allocate large arrays directly on heap to avoid stack overflow
        // Using vec!().into_boxed_slice().try_into() ensures no stack intermediary
        let blocks: Box<[[[BlockType; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]> = vec![[[BlockType::Air; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]
            .into_boxed_slice()
            .try_into()
            .unwrap();
        let light_levels: Box<[[[u8; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]> = vec![[[0u8; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]
            .into_boxed_slice()
            .try_into()
            .unwrap();

        let mut chunk = Self {
            blocks,
            light_levels,
            position: (chunk_x, chunk_z),
            vertices: Vec::new(),
            indices: Vec::new(),
            water_vertices: Vec::new(),
            water_indices: Vec::new(),
            transparent_vertices: Vec::new(),
            transparent_indices: Vec::new(),
            dirty: true,
            light_dirty: true,
            mesh_version: 0,
            modified: false,
        };
        chunk.generate_terrain();
        chunk
    }

    /// Creates a chunk with pre-loaded block data (from saved file)
    pub fn from_saved_data(chunk_x: i32, chunk_z: i32, blocks: Box<[[[BlockType; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]>) -> Self {
        let light_levels: Box<[[[u8; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]> = vec![[[0u8; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]
            .into_boxed_slice()
            .try_into()
            .unwrap();

        Self {
            blocks,
            light_levels,
            position: (chunk_x, chunk_z),
            vertices: Vec::new(),
            indices: Vec::new(),
            water_vertices: Vec::new(),
            water_indices: Vec::new(),
            transparent_vertices: Vec::new(),
            transparent_indices: Vec::new(),
            dirty: true,
            light_dirty: true,
            mesh_version: 0,
            modified: true, // Loaded chunks are considered modified (already saved once)
        }
    }

    fn generate_terrain(&mut self) {
        let perlin = Perlin::new(42);
        let biome_perlin = Perlin::new(44);     // very low frequency: ocean/land mask
        let mountain_perlin = Perlin::new(45);  // low frequency: mountain mask
        let detail_perlin = Perlin::new(46);    // higher frequency: small variation
        let jagged_perlin = Perlin::new(47);    // high frequency: jagged mountain detail

        // Cave noise generators
        let cave_perlin_1 = Perlin::new(100);   // Primary spaghetti cave noise
        let cave_perlin_2 = Perlin::new(101);   // Secondary spaghetti cave noise (perpendicular)
        let cheese_perlin = Perlin::new(102);   // Large cavern "cheese" caves
        let cave_mask_perlin = Perlin::new(103); // Controls cave density in different areas

        // Sky island noise generators (3D)
        let sky_island_perlin = Perlin::new(200);      // Primary 3D shape
        let sky_island_mask_perlin = Perlin::new(201); // 2D mask for sparse placement
        let sky_island_detail = Perlin::new(202);      // Detail variation

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
                let biome_scale = 0.0012; // Reduced from 0.0035 for ~3x larger oceans/continents
                let biome_raw = (biome_perlin.get([world_x * biome_scale, world_z * biome_scale]) + 1.0) * 0.5;
                let inland = smoothstep(0.35, 0.55, biome_raw); // Push more area into ocean while keeping smooth shores.

                // Mountain mask: 0 = no mountains, 1 = mountains.
                let mountain_scale = 0.002; // Reduced from 0.006 for ~3x larger mountain ranges
                let mountain_raw = (mountain_perlin.get([world_x * mountain_scale, world_z * mountain_scale]) + 1.0) * 0.5;
                let mountains = smoothstep(0.60, 0.80, mountain_raw) * inland; // Only allow strong mountains well inland.

                // --- Base terrain (plains) + local detail ---
                let base_scale = 0.007; // Reduced from 0.02 for ~3x larger terrain features
                let base_n = (perlin.get([world_x * base_scale, world_z * base_scale]) + 1.0) * 0.5;
                let base_height = 18.0 + base_n * 18.0; // ~18..36

                let detail_scale = 0.02; // Reduced from 0.06 for smoother terrain
                let detail = detail_perlin.get([world_x * detail_scale, world_z * detail_scale]) * 2.5;

                // --- Ocean shaping ---
                let ocean_drop = (1.0 - inland) * 16.0; // Pull terrain down near coasts/ocean.

                // --- Mountain shaping (jagged peaks) ---
                // Use higher frequency noise for sharper, more jagged peaks
                let ridge_scale = 0.015; // Reduced from 0.04 for wider mountain ridges
                let ridge = (perlin.get([world_x * ridge_scale, world_z * ridge_scale]).abs()) * 15.0;

                // Add extra high-frequency jagged detail for sharp, craggy look
                let jagged_scale = 0.04; // Reduced from 0.12 for larger jagged features
                let jagged = (jagged_perlin.get([world_x * jagged_scale, world_z * jagged_scale]).abs()) * 8.0;

                // Taller mountains: base 35 + up to 23 from noise = max ~58 blocks above base
                let mountain_lift = mountains * (35.0 + ridge + jagged);

                let height_f = base_height + detail + mountain_lift - ocean_drop;
                let height_i = height_f.round() as isize;

                let height = height_i
                    .clamp(0, (CHUNK_HEIGHT as isize) - 1) as usize;

                // Surface palette with biome-based block selection
                let sea = sea_level as usize;

                // Jagged transition noise for clumpy block type boundaries
                let transition_scale = 0.05; // Reduced from 0.15 for larger block type patches
                let transition_noise = (detail_perlin.get([world_x * transition_scale, world_z * transition_scale]) + 1.0) * 0.5;
                let transition_noise_2 = (jagged_perlin.get([world_x * transition_scale * 1.5, world_z * transition_scale * 1.5]) + 1.0) * 0.5;
                let jagged_offset = ((transition_noise * 8.0) + (transition_noise_2 * 4.0)) as isize - 6; // -6 to +6 block variation

                // Narrow sand band right at coast edge (3-6 blocks from water)
                // inland 0.0-0.15 is the immediate coast edge
                let is_beach = inland > 0.0 && inland < 0.15;

                // Mountain stone threshold with jagged variation - raised to allow grass higher
                let mountain_stone_threshold = sea_level + 18 + jagged_offset;
                let is_mountain_zone = mountains > 0.15;

                // Snow threshold - snow appears on the highest mountain peaks
                let snow_threshold = sea_level + 45 + jagged_offset; // Snow at very high elevations

                // Extra noise for grass patches above stone threshold
                let grass_patch_noise = transition_noise * 0.7 + transition_noise_2 * 0.3;

                for y in 0..=height {
                    let y_i = y as isize;

                    // Underwater blocks - mostly sand with sparse deep stone
                    if y <= sea {
                        // Very deep underwater = sparse stone patches, otherwise sand
                        let deep_threshold = (sea_level - 15) as usize; // Deeper threshold
                        // Use noise to make stone sparse (only ~20% of deep areas)
                        let stone_noise = transition_noise * transition_noise_2;
                        let is_deep_stone = y < deep_threshold && stone_noise > 0.4;
                        self.blocks[x][y][z] = if is_deep_stone {
                            BlockType::Stone
                        } else {
                            BlockType::Sand
                        };
                    }
                    // Above water surface blocks
                    else if y == height {
                        // Snow-capped mountain peaks
                        if is_mountain_zone && y_i > snow_threshold {
                            self.blocks[x][y][z] = BlockType::Snow;
                        }
                        // Mountains: stone surface with jagged grass patches
                        else if is_mountain_zone && y_i > mountain_stone_threshold {
                            // How far above threshold determines grass chance
                            let blocks_above = (y_i - mountain_stone_threshold) as f64;
                            // Grass can appear up to 8 blocks above threshold with decreasing probability
                            let grass_chance_threshold = 0.3 + (blocks_above * 0.1); // 0.3 at threshold, 1.1 at +8
                            if grass_patch_noise > grass_chance_threshold && blocks_above < 8.0 {
                                self.blocks[x][y][z] = BlockType::Grass;
                            } else {
                                self.blocks[x][y][z] = BlockType::Stone;
                            }
                        }
                        // Beach: narrow sand band at coast edge with jagged grass patches
                        else if is_beach {
                            // Grass can intrude into beach with decreasing probability toward water
                            // inland 0.0 = water edge, 0.15 = beach/grass boundary
                            let beach_depth = inland / 0.15; // 0.0 at water, 1.0 at grass
                            // More aggressive thresholds: grass intrudes significantly
                            // Near grass boundary: 90% grass, near water: 35% grass
                            let grass_intrusion_threshold = 0.65 - (beach_depth * 0.55);
                            if grass_patch_noise > grass_intrusion_threshold {
                                self.blocks[x][y][z] = BlockType::Grass;
                            } else {
                                self.blocks[x][y][z] = BlockType::Sand;
                            }
                        }
                        // Default: grass everywhere else
                        else {
                            self.blocks[x][y][z] = BlockType::Grass;
                        }
                    }
                    // Below surface blocks
                    else {
                        let depth_below_surface = height - y;

                        // Mountains: mostly stone with jagged transition
                        if is_mountain_zone && y_i > mountain_stone_threshold - 4 {
                            self.blocks[x][y][z] = BlockType::Stone;
                        }
                        // Near surface (1-3 blocks down)
                        else if depth_below_surface <= 3 {
                            // Beach areas get sand underneath, unless grass intruded
                            if is_beach {
                                let beach_depth = inland / 0.15;
                                let grass_intrusion_threshold = 0.65 - (beach_depth * 0.55);
                                if grass_patch_noise > grass_intrusion_threshold {
                                    self.blocks[x][y][z] = BlockType::Dirt; // Under grass
                                } else {
                                    self.blocks[x][y][z] = BlockType::Sand; // Under sand
                                }
                            } else {
                                self.blocks[x][y][z] = BlockType::Dirt;
                            }
                        }
                        // Deeper underground
                        else {
                            self.blocks[x][y][z] = BlockType::Stone;
                        }
                    }
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
        // Carve caves using 3D ridged noise for natural, connected tunnel networks
        // Uses multiple layers: winding tunnels, large caverns, and vertical shafts

        let cave_floor = 3;  // Don't carve below this (bedrock layer)
        let surface_opening_perlin = Perlin::new(104);  // Controls where caves can reach surface
        let cave_worm_perlin = Perlin::new(105);  // Additional worm-like cave layer
        let vertical_shaft_perlin = Perlin::new(106);  // Vertical shafts connecting caves

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
                let cave_density = (cave_mask_perlin.get([world_x * 0.002, world_z * 0.002]) + 1.0) * 0.5;

                // Surface opening chance - very rare spots where caves can reach surface
                let surface_opening_noise = surface_opening_perlin.get([world_x * 0.003, world_z * 0.003]);
                let allow_surface_opening = surface_opening_noise > 0.95; // Only ~2.5% of terrain can have openings
                let min_surface_depth = if allow_surface_opening { 2 } else { 8 }; // Even openings stay 2 blocks down, normal caves 8 blocks

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

                    // === Primary Cave Network (Ridged Noise) ===
                    // Using ridged noise creates more natural, connected tunnel networks
                    let cave_scale_1 = 0.025;
                    let cave_noise_1 = cave_perlin_1.get([
                        world_x * cave_scale_1,
                        world_y * cave_scale_1 * 0.4,  // Stretch vertically for horizontal bias
                        world_z * cave_scale_1
                    ]);
                    // Ridged noise: invert the absolute value to create ridges (tunnels along ridges)
                    let ridged_1 = 1.0 - cave_noise_1.abs() * 2.0;

                    // Second layer at different scale for variety
                    let cave_scale_2 = 0.04;
                    let cave_noise_2 = cave_perlin_2.get([
                        world_x * cave_scale_2 + 100.0,  // Offset to decorrelate
                        world_y * cave_scale_2 * 0.5,
                        world_z * cave_scale_2 + 100.0
                    ]);
                    let ridged_2 = 1.0 - cave_noise_2.abs() * 2.0;

                    // Combine ridged noise layers - tunnel forms where ridged value is high
                    let combined_ridged = (ridged_1 * 0.6 + ridged_2 * 0.4).max(ridged_1.max(ridged_2) * 0.8);
                    let tunnel_threshold = 0.75 + (1.0 - cave_density) * 0.12; // Increased from 0.65 for fewer caves
                    let is_tunnel = combined_ridged > tunnel_threshold && cave_density > 0.3;

                    // === Worm Caves (meandering tunnels) ===
                    // Creates longer, more connected cave systems
                    let worm_scale = 0.018;
                    let worm_noise = cave_worm_perlin.get([
                        world_x * worm_scale,
                        world_y * worm_scale * 0.3,  // Very horizontal bias
                        world_z * worm_scale
                    ]);
                    let worm_ridged = 1.0 - worm_noise.abs() * 1.8;
                    let worm_threshold = 0.80 + (1.0 - cave_density) * 0.08; // Increased from 0.72 for fewer caves
                    let is_worm = worm_ridged > worm_threshold && cave_density > 0.4;

                    // === Cheese Caves (larger caverns) ===
                    let cheese_scale = 0.012;
                    let cheese_value = cheese_perlin.get([
                        world_x * cheese_scale,
                        world_y * cheese_scale * 0.5,
                        world_z * cheese_scale
                    ]);

                    // Cheese caves stronger at lower elevations
                    let cheese_y_factor = (1.0 - (world_y / 35.0).min(1.0)).max(0.0);
                    let cheese_threshold = 0.60 + (1.0 - cave_density) * 0.12 - cheese_y_factor * 0.10; // Increased from 0.50 for fewer caves
                    let is_cheese = cheese_value > cheese_threshold && cave_density > 0.45;

                    // === Vertical Shafts (connect cave levels) ===
                    let shaft_scale = 0.05;
                    let shaft_noise = vertical_shaft_perlin.get([
                        world_x * shaft_scale,
                        world_z * shaft_scale
                    ]);
                    // Shafts are rare but connect caves vertically - must be deep underground
                    let is_shaft = shaft_noise > 0.90 && y < surface_y.saturating_sub(15); // Increased threshold, deeper requirement

                    // Carve the cave
                    if is_tunnel || is_worm || is_cheese || is_shaft {
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
                        let tree_noise = perlin.get([world_x * 0.02, world_z * 0.02]); // Reduced from 0.1 for larger forest patches
                        if tree_noise > 0.25 { // Lowered from 0.7 for more frequent trees
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
                        let noise_val = glow_perlin.get([world_x * 0.05, y as f64 * 0.05, world_z * 0.05]); // Reduced from 0.15 for larger clusters
                        if noise_val > 0.85 {
                            self.blocks[x][y][z] = BlockType::GlowStone;
                        }
                    }
                }
            }
        }

        // === Floating Sky Islands ===
        // Generate sparse, hilly islands at various heights with stalactites
        let sky_island_base_y = 60;          // Lower base height
        let sky_island_height_range = 40;    // More height variation (60-100)
        let stalactite_perlin = Perlin::new(203); // For bottom stalactites
        let hill_perlin = Perlin::new(204);       // For top hills

        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let world_x = (world_offset_x + x as i32) as f64;
                let world_z = (world_offset_z + z as i32) as f64;

                // 2D mask for very sparse island placement
                let mask_scale = 0.0017; // Reduced from 0.005 for ~3x larger island spacing
                let island_mask = (sky_island_mask_perlin.get([world_x * mask_scale, world_z * mask_scale]) + 1.0) * 0.5;

                if island_mask < 0.78 {
                    continue;
                }

                // Height of this island cluster
                let height_noise = (sky_island_detail.get([world_x * 0.001, world_z * 0.001]) + 1.0) * 0.5; // Reduced from 0.003
                let island_center_y = sky_island_base_y + (height_noise * sky_island_height_range as f64) as usize;

                // Hills on top - adds 0-4 blocks of height variation
                let hill_scale = 0.025; // Reduced from 0.08 for larger hills
                let hill_noise = (hill_perlin.get([world_x * hill_scale, world_z * hill_scale]) + 1.0) * 0.5;
                let hill_height = (hill_noise * 4.0) as usize;

                // Calculate "centeredness" - how close this point is to the core of the island
                // Use horizontal slice of island noise to determine center
                let island_scale = 0.008; // Reduced from 0.025 for ~3x larger islands
                let horizontal_island_noise = (sky_island_perlin.get([
                    world_x * island_scale,
                    island_center_y as f64 * island_scale * 0.25, // Sample at island center height
                    world_z * island_scale,
                ]) + 1.0) * 0.5;

                // Centeredness: higher noise = closer to island core
                let centeredness = smoothstep(0.5, 0.8, horizontal_island_noise);

                // Stalactites - longer in general, and much longer/thicker toward center
                let stalactite_scale = 0.05; // Reduced from 0.15 for larger stalactite patterns
                let stalactite_noise = (stalactite_perlin.get([world_x * stalactite_scale, world_z * stalactite_scale]) + 1.0) * 0.5;

                // Base stalactite depth (longer in general: 2-8 blocks at edges)
                let base_stalactite = if stalactite_noise > 0.25 {
                    2 + ((stalactite_noise - 0.25) * 10.0) as usize // 2-8 blocks
                } else {
                    0
                };

                // Center bonus: add up to 8 more blocks at the very center
                let center_bonus = (centeredness * 8.0) as usize;

                // Combined stalactite depth
                let stalactite_depth = base_stalactite + center_bonus;

                // Base island thickness
                let base_thickness = 5;
                let island_min_y = island_center_y.saturating_sub(base_thickness / 2 + stalactite_depth);
                let island_max_y = (island_center_y + base_thickness / 2 + hill_height + 2).min(CHUNK_HEIGHT);

                let island_strength = smoothstep(0.78, 0.90, island_mask);

                for y in island_min_y..island_max_y {
                    let world_y = y as f64;

                    // 3D noise for island shape (island_scale defined above)
                    let island_noise = sky_island_perlin.get([
                        world_x * island_scale,
                        world_y * island_scale * 0.25,
                        world_z * island_scale,
                    ]);

                    // Adjusted center accounting for hills
                    let effective_center_y = (island_center_y + hill_height / 2) as f64;
                    let effective_thickness = base_thickness as f64 + hill_height as f64 / 2.0 + stalactite_depth as f64 / 2.0;

                    // Vertical falloff
                    let y_dist = (world_y - effective_center_y).abs() / (effective_thickness / 2.0 + 1.0);
                    let y_falloff = (1.0 - y_dist.powi(2)).max(0.0);

                    // Bottom tapers more, with stalactites extending further at center
                    let is_below_center = world_y < effective_center_y;
                    let taper = if is_below_center {
                        // Stalactite regions taper less to allow them to extend down
                        // More centeredness = less taper = longer stalactites
                        let base_taper = 0.5;
                        let center_taper_bonus = centeredness * 0.4; // Up to 0.9 at center
                        y_falloff * (base_taper + center_taper_bonus)
                    } else {
                        y_falloff
                    };

                    let threshold = 0.45 - (island_strength * 0.25) - (taper * 0.2);

                    if island_noise > threshold && self.blocks[x][y][z] == BlockType::Air {
                        // Check if this is a surface block
                        let is_surface = y + 1 >= island_max_y || {
                            let above_y = (y + 1) as f64;
                            let above_noise = sky_island_perlin.get([
                                world_x * island_scale,
                                above_y * island_scale * 0.25,
                                world_z * island_scale,
                            ]);
                            let above_y_dist = (above_y - effective_center_y).abs() / (effective_thickness / 2.0 + 1.0);
                            let above_y_falloff = (1.0 - above_y_dist.powi(2)).max(0.0);
                            let above_taper = if above_y < effective_center_y {
                                let base_taper = 0.5;
                                let center_taper_bonus = centeredness * 0.4;
                                above_y_falloff * (base_taper + center_taper_bonus)
                            } else {
                                above_y_falloff
                            };
                            let above_threshold = 0.45 - (island_strength * 0.25) - (above_taper * 0.2);
                            above_noise <= above_threshold
                        };

                        // Depth from surface for block type
                        let mut depth_from_surface = 0;
                        for check_y in (y + 1)..island_max_y.min(y + 5) {
                            let check_world_y = check_y as f64;
                            let check_noise = sky_island_perlin.get([
                                world_x * island_scale,
                                check_world_y * island_scale * 0.25,
                                world_z * island_scale,
                            ]);
                            let check_y_dist = (check_world_y - effective_center_y).abs() / (effective_thickness / 2.0 + 1.0);
                            let check_y_falloff = (1.0 - check_y_dist.powi(2)).max(0.0);
                            let check_taper = if check_world_y < effective_center_y {
                                let base_taper = 0.5;
                                let center_taper_bonus = centeredness * 0.4;
                                check_y_falloff * (base_taper + center_taper_bonus)
                            } else {
                                check_y_falloff
                            };
                            let check_threshold = 0.45 - (island_strength * 0.25) - (check_taper * 0.2);
                            if check_noise > check_threshold {
                                depth_from_surface += 1;
                            } else {
                                break;
                            }
                        }

                        // Stalactites (below center) are always stone, top surface gets grass/dirt
                        self.blocks[x][y][z] = if is_below_center {
                            BlockType::Stone // Stalactites are always stone
                        } else if is_surface {
                            BlockType::Grass
                        } else if depth_from_surface < 3 {
                            BlockType::Dirt
                        } else {
                            BlockType::Stone
                        };
                    }
                }
            }
        }

        // === Sky Island Ponds ===
        // Create large ponds using flood-fill to ensure contiguous, properly-bounded water bodies
        // Minimum size: 6x6 (36 blocks). Ponds are excavated basins filled with water at a flat level.

        let pond_noise = Perlin::new(205);

        // Track which positions have been used for ponds (to avoid overlaps)
        let mut used_for_pond = [[false; CHUNK_SIZE]; CHUNK_SIZE];

        // Find pond seed points - locations where we'll try to create ponds
        let mut pond_seeds: Vec<(usize, usize, usize)> = Vec::new(); // (x, z, surface_y)

        for x in 6..CHUNK_SIZE - 6 {
            for z in 6..CHUNK_SIZE - 6 {
                let world_x = (world_offset_x + x as i32) as f64;
                let world_z = (world_offset_z + z as i32) as f64;

                // Check if we're on a floating island using the island mask
                let mask_scale = 0.005;
                let island_mask = (sky_island_mask_perlin.get([world_x * mask_scale, world_z * mask_scale]) + 1.0) * 0.5;
                if island_mask < 0.82 {
                    continue;
                }

                // Use noise to create sparse pond locations
                let pond_chance = (pond_noise.get([world_x * 0.009, world_z * 0.009]) + 1.0) * 0.5; // Reduced from 0.02
                if pond_chance < 0.50 {
                    continue;
                }

                // Find the grass surface at this location
                for y in (sky_island_base_y..CHUNK_HEIGHT - 5).rev() {
                    if self.blocks[x][y][z] != BlockType::Grass {
                        continue;
                    }

                    // Verify this is on a floating island (has air below at some point)
                    let mut is_floating = false;
                    for check_y in (sky_island_base_y.saturating_sub(10))..y {
                        if self.blocks[x][check_y][z] == BlockType::Air {
                            is_floating = true;
                            break;
                        }
                    }
                    if !is_floating {
                        continue;
                    }

                    // Check not already used
                    if used_for_pond[x][z] {
                        continue;
                    }

                    pond_seeds.push((x, z, y));
                    break;
                }
            }
        }

        // For each seed, try to create a pond using flood-fill
        for (seed_x, seed_z, seed_y) in pond_seeds {
            if used_for_pond[seed_x][seed_z] {
                continue;
            }

            // Flood-fill to find contiguous solid region
            // We'll find all positions that:
            // 1. Are within chunk bounds (with buffer)
            // 2. Have solid surface (grass/dirt) within a few Y levels of the seed
            // 3. Have solid ground below (at least 5 blocks deep for excavation)
            // 4. Are connected to the seed position

            let mut queue: Vec<(usize, usize)> = Vec::new();
            let mut visited = [[false; CHUNK_SIZE]; CHUNK_SIZE];
            let mut region: Vec<(usize, usize, usize)> = Vec::new(); // (x, z, surface_y)

            queue.push((seed_x, seed_z));
            visited[seed_x][seed_z] = true;

            let max_pond_radius = 12; // Maximum distance from seed

            while let Some((x, z)) = queue.pop() {
                // Check distance from seed (limit pond size for performance and aesthetics)
                let dx = (x as i32 - seed_x as i32).abs();
                let dz = (z as i32 - seed_z as i32).abs();
                if dx > max_pond_radius || dz > max_pond_radius {
                    continue;
                }

                // Check chunk bounds with buffer
                if x < 3 || x >= CHUNK_SIZE - 3 || z < 3 || z >= CHUNK_SIZE - 3 {
                    continue;
                }

                // Find surface at this position (within a few Y levels of seed)
                let mut surface_y = None;
                for y in (seed_y.saturating_sub(4)..=seed_y + 4).rev() {
                    if y < CHUNK_HEIGHT {
                        let block = self.blocks[x][y][z];
                        if block == BlockType::Grass || block == BlockType::Dirt {
                            surface_y = Some(y);
                            break;
                        }
                    }
                }

                let sy = match surface_y {
                    Some(y) => y,
                    None => continue, // No solid surface here
                };

                // Check we have solid ground below for excavation (5 blocks minimum)
                let mut solid_below = true;
                for depth in 1..=5 {
                    if sy < depth {
                        solid_below = false;
                        break;
                    }
                    let block = self.blocks[x][sy - depth][z];
                    if block == BlockType::Air {
                        solid_below = false;
                        break;
                    }
                }
                if !solid_below {
                    continue;
                }

                // Check immediate neighbors for air at surface level (edge detection)
                // If any direct neighbor is air, we're at the island edge - don't include
                let mut at_edge = false;
                for (nx, nz) in [(x.wrapping_sub(1), z), (x + 1, z), (x, z.wrapping_sub(1)), (x, z + 1)] {
                    if nx < CHUNK_SIZE && nz < CHUNK_SIZE {
                        if self.blocks[nx][sy][nz] == BlockType::Air {
                            at_edge = true;
                            break;
                        }
                    }
                }
                if at_edge {
                    continue;
                }

                // This position is valid - add to region
                region.push((x, z, sy));

                // Add unvisited neighbors to queue
                for (nx, nz) in [(x.wrapping_sub(1), z), (x + 1, z), (x, z.wrapping_sub(1)), (x, z + 1)] {
                    if nx < CHUNK_SIZE && nz < CHUNK_SIZE && !visited[nx][nz] && !used_for_pond[nx][nz] {
                        visited[nx][nz] = true;
                        queue.push((nx, nz));
                    }
                }
            }

            // Check if region is large enough (minimum 36 = 6x6)
            if region.len() < 36 {
                continue;
            }

            // Find the bounding box and check it's not too elongated
            let min_x = region.iter().map(|(x, _, _)| *x).min().unwrap();
            let max_x = region.iter().map(|(x, _, _)| *x).max().unwrap();
            let min_z = region.iter().map(|(_, z, _)| *z).min().unwrap();
            let max_z = region.iter().map(|(_, z, _)| *z).max().unwrap();
            let width = max_x - min_x + 1;
            let depth = max_z - min_z + 1;

            // Skip if too narrow in either dimension (want at least 6 in both)
            if width < 6 || depth < 6 {
                continue;
            }

            // Find water surface level (use the highest surface Y in the region)
            let water_level = region.iter().map(|(_, _, y)| *y).max().unwrap();

            // Calculate center of the region for depth calculations
            let center_x = (min_x + max_x) / 2;
            let center_z = (min_z + max_z) / 2;
            let max_dist = ((width.max(depth) / 2) as f64).max(1.0);

            // Mark all positions as used
            for (x, z, _) in &region {
                used_for_pond[*x][*z] = true;
            }

            // Excavate basin and fill with water
            for (x, z, _) in &region {
                // Calculate depth based on distance from center (deeper in middle)
                let dist_x = (*x as f64 - center_x as f64).abs();
                let dist_z = (*z as f64 - center_z as f64).abs();
                let dist = (dist_x * dist_x + dist_z * dist_z).sqrt();
                let normalized_dist = (dist / max_dist).min(1.0);

                let pond_depth = if normalized_dist < 0.3 {
                    4 // Center: 4 blocks deep
                } else if normalized_dist < 0.5 {
                    3 // Inner ring: 3 blocks deep
                } else if normalized_dist < 0.7 {
                    2 // Middle: 2 blocks deep
                } else {
                    1 // Edge: 1 block deep
                };

                // Excavate from water_level down to water_level - pond_depth + 1
                // Then fill with water
                let bottom_y = water_level.saturating_sub(pond_depth - 1);

                for y in bottom_y..=water_level {
                    if y >= CHUNK_HEIGHT {
                        continue;
                    }
                    let block = self.blocks[*x][y][*z];
                    // Don't break through air or stone
                    if block != BlockType::Air && block != BlockType::Stone {
                        self.blocks[*x][y][*z] = BlockType::Water;
                    }
                }

                // Place dirt floor under the water
                if bottom_y > 0 {
                    let floor_y = bottom_y - 1;
                    let floor_block = self.blocks[*x][floor_y][*z];
                    if floor_block != BlockType::Air && floor_block != BlockType::Stone && floor_block != BlockType::Water {
                        self.blocks[*x][floor_y][*z] = BlockType::Dirt;
                    }
                }
            }
        }

        // === Sky Island Trees ===
        // Generate trees on sky island grass blocks
        for x in 3..CHUNK_SIZE - 3 {
            for z in 3..CHUNK_SIZE - 3 {
                let world_x = (world_offset_x + x as i32) as f64;
                let world_z = (world_offset_z + z as i32) as f64;

                // Find grass blocks in the sky island range
                for y in sky_island_base_y..CHUNK_HEIGHT - 12 {
                    if self.blocks[x][y][z] != BlockType::Grass {
                        continue;
                    }

                    // Make sure this is a sky island (above normal terrain)
                    // Check that there's air below at some point (floating)
                    let mut is_floating = false;
                    for check_y in (sky_island_base_y - 10)..y {
                        if self.blocks[x][check_y][z] == BlockType::Air {
                            is_floating = true;
                            break;
                        }
                    }
                    if !is_floating {
                        continue;
                    }

                    // Tree placement noise
                    let tree_noise = perlin.get([world_x * 0.05, world_z * 0.05]); // Reduced from 0.15 for larger forest patches
                    if tree_noise < 0.4 { // Lowered from 0.6 for more trees on sky islands
                        continue;
                    }

                    // Check for nearby trees
                    let mut too_close = false;
                    'check: for check_x in x.saturating_sub(2)..=(x + 2).min(CHUNK_SIZE - 1) {
                        for check_z in z.saturating_sub(2)..=(z + 2).min(CHUNK_SIZE - 1) {
                            if check_x == x && check_z == z {
                                continue;
                            }
                            if y + 1 < CHUNK_HEIGHT && self.blocks[check_x][y + 1][check_z] == BlockType::Wood {
                                too_close = true;
                                break 'check;
                            }
                        }
                    }
                    if too_close {
                        continue;
                    }

                    // Generate a small tree (sky island trees are shorter)
                    let tree_seed = ((world_x as i64).wrapping_mul(73856093) ^ (world_z as i64).wrapping_mul(19349663) ^ (y as i64 * 83492791)) as u64;
                    let mut rng = StdRng::seed_from_u64(tree_seed);
                    let trunk_height = rng.gen_range(3..=5);

                    // Trunk
                    for trunk_y in (y + 1)..=(y + trunk_height).min(CHUNK_HEIGHT - 1) {
                        self.blocks[x][trunk_y][z] = BlockType::Wood;
                    }

                    // Leaves
                    let leaf_radius = 2;
                    let leaf_start_y = y + trunk_height - 1;
                    let leaf_height = 3;

                    for lx in x.saturating_sub(leaf_radius)..=(x + leaf_radius).min(CHUNK_SIZE - 1) {
                        for lz in z.saturating_sub(leaf_radius)..=(z + leaf_radius).min(CHUNK_SIZE - 1) {
                            for ly in leaf_start_y..(leaf_start_y + leaf_height).min(CHUNK_HEIGHT) {
                                let dx = (lx as i32 - x as i32).abs();
                                let dz = (lz as i32 - z as i32).abs();
                                let dy = ly - leaf_start_y;

                                // Skip corners
                                let is_corner = dx == leaf_radius as i32 && dz == leaf_radius as i32;
                                if is_corner {
                                    continue;
                                }

                                // Taper top
                                if dy >= leaf_height - 1 && (dx > 1 || dz > 1) {
                                    continue;
                                }

                                if self.blocks[lx][ly][lz] == BlockType::Air {
                                    self.blocks[lx][ly][lz] = BlockType::Leaves;
                                }
                            }
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
            self.modified = true; // Mark chunk as needing to be saved

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

    pub fn generate_mesh(neighbors: &ChunkNeighbors) -> (Vec<Vertex>, Vec<u16>, Vec<Vertex>, Vec<u16>, Vec<Vertex>, Vec<u16>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut water_vertices = Vec::new();
        let mut water_indices = Vec::new();
        let mut transparent_vertices = Vec::new();
        let mut transparent_indices = Vec::new();
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
                        // For semi-transparent blocks (ice), always draw faces against any different block
                        // For transparent blocks (leaves), draw faces against any different block type
                        let is_semi_transparent = block.is_semi_transparent();
                        let should_draw = if is_water {
                            neighbor_block.is_transparent_for_water() && neighbor_block != block
                        } else if is_semi_transparent {
                            // Semi-transparent blocks always draw all faces (except against same block type)
                            neighbor_block != block
                        } else {
                            (block.is_transparent() || neighbor_block.is_transparent() || neighbor_block.is_semi_transparent()) && neighbor_block != block
                        };

                        if should_draw {
                            let light = neighbors.get_light(nx, ny, nz);
                            let light_normalized = light as f32 / 15.0;

                            // Check if block above is solid (for grass/dirt texture selection)
                            let block_above = neighbors.get_block(x as i32, y as i32 + 1, z as i32);
                            let has_block_above = block_above.is_solid();

                            // Get texture info for this block
                            let face_textures = block.get_face_textures(has_block_above);
                            let tex_index = face_textures.get_for_face(face_idx);

                            // Get UVs for this texture (or default for non-textured)
                            let uvs = if tex_index != TEX_NONE {
                                get_face_uvs(tex_index)
                            } else {
                                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
                            };

                            // Calculate ambient occlusion for each vertex
                            let ao_values: [f32; 4] = std::array::from_fn(|v| {
                                let offsets = &AO_OFFSETS[face_idx][v];
                                let s1 = neighbors.get_block(
                                    x as i32 + offsets[0][0],
                                    y as i32 + offsets[0][1],
                                    z as i32 + offsets[0][2]
                                ).is_solid();
                                let s2 = neighbors.get_block(
                                    x as i32 + offsets[1][0],
                                    y as i32 + offsets[1][1],
                                    z as i32 + offsets[1][2]
                                ).is_solid();
                                let corner = neighbors.get_block(
                                    x as i32 + offsets[2][0],
                                    y as i32 + offsets[2][1],
                                    z as i32 + offsets[2][2]
                                ).is_solid();
                                calculate_ao(s1, s2, corner)
                            });

                            if is_water {
                                // Check if this is surface water (no water above)
                                let block_above = neighbors.get_block(x as i32, y as i32 + 1, z as i32);
                                let is_surface_water = block_above != BlockType::Water;

                                // Calculate edge flags for foam rendering as a bitmask
                                // Encodes which edges have solid neighbors (same value for all vertices to avoid interpolation)
                                // Bitmask: neg_x=1, pos_x=2, neg_z=4, pos_z=8 (divided by 16 to fit in 0-1)
                                let edge_flags: f32 = if face_idx == 2 {
                                    // Top face - check horizontal neighbors for shore foam
                                    let solid_neg_x = neighbors.get_block(x as i32 - 1, y as i32, z as i32).is_solid();
                                    let solid_pos_x = neighbors.get_block(x as i32 + 1, y as i32, z as i32).is_solid();
                                    let solid_neg_z = neighbors.get_block(x as i32, y as i32, z as i32 - 1).is_solid();
                                    let solid_pos_z = neighbors.get_block(x as i32, y as i32, z as i32 + 1).is_solid();

                                    // Encode as bitmask (all vertices get same value - no interpolation issues)
                                    let mut flags: u32 = 0;
                                    if solid_neg_x { flags |= 1; }  // bit 0
                                    if solid_pos_x { flags |= 2; }  // bit 1
                                    if solid_neg_z { flags |= 4; }  // bit 2
                                    if solid_pos_z { flags |= 8; }  // bit 3
                                    flags as f32 / 16.0  // Normalize to 0-1 range (max value is 15)
                                } else {
                                    0.0
                                };
                                // All 4 vertices get the same edge_flags value
                                let edge_factors: [f32; 4] = [edge_flags, edge_flags, edge_flags, edge_flags];

                                // Water uses separate mesh with wave factor encoded in alpha
                                let face_verts = create_water_face_vertices(
                                    world_pos, face_idx, light_normalized, tex_index, uvs, edge_factors, is_surface_water
                                );

                                let base_index = water_vertices.len() as u16;
                                water_vertices.extend_from_slice(&face_verts);

                                // Anisotropy fix: flip diagonal if it reduces AO discontinuity
                                if (ao_values[0] - ao_values[2]).abs() > (ao_values[1] - ao_values[3]).abs() {
                                    water_indices.extend_from_slice(&[
                                        base_index + 1, base_index + 2, base_index + 3,
                                        base_index + 3, base_index, base_index + 1,
                                    ]);
                                } else {
                                    water_indices.extend_from_slice(&[
                                        base_index, base_index + 1, base_index + 2,
                                        base_index + 2, base_index + 3, base_index,
                                    ]);
                                }
                            } else if is_semi_transparent {
                                // Semi-transparent blocks (ice) go to separate mesh rendered after opaque
                                use crate::block::create_face_vertices_with_alpha;
                                let alpha = block.get_alpha();
                                let face_verts = create_face_vertices_with_alpha(
                                    world_pos, block, face_idx, light_normalized, alpha, tex_index, uvs, ao_values
                                );

                                let base_index = transparent_vertices.len() as u16;
                                transparent_vertices.extend_from_slice(&face_verts);

                                // Anisotropy fix: flip diagonal if it reduces AO discontinuity
                                if (ao_values[0] - ao_values[2]).abs() > (ao_values[1] - ao_values[3]).abs() {
                                    transparent_indices.extend_from_slice(&[
                                        base_index + 1, base_index + 2, base_index + 3,
                                        base_index + 3, base_index, base_index + 1,
                                    ]);
                                } else {
                                    transparent_indices.extend_from_slice(&[
                                        base_index, base_index + 1, base_index + 2,
                                        base_index + 2, base_index + 3, base_index,
                                    ]);
                                }
                            } else {
                                let face_verts = create_face_vertices(world_pos, block, face_idx, light_normalized, tex_index, uvs, ao_values);

                                let base_index = vertices.len() as u16;
                                vertices.extend_from_slice(&face_verts);

                                // Anisotropy fix: flip diagonal if it reduces AO discontinuity
                                if (ao_values[0] - ao_values[2]).abs() > (ao_values[1] - ao_values[3]).abs() {
                                    indices.extend_from_slice(&[
                                        base_index + 1, base_index + 2, base_index + 3,
                                        base_index + 3, base_index, base_index + 1,
                                    ]);
                                    // For transparent blocks (leaves), also add back face (reversed winding)
                                    if block.is_transparent() {
                                        indices.extend_from_slice(&[
                                            base_index + 3, base_index + 2, base_index + 1,
                                            base_index + 1, base_index, base_index + 3,
                                        ]);
                                    }
                                } else {
                                    indices.extend_from_slice(&[
                                        base_index, base_index + 1, base_index + 2,
                                        base_index + 2, base_index + 3, base_index,
                                    ]);
                                    // For transparent blocks (leaves), also add back face (reversed winding)
                                    if block.is_transparent() {
                                        indices.extend_from_slice(&[
                                            base_index + 2, base_index + 1, base_index,
                                            base_index, base_index + 3, base_index + 2,
                                        ]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        (vertices, indices, water_vertices, water_indices, transparent_vertices, transparent_indices)
    }
}