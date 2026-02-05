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

// Biome types for terrain generation
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BiomeType {
    Desert,
    Forest,
    Mountains,
    Arctic,
    Ocean,
}

// Biome weights for smooth blending between biomes
#[derive(Clone, Copy, Debug)]
pub struct BiomeWeights {
    pub desert: f64,
    pub forest: f64,
    pub mountains: f64,
    pub arctic: f64,
    pub ocean: f64,
}

impl BiomeWeights {
    pub fn new() -> Self {
        Self {
            desert: 0.0,
            forest: 0.0,
            mountains: 0.0,
            arctic: 0.0,
            ocean: 0.0,
        }
    }

    pub fn normalize(&mut self) {
        let total = self.desert + self.forest + self.mountains + self.arctic + self.ocean;
        if total > 0.0 {
            self.desert /= total;
            self.forest /= total;
            self.mountains /= total;
            self.arctic /= total;
            self.ocean /= total;
        }
    }

    pub fn dominant(&self) -> BiomeType {
        let mut max_weight = self.desert;
        let mut dominant = BiomeType::Desert;

        if self.forest > max_weight {
            max_weight = self.forest;
            dominant = BiomeType::Forest;
        }
        if self.mountains > max_weight {
            max_weight = self.mountains;
            dominant = BiomeType::Mountains;
        }
        if self.arctic > max_weight {
            max_weight = self.arctic;
            dominant = BiomeType::Arctic;
        }
        if self.ocean > max_weight {
            dominant = BiomeType::Ocean;
        }
        dominant
    }
}

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
        // === Noise generators ===
        let perlin = Perlin::new(42);
        let detail_perlin = Perlin::new(46);
        let jagged_perlin = Perlin::new(47);

        // Biome noise generators - using large scale for smooth regions
        let temperature_perlin = Perlin::new(300);
        let humidity_perlin = Perlin::new(301);
        let continentalness_perlin = Perlin::new(302);
        let mountain_perlin = Perlin::new(303);

        // Transition/vein noise - creates organic, connected patterns at biome boundaries
        let vein_perlin = Perlin::new(600);
        let vein_detail = Perlin::new(601);

        // Sky island noise generators
        let sky_island_perlin = Perlin::new(200);
        let sky_island_mask_perlin = Perlin::new(201);
        let sky_island_detail = Perlin::new(202);

        // Oasis and special feature noise
        let oasis_perlin = Perlin::new(400);
        let glacier_perlin = Perlin::new(401);

        // Ocean island noise
        let island_perlin = Perlin::new(500);

        let world_offset_x = self.position.0 * CHUNK_SIZE as i32;
        let world_offset_z = self.position.1 * CHUNK_SIZE as i32;
        let sea_level: isize = 32;
        let sea = sea_level as usize;

        fn clamp01(v: f64) -> f64 {
            v.max(0.0).min(1.0)
        }

        fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
            let t = clamp01((x - edge0) / (edge1 - edge0));
            t * t * (3.0 - 2.0 * t)
        }

        fn smootherstep(edge0: f64, edge1: f64, x: f64) -> f64 {
            let t = clamp01((x - edge0) / (edge1 - edge0));
            t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
        }

        // Store biome data for later use
        let mut column_biomes: [[BiomeWeights; CHUNK_SIZE]; CHUNK_SIZE] = [[BiomeWeights::new(); CHUNK_SIZE]; CHUNK_SIZE];

        // === First pass: Generate base terrain with biomes ===
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let world_x = (world_offset_x + x as i32) as f64;
                let world_z = (world_offset_z + z as i32) as f64;

                // === Sample biome parameters ===
                let biome_scale = 0.0025; // 0.0025 for testing, 0.00025 for continent-sized regions

                let raw_temperature = (temperature_perlin.get([world_x * biome_scale, world_z * biome_scale]) + 1.0) * 0.5;
                let raw_humidity = (humidity_perlin.get([world_x * biome_scale * 1.1, world_z * biome_scale * 1.1 + 1000.0]) + 1.0) * 0.5;
                let raw_mountainess = (mountain_perlin.get([world_x * biome_scale * 1.5, world_z * biome_scale * 1.5 + 3000.0]) + 1.0) * 0.5;

                // Continentalness with wider coastal zone
                let continental_scale = biome_scale * 0.7;
                let raw_continentalness = (continentalness_perlin.get([world_x * continental_scale, world_z * continental_scale + 2000.0]) + 1.0) * 0.5;

                // === Calculate biome weights ===
                let mut biome_weights = BiomeWeights::new();

                // Ocean: wider transition zone (0.25 - 0.50)
                if raw_continentalness < 0.25 {
                    biome_weights.ocean = 1.0;
                } else if raw_continentalness < 0.50 {
                    biome_weights.ocean = smootherstep(0.50, 0.25, raw_continentalness);
                }

                let land_factor = 1.0 - biome_weights.ocean;

                if land_factor > 0.0 {
                    // Arctic: cold temperatures
                    let arctic_factor = smootherstep(0.35, 0.15, raw_temperature);
                    biome_weights.arctic = arctic_factor * land_factor;

                    // Desert: hot, dry, and inland (but NOT near ocean at all)
                    let desert_temp = smootherstep(0.55, 0.75, raw_temperature);
                    let desert_dry = smootherstep(0.50, 0.25, raw_humidity);
                    // Deserts require high continentalness to stay away from water
                    let desert_inland = smootherstep(0.60, 0.80, raw_continentalness);
                    biome_weights.desert = desert_temp * desert_dry * desert_inland * land_factor * (1.0 - biome_weights.arctic);

                    // Mountains
                    let mountain_factor = smootherstep(0.55, 0.75, raw_mountainess);
                    let remaining = land_factor * (1.0 - biome_weights.arctic) * (1.0 - biome_weights.desert * 0.8);
                    biome_weights.mountains = mountain_factor * remaining;

                    // Forest: everything else on land
                    let used = biome_weights.arctic + biome_weights.desert + biome_weights.mountains;
                    biome_weights.forest = (land_factor - used).max(0.0);
                }

                biome_weights.normalize();
                column_biomes[x][z] = biome_weights;

                let dominant_biome = biome_weights.dominant();

                // === Vein noise for organic biome transitions ===
                // This creates connected, branching patterns instead of polka dots
                let vein_scale = 0.02; // Medium frequency for visible veins
                let vein_n1 = vein_perlin.get([world_x * vein_scale, world_z * vein_scale]);
                let vein_n2 = vein_detail.get([world_x * vein_scale * 2.3, world_z * vein_scale * 2.3]);
                // Combine to create fractal vein pattern
                let vein_value = (vein_n1 * 0.7 + vein_n2 * 0.3 + 1.0) * 0.5; // 0-1

                // === Calculate terrain height ===
                let base_scale = 0.006;
                let base_n = (perlin.get([world_x * base_scale, world_z * base_scale]) + 1.0) * 0.5;

                let detail_scale = 0.06;
                let detail = detail_perlin.get([world_x * detail_scale, world_z * detail_scale]) * 1.5;

                // === Ocean islands with GRADUAL slopes ===
                let island_scale = 0.008; // Smaller scale = larger, smoother islands
                let island_noise = (island_perlin.get([world_x * island_scale + 5000.0, world_z * island_scale + 5000.0]) + 1.0) * 0.5;

                // Gradual island bump - squared falloff for smooth slopes
                let island_bump = if biome_weights.ocean > 0.5 && island_noise > 0.65 {
                    let island_strength = smootherstep(0.65, 0.85, island_noise);
                    // Squared for gradual slope, not linear
                    island_strength * island_strength * 22.0
                } else {
                    0.0
                };

                // === Height per biome ===
                // Ocean floor with gradual island rise
                let ocean_base = 12.0 + base_n * 15.0;
                let ocean_height = ocean_base + island_bump;

                // Desert: ALWAYS well above sea level (minimum 38)
                let desert_height = (40.0 + base_n * 5.0 + detail * 0.2).max(sea_level as f64 + 8.0);

                // Forest: gentle rolling hills
                let forest_height = 36.0 + base_n * 8.0 + detail;

                // Arctic: varied terrain
                let arctic_height = 35.0 + base_n * 10.0 + detail;

                // Mountains: dramatic peaks
                let ridge_scale = 0.015;
                let ridge = perlin.get([world_x * ridge_scale, world_z * ridge_scale]).abs() * 15.0;
                let jagged_scale = 0.04;
                let jagged = jagged_perlin.get([world_x * jagged_scale, world_z * jagged_scale]).abs() * 8.0;
                let mountain_height = 42.0 + base_n * 12.0 + ridge + jagged + detail;

                // === Blend heights using biome weights ===
                let mut blended_height =
                    biome_weights.ocean * ocean_height +
                    biome_weights.desert * desert_height +
                    biome_weights.forest * forest_height +
                    biome_weights.mountains * mountain_height +
                    biome_weights.arctic * arctic_height;

                // === CRITICAL: Ensure desert areas stay above sea level ===
                // If desert has significant weight, pull height UP toward desert height
                if biome_weights.desert > 0.2 {
                    let desert_pull = smootherstep(0.2, 0.6, biome_weights.desert);
                    let min_desert_height = sea_level as f64 + 6.0;
                    blended_height = blended_height.max(min_desert_height * desert_pull + blended_height * (1.0 - desert_pull));
                }

                // === Coastal smoothing - WIDER zone with gentler slopes ===
                let height_f = if raw_continentalness > 0.25 && raw_continentalness < 0.55 {
                    // Coastal transition zone (0.25 - 0.55 = 30% of range)
                    let shore_t = smootherstep(0.25, 0.55, raw_continentalness);
                    // Start from near ocean floor, not sea level
                    let min_shore = 20.0 + base_n * 8.0; // Underwater starting point
                    let max_shore = blended_height;
                    min_shore + (max_shore - min_shore) * shore_t
                } else {
                    blended_height
                };

                let height = (height_f.round() as isize).clamp(1, CHUNK_HEIGHT as isize - 1) as usize;

                // === Surface block noise for variations ===
                let transition_scale = 0.03;
                let transition_noise = (detail_perlin.get([world_x * transition_scale, world_z * transition_scale]) + 1.0) * 0.5;
                let transition_noise_2 = (jagged_perlin.get([world_x * transition_scale * 1.3, world_z * transition_scale * 1.3]) + 1.0) * 0.5;
                let jagged_offset = ((transition_noise * 5.0) + (transition_noise_2 * 3.0)) as isize - 4;

                let snow_threshold = sea_level + 38 + jagged_offset;
                let stone_threshold = sea_level + 12 + jagged_offset;
                let arctic_snow_threshold = sea_level + 3 + jagged_offset;
                let grass_patch_noise = transition_noise * 0.7 + transition_noise_2 * 0.3;
                let ice_noise = transition_noise * transition_noise_2;

                // === Fill terrain column with ORGANIC transitions ===
                for y in 0..=height {
                    let y_i = y as isize;
                    let depth_from_surface = height - y;

                    let block = if depth_from_surface == 0 && y > sea {
                        // === SURFACE BLOCK with organic vein transitions ===

                        // Check for transition zones between biomes
                        // Use vein_value to create connected patterns, not random dots

                        // Forest-Desert transition: dirt veins extending into desert
                        if biome_weights.desert > 0.15 && biome_weights.forest > 0.15 {
                            // In the transition zone
                            let desert_dominance = biome_weights.desert / (biome_weights.desert + biome_weights.forest);
                            // Vein threshold shifts based on dominance
                            // More desert = need higher vein value to get grass
                            let grass_threshold = 0.3 + desert_dominance * 0.5;
                            if vein_value > grass_threshold {
                                BlockType::Grass // Grass veins into desert
                            } else {
                                BlockType::Sand
                            }
                        }
                        // Forest-Mountain transition
                        else if biome_weights.mountains > 0.15 && biome_weights.forest > 0.15 {
                            let mountain_dominance = biome_weights.mountains / (biome_weights.mountains + biome_weights.forest);
                            if y_i > snow_threshold {
                                BlockType::Snow
                            } else if y_i > stone_threshold {
                                let stone_threshold_here = 0.3 + mountain_dominance * 0.4;
                                if vein_value > stone_threshold_here { BlockType::Grass } else { BlockType::Stone }
                            } else {
                                BlockType::Grass
                            }
                        }
                        // Forest-Arctic transition
                        else if biome_weights.arctic > 0.15 && biome_weights.forest > 0.15 {
                            let arctic_dominance = biome_weights.arctic / (biome_weights.arctic + biome_weights.forest);
                            let snow_threshold_here = 0.3 + arctic_dominance * 0.5;
                            if vein_value < snow_threshold_here { BlockType::Snow } else { BlockType::Grass }
                        }
                        // Pure dominant biome
                        else {
                            match dominant_biome {
                                BiomeType::Desert => BlockType::Sand,
                                BiomeType::Forest => BlockType::Grass,
                                BiomeType::Ocean => {
                                    if height > sea + 3 { BlockType::Grass } else { BlockType::Sand }
                                },
                                BiomeType::Mountains => {
                                    if y_i > snow_threshold {
                                        BlockType::Snow
                                    } else if y_i > stone_threshold {
                                        let blocks_above = (y_i - stone_threshold) as f64;
                                        if grass_patch_noise > 0.35 + (blocks_above * 0.06) && blocks_above < 12.0 {
                                            BlockType::Grass
                                        } else {
                                            BlockType::Stone
                                        }
                                    } else {
                                        BlockType::Grass
                                    }
                                },
                                BiomeType::Arctic => {
                                    if y_i > arctic_snow_threshold || ice_noise > 0.55 {
                                        BlockType::Snow
                                    } else if ice_noise > 0.25 {
                                        BlockType::Ice
                                    } else {
                                        BlockType::Snow
                                    }
                                },
                            }
                        }
                    } else if depth_from_surface == 0 && y <= sea {
                        // Underwater surface
                        BlockType::Sand
                    } else if depth_from_surface <= 3 {
                        // Near-surface with vein transitions
                        if biome_weights.desert > 0.15 && biome_weights.forest > 0.15 {
                            let desert_dominance = biome_weights.desert / (biome_weights.desert + biome_weights.forest);
                            let dirt_threshold = 0.3 + desert_dominance * 0.5;
                            if vein_value > dirt_threshold { BlockType::Dirt } else { BlockType::Sand }
                        } else {
                            match dominant_biome {
                                BiomeType::Desert => BlockType::Sand,
                                BiomeType::Forest => BlockType::Dirt,
                                BiomeType::Ocean => BlockType::Sand,
                                BiomeType::Mountains => {
                                    if y_i > stone_threshold - 4 { BlockType::Stone } else { BlockType::Dirt }
                                },
                                BiomeType::Arctic => {
                                    if y_i > arctic_snow_threshold { BlockType::Stone } else { BlockType::Dirt }
                                },
                            }
                        }
                    } else if depth_from_surface <= 6 {
                        match dominant_biome {
                            BiomeType::Desert => BlockType::Sand,
                            _ => BlockType::Stone,
                        }
                    } else {
                        BlockType::Stone
                    };

                    self.blocks[x][y][z] = block;
                }

                // === Fill water ===
                // Desert should NEVER have water (except oases)
                let is_desert_area = biome_weights.desert > 0.3;

                if !is_desert_area {
                    for y in height + 1..sea {
                        if self.blocks[x][y][z] == BlockType::Air {
                            if dominant_biome == BiomeType::Arctic && y == sea - 1 {
                                let ice_gap_noise = (glacier_perlin.get([world_x * 0.05, world_z * 0.05]) + 1.0) * 0.5;
                                if ice_gap_noise > 0.15 {
                                    self.blocks[x][y][z] = BlockType::Ice;
                                } else {
                                    self.blocks[x][y][z] = BlockType::Water;
                                }
                            } else {
                                self.blocks[x][y][z] = BlockType::Water;
                            }
                        }
                    }
                }

                // === Desert Oases ===
                if dominant_biome == BiomeType::Desert && biome_weights.desert > 0.7 {
                    let oasis_scale = 0.006;
                    let oasis_noise = (oasis_perlin.get([world_x * oasis_scale, world_z * oasis_scale]) + 1.0) * 0.5;

                    if oasis_noise > 0.90 {
                        let oasis_strength = smoothstep(0.90, 0.97, oasis_noise);
                        let depression_depth = (oasis_strength * 3.0) as usize + 1;

                        let water_y = height.saturating_sub(depression_depth);
                        if water_y > 10 && height > sea {
                            for dy in 0..=depression_depth {
                                let y = water_y + dy;
                                if y < CHUNK_HEIGHT {
                                    if dy < depression_depth / 2 + 1 {
                                        self.blocks[x][y][z] = BlockType::Water;
                                    } else {
                                        self.blocks[x][y][z] = BlockType::Dirt;
                                    }
                                }
                            }
                            if height < CHUNK_HEIGHT {
                                self.blocks[x][height][z] = BlockType::Grass;
                            }
                        }
                    }
                }

                // === Arctic Glaciers ===
                if dominant_biome == BiomeType::Arctic && height <= sea {
                    let glacier_scale = 0.015;
                    let glacier_noise = (glacier_perlin.get([world_x * glacier_scale, world_z * glacier_scale]) + 1.0) * 0.5;

                    if glacier_noise > 0.50 {
                        let glacier_height = ((glacier_noise - 0.50) * 16.0) as usize;
                        for gy in 0..glacier_height {
                            let y = sea + gy;
                            if y < CHUNK_HEIGHT {
                                let taper = 1.0 - (gy as f64 / glacier_height.max(1) as f64);
                                let taper_noise = (jagged_perlin.get([world_x * 0.08, y as f64 * 0.08, world_z * 0.08]) + 1.0) * 0.5;
                                if taper_noise < taper {
                                    self.blocks[x][y][z] = BlockType::Ice;
                                }
                            }
                        }
                    }
                }
            }
        }

        // NOTE: Cave generation removed - will be re-added later once biomes are stable

        // === Tree Generation with biome-aware density ===
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let world_x = (world_offset_x + x as i32) as f64;
                let world_z = (world_offset_z + z as i32) as f64;

                let biome_weights = column_biomes[x][z];
                let dominant_biome = biome_weights.dominant();

                // Skip arctic entirely
                if dominant_biome == BiomeType::Arctic {
                    continue;
                }

                // Find grass surface
                let mut height = 0;
                let mut found_grass = false;
                for y in (0..CHUNK_HEIGHT).rev() {
                    if self.blocks[x][y][z] == BlockType::Grass {
                        height = y;
                        found_grass = true;
                        break;
                    }
                }

                if !found_grass {
                    continue;
                }

                let max_tree_space = TREE_MAX_HEIGHT + 4;

                if height > sea && height < CHUNK_HEIGHT - max_tree_space && x > 3 && x < CHUNK_SIZE - 3 && z > 3 && z < CHUNK_SIZE - 3 {
                    // === Tree density based on BIOME WEIGHT, not just dominant ===
                    // Trees become sparser as forest weight decreases
                    let forest_weight = biome_weights.forest;

                    // Base threshold adjusted by forest weight
                    // Pure forest: threshold 0.3 (fairly dense but not overwhelming)
                    // Forest edge: threshold increases, fewer trees
                    let base_threshold = match dominant_biome {
                        BiomeType::Forest => 0.3 + (1.0 - forest_weight) * 0.5, // 0.3 to 0.8
                        BiomeType::Mountains => 0.6,
                        BiomeType::Desert => 0.95,
                        BiomeType::Ocean => 0.7,
                        BiomeType::Arctic => 0.95,
                    };

                    let height_variance = dominant_biome == BiomeType::Forest && forest_weight > 0.5;

                    let tree_noise = perlin.get([world_x * 0.02, world_z * 0.02]);
                    if tree_noise > base_threshold {
                        // Minimum spacing - forests get slightly closer trees but not packed
                        let min_spacing = if dominant_biome == BiomeType::Forest && forest_weight > 0.6 { 3 } else { 4 };
                        let mut too_close = false;
                        'check: for check_x in x.saturating_sub(min_spacing)..=(x + min_spacing).min(CHUNK_SIZE - 1) {
                            for check_z in z.saturating_sub(min_spacing)..=(z + min_spacing).min(CHUNK_SIZE - 1) {
                                if check_x == x && check_z == z {
                                    continue;
                                }
                                if height + 1 < CHUNK_HEIGHT && self.blocks[check_x][height + 1][check_z] == BlockType::Wood {
                                    too_close = true;
                                    break 'check;
                                }
                            }
                        }

                        if too_close {
                            continue;
                        }

                        let tree_seed = ((world_x as i64).wrapping_mul(73856093) ^ (world_z as i64).wrapping_mul(19349663)) as u64;
                        let mut rng = StdRng::seed_from_u64(tree_seed);

                        // Forest biome has much more height variation
                        let trunk_height = if height_variance {
                            // Forest: wide range from small bushes to tall trees
                            let height_roll = rng.gen::<f64>();
                            if height_roll < 0.2 {
                                rng.gen_range(3..=5)  // Short
                            } else if height_roll < 0.6 {
                                rng.gen_range(5..=8)  // Medium
                            } else if height_roll < 0.85 {
                                rng.gen_range(8..=11) // Tall
                            } else {
                                rng.gen_range(11..=14) // Very tall
                            }
                        } else {
                            let is_tall = rng.gen::<f64>() < TREE_TALL_CHANCE;
                            if is_tall {
                                rng.gen_range(TREE_MAX_HEIGHT - 2..=TREE_MAX_HEIGHT)
                            } else {
                                rng.gen_range(TREE_MIN_HEIGHT..=TREE_MIN_HEIGHT + 2)
                            }
                        };

                        // Generate trunk
                        for trunk_y in height + 1..=height + trunk_height {
                            if trunk_y < CHUNK_HEIGHT {
                                self.blocks[x][trunk_y][z] = BlockType::Wood;
                            }
                        }

                        // Branches for taller trees
                        let branch_start_y = height + trunk_height.saturating_sub(2);
                        let should_branch = rng.gen::<f64>() < TREE_BRANCH_CHANCE;
                        if should_branch && trunk_height >= 5 {
                            let num_branches = rng.gen_range(1..=3);
                            for _ in 0..num_branches {
                                let branch_y = rng.gen_range(branch_start_y..=height + trunk_height);
                                if branch_y < CHUNK_HEIGHT {
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

                        // Leaves - scale with trunk height
                        let leaf_radius = if trunk_height >= 10 { 4 } else if trunk_height >= 7 { 3 } else { 2 };
                        let leaf_start_y = height + trunk_height - 1;
                        let leaf_height = if trunk_height >= 10 { 6 } else if trunk_height >= 7 { 5 } else { 4 };

                        for lx in x.saturating_sub(leaf_radius)..=(x + leaf_radius).min(CHUNK_SIZE - 1) {
                            for lz in z.saturating_sub(leaf_radius)..=(z + leaf_radius).min(CHUNK_SIZE - 1) {
                                for ly in leaf_start_y..leaf_start_y + leaf_height {
                                    if ly < CHUNK_HEIGHT {
                                        let dx = (lx as i32 - x as i32).abs();
                                        let dz = (lz as i32 - z as i32).abs();
                                        let dy = ly - leaf_start_y;

                                        let is_corner = dx == leaf_radius as i32 && dz == leaf_radius as i32;
                                        let skip_corner = is_corner && (dy == 0 || dy >= leaf_height - 1);
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

        // === Oasis Trees (Desert) ===
        for x in 2..CHUNK_SIZE - 2 {
            for z in 2..CHUNK_SIZE - 2 {
                // Check for grass in desert (oasis indicator)
                for y in (sea_level as usize)..CHUNK_HEIGHT - 10 {
                    if self.blocks[x][y][z] != BlockType::Grass {
                        continue;
                    }

                    // Verify this is in a desert by checking for nearby sand
                    let mut near_sand = false;
                    for dx in -2i32..=2 {
                        for dz in -2i32..=2 {
                            let nx = (x as i32 + dx) as usize;
                            let nz = (z as i32 + dz) as usize;
                            if nx < CHUNK_SIZE && nz < CHUNK_SIZE {
                                if self.blocks[nx][y][nz] == BlockType::Sand {
                                    near_sand = true;
                                    break;
                                }
                            }
                        }
                        if near_sand { break; }
                    }

                    if !near_sand {
                        continue;
                    }

                    let world_x = (world_offset_x + x as i32) as f64;
                    let world_z = (world_offset_z + z as i32) as f64;

                    // Sparse trees at oases
                    let tree_noise = perlin.get([world_x * 0.1, world_z * 0.1]);
                    if tree_noise < 0.3 {
                        continue;
                    }

                    // Check for nearby trees
                    let mut too_close = false;
                    'check: for check_x in x.saturating_sub(2)..=(x + 2).min(CHUNK_SIZE - 1) {
                        for check_z in z.saturating_sub(2)..=(z + 2).min(CHUNK_SIZE - 1) {
                            if check_x == x && check_z == z { continue; }
                            if y + 1 < CHUNK_HEIGHT && self.blocks[check_x][y + 1][check_z] == BlockType::Wood {
                                too_close = true;
                                break 'check;
                            }
                        }
                    }
                    if too_close { continue; }

                    // Small palm-like trees
                    let tree_seed = ((world_x as i64).wrapping_mul(73856093) ^ (world_z as i64).wrapping_mul(19349663)) as u64;
                    let mut rng = StdRng::seed_from_u64(tree_seed);
                    let trunk_height = rng.gen_range(4..=6);

                    for trunk_y in y + 1..=(y + trunk_height).min(CHUNK_HEIGHT - 1) {
                        self.blocks[x][trunk_y][z] = BlockType::Wood;
                    }

                    // Small leaf canopy
                    let leaf_y = y + trunk_height;
                    if leaf_y < CHUNK_HEIGHT - 2 {
                        for lx in x.saturating_sub(2)..=(x + 2).min(CHUNK_SIZE - 1) {
                            for lz in z.saturating_sub(2)..=(z + 2).min(CHUNK_SIZE - 1) {
                                for ly in leaf_y..=(leaf_y + 2).min(CHUNK_HEIGHT - 1) {
                                    let dx = (lx as i32 - x as i32).abs();
                                    let dz = (lz as i32 - z as i32).abs();
                                    if dx + dz <= 3 && self.blocks[lx][ly][lz] == BlockType::Air {
                                        self.blocks[lx][ly][lz] = BlockType::Leaves;
                                    }
                                }
                            }
                        }
                    }
                    break; // One tree per column
                }
            }
        }

        // === GlowStone in caves ===
        let glow_perlin = Perlin::new(43);
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let world_x = (world_offset_x + x as i32) as f64;
                let world_z = (world_offset_z + z as i32) as f64;
                for y in 5..25 {
                    if self.blocks[x][y][z] == BlockType::Stone {
                        let noise_val = glow_perlin.get([world_x * 0.05, y as f64 * 0.05, world_z * 0.05]);
                        if noise_val > 0.85 {
                            self.blocks[x][y][z] = BlockType::GlowStone;
                        }
                    }
                }
            }
        }

        // === Floating Sky Islands (Desert Only) ===
        // Sky islands only appear over desert biomes and are smaller
        let sky_island_base_y = 65;           // Lower base
        let sky_island_height_range = 20;     // Less height variation
        let stalactite_perlin = Perlin::new(203);
        let hill_perlin = Perlin::new(204);

        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                // Only generate sky islands over desert biomes
                let biome_weights = column_biomes[x][z];
                if biome_weights.desert < 0.5 {
                    continue; // Skip non-desert areas
                }

                let world_x = (world_offset_x + x as i32) as f64;
                let world_z = (world_offset_z + z as i32) as f64;

                // Higher threshold = rarer islands, larger scale = smaller islands
                let mask_scale = 0.003; // Larger scale = smaller islands
                let island_mask = (sky_island_mask_perlin.get([world_x * mask_scale, world_z * mask_scale]) + 1.0) * 0.5;

                if island_mask < 0.85 { // Higher threshold = rarer
                    continue;
                }

                let height_noise = (sky_island_detail.get([world_x * 0.002, world_z * 0.002]) + 1.0) * 0.5;
                let island_center_y = sky_island_base_y + (height_noise * sky_island_height_range as f64) as usize;

                let hill_scale = 0.04; // Larger scale = smaller features
                let hill_noise = (hill_perlin.get([world_x * hill_scale, world_z * hill_scale]) + 1.0) * 0.5;
                let hill_height = (hill_noise * 2.0) as usize; // Smaller hills

                let island_scale = 0.015; // Larger scale = smaller islands
                let horizontal_island_noise = (sky_island_perlin.get([
                    world_x * island_scale,
                    island_center_y as f64 * island_scale * 0.3,
                    world_z * island_scale,
                ]) + 1.0) * 0.5;

                let centeredness = smoothstep(0.55, 0.85, horizontal_island_noise);

                let stalactite_scale = 0.08;
                let stalactite_noise = (stalactite_perlin.get([world_x * stalactite_scale, world_z * stalactite_scale]) + 1.0) * 0.5;

                // Smaller stalactites
                let base_stalactite = if stalactite_noise > 0.35 {
                    1 + ((stalactite_noise - 0.35) * 5.0) as usize
                } else {
                    0
                };

                let center_bonus = (centeredness * 4.0) as usize; // Smaller center bonus
                let stalactite_depth = base_stalactite + center_bonus;

                let base_thickness = 3; // Thinner islands
                let island_min_y = island_center_y.saturating_sub(base_thickness / 2 + stalactite_depth);
                let island_max_y = (island_center_y + base_thickness / 2 + hill_height + 1).min(CHUNK_HEIGHT);

                let island_strength = smoothstep(0.85, 0.95, island_mask);

                for y in island_min_y..island_max_y {
                    let world_y = y as f64;

                    let island_noise = sky_island_perlin.get([
                        world_x * island_scale,
                        world_y * island_scale * 0.3,
                        world_z * island_scale,
                    ]);

                    let effective_center_y = (island_center_y + hill_height / 2) as f64;
                    let effective_thickness = base_thickness as f64 + hill_height as f64 / 2.0 + stalactite_depth as f64 / 2.0;

                    let y_dist = (world_y - effective_center_y).abs() / (effective_thickness / 2.0 + 1.0);
                    let y_falloff = (1.0 - y_dist.powi(2)).max(0.0);

                    let is_below_center = world_y < effective_center_y;
                    let taper = if is_below_center {
                        let base_taper = 0.5;
                        let center_taper_bonus = centeredness * 0.3;
                        y_falloff * (base_taper + center_taper_bonus)
                    } else {
                        y_falloff
                    };

                    let threshold = 0.50 - (island_strength * 0.20) - (taper * 0.15); // Tighter threshold

                    if island_noise > threshold && self.blocks[x][y][z] == BlockType::Air {
                        let is_surface = y + 1 >= island_max_y || {
                            let above_y = (y + 1) as f64;
                            let above_noise = sky_island_perlin.get([
                                world_x * island_scale,
                                above_y * island_scale * 0.3,
                                world_z * island_scale,
                            ]);
                            let above_y_dist = (above_y - effective_center_y).abs() / (effective_thickness / 2.0 + 1.0);
                            let above_y_falloff = (1.0 - above_y_dist.powi(2)).max(0.0);
                            let above_taper = if above_y < effective_center_y {
                                let base_taper = 0.5;
                                let center_taper_bonus = centeredness * 0.3;
                                above_y_falloff * (base_taper + center_taper_bonus)
                            } else {
                                above_y_falloff
                            };
                            let above_threshold = 0.50 - (island_strength * 0.20) - (above_taper * 0.15);
                            above_noise <= above_threshold
                        };

                        let mut depth_from_surface = 0;
                        for check_y in (y + 1)..island_max_y.min(y + 4) {
                            let check_world_y = check_y as f64;
                            let check_noise = sky_island_perlin.get([
                                world_x * island_scale,
                                check_world_y * island_scale * 0.3,
                                world_z * island_scale,
                            ]);
                            let check_y_dist = (check_world_y - effective_center_y).abs() / (effective_thickness / 2.0 + 1.0);
                            let check_y_falloff = (1.0 - check_y_dist.powi(2)).max(0.0);
                            let check_taper = if check_world_y < effective_center_y {
                                let base_taper = 0.5;
                                let center_taper_bonus = centeredness * 0.3;
                                check_y_falloff * (base_taper + center_taper_bonus)
                            } else {
                                check_y_falloff
                            };
                            let check_threshold = 0.50 - (island_strength * 0.20) - (check_taper * 0.15);
                            if check_noise > check_threshold {
                                depth_from_surface += 1;
                            } else {
                                break;
                            }
                        }

                        // Desert sky islands are sandstone-like
                        self.blocks[x][y][z] = if is_below_center {
                            BlockType::Stone
                        } else if is_surface {
                            BlockType::Sand // Sandy top
                        } else if depth_from_surface < 2 {
                            BlockType::Sand
                        } else {
                            BlockType::Stone
                        };
                    }
                }
            }
        }

        // NOTE: Sky island trees removed - desert islands don't have trees
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