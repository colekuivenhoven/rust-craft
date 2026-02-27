use crate::block::{BlockType, Vertex, create_face_vertices, create_face_vertices_tinted, create_cross_model_vertices, create_water_face_vertices, AO_OFFSETS, calculate_ao};
use crate::lighting;
use crate::texture::{get_face_uvs, rotate_face_uvs, TEX_NONE, TEX_GRASS_TOP, TEX_GRASS_SIDE};
use cgmath::Vector3;
use noise::{NoiseFn, Perlin};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

pub const CHUNK_SIZE: usize = 16;
pub const CHUNK_HEIGHT: usize = 128;

// Biome types for terrain generation
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BiomeType {
    Desert,
    Forest,
    Mountains,
    Arctic,
    Ocean,
    Plains,
}

// Biome weights for smooth blending between biomes
#[derive(Clone, Copy, Debug)]
pub struct BiomeWeights {
    pub desert: f64,
    pub forest: f64,
    pub mountains: f64,
    pub arctic: f64,
    pub ocean: f64,
    pub plains: f64,
}

impl BiomeWeights {
    pub fn new() -> Self {
        Self {
            desert: 0.0,
            forest: 0.0,
            mountains: 0.0,
            arctic: 0.0,
            ocean: 0.0,
            plains: 0.0,
        }
    }

    pub fn normalize(&mut self) {
        let total = self.desert + self.forest + self.mountains + self.arctic + self.ocean + self.plains;
        if total > 0.0 {
            self.desert /= total;
            self.forest /= total;
            self.mountains /= total;
            self.arctic /= total;
            self.ocean /= total;
            self.plains /= total;
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
            max_weight = self.ocean;
            dominant = BiomeType::Ocean;
        }
        if self.plains > max_weight {
            dominant = BiomeType::Plains;
        }
        dominant
    }
}

pub struct Chunk {
    pub blocks: Box<[[[BlockType; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]>,
    pub light_levels: Box<[[[u8; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]>,
    pub biome_map: Vec<Vec<BiomeType>>,         // [x][z] dominant biome per column, for mesh coloring
    pub plains_weight_map: Vec<Vec<f32>>,       // [x][z] continuous plains weight for smooth color blending
    pub position: (i32, i32),
    pub master_seed: u32,
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

/// Helper struct to provide safe access to neighboring chunks during mesh generation.
/// Includes the 4 cardinal and 4 diagonal neighbors so that smooth-lighting and AO
/// samples near chunk corners can cross into diagonal chunks correctly.
pub struct ChunkNeighbors<'a> {
    pub center: &'a Chunk,
    pub left:         Option<&'a Chunk>,  // (x-1, z  )
    pub right:        Option<&'a Chunk>,  // (x+1, z  )
    pub front:        Option<&'a Chunk>,  // (x,   z+1)
    pub back:         Option<&'a Chunk>,  // (x,   z-1)
    pub front_left:   Option<&'a Chunk>,  // (x-1, z+1)
    pub front_right:  Option<&'a Chunk>,  // (x+1, z+1)
    pub back_left:    Option<&'a Chunk>,  // (x-1, z-1)
    pub back_right:   Option<&'a Chunk>,  // (x+1, z-1)
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
        let cs = CHUNK_SIZE as i32;
        match (x < 0, x >= cs, z < 0, z >= cs) {
            // Diagonals
            (true,  false, true,  false) => (self.back_left,   (x + cs) as usize, (z + cs) as usize),
            (true,  false, false, true)  => (self.front_left,  (x + cs) as usize, (z - cs) as usize),
            (false, true,  true,  false) => (self.back_right,  (x - cs) as usize, (z + cs) as usize),
            (false, true,  false, true)  => (self.front_right, (x - cs) as usize, (z - cs) as usize),
            // Cardinals
            (true,  false, false, false) => (self.left,  (x + cs) as usize, z as usize),
            (false, true,  false, false) => (self.right, (x - cs) as usize, z as usize),
            (false, false, true,  false) => (self.back,  x as usize, (z + cs) as usize),
            (false, false, false, true)  => (self.front, x as usize, (z - cs) as usize),
            // Center
            _ => (Some(self.center), x as usize, z as usize),
        }
    }
}

impl Chunk {
    pub fn new(chunk_x: i32, chunk_z: i32, master_seed: u32, cfg: &crate::config::TerrainConfig) -> Self {
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
            biome_map: vec![vec![BiomeType::Forest; CHUNK_SIZE]; CHUNK_SIZE],
            plains_weight_map: vec![vec![0.0f32; CHUNK_SIZE]; CHUNK_SIZE],
            position: (chunk_x, chunk_z),
            master_seed,
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
        chunk.generate_terrain(master_seed, cfg);
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
            biome_map: vec![vec![BiomeType::Forest; CHUNK_SIZE]; CHUNK_SIZE],
            plains_weight_map: vec![vec![0.0f32; CHUNK_SIZE]; CHUNK_SIZE],
            position: (chunk_x, chunk_z),
            master_seed: 0, // Not used for saved chunks (blocks already generated)
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

    fn generate_terrain(&mut self, master_seed: u32, cfg: &crate::config::TerrainConfig) {
        // Derive per-feature seeds from master seed
        let seed_base_terrain      = master_seed;
        let seed_glowstone         = master_seed.wrapping_add(1);
        let seed_detail            = master_seed.wrapping_add(4);
        let seed_jagged            = master_seed.wrapping_add(5);
        let seed_sky_island        = master_seed.wrapping_add(158);
        let seed_sky_island_mask   = master_seed.wrapping_add(159);
        let seed_sky_island_detail = master_seed.wrapping_add(160);
        let seed_stalactite        = master_seed.wrapping_add(161);
        let seed_hill              = master_seed.wrapping_add(162);
        let seed_temperature       = master_seed.wrapping_add(258);
        let seed_humidity          = master_seed.wrapping_add(259);
        let seed_continentalness   = master_seed.wrapping_add(260);
        let seed_mountain          = master_seed.wrapping_add(261);
        let seed_oasis             = master_seed.wrapping_add(358);
        let seed_glacier           = master_seed.wrapping_add(359);
        let seed_ocean_island      = master_seed.wrapping_add(458);
        let seed_vein              = master_seed.wrapping_add(558);
        let seed_vein_detail       = master_seed.wrapping_add(559);
        let seed_grass_tuft        = master_seed.wrapping_add(800);
        let seed_river             = master_seed.wrapping_add(900);
        let seed_river_warp1       = master_seed.wrapping_add(901);
        let seed_river_warp2       = master_seed.wrapping_add(902);
        let seed_river_depth       = master_seed.wrapping_add(903);

        // === Noise generators ===
        let perlin = Perlin::new(seed_base_terrain);
        let detail_perlin = Perlin::new(seed_detail);
        let jagged_perlin = Perlin::new(seed_jagged);

        // Biome noise generators - using large scale for smooth regions
        let temperature_perlin = Perlin::new(seed_temperature);
        let humidity_perlin = Perlin::new(seed_humidity);
        let continentalness_perlin = Perlin::new(seed_continentalness);
        let mountain_perlin = Perlin::new(seed_mountain);

        // Transition/vein noise - creates organic, connected patterns at biome boundaries
        let vein_perlin = Perlin::new(seed_vein);
        let vein_detail = Perlin::new(seed_vein_detail);

        // Sky island noise generators
        let sky_island_perlin = Perlin::new(seed_sky_island);
        let sky_island_mask_perlin = Perlin::new(seed_sky_island_mask);
        let sky_island_detail = Perlin::new(seed_sky_island_detail);

        // Oasis and special feature noise
        let oasis_perlin = Perlin::new(seed_oasis);
        let glacier_perlin = Perlin::new(seed_glacier);

        // Ocean island noise
        let island_perlin = Perlin::new(seed_ocean_island);

        // River domain warp noise (makes rivers wind naturally)
        let river_warp1 = Perlin::new(seed_river_warp1);
        let river_warp2 = Perlin::new(seed_river_warp2);
        // River depth variation noise (gives rivers varying water depth along their length)
        let river_depth_perlin = Perlin::new(seed_river_depth);

        let world_offset_x = self.position.0 * CHUNK_SIZE as i32;
        let world_offset_z = self.position.1 * CHUNK_SIZE as i32;
        let sea_level: isize = cfg.sea_level as isize;
        let sea = cfg.sea_level;

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

                // === River: Voronoi F2-F1 edge distance with domain warping ===
                // Domain warp offsets make rivers wind organically instead of following straight edges
                let river_edge_dist = {
                    let wx = world_x + river_warp1.get([world_x * cfg.river_winding_scale, world_z * cfg.river_winding_scale]) * cfg.river_winding_amplitude;
                    let wz = world_z + river_warp2.get([world_x * cfg.river_winding_scale + 100.0, world_z * cfg.river_winding_scale + 100.0]) * cfg.river_winding_amplitude;
                    let cell_size = cfg.river_cell_size;
                    let cell_x = (wx / cell_size).floor() as i32;
                    let cell_z = (wz / cell_size).floor() as i32;
                    let mut f1 = f64::MAX;
                    let mut f2 = f64::MAX;
                    for dx in -2i32..=2 {
                        for dz in -2i32..=2 {
                            let nx = cell_x + dx;
                            let nz = cell_z + dz;
                            // Hash two integers + seed into a stable pseudo-random value
                            let h: u64 = {
                                let mut v = (seed_river as u64)
                                    .wrapping_add((nx as i64 as u64).wrapping_mul(73856093))
                                    .wrapping_add((nz as i64 as u64).wrapping_mul(19349663));
                                v ^= v >> 33;
                                v = v.wrapping_mul(0xff51afd7ed558ccd);
                                v ^= v >> 33;
                                v
                            };
                            // Random offset within the Voronoi cell
                            let ox = (h & 0xFFFF) as f64 / 65536.0 * cell_size;
                            let oz = ((h >> 16) & 0xFFFF) as f64 / 65536.0 * cell_size;
                            let px = nx as f64 * cell_size + ox;
                            let pz = nz as f64 * cell_size + oz;
                            let d = ((wx - px).powi(2) + (wz - pz).powi(2)).sqrt();
                            if d < f1 { f2 = f1; f1 = d; } else if d < f2 { f2 = d; }
                        }
                    }
                    // F2-F1: 0 at Voronoi cell edges (river center), grows with distance from edge
                    (f2 - f1).max(0.0)
                };

                // === Sample biome parameters ===
                let raw_temperature = (temperature_perlin.get([world_x * cfg.biome_scale, world_z * cfg.biome_scale]) + 1.0) * 0.5;
                let raw_humidity = (humidity_perlin.get([world_x * cfg.biome_scale * 1.1, world_z * cfg.biome_scale * 1.1 + 1000.0]) + 1.0) * 0.5;
                let raw_mountainess = (mountain_perlin.get([world_x * cfg.biome_scale * 1.5, world_z * cfg.biome_scale * 1.5 + 3000.0]) + 1.0) * 0.5;

                // Continentalness with wider coastal zone
                let continental_scale = cfg.biome_scale * cfg.continental_scale_factor;
                let raw_continentalness = (continentalness_perlin.get([world_x * continental_scale, world_z * continental_scale + 2000.0]) + 1.0) * 0.5;

                // === Calculate biome weights ===
                let mut biome_weights = BiomeWeights::new();

                // Ocean: wider transition zone
                if raw_continentalness < cfg.ocean_threshold_deep {
                    biome_weights.ocean = 1.0;
                } else if raw_continentalness < cfg.ocean_threshold_shallow {
                    biome_weights.ocean = smootherstep(cfg.ocean_threshold_shallow, cfg.ocean_threshold_deep, raw_continentalness);
                }

                let land_factor = 1.0 - biome_weights.ocean;

                if land_factor > 0.0 {
                    // Arctic: cold temperatures
                    let arctic_factor = smootherstep(cfg.arctic_temp_threshold_high, cfg.arctic_temp_threshold_low, raw_temperature);
                    biome_weights.arctic = arctic_factor * land_factor;

                    // Desert: hot, dry, and inland (but NOT near ocean at all)
                    let desert_temp = smootherstep(cfg.desert_temp_threshold_low, cfg.desert_temp_threshold_high, raw_temperature);
                    let desert_dry = smootherstep(cfg.desert_humidity_threshold_high, cfg.desert_humidity_threshold_low, raw_humidity);
                    // Deserts require high continentalness to stay away from water
                    let desert_inland = smootherstep(cfg.desert_inland_threshold_low, cfg.desert_inland_threshold_high, raw_continentalness);
                    biome_weights.desert = desert_temp * desert_dry * desert_inland * land_factor * (1.0 - biome_weights.arctic);

                    // Mountains
                    let mountain_factor = smootherstep(cfg.mountain_threshold_low, cfg.mountain_threshold_high, raw_mountainess);
                    let remaining = land_factor * (1.0 - biome_weights.arctic) * (1.0 - biome_weights.desert * 0.8);
                    biome_weights.mountains = mountain_factor * remaining;

                    // Split remaining land between Forest and Plains based on humidity.
                    // Plains occupy lower-humidity land that isn't arctic, desert, or mountainous.
                    let used = biome_weights.arctic + biome_weights.desert + biome_weights.mountains;
                    let forest_plains_total = (land_factor - used).max(0.0);
                    let plains_humidity_factor = smootherstep(cfg.plains_humidity_threshold_high, cfg.plains_humidity_threshold_low, raw_humidity);
                    biome_weights.plains = plains_humidity_factor * forest_plains_total;
                    biome_weights.forest = forest_plains_total - biome_weights.plains;
                }

                biome_weights.normalize();
                column_biomes[x][z] = biome_weights;
                self.biome_map[x][z] = biome_weights.dominant();
                self.plains_weight_map[x][z] = biome_weights.plains as f32;

                let dominant_biome = biome_weights.dominant();

                // === River factors ===
                // Suppress rivers in ocean and desert — rivers are land features
                let no_river_zone = (biome_weights.ocean + biome_weights.desert).min(1.0);
                // river_factor: 1 at Voronoi cell edge (river center), 0 beyond river_width
                let river_factor = (1.0 - smootherstep(0.0, cfg.river_width, river_edge_dist)) * (1.0 - no_river_zone);
                // bank_factor: extends river_bank_width beyond the channel, for sand banks
                let bank_factor = (1.0 - smootherstep(cfg.river_width, cfg.river_width + cfg.river_bank_width, river_edge_dist)) * (1.0 - no_river_zone);

                // === Vein noise for organic biome transitions ===
                // This creates connected, branching patterns instead of polka dots
                let vein_n1 = vein_perlin.get([world_x * cfg.vein_scale, world_z * cfg.vein_scale]);
                let vein_n2 = vein_detail.get([world_x * cfg.vein_scale * 2.3, world_z * cfg.vein_scale * 2.3]);
                // Combine to create fractal vein pattern
                let vein_value = (vein_n1 * 0.7 + vein_n2 * 0.3 + 1.0) * 0.5; // 0-1

                // === Calculate terrain height ===
                let base_n = (perlin.get([world_x * cfg.terrain_base_scale, world_z * cfg.terrain_base_scale]) + 1.0) * 0.5;
                let detail = detail_perlin.get([world_x * cfg.terrain_detail_scale, world_z * cfg.terrain_detail_scale]) * cfg.terrain_detail_amplitude;

                // === Ocean islands with GRADUAL slopes ===
                let island_noise = (island_perlin.get([world_x * cfg.ocean_island_scale + 5000.0, world_z * cfg.ocean_island_scale + 5000.0]) + 1.0) * 0.5;

                // Compute raw island bump from noise alone (no ocean weight gate)
                let island_bump_raw = if island_noise > cfg.ocean_island_threshold {
                    let island_strength = smootherstep(cfg.ocean_island_threshold, cfg.ocean_island_strength_max, island_noise);
                    // Squared for gradual slope, not linear
                    island_strength * island_strength * cfg.ocean_island_max_bump
                } else {
                    0.0
                };

                // Gradual island bump - only applied in ocean-dominant areas
                let island_bump = if biome_weights.ocean > 0.5 {
                    island_bump_raw
                } else {
                    0.0
                };

                // === Height per biome ===
                // Ocean floor with gradual island rise
                let ocean_base = cfg.ocean_height_base + base_n * cfg.ocean_height_variation;
                let ocean_height = ocean_base + island_bump;

                // Desert: ALWAYS well above sea level
                let desert_height = (cfg.desert_height_base + base_n * cfg.desert_height_variation + detail * 0.2).max(sea_level as f64 + cfg.desert_min_above_sea);

                // Forest: gentle rolling hills
                let forest_height = cfg.forest_height_base + base_n * cfg.forest_height_variation + detail;

                // Plains: slightly flatter than forest, similar elevation
                let plains_height = cfg.plains_height_base + base_n * cfg.plains_height_variation + detail * 0.2;

                // Arctic: varied terrain
                let arctic_height = cfg.arctic_height_base + base_n * cfg.arctic_height_variation + detail;

                // Mountains: dramatic peaks
                let ridge = perlin.get([world_x * cfg.mountain_ridge_scale, world_z * cfg.mountain_ridge_scale]).abs() * cfg.mountain_ridge_amplitude;
                let jagged = jagged_perlin.get([world_x * cfg.mountain_jagged_scale, world_z * cfg.mountain_jagged_scale]).abs() * cfg.mountain_jagged_amplitude;
                let mountain_height = cfg.mountain_height_base + base_n * cfg.mountain_height_variation + ridge + jagged + detail;

                // === Blend heights using biome weights ===
                let mut blended_height =
                    biome_weights.ocean * ocean_height +
                    biome_weights.desert * desert_height +
                    biome_weights.forest * forest_height +
                    biome_weights.plains * plains_height +
                    biome_weights.mountains * mountain_height +
                    biome_weights.arctic * arctic_height;

                // === Ensure desert areas stay above sea level ===
                // If desert has significant weight, pull height UP toward desert height
                if biome_weights.desert > cfg.desert_pull_threshold {
                    let desert_pull = smootherstep(cfg.desert_pull_threshold, cfg.desert_pull_strength, biome_weights.desert);
                    let min_desert_height = sea_level as f64 + cfg.desert_min_above_sea - 2.0;
                    blended_height = blended_height.max(min_desert_height * desert_pull + blended_height * (1.0 - desert_pull));
                }

                // === Coastal smoothing - WIDER zone with gentler slopes ===
                let island_blend_start = cfg.coastal_start - cfg.island_coastal_blend_width;
                let base_shore = cfg.coastal_min_height + base_n * cfg.forest_height_variation;
                let height_f = if raw_continentalness > cfg.coastal_start && raw_continentalness < cfg.coastal_end {
                    // Coastal transition zone
                    let shore_t = smootherstep(cfg.coastal_start, cfg.coastal_end, raw_continentalness);
                    // Blend island height into shore start, fading out as we move inland
                    let min_shore = base_shore + island_bump_raw * (1.0 - shore_t);
                    let max_shore = blended_height;
                    min_shore + (max_shore - min_shore) * shore_t
                } else if raw_continentalness >= island_blend_start && raw_continentalness <= cfg.coastal_start {
                    // Island-to-coast blend zone: smooth ocean/island heights into coastal zone
                    let blend_t = smootherstep(island_blend_start, cfg.coastal_start, raw_continentalness);
                    // Target: what the coastal zone would give at cfg.coastal_start
                    let coastal_start_height = base_shore + island_bump_raw;
                    // At island_blend_start (blend_t=0): pure blended_height (ocean + islands)
                    // At cfg.coastal_start (blend_t=1): matches coastal zone's starting height
                    blended_height + (coastal_start_height - blended_height) * blend_t
                } else {
                    blended_height
                };

                // === River carving: smoothly lower terrain to sea level at river channels ===
                // Depth varies along the river for a natural feel; deeper = lower riverbed
                let depth_noise_val = river_depth_perlin.get([world_x * 0.02, world_z * 0.02]);
                let river_depth = (cfg.river_depth_avg as f64 + depth_noise_val * cfg.river_depth_variation)
                    .max(1.0)
                    .round() as usize;
                // Carve target: surface block sits here; water fills from (target+1) up to sea_level-1
                let river_carve_target = cfg.sea_level as f64 - 1.0 - river_depth as f64;
                // Sand banks limited to within river_bank_y_range blocks above the riverbed
                let river_bank_max_height = (river_carve_target.round() as usize).saturating_add(cfg.river_bank_y_range);
                let height_f = if river_factor > 0.0 && height_f > river_carve_target {
                    height_f * (1.0 - river_factor) + river_carve_target * river_factor
                } else {
                    height_f
                };

                let height = (height_f.round() as isize).clamp(1, CHUNK_HEIGHT as isize - 1) as usize;

                // === Surface block noise for variations ===
                let transition_noise = (detail_perlin.get([world_x * cfg.surface_transition_scale, world_z * cfg.surface_transition_scale]) + 1.0) * 0.5;
                let transition_noise_2 = (jagged_perlin.get([world_x * cfg.surface_transition_scale * 1.3, world_z * cfg.surface_transition_scale * 1.3]) + 1.0) * 0.5;
                let jagged_offset = ((transition_noise * 5.0) + (transition_noise_2 * 3.0)) as isize - 4;

                let snow_threshold = sea_level + cfg.snow_threshold_offset + jagged_offset;
                let stone_threshold = sea_level + cfg.stone_threshold_offset + jagged_offset;
                let arctic_snow_threshold = sea_level + cfg.arctic_snow_threshold_offset + jagged_offset;
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

                        // Forest/Plains combined weight for vein transitions
                        let forest_or_plains = biome_weights.forest + biome_weights.plains;

                        // (Forest|Plains)-Desert transition: dirt veins extending into desert
                        if biome_weights.desert > 0.15 && forest_or_plains > 0.15 {
                            let desert_dominance = biome_weights.desert / (biome_weights.desert + forest_or_plains);
                            let grass_threshold = 0.3 + desert_dominance * 0.5;
                            if vein_value > grass_threshold {
                                BlockType::Dirt
                            } else {
                                BlockType::Sand
                            }
                        }
                        // (Forest|Plains)-Mountain transition
                        else if biome_weights.mountains > 0.15 && forest_or_plains > 0.15 {
                            let mountain_dominance = biome_weights.mountains / (biome_weights.mountains + forest_or_plains);
                            if y_i > snow_threshold {
                                BlockType::Snow
                            } else if y_i > stone_threshold {
                                let stone_threshold_here = 0.3 + mountain_dominance * 0.4;
                                if vein_value > stone_threshold_here { BlockType::Dirt } else { BlockType::Stone }
                            } else {
                                BlockType::Dirt
                            }
                        }
                        // (Forest|Plains)-Arctic transition
                        else if biome_weights.arctic > 0.15 && forest_or_plains > 0.15 {
                            let arctic_dominance = biome_weights.arctic / (biome_weights.arctic + forest_or_plains);
                            let snow_threshold_here = 0.3 + arctic_dominance * 0.5;
                            if vein_value < snow_threshold_here { BlockType::Snow } else { BlockType::Dirt }
                        }
                        // Pure dominant biome
                        else {
                            match dominant_biome {
                                BiomeType::Desert => BlockType::Sand,
                                BiomeType::Forest => BlockType::Dirt,
                                BiomeType::Plains => BlockType::Dirt,
                                BiomeType::Ocean => {
                                    if height > sea + cfg.ocean_island_grass_start { BlockType::Dirt } else { BlockType::Sand }
                                },
                                BiomeType::Mountains => {
                                    if y_i > snow_threshold {
                                        BlockType::Snow
                                    } else if y_i > stone_threshold {
                                        let blocks_above = (y_i - stone_threshold) as f64;
                                        if grass_patch_noise > cfg.grass_patch_base_threshold + (blocks_above * cfg.grass_patch_height_factor) && blocks_above < cfg.grass_patch_max_height {
                                            BlockType::Dirt
                                        } else {
                                            BlockType::Stone
                                        }
                                    } else {
                                        BlockType::Dirt
                                    }
                                },
                                BiomeType::Arctic => {
                                    if y_i > arctic_snow_threshold || ice_noise > cfg.arctic_full_ice_threshold {
                                        BlockType::Snow
                                    } else if ice_noise > cfg.arctic_ice_threshold {
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
                    } else if depth_from_surface <= cfg.depth_near_surface {
                        // Near-surface with vein transitions
                        let forest_or_plains = biome_weights.forest + biome_weights.plains;
                        if biome_weights.desert > 0.15 && forest_or_plains > 0.15 {
                            let desert_dominance = biome_weights.desert / (biome_weights.desert + forest_or_plains);
                            let dirt_threshold = 0.3 + desert_dominance * 0.5;
                            if vein_value > dirt_threshold { BlockType::Dirt } else { BlockType::Sand }
                        } else {
                            match dominant_biome {
                                BiomeType::Desert => BlockType::Sand,
                                BiomeType::Forest => BlockType::Dirt,
                                BiomeType::Plains => BlockType::Dirt,
                                BiomeType::Ocean => BlockType::Sand,
                                BiomeType::Mountains => {
                                    if y_i > stone_threshold - 4 { BlockType::Stone } else { BlockType::Dirt }
                                },
                                BiomeType::Arctic => {
                                    if y_i > arctic_snow_threshold { BlockType::Stone } else { BlockType::Dirt }
                                },
                            }
                        }
                    } else if depth_from_surface <= cfg.depth_transition {
                        match dominant_biome {
                            BiomeType::Desert => BlockType::Sand,
                            _ => BlockType::Stone,
                        }
                    } else {
                        BlockType::Stone
                    };

                    // === River sand overrides ===
                    // Both surface banks and subsurface layers are capped by river_bank_max_height
                    // so sand cannot appear high up on mountain slopes above the river channel.
                    let block = if height <= river_bank_max_height {
                        if depth_from_surface == 0 && bank_factor > 0.0 {
                            BlockType::Sand
                        } else if depth_from_surface >= 1 && depth_from_surface <= cfg.river_sand_depth && river_factor > 0.0 {
                            BlockType::Sand
                        } else {
                            block
                        }
                    } else {
                        block
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
                                let ice_gap_noise = (glacier_perlin.get([world_x * cfg.glacier_ice_gap_scale, world_z * cfg.glacier_ice_gap_scale]) + 1.0) * 0.5;
                                if ice_gap_noise > cfg.glacier_ice_gap_threshold {
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
                if dominant_biome == BiomeType::Desert && biome_weights.desert > cfg.oasis_min_desert_weight {
                    let oasis_noise = (oasis_perlin.get([world_x * cfg.oasis_scale, world_z * cfg.oasis_scale]) + 1.0) * 0.5;

                    if oasis_noise > cfg.oasis_threshold {
                        let oasis_strength = smoothstep(cfg.oasis_threshold, cfg.oasis_strength_max, oasis_noise);
                        let depression_depth = (oasis_strength * cfg.oasis_max_depression as f64) as usize + 1;

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
                                self.blocks[x][height][z] = BlockType::Dirt;
                            }
                        }
                    }
                }

                // === Arctic Glaciers ===
                if dominant_biome == BiomeType::Arctic && height <= sea {
                    let glacier_noise = (glacier_perlin.get([world_x * cfg.glacier_scale, world_z * cfg.glacier_scale]) + 1.0) * 0.5;

                    if glacier_noise > cfg.glacier_threshold {
                        let glacier_height = ((glacier_noise - cfg.glacier_threshold) * cfg.glacier_max_height) as usize;
                        for gy in 0..glacier_height {
                            let y = sea + gy;
                            if y < CHUNK_HEIGHT {
                                let taper = 1.0 - (gy as f64 / glacier_height.max(1) as f64);
                                let taper_noise = (jagged_perlin.get([world_x * cfg.glacier_taper_scale, y as f64 * cfg.glacier_taper_scale, world_z * cfg.glacier_taper_scale]) + 1.0) * 0.5;
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
                    if self.blocks[x][y][z] == BlockType::Dirt {
                        height = y;
                        found_grass = true;
                        break;
                    }
                }

                if !found_grass {
                    continue;
                }

                let max_tree_space = cfg.tree_max_height + cfg.leaf_height_large;

                if height > sea && height < CHUNK_HEIGHT - max_tree_space && x > cfg.tree_border_buffer && x < CHUNK_SIZE - cfg.tree_border_buffer && z > cfg.tree_border_buffer && z < CHUNK_SIZE - cfg.tree_border_buffer {
                    // === Tree density based on BIOME WEIGHT, not just dominant ===
                    // Trees become sparser as forest weight decreases
                    let forest_weight = biome_weights.forest;

                    // Base threshold adjusted by forest weight
                    // Pure forest: threshold 0.3 (fairly dense but not overwhelming)
                    // Forest edge: threshold increases, fewer trees
                    let base_threshold = match dominant_biome {
                        BiomeType::Forest => cfg.tree_threshold_forest_base + (1.0 - forest_weight) * cfg.tree_threshold_forest_edge_add,
                        BiomeType::Mountains => cfg.tree_threshold_mountain,
                        BiomeType::Desert => cfg.tree_threshold_desert,
                        BiomeType::Ocean => cfg.tree_threshold_ocean,
                        BiomeType::Arctic => cfg.tree_threshold_arctic,
                        BiomeType::Plains => cfg.tree_threshold_plains,
                    };

                    let height_variance = dominant_biome == BiomeType::Forest && forest_weight > 0.5;

                    let tree_noise = perlin.get([world_x * cfg.tree_noise_scale, world_z * cfg.tree_noise_scale]);
                    if tree_noise > base_threshold {
                        // Minimum spacing - forests get slightly closer trees but not packed
                        let min_spacing = if dominant_biome == BiomeType::Forest && forest_weight > cfg.tree_spacing_forest_weight_threshold {
                        cfg.tree_spacing_forest_dense
                    } else if dominant_biome == BiomeType::Plains {
                        cfg.tree_spacing_plains
                    } else {
                        cfg.tree_spacing_default
                    };
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

                        let tree_seed = ((world_x as i64).wrapping_mul(cfg.tree_seed_hash_1) ^ (world_z as i64).wrapping_mul(cfg.tree_seed_hash_2)) as u64;
                        let mut rng = StdRng::seed_from_u64(tree_seed);

                        // Forest biome has much more height variation
                        let trunk_height = if height_variance {
                            // Forest: wide range from small bushes to tall trees
                            let height_roll = rng.gen::<f64>();
                            if height_roll < cfg.forest_tree_short_chance {
                                rng.gen_range(cfg.forest_tree_short_min..=cfg.forest_tree_short_max)
                            } else if height_roll < cfg.forest_tree_medium_chance {
                                rng.gen_range(cfg.forest_tree_medium_min..=cfg.forest_tree_medium_max)
                            } else if height_roll < cfg.forest_tree_tall_chance {
                                rng.gen_range(cfg.forest_tree_tall_min..=cfg.forest_tree_tall_max)
                            } else {
                                rng.gen_range(cfg.forest_tree_very_tall_min..=cfg.forest_tree_very_tall_max)
                            }
                        } else {
                            let is_tall = rng.gen::<f64>() < cfg.tree_tall_chance;
                            if is_tall {
                                rng.gen_range(cfg.tree_max_height - 2..=cfg.tree_max_height)
                            } else {
                                rng.gen_range(cfg.tree_min_height..=cfg.tree_min_height + 2)
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
                        let branch_chance = if dominant_biome == BiomeType::Plains { cfg.plains_tree_branch_chance } else { cfg.tree_branch_chance };
                        let should_branch = rng.gen::<f64>() < branch_chance;
                        if should_branch && trunk_height >= cfg.branch_min_trunk_height {
                            let num_branches = rng.gen_range(cfg.branch_count_min..=cfg.branch_count_max);
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

                        // Leaves - biome-dependent shape
                        let leaf_start_y = height + trunk_height - 1;
                        if dominant_biome == BiomeType::Plains {
                            // Flat, spreading canopy: each layer up shrinks radius by 1
                            let p_radius = cfg.plains_leaf_radius;
                            let p_height = cfg.plains_leaf_height;
                            for lx in x.saturating_sub(p_radius)..=(x + p_radius).min(CHUNK_SIZE - 1) {
                                for lz in z.saturating_sub(p_radius)..=(z + p_radius).min(CHUNK_SIZE - 1) {
                                    let dx = (lx as i32 - x as i32).abs();
                                    let dz = (lz as i32 - z as i32).abs();
                                    for dy in 0..p_height {
                                        let ly = leaf_start_y + dy;
                                        if ly < CHUNK_HEIGHT {
                                            let effective_r = p_radius.saturating_sub(dy) as i32;
                                            let is_corner = dx == effective_r && dz == effective_r;
                                            if dx <= effective_r && dz <= effective_r && !is_corner && self.blocks[lx][ly][lz] == BlockType::Air {
                                                self.blocks[lx][ly][lz] = BlockType::Leaves;
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            // Standard canopy - scale with trunk height
                            let leaf_radius = if trunk_height >= cfg.tree_height_large_threshold { cfg.leaf_radius_large } else if trunk_height >= cfg.tree_height_medium_threshold { cfg.leaf_radius_medium } else { cfg.leaf_radius_small };
                            let leaf_height = if trunk_height >= cfg.tree_height_large_threshold { cfg.leaf_height_large } else if trunk_height >= cfg.tree_height_medium_threshold { cfg.leaf_height_medium } else { cfg.leaf_height_small };

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
        }

        // === Oasis Trees (Desert) ===
        for x in cfg.oasis_tree_spacing..CHUNK_SIZE - cfg.oasis_tree_spacing {
            for z in cfg.oasis_tree_spacing..CHUNK_SIZE - cfg.oasis_tree_spacing {
                // Check for grass in desert (oasis indicator)
                for y in cfg.sea_level..CHUNK_HEIGHT - 10 {
                    if self.blocks[x][y][z] != BlockType::Dirt {
                        continue;
                    }

                    // Verify this is in a desert by checking for nearby sand
                    let mut near_sand = false;
                    for dx in -(cfg.oasis_tree_spacing as i32)..=(cfg.oasis_tree_spacing as i32) {
                        for dz in -(cfg.oasis_tree_spacing as i32)..=(cfg.oasis_tree_spacing as i32) {
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
                    let tree_noise = perlin.get([world_x * cfg.oasis_tree_noise_scale, world_z * cfg.oasis_tree_noise_scale]);
                    if tree_noise < cfg.oasis_tree_threshold {
                        continue;
                    }

                    // Check for nearby trees
                    let mut too_close = false;
                    'check: for check_x in x.saturating_sub(cfg.oasis_tree_spacing)..=(x + cfg.oasis_tree_spacing).min(CHUNK_SIZE - 1) {
                        for check_z in z.saturating_sub(cfg.oasis_tree_spacing)..=(z + cfg.oasis_tree_spacing).min(CHUNK_SIZE - 1) {
                            if check_x == x && check_z == z { continue; }
                            if y + 1 < CHUNK_HEIGHT && self.blocks[check_x][y + 1][check_z] == BlockType::Wood {
                                too_close = true;
                                break 'check;
                            }
                        }
                    }
                    if too_close { continue; }

                    // Small palm-like trees
                    let tree_seed = ((world_x as i64).wrapping_mul(cfg.tree_seed_hash_1) ^ (world_z as i64).wrapping_mul(cfg.tree_seed_hash_2)) as u64;
                    let mut rng = StdRng::seed_from_u64(tree_seed);
                    let trunk_height = rng.gen_range(cfg.oasis_tree_min_height..=cfg.oasis_tree_max_height);

                    for trunk_y in y + 1..=(y + trunk_height).min(CHUNK_HEIGHT - 1) {
                        self.blocks[x][trunk_y][z] = BlockType::Wood;
                    }

                    // Small leaf canopy
                    let leaf_y = y + trunk_height;
                    if leaf_y < CHUNK_HEIGHT - 2 {
                        for lx in x.saturating_sub(cfg.oasis_tree_leaf_radius)..=(x + cfg.oasis_tree_leaf_radius).min(CHUNK_SIZE - 1) {
                            for lz in z.saturating_sub(cfg.oasis_tree_leaf_radius)..=(z + cfg.oasis_tree_leaf_radius).min(CHUNK_SIZE - 1) {
                                for ly in leaf_y..=(leaf_y + 2).min(CHUNK_HEIGHT - 1) {
                                    let dx = (lx as i32 - x as i32).abs();
                                    let dz = (lz as i32 - z as i32).abs();
                                    if dx + dz <= cfg.oasis_tree_leaf_max_dist && self.blocks[lx][ly][lz] == BlockType::Air {
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
        let glow_perlin = Perlin::new(seed_glowstone);
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let world_x = (world_offset_x + x as i32) as f64;
                let world_z = (world_offset_z + z as i32) as f64;
                for y in cfg.glowstone_min_y..cfg.glowstone_max_y {
                    if self.blocks[x][y][z] == BlockType::Stone {
                        let noise_val = glow_perlin.get([world_x * cfg.glowstone_scale, y as f64 * cfg.glowstone_scale, world_z * cfg.glowstone_scale]);
                        if noise_val > cfg.glowstone_threshold {
                            self.blocks[x][y][z] = BlockType::GlowStone;
                        }
                    }
                }
            }
        }

        // === Floating Sky Islands ===
        // Sky islands appear over desert and ocean biomes
        let stalactite_perlin = Perlin::new(seed_stalactite);
        let hill_perlin = Perlin::new(seed_hill);

        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                // Only generate sky islands over desert or ocean biomes.
                // Use the stronger of the two eligible biome weights as the fade input.
                let biome_weights = column_biomes[x][z];
                let eligible_weight = biome_weights.ocean.max(biome_weights.desert);
                if eligible_weight < cfg.sky_island_min_biome_weight {
                    continue;
                }

                let world_x = (world_offset_x + x as i32) as f64;
                let world_z = (world_offset_z + z as i32) as f64;

                // Higher threshold = rarer islands, larger scale = smaller islands
                let island_mask = (sky_island_mask_perlin.get([world_x * cfg.sky_island_mask_scale, world_z * cfg.sky_island_mask_scale]) + 1.0) * 0.5;

                if island_mask < cfg.sky_island_mask_threshold {
                    continue;
                }

                let height_noise = (sky_island_detail.get([world_x * cfg.sky_island_detail_scale, world_z * cfg.sky_island_detail_scale]) + 1.0) * 0.5;
                let island_center_y = cfg.sky_island_base_y + (height_noise * cfg.sky_island_height_range as f64) as usize;

                let hill_noise = (hill_perlin.get([world_x * cfg.sky_island_hill_scale, world_z * cfg.sky_island_hill_scale]) + 1.0) * 0.5;
                let hill_height = (hill_noise * cfg.sky_island_max_hill_height) as usize;

                let horizontal_island_noise = (sky_island_perlin.get([
                    world_x * cfg.sky_island_scale,
                    island_center_y as f64 * cfg.sky_island_scale * 0.3,
                    world_z * cfg.sky_island_scale,
                ]) + 1.0) * 0.5;

                let centeredness = smoothstep(0.55, 0.85, horizontal_island_noise);

                let stalactite_noise = (stalactite_perlin.get([world_x * cfg.sky_island_stalactite_scale, world_z * cfg.sky_island_stalactite_scale]) + 1.0) * 0.5;

                // Smaller stalactites
                let base_stalactite = if stalactite_noise > cfg.sky_island_stalactite_threshold {
                    1 + ((stalactite_noise - cfg.sky_island_stalactite_threshold) * cfg.sky_island_stalactite_scale_factor) as usize
                } else {
                    0
                };

                let center_bonus = (centeredness * cfg.sky_island_center_bonus) as usize;
                let stalactite_depth = base_stalactite + center_bonus;

                let island_min_y = island_center_y.saturating_sub(cfg.sky_island_base_thickness / 2 + stalactite_depth);
                let island_max_y = (island_center_y + cfg.sky_island_base_thickness / 2 + hill_height + 1).min(CHUNK_HEIGHT);

                // Biome edge fade: eligible_weight runs from sky_island_min_biome_weight (edge)
                // to sky_island_biome_fade_end (well inside biome). Multiplying island_strength
                // by this factor raises the voxel threshold near the boundary so the island
                // thins out and disappears organically instead of being cut off with a flat wall.
                let biome_fade = smoothstep(cfg.sky_island_min_biome_weight, cfg.sky_island_biome_fade_end, eligible_weight);
                let island_strength = smoothstep(cfg.sky_island_mask_threshold, cfg.sky_island_strength_threshold, island_mask) * biome_fade;

                for y in island_min_y..island_max_y {
                    let world_y = y as f64;

                    let island_noise = sky_island_perlin.get([
                        world_x * cfg.sky_island_scale,
                        world_y * cfg.sky_island_scale * 0.3,
                        world_z * cfg.sky_island_scale,
                    ]);

                    let effective_center_y = (island_center_y + hill_height / 2) as f64;
                    let effective_thickness = cfg.sky_island_base_thickness as f64 + hill_height as f64 / 2.0 + stalactite_depth as f64 / 2.0;

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

                    let threshold = 0.50 - (island_strength * 0.20) - (taper * 0.15) + (1.0 - biome_fade) * 0.60;

                    if island_noise > threshold && self.blocks[x][y][z] == BlockType::Air {
                        let is_surface = y + 1 >= island_max_y || {
                            let above_y = (y + 1) as f64;
                            let above_noise = sky_island_perlin.get([
                                world_x * cfg.sky_island_scale,
                                above_y * cfg.sky_island_scale * 0.3,
                                world_z * cfg.sky_island_scale,
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
                            let above_threshold = 0.50 - (island_strength * 0.20) - (above_taper * 0.15) + (1.0 - biome_fade) * 0.60;
                            above_noise <= above_threshold
                        };

                        let mut depth_from_surface = 0;
                        for check_y in (y + 1)..island_max_y.min(y + 4) {
                            let check_world_y = check_y as f64;
                            let check_noise = sky_island_perlin.get([
                                world_x * cfg.sky_island_scale,
                                check_world_y * cfg.sky_island_scale * 0.3,
                                world_z * cfg.sky_island_scale,
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
                            let check_threshold = 0.50 - (island_strength * 0.20) - (check_taper * 0.15) + (1.0 - biome_fade) * 0.60;
                            if check_noise > check_threshold {
                                depth_from_surface += 1;
                            } else {
                                break;
                            }
                        }

                        // Block types depend on biome
                        self.blocks[x][y][z] = if biome_weights.desert >= biome_weights.ocean {
                            // Desert sky islands are sandstone-like
                            if is_below_center {
                                BlockType::Stone
                            } else if is_surface {
                                BlockType::Sand
                            } else if depth_from_surface < 2 {
                                BlockType::Sand
                            } else {
                                BlockType::Stone
                            }
                        } else {
                            // Ocean sky islands are grassy
                            if is_below_center {
                                BlockType::Stone
                            } else if is_surface {
                                BlockType::Dirt
                            } else if depth_from_surface < 2 {
                                BlockType::Dirt
                            } else {
                                BlockType::Stone
                            }
                        };
                    }
                }
            }
        }

        // === Sky Island Trees (Ocean islands only) ===
        // Dense forest with wide height variety on ocean sky islands
        for x in cfg.tree_border_buffer..CHUNK_SIZE - cfg.tree_border_buffer {
            for z in cfg.tree_border_buffer..CHUNK_SIZE - cfg.tree_border_buffer {
                for y in (cfg.sky_island_base_y..CHUNK_HEIGHT.saturating_sub(cfg.tree_max_height)).rev() {
                    if self.blocks[x][y][z] != BlockType::Dirt {
                        continue;
                    }
                    // Verify there's air above (actual surface)
                    if y + 1 >= CHUNK_HEIGHT || self.blocks[x][y + 1][z] != BlockType::Air {
                        continue;
                    }

                    let world_x = (world_offset_x + x as i32) as f64;
                    let world_z = (world_offset_z + z as i32) as f64;

                    // Dense placement - low threshold so most grass blocks get a tree
                    let tree_noise = perlin.get([world_x * cfg.tree_noise_scale, world_z * cfg.tree_noise_scale]);
                    if tree_noise < cfg.tree_threshold_forest_base {
                        continue;
                    }

                    // Tight spacing for dense canopy
                    let mut too_close = false;
                    'sky_check: for check_x in x.saturating_sub(cfg.tree_spacing_forest_dense)..=(x + cfg.tree_spacing_forest_dense).min(CHUNK_SIZE - 1) {
                        for check_z in z.saturating_sub(cfg.tree_spacing_forest_dense)..=(z + cfg.tree_spacing_forest_dense).min(CHUNK_SIZE - 1) {
                            if check_x == x && check_z == z { continue; }
                            if y + 1 < CHUNK_HEIGHT && self.blocks[check_x][y + 1][check_z] == BlockType::Wood {
                                too_close = true;
                                break 'sky_check;
                            }
                        }
                    }
                    if too_close { continue; }

                    // Wide height variety - same distribution as forest biome
                    let tree_seed = ((world_x as i64).wrapping_mul(cfg.tree_seed_hash_1) ^ (world_z as i64).wrapping_mul(cfg.tree_seed_hash_2)) as u64;
                    let mut rng = StdRng::seed_from_u64(tree_seed);
                    let height_roll = rng.gen::<f64>();
                    let trunk_height = if height_roll < cfg.forest_tree_short_chance {
                        rng.gen_range(cfg.forest_tree_short_min..=cfg.forest_tree_short_max)
                    } else if height_roll < cfg.forest_tree_medium_chance {
                        rng.gen_range(cfg.forest_tree_medium_min..=cfg.forest_tree_medium_max)
                    } else if height_roll < cfg.forest_tree_tall_chance {
                        rng.gen_range(cfg.forest_tree_tall_min..=cfg.forest_tree_tall_max)
                    } else {
                        rng.gen_range(cfg.forest_tree_very_tall_min..=cfg.forest_tree_very_tall_max)
                    };

                    // Trunk
                    for trunk_y in (y + 1)..=(y + trunk_height).min(CHUNK_HEIGHT - 1) {
                        self.blocks[x][trunk_y][z] = BlockType::Wood;
                    }

                    // Branches for taller trees
                    let branch_start_y = y + trunk_height.saturating_sub(2);
                    let should_branch = rng.gen::<f64>() < cfg.tree_branch_chance;
                    if should_branch && trunk_height >= cfg.branch_min_trunk_height {
                        let num_branches = rng.gen_range(cfg.branch_count_min..=cfg.branch_count_max);
                        for _ in 0..num_branches {
                            let branch_y = rng.gen_range(branch_start_y..=y + trunk_height);
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

                    // Leaves - scale canopy with trunk height
                    let leaf_radius = if trunk_height >= cfg.tree_height_large_threshold { cfg.leaf_radius_large } else if trunk_height >= cfg.tree_height_medium_threshold { cfg.leaf_radius_medium } else { cfg.leaf_radius_small };
                    let leaf_height = if trunk_height >= cfg.tree_height_large_threshold { cfg.leaf_height_large } else if trunk_height >= cfg.tree_height_medium_threshold { cfg.leaf_height_medium } else { cfg.leaf_height_small };
                    let leaf_start_y = y + trunk_height - 1;

                    for lx in x.saturating_sub(leaf_radius)..=(x + leaf_radius).min(CHUNK_SIZE - 1) {
                        for lz in z.saturating_sub(leaf_radius)..=(z + leaf_radius).min(CHUNK_SIZE - 1) {
                            for ly in leaf_start_y..=(leaf_start_y + leaf_height).min(CHUNK_HEIGHT - 1) {
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

                    break; // One tree per column
                }
            }
        }

        // === Grass Tufts ===
        // Spawn cross-model grass tufts on grass blocks using noise for natural clustering.
        // Plains uses a higher threshold for regular tufts (GrassTuftTall dominates there instead).
        let tuft_noise = Perlin::new(seed_grass_tuft);
        let tall_tuft_noise = Perlin::new(seed_grass_tuft.wrapping_add(1));
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let world_x = (world_offset_x + x as i32) as f64;
                let world_z = (world_offset_z + z as i32) as f64;

                let is_plains = column_biomes[x][z].dominant() == BiomeType::Plains;

                // Regular GrassTuft: sparse in Plains (tall tufts take priority there)
                let regular_threshold = if is_plains { cfg.grass_tuft_threshold + 0.5 } else { cfg.grass_tuft_threshold };
                let regular_noise = tuft_noise.get([world_x * cfg.grass_tuft_noise_scale, world_z * cfg.grass_tuft_noise_scale]);
                if regular_noise >= regular_threshold {
                    for y in (cfg.sea_level..CHUNK_HEIGHT - 1).rev() {
                        if self.blocks[x][y][z] == BlockType::Dirt && self.blocks[x][y + 1][z] == BlockType::Air {
                            self.blocks[x][y + 1][z] = BlockType::GrassTuft;
                            break;
                        }
                    }
                }

                // GrassTuftTall: sparse globally, much more frequent in Plains.
                // Placed after regular tufts so it can overwrite them.
                let tall_threshold = if is_plains { cfg.grass_tuft_tall_plains_threshold } else { cfg.grass_tuft_tall_threshold };
                let tall_noise = tall_tuft_noise.get([world_x * cfg.grass_tuft_noise_scale, world_z * cfg.grass_tuft_noise_scale]);
                if tall_noise >= tall_threshold {
                    for y in (cfg.sea_level..CHUNK_HEIGHT - 1).rev() {
                        if self.blocks[x][y][z] == BlockType::Dirt &&
                            (self.blocks[x][y + 1][z] == BlockType::Air || self.blocks[x][y + 1][z] == BlockType::GrassTuft)
                        {
                            self.blocks[x][y + 1][z] = BlockType::GrassTuftTall;
                            break;
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

    pub fn generate_mesh(neighbors: &ChunkNeighbors, smooth_lighting: bool) -> (Vec<Vertex>, Vec<u16>, Vec<Vertex>, Vec<u16>, Vec<Vertex>, Vec<u16>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut water_vertices = Vec::new();
        let mut water_indices = Vec::new();
        let mut transparent_vertices = Vec::new();
        let mut transparent_indices = Vec::new();
        let chunk = neighbors.center;

        let world_offset_x = chunk.position.0 * CHUNK_SIZE as i32;
        let world_offset_z = chunk.position.1 * CHUNK_SIZE as i32;

        // Noise for leaf color variation (green ↔ orange)
        let leaf_color_noise = Perlin::new(neighbors.center.master_seed.wrapping_add(700));

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

                    // Cross-model blocks (grass tufts) use special geometry instead of cube faces
                    if block.is_cross_model() {
                        let world_pos = Vector3::new(
                            (world_offset_x + x as i32) as f32,
                            y as f32,
                            (world_offset_z + z as i32) as f32,
                        );

                        let face_textures = block.get_face_textures(false);
                        let tex_index = face_textures.get_for_face(0);
                        let uvs = if tex_index != TEX_NONE { get_face_uvs(tex_index) } else { [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]] };

                        // Sample light from above the block for consistent brightness
                        let light = neighbors.get_light(x as i32, y as i32 + 1, z as i32);
                        let light_normalized = light as f32 / 15.0;

                        // Use same leaf color noise for consistent biome-wide coloring
                        let wx = (world_offset_x + x as i32) as f64;
                        let wy = y as f64;
                        let wz = (world_offset_z + z as i32) as f64;
                        let noise_val = leaf_color_noise.get([wx * 0.02, wy * 0.02, wz * 0.02]);
                        let t_raw = (noise_val as f32 * 0.5 + 0.5).clamp(0.0, 1.0);
                        // Plains: smoothly bias t into the upper (orange) half based on plains weight
                        let plains_w = chunk.plains_weight_map[x][z];
                        let t = t_raw + plains_w * 0.5 * (1.0 - t_raw);
                        let tint = [
                            0.3 + t * (1.0 - 0.3),
                            0.95 + t * (0.65 - 0.95),
                            0.2 + t * (0.1 - 0.2),
                        ];

                        // Deterministic hash for per-instance scale and angle variation
                        let wx_i = world_offset_x + x as i32;
                        let wz_i = world_offset_z + z as i32;
                        let cross_hash = (wx_i.wrapping_mul(73856093) ^ (y as i32).wrapping_mul(19349663) ^ wz_i.wrapping_mul(83492791)) as u32;

                        let (cross_verts, cross_indices) = create_cross_model_vertices(
                            world_pos, light_normalized, tex_index, uvs, tint, cross_hash,
                        );

                        let base_index = vertices.len() as u16;
                        vertices.extend_from_slice(&cross_verts);
                        for &idx in &cross_indices {
                            indices.push(base_index + idx);
                        }
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
                        // For transparent blocks (leaves, air, etc.), draw faces against any different block type
                        let is_semi_transparent = block.is_semi_transparent();
                        // Track leaf-leaf boundaries so the winning side can emit a double-sided face.
                        let is_leaf_leaf_face = block == BlockType::Leaves && neighbor_block == BlockType::Leaves;
                        let should_draw = if is_water {
                            neighbor_block.is_transparent_for_water() && neighbor_block != block
                        } else if is_semi_transparent {
                            // Semi-transparent blocks always draw all faces (except against same block type)
                            neighbor_block != block
                        } else if is_leaf_leaf_face {
                            // For two adjacent leaf blocks, only the "positive-direction" side renders,
                            // but it emits a double-sided face so it is visible from both directions.
                            dx + dy + dz > 0
                        } else if block == BlockType::Leaves {
                            true
                        } else {
                            (block.is_transparent() || neighbor_block.is_transparent() || neighbor_block.is_semi_transparent()) && neighbor_block != block
                        };

                        if should_draw {
                            let light = neighbors.get_light(nx, ny, nz);

                            // Compute per-vertex light: smooth mode averages the face-adjacent block
                            // with its three AO-corner neighbors; flat mode uses only the face block.
                            //
                            // For opaque solid AO-neighbors we fall back to l0 (the face-adjacent
                            // light) instead of sampling their internal value (which is always 0).
                            // Solid blocks carry no meaningful open-air light — AO already handles
                            // the corner darkening from solid geometry, so averaging in their 0s
                            // would double-darken edges and corners.
                            let light_values: [f32; 4] = if smooth_lighting {
                                std::array::from_fn(|v| {
                                    let offsets = &AO_OFFSETS[face_idx][v];
                                    let l0 = light as f32;

                                    // Sample a neighbor's light, but use l0 for opaque solids.
                                    let mut sample = |ox: i32, oy: i32, oz: i32| -> f32 {
                                        let b = neighbors.get_block(
                                            x as i32 + ox, y as i32 + oy, z as i32 + oz,
                                        );
                                        if b.is_solid() && !b.is_transparent() {
                                            l0
                                        } else {
                                            neighbors.get_light(
                                                x as i32 + ox, y as i32 + oy, z as i32 + oz,
                                            ) as f32
                                        }
                                    };

                                    let l1 = sample(offsets[0][0], offsets[0][1], offsets[0][2]);
                                    let l2 = sample(offsets[1][0], offsets[1][1], offsets[1][2]);
                                    let l3 = sample(offsets[2][0], offsets[2][1], offsets[2][2]);
                                    (l0 + l1 + l2 + l3) / (4.0 * 15.0)
                                })
                            } else {
                                [light as f32 / 15.0; 4]
                            };

                            // Check if block above is solid (for grass/dirt texture selection)
                            let block_above = neighbors.get_block(x as i32, y as i32 + 1, z as i32);
                            let has_block_above = block_above.is_solid();

                            // Get texture info for this block
                            let face_textures = block.get_face_textures(has_block_above);
                            let tex_index = face_textures.get_for_face(face_idx);

                            // Get UVs for this texture (or default for non-textured)
                            let uvs = if tex_index != TEX_NONE {
                                let base_uvs = get_face_uvs(tex_index);
                                // Rotate leaf texture UVs randomly per face for visual variety
                                if block == BlockType::Leaves {
                                    let wx = world_offset_x + x as i32;
                                    let wy = y as i32;
                                    let wz = world_offset_z + z as i32;
                                    let hash = wx.wrapping_mul(73856093) ^ wy.wrapping_mul(19349663) ^ wz.wrapping_mul(83492791) ^ (face_idx as i32).wrapping_mul(48611);
                                    let rotation = (hash as u32 % 4) as usize;
                                    rotate_face_uvs(base_uvs, rotation)
                                } else {
                                    base_uvs
                                }
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
                                    world_pos, face_idx, light_values, tex_index, uvs, edge_factors, is_surface_water
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
                                    world_pos, block, face_idx, light_values, alpha, tex_index, uvs, ao_values
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
                                // Determine if this is exposed dirt (grass rendering)
                                let is_grass_dirt = block == BlockType::Dirt && !has_block_above && !block_above.is_water();

                                // Compute noise tint parameter for leaves and grass (shared noise)
                                let needs_tint = block == BlockType::Leaves || is_grass_dirt;
                                let tint_t = if needs_tint {
                                    let wx = (world_offset_x + x as i32) as f64;
                                    let wy = y as f64;
                                    let wz = (world_offset_z + z as i32) as f64;
                                    let noise_val = leaf_color_noise.get([wx * 0.02, wy * 0.02, wz * 0.02]);
                                    let t_raw = (noise_val as f32 * 0.5 + 0.5).clamp(0.0, 1.0);
                                    // Plains: smoothly bias t into the upper (orange) half based on plains weight
                                    let plains_w = chunk.plains_weight_map[x][z];
                                    t_raw + plains_w * 0.5 * (1.0 - t_raw)
                                } else {
                                    0.0
                                };
                                let tint = if needs_tint {
                                    [
                                        0.3 + tint_t * (1.0 - 0.3),
                                        0.95 + tint_t * (0.65 - 0.95),
                                        0.2 + tint_t * (0.1 - 0.2),
                                    ]
                                } else {
                                    [1.0, 1.0, 1.0]
                                };

                                // For exposed dirt top face: use grass_top texture with tint
                                let (actual_tex, actual_uvs) = if is_grass_dirt && face_idx == 2 {
                                    let grass_uvs = get_face_uvs(TEX_GRASS_TOP);
                                    (TEX_GRASS_TOP, grass_uvs)
                                } else {
                                    (tex_index, uvs)
                                };

                                let face_verts = if block == BlockType::Leaves || (is_grass_dirt && face_idx == 2) {
                                    // Leaves: always tinted. Grass dirt top face: grass_top with tint.
                                    create_face_vertices_tinted(world_pos, face_idx, light_values, actual_tex, actual_uvs, ao_values, tint)
                                } else if is_grass_dirt && face_idx != 2 && face_idx != 3 {
                                    // Side faces of exposed dirt: pack overlay index (bits 16-23) and
                                    // tint parameter (bits 24-31) into tex_index. Shader reconstructs
                                    // tint and uses vertex color (dirt color) for the base texture.
                                    let tint_byte = (tint_t * 255.0) as u32;
                                    let packed_tex = tex_index | ((TEX_GRASS_SIDE + 1) << 16) | (tint_byte << 24);
                                    create_face_vertices(world_pos, block, face_idx, light_values, packed_tex, uvs, ao_values)
                                } else {
                                    create_face_vertices(world_pos, block, face_idx, light_values, actual_tex, actual_uvs, ao_values)
                                };

                                let base_index = vertices.len() as u16;
                                vertices.extend_from_slice(&face_verts);

                                // Anisotropy fix: flip diagonal if it reduces AO discontinuity
                                if (ao_values[0] - ao_values[2]).abs() > (ao_values[1] - ao_values[3]).abs() {
                                    indices.extend_from_slice(&[
                                        base_index + 1, base_index + 2, base_index + 3,
                                        base_index + 3, base_index, base_index + 1,
                                    ]);
                                    // Leaf-leaf boundary: also emit reversed winding so the face is
                                    // visible from both sides without needing a second block to contribute.
                                    if is_leaf_leaf_face {
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
                                    // Leaf-leaf boundary: also emit reversed winding so the face is
                                    // visible from both sides without needing a second block to contribute.
                                    if is_leaf_leaf_face {
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