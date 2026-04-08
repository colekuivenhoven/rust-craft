use crate::block::{BlockType, Vertex, create_face_vertices, create_face_vertices_tinted, create_cross_model_vertices, create_vine_face_vertices, create_water_face_vertices, AO_OFFSETS, calculate_ao};
use crate::lighting;
use crate::texture::{get_face_uvs, rotate_face_uvs, TEX_NONE, TEX_GRASS_TOP, TEX_GRASS_SIDE, TEX_VINES, TEX_VINES_END};
pub use crate::terrain::biome::{BiomeType, BiomeWeights};
use cgmath::Vector3;
use noise::{NoiseFn, Perlin};

pub const CHUNK_SIZE: usize = 16;
pub const CHUNK_HEIGHT: usize = 128;

/// Water level constants: 0 = no water, 1-7 = flowing (height = level/8), 8 = source (full block).
pub const WATER_LEVEL_SOURCE: u8 = 8;
pub const WATER_LEVEL_MAX_FLOW: u8 = 7;
pub const WATER_LEVEL_MIN_FLOW: u8 = 1;
pub const WATER_SURFACE_HEIGHT: f32 = 1.0;

pub struct Chunk {
    pub blocks: Box<[[[BlockType; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]>,
    pub water_levels: Box<[[[u8; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]>,
    pub light_levels: Box<[[[u8; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]>,
    pub biome_map: Vec<Vec<BiomeType>>,         // [x][z] dominant biome per column, for mesh coloring
    pub plains_weight_map: Vec<Vec<f32>>,       // [x][z] continuous plains weight for smooth color blending
    pub position: (i32, i32),
    pub master_seed: u32,
    pub moss_threshold: f64,               // sky_castle_moss_threshold from cfg, used in meshing
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

    pub fn get_water_level(&self, x: i32, y: i32, z: i32) -> u8 {
        if y < 0 || y >= CHUNK_HEIGHT as i32 {
            return 0;
        }

        let (target_chunk, lx, lz) = self.resolve_coordinates(x, z);

        if lx >= CHUNK_SIZE || lz >= CHUNK_SIZE {
            return 0;
        }

        match target_chunk {
            Some(chunk) => chunk.water_levels[lx][y as usize][lz],
            None => 0,
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
        let water_levels: Box<[[[u8; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]> = vec![[[0u8; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]
            .into_boxed_slice()
            .try_into()
            .unwrap();
        let light_levels: Box<[[[u8; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]> = vec![[[0u8; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]
            .into_boxed_slice()
            .try_into()
            .unwrap();

        let mut chunk = Self {
            blocks,
            water_levels,
            light_levels,
            biome_map: vec![vec![BiomeType::Forest; CHUNK_SIZE]; CHUNK_SIZE],
            plains_weight_map: vec![vec![0.0f32; CHUNK_SIZE]; CHUNK_SIZE],
            position: (chunk_x, chunk_z),
            master_seed,
            moss_threshold: cfg.sky_castle_moss_threshold,
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
        crate::terrain::generation::generate_terrain(&mut chunk, master_seed, cfg);
        chunk
    }

    /// Creates a chunk with pre-loaded block data (from saved file)
    pub fn from_saved_data(chunk_x: i32, chunk_z: i32, blocks: Box<[[[BlockType; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]>, water_levels: Box<[[[u8; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]>) -> Self {
        let light_levels: Box<[[[u8; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]> = vec![[[0u8; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]
            .into_boxed_slice()
            .try_into()
            .unwrap();

        Self {
            blocks,
            water_levels,
            light_levels,
            biome_map: vec![vec![BiomeType::Forest; CHUNK_SIZE]; CHUNK_SIZE],
            plains_weight_map: vec![vec![0.0f32; CHUNK_SIZE]; CHUNK_SIZE],
            position: (chunk_x, chunk_z),
            master_seed: 0, // Not used for saved chunks (blocks already generated)
            moss_threshold: 0.0,
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

    pub fn get_water_level(&self, x: usize, y: usize, z: usize) -> u8 {
        if x >= CHUNK_SIZE || y >= CHUNK_HEIGHT || z >= CHUNK_SIZE {
            0
        } else {
            self.water_levels[x][y][z]
        }
    }

    pub fn set_water_level(&mut self, x: usize, y: usize, z: usize, level: u8) {
        if x < CHUNK_SIZE && y < CHUNK_HEIGHT && z < CHUNK_SIZE {
            self.water_levels[x][y][z] = level;
            self.dirty = true;
            self.modified = true;
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

                        // Sample light from the block's own position. Sampling y+1 breaks when
                        // a solid block sits directly overhead (its stored light is 0), making
                        // tufts under overhangs completely black. The tuft position itself gets
                        // correct light via horizontal propagation from adjacent air.
                        let light_above = neighbors.get_light(x as i32, y as i32 + 1, z as i32);
                        let light_self  = neighbors.get_light(x as i32, y as i32,     z as i32);
                        let light = light_above.max(light_self);
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

                    // Vine blocks: emit one thin face-panel per adjacent solid block.
                    if block == BlockType::Vines {
                        let world_pos = Vector3::new(
                            (world_offset_x + x as i32) as f32,
                            y as f32,
                            (world_offset_z + z as i32) as f32,
                        );

                        // Last vine in column should use TEX_VINES_END
                        let block_below = neighbors.get_block(x as i32, y as i32 - 1, z as i32);
                        let tex_index = if block_below == BlockType::Vines {
                            TEX_VINES
                        } else {
                            TEX_VINES_END
                        };
                        let uvs = if tex_index != TEX_NONE { get_face_uvs(tex_index) } else { [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]] };

                        // Same fix as cross-model blocks: max of self and above so that vines
                        // against a solid overhang still get horizontally-propagated light.
                        let light_above = neighbors.get_light(x as i32, y as i32 + 1, z as i32);
                        let light_self  = neighbors.get_light(x as i32, y as i32,     z as i32);
                        let light = light_above.max(light_self);
                        let light_normalized = light as f32 / 15.0;

                        // Foliage color tint — same noise as leaves/grass.
                        let wx = (world_offset_x + x as i32) as f64;
                        let wy = y as f64;
                        let wz = (world_offset_z + z as i32) as f64;
                        let noise_val = leaf_color_noise.get([wx * 0.02, wy * 0.02, wz * 0.02]);
                        let t_raw = (noise_val as f32 * 0.5 + 0.5).clamp(0.0, 1.0);
                        let plains_w = chunk.plains_weight_map[x][z];
                        let t = t_raw + plains_w * 0.5 * (1.0 - t_raw);
                        let vine_tint = [
                            0.3 + t * (1.0 - 0.3),
                            0.95 + t * (0.65 - 0.95),
                            0.2 + t * (0.1 - 0.2),
                        ];

                        // wall_dir: 0=-X 1=+X 2=-Z 3=+Z
                        const HORIZ: [(i32, i32, u8); 4] = [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)];

                        // Collect which wall directions to face. Hanging vine blocks (below
                        // the attachment row) have no direct solid horizontal neighbour, so we
                        // walk upward through the vine column to inherit the attachment block's
                        // wall direction — this makes the entire draping column visible.
                        let mut wall_dirs: Vec<u8> = Vec::new();
                        for (dx, dz, wd) in HORIZ {
                            let nb = neighbors.get_block(x as i32 + dx, y as i32, z as i32 + dz);
                            if nb.is_solid() && !nb.is_water() {
                                wall_dirs.push(wd);
                            }
                        }
                        if wall_dirs.is_empty() {
                            'upwalk: for up in 1i32..=20 {
                                let vy = y as i32 + up;
                                if neighbors.get_block(x as i32, vy, z as i32) != BlockType::Vines { break; }
                                for (dx, dz, wd) in HORIZ {
                                    let nb = neighbors.get_block(x as i32 + dx, vy, z as i32 + dz);
                                    if nb.is_solid() && !nb.is_water() {
                                        wall_dirs.push(wd);
                                        break 'upwalk;
                                    }
                                }
                            }
                        }
                        for wall_dir in wall_dirs {
                            let (verts, idxs) = create_vine_face_vertices(world_pos, wall_dir, light_normalized, tex_index, uvs, vine_tint);
                            let base = vertices.len() as u16;
                            vertices.extend_from_slice(&verts);
                            for &i in &idxs { indices.push(base + i); }
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
                            let light_values: [f32; 4] = if block.get_light_emission() > 0 {
                                // Emissive blocks: encode emission in light_level > 1.0
                                // Shader detects this and applies bloom/glow effect
                                let emit = block.get_light_emission() as f32 / 15.0;
                                [2.0 + emit; 4]
                            } else if smooth_lighting {
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

                                // Compute per-corner heights and wave scales for water surface slopes
                                // Corner order: [0]=(-X,+Z), [1]=(+X,+Z), [2]=(+X,-Z), [3]=(-X,-Z)
                                let (corner_heights, corner_wave_scales) = if is_surface_water {
                                    use crate::block::WAVE_AMPLITUDE;
                                    let base_y = y as f32;

                                    // For each corner, find the max water level among the 4 blocks sharing it
                                    let corner_offsets: [(i32, i32); 4] = [
                                        (-1, 1),  // corner 0: (-X, +Z)
                                        (1, 1),   // corner 1: (+X, +Z)
                                        (1, -1),  // corner 2: (+X, -Z)
                                        (-1, -1), // corner 3: (-X, -Z)
                                    ];

                                    let mut heights = [0.0f32; 4];
                                    let mut wave_scales = [0.0f32; 4];
                                    for (i, &(cx, cz)) in corner_offsets.iter().enumerate() {
                                        let positions = [
                                            (x as i32, z as i32),
                                            (x as i32 + cx, z as i32),
                                            (x as i32, z as i32 + cz),
                                            (x as i32 + cx, z as i32 + cz),
                                        ];

                                        let mut max_level: u8 = 0;
                                        let mut any_above_water = false;
                                        for &(px, pz) in &positions {
                                            let b = neighbors.get_block(px, y as i32, pz);
                                            if b == BlockType::Water {
                                                let lvl = neighbors.get_water_level(px, y as i32, pz);
                                                max_level = max_level.max(lvl);
                                            }
                                            let above = neighbors.get_block(px, y as i32 + 1, pz);
                                            if above == BlockType::Water {
                                                any_above_water = true;
                                            }
                                        }

                                        // Wave scale proportional to water level at this corner
                                        let height_frac;
                                        if any_above_water || max_level >= WATER_LEVEL_SOURCE {
                                            height_frac = WATER_SURFACE_HEIGHT;
                                        } else if max_level > 0 {
                                            height_frac = max_level as f32 / WATER_LEVEL_SOURCE as f32;
                                        } else {
                                            let own_level = neighbors.get_water_level(x as i32, y as i32, z as i32);
                                            height_frac = own_level as f32 / WATER_LEVEL_SOURCE as f32;
                                        }

                                        let wave_scale = height_frac;
                                        let wave_offset = 0.0 * WAVE_AMPLITUDE * wave_scale;
                                        heights[i] = base_y + height_frac - wave_offset;
                                        wave_scales[i] = wave_scale;
                                    }
                                    (heights, wave_scales)
                                } else {
                                    // Non-surface water: full block height, no waves
                                    let full = y as f32 + 1.0;
                                    ([full, full, full, full], [0.0f32; 4])
                                };

                                // Calculate edge flags for foam rendering as a bitmask
                                // Encodes which edges have solid neighbors (same value for all vertices to avoid interpolation)
                                // Bitmask: neg_x=1, pos_x=2, neg_z=4, pos_z=8 (divided by 16 to fit in 0-1)
                                let edge_flags: f32 = if face_idx == 2 {
                                    // Top face - check horizontal neighbors for shore foam
                                    let foams_water = |b: BlockType| b.is_solid() || matches!(b, BlockType::GrassTuft | BlockType::GrassTuftTall);
                                    let solid_neg_x = foams_water(neighbors.get_block(x as i32 - 1, y as i32, z as i32));
                                    let solid_pos_x = foams_water(neighbors.get_block(x as i32 + 1, y as i32, z as i32));
                                    let solid_neg_z = foams_water(neighbors.get_block(x as i32, y as i32, z as i32 - 1));
                                    let solid_pos_z = foams_water(neighbors.get_block(x as i32, y as i32, z as i32 + 1));

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
                                    world_pos, face_idx, light_values, tex_index, uvs, edge_factors, is_surface_water, corner_heights, corner_wave_scales
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

                                // Exposed castle Cobblestone/Stone gets a moss overlay (same technique
                                // as grass-on-dirt). Noise-gated so only ~half of eligible blocks are
                                // mossy, avoiding a uniform carpet look.
                                let is_mossy_stone = (block == BlockType::Cobblestone || block == BlockType::Stone)
                                    && !has_block_above && !block_above.is_water()
                                    && {
                                        let wx_m = (world_offset_x + x as i32) as f64;
                                        let wz_m = (world_offset_z + z as i32) as f64;
                                        leaf_color_noise.get([wx_m * 0.09 + 50.3, wz_m * 0.09 + 50.3]) > neighbors.center.moss_threshold
                                    };

                                let has_overlay = is_grass_dirt || is_mossy_stone;

                                // Compute noise tint parameter for leaves, grass, and moss (shared noise)
                                let needs_tint = block == BlockType::Leaves || has_overlay;
                                let tint_t = if needs_tint {
                                    let wx = (world_offset_x + x as i32) as f64;
                                    let wy = y as f64;
                                    let wz = (world_offset_z + z as i32) as f64;
                                    let noise_val = leaf_color_noise.get([wx * 0.02, wy * 0.02, wz * 0.02]);
                                    let t_raw = (noise_val as f32 * 0.5 + 0.5).clamp(0.0, 1.0);
                                    // Same formula for all foliage — plains bias shifts toward orange.
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

                                // For exposed top faces (dirt or mossy stone): use grass_top texture
                                let (actual_tex, actual_uvs) = if has_overlay && face_idx == 2 {
                                    let grass_uvs = get_face_uvs(TEX_GRASS_TOP);
                                    (TEX_GRASS_TOP, grass_uvs)
                                } else {
                                    (tex_index, uvs)
                                };

                                let face_verts = if block == BlockType::Leaves || (has_overlay && face_idx == 2) {
                                    // Leaves: always tinted. Overlay top face: grass_top with tint.
                                    create_face_vertices_tinted(world_pos, face_idx, light_values, actual_tex, actual_uvs, ao_values, tint)
                                } else if has_overlay && face_idx != 2 && face_idx != 3 {
                                    // Side faces of overlay blocks: pack overlay index (bits 16-23) and
                                    // tint parameter (bits 24-31) into tex_index. Shader reconstructs
                                    // tint and uses vertex color (base texture) for the main surface.
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