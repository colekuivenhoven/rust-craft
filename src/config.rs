/// Global configuration for the game
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Terrain generation configuration — all values are loaded from config.toml at startup.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TerrainConfig {
    // ── Terrain ──────────────────────────────────────────────────────────────
    pub sea_level: usize,
    pub terrain_base_scale: f64,
    pub terrain_detail_scale: f64,
    pub terrain_detail_amplitude: f64,
    pub mountain_ridge_scale: f64,
    pub mountain_ridge_amplitude: f64,
    pub mountain_jagged_scale: f64,
    pub mountain_jagged_amplitude: f64,

    // ── Biome distribution ────────────────────────────────────────────────────
    pub biome_scale: f64,
    pub biome_scale_multiplier: f64,
    pub continental_scale_factor: f64,
    pub ocean_threshold_deep: f64,
    pub ocean_threshold_shallow: f64,
    pub arctic_temp_threshold_high: f64,
    pub arctic_temp_threshold_low: f64,
    pub desert_temp_threshold_low: f64,
    pub desert_temp_threshold_high: f64,
    pub desert_humidity_threshold_high: f64,
    pub desert_humidity_threshold_low: f64,
    pub desert_inland_threshold_low: f64,
    pub desert_inland_threshold_high: f64,
    pub mountain_threshold_low: f64,
    pub mountain_threshold_high: f64,
    pub vein_scale: f64,

    // ── Plains biome ──────────────────────────────────────────────────────────
    pub plains_humidity_threshold_high: f64,
    pub plains_humidity_threshold_low: f64,

    // ── Biome heights ─────────────────────────────────────────────────────────
    pub ocean_height_base: f64,
    pub ocean_height_variation: f64,
    pub desert_height_base: f64,
    pub desert_height_variation: f64,
    pub desert_min_above_sea: f64,
    pub desert_pull_threshold: f64,
    pub desert_pull_strength: f64,
    pub forest_height_base: f64,
    pub forest_height_variation: f64,
    pub plains_height_base: f64,
    pub plains_height_variation: f64,
    pub arctic_height_base: f64,
    pub arctic_height_variation: f64,
    pub mountain_height_base: f64,
    pub mountain_height_variation: f64,
    pub coastal_start: f64,
    pub coastal_end: f64,
    pub coastal_min_height: f64,
    pub island_coastal_blend_width: f64,

    // ── Surface blocks ────────────────────────────────────────────────────────
    pub snow_threshold_offset: isize,
    pub stone_threshold_offset: isize,
    pub arctic_snow_threshold_offset: isize,
    pub surface_transition_scale: f64,
    pub grass_patch_base_threshold: f64,
    pub grass_patch_height_factor: f64,
    pub grass_patch_max_height: f64,
    pub arctic_ice_threshold: f64,
    pub arctic_full_ice_threshold: f64,

    // ── Ocean islands ─────────────────────────────────────────────────────────
    pub ocean_island_scale: f64,
    pub ocean_island_threshold: f64,
    pub ocean_island_strength_max: f64,
    pub ocean_island_max_bump: f64,
    pub ocean_island_grass_start: usize,

    // ── Trees ─────────────────────────────────────────────────────────────────
    pub tree_min_height: usize,
    pub tree_max_height: usize,
    pub tree_branch_chance: f64,
    pub tree_tall_chance: f64,
    pub tree_noise_scale: f64,
    pub tree_seed_hash_1: i64,
    pub tree_seed_hash_2: i64,
    pub tree_threshold_forest_base: f64,
    pub tree_threshold_forest_edge_add: f64,
    pub tree_threshold_mountain: f64,
    pub tree_threshold_desert: f64,
    pub tree_threshold_ocean: f64,
    pub tree_threshold_arctic: f64,
    pub tree_threshold_plains: f64,
    pub tree_spacing_plains: usize,
    pub tree_spacing_forest_dense: usize,
    pub tree_spacing_default: usize,
    pub tree_spacing_forest_weight_threshold: f64,
    pub forest_tree_short_chance: f64,
    pub forest_tree_medium_chance: f64,
    pub forest_tree_tall_chance: f64,
    pub forest_tree_short_min: usize,
    pub forest_tree_short_max: usize,
    pub forest_tree_medium_min: usize,
    pub forest_tree_medium_max: usize,
    pub forest_tree_tall_min: usize,
    pub forest_tree_tall_max: usize,
    pub forest_tree_very_tall_min: usize,
    pub forest_tree_very_tall_max: usize,
    pub leaf_radius_small: usize,
    pub leaf_radius_medium: usize,
    pub leaf_radius_large: usize,
    pub leaf_height_small: usize,
    pub leaf_height_medium: usize,
    pub leaf_height_large: usize,
    pub tree_height_large_threshold: usize,
    pub tree_height_medium_threshold: usize,
    pub branch_min_trunk_height: usize,
    pub branch_count_min: usize,
    pub branch_count_max: usize,
    pub tree_border_buffer: usize,
    pub plains_tree_branch_chance: f64,  // Branch chance for Plains trees (higher = more branchy)
    pub plains_leaf_radius: usize,       // Leaf radius for the flat Plains canopy
    pub plains_leaf_height: usize,       // Vertical layers for the flat Plains canopy

    // ── Oasis ─────────────────────────────────────────────────────────────────
    pub oasis_scale: f64,
    pub oasis_threshold: f64,
    pub oasis_strength_max: f64,
    pub oasis_max_depression: usize,
    pub oasis_min_desert_weight: f64,
    pub oasis_tree_noise_scale: f64,
    pub oasis_tree_threshold: f64,
    pub oasis_tree_min_height: usize,
    pub oasis_tree_max_height: usize,
    pub oasis_tree_spacing: usize,
    pub oasis_tree_leaf_radius: usize,
    pub oasis_tree_leaf_max_dist: i32,

    // ── Rivers ────────────────────────────────────────────────────────────────
    pub river_cell_size: f64,
    pub river_width: f64,
    pub river_bank_width: f64,
    pub river_sand_depth: usize,
    pub river_depth_avg: usize,
    pub river_depth_variation: f64,
    pub river_bank_y_range: usize,
    pub river_winding_scale: f64,
    pub river_winding_amplitude: f64,

    // ── Glacier ───────────────────────────────────────────────────────────────
    pub glacier_scale: f64,
    pub glacier_threshold: f64,
    pub glacier_max_height: f64,
    pub glacier_ice_gap_scale: f64,
    pub glacier_ice_gap_threshold: f64,
    pub glacier_taper_scale: f64,

    // ── Grass tufts ───────────────────────────────────────────────────────────
    pub grass_tuft_noise_scale: f64,
    pub grass_tuft_threshold: f64,
    pub grass_tuft_tall_threshold: f64,
    pub grass_tuft_tall_plains_threshold: f64,

    // ── Glowstone ─────────────────────────────────────────────────────────────
    pub glowstone_min_y: usize,
    pub glowstone_max_y: usize,
    pub glowstone_scale: f64,
    pub glowstone_threshold: f64,

    // ── Sky islands ───────────────────────────────────────────────────────────
    pub sky_island_base_y: usize,
    pub sky_island_height_range: usize,
    pub sky_island_min_biome_weight: f64,
    pub sky_island_mask_scale: f64,
    pub sky_island_detail_scale: f64,
    pub sky_island_scale: f64,
    pub sky_island_hill_scale: f64,
    pub sky_island_stalactite_scale: f64,
    pub sky_island_mask_threshold: f64,
    pub sky_island_strength_threshold: f64,
    pub sky_island_base_thickness: usize,
    pub sky_island_max_hill_height: f64,
    pub sky_island_stalactite_threshold: f64,
    pub sky_island_stalactite_scale_factor: f64,
    pub sky_island_center_bonus: f64,
    pub sky_island_biome_fade_end: f64,

    // ── Depth layers ──────────────────────────────────────────────────────────
    pub depth_near_surface: usize,
    pub depth_transition: usize,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            sea_level: 32,
            terrain_base_scale: 0.006,
            terrain_detail_scale: 0.06,
            terrain_detail_amplitude: 1.5,
            mountain_ridge_scale: 0.015,
            mountain_ridge_amplitude: 15.0,
            mountain_jagged_scale: 0.04,
            mountain_jagged_amplitude: 8.0,

            biome_scale: 0.0025,
            biome_scale_multiplier: 0.00025,
            continental_scale_factor: 0.2,
            ocean_threshold_deep: 0.25,
            ocean_threshold_shallow: 0.50,
            arctic_temp_threshold_high: 0.35,
            arctic_temp_threshold_low: 0.15,
            desert_temp_threshold_low: 0.55,
            desert_temp_threshold_high: 0.75,
            desert_humidity_threshold_high: 0.50,
            desert_humidity_threshold_low: 0.25,
            desert_inland_threshold_low: 0.60,
            desert_inland_threshold_high: 0.80,
            mountain_threshold_low: 0.55,
            mountain_threshold_high: 0.75,
            vein_scale: 0.02,
            plains_humidity_threshold_high: 0.50,
            plains_humidity_threshold_low: 0.15,

            ocean_height_base: 12.0,
            ocean_height_variation: 15.0,
            desert_height_base: 40.0,
            desert_height_variation: 5.0,
            desert_min_above_sea: 8.0,
            desert_pull_threshold: 0.2,
            desert_pull_strength: 0.6,
            forest_height_base: 36.0,
            forest_height_variation: 8.0,
            plains_height_base: 36.0,
            plains_height_variation: 3.5,
            arctic_height_base: 35.0,
            arctic_height_variation: 10.0,
            mountain_height_base: 42.0,
            mountain_height_variation: 12.0,
            coastal_start: 0.35,
            coastal_end: 0.45,
            coastal_min_height: 12.0,
            island_coastal_blend_width: 0.15,

            snow_threshold_offset: 38,
            stone_threshold_offset: 12,
            arctic_snow_threshold_offset: 3,
            surface_transition_scale: 0.03,
            grass_patch_base_threshold: 0.35,
            grass_patch_height_factor: 0.06,
            grass_patch_max_height: 12.0,
            arctic_ice_threshold: 0.25,
            arctic_full_ice_threshold: 0.55,

            ocean_island_scale: 0.01,
            ocean_island_threshold: 0.4,
            ocean_island_strength_max: 0.85,
            ocean_island_max_bump: 14.0,
            ocean_island_grass_start: 1,

            tree_min_height: 4,
            tree_max_height: 12,
            tree_branch_chance: 0.6,
            tree_tall_chance: 0.3,
            tree_noise_scale: 0.02,
            tree_seed_hash_1: 73856093,
            tree_seed_hash_2: 19349663,
            tree_threshold_forest_base: 0.3,
            tree_threshold_forest_edge_add: 0.5,
            tree_threshold_mountain: 0.6,
            tree_threshold_desert: 0.95,
            tree_threshold_ocean: 0.7,
            tree_threshold_arctic: 0.95,
            tree_threshold_plains: 0.70,
            tree_spacing_plains: 8,
            tree_spacing_forest_dense: 4,
            tree_spacing_default: 5,
            tree_spacing_forest_weight_threshold: 0.5,
            forest_tree_short_chance: 0.2,
            forest_tree_medium_chance: 0.6,
            forest_tree_tall_chance: 0.85,
            forest_tree_short_min: 3,
            forest_tree_short_max: 5,
            forest_tree_medium_min: 5,
            forest_tree_medium_max: 8,
            forest_tree_tall_min: 8,
            forest_tree_tall_max: 11,
            forest_tree_very_tall_min: 11,
            forest_tree_very_tall_max: 14,
            leaf_radius_small: 2,
            leaf_radius_medium: 3,
            leaf_radius_large: 4,
            leaf_height_small: 4,
            leaf_height_medium: 5,
            leaf_height_large: 6,
            tree_height_large_threshold: 10,
            tree_height_medium_threshold: 7,
            branch_min_trunk_height: 5,
            branch_count_min: 1,
            branch_count_max: 3,
            tree_border_buffer: 2,
            plains_tree_branch_chance: 0.9,
            plains_leaf_radius: 4,
            plains_leaf_height: 2,

            oasis_scale: 0.006,
            oasis_threshold: 0.90,
            oasis_strength_max: 0.97,
            oasis_max_depression: 3,
            oasis_min_desert_weight: 0.7,
            oasis_tree_noise_scale: 0.1,
            oasis_tree_threshold: 0.3,
            oasis_tree_min_height: 4,
            oasis_tree_max_height: 6,
            oasis_tree_spacing: 2,
            oasis_tree_leaf_radius: 2,
            oasis_tree_leaf_max_dist: 3,

            river_cell_size: 250.0,
            river_width: 8.0,
            river_bank_width: 2.0,
            river_sand_depth: 2,
            river_depth_avg: 3,
            river_depth_variation: 2.0,
            river_bank_y_range: 3,
            river_winding_scale: 0.003,
            river_winding_amplitude: 15.0,

            glacier_scale: 0.015,
            glacier_threshold: 0.50,
            glacier_max_height: 16.0,
            glacier_ice_gap_scale: 0.05,
            glacier_ice_gap_threshold: 0.15,
            glacier_taper_scale: 0.08,

            grass_tuft_noise_scale: 0.15,
            grass_tuft_threshold: -0.1,
            grass_tuft_tall_threshold: 0.65,
            grass_tuft_tall_plains_threshold: 0.05,

            glowstone_min_y: 5,
            glowstone_max_y: 25,
            glowstone_scale: 0.05,
            glowstone_threshold: 0.85,

            sky_island_base_y: 65,
            sky_island_height_range: 20,
            sky_island_min_biome_weight: 0.5,
            sky_island_mask_scale: 0.003,
            sky_island_detail_scale: 0.002,
            sky_island_scale: 0.015,
            sky_island_hill_scale: 0.04,
            sky_island_stalactite_scale: 0.08,
            sky_island_mask_threshold: 0.7,
            sky_island_strength_threshold: 0.85,
            sky_island_base_thickness: 3,
            sky_island_max_hill_height: 2.0,
            sky_island_stalactite_threshold: 0.35,
            sky_island_stalactite_scale_factor: 5.0,
            sky_island_center_bonus: 4.0,
            sky_island_biome_fade_end: 0.75,

            depth_near_surface: 3,
            depth_transition: 6,
        }
    }
}

impl TerrainConfig {
    /// Load terrain configuration from config.toml. Missing fields fall back to defaults.
    pub fn load_or_create(path: &Path) -> Self {
        match fs::read_to_string(path) {
            Ok(contents) => {
                match toml::from_str(&contents) {
                    Ok(config) => {
                        log::info!("Loaded terrain config from {}", path.display());
                        config
                    }
                    Err(e) => {
                        log::warn!("Failed to parse terrain config from {}: {}, using defaults", path.display(), e);
                        Self::default()
                    }
                }
            }
            Err(_) => {
                log::warn!("Config file not found at {}, using default terrain config", path.display());
                Self::default()
            }
        }
    }
}

/// World generation configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WorldConfig {
    pub master_seed: u32,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self { master_seed: 52 }
    }
}

impl WorldConfig {
    /// Load configuration from a TOML file, or create default if it doesn't exist
    pub fn load_or_create(path: &Path) -> Self {
        match fs::read_to_string(path) {
            Ok(contents) => {
                match toml::from_str(&contents) {
                    Ok(config) => {
                        log::info!("Loaded world config from {}", path.display());
                        config
                    }
                    Err(e) => {
                        log::warn!("Failed to parse world config: {}, using defaults", e);
                        Self::default()
                    }
                }
            }
            Err(_) => {
                log::info!("No config found, using default world config");
                let default = Self::default();
                let _ = default.save(path);
                default
            }
        }
    }

    /// Save configuration to a TOML file
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let toml_string = toml::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        fs::write(path, toml_string)?;
        Ok(())
    }
}

/// Fog configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FogConfig {
    pub enabled: bool,
    pub start: f32,
    pub end: f32,
    pub use_square_fog: bool,
}

/// Cloud configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CloudConfig {
    pub height: f32,
    pub pixel_size: f32,
    pub threshold: f64,
    pub noise_scale: f64,
    /// Cloud drift speed in world units per second (X axis).
    /// Z drift is automatically 30% of this value.
    pub noise_offset_change_speed: f64,
}

impl Default for FogConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            start: 200.0,
            end: 300.0,
            use_square_fog: false,
        }
    }
}

impl FogConfig {
    /// Load configuration from a TOML file, or create default if it doesn't exist
    pub fn load_or_create(path: &Path) -> Self {
        match fs::read_to_string(path) {
            Ok(contents) => {
                match toml::from_str(&contents) {
                    Ok(config) => {
                        log::info!("Loaded fog config from {}", path.display());
                        config
                    }
                    Err(e) => {
                        log::warn!("Failed to parse fog config: {}, using defaults", e);
                        let default = Self::default();
                        let _ = default.save(path);
                        default
                    }
                }
            }
            Err(_) => {
                log::info!("No fog config found, creating default at {}", path.display());
                let default = Self::default();
                let _ = default.save(path);
                default
            }
        }
    }

    /// Save configuration to a TOML file
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let toml_string = toml::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        fs::write(path, toml_string)?;
        log::info!("Saved fog config to {}", path.display());
        Ok(())
    }
}

/// Saved player state persisted between sessions.
/// Binary format (version 1): [u8 version | f32 x | f32 y | f32 z | f32 yaw | f32 pitch | u8 flags]
/// flags bit 0 = show_chunk_outlines, bit 1 = noclip_mode, bit 2 = show_enemy_hitboxes
#[derive(Debug, Clone, Copy)]
pub struct PlayerSave {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub show_chunk_outlines: bool,
    pub noclip_mode: bool,
    pub show_enemy_hitboxes: bool,
}

const PLAYER_FILE_VERSION: u8 = 1;
const PLAYER_SAVE_LEN: usize = 22; // 1 + 5*4 + 1

impl PlayerSave {
    const PATH: &'static str = "saves/player.dat";

    /// Load a saved player state, returning `None` if no save file exists or is invalid.
    pub fn load() -> Option<Self> {
        let data = fs::read(Self::PATH).ok()?;
        if data.len() < PLAYER_SAVE_LEN || data[0] != PLAYER_FILE_VERSION {
            return None;
        }
        let x     = f32::from_le_bytes([data[1],  data[2],  data[3],  data[4]]);
        let y     = f32::from_le_bytes([data[5],  data[6],  data[7],  data[8]]);
        let z     = f32::from_le_bytes([data[9],  data[10], data[11], data[12]]);
        let yaw   = f32::from_le_bytes([data[13], data[14], data[15], data[16]]);
        let pitch = f32::from_le_bytes([data[17], data[18], data[19], data[20]]);
        let flags = data[21];
        log::info!("Loaded player save from {}", Self::PATH);
        Some(Self {
            x, y, z, yaw, pitch,
            show_chunk_outlines: flags & 0x01 != 0,
            noclip_mode:         flags & 0x02 != 0,
            show_enemy_hitboxes: flags & 0x04 != 0,
        })
    }

    /// Persist the player state to disk.
    pub fn save(&self) {
        let _ = fs::create_dir_all("saves");
        let mut data = Vec::with_capacity(PLAYER_SAVE_LEN);
        data.push(PLAYER_FILE_VERSION);
        data.extend_from_slice(&self.x.to_le_bytes());
        data.extend_from_slice(&self.y.to_le_bytes());
        data.extend_from_slice(&self.z.to_le_bytes());
        data.extend_from_slice(&self.yaw.to_le_bytes());
        data.extend_from_slice(&self.pitch.to_le_bytes());
        let mut flags = 0u8;
        if self.show_chunk_outlines { flags |= 0x01; }
        if self.noclip_mode         { flags |= 0x02; }
        if self.show_enemy_hitboxes { flags |= 0x04; }
        data.push(flags);
        if let Err(e) = fs::write(Self::PATH, data) {
            log::warn!("Failed to write player save: {}", e);
        } else {
            log::info!("Saved player state to {}", Self::PATH);
        }
    }
}

//* —— Fog uniform data sent to GPU —————————————————————————————————————————————————————————————————————
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FogUniform {
    pub start: f32,
    pub end: f32,
    pub enabled: f32,       // 1.0 = enabled, 0.0 = disabled
    pub use_square_fog: f32, // 1.0 = square fog (Chebyshev), 0.0 = circular fog (Euclidean)
}

impl From<FogConfig> for FogUniform {
    fn from(config: FogConfig) -> Self {
        Self {
            start: config.start,
            end: config.end,
            enabled: if config.enabled { 1.0 } else { 0.0 },
            use_square_fog: if config.use_square_fog { 1.0 } else { 0.0 },
        }
    }
}

impl Default for CloudConfig {
    fn default() -> Self {
        Self {
            height: 150.0,
            pixel_size: 8.0,
            threshold: 0.65,
            noise_scale: 0.005,
            noise_offset_change_speed: 5.0, // world units per second X drift
        }
    }
}

impl CloudConfig {
    /// Load configuration from a TOML file, or create default if it doesn't exist
    pub fn load_or_create(path: &Path) -> Self {
        match fs::read_to_string(path) {
            Ok(contents) => {
                match toml::from_str(&contents) {
                    Ok(config) => {
                        log::info!("Loaded cloud config from {}", path.display());
                        config
                    }
                    Err(e) => {
                        log::warn!("Failed to parse cloud config: {}, using defaults", e);
                        let default = Self::default();
                        let _ = default.save(path);
                        default
                    }
                }
            }
            Err(_) => {
                log::info!("No cloud config found, creating default at {}", path.display());
                let default = Self::default();
                let _ = default.save(path);
                default
            }
        }
    }

    /// Save configuration to a TOML file
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let toml_string = toml::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        fs::write(path, toml_string)?;
        log::info!("Saved cloud config to {}", path.display());
        Ok(())
    }
}
