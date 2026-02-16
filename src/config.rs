/// Global configuration for the game
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

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
            noise_offset_change_speed: 0.01,
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
