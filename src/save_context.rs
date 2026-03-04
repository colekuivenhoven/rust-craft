//! Tracks the currently-active world's save directory.
//! Call `set_world` once before starting the game; all other save helpers
//! use the stored value automatically.

use std::sync::RwLock;

/// The active world save root, e.g. `"saves/myworld"`.
/// Empty until `set_world` is called.
static WORLD_SAVE_DIR: RwLock<String> = RwLock::new(String::new());

/// Activate a world by name, creating its save directories immediately.
pub fn set_world(name: &str) {
    *WORLD_SAVE_DIR.write().unwrap() = format!("saves/{}", name);
    let _ = std::fs::create_dir_all(chunks_dir());
}

/// Returns the world root, e.g. `"saves/myworld"`.
pub fn world_dir() -> String {
    WORLD_SAVE_DIR.read().unwrap().clone()
}

/// Returns the chunk storage directory, e.g. `"saves/myworld/chunks"`.
pub fn chunks_dir() -> String {
    format!("{}/chunks", world_dir())
}

/// Returns the player save file path.
pub fn player_path() -> String {
    format!("{}/player.dat", world_dir())
}

/// Returns the enemy save file path.
pub fn enemies_path() -> String {
    format!("{}/enemies.dat", world_dir())
}

/// Returns the per-world config file path (holds `master_seed` etc.).
pub fn world_config_path() -> String {
    format!("{}/world.toml", world_dir())
}

/// Delete a world's save directory permanently.
pub fn delete_world(name: &str) {
    let path = std::path::Path::new("saves").join(name);
    let _ = std::fs::remove_dir_all(path);
}

/// Return a list of available world names by scanning the `saves/` directory.
pub fn list_worlds() -> Vec<String> {
    let saves_dir = std::path::Path::new("saves");
    if !saves_dir.exists() {
        return vec![];
    }
    let mut worlds: Vec<String> = std::fs::read_dir(saves_dir)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            // Must be a directory that contains a world.toml
            if path.is_dir() && path.join("world.toml").exists() {
                path.file_name()?.to_str().map(|s| s.to_string())
            } else {
                None
            }
        })
        .collect();
    worlds.sort();
    worlds
}
