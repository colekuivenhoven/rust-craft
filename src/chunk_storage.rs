use crate::block::BlockType;
use crate::chunk::{Chunk, CHUNK_SIZE, CHUNK_HEIGHT};
use std::fs::{self, File};
use std::io::{Read, Write, BufReader, BufWriter};
use std::path::PathBuf;

const SAVE_DIR: &str = "saves/chunks";
const CHUNK_FILE_VERSION: u8 = 1;

/// Returns the file path for a chunk's save file
fn get_chunk_path(chunk_x: i32, chunk_z: i32) -> PathBuf {
    PathBuf::from(SAVE_DIR).join(format!("{}_{}.chunk", chunk_x, chunk_z))
}

/// Ensures the save directory exists
fn ensure_save_dir() -> std::io::Result<()> {
    fs::create_dir_all(SAVE_DIR)
}

/// Saves a chunk to disk
/// File format:
/// - 1 byte: version
/// - 4 bytes: chunk_x (i32 little endian)
/// - 4 bytes: chunk_z (i32 little endian)
/// - CHUNK_SIZE * CHUNK_HEIGHT * CHUNK_SIZE bytes: block IDs
pub fn save_chunk(chunk: &Chunk) -> std::io::Result<()> {
    ensure_save_dir()?;

    let path = get_chunk_path(chunk.position.0, chunk.position.1);
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write header
    writer.write_all(&[CHUNK_FILE_VERSION])?;
    writer.write_all(&chunk.position.0.to_le_bytes())?;
    writer.write_all(&chunk.position.1.to_le_bytes())?;

    // Write block data
    for x in 0..CHUNK_SIZE {
        for y in 0..CHUNK_HEIGHT {
            for z in 0..CHUNK_SIZE {
                writer.write_all(&[chunk.blocks[x][y][z].get_id()])?;
            }
        }
    }

    writer.flush()?;
    Ok(())
}

/// Attempts to load a chunk from disk
/// Returns Some(Chunk) if the chunk was loaded successfully, None if no save exists
pub fn load_chunk(chunk_x: i32, chunk_z: i32) -> Option<Chunk> {
    let path = get_chunk_path(chunk_x, chunk_z);

    if !path.exists() {
        return None;
    }

    let file = match File::open(&path) {
        Ok(f) => f,
        Err(_) => return None,
    };

    let mut reader = BufReader::new(file);

    // Read and verify header
    let mut version = [0u8; 1];
    if reader.read_exact(&mut version).is_err() {
        return None;
    }
    if version[0] != CHUNK_FILE_VERSION {
        // Version mismatch - regenerate chunk
        return None;
    }

    let mut pos_x_bytes = [0u8; 4];
    let mut pos_z_bytes = [0u8; 4];
    if reader.read_exact(&mut pos_x_bytes).is_err() || reader.read_exact(&mut pos_z_bytes).is_err() {
        return None;
    }

    let saved_x = i32::from_le_bytes(pos_x_bytes);
    let saved_z = i32::from_le_bytes(pos_z_bytes);

    // Verify position matches
    if saved_x != chunk_x || saved_z != chunk_z {
        return None;
    }

    // Read block data
    let mut blocks: Box<[[[BlockType; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]> =
        vec![[[BlockType::Air; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]
            .into_boxed_slice()
            .try_into()
            .unwrap();

    let mut block_id = [0u8; 1];
    for x in 0..CHUNK_SIZE {
        for y in 0..CHUNK_HEIGHT {
            for z in 0..CHUNK_SIZE {
                if reader.read_exact(&mut block_id).is_err() {
                    return None;
                }
                blocks[x][y][z] = BlockType::from_id(block_id[0]);
            }
        }
    }

    Some(Chunk::from_saved_data(chunk_x, chunk_z, blocks))
}

/// Checks if a saved chunk exists for the given coordinates
pub fn has_saved_chunk(chunk_x: i32, chunk_z: i32) -> bool {
    get_chunk_path(chunk_x, chunk_z).exists()
}

/// Deletes a saved chunk file (useful for resetting world)
#[allow(dead_code)]
pub fn delete_saved_chunk(chunk_x: i32, chunk_z: i32) -> std::io::Result<()> {
    let path = get_chunk_path(chunk_x, chunk_z);
    if path.exists() {
        fs::remove_file(path)?;
    }
    Ok(())
}
