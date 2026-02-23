use crate::block::BlockType;
use crate::chunk::{Chunk, CHUNK_SIZE, CHUNK_HEIGHT};
use crate::chunk_storage;
use crate::lighting;
use std::collections::HashSet;
use std::sync::mpsc::{channel, Sender, Receiver, TryRecvError};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// Request to load or generate a chunk
pub struct ChunkLoadRequest {
    pub position: (i32, i32),
    pub priority: f32, // Lower = higher priority (distance squared)
}

/// Result of a chunk load/generate operation
pub struct ChunkLoadResult {
    pub position: (i32, i32),
    pub chunk: Chunk,
}

/// Request to save a chunk
pub struct ChunkSaveRequest {
    pub position: (i32, i32),
    pub blocks: Box<[[[BlockType; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]>,
    pub modified: bool,
}

/// Background chunk loader that handles generation and I/O off the main thread
pub struct ChunkLoader {
    // Channel for sending load requests to the worker
    load_request_tx: Sender<ChunkLoadRequest>,
    // Channel for receiving completed chunks
    load_result_rx: Receiver<ChunkLoadResult>,
    // Channel for sending save requests
    save_request_tx: Option<Sender<ChunkSaveRequest>>,
    // Worker thread handles (Option so we can take them for joining)
    _load_thread: Option<JoinHandle<()>>,
    save_thread: Option<JoinHandle<()>>,
    // Track pending requests to avoid duplicates
    pending_loads: HashSet<(i32, i32)>,
    // Track pending saves for shutdown
    pending_saves: Arc<AtomicUsize>,
    // Maximum chunks to receive per frame
    pub max_chunks_per_frame: usize,
}

impl ChunkLoader {
    pub fn new(master_seed: u32, terrain_cfg: Arc<crate::config::TerrainConfig>) -> Self {
        // Create channels for load requests/results
        let (load_request_tx, load_request_rx) = channel::<ChunkLoadRequest>();
        let (load_result_tx, load_result_rx) = channel::<ChunkLoadResult>();

        // Create channel for save requests
        let (save_request_tx, save_request_rx) = channel::<ChunkSaveRequest>();

        // Track pending saves
        let pending_saves = Arc::new(AtomicUsize::new(0));
        let pending_saves_clone = pending_saves.clone();

        // Spawn load worker thread
        let load_thread = thread::Builder::new()
            .name("chunk-loader".to_string())
            .stack_size(8 * 1024 * 1024) // 8MB stack for chunk generation
            .spawn(move || {
                Self::load_worker(load_request_rx, load_result_tx, master_seed, terrain_cfg);
            })
            .expect("Failed to spawn chunk loader thread");

        // Spawn save worker thread
        let save_thread = thread::Builder::new()
            .name("chunk-saver".to_string())
            .stack_size(4 * 1024 * 1024) // 4MB stack
            .spawn(move || {
                Self::save_worker(save_request_rx, pending_saves_clone);
            })
            .expect("Failed to spawn chunk saver thread");

        Self {
            load_request_tx,
            load_result_rx,
            save_request_tx: Some(save_request_tx),
            _load_thread: Some(load_thread),
            save_thread: Some(save_thread),
            pending_loads: HashSet::new(),
            pending_saves,
            max_chunks_per_frame: 1, // Process 1 chunk per frame to minimize stuttering
        }
    }

    /// Worker function that runs in the load thread
    fn load_worker(rx: Receiver<ChunkLoadRequest>, tx: Sender<ChunkLoadResult>, master_seed: u32, terrain_cfg: Arc<crate::config::TerrainConfig>) {
        // Collect requests and sort by priority
        let mut requests: Vec<ChunkLoadRequest> = Vec::new();

        loop {
            // First, drain all pending requests
            loop {
                match rx.try_recv() {
                    Ok(request) => requests.push(request),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => return,
                }
            }

            // If no requests, block waiting for one
            if requests.is_empty() {
                match rx.recv() {
                    Ok(request) => requests.push(request),
                    Err(_) => return, // Channel closed
                }
                continue;
            }

            // Sort by priority (lowest first = closest to player)
            requests.sort_by(|a, b| a.priority.partial_cmp(&b.priority).unwrap());

            // Process the highest priority request
            let request = requests.remove(0);
            let (cx, cz) = request.position;

            // Try to load from disk first, otherwise generate
            let mut chunk = if let Some(loaded) = chunk_storage::load_chunk(cx, cz) {
                loaded
            } else {
                Chunk::new(cx, cz, master_seed, &terrain_cfg)
            };

            // Calculate lighting and mark as done so main thread doesn't recalculate
            lighting::calculate_chunk_lighting(&mut chunk);
            chunk.light_dirty = false;

            // Send result back
            if tx.send(ChunkLoadResult {
                position: (cx, cz),
                chunk,
            }).is_err() {
                return; // Main thread dropped receiver
            }
        }
    }

    /// Worker function that runs in the save thread
    fn save_worker(rx: Receiver<ChunkSaveRequest>, pending_saves: Arc<AtomicUsize>) {
        loop {
            match rx.recv() {
                Ok(request) => {
                    if request.modified {
                        // Create a minimal chunk just for saving
                        let chunk = Chunk::from_saved_data(
                            request.position.0,
                            request.position.1,
                            request.blocks,
                        );
                        if let Err(e) = chunk_storage::save_chunk(&chunk) {
                            eprintln!("Background save failed for {:?}: {}", request.position, e);
                        }
                    }
                    // Decrement pending saves counter
                    pending_saves.fetch_sub(1, Ordering::SeqCst);
                }
                Err(_) => return, // Channel closed
            }
        }
    }

    /// Request a chunk to be loaded/generated
    pub fn request_chunk(&mut self, position: (i32, i32), priority: f32) {
        if self.pending_loads.contains(&position) {
            return; // Already requested
        }

        self.pending_loads.insert(position);
        let _ = self.load_request_tx.send(ChunkLoadRequest { position, priority });
    }

    /// Queue a chunk to be saved in the background
    pub fn queue_save(&self, position: (i32, i32), blocks: Box<[[[BlockType; CHUNK_SIZE]; CHUNK_HEIGHT]; CHUNK_SIZE]>, modified: bool) {
        if let Some(ref tx) = self.save_request_tx {
            self.pending_saves.fetch_add(1, Ordering::SeqCst);
            if tx.send(ChunkSaveRequest {
                position,
                blocks,
                modified,
            }).is_err() {
                // Channel closed, decrement counter
                self.pending_saves.fetch_sub(1, Ordering::SeqCst);
            }
        }
    }

    /// Receive completed chunks (call once per frame)
    /// Returns up to max_chunks_per_frame chunks
    pub fn receive_chunks(&mut self) -> Vec<ChunkLoadResult> {
        let mut results = Vec::new();

        for _ in 0..self.max_chunks_per_frame {
            match self.load_result_rx.try_recv() {
                Ok(result) => {
                    self.pending_loads.remove(&result.position);
                    results.push(result);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }

        results
    }

    /// Check if a chunk load is pending
    pub fn is_pending(&self, position: &(i32, i32)) -> bool {
        self.pending_loads.contains(position)
    }

    /// Get number of pending loads
    pub fn pending_count(&self) -> usize {
        self.pending_loads.len()
    }

    /// Shutdown the loader, waiting for all pending saves to complete
    pub fn shutdown(&mut self) {
        // Drop the save sender to signal the worker to exit after processing remaining requests
        self.save_request_tx = None;

        // Wait for pending saves with a timeout
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(5);
        while self.pending_saves.load(Ordering::SeqCst) > 0 {
            if start.elapsed() > timeout {
                eprintln!("Warning: Timed out waiting for {} pending chunk saves",
                    self.pending_saves.load(Ordering::SeqCst));
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Join the save thread
        if let Some(handle) = self.save_thread.take() {
            let _ = handle.join();
        }
    }
}
