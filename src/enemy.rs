use crate::block::{Vertex, LineVertex};
use crate::chunk::CHUNK_SIZE;
use crate::texture::TEX_NONE;
use crate::world::World;
use cgmath::{Point3, Vector3, InnerSpace};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write, BufReader, BufWriter};

const GRAVITY: f32 = -20.0;
const BOUNCE_UP_MIN: f32 = 8.0;
const BOUNCE_UP_MAX: f32 = 10.0;
const WANDER_SPEED: f32 = 3.0;
const CHASE_SPEED: f32 = 5.0;
const DETECTION_RANGE: f32 = 16.0;
const SPAWN_MIN_DIST: f32 = 20.0;
const SPAWN_MAX_DIST: f32 = 40.0;
const SLIME_SIZE: f32 = 0.8;
const SLIME_DAMAGE: f32 = 8.0;
const SAVE_PATH: &str = "saves/enemies.dat";
const ENEMY_FILE_VERSION: u8 = 1;

// ── Enemy types ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum EnemyKind {
    SlimeCube {
        squash: f32,
        color: [f32; 3],
    },
}

// ── Enemy ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Enemy {
    pub position: Point3<f32>,
    pub velocity: Vector3<f32>,
    pub size: f32,
    pub health: f32,
    pub max_health: f32,
    pub attack_damage: f32,
    pub alive: bool,
    pub on_ground: bool,
    pub kind: EnemyKind,
    pub facing_yaw: f32,
    bounce_timer: f32,
    was_on_ground: bool,
    rng_state: u32,
    last_land_x: f32,
    last_land_z: f32,
    stuck_count: u8,
}

impl Enemy {
    pub fn new_slime(position: Point3<f32>, seed: u32) -> Self {
        let r = 0.2 + (simple_hash(seed) % 100) as f32 * 0.002;
        let g = 0.75 + (simple_hash(seed.wrapping_add(1)) % 100) as f32 * 0.002;
        let b = 0.1 + (simple_hash(seed.wrapping_add(2)) % 100) as f32 * 0.002;

        Self {
            position,
            velocity: Vector3::new(0.0, 0.0, 0.0),
            size: SLIME_SIZE,
            health: 20.0,
            max_health: 20.0,
            attack_damage: SLIME_DAMAGE,
            alive: true,
            on_ground: false,
            kind: EnemyKind::SlimeCube {
                squash: 0.0,
                color: [r, g, b],
            },
            facing_yaw: 0.0,
            bounce_timer: 0.0,
            was_on_ground: false,
            rng_state: seed,
            last_land_x: position.x,
            last_land_z: position.z,
            stuck_count: 0,
        }
    }

    /// Returns the chunk position this enemy is in
    pub fn chunk_pos(&self) -> (i32, i32) {
        (
            (self.position.x / CHUNK_SIZE as f32).floor() as i32,
            (self.position.z / CHUNK_SIZE as f32).floor() as i32,
        )
    }

    pub fn update(&mut self, dt: f32, player_pos: Point3<f32>, world: &World) {
        if !self.alive {
            return;
        }

        let to_player = player_pos - self.position;
        let dist_to_player = to_player.magnitude();

        // Face toward player when chasing
        if dist_to_player < DETECTION_RANGE {
            self.facing_yaw = to_player.z.atan2(to_player.x);
        }

        // Bounce AI: only decide when on ground
        self.bounce_timer -= dt;
        if self.on_ground && self.bounce_timer <= 0.0 {
            self.bounce(dist_to_player, to_player);
        }

        // Gravity
        self.velocity.y += GRAVITY * dt;
        // Clamp terminal velocity
        if self.velocity.y < -30.0 {
            self.velocity.y = -30.0;
        }

        // Apply velocity
        let new_pos = self.position + self.velocity * dt;

        // Collision detection
        self.apply_collisions(new_pos, world);

        // Track repeated landings at the same position (stuck detection)
        if self.on_ground && !self.was_on_ground {
            let dx = (self.position.x - self.last_land_x).abs();
            let dz = (self.position.z - self.last_land_z).abs();
            if dx < 0.5 && dz < 0.5 {
                self.stuck_count = self.stuck_count.saturating_add(1);
            } else {
                self.stuck_count = 0;
            }
            self.last_land_x = self.position.x;
            self.last_land_z = self.position.z;

            // After 3 consecutive bounces landing in the same spot, escape
            if self.stuck_count >= 3 {
                let radius = self.size * 0.4;
                let height = self.size;
                if self.is_stuck(radius, height, world) {
                    if let Some(clear) = self.find_nearest_clear(world) {
                        self.position = clear;
                        self.velocity = Vector3::new(0.0, 0.0, 0.0);
                        self.on_ground = false;
                    }
                }
                self.stuck_count = 0;
            }
        }

        // Squash/stretch animation
        self.update_squash(dt);
    }

    fn bounce(&mut self, dist_to_player: f32, to_player: Vector3<f32>) {
        let jump_vel = pseudo_rand_range(&mut self.rng_state, BOUNCE_UP_MIN, BOUNCE_UP_MAX);
        self.velocity.y = jump_vel;

        if dist_to_player < DETECTION_RANGE {
            let dir = Vector3::new(to_player.x, 0.0, to_player.z);
            if dir.magnitude() > 0.1 {
                let dir = dir.normalize();
                let speed = pseudo_rand_range(&mut self.rng_state, CHASE_SPEED * 0.8, CHASE_SPEED);
                self.velocity.x = dir.x * speed;
                self.velocity.z = dir.z * speed;
            }
            self.bounce_timer = pseudo_rand_range(&mut self.rng_state, 0.3, 0.8);
        } else {
            let angle = pseudo_rand_range(&mut self.rng_state, 0.0, std::f32::consts::TAU);
            let speed = pseudo_rand_range(&mut self.rng_state, WANDER_SPEED * 0.5, WANDER_SPEED);
            self.velocity.x = angle.cos() * speed;
            self.velocity.z = angle.sin() * speed;
            self.facing_yaw = angle;
            self.bounce_timer = pseudo_rand_range(&mut self.rng_state, 1.0, 3.0);
        }
    }

    fn apply_collisions(&mut self, new_pos: Point3<f32>, world: &World) {
        let radius = self.size * 0.4;
        let height = self.size;

        // --- X axis collision ---
        if self.velocity.x != 0.0 {
            let test_x = Point3::new(new_pos.x, self.position.y, self.position.z);
            if self.check_horizontal_collision(test_x, radius, height, world) {
                self.velocity.x = 0.0;
            } else {
                self.position.x = new_pos.x;
            }
        }

        // --- Z axis collision (using updated X) ---
        if self.velocity.z != 0.0 {
            let test_z = Point3::new(self.position.x, self.position.y, new_pos.z);
            if self.check_horizontal_collision(test_z, radius, height, world) {
                self.velocity.z = 0.0;
            } else {
                self.position.z = new_pos.z;
            }
        }

        // --- Y axis: ground and ceiling ---
        self.was_on_ground = self.on_ground;
        self.on_ground = false;

        // Build check positions using the CORRECTED X/Z position
        let check_positions = [
            (self.position.x, self.position.z),
            (self.position.x + radius * 0.7, self.position.z),
            (self.position.x - radius * 0.7, self.position.z),
            (self.position.x, self.position.z + radius * 0.7),
            (self.position.x, self.position.z - radius * 0.7),
        ];

        let old_feet_y = self.position.y;

        if self.velocity.y <= 0.0 {
            // Falling: check ground
            let feet_y = new_pos.y;
            let mut highest_ground = f32::NEG_INFINITY;
            let mut found_ground = false;

            for (cx, cz) in &check_positions {
                let block_y = feet_y.floor() as i32;
                let bx = cx.floor() as i32;
                let bz = cz.floor() as i32;
                let block = world.get_block_world(bx, block_y, bz);
                if block.is_solid() {
                    let block_top = (block_y + 1) as f32;
                    // Only land if we were above this block before
                    if feet_y < block_top && old_feet_y >= block_top - 0.01 {
                        highest_ground = highest_ground.max(block_top);
                        found_ground = true;
                    }
                }
            }

            if found_ground {
                self.position.y = highest_ground;
                self.velocity.y = 0.0;
                self.on_ground = true;
                // Apply friction when landing
                self.velocity.x *= 0.5;
                self.velocity.z *= 0.5;
            } else {
                self.position.y = new_pos.y;

                // Standing hysteresis check - prevent gliding off edges
                let standing_feet_y = self.position.y - 0.05;
                for (cx, cz) in &check_positions {
                    let bx = cx.floor() as i32;
                    let bz = cz.floor() as i32;
                    let block_y = standing_feet_y.floor() as i32;
                    let block = world.get_block_world(bx, block_y, bz);
                    if block.is_solid() {
                        let block_top = (block_y + 1) as f32;
                        if (self.position.y - block_top).abs() < 0.05 {
                            self.on_ground = true;
                            self.position.y = block_top;
                            break;
                        }
                    }
                }
            }
        } else {
            // Rising: check ceiling at multiple points
            let mut hit_ceiling = false;
            for (cx, cz) in &check_positions {
                let head_y = (new_pos.y + height).floor() as i32;
                let bx = cx.floor() as i32;
                let bz = cz.floor() as i32;
                if world.get_block_world(bx, head_y, bz).is_solid() {
                    hit_ceiling = true;
                    break;
                }
            }
            if hit_ceiling {
                self.velocity.y = 0.0;
            } else {
                self.position.y = new_pos.y;
            }
        }

        // Prevent falling into void
        if self.position.y < 0.0 {
            self.position.y = 0.0;
            self.velocity.y = 0.0;
            self.on_ground = true;
        }

        // Anti-stuck: if center of enemy is embedded in a solid block, escape immediately
        let center_bx = self.position.x.floor() as i32;
        let center_by = (self.position.y + height * 0.5).floor() as i32;
        let center_bz = self.position.z.floor() as i32;
        if world.get_block_world(center_bx, center_by, center_bz).is_solid() {
            if let Some(clear) = self.find_nearest_clear(world) {
                self.position = clear;
                self.velocity = Vector3::new(0.0, 0.0, 0.0);
                self.on_ground = false;
                self.stuck_count = 0;
            }
        }
    }

    fn check_horizontal_collision(
        &self, pos: Point3<f32>, radius: f32, height: f32, world: &World,
    ) -> bool {
        // Check at 3 heights: feet, middle, head
        let heights = [0.1, height * 0.5, height - 0.1];
        // Check in all 4 cardinal directions at radius distance
        let offsets = [
            (radius, 0.0),
            (-radius, 0.0),
            (0.0, radius),
            (0.0, -radius),
        ];

        for &h in &heights {
            for &(ox, oz) in &offsets {
                let bx = (pos.x + ox).floor() as i32;
                let by = (pos.y + h).floor() as i32;
                let bz = (pos.z + oz).floor() as i32;
                if world.get_block_world(bx, by, bz).is_solid() {
                    return true;
                }
            }
        }
        false
    }

    /// Check if the enemy is blocked on at least 2 horizontal sides (boxed in)
    fn is_stuck(&self, radius: f32, _height: f32, world: &World) -> bool {
        let by = (self.position.y + 0.1).floor() as i32;
        let sides = [
            (radius + 0.1, 0.0),
            (-(radius + 0.1), 0.0),
            (0.0, radius + 0.1),
            (0.0, -(radius + 0.1)),
        ];
        let mut blocked = 0;
        for &(ox, oz) in &sides {
            let bx = (self.position.x + ox).floor() as i32;
            let bz = (self.position.z + oz).floor() as i32;
            if world.get_block_world(bx, by, bz).is_solid() {
                blocked += 1;
            }
        }
        blocked >= 2
    }

    /// Search in expanding rings for the nearest clear position
    /// (solid ground below, 2 air blocks above)
    fn find_nearest_clear(&self, world: &World) -> Option<Point3<f32>> {
        let base_x = self.position.x.floor() as i32;
        let base_y = self.position.y.floor() as i32;
        let base_z = self.position.z.floor() as i32;

        // Search expanding rings: radius 0..6, y offsets -3..+8
        for r in 0..7_i32 {
            for dx in -r..=r {
                for dz in -r..=r {
                    // Only check the ring perimeter (skip interior, already checked)
                    if r > 0 && dx.abs() != r && dz.abs() != r {
                        continue;
                    }
                    let x = base_x + dx;
                    let z = base_z + dz;
                    // Scan vertically for a valid landing spot
                    for dy in -3..=8_i32 {
                        let y = base_y + dy;
                        if y < 1 { continue; }
                        let ground_solid = world.get_block_world(x, y - 1, z).is_solid();
                        let feet_clear = !world.get_block_world(x, y, z).is_solid();
                        let head_clear = !world.get_block_world(x, y + 1, z).is_solid();
                        if ground_solid && feet_clear && head_clear {
                            return Some(Point3::new(x as f32 + 0.5, y as f32, z as f32 + 0.5));
                        }
                    }
                }
            }
        }
        None
    }

    fn update_squash(&mut self, dt: f32) {
        let EnemyKind::SlimeCube { ref mut squash, .. } = self.kind;
        if self.on_ground && !self.was_on_ground {
            *squash = -0.4;
        } else if !self.on_ground && self.velocity.y > 0.0 {
            let target = 0.3 * (self.velocity.y / 10.0).clamp(0.0, 1.0);
            *squash = *squash + (target - *squash) * dt * 12.0;
        }
        *squash += (0.0 - *squash) * dt * 8.0;
    }

    // ── Serialization ────────────────────────────────────────

    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        // Kind tag
        let kind_id: u8 = match &self.kind {
            EnemyKind::SlimeCube { .. } => 0,
        };
        w.write_all(&[kind_id])?;

        // Position
        w.write_all(&self.position.x.to_le_bytes())?;
        w.write_all(&self.position.y.to_le_bytes())?;
        w.write_all(&self.position.z.to_le_bytes())?;

        // Health
        w.write_all(&self.health.to_le_bytes())?;

        // Facing yaw
        w.write_all(&self.facing_yaw.to_le_bytes())?;

        // RNG state (for deterministic color reproduction)
        w.write_all(&self.rng_state.to_le_bytes())?;

        // Color (stored directly since it's derived from seed but we want exact match)
        let EnemyKind::SlimeCube { color, .. } = &self.kind;
        w.write_all(&color[0].to_le_bytes())?;
        w.write_all(&color[1].to_le_bytes())?;
        w.write_all(&color[2].to_le_bytes())?;

        Ok(())
    }

    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let mut buf1 = [0u8; 1];
        let mut buf4 = [0u8; 4];

        // Kind tag
        r.read_exact(&mut buf1)?;
        let kind_id = buf1[0];

        // Position
        r.read_exact(&mut buf4)?;
        let px = f32::from_le_bytes(buf4);
        r.read_exact(&mut buf4)?;
        let py = f32::from_le_bytes(buf4);
        r.read_exact(&mut buf4)?;
        let pz = f32::from_le_bytes(buf4);

        // Health
        r.read_exact(&mut buf4)?;
        let health = f32::from_le_bytes(buf4);

        // Facing yaw
        r.read_exact(&mut buf4)?;
        let facing_yaw = f32::from_le_bytes(buf4);

        // RNG state
        r.read_exact(&mut buf4)?;
        let rng_state = u32::from_le_bytes(buf4);

        match kind_id {
            0 => {
                // SlimeCube: read color
                r.read_exact(&mut buf4)?;
                let cr = f32::from_le_bytes(buf4);
                r.read_exact(&mut buf4)?;
                let cg = f32::from_le_bytes(buf4);
                r.read_exact(&mut buf4)?;
                let cb = f32::from_le_bytes(buf4);

                Ok(Self {
                    position: Point3::new(px, py, pz),
                    velocity: Vector3::new(0.0, 0.0, 0.0),
                    size: SLIME_SIZE,
                    health,
                    max_health: 20.0,
                    attack_damage: SLIME_DAMAGE,
                    alive: true,
                    on_ground: false,
                    kind: EnemyKind::SlimeCube {
                        squash: 0.0,
                        color: [cr, cg, cb],
                    },
                    facing_yaw,
                    bounce_timer: 0.0,
                    was_on_ground: false,
                    rng_state,
                    last_land_x: px,
                    last_land_z: pz,
                    stuck_count: 0,
                })
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Unknown enemy kind",
            )),
        }
    }
}

// ── EnemyManager ─────────────────────────────────────────────

pub struct EnemyManager {
    pub enemies: Vec<Enemy>,
    /// Enemies stored by chunk position when their chunk is unloaded
    shelved: HashMap<(i32, i32), Vec<Enemy>>,
    spawn_timer: f32,
    spawn_interval: f32,
    max_enemies: usize,
    next_seed: u32,
}

impl EnemyManager {
    pub fn new(spawn_interval: f32, max_enemies: usize) -> Self {
        let mut mgr = Self {
            enemies: Vec::new(),
            shelved: HashMap::new(),
            spawn_timer: 0.0,
            spawn_interval,
            max_enemies,
            next_seed: 42,
        };
        mgr.load_from_disk();
        mgr
    }

    pub fn update(&mut self, dt: f32, player_pos: Point3<f32>, world: &World) {
        // Shelve enemies whose chunks are no longer loaded
        let mut to_shelve = Vec::new();
        for (i, enemy) in self.enemies.iter().enumerate() {
            let cp = enemy.chunk_pos();
            if !world.chunks.contains_key(&cp) {
                to_shelve.push(i);
            }
        }
        // Remove from back to front to preserve indices
        for &i in to_shelve.iter().rev() {
            let enemy = self.enemies.remove(i);
            let cp = enemy.chunk_pos();
            self.shelved.entry(cp).or_default().push(enemy);
        }

        // Unshelve enemies whose chunks are now loaded
        let loaded_chunks: Vec<(i32, i32)> = self.shelved.keys()
            .filter(|cp| world.chunks.contains_key(cp))
            .copied()
            .collect();
        for cp in loaded_chunks {
            if let Some(enemies) = self.shelved.remove(&cp) {
                self.enemies.extend(enemies);
            }
        }

        // Update active enemies
        for enemy in &mut self.enemies {
            enemy.update(dt, player_pos, world);
        }

        // Remove dead enemies
        self.enemies.retain(|e| e.alive);

        // Spawn new enemies (only count active, not shelved)
        self.spawn_timer += dt;
        if self.spawn_timer >= self.spawn_interval && self.enemies.len() < self.max_enemies {
            self.try_spawn(player_pos, world);
            self.spawn_timer = 0.0;
        }
    }

    fn try_spawn(&mut self, player_pos: Point3<f32>, world: &World) {
        let seed = self.next_seed;
        self.next_seed = self.next_seed.wrapping_add(73856093);

        let angle = pseudo_rand_range(&mut self.next_seed, 0.0, std::f32::consts::TAU);
        let distance = pseudo_rand_range(&mut self.next_seed, SPAWN_MIN_DIST, SPAWN_MAX_DIST);

        let spawn_x = player_pos.x + angle.cos() * distance;
        let spawn_z = player_pos.z + angle.sin() * distance;

        // Find ground: scan downward
        let start_y = player_pos.y as i32 + 20;
        let mut spawn_y = None;
        for y in (1..start_y).rev() {
            let bx = spawn_x.floor() as i32;
            let bz = spawn_z.floor() as i32;
            if world.get_block_world(bx, y, bz).is_solid()
                && !world.get_block_world(bx, y + 1, bz).is_solid()
                && !world.get_block_world(bx, y + 2, bz).is_solid()
            {
                spawn_y = Some((y + 1) as f32);
                break;
            }
        }

        if let Some(y) = spawn_y {
            let pos = Point3::new(spawn_x, y, spawn_z);
            self.enemies.push(Enemy::new_slime(pos, seed));
        }
    }

    pub fn check_player_damage(&self, player_pos: Point3<f32>) -> f32 {
        let mut total_damage = 0.0;
        for enemy in &self.enemies {
            if !enemy.alive { continue; }
            let dist = (enemy.position - player_pos).magnitude();
            let touch_dist = enemy.size * 0.5 + 0.25;
            if dist < touch_dist {
                total_damage += enemy.attack_damage;
            }
        }
        total_damage
    }

    // ── Persistence ──────────────────────────────────────────

    /// Save all enemies (active + shelved) to disk
    pub fn save_to_disk(&self) {
        let _ = fs::create_dir_all("saves");

        let file = match File::create(SAVE_PATH) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Failed to save enemies: {}", e);
                return;
            }
        };
        let mut w = BufWriter::new(file);

        // Collect all enemies: active + all shelved
        let all_enemies: Vec<&Enemy> = self.enemies.iter()
            .chain(self.shelved.values().flat_map(|v| v.iter()))
            .filter(|e| e.alive)
            .collect();

        let count = all_enemies.len() as u32;

        if w.write_all(&[ENEMY_FILE_VERSION]).is_err() { return; }
        if w.write_all(&count.to_le_bytes()).is_err() { return; }

        for enemy in all_enemies {
            if enemy.serialize(&mut w).is_err() {
                eprintln!("Failed to serialize enemy");
                return;
            }
        }

        let _ = w.flush();
    }

    /// Load enemies from disk into shelved storage
    fn load_from_disk(&mut self) {
        let file = match File::open(SAVE_PATH) {
            Ok(f) => f,
            Err(_) => return, // No save file, that's fine
        };
        let mut r = BufReader::new(file);

        let mut buf1 = [0u8; 1];
        let mut buf4 = [0u8; 4];

        // Version check
        if r.read_exact(&mut buf1).is_err() { return; }
        if buf1[0] != ENEMY_FILE_VERSION { return; }

        // Count
        if r.read_exact(&mut buf4).is_err() { return; }
        let count = u32::from_le_bytes(buf4) as usize;

        for _ in 0..count {
            match Enemy::deserialize(&mut r) {
                Ok(enemy) => {
                    // Put into shelved by chunk position; they'll be unshelved
                    // when their chunk loads
                    let cp = enemy.chunk_pos();
                    self.shelved.entry(cp).or_default().push(enemy);
                }
                Err(_) => break,
            }
        }
    }
}

// ── Vertex generation ────────────────────────────────────────

pub fn create_enemy_vertices(enemy: &Enemy) -> Vec<Vertex> {
    match &enemy.kind {
        EnemyKind::SlimeCube { squash, color } => {
            create_slime_vertices(enemy, *squash, *color)
        }
    }
}

fn create_slime_vertices(enemy: &Enemy, squash: f32, color: [f32; 3]) -> Vec<Vertex> {
    let mut vertices = Vec::with_capacity(5 * 24);

    let size = enemy.size;
    let half_w = size * 0.5 * (1.0 - squash * 0.5);
    let half_h = size * 0.5 * (1.0 + squash);
    let half_d = size * 0.5 * (1.0 - squash * 0.5);
    let pos = enemy.position;
    let yaw = enemy.facing_yaw;

    // Body
    add_rotated_cube(
        &mut vertices,
        pos, Vector3::new(0.0, half_h, 0.0),
        half_w, half_h, half_d,
        color, yaw,
    );

    // Eyes
    let eye_w = size * 0.08;
    let eye_h = size * 0.10;
    let eye_d = size * 0.03;
    let eye_color = [0.05, 0.05, 0.05];

    let eye_fwd = half_w + 0.002;
    let eye_up = half_h * 0.3;
    let eye_spread = size * 0.3;

    let left_offset = rotate_yaw(eye_fwd, eye_up, eye_spread, yaw);
    add_rotated_cube(
        &mut vertices,
        pos, Vector3::new(left_offset[0], half_h + left_offset[1], left_offset[2]),
        eye_d, eye_h, eye_w,
        eye_color, yaw,
    );

    let right_offset = rotate_yaw(eye_fwd, eye_up, -eye_spread, yaw);
    add_rotated_cube(
        &mut vertices,
        pos, Vector3::new(right_offset[0], half_h + right_offset[1], right_offset[2]),
        eye_d, eye_h, eye_w,
        eye_color, yaw,
    );

    // Pupils
    let pupil_size = size * 0.04;
    let pupil_d = size * 0.01;
    let pupil_color = [0.95, 0.95, 0.95];
    let pupil_fwd = eye_fwd + 0.002;
    let pupil_up = eye_up + eye_h * 0.2;

    let left_pupil = rotate_yaw(pupil_fwd, pupil_up, eye_spread, yaw);
    add_rotated_cube(
        &mut vertices,
        pos, Vector3::new(left_pupil[0], half_h + left_pupil[1], left_pupil[2]),
        pupil_d, pupil_size, pupil_size,
        pupil_color, yaw,
    );

    let right_pupil = rotate_yaw(pupil_fwd, pupil_up, -eye_spread, yaw);
    add_rotated_cube(
        &mut vertices,
        pos, Vector3::new(right_pupil[0], half_h + right_pupil[1], right_pupil[2]),
        pupil_d, pupil_size, pupil_size,
        pupil_color, yaw,
    );

    vertices
}

pub fn generate_enemy_indices(num_cubes: usize) -> Vec<u16> {
    let mut indices = Vec::with_capacity(num_cubes * 36);
    for cube_idx in 0..num_cubes {
        let base = (cube_idx * 24) as u16;
        for face in 0..6u16 {
            let fb = base + face * 4;
            indices.push(fb);
            indices.push(fb + 1);
            indices.push(fb + 2);
            indices.push(fb + 2);
            indices.push(fb + 3);
            indices.push(fb);
        }
    }
    indices
}

/// Generate LineVertex wireframe for enemy collision AABBs (debug visualization)
pub fn create_enemy_collision_outlines(enemies: &[Enemy]) -> Vec<LineVertex> {
    let mut verts = Vec::new();
    for enemy in enemies {
        if !enemy.alive { continue; }
        let radius = enemy.size * 0.4;
        let height = enemy.size;
        let px = enemy.position.x;
        let py = enemy.position.y;
        let pz = enemy.position.z;

        let x0 = px - radius;
        let x1 = px + radius;
        let y0 = py;
        let y1 = py + height;
        let z0 = pz - radius;
        let z1 = pz + radius;

        // 4 vertical edges
        verts.extend_from_slice(&[
            LineVertex { position: [x0, y0, z0] }, LineVertex { position: [x0, y1, z0] },
            LineVertex { position: [x1, y0, z0] }, LineVertex { position: [x1, y1, z0] },
            LineVertex { position: [x0, y0, z1] }, LineVertex { position: [x0, y1, z1] },
            LineVertex { position: [x1, y0, z1] }, LineVertex { position: [x1, y1, z1] },
        ]);
        // 4 bottom edges
        verts.extend_from_slice(&[
            LineVertex { position: [x0, y0, z0] }, LineVertex { position: [x1, y0, z0] },
            LineVertex { position: [x1, y0, z0] }, LineVertex { position: [x1, y0, z1] },
            LineVertex { position: [x1, y0, z1] }, LineVertex { position: [x0, y0, z1] },
            LineVertex { position: [x0, y0, z1] }, LineVertex { position: [x0, y0, z0] },
        ]);
        // 4 top edges
        verts.extend_from_slice(&[
            LineVertex { position: [x0, y1, z0] }, LineVertex { position: [x1, y1, z0] },
            LineVertex { position: [x1, y1, z0] }, LineVertex { position: [x1, y1, z1] },
            LineVertex { position: [x1, y1, z1] }, LineVertex { position: [x0, y1, z1] },
            LineVertex { position: [x0, y1, z1] }, LineVertex { position: [x0, y1, z0] },
        ]);
    }
    verts
}

// ── Geometry helpers ─────────────────────────────────────────

fn rotate_yaw(x: f32, y: f32, z: f32, yaw: f32) -> [f32; 3] {
    let c = yaw.cos();
    let s = yaw.sin();
    [x * c - z * s, y, x * s + z * c]
}

fn add_rotated_cube(
    vertices: &mut Vec<Vertex>,
    center: Point3<f32>,
    offset: Vector3<f32>,
    half_w: f32,
    half_h: f32,
    half_d: f32,
    color: [f32; 3],
    yaw: f32,
) {
    let cx = center.x + offset.x;
    let cy = center.y + offset.y;
    let cz = center.z + offset.z;

    let local_corners = [
        (-half_w, -half_h, -half_d),
        ( half_w, -half_h, -half_d),
        ( half_w,  half_h, -half_d),
        (-half_w,  half_h, -half_d),
        (-half_w, -half_h,  half_d),
        ( half_w, -half_h,  half_d),
        ( half_w,  half_h,  half_d),
        (-half_w,  half_h,  half_d),
    ];

    let mut corners = Vec::with_capacity(8);
    for (lx, ly, lz) in local_corners {
        let r = rotate_yaw(lx, ly, lz, yaw);
        corners.push([cx + r[0], cy + r[1], cz + r[2]]);
    }

    push_cube_verts(vertices, corners, color);
}

fn push_cube_verts(vertices: &mut Vec<Vertex>, corners: Vec<[f32; 3]>, color: [f32; 3]) {
    let faces = [
        [4, 5, 6, 7],
        [1, 0, 3, 2],
        [7, 6, 2, 3],
        [0, 1, 5, 4],
        [5, 1, 2, 6],
        [0, 4, 7, 3],
    ];

    for face_indices in faces.iter() {
        let p0 = Vector3::new(corners[face_indices[0]][0], corners[face_indices[0]][1], corners[face_indices[0]][2]);
        let p1 = Vector3::new(corners[face_indices[1]][0], corners[face_indices[1]][1], corners[face_indices[1]][2]);
        let p2 = Vector3::new(corners[face_indices[2]][0], corners[face_indices[2]][1], corners[face_indices[2]][2]);

        let edge1 = p1 - p0;
        let edge2 = p2 - p0;
        let normal = edge1.cross(edge2).normalize();
        let normal_arr = [normal.x, normal.y, normal.z];

        for &idx in face_indices.iter() {
            vertices.push(Vertex {
                position: corners[idx],
                color,
                normal: normal_arr,
                light_level: 1.0,
                alpha: 1.0,
                uv: [0.0, 0.0],
                tex_index: TEX_NONE,
                ao: 1.0,
            });
        }
    }
}

// ── Deterministic RNG helpers ────────────────────────────────

fn simple_hash(mut x: u32) -> u32 {
    x = x.wrapping_mul(73856093);
    x ^= x >> 16;
    x = x.wrapping_mul(2654435761);
    x ^= x >> 13;
    x
}

fn pseudo_rand(state: &mut u32) -> f32 {
    *state = simple_hash(*state);
    (*state & 0xFFFF) as f32 / 65535.0
}

fn pseudo_rand_range(state: &mut u32, min: f32, max: f32) -> f32 {
    min + pseudo_rand(state) * (max - min)
}
