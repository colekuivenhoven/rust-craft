use cgmath::{Point3, Vector3, InnerSpace};
use rand::Rng;
use crate::block::{BlockType, Vertex};
use crate::world::World;
use crate::texture::TEX_NONE;

// ============================================================================
// Configuration Constants
// ============================================================================

// Spawning
pub const BIRD_SPAWN_INTERVAL: f32 = 0.5;      // Seconds between spawn attempts
pub const BIRD_MAX_COUNT: usize = 150;          // Maximum birds in world
pub const BIRD_SPAWN_DISTANCE: f32 = 40.0;     // Distance from player to spawn
pub const BIRD_DESPAWN_DISTANCE: f32 = 80.0;   // Distance at which birds despawn

// Flocking (Boids)
pub const BOID_SEPARATION_RADIUS: f32 = 2.0;   // Avoid birds closer than this
pub const BOID_ALIGNMENT_RADIUS: f32 = 8.0;    // Align with birds within this radius
pub const BOID_COHESION_RADIUS: f32 = 10.0;    // Move toward center of birds within this
pub const BOID_SEPARATION_WEIGHT: f32 = 1.5;   // Weight for separation force
pub const BOID_ALIGNMENT_WEIGHT: f32 = 1.0;    // Weight for alignment force
pub const BOID_COHESION_WEIGHT: f32 = 1.0;     // Weight for cohesion force
pub const BOID_FLOCK_CHANCE: f32 = 0.7;        // Probability a bird will try to flock

// Flight
pub const BIRD_FLY_SPEED: f32 = 8.0;           // Base flight speed
pub const BIRD_TURN_RATE: f32 = 2.0;           // Radians per second turning
pub const BIRD_MIN_HEIGHT: f32 = 35.0;         // Minimum flight altitude
pub const BIRD_MAX_HEIGHT: f32 = 90.0;         // Maximum flight altitude
pub const BIRD_FLIGHT_RANDOMNESS: f32 = 0.3;   // Random direction change factor
pub const BIRD_HEIGHT_CHANGE_RATE: f32 = 3.0;  // Vertical movement speed

// Landing
pub const BIRD_LANDING_CHECK_INTERVAL: f32 = 3.0; // How often to check for landing spots
pub const BIRD_LANDING_CHANCE: f32 = 0.15;        // Probability to land when spot found
pub const BIRD_PERCH_DURATION_MIN: f32 = 5.0;     // Minimum time perched
pub const BIRD_PERCH_DURATION_MAX: f32 = 20.0;    // Maximum time perched
pub const BIRD_LANDING_SCAN_RADIUS: i32 = 15;     // Radius to search for trees
pub const BIRD_SOLID_LANDING_CHANCE: f32 = 0.1;   // Chance to land on non-leaves solid blocks

// Obstacle avoidance
pub const BIRD_LOOK_AHEAD_DISTANCE: f32 = 4.0;    // How far ahead to check for obstacles
pub const BIRD_AVOIDANCE_STRENGTH: f32 = 4.0;     // How strongly to steer away from obstacles

// Flight dynamics
pub const BIRD_MAX_PITCH: f32 = 0.5;              // Maximum pitch angle (radians)
pub const BIRD_MAX_ROLL: f32 = 0.6;               // Maximum roll angle when turning (radians)
pub const BIRD_ROTATION_SMOOTHING: f32 = 4.0;    // Smoothing factor for pitch/roll changes (higher = faster)

// Animation
pub const WING_FLAP_SPEED: f32 = 12.0;         // Wing flaps per second while flying
pub const WING_FLAP_AMPLITUDE: f32 = 0.4;      // Maximum wing rotation (radians)
pub const LEG_WALK_SPEED: f32 = 8.0;           // Leg movement cycles per second
pub const LEG_WALK_AMPLITUDE: f32 = 0.2;       // Maximum leg swing distance

// Bird size
pub const BIRD_BASE_SIZE: f32 = 1.0;           // Base size multiplier

// ============================================================================
// BirdState Enum
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BirdState {
    Flying,      // Normal flight, may be flocking
    Descending,  // Flying down toward a landing spot
    Perched,     // Sitting on a Leaves block
    Walking,     // Moving around on the perch
    TakingOff,   // Transitioning from perched to flying
}

// ============================================================================
// Bird Struct
// ============================================================================

#[derive(Debug, Clone)]
pub struct Bird {
    pub position: Point3<f32>,
    pub velocity: Vector3<f32>,
    pub direction: f32,              // Yaw angle in radians (heading)
    pub pitch: f32,                  // Pitch angle (up/down tilt)
    pub roll: f32,                   // Roll angle (banking when turning)
    pub target_direction: f32,       // Target yaw for smooth turning
    pub state: BirdState,
    pub state_timer: f32,            // Time remaining in current state
    pub flock_id: Option<usize>,     // ID of flock this bird belongs to
    pub target_perch: Option<Point3<i32>>,  // Block position of target landing spot
    pub wing_phase: f32,             // Current phase of wing flap animation (0.0 - TAU)
    pub leg_phase: f32,              // Current phase of leg walk animation (0.0 - TAU)
    pub speed_modifier: f32,         // 0.8 - 1.2 multiplier for this bird
    pub size_modifier: f32,          // 0.9 - 1.1 multiplier for visual size
    pub target_height: f32,          // Current target flying height
    pub random_timer: f32,           // Timer for random direction changes
}

impl Bird {
    pub fn new(position: Point3<f32>) -> Self {
        let mut rng = rand::thread_rng();
        let initial_dir = rng.gen_range(0.0..std::f32::consts::TAU);
        Self {
            position,
            velocity: Vector3::new(0.0, 0.0, 0.0),
            direction: initial_dir,
            pitch: 0.0,
            roll: 0.0,
            target_direction: initial_dir,
            state: BirdState::Flying,
            state_timer: rng.gen_range(2.0..5.0),
            flock_id: None,
            target_perch: None,
            wing_phase: rng.gen_range(0.0..std::f32::consts::TAU),
            leg_phase: 0.0,
            speed_modifier: rng.gen_range(0.8..1.2),
            size_modifier: rng.gen_range(0.9..1.1),
            target_height: rng.gen_range(BIRD_MIN_HEIGHT..BIRD_MAX_HEIGHT),
            random_timer: rng.gen_range(1.0..3.0),
        }
    }

    pub fn update(&mut self, dt: f32, world: &World, boid_force: Vector3<f32>) {
        match self.state {
            BirdState::Flying => self.update_flying(dt, world, boid_force),
            BirdState::Descending => self.update_descending(dt),
            BirdState::Perched => self.update_perched(dt),
            BirdState::Walking => self.update_walking(dt),
            BirdState::TakingOff => self.update_takeoff(dt),
        }
        self.update_animations(dt);
    }

    fn update_flying(&mut self, dt: f32, world: &World, boid_force: Vector3<f32>) {
        let mut rng = rand::thread_rng();

        // Random direction changes
        self.random_timer -= dt;
        if self.random_timer <= 0.0 {
            self.target_direction += rng.gen_range(-BIRD_FLIGHT_RANDOMNESS * 2.0..BIRD_FLIGHT_RANDOMNESS * 2.0);
            self.target_height = rng.gen_range(BIRD_MIN_HEIGHT..BIRD_MAX_HEIGHT);
            self.random_timer = rng.gen_range(2.0..5.0);
        }

        // Apply boid forces to target direction
        if boid_force.magnitude() > 0.01 {
            let target_yaw = boid_force.z.atan2(boid_force.x);
            let mut yaw_diff = target_yaw - self.target_direction;
            while yaw_diff > std::f32::consts::PI { yaw_diff -= std::f32::consts::TAU; }
            while yaw_diff < -std::f32::consts::PI { yaw_diff += std::f32::consts::TAU; }
            self.target_direction += yaw_diff * 0.5;
        }

        // Obstacle avoidance - look ahead in flight direction
        let avoidance = self.calculate_avoidance(world);
        if avoidance.magnitude() > 0.01 {
            // Steer away from obstacles
            let avoid_yaw = avoidance.z.atan2(avoidance.x);
            self.target_direction = avoid_yaw;
            // Also adjust target height based on vertical avoidance
            if avoidance.y > 0.1 {
                self.target_height = (self.position.y + 10.0).min(BIRD_MAX_HEIGHT);
            } else if avoidance.y < -0.1 {
                self.target_height = (self.position.y - 5.0).max(BIRD_MIN_HEIGHT);
            }
        }

        // Normalize target direction
        while self.target_direction > std::f32::consts::TAU { self.target_direction -= std::f32::consts::TAU; }
        while self.target_direction < 0.0 { self.target_direction += std::f32::consts::TAU; }

        // Smooth turning toward target direction
        let mut yaw_diff = self.target_direction - self.direction;
        while yaw_diff > std::f32::consts::PI { yaw_diff -= std::f32::consts::TAU; }
        while yaw_diff < -std::f32::consts::PI { yaw_diff += std::f32::consts::TAU; }

        // Smooth exponential interpolation for direction
        let turn_factor = 1.0 - (-BIRD_TURN_RATE * dt).exp();
        self.direction += yaw_diff * turn_factor;

        // Normalize direction
        while self.direction > std::f32::consts::TAU { self.direction -= std::f32::consts::TAU; }
        while self.direction < 0.0 { self.direction += std::f32::consts::TAU; }

        // Roll based on turning (bank into turns) - smooth exponential interpolation
        let target_roll = (yaw_diff * 1.5).clamp(-BIRD_MAX_ROLL, BIRD_MAX_ROLL);
        let roll_factor = 1.0 - (-BIRD_ROTATION_SMOOTHING * dt).exp();
        self.roll += (target_roll - self.roll) * roll_factor;

        // Height adjustment and pitch
        let height_diff = self.target_height - self.position.y;
        let vertical_speed = height_diff.clamp(-BIRD_HEIGHT_CHANGE_RATE, BIRD_HEIGHT_CHANGE_RATE);

        // Pitch based on vertical movement - smooth exponential interpolation
        // Positive pitch = nose up when going up
        let target_pitch = (vertical_speed / BIRD_HEIGHT_CHANGE_RATE * BIRD_MAX_PITCH).clamp(-BIRD_MAX_PITCH, BIRD_MAX_PITCH);
        let pitch_factor = 1.0 - (-BIRD_ROTATION_SMOOTHING * dt).exp();
        self.pitch += (target_pitch - self.pitch) * pitch_factor;

        // Clamp height targets
        if self.position.y < BIRD_MIN_HEIGHT {
            self.target_height = BIRD_MIN_HEIGHT + 5.0;
        } else if self.position.y > BIRD_MAX_HEIGHT {
            self.target_height = BIRD_MAX_HEIGHT - 5.0;
        }

        // Calculate velocity from direction and pitch
        let speed = BIRD_FLY_SPEED * self.speed_modifier;
        let cos_pitch = self.pitch.cos();
        self.velocity.x = self.direction.cos() * speed * cos_pitch;
        self.velocity.z = self.direction.sin() * speed * cos_pitch;
        self.velocity.y = vertical_speed;

        // Apply velocity
        self.position += self.velocity * dt;
    }

    fn calculate_avoidance(&self, world: &World) -> Vector3<f32> {
        let mut avoidance = Vector3::new(0.0, 0.0, 0.0);

        // Check multiple points ahead
        let forward_x = self.direction.cos();
        let forward_z = self.direction.sin();

        for dist in [1.5, 3.0, BIRD_LOOK_AHEAD_DISTANCE] {
            let check_x = (self.position.x + forward_x * dist) as i32;
            let check_z = (self.position.z + forward_z * dist) as i32;

            // Check at current height and slightly above/below
            for dy in [-1, 0, 1] {
                let check_y = (self.position.y + dy as f32) as i32;
                let block = world.get_block_world(check_x, check_y, check_z);

                if block.is_solid() {
                    // Found obstacle, steer away
                    let weight = BIRD_AVOIDANCE_STRENGTH / dist;

                    // Steer opposite to the obstacle direction
                    avoidance.x -= forward_x * weight;
                    avoidance.z -= forward_z * weight;

                    // Steer up or down based on where obstacle is
                    if dy <= 0 {
                        avoidance.y += weight; // Obstacle below/at level, go up
                    } else {
                        avoidance.y -= weight * 0.5; // Obstacle above, go down
                    }
                }
            }
        }

        // Also check sides for tight spaces
        let right_x = forward_z;
        let right_z = -forward_x;

        for side_mult in [-1.0_f32, 1.0_f32] {
            let check_x = (self.position.x + right_x * side_mult * 1.5) as i32;
            let check_z = (self.position.z + right_z * side_mult * 1.5) as i32;
            let check_y = self.position.y as i32;

            let block = world.get_block_world(check_x, check_y, check_z);
            if block.is_solid() {
                // Steer away from side obstacle
                avoidance.x -= right_x * side_mult * BIRD_AVOIDANCE_STRENGTH * 0.5;
                avoidance.z -= right_z * side_mult * BIRD_AVOIDANCE_STRENGTH * 0.5;
            }
        }

        avoidance
    }

    fn update_descending(&mut self, dt: f32) {
        if let Some(target) = self.target_perch {
            let target_pos = Point3::new(
                target.x as f32 + 0.5,
                target.y as f32 + 1.0,
                target.z as f32 + 0.5,
            );
            let to_target = target_pos - self.position;
            let distance = to_target.magnitude();

            if distance < 0.3 {
                // Land
                self.state = BirdState::Perched;
                self.position = target_pos;
                self.velocity = Vector3::new(0.0, 0.0, 0.0);
                self.state_timer = rand::thread_rng().gen_range(BIRD_PERCH_DURATION_MIN..BIRD_PERCH_DURATION_MAX);
            } else {
                // Continue descending
                let descent_speed = BIRD_FLY_SPEED * 0.5;
                self.velocity = to_target.normalize() * descent_speed;
                self.position += self.velocity * dt;

                // Update direction to face target
                self.direction = to_target.z.atan2(to_target.x);
            }
        } else {
            // No target, go back to flying
            self.state = BirdState::Flying;
        }
    }

    fn update_perched(&mut self, dt: f32) {
        let mut rng = rand::thread_rng();
        self.state_timer -= dt;
        self.velocity = Vector3::new(0.0, 0.0, 0.0);

        if self.state_timer <= 0.0 {
            // Time to do something
            if rng.gen::<f32>() < 0.3 {
                // Walk around
                self.state = BirdState::Walking;
                self.state_timer = rng.gen_range(1.0..3.0);
            } else {
                // Take off
                self.state = BirdState::TakingOff;
                self.state_timer = 0.5;
            }
        }
    }

    fn update_walking(&mut self, dt: f32) {
        let mut rng = rand::thread_rng();
        self.state_timer -= dt;

        // Small random movements
        let walk_speed = 0.5;
        self.velocity.x = self.direction.cos() * walk_speed;
        self.velocity.z = self.direction.sin() * walk_speed;

        // Occasionally change direction
        if rng.gen::<f32>() < dt * 2.0 {
            self.direction += rng.gen_range(-1.0..1.0);
        }

        // Very small movement
        self.position.x += self.velocity.x * dt * 0.1;
        self.position.z += self.velocity.z * dt * 0.1;

        if self.state_timer <= 0.0 {
            self.state = BirdState::Perched;
            self.state_timer = rng.gen_range(2.0..8.0);
        }
    }

    fn update_takeoff(&mut self, dt: f32) {
        self.state_timer -= dt;

        // Accelerate upward
        self.velocity.y = 8.0;
        self.velocity.x = self.direction.cos() * BIRD_FLY_SPEED * 0.5;
        self.velocity.z = self.direction.sin() * BIRD_FLY_SPEED * 0.5;
        self.position += self.velocity * dt;

        if self.state_timer <= 0.0 {
            self.state = BirdState::Flying;
            self.target_perch = None;
            self.target_height = self.position.y + 10.0;
        }
    }

    fn update_animations(&mut self, dt: f32) {
        match self.state {
            BirdState::Flying | BirdState::Descending | BirdState::TakingOff => {
                self.wing_phase += WING_FLAP_SPEED * dt;
                if self.wing_phase > std::f32::consts::TAU {
                    self.wing_phase -= std::f32::consts::TAU;
                }
                self.leg_phase = 0.0; // Legs tucked while flying
            }
            BirdState::Walking => {
                self.wing_phase = 0.0; // Wings folded
                self.leg_phase += LEG_WALK_SPEED * dt;
                if self.leg_phase > std::f32::consts::TAU {
                    self.leg_phase -= std::f32::consts::TAU;
                }
            }
            BirdState::Perched => {
                self.wing_phase = 0.0;
                self.leg_phase = 0.0;
            }
        }
    }
}

// ============================================================================
// BirdManager
// ============================================================================

pub struct BirdManager {
    pub birds: Vec<Bird>,
    spawn_timer: f32,
    landing_check_timer: f32,
    next_flock_id: usize,
}

impl BirdManager {
    pub fn new() -> Self {
        Self {
            birds: Vec::new(),
            spawn_timer: 0.0,
            landing_check_timer: 0.0,
            next_flock_id: 0,
        }
    }

    pub fn update(&mut self, dt: f32, player_pos: Point3<f32>, world: &World) {
        // Spawn new birds
        self.spawn_timer += dt;
        if self.spawn_timer >= BIRD_SPAWN_INTERVAL && self.birds.len() < BIRD_MAX_COUNT {
            self.try_spawn_bird(player_pos, world);
            self.spawn_timer = 0.0;
        }

        // Calculate boid forces for all birds
        let boid_forces = self.calculate_boid_forces();

        // Update each bird
        for (i, bird) in self.birds.iter_mut().enumerate() {
            let force = boid_forces.get(i).copied().unwrap_or(Vector3::new(0.0, 0.0, 0.0));
            bird.update(dt, world, force);
        }

        // Check for landing opportunities
        self.landing_check_timer += dt;
        if self.landing_check_timer >= BIRD_LANDING_CHECK_INTERVAL {
            self.check_for_landing_spots(world);
            self.landing_check_timer = 0.0;
        }

        // Despawn far birds
        self.birds.retain(|bird| {
            let dx = bird.position.x - player_pos.x;
            let dz = bird.position.z - player_pos.z;
            let dist = (dx * dx + dz * dz).sqrt();
            dist < BIRD_DESPAWN_DISTANCE
        });

        // Update flock assignments periodically
        self.update_flocks();
    }

    fn calculate_boid_forces(&self) -> Vec<Vector3<f32>> {
        self.birds.iter().enumerate().map(|(i, bird)| {
            if bird.state != BirdState::Flying {
                return Vector3::new(0.0, 0.0, 0.0);
            }

            let mut separation = Vector3::new(0.0, 0.0, 0.0);
            let mut alignment = Vector3::new(0.0, 0.0, 0.0);
            let mut cohesion = Vector3::new(0.0, 0.0, 0.0);
            let mut sep_count = 0;
            let mut align_count = 0;
            let mut cohesion_count = 0;

            for (j, other) in self.birds.iter().enumerate() {
                if i == j || other.state != BirdState::Flying {
                    continue;
                }

                let diff = bird.position - other.position;
                let dist = diff.magnitude();

                // Separation - avoid birds that are too close
                if dist < BOID_SEPARATION_RADIUS && dist > 0.001 {
                    separation += diff.normalize() / dist;
                    sep_count += 1;
                }

                // Only align/cohere with same flock or unflocked birds
                let same_flock = bird.flock_id.is_some() && bird.flock_id == other.flock_id;
                let both_unflocked = bird.flock_id.is_none() && other.flock_id.is_none();

                if same_flock || both_unflocked {
                    // Alignment - match velocity of nearby birds
                    if dist < BOID_ALIGNMENT_RADIUS {
                        alignment += other.velocity;
                        align_count += 1;
                    }

                    // Cohesion - move toward center of nearby birds
                    if dist < BOID_COHESION_RADIUS {
                        cohesion += Vector3::new(other.position.x, other.position.y, other.position.z);
                        cohesion_count += 1;
                    }
                }
            }

            let mut force = Vector3::new(0.0, 0.0, 0.0);

            if sep_count > 0 {
                let avg_sep = separation / sep_count as f32;
                force += avg_sep * BOID_SEPARATION_WEIGHT;
            }

            if align_count > 0 {
                let avg_vel = alignment / align_count as f32;
                if avg_vel.magnitude() > 0.001 {
                    let steer = avg_vel.normalize() - bird.velocity.normalize();
                    force += steer * BOID_ALIGNMENT_WEIGHT;
                }
            }

            if cohesion_count > 0 {
                let center = cohesion / cohesion_count as f32;
                let to_center = center - Vector3::new(bird.position.x, bird.position.y, bird.position.z);
                if to_center.magnitude() > 0.001 {
                    force += to_center.normalize() * BOID_COHESION_WEIGHT;
                }
            }

            force
        }).collect()
    }

    fn try_spawn_bird(&mut self, player_pos: Point3<f32>, world: &World) {
        let mut rng = rand::thread_rng();
        let angle = rng.gen_range(0.0..std::f32::consts::TAU);
        let distance = rng.gen_range(BIRD_SPAWN_DISTANCE * 0.8..BIRD_SPAWN_DISTANCE);

        let spawn_x = player_pos.x + angle.cos() * distance;
        let spawn_z = player_pos.z + angle.sin() * distance;
        let spawn_y = rng.gen_range(BIRD_MIN_HEIGHT..BIRD_MAX_HEIGHT);

        // Verify not inside terrain
        let block = world.get_block_world(spawn_x as i32, spawn_y as i32, spawn_z as i32);
        if !block.is_solid() {
            let mut bird = Bird::new(Point3::new(spawn_x, spawn_y, spawn_z));

            // Chance to join existing flock
            if rng.gen::<f32>() < BOID_FLOCK_CHANCE {
                if let Some(flock_id) = self.find_nearby_flock(Point3::new(spawn_x, spawn_y, spawn_z)) {
                    bird.flock_id = Some(flock_id);
                } else {
                    // Create new flock
                    bird.flock_id = Some(self.next_flock_id);
                    self.next_flock_id += 1;
                }
            }

            self.birds.push(bird);
        }
    }

    fn find_nearby_flock(&self, pos: Point3<f32>) -> Option<usize> {
        for bird in &self.birds {
            if bird.state == BirdState::Flying {
                if let Some(flock_id) = bird.flock_id {
                    let dx = bird.position.x - pos.x;
                    let dy = bird.position.y - pos.y;
                    let dz = bird.position.z - pos.z;
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist < BOID_COHESION_RADIUS * 2.0 {
                        return Some(flock_id);
                    }
                }
            }
        }
        None
    }

    fn update_flocks(&mut self) {
        // Unflocked birds may join nearby flocks
        let mut rng = rand::thread_rng();

        for i in 0..self.birds.len() {
            if self.birds[i].flock_id.is_none() && self.birds[i].state == BirdState::Flying {
                if rng.gen::<f32>() < 0.01 { // Small chance per frame
                    let pos = self.birds[i].position;
                    if let Some(flock_id) = self.find_nearby_flock(pos) {
                        self.birds[i].flock_id = Some(flock_id);
                    }
                }
            }
        }
    }

    fn check_for_landing_spots(&mut self, world: &World) {
        let mut rng = rand::thread_rng();

        for bird in &mut self.birds {
            if bird.state != BirdState::Flying {
                continue;
            }

            // Random chance to look for landing spot
            if rng.gen::<f32>() > BIRD_LANDING_CHANCE {
                continue;
            }

            // Search for leaves blocks below
            if let Some(perch) = find_landing_spot(bird.position, world) {
                bird.target_perch = Some(perch);
                bird.state = BirdState::Descending;
            }
        }
    }
}

fn find_landing_spot(pos: Point3<f32>, world: &World) -> Option<Point3<i32>> {
    let mut rng = rand::thread_rng();
    let mut candidates = Vec::new();

    let center_x = pos.x as i32;
    let center_z = pos.z as i32;

    // Sample a few random positions within scan radius
    for _ in 0..10 {
        let dx = rng.gen_range(-BIRD_LANDING_SCAN_RADIUS..=BIRD_LANDING_SCAN_RADIUS);
        let dz = rng.gen_range(-BIRD_LANDING_SCAN_RADIUS..=BIRD_LANDING_SCAN_RADIUS);
        let x = center_x + dx;
        let z = center_z + dz;

        // Search downward for landing spots
        for y in (30..(pos.y as i32)).rev() {
            let block = world.get_block_world(x, y, z);

            // Check if there's air above (can land)
            let above = world.get_block_world(x, y + 1, z);
            if above != BlockType::Air {
                if block.is_solid() {
                    break; // Can't land here, keep searching
                }
                continue;
            }

            // Prefer leaves blocks (trees)
            if block == BlockType::Leaves {
                candidates.push(Point3::new(x, y, z));
                break;
            }

            // Small chance to land on other solid blocks (not water, not air)
            if block.is_solid() && block != BlockType::Water && block != BlockType::Boundary {
                if rng.gen::<f32>() < BIRD_SOLID_LANDING_CHANCE {
                    candidates.push(Point3::new(x, y, z));
                }
                break; // Hit solid ground, stop searching this column
            }
        }
    }

    if candidates.is_empty() {
        None
    } else {
        Some(candidates[rng.gen_range(0..candidates.len())])
    }
}

// ============================================================================
// Bird Rendering
// ============================================================================

/// Creates all vertices for rendering a bird with its current animation state
/// Bird parts: head, beak, eyes, body, left wing, right wing, tail feathers, left leg, right leg
pub fn create_bird_vertices(bird: &Bird) -> Vec<Vertex> {
    let size = BIRD_BASE_SIZE * bird.size_modifier;
    let mut vertices = Vec::new();

    // Colors
    let body_color = [0.95, 0.95, 0.95];  // White
    let beak_color = [1.0, 0.5, 0.1];     // Orange
    let leg_color = [1.0, 0.5, 0.1];      // Orange
    let eye_color = [0.05, 0.05, 0.05];   // Black

    // Get pitch and roll for flying states, zero for perched
    let (pitch, roll) = match bird.state {
        BirdState::Flying | BirdState::Descending | BirdState::TakingOff => (bird.pitch, bird.roll),
        _ => (0.0, 0.0),
    };

    let pos = bird.position;
    let yaw = bird.direction;

    // === BODY (oval-ish, elongated front-to-back) ===
    // X = forward/backward, Z = side-to-side
    let body_length = size * 0.35;  // X dimension - front to back
    let body_height = size * 0.3;   // Y dimension
    let body_width = size * 0.25;   // Z dimension - side to side
    add_rotated_cube_full(
        &mut vertices,
        pos,
        Vector3::new(0.0, 0.0, 0.0),
        body_length, body_height, body_width,
        body_color,
        yaw, pitch, roll,
    );

    // === HEAD (distinct cube above and in front of body) ===
    let head_offset = rotate_point_full(size * 0.3, size * 0.22, 0.0, yaw, pitch, roll);
    add_rotated_cube_full(
        &mut vertices,
        pos,
        Vector3::new(head_offset[0], head_offset[1], head_offset[2]),
        size * 0.2, size * 0.2, size * 0.18,
        body_color,
        yaw, pitch, roll,
    );

    // === EYES (small black squares on each side of head) ===
    let eye_size = size * 0.05;
    // Left eye - on left side of head, slightly forward
    let left_eye_local = rotate_point_full(size * 0.32, size * 0.26, size * 0.08, yaw, pitch, roll);
    add_rotated_cube_full(
        &mut vertices,
        pos,
        Vector3::new(left_eye_local[0], left_eye_local[1], left_eye_local[2]),
        eye_size, eye_size, eye_size * 0.3,
        eye_color,
        yaw, pitch, roll,
    );
    // Right eye - on right side of head, slightly forward
    let right_eye_local = rotate_point_full(size * 0.32, size * 0.26, -size * 0.08, yaw, pitch, roll);
    add_rotated_cube_full(
        &mut vertices,
        pos,
        Vector3::new(right_eye_local[0], right_eye_local[1], right_eye_local[2]),
        eye_size, eye_size, eye_size * 0.3,
        eye_color,
        yaw, pitch, roll,
    );

    // === BEAK (small pointed cube at front of head) ===
    let beak_offset = rotate_point_full(size * 0.42, size * 0.16, 0.0, yaw, pitch, roll);
    add_rotated_cube_full(
        &mut vertices,
        pos,
        Vector3::new(beak_offset[0], beak_offset[1], beak_offset[2]),
        size * 0.1, size * 0.05, size * 0.06,
        beak_color,
        yaw, pitch, roll,
    );

    // === TAIL FEATHERS (flat piece extending backward) ===
    // X = length extending backward (longer), Z = width side-to-side (narrower)
    let tail_offset = rotate_point_full(-size * 0.25, size * 0.02, 0.0, yaw, pitch, roll);
    add_rotated_cube_full(
        &mut vertices,
        pos,
        Vector3::new(tail_offset[0], tail_offset[1], tail_offset[2]),
        size * 0.28, size * 0.02, size * 0.12,
        body_color,
        yaw, pitch, roll,
    );

    // === WINGS (rotate around bird's forward axis for flapping) ===
    // Wing flap angle: positive = wings up, negative = wings down
    let wing_angle = bird.wing_phase.sin() * WING_FLAP_AMPLITUDE;

    // Wing dimensions (in bird's local space before direction rotation):
    // - Local X = bird's forward direction
    // - Local Z = bird's left/right (wing extends this way)
    // - Local Y = up
    let wing_chord = size * 0.12;     // Front-to-back (X dimension) - small
    let wing_thickness = size * 0.02; // Up-down (Y dimension) - very thin
    let wing_span = size * 0.4;       // How far wing extends sideways (Z dimension) - long

    // Joint is at body's side edge
    let wing_joint = body_width * 0.5;

    // Left wing - extends in +Z direction (bird's left after rotation)
    add_wing_cube_full(
        &mut vertices,
        pos,
        wing_chord, wing_thickness, wing_span,
        body_color,
        yaw, pitch, roll,
        wing_angle,
        true,  // is left wing
        wing_joint,
    );

    // Right wing - extends in -Z direction (bird's right after rotation)
    add_wing_cube_full(
        &mut vertices,
        pos,
        wing_chord, wing_thickness, wing_span,
        body_color,
        yaw, pitch, roll,
        -wing_angle, // Opposite flap direction
        false, // is right wing
        wing_joint,
    );

    // === LEGS (animated when walking, tucked when flying) ===
    let leg_swing = if bird.state == BirdState::Walking {
        bird.leg_phase.sin() * LEG_WALK_AMPLITUDE
    } else {
        0.0
    };

    // Only show legs when perched or walking
    if bird.state == BirdState::Perched || bird.state == BirdState::Walking {
        let leg_spread = size * 0.08; // How far apart legs are (side to side)

        // Left leg
        let left_leg_local = rotate_point_full(leg_swing, -size * 0.22, leg_spread, yaw, 0.0, 0.0);
        add_rotated_cube_full(
            &mut vertices,
            pos,
            Vector3::new(left_leg_local[0], left_leg_local[1], left_leg_local[2]),
            size * 0.04, size * 0.15, size * 0.04,
            leg_color,
            yaw, 0.0, 0.0,
        );

        // Right leg (opposite swing phase)
        let right_leg_local = rotate_point_full(-leg_swing, -size * 0.22, -leg_spread, yaw, 0.0, 0.0);
        add_rotated_cube_full(
            &mut vertices,
            pos,
            Vector3::new(right_leg_local[0], right_leg_local[1], right_leg_local[2]),
            size * 0.04, size * 0.15, size * 0.04,
            leg_color,
            yaw, 0.0, 0.0,
        );
    }

    vertices
}

/// Adds a rotated cube to the vertex list
fn add_rotated_cube(
    vertices: &mut Vec<Vertex>,
    center: Point3<f32>,
    offset: Vector3<f32>,
    width: f32,
    height: f32,
    depth: f32,
    color: [f32; 3],
    cos_dir: f32,
    sin_dir: f32,
) {
    let half_w = width / 2.0;
    let half_h = height / 2.0;
    let half_d = depth / 2.0;

    // Apply offset (already in world-rotated space)
    let cx = center.x + offset.x;
    let cy = center.y + offset.y;
    let cz = center.z + offset.z;

    // Generate 8 corners relative to center, rotated by direction
    let corners = [
        rotate_point(-half_w, -half_h, -half_d, cos_dir, sin_dir),
        rotate_point( half_w, -half_h, -half_d, cos_dir, sin_dir),
        rotate_point( half_w,  half_h, -half_d, cos_dir, sin_dir),
        rotate_point(-half_w,  half_h, -half_d, cos_dir, sin_dir),
        rotate_point(-half_w, -half_h,  half_d, cos_dir, sin_dir),
        rotate_point( half_w, -half_h,  half_d, cos_dir, sin_dir),
        rotate_point( half_w,  half_h,  half_d, cos_dir, sin_dir),
        rotate_point(-half_w,  half_h,  half_d, cos_dir, sin_dir),
    ];

    // Translate corners to world position
    let corners: Vec<[f32; 3]> = corners.iter().map(|c| {
        [cx + c[0], cy + c[1], cz + c[2]]
    }).collect();

    let light_level = 1.0;
    let alpha = 1.0;
    let uv = [0.0, 0.0];
    let tex_index = TEX_NONE;

    // Front face (+Z local, rotated)
    let front_normal = rotate_point(0.0, 0.0, 1.0, cos_dir, sin_dir);
    vertices.push(Vertex { position: corners[4], color, normal: front_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[5], color, normal: front_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[6], color, normal: front_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[7], color, normal: front_normal, light_level, alpha, uv, tex_index });

    // Back face (-Z local, rotated)
    let back_normal = rotate_point(0.0, 0.0, -1.0, cos_dir, sin_dir);
    vertices.push(Vertex { position: corners[1], color, normal: back_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[0], color, normal: back_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[3], color, normal: back_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[2], color, normal: back_normal, light_level, alpha, uv, tex_index });

    // Top face (+Y)
    let top_normal = [0.0, 1.0, 0.0];
    vertices.push(Vertex { position: corners[7], color, normal: top_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[6], color, normal: top_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[2], color, normal: top_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[3], color, normal: top_normal, light_level, alpha, uv, tex_index });

    // Bottom face (-Y)
    let bottom_normal = [0.0, -1.0, 0.0];
    vertices.push(Vertex { position: corners[0], color, normal: bottom_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[1], color, normal: bottom_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[5], color, normal: bottom_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[4], color, normal: bottom_normal, light_level, alpha, uv, tex_index });

    // Right face (+X local, rotated)
    let right_normal = rotate_point(1.0, 0.0, 0.0, cos_dir, sin_dir);
    vertices.push(Vertex { position: corners[5], color, normal: right_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[1], color, normal: right_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[2], color, normal: right_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[6], color, normal: right_normal, light_level, alpha, uv, tex_index });

    // Left face (-X local, rotated)
    let left_normal = rotate_point(-1.0, 0.0, 0.0, cos_dir, sin_dir);
    vertices.push(Vertex { position: corners[0], color, normal: left_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[4], color, normal: left_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[7], color, normal: left_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[3], color, normal: left_normal, light_level, alpha, uv, tex_index });
}

/// Rotate a point around Y axis only (for normals and simple rotation)
fn rotate_point(x: f32, y: f32, z: f32, cos_dir: f32, sin_dir: f32) -> [f32; 3] {
    [
        x * cos_dir - z * sin_dir,
        y,
        x * sin_dir + z * cos_dir,
    ]
}

/// Rotate a point with full yaw, pitch, and roll
/// Order: roll (around local Z) -> pitch (around local X after roll) -> yaw (around Y)
/// Positive pitch = nose up, positive roll = bank right
fn rotate_point_full(x: f32, y: f32, z: f32, yaw: f32, pitch: f32, roll: f32) -> [f32; 3] {
    let cos_yaw = yaw.cos();
    let sin_yaw = yaw.sin();
    let cos_pitch = pitch.cos();
    let sin_pitch = pitch.sin();
    let cos_roll = roll.cos();
    let sin_roll = roll.sin();

    // Apply roll (rotation around local X axis - banking)
    // Positive roll = right wing down
    let x1 = x;
    let y1 = y * cos_roll - z * sin_roll;
    let z1 = y * sin_roll + z * cos_roll;

    // Apply pitch (rotation around local Z axis - nose up/down)
    // Positive pitch = nose up (standard aircraft convention)
    let x2 = x1 * cos_pitch - y1 * sin_pitch;
    let y2 = x1 * sin_pitch + y1 * cos_pitch;
    let z2 = z1;

    // Apply yaw (rotation around Y axis - heading)
    [
        x2 * cos_yaw - z2 * sin_yaw,
        y2,
        x2 * sin_yaw + z2 * cos_yaw,
    ]
}

/// Adds a rotated cube with full yaw/pitch/roll rotation
fn add_rotated_cube_full(
    vertices: &mut Vec<Vertex>,
    center: Point3<f32>,
    offset: Vector3<f32>,
    width: f32,
    height: f32,
    depth: f32,
    color: [f32; 3],
    yaw: f32,
    pitch: f32,
    roll: f32,
) {
    let half_w = width / 2.0;
    let half_h = height / 2.0;
    let half_d = depth / 2.0;

    // Apply offset (already in world space)
    let cx = center.x + offset.x;
    let cy = center.y + offset.y;
    let cz = center.z + offset.z;

    // Generate 8 corners relative to center, rotated by yaw/pitch/roll
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

    let corners: Vec<[f32; 3]> = local_corners.iter().map(|(lx, ly, lz)| {
        let rotated = rotate_point_full(*lx, *ly, *lz, yaw, pitch, roll);
        [cx + rotated[0], cy + rotated[1], cz + rotated[2]]
    }).collect();

    let light_level = 1.0;
    let alpha = 1.0;
    let uv = [0.0, 0.0];
    let tex_index = TEX_NONE;

    // Front face (+Z local, rotated)
    let front_normal = rotate_point_full(0.0, 0.0, 1.0, yaw, pitch, roll);
    vertices.push(Vertex { position: corners[4], color, normal: front_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[5], color, normal: front_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[6], color, normal: front_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[7], color, normal: front_normal, light_level, alpha, uv, tex_index });

    // Back face (-Z local, rotated)
    let back_normal = rotate_point_full(0.0, 0.0, -1.0, yaw, pitch, roll);
    vertices.push(Vertex { position: corners[1], color, normal: back_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[0], color, normal: back_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[3], color, normal: back_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[2], color, normal: back_normal, light_level, alpha, uv, tex_index });

    // Top face (+Y)
    let top_normal = rotate_point_full(0.0, 1.0, 0.0, yaw, pitch, roll);
    vertices.push(Vertex { position: corners[7], color, normal: top_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[6], color, normal: top_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[2], color, normal: top_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[3], color, normal: top_normal, light_level, alpha, uv, tex_index });

    // Bottom face (-Y)
    let bottom_normal = rotate_point_full(0.0, -1.0, 0.0, yaw, pitch, roll);
    vertices.push(Vertex { position: corners[0], color, normal: bottom_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[1], color, normal: bottom_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[5], color, normal: bottom_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[4], color, normal: bottom_normal, light_level, alpha, uv, tex_index });

    // Right face (+X local, rotated)
    let right_normal = rotate_point_full(1.0, 0.0, 0.0, yaw, pitch, roll);
    vertices.push(Vertex { position: corners[5], color, normal: right_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[1], color, normal: right_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[2], color, normal: right_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[6], color, normal: right_normal, light_level, alpha, uv, tex_index });

    // Left face (-X local, rotated)
    let left_normal = rotate_point_full(-1.0, 0.0, 0.0, yaw, pitch, roll);
    vertices.push(Vertex { position: corners[0], color, normal: left_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[4], color, normal: left_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[7], color, normal: left_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[3], color, normal: left_normal, light_level, alpha, uv, tex_index });
}

/// Adds a wing cube with full yaw/pitch/roll rotation plus flapping animation
fn add_wing_cube_full(
    vertices: &mut Vec<Vertex>,
    center: Point3<f32>,
    chord: f32,      // X dimension (front-to-back)
    thickness: f32,  // Y dimension (thin)
    span: f32,       // Z dimension (how far wing extends from body)
    color: [f32; 3],
    yaw: f32,
    pitch: f32,
    roll: f32,
    flap_angle: f32,
    is_left: bool,
    joint_offset: f32, // How far the joint is from bird center
) {
    let half_chord = chord / 2.0;
    let half_thick = thickness / 2.0;

    // Wing extends from joint outward
    let (z_near, z_far) = if is_left {
        (joint_offset, joint_offset + span)
    } else {
        (-joint_offset - span, -joint_offset)
    };

    let pivot_z = if is_left { joint_offset } else { -joint_offset };

    // Flap rotation is around the X axis (bird's forward direction in local space)
    let cos_flap = flap_angle.cos();
    let sin_flap = flap_angle.sin();

    // 8 corners in local coordinates (before flap and direction rotation)
    let local_corners = [
        (-half_chord, -half_thick, z_near),
        ( half_chord, -half_thick, z_near),
        ( half_chord,  half_thick, z_near),
        (-half_chord,  half_thick, z_near),
        (-half_chord, -half_thick, z_far),
        ( half_chord, -half_thick, z_far),
        ( half_chord,  half_thick, z_far),
        (-half_chord,  half_thick, z_far),
    ];

    let mut corners = Vec::with_capacity(8);

    for (lx, ly, lz) in local_corners {
        // 1. Apply flap rotation around X axis at the pivot point
        let rel_z = lz - pivot_z;
        let rel_y = ly;
        let rot_y = rel_y * cos_flap - rel_z * sin_flap;
        let rot_z = rel_y * sin_flap + rel_z * cos_flap;
        let flapped_x = lx;
        let flapped_y = rot_y;
        let flapped_z = rot_z + pivot_z;

        // 2. Apply full body rotation (yaw, pitch, roll)
        let world_rot = rotate_point_full(flapped_x, flapped_y, flapped_z, yaw, pitch, roll);
        corners.push([
            center.x + world_rot[0],
            center.y + world_rot[1],
            center.z + world_rot[2],
        ]);
    }

    let light_level = 1.0;
    let alpha = 1.0;
    let uv = [0.0, 0.0];
    let tex_index = TEX_NONE;

    // Front face
    let front_normal = rotate_point_full(0.0, 0.0, 1.0, yaw, pitch, roll);
    vertices.push(Vertex { position: corners[4], color, normal: front_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[5], color, normal: front_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[6], color, normal: front_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[7], color, normal: front_normal, light_level, alpha, uv, tex_index });

    // Back face
    let back_normal = rotate_point_full(0.0, 0.0, -1.0, yaw, pitch, roll);
    vertices.push(Vertex { position: corners[1], color, normal: back_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[0], color, normal: back_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[3], color, normal: back_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[2], color, normal: back_normal, light_level, alpha, uv, tex_index });

    // Top face
    let top_normal = rotate_point_full(0.0, 1.0, 0.0, yaw, pitch, roll);
    vertices.push(Vertex { position: corners[7], color, normal: top_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[6], color, normal: top_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[2], color, normal: top_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[3], color, normal: top_normal, light_level, alpha, uv, tex_index });

    // Bottom face
    let bottom_normal = rotate_point_full(0.0, -1.0, 0.0, yaw, pitch, roll);
    vertices.push(Vertex { position: corners[0], color, normal: bottom_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[1], color, normal: bottom_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[5], color, normal: bottom_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[4], color, normal: bottom_normal, light_level, alpha, uv, tex_index });

    // Right face
    let right_normal = rotate_point_full(1.0, 0.0, 0.0, yaw, pitch, roll);
    vertices.push(Vertex { position: corners[5], color, normal: right_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[1], color, normal: right_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[2], color, normal: right_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[6], color, normal: right_normal, light_level, alpha, uv, tex_index });

    // Left face
    let left_normal = rotate_point_full(-1.0, 0.0, 0.0, yaw, pitch, roll);
    vertices.push(Vertex { position: corners[0], color, normal: left_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[4], color, normal: left_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[7], color, normal: left_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[3], color, normal: left_normal, light_level, alpha, uv, tex_index });
}

/// Adds a wing cube that rotates around the bird's forward axis (flapping motion)
/// In bird's local coordinates: X = forward, Y = up, Z = left/right
/// Wing extends along Z axis, flaps by rotating around X axis
#[allow(dead_code)]
fn add_wing_cube(
    vertices: &mut Vec<Vertex>,
    center: Point3<f32>,
    _offset: Vector3<f32>,
    chord: f32,      // X dimension (front-to-back)
    thickness: f32,  // Y dimension (thin)
    span: f32,       // Z dimension (how far wing extends from body)
    color: [f32; 3],
    cos_dir: f32,
    sin_dir: f32,
    flap_angle: f32,
    is_left: bool,
    joint_offset: f32, // How far the joint is from bird center
) {
    let half_chord = chord / 2.0;
    let half_thick = thickness / 2.0;

    // Wing extends from joint outward
    // Left wing: joint at +Z (body edge), extends further in +Z
    // Right wing: joint at -Z (body edge), extends further in -Z
    let (z_near, z_far) = if is_left {
        (joint_offset, joint_offset + span)
    } else {
        (-joint_offset - span, -joint_offset)
    };

    // The joint (pivot for flapping) is at z = joint_offset (or -joint_offset for right)
    let pivot_z = if is_left { joint_offset } else { -joint_offset };

    // Flap rotation is around the X axis (bird's forward direction in local space)
    let cos_flap = flap_angle.cos();
    let sin_flap = flap_angle.sin();

    // 8 corners in local coordinates (before flap and direction rotation)
    let local_corners = [
        (-half_chord, -half_thick, z_near),
        ( half_chord, -half_thick, z_near),
        ( half_chord,  half_thick, z_near),
        (-half_chord,  half_thick, z_near),
        (-half_chord, -half_thick, z_far),
        ( half_chord, -half_thick, z_far),
        ( half_chord,  half_thick, z_far),
        (-half_chord,  half_thick, z_far),
    ];

    let mut corners = Vec::with_capacity(8);

    for (lx, ly, lz) in local_corners {
        // Rotate around X axis at the pivot point (joint)
        // 1. Translate so pivot is at origin (only Z needs adjustment)
        let rel_z = lz - pivot_z;
        let rel_y = ly;

        // 2. Rotate in Y-Z plane (rotation around X axis)
        let rot_y = rel_y * cos_flap - rel_z * sin_flap;
        let rot_z = rel_y * sin_flap + rel_z * cos_flap;

        // 3. Translate back
        let final_x = lx;
        let final_y = rot_y;
        let final_z = rot_z + pivot_z;

        // 4. Apply bird's direction rotation (around Y axis)
        let world_rot = rotate_point(final_x, final_y, final_z, cos_dir, sin_dir);
        corners.push([
            center.x + world_rot[0],
            center.y + world_rot[1],
            center.z + world_rot[2],
        ]);
    }

    let light_level = 1.0;
    let alpha = 1.0;
    let uv = [0.0, 0.0];
    let tex_index = TEX_NONE;

    // Front face (+Z local after rotation = wing tip side for left wing)
    let front_normal = rotate_point(0.0, 0.0, 1.0, cos_dir, sin_dir);
    vertices.push(Vertex { position: corners[4], color, normal: front_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[5], color, normal: front_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[6], color, normal: front_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[7], color, normal: front_normal, light_level, alpha, uv, tex_index });

    // Back face (-Z local = body side for left wing)
    let back_normal = rotate_point(0.0, 0.0, -1.0, cos_dir, sin_dir);
    vertices.push(Vertex { position: corners[1], color, normal: back_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[0], color, normal: back_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[3], color, normal: back_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[2], color, normal: back_normal, light_level, alpha, uv, tex_index });

    // Top face (+Y)
    let top_normal = [0.0, 1.0, 0.0];
    vertices.push(Vertex { position: corners[7], color, normal: top_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[6], color, normal: top_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[2], color, normal: top_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[3], color, normal: top_normal, light_level, alpha, uv, tex_index });

    // Bottom face (-Y)
    let bottom_normal = [0.0, -1.0, 0.0];
    vertices.push(Vertex { position: corners[0], color, normal: bottom_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[1], color, normal: bottom_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[5], color, normal: bottom_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[4], color, normal: bottom_normal, light_level, alpha, uv, tex_index });

    // Right face (+X local = bird's forward)
    let right_normal = rotate_point(1.0, 0.0, 0.0, cos_dir, sin_dir);
    vertices.push(Vertex { position: corners[5], color, normal: right_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[1], color, normal: right_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[2], color, normal: right_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[6], color, normal: right_normal, light_level, alpha, uv, tex_index });

    // Left face (-X local = bird's backward)
    let left_normal = rotate_point(-1.0, 0.0, 0.0, cos_dir, sin_dir);
    vertices.push(Vertex { position: corners[0], color, normal: left_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[4], color, normal: left_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[7], color, normal: left_normal, light_level, alpha, uv, tex_index });
    vertices.push(Vertex { position: corners[3], color, normal: left_normal, light_level, alpha, uv, tex_index });
}

/// Generate indices for N cubes (each cube has 24 vertices, 36 indices)
pub fn generate_bird_indices(num_cubes: usize) -> Vec<u16> {
    let mut indices = Vec::with_capacity(num_cubes * 36);
    for cube_idx in 0..num_cubes {
        let base = (cube_idx * 24) as u16;
        // Each face: 2 triangles
        for face in 0..6 {
            let face_base = base + (face * 4) as u16;
            indices.push(face_base);
            indices.push(face_base + 1);
            indices.push(face_base + 2);
            indices.push(face_base + 2);
            indices.push(face_base + 3);
            indices.push(face_base);
        }
    }
    indices
}
