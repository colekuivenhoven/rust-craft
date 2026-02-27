use cgmath::{Point3, Vector3, InnerSpace};
use rand::Rng;
use crate::block::{BlockType, Vertex};
use crate::world::World;
use crate::texture::TEX_NONE;

// ============================================================================
// Configuration Constants
// ============================================================================

// Spawning
pub const BIRD_SPAWN_INTERVAL: f32 = 5.0;      // Seconds between spawn attempts
pub const BIRD_MAX_COUNT: usize = 150;         // Maximum birds in world
pub const BIRD_SPAWN_DISTANCE: f32 = 40.0;     // Distance from player to spawn
pub const BIRD_DESPAWN_DISTANCE: f32 = 100.0;  // Distance at which birds despawn

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
pub const BIRD_TURN_RATE: f32 = 3.0;           // Radians per second turning
pub const BIRD_MIN_HEIGHT: f32 = 35.0;         // Minimum flight altitude
pub const BIRD_MAX_HEIGHT: f32 = 120.0;        // Maximum flight altitude
pub const BIRD_FLIGHT_RANDOMNESS: f32 = 0.5;   // Random direction change factor
pub const BIRD_HEIGHT_CHANGE_RATE: f32 = 4.0;  // Vertical movement speed

// Landing
pub const BIRD_LANDING_CHECK_INTERVAL: f32 = 3.0; // How often to check for landing spots
pub const BIRD_LANDING_CHANCE: f32 = 0.15;        // Probability to land when spot found
pub const BIRD_PERCH_DURATION_MIN: f32 = 5.0;     // Minimum time perched
pub const BIRD_PERCH_DURATION_MAX: f32 = 20.0;    // Maximum time perched
pub const BIRD_LANDING_SCAN_RADIUS: i32 = 15;     // Radius to search for trees
pub const BIRD_SOLID_LANDING_CHANCE: f32 = 0.1;   // Chance to land on non-leaves solid blocks

// Obstacle avoidance
pub const BIRD_LOOK_AHEAD_DISTANCE: f32 = 10.0;    // How far ahead to check for obstacles
pub const BIRD_AVOIDANCE_STRENGTH: f32 = 6.0;      // How strongly to steer away from obstacles

// Flight dynamics
pub const BIRD_MAX_PITCH: f32 = 0.8;              // Maximum pitch angle (radians)
pub const BIRD_MAX_ROLL: f32 = 0.8;               // Maximum roll angle when turning (radians)
pub const BIRD_ROTATION_SMOOTHING: f32 = 5.0;     // Smoothing factor for pitch/roll changes

// Animation
pub const WING_FLAP_SPEED: f32 = 15.0;         // Wing flaps per second while flying
pub const WING_FLAP_AMPLITUDE: f32 = 0.6;      // Maximum wing rotation (radians)
pub const LEG_WALK_SPEED: f32 = 10.0;          // Leg movement cycles per second
pub const LEG_WALK_AMPLITUDE: f32 = 0.3;       // Maximum leg swing distance

// Bird size
pub const BIRD_BASE_SIZE: f32 = 0.6;           // Base size multiplier

// ============================================================================
// Bird Colors
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BirdColor {
    Red,    // Scarlet Macaw-ish
    Blue,   // Blue Jay-ish
    Green,  // Parrot-ish
    Yellow, // Canary-ish
    White,  // Seagull/Dove
    Brown,  // Sparrow
}

impl BirdColor {
    pub fn random() -> Self {
        let mut rng = rand::thread_rng();
        match rng.gen_range(0..6) {
            0 => BirdColor::Red,
            1 => BirdColor::Blue,
            2 => BirdColor::Green,
            3 => BirdColor::Yellow,
            4 => BirdColor::White,
            _ => BirdColor::Brown,
        }
    }

    pub fn primary_rgb(&self) -> [f32; 3] {
        match self {
            BirdColor::Red => [0.9, 0.2, 0.2],
            BirdColor::Blue => [0.2, 0.4, 0.9],
            BirdColor::Green => [0.2, 0.8, 0.3],
            BirdColor::Yellow => [0.95, 0.85, 0.2],
            BirdColor::White => [0.95, 0.95, 0.95],
            BirdColor::Brown => [0.55, 0.40, 0.25],
        }
    }

    pub fn secondary_rgb(&self) -> [f32; 3] {
        // Wing tips / accent color
        match self {
            BirdColor::Red => [0.9, 0.8, 0.2],     // Yellow on red bird
            BirdColor::Blue => [0.1, 0.1, 0.3],    // Darker blue
            BirdColor::Green => [0.8, 0.2, 0.2],   // Red on green bird
            BirdColor::Yellow => [0.9, 0.9, 0.9],  // White on yellow bird
            BirdColor::White => [0.7, 0.7, 0.7],   // Grey on white bird
            BirdColor::Brown => [0.3, 0.2, 0.1],   // Dark brown
        }
    }
}

// ============================================================================
// BirdState Enum
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BirdState {
    Flying,      // Normal flight, may be flocking
    Descending,  // Flying down toward a landing spot
    Perched,     // Sitting on a block
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
    pub wing_phase: f32,             // Current phase of wing flap animation
    pub leg_phase: f32,              // Current phase of leg walk animation
    pub speed_modifier: f32,         // 0.8 - 1.2 multiplier for this bird
    pub size_modifier: f32,          // 0.9 - 1.1 multiplier for visual size
    pub target_height: f32,          // Current target flying height
    pub random_timer: f32,           // Timer for random direction changes
    pub color: BirdColor,            // Visual color
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
            color: BirdColor::random(),
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
            self.target_direction += yaw_diff * 0.6; // Stronger influence
            
            // Vertical alignment with flock
            if boid_force.y.abs() > 0.1 {
                self.target_height += boid_force.y * 5.0;
            }
        }

        // Obstacle avoidance - look ahead in flight direction
        let avoidance = self.calculate_avoidance(world);
        if avoidance.magnitude() > 0.01 {
            // Steer away from obstacles
            let avoid_yaw = avoidance.z.atan2(avoidance.x);
            self.target_direction = avoid_yaw;
            // Also adjust target height based on vertical avoidance
            if avoidance.y > 0.1 {
                self.target_height = (self.position.y + 15.0).min(BIRD_MAX_HEIGHT);
            } else if avoidance.y < -0.1 {
                self.target_height = (self.position.y - 10.0).max(BIRD_MIN_HEIGHT);
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
        let target_roll = (yaw_diff * 2.0).clamp(-BIRD_MAX_ROLL, BIRD_MAX_ROLL);
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

        for dist in [2.0, 5.0, BIRD_LOOK_AHEAD_DISTANCE] {
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
            let check_x = (self.position.x + right_x * side_mult * 2.0) as i32;
            let check_z = (self.position.z + right_z * side_mult * 2.0) as i32;
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
                target.y as f32 + 1.0, // Land ON TOP of the block
                target.z as f32 + 0.5,
            );
            let to_target = target_pos - self.position;
            let distance = to_target.magnitude();

            if distance < 0.5 {
                // Land
                self.state = BirdState::Perched;
                self.position = target_pos;
                self.velocity = Vector3::new(0.0, 0.0, 0.0);
                self.pitch = 0.0;
                self.roll = 0.0;
                self.state_timer = rand::thread_rng().gen_range(BIRD_PERCH_DURATION_MIN..BIRD_PERCH_DURATION_MAX);
            } else {
                // Continue descending
                let descent_speed = BIRD_FLY_SPEED * 0.6;
                let dir = to_target.normalize();
                self.velocity = dir * descent_speed;
                self.position += self.velocity * dt;

                // Update direction to face target
                let target_yaw = dir.z.atan2(dir.x);
                // Smooth turn
                let mut yaw_diff = target_yaw - self.direction;
                while yaw_diff > std::f32::consts::PI { yaw_diff -= std::f32::consts::TAU; }
                while yaw_diff < -std::f32::consts::PI { yaw_diff += std::f32::consts::TAU; }
                self.direction += yaw_diff * dt * 5.0;
                
                // Pitch down to look at landing spot
                self.pitch = -0.3;
                self.roll = 0.0;
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
        self.pitch = 0.0;
        self.roll = 0.0;

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
        let walk_speed = 0.8;
        self.velocity.x = self.direction.cos() * walk_speed;
        self.velocity.z = self.direction.sin() * walk_speed;
        
        // Occasionally change direction
        if rng.gen::<f32>() < dt * 3.0 {
            self.direction += rng.gen_range(-1.5..1.5);
        }

        // Apply movement
        self.position.x += self.velocity.x * dt;
        self.position.z += self.velocity.z * dt;

        if self.state_timer <= 0.0 {
            self.state = BirdState::Perched;
            self.state_timer = rng.gen_range(2.0..6.0);
        }
    }

    fn update_takeoff(&mut self, dt: f32) {
        self.state_timer -= dt;

        // Accelerate upward and forward
        self.velocity.y = 5.0;
        self.velocity.x = self.direction.cos() * BIRD_FLY_SPEED * 0.6;
        self.velocity.z = self.direction.sin() * BIRD_FLY_SPEED * 0.6;
        self.position += self.velocity * dt;
        self.pitch = 0.5; // Pitch up

        if self.state_timer <= 0.0 {
            self.state = BirdState::Flying;
            self.target_perch = None;
            self.target_height = self.position.y + 20.0;
        }
    }

    fn update_animations(&mut self, dt: f32) {
        match self.state {
            BirdState::Flying | BirdState::Descending | BirdState::TakingOff => {
                let flap_speed = if self.state == BirdState::Descending { WING_FLAP_SPEED * 0.5 } else { WING_FLAP_SPEED };
                self.wing_phase += flap_speed * dt;
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
                    // Match flock color
                    if let Some(flock_mate) = self.birds.iter().find(|b| b.flock_id == Some(flock_id)) {
                        bird.color = flock_mate.color;
                    }
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
                        // Adopt flock color
                         if let Some(flock_mate) = self.birds.iter().find(|b| b.flock_id == Some(flock_id)) {
                            self.birds[i].color = flock_mate.color;
                        }
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
/// Bird parts: head, body, tail, wings, legs
pub fn create_bird_vertices(bird: &Bird) -> Vec<Vertex> {
    let size = BIRD_BASE_SIZE * bird.size_modifier;
    let mut vertices = Vec::new();

    // Colors
    let primary_color = bird.color.primary_rgb();
    let secondary_color = bird.color.secondary_rgb();
    let beak_color = [1.0, 0.6, 0.1];     // Orange/Yellow beak
    let leg_color = [0.4, 0.3, 0.2];      // Dark grey/brown legs
    let eye_color = [0.05, 0.05, 0.05];   // Black

    // Get pitch and roll for flying states, zero for perched
    let (pitch, roll) = match bird.state {
        BirdState::Flying | BirdState::Descending | BirdState::TakingOff => (bird.pitch, bird.roll),
        _ => (0.0, 0.0),
    };

    let pos = bird.position;
    let yaw = bird.direction;

    // === 1. BODY (Main Torso) ===
    // Central slightly oval shape
    add_rotated_cube_full(
        &mut vertices,
        pos,
        Vector3::new(0.0, 0.0, 0.0),
        size * 0.40, size * 0.28, size * 0.28, // L, H, W
        primary_color,
        yaw, pitch, roll,
    );

    // === 2. HEAD (Sphere-like cube) ===
    // Placed at the front, slightly up
    let head_pos = rotate_point_full(size * 0.25, size * 0.18, 0.0, yaw, pitch, roll);
    add_rotated_cube_full(
        &mut vertices,
        pos,
        Vector3::new(head_pos[0], head_pos[1], head_pos[2]),
        size * 0.22, size * 0.22, size * 0.22,
        primary_color,
        yaw, pitch, roll,
    );

    // === 3. BEAK ===
    let beak_pos = rotate_point_full(size * 0.40, size * 0.16, 0.0, yaw, pitch, roll);
    add_rotated_cube_full(
        &mut vertices,
        pos,
        Vector3::new(beak_pos[0], beak_pos[1], beak_pos[2]),
        size * 0.12, size * 0.06, size * 0.08,
        beak_color,
        yaw, pitch, roll,
    );

    // === 4. EYES ===
    let eye_size = size * 0.05;
    let left_eye = rotate_point_full(size * 0.28, size * 0.22, size * 0.115, yaw, pitch, roll);
    add_rotated_cube_full(
        &mut vertices,
        pos,
        Vector3::new(left_eye[0], left_eye[1], left_eye[2]),
        eye_size, eye_size, eye_size * 0.2,
        eye_color,
        yaw, pitch, roll,
    );
    let right_eye = rotate_point_full(size * 0.28, size * 0.22, -size * 0.115, yaw, pitch, roll);
    add_rotated_cube_full(
        &mut vertices,
        pos,
        Vector3::new(right_eye[0], right_eye[1], right_eye[2]),
        eye_size, eye_size, eye_size * 0.2,
        eye_color,
        yaw, pitch, roll,
    );

    // === 5. TAIL ===
    // Tapers out the back
    let tail_pos = rotate_point_full(-size * 0.35, -size * 0.05, 0.0, yaw, pitch, roll);
    add_rotated_cube_full(
        &mut vertices,
        pos,
        Vector3::new(tail_pos[0], tail_pos[1], tail_pos[2]),
        size * 0.4, size * 0.05, size * 0.20,
        secondary_color,
        yaw, pitch - 0.1, roll, // Slight droop
    );

    // === 6. WINGS (Articulated) ===
    let wing_flap = bird.wing_phase.sin() * WING_FLAP_AMPLITUDE;
    let wing_base_z = size * 0.15; // Body width / 2 approx

    // Left Wing
    add_full_wing_render(
        &mut vertices,
        pos,
        size,
        primary_color, secondary_color,
        yaw, pitch, roll,
        wing_flap,
        true, // is left
        wing_base_z
    );

    // Right Wing
    add_full_wing_render(
        &mut vertices,
        pos,
        size,
        primary_color, secondary_color,
        yaw, pitch, roll,
        wing_flap, // Symmetric flap phase
        false, // is right
        wing_base_z
    );

    // === 7. LEGS ===
    let leg_offset_x = -size * 0.05;
    let leg_offset_z = size * 0.1;
    let leg_height = size * 0.18;
    
    // Animation for walking
    let left_leg_angle = if bird.state == BirdState::Walking { bird.leg_phase.sin() * LEG_WALK_AMPLITUDE } else { 0.0 };
    let right_leg_angle = if bird.state == BirdState::Walking { (bird.leg_phase + std::f32::consts::PI).sin() * LEG_WALK_AMPLITUDE } else { 0.0 };

    // Only draw extended legs if walking or perched. 
    if bird.state == BirdState::Flying || bird.state == BirdState::Descending || bird.state == BirdState::TakingOff {
        // Tucked legs
        let l_tuck = rotate_point_full(leg_offset_x, -size * 0.14, leg_offset_z, yaw, pitch, roll);
        add_rotated_cube_full(&mut vertices, pos, Vector3::new(l_tuck[0], l_tuck[1], l_tuck[2]), size*0.08, size*0.04, size*0.04, leg_color, yaw, pitch, roll);
        
        let r_tuck = rotate_point_full(leg_offset_x, -size * 0.14, -leg_offset_z, yaw, pitch, roll);
        add_rotated_cube_full(&mut vertices, pos, Vector3::new(r_tuck[0], r_tuck[1], r_tuck[2]), size*0.08, size*0.04, size*0.04, leg_color, yaw, pitch, roll);
    } else {
        // Extended legs
        // Left
        let l_leg_pos = rotate_point_full(leg_offset_x, -size * 0.14 - leg_height/2.0, leg_offset_z, yaw, 0.0, 0.0);
        add_rotated_cube_full(
            &mut vertices, 
            pos, 
            Vector3::new(l_leg_pos[0], l_leg_pos[1], l_leg_pos[2]), 
            size*0.04, leg_height, size*0.04, 
            leg_color, 
            yaw, 0.0, left_leg_angle 
        );
         // Right
        let r_leg_pos = rotate_point_full(leg_offset_x, -size * 0.14 - leg_height/2.0, -leg_offset_z, yaw, 0.0, 0.0);
        add_rotated_cube_full(
            &mut vertices, 
            pos, 
            Vector3::new(r_leg_pos[0], r_leg_pos[1], r_leg_pos[2]), 
            size*0.04, leg_height, size*0.04, 
            leg_color, 
            yaw, 0.0, right_leg_angle
        );
    }

    vertices
}

// Renders a two-part articulated wing
fn add_full_wing_render(
    vertices: &mut Vec<Vertex>,
    center: Point3<f32>,
    size: f32,
    inner_color: [f32; 3],
    outer_color: [f32; 3],
    yaw: f32, pitch: f32, roll: f32,
    flap_angle: f32,
    is_left: bool,
    shoulder_z: f32,
) {
    let side = if is_left { 1.0 } else { -1.0 };
    
    let inner_span = size * 0.35;
    let chord = size * 0.30;
    let thickness = size * 0.04;

    // Angles
    let inner_rot = flap_angle;
    // Outer wing amplifies the flap slightly for fluidity
    let outer_rot = flap_angle * 1.3 + (if flap_angle.sin() > 0.0 { 0.2 } else { -0.1 }); 

    // --- INNER WING ---
    let mut inner_corners = Vec::new();
    let box_inner = get_wing_box(inner_span, chord, thickness, side);
    
    for p in box_inner.iter() {
        // 1. Flap Rotation
        let (fx, fy, fz) = rotate_x(p[0], p[1], p[2], inner_rot * side);
        
        // 2. Shoulder Offset
        let sx = fx;
        let sy = fy + size * 0.15; // Shoulder height
        let sz = fz + shoulder_z * side;
        
        // 3. Body Rotation
        let final_rot = rotate_point_full(sx, sy, sz, yaw, pitch, roll);
        inner_corners.push([center.x + final_rot[0], center.y + final_rot[1], center.z + final_rot[2]]);
    }
    push_cube_verts(vertices, inner_corners, inner_color);

    // --- OUTER WING ---
    let mut outer_corners = Vec::new();
    let box_outer = get_wing_box(inner_span, chord * 0.8, thickness * 0.8, side);
    
    // FIX: Calculate Elbow Pivot by rotating the actual tip of the inner wing.
    // The inner wing extends from 0 to (inner_span * side) along Z.
    let (_, elbow_y, elbow_z) = rotate_x(0.0, 0.0, inner_span * side, inner_rot * side);
    
    for p in box_outer.iter() {
        // 1. Flap (Outer rotation relative to horizontal, centered at elbow)
        let (fx, fy, fz) = rotate_x(p[0], p[1], p[2], outer_rot * side);
        
        // 2. Move to Elbow position (relative to shoulder)
        let ex = fx;
        let ey = fy + elbow_y;
        let ez = fz + elbow_z;
        
        // 3. Move to Shoulder (relative to body center)
        let sx = ex;
        let sy = ey + size * 0.15;
        let sz = ez + shoulder_z * side;
        
        // 4. Body Rotation
        let final_rot = rotate_point_full(sx, sy, sz, yaw, pitch, roll);
        outer_corners.push([center.x + final_rot[0], center.y + final_rot[1], center.z + final_rot[2]]);
    }
    push_cube_verts(vertices, outer_corners, outer_color);
}

// Helper: Get 8 corners of a box extending along Z (or -Z) from origin
fn get_wing_box(span: f32, chord: f32, thickness: f32, side: f32) -> [[f32; 3]; 8] {
    let h_ch = chord / 2.0;
    let h_th = thickness / 2.0;
    
    // If side > 0 (Left), extend 0 to +span
    // If side < 0 (Right), extend 0 to -span
    let z_near = 0.0;
    let z_far = span * side;
    
    [
        [-h_ch, -h_th, z_near], // 0
        [ h_ch, -h_th, z_near], // 1
        [ h_ch,  h_th, z_near], // 2
        [-h_ch,  h_th, z_near], // 3
        [-h_ch, -h_th, z_far],  // 4
        [ h_ch, -h_th, z_far],  // 5
        [ h_ch,  h_th, z_far],  // 6
        [-h_ch,  h_th, z_far],  // 7
    ]
}

// Helper: Rotate around X axis
fn rotate_x(x: f32, y: f32, z: f32, angle: f32) -> (f32, f32, f32) {
    let c = angle.cos();
    let s = angle.sin();
    (x, y * c - z * s, y * s + z * c)
}

// Helper: Push standard cube faces given 8 corners
fn push_cube_verts(vertices: &mut Vec<Vertex>, corners: Vec<[f32; 3]>, color: [f32; 3]) {
    // Indices for faces
    let faces = [
        [4, 5, 6, 7], // Front
        [1, 0, 3, 2], // Back
        [7, 6, 2, 3], // Top
        [0, 1, 5, 4], // Bottom
        [5, 1, 2, 6], // Right
        [0, 4, 7, 3], // Left
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
                ao: 1.0
            });
        }
    }
}

/// Rotate a point with full yaw, pitch, and roll
fn rotate_point_full(x: f32, y: f32, z: f32, yaw: f32, pitch: f32, roll: f32) -> [f32; 3] {
    let cos_yaw = yaw.cos();
    let sin_yaw = yaw.sin();
    let cos_pitch = pitch.cos();
    let sin_pitch = pitch.sin();
    let cos_roll = roll.cos();
    let sin_roll = roll.sin();

    // 1. Roll (around X)
    let x1 = x;
    let y1 = y * cos_roll - z * sin_roll;
    let z1 = y * sin_roll + z * cos_roll;

    // 2. Pitch (around Z)
    let x2 = x1 * cos_pitch - y1 * sin_pitch;
    let y2 = x1 * sin_pitch + y1 * cos_pitch;
    let z2 = z1;

    // 3. Yaw (around Y)
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
        let rotated = rotate_point_full(lx, ly, lz, yaw, pitch, roll);
        corners.push([cx + rotated[0], cy + rotated[1], cz + rotated[2]]);
    }
    
    push_cube_verts(vertices, corners, color);
}

// Generate indices (standard)
pub fn generate_bird_indices(num_cubes: usize) -> Vec<u16> {
    let mut indices = Vec::with_capacity(num_cubes * 36);
    for cube_idx in 0..num_cubes {
        let base = (cube_idx * 24) as u16;
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