use cgmath::{Point3, Vector3, InnerSpace};
use rand::Rng;
use crate::block::{BlockType, Vertex};
use crate::world::World;
use crate::texture::TEX_NONE;

// ============================================================================
// Configuration Constants
// ============================================================================

// Spawning
pub const FISH_SPAWN_INTERVAL: f32 = 0.3;       // Seconds between spawn attempts
pub const FISH_MAX_COUNT: usize = 200;          // Maximum fish in world
pub const FISH_SPAWN_DISTANCE: f32 = 30.0;      // Distance from player to spawn
pub const FISH_DESPAWN_DISTANCE: f32 = 80.0;    // Distance at which fish despawn

// Schooling (Boids) - Fish school more tightly than birds flock
pub const BOID_SEPARATION_RADIUS: f32 = 1.0;    // Avoid fish closer than this
pub const BOID_ALIGNMENT_RADIUS: f32 = 6.0;     // Align with fish within this radius
pub const BOID_COHESION_RADIUS: f32 = 8.0;      // Move toward center of fish within this
pub const BOID_SEPARATION_WEIGHT: f32 = 2.0;    // Weight for separation force
pub const BOID_ALIGNMENT_WEIGHT: f32 = 1.5;     // Weight for alignment force
pub const BOID_COHESION_WEIGHT: f32 = 1.2;      // Weight for cohesion force
pub const BOID_SCHOOL_CHANCE: f32 = 0.95;       // Much higher than birds (0.7)

// Swimming
pub const FISH_SWIM_SPEED: f32 = 4.0;           // Base swim speed
pub const FISH_TURN_RATE: f32 = 3.0;            // Radians per second turning
pub const FISH_VERTICAL_SPEED: f32 = 1.5;       // Vertical movement speed
pub const FISH_SWIM_RANDOMNESS: f32 = 0.3;      // Random direction change factor

// Obstacle avoidance
pub const FISH_LOOK_AHEAD_DISTANCE: f32 = 5.0;  // How far ahead to check for obstacles
pub const FISH_AVOIDANCE_STRENGTH: f32 = 5.0;   // How strongly to steer away from obstacles

// Animation
pub const WIGGLE_SPEED: f32 = 10.0;             // Wiggle cycles per second
pub const WIGGLE_AMPLITUDE: f32 = 0.25;         // Maximum wiggle angle (radians)
pub const TAIL_WIGGLE_MULTIPLIER: f32 = 1.5;    // Tail wiggles more than body

// Fish size
pub const FISH_BASE_SIZE: f32 = 1.0;            // Base size multiplier

// ============================================================================
// Fish Colors
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FishColor {
    Red,
    Blue,
    Green,
    Yellow,
    Purple,
    Brown,
}

impl FishColor {
    pub fn random() -> Self {
        let mut rng = rand::thread_rng();
        match rng.gen_range(0..6) {
            0 => FishColor::Red,
            1 => FishColor::Blue,
            2 => FishColor::Green,
            3 => FishColor::Yellow,
            4 => FishColor::Purple,
            _ => FishColor::Brown,
        }
    }

    pub fn to_rgb(&self) -> [f32; 3] {
        match self {
            FishColor::Red => [0.9, 0.2, 0.15],
            FishColor::Blue => [0.2, 0.4, 0.9],
            FishColor::Green => [0.2, 0.8, 0.3],
            FishColor::Yellow => [0.95, 0.85, 0.2],
            FishColor::Purple => [0.6, 0.2, 0.8],
            FishColor::Brown => [0.55, 0.35, 0.2],
        }
    }

    /// Slightly darker belly color
    pub fn belly_rgb(&self) -> [f32; 3] {
        let base = self.to_rgb();
        [
            (base[0] * 0.7 + 0.3).min(1.0),
            (base[1] * 0.7 + 0.3).min(1.0),
            (base[2] * 0.7 + 0.3).min(1.0),
        ]
    }
}

// ============================================================================
// Fish Struct
// ============================================================================

#[derive(Debug, Clone)]
pub struct Fish {
    pub position: Point3<f32>,
    pub velocity: Vector3<f32>,
    pub direction: f32,              // Yaw angle in radians (heading)
    pub pitch: f32,                  // Pitch angle (up/down tilt)
    pub target_direction: f32,       // Target yaw for smooth turning
    pub target_pitch: f32,           // Target pitch for smooth vertical movement
    pub school_id: Option<usize>,    // ID of school this fish belongs to
    pub wiggle_phase: f32,           // Current phase of wiggle animation
    pub speed_modifier: f32,         // 0.8 - 1.2 multiplier for this fish
    pub size_modifier: f32,          // 0.8 - 1.2 multiplier for visual size
    pub color: FishColor,            // Fish color
    pub random_timer: f32,           // Timer for random direction changes
}

impl Fish {
    pub fn new(position: Point3<f32>) -> Self {
        let mut rng = rand::thread_rng();
        let initial_dir = rng.gen_range(0.0..std::f32::consts::TAU);
        Self {
            position,
            velocity: Vector3::new(0.0, 0.0, 0.0),
            direction: initial_dir,
            pitch: 0.0,
            target_direction: initial_dir,
            target_pitch: 0.0,
            school_id: None,
            wiggle_phase: rng.gen_range(0.0..std::f32::consts::TAU),
            speed_modifier: rng.gen_range(0.8..1.2),
            size_modifier: rng.gen_range(0.8..1.2),
            color: FishColor::random(),
            random_timer: rng.gen_range(1.0..3.0),
        }
    }

    pub fn update(&mut self, dt: f32, world: &World, boid_force: Vector3<f32>) {
        let mut rng = rand::thread_rng();

        // Check if still in water
        let current_block = world.get_block_world(
            self.position.x as i32,
            self.position.y as i32,
            self.position.z as i32,
        );

        if current_block != BlockType::Water {
            // Try to find water nearby
            self.target_pitch = -0.5; // Swim down to find water
        }

        // Random direction changes
        self.random_timer -= dt;
        if self.random_timer <= 0.0 {
            self.target_direction += rng.gen_range(-FISH_SWIM_RANDOMNESS * 2.0..FISH_SWIM_RANDOMNESS * 2.0);
            self.target_pitch = rng.gen_range(-0.3..0.3);
            self.random_timer = rng.gen_range(1.5..4.0);
        }

        // Apply boid forces to target direction
        if boid_force.magnitude() > 0.01 {
            let target_yaw = boid_force.z.atan2(boid_force.x);
            let mut yaw_diff = target_yaw - self.target_direction;
            while yaw_diff > std::f32::consts::PI { yaw_diff -= std::f32::consts::TAU; }
            while yaw_diff < -std::f32::consts::PI { yaw_diff += std::f32::consts::TAU; }
            self.target_direction += yaw_diff * 0.6;

            // Vertical alignment with school
            if boid_force.y.abs() > 0.01 {
                self.target_pitch += boid_force.y * 0.1;
            }
        }

        // Obstacle avoidance
        let avoidance = self.calculate_avoidance(world);
        if avoidance.magnitude() > 0.01 {
            let avoid_yaw = avoidance.z.atan2(avoidance.x);
            self.target_direction = avoid_yaw;
            if avoidance.y.abs() > 0.1 {
                self.target_pitch = avoidance.y.signum() * 0.5;
            }
        }

        // Normalize target direction
        while self.target_direction > std::f32::consts::TAU { self.target_direction -= std::f32::consts::TAU; }
        while self.target_direction < 0.0 { self.target_direction += std::f32::consts::TAU; }

        // Smooth turning toward target direction
        let mut yaw_diff = self.target_direction - self.direction;
        while yaw_diff > std::f32::consts::PI { yaw_diff -= std::f32::consts::TAU; }
        while yaw_diff < -std::f32::consts::PI { yaw_diff += std::f32::consts::TAU; }

        let turn_factor = 1.0 - (-FISH_TURN_RATE * dt).exp();
        self.direction += yaw_diff * turn_factor;

        // Normalize direction
        while self.direction > std::f32::consts::TAU { self.direction -= std::f32::consts::TAU; }
        while self.direction < 0.0 { self.direction += std::f32::consts::TAU; }

        // Smooth pitch changes
        self.target_pitch = self.target_pitch.clamp(-0.6, 0.6);
        let pitch_factor = 1.0 - (-FISH_TURN_RATE * dt).exp();
        self.pitch += (self.target_pitch - self.pitch) * pitch_factor;

        // Calculate velocity from direction and pitch
        let speed = FISH_SWIM_SPEED * self.speed_modifier;
        let cos_pitch = self.pitch.cos();
        self.velocity.x = self.direction.cos() * speed * cos_pitch;
        self.velocity.z = self.direction.sin() * speed * cos_pitch;
        self.velocity.y = self.pitch.sin() * FISH_VERTICAL_SPEED;

        // Apply velocity
        self.position += self.velocity * dt;

        // Update wiggle animation - faster when moving faster
        let speed_factor = self.velocity.magnitude() / FISH_SWIM_SPEED;
        self.wiggle_phase += WIGGLE_SPEED * dt * speed_factor.max(0.5);
        if self.wiggle_phase > std::f32::consts::TAU {
            self.wiggle_phase -= std::f32::consts::TAU;
        }
    }

    fn calculate_avoidance(&self, world: &World) -> Vector3<f32> {
        let mut avoidance = Vector3::new(0.0, 0.0, 0.0);

        let forward_x = self.direction.cos();
        let forward_z = self.direction.sin();

        // Check ahead for solid blocks
        for dist in [1.0, 2.0, FISH_LOOK_AHEAD_DISTANCE] {
            let check_x = (self.position.x + forward_x * dist) as i32;
            let check_z = (self.position.z + forward_z * dist) as i32;

            for dy in [-1, 0, 1] {
                let check_y = (self.position.y + dy as f32) as i32;
                let block = world.get_block_world(check_x, check_y, check_z);

                // Avoid solid blocks and air (fish want to stay in water)
                if block.is_solid() || block == BlockType::Air {
                    let weight = FISH_AVOIDANCE_STRENGTH / dist;
                    avoidance.x -= forward_x * weight;
                    avoidance.z -= forward_z * weight;

                    if block == BlockType::Air {
                        avoidance.y -= weight; // Swim down if near air
                    } else if dy <= 0 {
                        avoidance.y += weight;
                    } else {
                        avoidance.y -= weight * 0.5;
                    }
                }
            }
        }

        // Check sides
        let right_x = forward_z;
        let right_z = -forward_x;

        for side_mult in [-1.0_f32, 1.0_f32] {
            let check_x = (self.position.x + right_x * side_mult * 1.0) as i32;
            let check_z = (self.position.z + right_z * side_mult * 1.0) as i32;
            let check_y = self.position.y as i32;

            let block = world.get_block_world(check_x, check_y, check_z);
            if block.is_solid() || block == BlockType::Air {
                avoidance.x -= right_x * side_mult * FISH_AVOIDANCE_STRENGTH * 0.5;
                avoidance.z -= right_z * side_mult * FISH_AVOIDANCE_STRENGTH * 0.5;
            }
        }

        avoidance
    }
}

// ============================================================================
// FishManager
// ============================================================================

pub struct FishManager {
    pub fish: Vec<Fish>,
    spawn_timer: f32,
    next_school_id: usize,
}

impl FishManager {
    pub fn new() -> Self {
        Self {
            fish: Vec::new(),
            spawn_timer: 0.0,
            next_school_id: 0,
        }
    }

    pub fn update(&mut self, dt: f32, player_pos: Point3<f32>, world: &World) {
        // Spawn new fish
        self.spawn_timer += dt;
        if self.spawn_timer >= FISH_SPAWN_INTERVAL && self.fish.len() < FISH_MAX_COUNT {
            self.try_spawn_fish(player_pos, world);
            self.spawn_timer = 0.0;
        }

        // Calculate boid forces for all fish
        let boid_forces = self.calculate_boid_forces();

        // Update each fish
        for (i, fish) in self.fish.iter_mut().enumerate() {
            let force = boid_forces.get(i).copied().unwrap_or(Vector3::new(0.0, 0.0, 0.0));
            fish.update(dt, world, force);
        }

        // Despawn far fish or fish out of water for too long
        self.fish.retain(|fish| {
            let dx = fish.position.x - player_pos.x;
            let dz = fish.position.z - player_pos.z;
            let dist = (dx * dx + dz * dz).sqrt();
            dist < FISH_DESPAWN_DISTANCE
        });

        // Update school assignments
        self.update_schools();
    }

    fn calculate_boid_forces(&self) -> Vec<Vector3<f32>> {
        self.fish.iter().enumerate().map(|(i, fish)| {
            let mut separation = Vector3::new(0.0, 0.0, 0.0);
            let mut alignment = Vector3::new(0.0, 0.0, 0.0);
            let mut cohesion = Vector3::new(0.0, 0.0, 0.0);
            let mut sep_count = 0;
            let mut align_count = 0;
            let mut cohesion_count = 0;

            for (j, other) in self.fish.iter().enumerate() {
                if i == j {
                    continue;
                }

                let diff = fish.position - other.position;
                let dist = diff.magnitude();

                // Separation - avoid fish that are too close
                if dist < BOID_SEPARATION_RADIUS && dist > 0.001 {
                    separation += diff.normalize() / dist;
                    sep_count += 1;
                }

                // Only align/cohere with same school
                let same_school = fish.school_id.is_some() && fish.school_id == other.school_id;
                let both_unschooled = fish.school_id.is_none() && other.school_id.is_none();

                if same_school || both_unschooled {
                    // Alignment - match velocity of nearby fish
                    if dist < BOID_ALIGNMENT_RADIUS {
                        alignment += other.velocity;
                        align_count += 1;
                    }

                    // Cohesion - move toward center of nearby fish
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
                    let steer = avg_vel.normalize() - fish.velocity.normalize();
                    force += steer * BOID_ALIGNMENT_WEIGHT;
                }
            }

            if cohesion_count > 0 {
                let center = cohesion / cohesion_count as f32;
                let to_center = center - Vector3::new(fish.position.x, fish.position.y, fish.position.z);
                if to_center.magnitude() > 0.001 {
                    force += to_center.normalize() * BOID_COHESION_WEIGHT;
                }
            }

            force
        }).collect()
    }

    fn try_spawn_fish(&mut self, player_pos: Point3<f32>, world: &World) {
        let mut rng = rand::thread_rng();
        let angle = rng.gen_range(0.0..std::f32::consts::TAU);
        let distance = rng.gen_range(FISH_SPAWN_DISTANCE * 0.5..FISH_SPAWN_DISTANCE);

        let spawn_x = player_pos.x + angle.cos() * distance;
        let spawn_z = player_pos.z + angle.sin() * distance;

        // Search for water blocks in a column
        for y in (1..100).rev() {
            let block = world.get_block_world(spawn_x as i32, y, spawn_z as i32);
            if block == BlockType::Water {
                // Found water, spawn here
                let mut fish = Fish::new(Point3::new(spawn_x, y as f32 + 0.5, spawn_z));

                // High chance to join existing school
                if rng.gen::<f32>() < BOID_SCHOOL_CHANCE {
                    if let Some(school_id) = self.find_nearby_school(fish.position) {
                        fish.school_id = Some(school_id);
                        // Match color with school
                        if let Some(school_fish) = self.fish.iter().find(|f| f.school_id == Some(school_id)) {
                            fish.color = school_fish.color;
                        }
                    } else {
                        // Create new school
                        fish.school_id = Some(self.next_school_id);
                        self.next_school_id += 1;
                    }
                }

                self.fish.push(fish);
                return;
            }
            if block.is_solid() {
                break; // Hit ground before finding water
            }
        }
    }

    fn find_nearby_school(&self, pos: Point3<f32>) -> Option<usize> {
        for fish in &self.fish {
            if let Some(school_id) = fish.school_id {
                let dx = fish.position.x - pos.x;
                let dy = fish.position.y - pos.y;
                let dz = fish.position.z - pos.z;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                if dist < BOID_COHESION_RADIUS * 2.0 {
                    return Some(school_id);
                }
            }
        }
        None
    }

    fn update_schools(&mut self) {
        // Unschooled fish may join nearby schools
        let mut rng = rand::thread_rng();

        for i in 0..self.fish.len() {
            if self.fish[i].school_id.is_none() {
                if rng.gen::<f32>() < 0.02 { // Higher chance than birds
                    let pos = self.fish[i].position;
                    if let Some(school_id) = self.find_nearby_school(pos) {
                        // Also adopt the school's color
                        let school_color = self.fish.iter()
                            .find(|f| f.school_id == Some(school_id))
                            .map(|f| f.color);

                        self.fish[i].school_id = Some(school_id);
                        if let Some(color) = school_color {
                            self.fish[i].color = color;
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Fish Rendering
// ============================================================================

/// Creates all vertices for rendering a fish with wiggle animation
/// Fish segments: head, body, tail
pub fn create_fish_vertices(fish: &Fish) -> Vec<Vertex> {
    let size = FISH_BASE_SIZE * fish.size_modifier;
    let mut vertices = Vec::new();

    let body_color = fish.color.to_rgb();
    let belly_color = fish.color.belly_rgb();
    let eye_color = [0.05, 0.05, 0.05]; // Black
    let fin_color = [
        body_color[0] * 0.7,
        body_color[1] * 0.7,
        body_color[2] * 0.7,
    ];

    let pos = fish.position;
    let yaw = fish.direction;
    let pitch = fish.pitch;

    // Wiggle animation - sine wave propagates from head to tail
    let wiggle_base = fish.wiggle_phase.sin() * WIGGLE_AMPLITUDE;
    let body_wiggle = (fish.wiggle_phase + 1.0).sin() * WIGGLE_AMPLITUDE;
    let tail_wiggle = (fish.wiggle_phase + 2.0).sin() * WIGGLE_AMPLITUDE * TAIL_WIGGLE_MULTIPLIER;

    // === HEAD (front segment, slight wiggle) ===
    let head_yaw = yaw + wiggle_base * 0.3;
    let head_offset = rotate_point_fish(size * 0.25, 0.0, 0.0, yaw, pitch);
    add_fish_segment(
        &mut vertices,
        pos,
        Vector3::new(head_offset[0], head_offset[1], head_offset[2]),
        size * 0.2,   // length
        size * 0.15,  // height
        size * 0.12,  // width
        body_color,
        belly_color,
        head_yaw, pitch,
    );

    // === EYES (on sides of head) ===
    let eye_size = size * 0.04;
    // Left eye
    let left_eye_offset = rotate_point_fish(size * 0.30, size * 0.02, size * 0.08, yaw, pitch);
    add_cube_simple(
        &mut vertices,
        pos,
        Vector3::new(left_eye_offset[0], left_eye_offset[1], left_eye_offset[2]),
        eye_size, eye_size, eye_size * 0.5,
        eye_color,
        yaw, pitch,
    );
    // Right eye
    let right_eye_offset = rotate_point_fish(size * 0.30, size * 0.02, -size * 0.08, yaw, pitch);
    add_cube_simple(
        &mut vertices,
        pos,
        Vector3::new(right_eye_offset[0], right_eye_offset[1], right_eye_offset[2]),
        eye_size, eye_size, eye_size * 0.5,
        eye_color,
        yaw, pitch,
    );

    // === BODY (middle segment, medium wiggle) ===
    let body_yaw = yaw + body_wiggle;
    add_fish_segment(
        &mut vertices,
        pos,
        Vector3::new(0.0, 0.0, 0.0),
        size * 0.3,   // length
        size * 0.18,  // height (tallest part)
        size * 0.14,  // width
        body_color,
        belly_color,
        body_yaw, pitch,
    );

    // === DORSAL FIN (on top of body) ===
    let dorsal_offset = rotate_point_fish(0.0, size * 0.15, 0.0, body_yaw, pitch);
    add_cube_simple(
        &mut vertices,
        pos,
        Vector3::new(dorsal_offset[0], dorsal_offset[1], dorsal_offset[2]),
        size * 0.15, size * 0.1, size * 0.02,
        fin_color,
        body_yaw, pitch,
    );

    // === SIDE FINS (pectoral fins) ===
    // Left fin
    let left_fin_offset = rotate_point_fish(size * 0.05, -size * 0.05, size * 0.1, body_yaw, pitch);
    add_cube_simple(
        &mut vertices,
        pos,
        Vector3::new(left_fin_offset[0], left_fin_offset[1], left_fin_offset[2]),
        size * 0.08, size * 0.02, size * 0.12,
        fin_color,
        body_yaw, pitch,
    );
    // Right fin
    let right_fin_offset = rotate_point_fish(size * 0.05, -size * 0.05, -size * 0.1, body_yaw, pitch);
    add_cube_simple(
        &mut vertices,
        pos,
        Vector3::new(right_fin_offset[0], right_fin_offset[1], right_fin_offset[2]),
        size * 0.08, size * 0.02, size * 0.12,
        fin_color,
        body_yaw, pitch,
    );

    // === TAIL (back segment, strongest wiggle) ===
    let tail_yaw = yaw + tail_wiggle;
    let tail_offset = rotate_point_fish(-size * 0.25, 0.0, 0.0, yaw, pitch);
    add_fish_segment(
        &mut vertices,
        pos,
        Vector3::new(tail_offset[0], tail_offset[1], tail_offset[2]),
        size * 0.15,  // length
        size * 0.1,   // height (tapers)
        size * 0.08,  // width (tapers)
        body_color,
        belly_color,
        tail_yaw, pitch,
    );

    // === TAIL FIN (caudal fin) ===
    let tail_fin_offset = rotate_point_fish(-size * 0.38, 0.0, 0.0, tail_yaw, pitch);
    add_cube_simple(
        &mut vertices,
        pos,
        Vector3::new(tail_fin_offset[0], tail_fin_offset[1], tail_fin_offset[2]),
        size * 0.02, size * 0.18, size * 0.15,
        fin_color,
        tail_yaw, pitch,
    );

    vertices
}

/// Rotate a point with yaw and pitch (fish don't roll)
fn rotate_point_fish(x: f32, y: f32, z: f32, yaw: f32, pitch: f32) -> [f32; 3] {
    let cos_yaw = yaw.cos();
    let sin_yaw = yaw.sin();
    let cos_pitch = pitch.cos();
    let sin_pitch = pitch.sin();

    // Apply pitch (rotation around Z axis)
    let x1 = x * cos_pitch - y * sin_pitch;
    let y1 = x * sin_pitch + y * cos_pitch;
    let z1 = z;

    // Apply yaw (rotation around Y axis)
    [
        x1 * cos_yaw - z1 * sin_yaw,
        y1,
        x1 * sin_yaw + z1 * cos_yaw,
    ]
}

/// Adds a fish body segment (uses different colors for top/bottom)
fn add_fish_segment(
    vertices: &mut Vec<Vertex>,
    center: Point3<f32>,
    offset: Vector3<f32>,
    length: f32,  // X dimension
    height: f32,  // Y dimension
    width: f32,   // Z dimension
    top_color: [f32; 3],
    bottom_color: [f32; 3],
    yaw: f32,
    pitch: f32,
) {
    let half_l = length / 2.0;
    let half_h = height / 2.0;
    let half_w = width / 2.0;

    let cx = center.x + offset.x;
    let cy = center.y + offset.y;
    let cz = center.z + offset.z;

    let local_corners = [
        (-half_l, -half_h, -half_w),
        ( half_l, -half_h, -half_w),
        ( half_l,  half_h, -half_w),
        (-half_l,  half_h, -half_w),
        (-half_l, -half_h,  half_w),
        ( half_l, -half_h,  half_w),
        ( half_l,  half_h,  half_w),
        (-half_l,  half_h,  half_w),
    ];

    let corners: Vec<[f32; 3]> = local_corners.iter().map(|(lx, ly, lz)| {
        let rotated = rotate_point_fish(*lx, *ly, *lz, yaw, pitch);
        [cx + rotated[0], cy + rotated[1], cz + rotated[2]]
    }).collect();

    let light_level = 1.0;
    let alpha = 1.0;
    let uv = [0.0, 0.0];
    let tex_index = TEX_NONE;

    // Front face (+Z)
    let front_normal = rotate_point_fish(0.0, 0.0, 1.0, yaw, pitch);
    vertices.push(Vertex { position: corners[4], color: top_color, normal: front_normal, light_level, alpha, uv, tex_index, ao: 1.0 });
    vertices.push(Vertex { position: corners[5], color: top_color, normal: front_normal, light_level, alpha, uv, tex_index, ao: 1.0 });
    vertices.push(Vertex { position: corners[6], color: top_color, normal: front_normal, light_level, alpha, uv, tex_index, ao: 1.0 });
    vertices.push(Vertex { position: corners[7], color: top_color, normal: front_normal, light_level, alpha, uv, tex_index, ao: 1.0 });

    // Back face (-Z)
    let back_normal = rotate_point_fish(0.0, 0.0, -1.0, yaw, pitch);
    vertices.push(Vertex { position: corners[1], color: top_color, normal: back_normal, light_level, alpha, uv, tex_index, ao: 1.0 });
    vertices.push(Vertex { position: corners[0], color: top_color, normal: back_normal, light_level, alpha, uv, tex_index, ao: 1.0 });
    vertices.push(Vertex { position: corners[3], color: top_color, normal: back_normal, light_level, alpha, uv, tex_index, ao: 1.0 });
    vertices.push(Vertex { position: corners[2], color: top_color, normal: back_normal, light_level, alpha, uv, tex_index, ao: 1.0 });

    // Top face (+Y) - main body color
    let top_normal = rotate_point_fish(0.0, 1.0, 0.0, yaw, pitch);
    vertices.push(Vertex { position: corners[7], color: top_color, normal: top_normal, light_level, alpha, uv, tex_index, ao: 1.0 });
    vertices.push(Vertex { position: corners[6], color: top_color, normal: top_normal, light_level, alpha, uv, tex_index, ao: 1.0 });
    vertices.push(Vertex { position: corners[2], color: top_color, normal: top_normal, light_level, alpha, uv, tex_index, ao: 1.0 });
    vertices.push(Vertex { position: corners[3], color: top_color, normal: top_normal, light_level, alpha, uv, tex_index, ao: 1.0 });

    // Bottom face (-Y) - lighter belly color
    let bottom_normal = rotate_point_fish(0.0, -1.0, 0.0, yaw, pitch);
    vertices.push(Vertex { position: corners[0], color: bottom_color, normal: bottom_normal, light_level, alpha, uv, tex_index, ao: 1.0 });
    vertices.push(Vertex { position: corners[1], color: bottom_color, normal: bottom_normal, light_level, alpha, uv, tex_index, ao: 1.0 });
    vertices.push(Vertex { position: corners[5], color: bottom_color, normal: bottom_normal, light_level, alpha, uv, tex_index, ao: 1.0 });
    vertices.push(Vertex { position: corners[4], color: bottom_color, normal: bottom_normal, light_level, alpha, uv, tex_index, ao: 1.0 });

    // Right face (+X)
    let right_normal = rotate_point_fish(1.0, 0.0, 0.0, yaw, pitch);
    vertices.push(Vertex { position: corners[5], color: top_color, normal: right_normal, light_level, alpha, uv, tex_index, ao: 1.0 });
    vertices.push(Vertex { position: corners[1], color: top_color, normal: right_normal, light_level, alpha, uv, tex_index, ao: 1.0 });
    vertices.push(Vertex { position: corners[2], color: top_color, normal: right_normal, light_level, alpha, uv, tex_index, ao: 1.0 });
    vertices.push(Vertex { position: corners[6], color: top_color, normal: right_normal, light_level, alpha, uv, tex_index, ao: 1.0 });

    // Left face (-X)
    let left_normal = rotate_point_fish(-1.0, 0.0, 0.0, yaw, pitch);
    vertices.push(Vertex { position: corners[0], color: top_color, normal: left_normal, light_level, alpha, uv, tex_index, ao: 1.0 });
    vertices.push(Vertex { position: corners[4], color: top_color, normal: left_normal, light_level, alpha, uv, tex_index, ao: 1.0 });
    vertices.push(Vertex { position: corners[7], color: top_color, normal: left_normal, light_level, alpha, uv, tex_index, ao: 1.0 });
    vertices.push(Vertex { position: corners[3], color: top_color, normal: left_normal, light_level, alpha, uv, tex_index, ao: 1.0 });
}

/// Simple cube for fins, eyes, etc
fn add_cube_simple(
    vertices: &mut Vec<Vertex>,
    center: Point3<f32>,
    offset: Vector3<f32>,
    length: f32,
    height: f32,
    width: f32,
    color: [f32; 3],
    yaw: f32,
    pitch: f32,
) {
    add_fish_segment(vertices, center, offset, length, height, width, color, color, yaw, pitch);
}

/// Generate indices for N cubes (each cube has 24 vertices, 36 indices)
pub fn generate_fish_indices(num_cubes: usize) -> Vec<u16> {
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
