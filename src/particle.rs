use crate::block::BlockType;
use cgmath::{Point3, Vector3};
use rand::Rng;

// Particle constants
const PARTICLE_COUNT: usize = 25;
const PARTICLE_GRAVITY: f32 = -12.0;
const PARTICLE_SIZE_MIN: f32 = 0.05;
const PARTICLE_SIZE_MAX: f32 = 0.1;
const PARTICLE_LIFETIME_MIN: f32 = 0.3;
const PARTICLE_LIFETIME_MAX: f32 = 0.8;
const PARTICLE_VELOCITY_HORIZONTAL: f32 = 4.0;
const PARTICLE_VELOCITY_UP_MIN: f32 = 2.0;
const PARTICLE_VELOCITY_UP_MAX: f32 = 6.0;

#[derive(Debug, Clone)]
pub struct Particle {
    pub position: Point3<f32>,
    pub velocity: Vector3<f32>,
    pub color: [f32; 3],
    pub lifetime: f32,
    pub max_lifetime: f32,
    pub size: f32,
}

impl Particle {
    pub fn new(
        position: Point3<f32>,
        velocity: Vector3<f32>,
        color: [f32; 3],
        lifetime: f32,
        size: f32,
    ) -> Self {
        Self {
            position,
            velocity,
            color,
            lifetime,
            max_lifetime: lifetime,
            size,
        }
    }

    /// Get the current alpha based on remaining lifetime (fades out)
    pub fn get_alpha(&self) -> f32 {
        (self.lifetime / self.max_lifetime).max(0.0).min(1.0)
    }

    /// Check if the particle should be removed
    pub fn is_expired(&self) -> bool {
        self.lifetime <= 0.0
    }
}

pub struct ParticleManager {
    pub particles: Vec<Particle>,
}

impl ParticleManager {
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
        }
    }

    /// Spawn particles when a block is broken
    pub fn spawn_block_break(&mut self, block_pos: Point3<f32>, block_type: BlockType) {
        let color = block_type.get_color();
        let mut rng = rand::thread_rng();

        // Center of the block
        let center = Point3::new(
            block_pos.x + 0.5,
            block_pos.y + 0.5,
            block_pos.z + 0.5,
        );

        for _ in 0..PARTICLE_COUNT {
            // Random velocity in all directions, biased upward
            let velocity = Vector3::new(
                rng.gen_range(-PARTICLE_VELOCITY_HORIZONTAL..PARTICLE_VELOCITY_HORIZONTAL),
                rng.gen_range(PARTICLE_VELOCITY_UP_MIN..PARTICLE_VELOCITY_UP_MAX),
                rng.gen_range(-PARTICLE_VELOCITY_HORIZONTAL..PARTICLE_VELOCITY_HORIZONTAL),
            );

            // Random offset from center
            let offset = Vector3::new(
                rng.gen_range(-0.3..0.3),
                rng.gen_range(-0.3..0.3),
                rng.gen_range(-0.3..0.3),
            );

            let spawn_pos = Point3::new(
                center.x + offset.x,
                center.y + offset.y,
                center.z + offset.z,
            );

            let lifetime = rng.gen_range(PARTICLE_LIFETIME_MIN..PARTICLE_LIFETIME_MAX);
            let size = rng.gen_range(PARTICLE_SIZE_MIN..PARTICLE_SIZE_MAX);

            // Slight color variation for visual interest
            let color_var = rng.gen_range(-0.1..0.1);
            let varied_color = [
                (color[0] + color_var).clamp(0.0, 1.0),
                (color[1] + color_var).clamp(0.0, 1.0),
                (color[2] + color_var).clamp(0.0, 1.0),
            ];

            self.particles.push(Particle::new(
                spawn_pos,
                velocity,
                varied_color,
                lifetime,
                size,
            ));
        }
    }

    /// Update all particles
    pub fn update(&mut self, dt: f32) {
        for particle in &mut self.particles {
            // Update lifetime
            particle.lifetime -= dt;

            // Apply gravity
            particle.velocity.y += PARTICLE_GRAVITY * dt;

            // Apply velocity
            particle.position.x += particle.velocity.x * dt;
            particle.position.y += particle.velocity.y * dt;
            particle.position.z += particle.velocity.z * dt;

            // Shrink as they fade
            let life_ratio = particle.get_alpha();
            particle.size *= 0.99 + (0.01 * life_ratio);
        }

        // Remove expired particles
        self.particles.retain(|p| !p.is_expired());
    }

    /// Get the number of active particles
    pub fn count(&self) -> usize {
        self.particles.len()
    }
}

impl Default for ParticleManager {
    fn default() -> Self {
        Self::new()
    }
}
