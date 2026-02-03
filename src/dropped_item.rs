use crate::block::BlockType;
use crate::world::World;
use cgmath::{InnerSpace, Point3, Vector3};
use rand::Rng;

// Physics constants
const GRAVITY: f32 = -15.0;
const POP_VELOCITY_MIN: f32 = 2.0;
const POP_VELOCITY_MAX: f32 = 5.0;
const POP_VELOCITY_UP_MIN: f32 = 3.0;
const POP_VELOCITY_UP_MAX: f32 = 5.0;
const MAGNETIC_RANGE: f32 = 4.0;
const MAGNETIC_FORCE: f32 = 20.0;
const COLLECTION_RANGE: f32 = 1.5;
const GROUND_FRICTION: f32 = 0.9;
const ITEM_SIZE: f32 = 0.25; // 1/4 of a block
const ITEM_VALUE: f32 = 0.25; // Each mini-block is worth 0.25
const DESPAWN_TIME: f32 = 300.0; // 5 minutes

#[derive(Debug, Clone)]
pub struct DroppedItem {
    pub position: Point3<f32>,
    pub velocity: Vector3<f32>,
    pub block_type: BlockType,
    pub value: f32,
    pub lifetime: f32,
    pub on_ground: bool,
}

impl DroppedItem {
    pub fn new(position: Point3<f32>, velocity: Vector3<f32>, block_type: BlockType) -> Self {
        Self {
            position,
            velocity,
            block_type,
            value: ITEM_VALUE,
            lifetime: DESPAWN_TIME,
            on_ground: false,
        }
    }

    pub fn get_size(&self) -> f32 {
        ITEM_SIZE
    }
}

/// Represents a collected item with its data
pub struct CollectedItem {
    pub block_type: BlockType,
    pub value: f32,
}

pub struct DroppedItemManager {
    pub items: Vec<DroppedItem>,
}

impl DroppedItemManager {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Spawn 4 mini-blocks when a block is broken
    pub fn spawn_drops(&mut self, block_pos: Point3<f32>, block_type: BlockType) {
        let mut rng = rand::thread_rng();

        // Center of the block
        let center = Point3::new(
            block_pos.x + 0.5,
            block_pos.y + 0.5,
            block_pos.z + 0.5,
        );

        // Spawn 4 mini-blocks
        for _ in 0..4 {
            // Random horizontal angle
            let angle = rng.gen_range(0.0..std::f32::consts::TAU);
            let horizontal_speed = rng.gen_range(POP_VELOCITY_MIN..POP_VELOCITY_MAX);
            let vertical_speed = rng.gen_range(POP_VELOCITY_UP_MIN..POP_VELOCITY_UP_MAX);

            let velocity = Vector3::new(
                angle.cos() * horizontal_speed,
                vertical_speed,
                angle.sin() * horizontal_speed,
            );

            // Slight random offset from center
            let offset = Vector3::new(
                rng.gen_range(-0.2..0.2),
                rng.gen_range(-0.1..0.1),
                rng.gen_range(-0.2..0.2),
            );

            let spawn_pos = Point3::new(
                center.x + offset.x,
                center.y + offset.y,
                center.z + offset.z,
            );

            self.items.push(DroppedItem::new(spawn_pos, velocity, block_type));
        }
    }

    /// Update all dropped items and return collected items
    /// This handles physics, collection, and expiration internally
    pub fn update(&mut self, dt: f32, player_pos: Point3<f32>, world: &World) -> Vec<CollectedItem> {
        let mut collected = Vec::new();

        // Use retain to filter items, collecting those that touch the player
        self.items.retain_mut(|item| {
            // Update lifetime
            item.lifetime -= dt;
            if item.lifetime <= 0.0 {
                // Item expired, remove it
                return false;
            }

            // Calculate distance to player
            let to_player = player_pos - item.position;
            let dist_to_player = to_player.magnitude();

            // Check for collection
            if dist_to_player < COLLECTION_RANGE {
                // Collect this item
                collected.push(CollectedItem {
                    block_type: item.block_type,
                    value: item.value,
                });
                return false; // Remove from items
            }

            // Magnetic attraction when close
            if dist_to_player < MAGNETIC_RANGE && dist_to_player > 0.1 {
                let dir = to_player.normalize();
                item.velocity += dir * MAGNETIC_FORCE * dt;
            }

            // Gravity (only if not on ground)
            if !item.on_ground {
                item.velocity.y += GRAVITY * dt;
            }

            // Apply velocity
            let new_pos = Point3::new(
                item.position.x + item.velocity.x * dt,
                item.position.y + item.velocity.y * dt,
                item.position.z + item.velocity.z * dt,
            );

            // Collision detection
            let half_size = ITEM_SIZE * 0.5;

            // Check ground collision
            let block_below = world.get_block_world(
                new_pos.x.floor() as i32,
                (new_pos.y - half_size).floor() as i32,
                new_pos.z.floor() as i32,
            );

            if block_below.is_solid() && item.velocity.y < 0.0 {
                // Land on top of block
                let ground_y = (new_pos.y - half_size).floor() + 1.0 + half_size;
                item.position.y = ground_y;
                item.on_ground = true;
                item.velocity.y = 0.0;
                // Apply friction
                item.velocity.x *= GROUND_FRICTION;
                item.velocity.z *= GROUND_FRICTION;
                // Stop very slow movement
                if item.velocity.x.abs() < 0.01 {
                    item.velocity.x = 0.0;
                }
                if item.velocity.z.abs() < 0.01 {
                    item.velocity.z = 0.0;
                }
            } else {
                item.on_ground = false;
                item.position.y = new_pos.y;
            }

            // Horizontal collision - X axis
            let block_x = world.get_block_world(
                (new_pos.x + half_size * item.velocity.x.signum()).floor() as i32,
                item.position.y.floor() as i32,
                item.position.z.floor() as i32,
            );
            if block_x.is_solid() {
                item.velocity.x = -item.velocity.x * 0.3; // Bounce slightly
            } else {
                item.position.x = new_pos.x;
            }

            // Horizontal collision - Z axis
            let block_z = world.get_block_world(
                item.position.x.floor() as i32,
                item.position.y.floor() as i32,
                (new_pos.z + half_size * item.velocity.z.signum()).floor() as i32,
            );
            if block_z.is_solid() {
                item.velocity.z = -item.velocity.z * 0.3; // Bounce slightly
            } else {
                item.position.z = new_pos.z;
            }

            true // Keep this item
        });

        collected
    }
}

impl Default for DroppedItemManager {
    fn default() -> Self {
        Self::new()
    }
}
