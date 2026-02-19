use crate::block::BlockType;
use crate::world::World;
use cgmath::{Point3, Vector3};
use rand::Rng;

// Physics constants
const GRAVITY: f32 = -15.0;
const POP_VELOCITY_MIN: f32 = 2.0;
const POP_VELOCITY_MAX: f32 = 5.0;
const POP_VELOCITY_UP_MIN: f32 = 3.0;
const POP_VELOCITY_UP_MAX: f32 = 5.0;
const GROUND_FRICTION: f32 = 0.9;
const ITEM_SIZE: f32 = 0.25; // visual size: 1/4 of a block
const ITEM_VALUE: f32 = 0.25;
const DESPAWN_TIME: f32 = 300.0; // 5 minutes

/// Hitbox half-extent used for raycast picking (larger than visual to aid clicking)
const PICKUP_HALF: f32 = 0.25;

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

/// Data returned when an item is picked up
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

    /// Spawn 4 mini-blocks when a block is broken.
    pub fn spawn_drops(&mut self, block_pos: Point3<f32>, block_type: BlockType) {
        let mut rng = rand::thread_rng();

        let center = Point3::new(
            block_pos.x + 0.5,
            block_pos.y + 0.5,
            block_pos.z + 0.5,
        );

        for _ in 0..4 {
            let angle = rng.gen_range(0.0..std::f32::consts::TAU);
            let h_speed = rng.gen_range(POP_VELOCITY_MIN..POP_VELOCITY_MAX);
            let v_speed = rng.gen_range(POP_VELOCITY_UP_MIN..POP_VELOCITY_UP_MAX);

            let velocity = Vector3::new(
                angle.cos() * h_speed,
                v_speed,
                angle.sin() * h_speed,
            );
            let offset = Vector3::new(
                rng.gen_range(-0.2..0.2),
                rng.gen_range(-0.1..0.1),
                rng.gen_range(-0.2..0.2),
            );
            self.items.push(DroppedItem::new(
                center + offset,
                velocity,
                block_type,
            ));
        }
    }

    /// Advance physics for all items (gravity, collision, lifetime).
    /// No automatic collection — the player must right-click.
    pub fn update(&mut self, dt: f32, world: &World) {
        self.items.retain_mut(|item| {
            item.lifetime -= dt;
            if item.lifetime <= 0.0 {
                return false;
            }

            // Gravity
            if !item.on_ground {
                item.velocity.y += GRAVITY * dt;
            }

            let new_pos = Point3::new(
                item.position.x + item.velocity.x * dt,
                item.position.y + item.velocity.y * dt,
                item.position.z + item.velocity.z * dt,
            );

            let half_size = ITEM_SIZE * 0.5;

            // Ground collision (Y)
            let block_below = world.get_block_world(
                new_pos.x.floor() as i32,
                (new_pos.y - half_size).floor() as i32,
                new_pos.z.floor() as i32,
            );
            if block_below.is_solid() && item.velocity.y < 0.0 {
                let ground_y = (new_pos.y - half_size).floor() + 1.0 + half_size;
                item.position.y = ground_y;
                item.on_ground = true;
                item.velocity.y = 0.0;
                item.velocity.x *= GROUND_FRICTION;
                item.velocity.z *= GROUND_FRICTION;
                if item.velocity.x.abs() < 0.01 { item.velocity.x = 0.0; }
                if item.velocity.z.abs() < 0.01 { item.velocity.z = 0.0; }
            } else {
                item.on_ground = false;
                item.position.y = new_pos.y;
            }

            // Horizontal collision X
            let block_x = world.get_block_world(
                (new_pos.x + half_size * item.velocity.x.signum()).floor() as i32,
                item.position.y.floor() as i32,
                item.position.z.floor() as i32,
            );
            if block_x.is_solid() {
                item.velocity.x = -item.velocity.x * 0.3;
            } else {
                item.position.x = new_pos.x;
            }

            // Horizontal collision Z
            let block_z = world.get_block_world(
                item.position.x.floor() as i32,
                item.position.y.floor() as i32,
                (new_pos.z + half_size * item.velocity.z.signum()).floor() as i32,
            );
            if block_z.is_solid() {
                item.velocity.z = -item.velocity.z * 0.3;
            } else {
                item.position.z = new_pos.z;
            }

            true
        });
    }

    /// Returns the index of the closest dropped item whose pickup hitbox the
    /// ray (origin + t*direction) intersects, within `max_dist`.
    /// Returns `None` if no item is in the crosshair.
    pub fn raycast_item(
        &self,
        origin: Point3<f32>,
        direction: Vector3<f32>,
        max_dist: f32,
    ) -> Option<usize> {
        let mut best_t = max_dist;
        let mut best_idx = None;

        for (i, item) in self.items.iter().enumerate() {
            if let Some(t) = ray_aabb_hit(origin, direction, item.position, PICKUP_HALF) {
                if t < best_t {
                    best_t = t;
                    best_idx = Some(i);
                }
            }
        }

        best_idx
    }

    /// Remove the item at `index` and return its data for adding to inventory.
    /// Panics if index is out of bounds — always guard with a bounds check.
    pub fn collect_item(&mut self, index: usize) -> CollectedItem {
        let item = self.items.remove(index);
        CollectedItem {
            block_type: item.block_type,
            value: item.value,
        }
    }
}

impl Default for DroppedItemManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Ray–AABB intersection helper ─────────────────────────────────────────────
// Returns the entry t-value along the ray, or None if the ray misses the box.
fn ray_aabb_hit(
    origin: Point3<f32>,
    dir: Vector3<f32>,
    center: Point3<f32>,
    half: f32,
) -> Option<f32> {
    let inv_x = if dir.x != 0.0 { 1.0 / dir.x } else { f32::MAX };
    let inv_y = if dir.y != 0.0 { 1.0 / dir.y } else { f32::MAX };
    let inv_z = if dir.z != 0.0 { 1.0 / dir.z } else { f32::MAX };

    let t1 = (center.x - half - origin.x) * inv_x;
    let t2 = (center.x + half - origin.x) * inv_x;
    let t3 = (center.y - half - origin.y) * inv_y;
    let t4 = (center.y + half - origin.y) * inv_y;
    let t5 = (center.z - half - origin.z) * inv_z;
    let t6 = (center.z + half - origin.z) * inv_z;

    let tmin = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
    let tmax = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

    if tmax < 0.0 || tmin > tmax {
        None
    } else {
        Some(if tmin < 0.0 { tmax } else { tmin })
    }
}
