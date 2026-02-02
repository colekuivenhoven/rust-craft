use crate::block::BlockType;
use crate::inventory::Inventory;
use crate::world::World;
use cgmath::{Point3, Vector3, InnerSpace};

pub struct Player {
    pub position: Point3<f32>,
    pub inventory: Inventory,
    pub health: f32,
    pub max_health: f32,
    pub reach_distance: f32,
}

impl Player {
    pub fn new(position: Point3<f32>) -> Self {
        let mut inventory = Inventory::new(9);

        // Give player starting items
        inventory.add_item(BlockType::Wood, 64);
        inventory.add_item(BlockType::Stone, 64);
        inventory.add_item(BlockType::Dirt, 64);
        inventory.add_item(BlockType::GlowStone, 64);
        inventory.add_item(BlockType::Water, 64);
        inventory.add_item(BlockType::Ice, 64);
        inventory.add_item(BlockType::Snow, 64);

        Self {
            position,
            inventory,
            health: 100.0,
            max_health: 100.0,
            reach_distance: 5.0,
        }
    }

    pub fn take_damage(&mut self, damage: f32) {
        self.health -= damage;
        if self.health < 0.0 {
            self.health = 0.0;
        }
    }

    pub fn heal(&mut self, amount: f32) {
        self.health = (self.health + amount).min(self.max_health);
    }

    pub fn is_alive(&self) -> bool {
        self.health > 0.0
    }

    /// Raycast using DDA algorithm to find the block being looked at and which face was hit.
    /// Returns (block_x, block_y, block_z, face_normal) where face_normal points outward from the hit face.
    pub fn raycast_block(
        &self,
        direction: Vector3<f32>,
        world: &World,
    ) -> Option<(i32, i32, i32, Vector3<i32>)> {
        // DDA (Digital Differential Analyzer) raycasting algorithm
        // This properly tracks which face boundary is crossed

        let origin = self.position;
        let dir = direction.normalize();

        // Current voxel coordinates
        let mut x = origin.x.floor() as i32;
        let mut y = origin.y.floor() as i32;
        let mut z = origin.z.floor() as i32;

        // Direction to step in each axis (+1 or -1)
        let step_x = if dir.x >= 0.0 { 1 } else { -1 };
        let step_y = if dir.y >= 0.0 { 1 } else { -1 };
        let step_z = if dir.z >= 0.0 { 1 } else { -1 };

        // How far along the ray we must move for each component to cross a voxel boundary
        // (in units of t along the ray)
        let t_delta_x = if dir.x != 0.0 { (1.0 / dir.x).abs() } else { f32::MAX };
        let t_delta_y = if dir.y != 0.0 { (1.0 / dir.y).abs() } else { f32::MAX };
        let t_delta_z = if dir.z != 0.0 { (1.0 / dir.z).abs() } else { f32::MAX };

        // Distance to the next voxel boundary for each axis
        let mut t_max_x = if dir.x != 0.0 {
            if dir.x > 0.0 {
                ((x + 1) as f32 - origin.x) / dir.x
            } else {
                (x as f32 - origin.x) / dir.x
            }
        } else {
            f32::MAX
        };

        let mut t_max_y = if dir.y != 0.0 {
            if dir.y > 0.0 {
                ((y + 1) as f32 - origin.y) / dir.y
            } else {
                (y as f32 - origin.y) / dir.y
            }
        } else {
            f32::MAX
        };

        let mut t_max_z = if dir.z != 0.0 {
            if dir.z > 0.0 {
                ((z + 1) as f32 - origin.z) / dir.z
            } else {
                (z as f32 - origin.z) / dir.z
            }
        } else {
            f32::MAX
        };

        // Track which face we entered from (the normal points back toward the player)
        let mut face_normal = Vector3::new(0, 0, 0);

        // Maximum distance to search
        let max_distance = self.reach_distance;
        let mut distance = 0.0;

        // Check starting block first
        if world.get_block_world(x, y, z).is_solid() {
            // We're inside a solid block, return it with no specific face
            return Some((x, y, z, Vector3::new(0, 1, 0)));
        }

        while distance < max_distance {
            // Step to the next voxel boundary (whichever axis is closest)
            if t_max_x < t_max_y && t_max_x < t_max_z {
                // X boundary is closest
                distance = t_max_x;
                t_max_x += t_delta_x;
                x += step_x;
                // Face normal points opposite to our travel direction
                face_normal = Vector3::new(-step_x, 0, 0);
            } else if t_max_y < t_max_z {
                // Y boundary is closest
                distance = t_max_y;
                t_max_y += t_delta_y;
                y += step_y;
                face_normal = Vector3::new(0, -step_y, 0);
            } else {
                // Z boundary is closest
                distance = t_max_z;
                t_max_z += t_delta_z;
                z += step_z;
                face_normal = Vector3::new(0, 0, -step_z);
            }

            if distance > max_distance {
                break;
            }

            // Check if this voxel is solid
            if world.get_block_world(x, y, z).is_solid() {
                return Some((x, y, z, face_normal));
            }
        }

        None
    }
}
