use cgmath::*;

pub struct Camera {
    pub position: Point3<f32>,
    pub yaw: Rad<f32>,
    pub pitch: Rad<f32>,
}

impl Camera {
    pub fn new(position: Point3<f32>) -> Self {
        Self {
            position,
            yaw: Rad(0.0),
            pitch: Rad(0.0),
        }
    }

    pub fn get_view_matrix(&self) -> Matrix4<f32> {
        let direction = self.get_direction();
        Matrix4::look_to_rh(self.position, direction, Vector3::unit_y())
    }

    pub fn get_direction(&self) -> Vector3<f32> {
        Vector3::new(
            self.yaw.0.cos() * self.pitch.0.cos(),
            self.pitch.0.sin(),
            self.yaw.0.sin() * self.pitch.0.cos(),
        )
        .normalize()
    }

    pub fn get_horizontal_direction(&self) -> Vector3<f32> {
        Vector3::new(self.yaw.0.cos(), 0.0, self.yaw.0.sin()).normalize()
    }
}

pub struct CameraController {
    pub speed: f32,
    pub sensitivity: f32,
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub jump_held: bool,
    pub velocity: Vector3<f32>,
    pub on_ground: bool,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            speed,
            sensitivity,
            forward: false,
            backward: false,
            left: false,
            right: false,
            jump_held: false,
            velocity: Vector3::new(0.0, 0.0, 0.0),
            on_ground: false,
        }
    }

    pub fn process_mouse(&mut self, camera: &mut Camera, dx: f32, dy: f32) {
        camera.yaw += Rad(dx * self.sensitivity);
        camera.pitch -= Rad(dy * self.sensitivity);

        // Clamp pitch to prevent camera flipping
        let max_pitch = std::f32::consts::FRAC_PI_2 - 0.1;
        camera.pitch.0 = camera.pitch.0.clamp(-max_pitch, max_pitch);
    }

    pub fn update_camera(&mut self, camera: &mut Camera, dt: f32, world: &crate::world::World) {
        let gravity = -20.0;
        let jump_velocity = 8.0;

        // Get movement direction (horizontal only for WASD)
        let forward = camera.get_horizontal_direction();
        let right = forward.cross(Vector3::unit_y()).normalize();

        // Calculate desired horizontal movement
        let mut move_dir = Vector3::new(0.0, 0.0, 0.0);
        if self.forward {
            move_dir += forward;
        }
        if self.backward {
            move_dir -= forward;
        }
        if self.left {
            move_dir -= right;
        }
        if self.right {
            move_dir += right;
        }

        // Normalize and apply speed
        if move_dir.magnitude2() > 0.0 {
            move_dir = move_dir.normalize() * self.speed;
        }

        // Set horizontal velocity directly for responsive controls
        self.velocity.x = move_dir.x;
        self.velocity.z = move_dir.z;

        // Apply gravity
        self.velocity.y += gravity * dt;

        // Jump if on ground and space is held
        if self.jump_held && self.on_ground {
            self.velocity.y = jump_velocity;
            self.on_ground = false;
        }

        // Calculate new position
        let mut new_pos = camera.position;
        new_pos.x += self.velocity.x * dt;
        new_pos.y += self.velocity.y * dt;
        new_pos.z += self.velocity.z * dt;

        // Simple collision detection - check feet position
        let player_height = 1.6; // Player is ~1.6 blocks tall, camera at eye level
        let feet_y = new_pos.y - player_height;
        let old_feet_y = camera.position.y - player_height;

        // Check multiple positions for player width to prevent corner clipping
        let player_radius = 0.25;
        let check_positions = [
            (new_pos.x, new_pos.z),
            (new_pos.x + player_radius, new_pos.z),
            (new_pos.x - player_radius, new_pos.z),
            (new_pos.x, new_pos.z + player_radius),
            (new_pos.x, new_pos.z - player_radius),
        ];

        // Ceiling collision - check when jumping upward
        if self.velocity.y > 0.0 {
            let head_y = new_pos.y; // Camera is at eye level, check slightly above
            let old_head_y = camera.position.y;

            for (check_x, check_z) in check_positions.iter() {
                let block_y = head_y.floor() as i32;
                let block = world.get_block_world(
                    check_x.floor() as i32,
                    block_y,
                    check_z.floor() as i32,
                );

                if block.is_solid() {
                    let block_bottom = block_y as f32;
                    // If head would enter this block and we were below it before
                    if head_y >= block_bottom && old_head_y < block_bottom + 0.01 {
                        new_pos.y = block_bottom;
                        self.velocity.y = 0.0;
                        break;
                    }
                }
            }
        }

        // Ground collision - only when falling, check if we're crossing into a block from above
        let feet_y = new_pos.y - player_height;
        let mut highest_ground = f32::NEG_INFINITY;
        let mut found_ground = false;

        if self.velocity.y <= 0.0 {
            for (check_x, check_z) in check_positions.iter() {
                // Check the block at feet level
                let block_y = feet_y.floor() as i32;
                let block = world.get_block_world(
                    check_x.floor() as i32,
                    block_y,
                    check_z.floor() as i32,
                );

                if block.is_solid() {
                    let block_top = (block_y + 1) as f32;
                    // Only land if we're actually penetrating the block AND we were above it before
                    if feet_y < block_top && old_feet_y >= block_top - 0.01 {
                        highest_ground = highest_ground.max(block_top);
                        found_ground = true;
                    }
                }
            }
        }

        // Apply ground collision
        if found_ground {
            new_pos.y = highest_ground + player_height;
            self.velocity.y = 0.0;
            self.on_ground = true;
        } else {
            // Check if we're standing on ground (for when not moving vertically)
            let standing_feet_y = new_pos.y - player_height - 0.05; // Small offset below feet
            let mut standing_on_ground = false;
            for (check_x, check_z) in check_positions.iter() {
                let block = world.get_block_world(
                    check_x.floor() as i32,
                    standing_feet_y.floor() as i32,
                    check_z.floor() as i32,
                );
                if block.is_solid() {
                    let block_top = (standing_feet_y.floor() as i32 + 1) as f32;
                    // Check if we're very close to standing on this block
                    if (new_pos.y - player_height - block_top).abs() < 0.05 {
                        standing_on_ground = true;
                        break;
                    }
                }
            }
            self.on_ground = standing_on_ground;
        }

        // Horizontal collision - check at multiple heights (feet, body, head)
        let feet_check_y = new_pos.y - player_height + 0.1; // Just above feet
        let body_y = new_pos.y - 1.0;
        let head_y = new_pos.y - 0.1; // Just below eye level

        // Check X collision
        let check_x = if self.velocity.x > 0.0 {
            new_pos.x + player_radius
        } else {
            new_pos.x - player_radius
        };
        let block_x1 = world.get_block_world(check_x.floor() as i32, head_y.floor() as i32, new_pos.z.floor() as i32);
        let block_x2 = world.get_block_world(check_x.floor() as i32, body_y.floor() as i32, new_pos.z.floor() as i32);
        let block_x3 = world.get_block_world(check_x.floor() as i32, feet_check_y.floor() as i32, new_pos.z.floor() as i32);
        if block_x1.is_solid() || block_x2.is_solid() || block_x3.is_solid() {
            new_pos.x = camera.position.x;
            self.velocity.x = 0.0;
        }

        // Check Z collision
        let check_z = if self.velocity.z > 0.0 {
            new_pos.z + player_radius
        } else {
            new_pos.z - player_radius
        };
        let block_z1 = world.get_block_world(new_pos.x.floor() as i32, head_y.floor() as i32, check_z.floor() as i32);
        let block_z2 = world.get_block_world(new_pos.x.floor() as i32, body_y.floor() as i32, check_z.floor() as i32);
        let block_z3 = world.get_block_world(new_pos.x.floor() as i32, feet_check_y.floor() as i32, check_z.floor() as i32);
        if block_z1.is_solid() || block_z2.is_solid() || block_z3.is_solid() {
            new_pos.z = camera.position.z;
            self.velocity.z = 0.0;
        }

        camera.position = new_pos;
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    view_position: [f32; 4],
    near: f32,
    far: f32,
    _padding: [f32; 2],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Matrix4::identity().into(),
            view_position: [0.0; 4],
            near: 0.1,
            far: 1000.0,
            _padding: [0.0; 2],
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera, projection: &Projection) {
        self.view_position = [camera.position.x, camera.position.y, camera.position.z, 1.0];
        self.view_proj = (projection.calc_matrix() * camera.get_view_matrix()).into();
        self.near = projection.znear;
        self.far = projection.zfar;
    }
}

pub struct Projection {
    aspect: f32,
    fovy: Rad<f32>,
    pub znear: f32,
    pub zfar: f32,
}

impl Projection {
    pub fn new(width: u32, height: u32, fovy: Rad<f32>, znear: f32, zfar: f32) -> Self {
        Self {
            aspect: width as f32 / height as f32,
            fovy,
            znear,
            zfar,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    pub fn calc_matrix(&self) -> Matrix4<f32> {
        perspective(self.fovy, self.aspect, self.znear, self.zfar)
    }
}
