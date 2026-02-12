use cgmath::*;

// === SMOOTHING CONSTANTS ===

/// How quickly horizontal movement reaches target speed (higher = snappier). Units: 1/s.
pub const MOVEMENT_ACCEL: f32 = 18.0;
/// How quickly horizontal movement decays to zero when keys are released (higher = snappier). Units: 1/s.
pub const MOVEMENT_DECEL: f32 = 12.0;
/// Mouse look smoothing time constant in seconds (0.0 = instant/no smoothing, higher = smoother).
pub const MOUSE_SMOOTHING: f32 = 0.025;
/// Motion blur intensity (0.0 = disabled, 1.0 = heavy blur).
pub const MOTION_BLUR_AMOUNT: f32 = 0.35;
/// Number of samples for the motion blur shader (higher = smoother blur, more expensive).
pub const MOTION_BLUR_SAMPLES: u32 = 8;

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
    pub shift_held: bool,
    pub velocity: Vector3<f32>,
    pub on_ground: bool,
    pub last_fall_velocity: f32,  // Captures velocity.y at moment of landing (for fall damage)
    // Mouse look smoothing: raw input accumulates into target, camera lerps toward it
    pub target_yaw: f32,
    pub target_pitch: f32,
    // Camera rotation velocity (radians/sec) — used by motion blur
    pub yaw_velocity: f32,
    pub pitch_velocity: f32,
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
            shift_held: false,
            velocity: Vector3::new(0.0, 0.0, 0.0),
            on_ground: false,
            last_fall_velocity: 0.0,
            target_yaw: 0.0,
            target_pitch: 0.0,
            yaw_velocity: 0.0,
            pitch_velocity: 0.0,
        }
    }

    pub fn process_mouse(&mut self, dx: f32, dy: f32) {
        // Accumulate raw mouse input into targets; camera lerps toward these in update_camera
        self.target_yaw += dx * self.sensitivity;
        self.target_pitch -= dy * self.sensitivity;

        let max_pitch = std::f32::consts::FRAC_PI_2 - 0.1;
        self.target_pitch = self.target_pitch.clamp(-max_pitch, max_pitch);
    }

    /// Updates camera position with physics. Returns true if camera is underwater.
    pub fn update_camera(&mut self, camera: &mut Camera, dt: f32, world: &crate::world::World, noclip: bool) -> bool {
        // --- Mouse look smoothing ---
        let prev_yaw = camera.yaw.0;
        let prev_pitch = camera.pitch.0;

        if MOUSE_SMOOTHING <= 0.0 {
            camera.yaw = Rad(self.target_yaw);
            camera.pitch = Rad(self.target_pitch);
        } else {
            let factor = 1.0 - (-dt / MOUSE_SMOOTHING).exp();
            camera.yaw = Rad(camera.yaw.0 + (self.target_yaw - camera.yaw.0) * factor);
            camera.pitch = Rad(camera.pitch.0 + (self.target_pitch - camera.pitch.0) * factor);
        }

        // Track rotation velocity for motion blur
        if dt > 0.0 {
            self.yaw_velocity = (camera.yaw.0 - prev_yaw) / dt;
            self.pitch_velocity = (camera.pitch.0 - prev_pitch) / dt;
        }

        let player_height = 1.6; // Player is ~1.6 blocks tall, camera at eye level

        // Check if camera (head) is underwater
        let camera_block = world.get_block_world(
            camera.position.x.floor() as i32,
            camera.position.y.floor() as i32,
            camera.position.z.floor() as i32,
        );
        let camera_underwater = camera_block.is_water();

        // Check if player body is in water (check at body level, slightly below camera)
        let body_y = camera.position.y - 0.8;
        let body_block = world.get_block_world(
            camera.position.x.floor() as i32,
            body_y.floor() as i32,
            camera.position.z.floor() as i32,
        );
        let in_water = body_block.is_water();

        // Noclip mode: free flight with no collision, quadrupled speed
        if noclip {
            let fly_speed = self.speed * 4.0;

            // Get movement direction (use camera direction for forward/backward)
            let forward = camera.get_horizontal_direction();
            let right = forward.cross(Vector3::unit_y()).normalize();

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

            // Vertical movement: space to fly up, shift to fly down
            if self.jump_held {
                move_dir.y += 1.0;
            }
            if self.shift_held {
                move_dir.y -= 1.0;
            }

            // Normalize and apply speed
            let has_input = move_dir.magnitude2() > 0.0;
            if has_input {
                move_dir = move_dir.normalize() * fly_speed;
            }

            // Smoothly interpolate noclip velocity
            let rate = if has_input { MOVEMENT_ACCEL } else { MOVEMENT_DECEL };
            let factor = 1.0 - (-rate * dt).exp();
            self.velocity.x += (move_dir.x - self.velocity.x) * factor;
            self.velocity.y += (move_dir.y - self.velocity.y) * factor;
            self.velocity.z += (move_dir.z - self.velocity.z) * factor;

            // Apply smoothed velocity (no collision)
            camera.position.x += self.velocity.x * dt;
            camera.position.y += self.velocity.y * dt;
            camera.position.z += self.velocity.z * dt;

            self.on_ground = false;
            return camera_underwater;
        }

        // Swimming mode: different physics when in water
        if in_water {
            let swim_speed = self.speed * 0.5; // Slower in water
            let sink_speed = -2.0; // Slow sinking
            let swim_up_speed = 8.0;
            let swim_down_speed = -5.0;

            // Get movement direction
            let forward = camera.get_horizontal_direction();
            let right = forward.cross(Vector3::unit_y()).normalize();

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

            // Normalize horizontal movement
            let has_input = move_dir.magnitude2() > 0.0;
            if has_input {
                move_dir = move_dir.normalize() * swim_speed;
            }

            // Vertical swimming: space to swim up, shift to swim down
            let mut vertical_velocity = sink_speed; // Default: slowly sink
            if self.jump_held {
                vertical_velocity = swim_up_speed;
            } else if self.shift_held {
                vertical_velocity = swim_down_speed;
            }

            // Smoothly interpolate horizontal water movement
            let rate = if has_input { MOVEMENT_ACCEL } else { MOVEMENT_DECEL };
            let factor = 1.0 - (-rate * dt).exp();
            self.velocity.x += (move_dir.x - self.velocity.x) * factor;
            self.velocity.z += (move_dir.z - self.velocity.z) * factor;
            // Vertical set directly so the player can break the surface and exit water
            self.velocity.y = vertical_velocity;

            // Calculate new position
            let mut new_pos = camera.position;
            new_pos.x += self.velocity.x * dt;
            new_pos.y += self.velocity.y * dt;
            new_pos.z += self.velocity.z * dt;

            // Simple collision (still collide with solid blocks underwater)
            let player_radius = 0.25;
            let feet_check_y = new_pos.y - player_height + 0.1;
            let body_check_y = new_pos.y - 1.0;
            let head_check_y = new_pos.y - 0.1;

            // Check X collision
            let check_x = if self.velocity.x > 0.0 { new_pos.x + player_radius } else { new_pos.x - player_radius };
            if world.get_block_world(check_x.floor() as i32, head_check_y.floor() as i32, new_pos.z.floor() as i32).is_solid()
                || world.get_block_world(check_x.floor() as i32, body_check_y.floor() as i32, new_pos.z.floor() as i32).is_solid()
                || world.get_block_world(check_x.floor() as i32, feet_check_y.floor() as i32, new_pos.z.floor() as i32).is_solid()
            {
                new_pos.x = camera.position.x;
            }

            // Check Z collision
            let check_z = if self.velocity.z > 0.0 { new_pos.z + player_radius } else { new_pos.z - player_radius };
            if world.get_block_world(new_pos.x.floor() as i32, head_check_y.floor() as i32, check_z.floor() as i32).is_solid()
                || world.get_block_world(new_pos.x.floor() as i32, body_check_y.floor() as i32, check_z.floor() as i32).is_solid()
                || world.get_block_world(new_pos.x.floor() as i32, feet_check_y.floor() as i32, check_z.floor() as i32).is_solid()
            {
                new_pos.z = camera.position.z;
            }

            // Ground collision when swimming down
            let feet_y = new_pos.y - player_height;
            let ground_block = world.get_block_world(new_pos.x.floor() as i32, feet_y.floor() as i32, new_pos.z.floor() as i32);
            if ground_block.is_solid() {
                let block_top = (feet_y.floor() as i32 + 1) as f32;
                if feet_y < block_top {
                    new_pos.y = block_top + player_height;
                    self.velocity.y = 0.0;
                }
            }

            self.on_ground = false;
            camera.position = new_pos;
            return camera_underwater;
        }

        // Normal mode with gravity and collision
        let gravity = -30.0;
        let jump_velocity = 10.0;

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
        let has_input = move_dir.magnitude2() > 0.0;
        if has_input {
            move_dir = move_dir.normalize() * self.speed;
        }

        // Smoothly accelerate/decelerate horizontal velocity
        let rate = if has_input { MOVEMENT_ACCEL } else { MOVEMENT_DECEL };
        let factor = 1.0 - (-rate * dt).exp();
        self.velocity.x += (move_dir.x - self.velocity.x) * factor;
        self.velocity.z += (move_dir.z - self.velocity.z) * factor;

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
        let old_feet_y = camera.position.y - player_height;

        // Check multiple positions for player width to prevent corner clipping
        let player_radius = 0.25;

        // Horizontal collision FIRST - this prevents wall blocks from being detected as ground
        // Check at multiple heights (feet, body, head)
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

        // Now calculate check_positions with the corrected horizontal position
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
            // Capture fall velocity before zeroing (for fall damage calculation)
            if !self.on_ground {
                self.last_fall_velocity = self.velocity.y;
            }
            new_pos.y = highest_ground + player_height;
            self.velocity.y = 0.0;
            self.on_ground = true;
        } else {
            // Check if we're standing on ground (only when not moving upward)
            // This prevents "gliding" up diagonal blocks by spam-jumping
            if self.velocity.y <= 0.0 {
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
            } else {
                self.on_ground = false;
            }
        }

        camera.position = new_pos;
        camera_underwater
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

/// A plane in 3D space represented as ax + by + cz + d = 0
#[derive(Clone, Copy)]
pub struct Plane {
    pub normal: Vector3<f32>,
    pub d: f32,
}

impl Plane {
    /// Returns the signed distance from the plane to a point
    pub fn distance_to_point(&self, point: Vector3<f32>) -> f32 {
        self.normal.dot(point) + self.d
    }
}

/// View frustum for culling chunks that are not visible
pub struct Frustum {
    planes: [Plane; 6], // left, right, bottom, top, near, far
}

impl Frustum {
    /// Extract frustum planes from a view-projection matrix
    pub fn from_view_proj(vp: &Matrix4<f32>) -> Self {
        let m = vp;

        // Extract rows
        let row0 = Vector4::new(m[0][0], m[1][0], m[2][0], m[3][0]);
        let row1 = Vector4::new(m[0][1], m[1][1], m[2][1], m[3][1]);
        let row2 = Vector4::new(m[0][2], m[1][2], m[2][2], m[3][2]);
        let row3 = Vector4::new(m[0][3], m[1][3], m[2][3], m[3][3]);

        // Extract planes using Gribb/Hartmann method
        let planes_raw = [
            row3 + row0, // Left
            row3 - row0, // Right
            row3 + row1, // Bottom
            row3 - row1, // Top
            row3 + row2, // Near
            row3 - row2, // Far
        ];

        let mut planes = [Plane { normal: Vector3::zero(), d: 0.0 }; 6];
        for (i, p) in planes_raw.iter().enumerate() {
            let len = Vector3::new(p.x, p.y, p.z).magnitude();
            if len > 0.0001 {
                planes[i] = Plane {
                    normal: Vector3::new(p.x / len, p.y / len, p.z / len),
                    d: p.w / len,
                };
            }
        }

        Frustum { planes }
    }

    /// Test if an axis-aligned bounding box is visible (intersects or is inside the frustum)
    /// Returns true if the AABB should be rendered
    pub fn is_box_visible(&self, min: Vector3<f32>, max: Vector3<f32>) -> bool {
        for plane in &self.planes {
            // Find the corner of the AABB most in the direction of the plane normal (p-vertex)
            let p = Vector3::new(
                if plane.normal.x >= 0.0 { max.x } else { min.x },
                if plane.normal.y >= 0.0 { max.y } else { min.y },
                if plane.normal.z >= 0.0 { max.z } else { min.z },
            );

            // If the p-vertex is outside, the box is completely outside
            if plane.distance_to_point(p) < 0.0 {
                return false;
            }
        }
        true
    }
}
