use crate::block::Vertex;
use crate::texture::TEX_NONE;
use crate::world::World;
use cgmath::{Point3, Vector3, InnerSpace};

// ── Dimensions ──────────────────────────────────────────────
const HUMANOID_SIZE_W: f32 = 1.2;
const HUMANOID_SIZE_H: f32 = 1.8;

// Body part dimensions (width, height, depth)
const HEAD_W: f32 = 0.30;
const HEAD_H: f32 = 0.30;
const HEAD_D: f32 = 0.30;

const NECK_W: f32 = 0.12;
const NECK_H: f32 = 0.08;
const NECK_D: f32 = 0.12;

const CHEST_W: f32 = 0.40;
const CHEST_H: f32 = 0.28;
const CHEST_D: f32 = 0.22;

const TORSO_W: f32 = 0.34;
const TORSO_H: f32 = 0.18;
const TORSO_D: f32 = 0.15;

const HIPS_W: f32 = 0.34;
const HIPS_H: f32 = 0.12;
const HIPS_D: f32 = 0.20;

const UPPER_ARM_W: f32 = 0.10;
const UPPER_ARM_H: f32 = 0.26;
const UPPER_ARM_D: f32 = 0.10;

const LOWER_ARM_W: f32 = 0.09;
const LOWER_ARM_H: f32 = 0.24;
const LOWER_ARM_D: f32 = 0.09;

const HAND_W: f32 = 0.10;
const HAND_H: f32 = 0.06;
const HAND_D: f32 = 0.08;

const FINGER_W: f32 = 0.025;
const FINGER_H: f32 = 0.06;
const FINGER_D: f32 = 0.025;

const THUMB_W: f32 = 0.03;
const THUMB_H: f32 = 0.05;
const THUMB_D: f32 = 0.03;

const UPPER_LEG_W: f32 = 0.13;
const UPPER_LEG_H: f32 = 0.30;
const UPPER_LEG_D: f32 = 0.13;

const LOWER_LEG_W: f32 = 0.11;
const LOWER_LEG_H: f32 = 0.28;
const LOWER_LEG_D: f32 = 0.11;

const HEEL_W: f32 = 0.11;
const HEEL_H: f32 = 0.06;
const HEEL_D: f32 = 0.09;

const FOOT_FRONT_W: f32 = 0.11;
const FOOT_FRONT_H: f32 = 0.05;
const FOOT_FRONT_D: f32 = 0.11;

const EAR_W: f32 = 0.04;
const EAR_H: f32 = 0.08;
const EAR_D: f32 = 0.06;

const EYE_OUTER_W: f32 = 0.08;
const EYE_OUTER_H: f32 = 0.05;
const EYE_OUTER_D: f32 = 0.02;

const EYE_INNER_W: f32 = 0.04;
const EYE_INNER_H: f32 = 0.04;
const EYE_INNER_D: f32 = 0.02;

const EYELID_W: f32 = 0.082;
const EYELID_H: f32 = 0.052;
const EYELID_D: f32 = 0.022;

const LIP_W: f32 = 0.12;
const LIP_H: f32 = 0.025;
const LIP_D: f32 = 0.02;

// ── Colors ──────────────────────────────────────────────────
const SKIN_COLOR: [f32; 3] = [0.51, 0.32, 0.21];
const SKIN_COLOR_DARK: [f32; 3] = [0.41, 0.22, 0.11];
const EYE_WHITE: [f32; 3] = [0.95, 0.95, 0.95];
const EYE_DARK: [f32; 3] = [0.12, 0.08, 0.06];
const LIP_COLOR: [f32; 3] = [0.72, 0.45, 0.42];
const HAIR_COLOR: [f32; 3] = [0.10, 0.07, 0.04];
const CLOTH_COLOR: [f32; 3] = [0.82, 0.74, 0.42]; // Faded yellow
const BELT_COLOR: [f32; 3] = [0.45, 0.28, 0.12];  // Brown leather

// ── Breechcloth Dimensions ───────────────────────────────────
const BELT_W: f32 = HIPS_W + 0.02;
const BELT_H: f32 = 0.04;
const BELT_D: f32 = HIPS_D + 0.02;

const CLOTH_W: f32 = HIPS_W * 0.5; // Half waist width
const CLOTH_D: f32 = 0.025;         // Thin cloth slab

// ── Hair & Eyebrow Dimensions ────────────────────────────────
const EYEBROW_W: f32 = 0.09;
const EYEBROW_H: f32 = 0.02;
const EYEBROW_D: f32 = 0.015;

const HAIR_CAP_W: f32 = 0.32;  // Slightly wider than HEAD_W
const HAIR_CAP_H: f32 = 0.05;
const HAIR_CAP_D: f32 = 0.20;  // Covers back portion of head top only

const HAIR_BACK_W: f32 = 0.30;
const HAIR_BACK_D: f32 = 0.06;

const HAIR_SIDE_W: f32 = 0.09;
const HAIR_SIDE_D: f32 = 0.09;

// ── Feather Headband Dimensions ──────────────────────────────
//const BAND_COLOR: [f32; 3] = [0.88, 0.42, 0.42]; // Light red
//const BAND_COLOR: [f32; 3] = [0.88, 0.76, 0.42]; // mustard yellow
const BAND_COLOR: [f32; 3] = [0.88, 0.12, 0.12]; // blood red
const BAND_W: f32 = HEAD_W + 0.01;
const BAND_H: f32 = 0.025;
const BAND_D: f32 = HEAD_D + 0.01;

const FEATHER_COLOR: [f32; 3] = [0.96, 0.96, 0.96]; // White
const FEATHER_QUILL_COLOR: [f32; 3] = [0.72, 0.72, 0.72]; // Light grey
const FEATHER_W: f32 = 0.06;
const FEATHER_H: f32 = 0.2;
const FEATHER_D: f32 = 0.02;
const FEATHER_QUILL_W: f32 = 0.012;

// ── AI constants ────────────────────────────────────────────
const WALK_SPEED: f32 = 2.0;
const RUN_SPEED: f32 = 5.5;
const DETECTION_RANGE: f32 = 18.0;
const ATTACK_RANGE: f32 = 2.0;
const PUNCH_DAMAGE: f32 = 200.0;
const PUNCH_COOLDOWN: f32 = 0.8;
const JUMP_VEL: f32 = 8.0;
const WANDER_PAUSE_MIN: f32 = 2.0;
const WANDER_PAUSE_MAX: f32 = 5.0;
const WANDER_WALK_MIN: f32 = 2.0;
const WANDER_WALK_MAX: f32 = 5.0;
const OBSTACLE_CHECK_DIST: f32 = 1.6;
const ACCEL_RATE: f32 = 6.0;  // How fast humanoid accelerates/decelerates (higher = snappier)

// ── Animation ───────────────────────────────────────────────

struct AnimValue {
    pub value: f32,
    peak: f32,
}

impl AnimValue {
    fn new() -> Self { Self { value: 0.0, peak: 0.0 } }

    // Sets value to amt * pt, and remembers amt as the peak for reset()
    fn update(&mut self, amt: f32, pt: f32) {
        self.peak = amt;
        self.value = amt * pt;
    }

    // Lerps value from `start` to `end` over pt (0.0 → 1.0)
    fn update_from(&mut self, start: f32, end: f32, pt: f32) {
        self.peak = end;
        self.value = start + (end - start) * pt;
    }

    // Eases value back from peak to 0 over pt (0.0 → 1.0)
    fn reset(&mut self, pt: f32) {
        self.value = self.peak * (1.0 - pt);
    }
}

struct BodySegmentState {
    pub pitch: AnimValue,
    pub yaw: AnimValue,
    pub roll: AnimValue,
}

impl BodySegmentState {
    fn new() -> Self { Self { pitch: AnimValue::new(), yaw: AnimValue::new(), roll: AnimValue::new() } }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HumanoidAnim {
    Idle,
    Walking,
    Running,
    Jumping,
    Punching,
    Death,
}

#[derive(Debug, Clone)]
pub struct HumanoidPose {
    // Limb swing angles (positive = forward)
    pub left_shoulder: f32,
    pub right_shoulder: f32,
    pub left_arm_spread: f32,
    pub right_arm_spread: f32,
    pub left_elbow: f32,
    pub right_elbow: f32,
    pub left_hip: f32,
    pub right_hip: f32,
    pub left_knee: f32,
    pub right_knee: f32,
    pub left_foot: f32, // New: Ankle flexion
    pub right_foot: f32, // New: Ankle flexion
    
    // Head orientation
    pub head_pitch: f32,
    pub head_yaw: f32,
    
    // Body orientation
    pub body_pitch: f32,
    pub torso_yaw: f32, 
    pub hips_yaw: f32, // New: Independent hip rotation (vital for punches)
    
    // Vertical bounce
    pub bob_y: f32,
    
    // Eyelids (0.0 = Open, 1.0 = Closed)
    pub eyelid_left: f32,
    pub eyelid_right: f32,

    // Eyebrow anger (0.0 = neutral, 1.0 = fully furrowed inward)
    pub brow_anger: f32,
}

impl Default for HumanoidPose {
    fn default() -> Self {
        Self {
            left_shoulder: 0.0, right_shoulder: 0.0,
            left_arm_spread: 0.0, right_arm_spread: 0.0,
            left_elbow: 0.0, right_elbow: 0.0,
            left_hip: 0.0, right_hip: 0.0,
            left_knee: 0.0, right_knee: 0.0,
            left_foot: 0.0, right_foot: 0.0,
            head_pitch: 0.0, head_yaw: 0.0,
            body_pitch: 0.0, torso_yaw: 0.0, hips_yaw: 0.0,
            bob_y: 0.0,
            eyelid_left: 0.0, eyelid_right: 0.0,
            brow_anger: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WanderState {
    Standing,
    Walking,
}

#[derive(Debug, Clone)]
pub struct HumanoidState {
    pub anim: HumanoidAnim,
    pub prev_anim: HumanoidAnim,
    pub blend_timer: f32,
    pub anim_phase: f32,
    pub pose: HumanoidPose,
    pub punch_timer: f32,
    pub punch_cooldown: f32,
    pub punch_hit_applied: bool,
    pub punch_right: bool,
    pub wander_state: WanderState,
    pub wander_timer: f32,
    pub wander_dir: f32,
    pub on_ground: bool,
    pub was_on_ground: bool,
    pub rng_state: u32,
}

impl HumanoidState {
    pub fn new(seed: u32) -> Self {
        Self {
            anim: HumanoidAnim::Idle,
            prev_anim: HumanoidAnim::Idle,
            blend_timer: 0.0,
            anim_phase: 0.0,
            pose: HumanoidPose::default(),
            punch_timer: 0.0,
            punch_cooldown: 0.0,
            punch_hit_applied: false,
            punch_right: true,
            wander_state: WanderState::Standing,
            wander_timer: pseudo_rand_range_s(seed, WANDER_PAUSE_MIN, WANDER_PAUSE_MAX),
            wander_dir: pseudo_rand_range_s(seed.wrapping_add(1), 0.0, std::f32::consts::TAU),
            on_ground: false,
            was_on_ground: false,
            rng_state: seed,
        }
    }
}

pub fn humanoid_size_w() -> f32 { HUMANOID_SIZE_W }
pub fn humanoid_size_h() -> f32 { HUMANOID_SIZE_H }
pub fn humanoid_damage() -> f32 { PUNCH_DAMAGE }

const BLEND_DURATION: f32 = 0.15; // Faster blend for snappier feel

// ── AI Update ───────────────────────────────────────────────

pub fn update_humanoid(
    position: &mut Point3<f32>,
    velocity: &mut Vector3<f32>,
    facing_yaw: &mut f32,
    target_yaw: &mut f32,
    alive: bool,
    death_timer: f32,
    state: &mut HumanoidState,
    dt: f32,
    player_pos: Point3<f32>,
    world: &World,
) {
    if !alive || death_timer >= 0.0 { 
        if state.anim != HumanoidAnim::Death { set_anim(state, HumanoidAnim::Death); }
        update_animation(state, dt);
        return;
    }

    let to_player = player_pos - *position;
    let dist = (to_player.x * to_player.x + to_player.z * to_player.z).sqrt();

    // Punch cooldown
    if state.punch_cooldown > 0.0 { state.punch_cooldown -= dt; }
    if state.punch_timer > 0.0 {
        state.punch_timer -= dt;
        if state.punch_timer <= 0.0 { set_anim(state, HumanoidAnim::Idle); }
    }

    // AI Logic
    if dist < ATTACK_RANGE && dist < DETECTION_RANGE {
        *target_yaw = (-to_player.x).atan2(to_player.z);

        let brake = (ACCEL_RATE * dt).min(1.0);
        velocity.x -= velocity.x * brake;
        velocity.z -= velocity.z * brake;
        if state.punch_cooldown <= 0.0 && state.punch_timer <= 0.0 {
            set_anim(state, HumanoidAnim::Punching);
            state.punch_timer = 0.5;
            state.punch_cooldown = PUNCH_COOLDOWN;
            state.punch_hit_applied = false;
        }
    } else if dist < DETECTION_RANGE {
        // Run
        let dir_x = to_player.x / dist;
        let dir_z = to_player.z / dist;
        *target_yaw = (-dir_x).atan2(dir_z);

        handle_movement(position, velocity, target_yaw, state, dir_x, dir_z, RUN_SPEED, HumanoidAnim::Running, world, dt);
    } else {
        // Wander
        wander(position, velocity, target_yaw, state, dt, world);
    }

    // Rotation smoothing
    let mut delta = *target_yaw - *facing_yaw;
    while delta > std::f32::consts::PI { delta -= std::f32::consts::TAU; }
    while delta < -std::f32::consts::PI { delta += std::f32::consts::TAU; }
    *facing_yaw += delta * (8.0 * dt).min(1.0);

    if !state.on_ground && state.anim != HumanoidAnim::Punching {
        set_anim(state, HumanoidAnim::Jumping);
    }

    update_animation(state, dt);
}

fn handle_movement(
    pos: &mut Point3<f32>, vel: &mut Vector3<f32>, target_yaw: &mut f32,
    state: &mut HumanoidState, dir_x: f32, dir_z: f32, speed: f32,
    move_anim: HumanoidAnim, world: &World, dt: f32,
) {
    let ahead_x = pos.x + dir_x * OBSTACLE_CHECK_DIST;
    let ahead_z = pos.z + dir_z * OBSTACLE_CHECK_DIST;
    let feet_y = (pos.y + 0.1).floor() as i32;
    let block_low = world.get_block_world(ahead_x as i32, feet_y, ahead_z as i32);
    let block_mid = world.get_block_world(ahead_x as i32, feet_y + 1, ahead_z as i32);

    if block_low.is_solid() && !block_mid.is_solid() && state.on_ground {
        vel.y = JUMP_VEL;
        state.on_ground = false;
        set_anim(state, HumanoidAnim::Jumping);
    } else if block_mid.is_solid() && state.on_ground {
         // Try to turn
         state.wander_dir = *target_yaw + std::f32::consts::FRAC_PI_2;
    }

    let target_vx = dir_x * speed;
    let target_vz = dir_z * speed;
    let t = (ACCEL_RATE * dt).min(1.0);
    vel.x += (target_vx - vel.x) * t;
    vel.z += (target_vz - vel.z) * t;

    if state.on_ground && state.anim != HumanoidAnim::Punching {
        set_anim(state, move_anim);
    }
}

fn wander(
    pos: &mut Point3<f32>, vel: &mut Vector3<f32>, target_yaw: &mut f32,
    state: &mut HumanoidState, dt: f32, world: &World
) {
    state.wander_timer -= dt;
    match state.wander_state {
        WanderState::Standing => {
            let brake = (ACCEL_RATE * dt).min(1.0);
            vel.x -= vel.x * brake;
            vel.z -= vel.z * brake;
            if state.anim != HumanoidAnim::Punching { set_anim(state, HumanoidAnim::Idle); }
            if state.wander_timer <= 0.0 {
                state.wander_state = WanderState::Walking;
                state.wander_timer = pseudo_rand_range_m(&mut state.rng_state, WANDER_WALK_MIN, WANDER_WALK_MAX);
                state.wander_dir = pseudo_rand_range_m(&mut state.rng_state, 0.0, std::f32::consts::TAU);
            }
        }
        WanderState::Walking => {
            let dir_x = -state.wander_dir.sin();
            let dir_z = state.wander_dir.cos();

            *target_yaw = state.wander_dir;
            handle_movement(pos, vel, target_yaw, state, dir_x, dir_z, WALK_SPEED, HumanoidAnim::Walking, world, dt);
            
            if state.wander_timer <= 0.0 {
                state.wander_state = WanderState::Standing;
                state.wander_timer = pseudo_rand_range_m(&mut state.rng_state, WANDER_PAUSE_MIN, WANDER_PAUSE_MAX);
            }
        }
    }
}

fn set_anim(state: &mut HumanoidState, anim: HumanoidAnim) {
    if state.anim != anim {
        state.prev_anim = state.anim;
        state.anim = anim;
        state.blend_timer = BLEND_DURATION;
        if anim == HumanoidAnim::Punching { 
            state.anim_phase = 0.0; 
            state.punch_right = !state.punch_right; // Alternate each punch
        }
    }
}

fn update_animation(state: &mut HumanoidState, dt: f32) {
    if state.blend_timer > 0.0 { state.blend_timer = (state.blend_timer - dt).max(0.0); }

    let speed = match state.anim {
        HumanoidAnim::Idle => 1.0,
        HumanoidAnim::Walking => 10.0, // Snappier walk
        HumanoidAnim::Running => 15.0, // High cadence
        HumanoidAnim::Jumping => 2.0,
        HumanoidAnim::Punching => 1.0, 
        HumanoidAnim::Death => 1.5,
    };
    state.anim_phase += dt * speed;

    let target = compute_pose(state.anim, state.anim_phase, state.punch_right);
    let blend_t = if state.blend_timer > 0.0 { state.blend_timer / BLEND_DURATION } else { 0.0 };

    if blend_t > 0.001 {
        let prev = compute_pose(state.prev_anim, state.anim_phase, state.punch_right);
        state.pose = lerp_pose(&prev, &target, 1.0 - blend_t);
    } else {
        state.pose = target;
    }
}

// ── Animations/Poses Logic ──────────────────────────────────────────────
fn compute_pose(anim: HumanoidAnim, phase: f32, punch_right: bool) -> HumanoidPose {
    let mut p = HumanoidPose::default();
    match anim {
        HumanoidAnim::Idle => {
            let breath = (phase * 1.5).sin() * 0.02;
            let drift = (phase * 0.5).sin() * 0.03;
            let is_blinking = if (phase * 0.8).rem_euclid(4.0) > 3.85 { 1.0 } else { 0.0 };

            p.left_shoulder = -0.05 + drift;
            p.right_shoulder = -0.05 - drift;
            p.left_elbow = -0.1 + drift * 0.5;
            p.right_elbow = -0.1 - drift * 0.5;
            p.head_pitch = breath;
            p.head_yaw = drift * 0.3;
            p.torso_yaw = drift * 0.5;
            p.bob_y = breath * 0.5;
            p.eyelid_left = is_blinking;
            p.eyelid_right = is_blinking;
        }
        HumanoidAnim::Walking => {
            let sway = 0.5;
            let lh = phase.sin() * sway;
            let rh = (phase + std::f32::consts::PI).sin() * sway;
            
            // Double bounce: lowest at contact (0, PI), highest at passing (PI/2, 3PI/2)
            let bob = (phase * 2.0).sin().abs() * 0.06;

            p.left_hip = lh;
            p.right_hip = rh;
            
            // Inverse Kinematics-ish knee bend: Bend more when bringing leg forward
            p.left_knee = (phase.sin().max(0.0)) * 0.8;
            p.right_knee = ((phase + std::f32::consts::PI).sin().max(0.0)) * 0.8;

            // Arms opposite to legs
            p.left_shoulder = rh * 0.8;
            p.right_shoulder = lh * 0.8;
            p.left_elbow = -0.2 - (rh * 0.2).max(0.0);
            p.right_elbow = -0.2 - (lh * 0.2).max(0.0);
            p.left_arm_spread = 0.1;
            p.right_arm_spread = 0.1;

            p.bob_y = bob;
            p.body_pitch = 0.05;
        }
        HumanoidAnim::Running => {
            let sway = 1.4;
            let lh = phase.sin() * sway;
            let rh = (phase + std::f32::consts::PI).sin() * sway;
            
            // Bouncy run
            let bob = (phase * 2.0).sin().abs() * 0.12;
            
            p.left_hip = lh;
            p.right_hip = rh;
            
            // Exaggerated knees
            p.left_knee = (phase.sin().max(-0.2) + 0.2) * 1.8;
            p.right_knee = ((phase + std::f32::consts::PI).sin().max(-0.2) + 0.2) * 1.8;

            // Tuck arms
            p.left_shoulder = rh * 1.1;
            p.right_shoulder = lh * 1.1;
            p.left_elbow = -1.8; // Bent 90 deg roughly
            p.right_elbow = -1.8;
            p.left_arm_spread = 0.2;
            p.right_arm_spread = 0.2;

            p.body_pitch = 0.3; // Lean forward
            p.head_pitch = -0.2; // Look up
            p.bob_y = bob;

            p.eyelid_left = 0.4; p.eyelid_right = 0.4;
            p.brow_anger = 0.5;
        }
        HumanoidAnim::Jumping => {
            p.left_shoulder = -0.5; p.right_shoulder = -0.5;
            p.left_arm_spread = 1.0; p.right_arm_spread = 1.0;
            p.left_elbow = -0.5; p.right_elbow = -0.5;
            // Scissor legs
            p.left_hip = 0.8; p.right_hip = -0.5;
            p.left_knee = 0.5; p.right_knee = 1.0;
            p.bob_y = 0.2;
        }
        HumanoidAnim::Punching => {
            const PHASE_SPEED: f32 = 2.0; // Animation speed. should be 2.0
            let t = (phase * PHASE_SPEED).clamp(0.0, 1.0);
            
            // Punch mechanics.
            let mut arm = BodySegmentState::new();
            let mut forearm = BodySegmentState::new();
            let mut body = BodySegmentState::new();
            let mut hip = BodySegmentState::new();

            // Windup
            if t < 0.3 {
                let wt = t / 0.3;

                arm.pitch.update(1.0, wt);
                arm.yaw.update(1.0, wt);
                forearm.pitch.update_from(0.0, -2.2, wt);
                body.roll.update_from(0.0, -1.0, wt);
                hip.roll.update_from(0.0, -0.4, wt);
            } 
            // Thrust
            else if t < 0.5 {
                let st = (t - 0.3) / 0.2;

                arm.pitch.update_from(1.0, -2.0, st);
                arm.yaw.update_from(1.0, 1.5, st);
                forearm.pitch.update_from(-2.2, -0.5, st);
                body.roll.update_from(-1.0, 0.5, st);
                hip.roll.update_from(-0.4, 0.2, st);
            } 
            // Recover
            else {
                let rt = (t - 0.5) / 0.5;

                arm.pitch.update_from(-2.0, 0.0, rt);
                arm.yaw.update_from(1.5, 0.0, rt);
                forearm.pitch.update_from(-0.5, 0.0, rt);
                body.roll.update_from(0.5, 0.0, rt);
                hip.roll.update_from(0.2, 0.0, rt);
            }

            let twist_sign = if punch_right { 1.0 } else { -1.0 };

            if punch_right {
                p.right_shoulder = arm.pitch.value;
                p.right_elbow = forearm.pitch.value;
                p.right_arm_spread = arm.yaw.value * twist_sign;
                p.left_shoulder = 0.5;
                p.left_elbow = -2.2;
                p.left_arm_spread = 0.5;
            } else {
                p.left_shoulder = arm.pitch.value;
                p.left_elbow = forearm.pitch.value;
                p.left_arm_spread = -arm.yaw.value * twist_sign;
                p.right_shoulder = 0.5;
                p.right_elbow = -2.2;
                p.right_arm_spread = 0.5;
            }

            // Body twist
            p.torso_yaw = body.roll.value * twist_sign;
            p.hips_yaw = hip.roll.value * twist_sign;

            // Plant feet widely
            p.left_hip = 0.4;
            p.right_hip = -0.4;
            p.left_knee = 0.3;
            p.right_knee = 0.5;
            
            // Eyelids
            p.eyelid_left = 0.6; p.eyelid_right = 0.6;
            p.brow_anger = 0.5;
        }
        HumanoidAnim::Death => {
            let t = (phase * 0.8).min(1.0);
            let fall = 1.0 - (1.0 - t) * (1.0 - t); // Ease out
            p.body_pitch = -1.5 * fall;
            p.bob_y = -0.6 * fall;
            p.left_arm_spread = 0.5 * fall; p.right_arm_spread = 0.5 * fall;
            p.left_knee = 0.5 * fall; p.right_knee = 0.2 * fall;
            p.eyelid_left = 1.0; p.eyelid_right = 1.0;
        }
    }
    p
}

fn lerp_pose(a: &HumanoidPose, b: &HumanoidPose, t: f32) -> HumanoidPose {
    HumanoidPose {
        left_shoulder: lerp(a.left_shoulder, b.left_shoulder, t),
        right_shoulder: lerp(a.right_shoulder, b.right_shoulder, t),
        left_arm_spread: lerp(a.left_arm_spread, b.left_arm_spread, t),
        right_arm_spread: lerp(a.right_arm_spread, b.right_arm_spread, t),
        left_elbow: lerp(a.left_elbow, b.left_elbow, t),
        right_elbow: lerp(a.right_elbow, b.right_elbow, t),
        left_hip: lerp(a.left_hip, b.left_hip, t),
        right_hip: lerp(a.right_hip, b.right_hip, t),
        left_knee: lerp(a.left_knee, b.left_knee, t),
        right_knee: lerp(a.right_knee, b.right_knee, t),
        left_foot: lerp(a.left_foot, b.left_foot, t),
        right_foot: lerp(a.right_foot, b.right_foot, t),
        head_pitch: lerp(a.head_pitch, b.head_pitch, t),
        head_yaw: lerp(a.head_yaw, b.head_yaw, t),
        body_pitch: lerp(a.body_pitch, b.body_pitch, t),
        torso_yaw: lerp(a.torso_yaw, b.torso_yaw, t),
        hips_yaw: lerp(a.hips_yaw, b.hips_yaw, t),
        bob_y: lerp(a.bob_y, b.bob_y, t),
        eyelid_left: lerp(a.eyelid_left, b.eyelid_left, t),
        eyelid_right: lerp(a.eyelid_right, b.eyelid_right, t),
        brow_anger: lerp(a.brow_anger, b.brow_anger, t),
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }

// ── Vertex Generation ───────────────────────────────────────
pub fn create_humanoid_vertices(
    position: Point3<f32>,
    yaw: f32,
    pose: &HumanoidPose,
    damage_flash: f32,
    death_timer: f32,
) -> Vec<Vertex> {
    let mut verts = Vec::with_capacity(90 * 24); 

    let flash_t = (damage_flash / 0.3).clamp(0.0, 1.0);
    let skin = tint(SKIN_COLOR, flash_t);
    let skin_dark = tint(SKIN_COLOR_DARK, flash_t);
    let lip = tint(LIP_COLOR, flash_t);

    let death_pitch = if death_timer >= 0.0 { pose.body_pitch } else { 0.0 };
    let mut p = position;
    p.y += pose.bob_y;

    let body_lean = if death_timer < 0.0 { pose.body_pitch } else { 0.0 };
    
    let torso_yaw = yaw + pose.torso_yaw;
    let hips_yaw = yaw + pose.hips_yaw; // Independent hips

    // ─── Hips & Torso ───
    let foot_h = HEEL_H;
    let lower_leg_top = foot_h + LOWER_LEG_H;
    let upper_leg_top = lower_leg_top + UPPER_LEG_H;
    let hips_cy = upper_leg_top + HIPS_H * 0.5;

    // Hips
    add_body_cube(&mut verts, p, hips_yaw, body_lean, death_pitch, 0.0,
        0.0, hips_cy, 0.0, HIPS_W, HIPS_H, HIPS_D, skin);

    let hips_top = upper_leg_top + HIPS_H;
    let torso_cy = hips_top + TORSO_H * 0.5;
    add_body_cube(&mut verts, p, torso_yaw, body_lean, death_pitch, 0.0,
        0.0, torso_cy, 0.0, TORSO_W, TORSO_H, TORSO_D, skin);

    let torso_top = hips_top + TORSO_H;
    let chest_cy = torso_top + CHEST_H * 0.5;
    add_body_cube(&mut verts, p, torso_yaw, body_lean, death_pitch, 0.0,
        0.0, chest_cy, 0.0, CHEST_W, CHEST_H, CHEST_D, skin);

    let chest_top = torso_top + CHEST_H;
    let neck_cy = chest_top + NECK_H * 0.5;
    add_body_cube(&mut verts, p, torso_yaw, body_lean, death_pitch, 0.0,
        0.0, neck_cy, 0.0, NECK_W, NECK_H, NECK_D, skin);

    // ─── Head ───
    let neck_top = chest_top + NECK_H;
    let head_cy = neck_top + HEAD_H * 0.5;
    let head_total_yaw = torso_yaw + pose.head_yaw;
    
    add_body_cube(&mut verts, p, head_total_yaw, body_lean, death_pitch, pose.head_pitch,
        0.0, head_cy, 0.0, HEAD_W, HEAD_H, HEAD_D, skin);

    // Eyes
    let eye_y = head_cy + HEAD_H * 0.1;
    let eye_fwd = HEAD_D * 0.5 + EYE_OUTER_D * 0.5;
    let eye_spread = HEAD_W * 0.22;

    // Left eye (Left is +X or -X? Standard right-hand rule: +X is Right, -X is Left. Let's assume -X Left)
    add_body_cube(&mut verts, p, head_total_yaw, body_lean, death_pitch, pose.head_pitch,
        -eye_spread, eye_y, eye_fwd,  // X=Spread, Z=Fwd
        EYE_OUTER_W, EYE_OUTER_H, EYE_OUTER_D, EYE_WHITE); // Swap W/D dims so eye is flat on face
        
    add_body_cube(&mut verts, p, head_total_yaw, body_lean, death_pitch, pose.head_pitch,
        -eye_spread, eye_y - 0.005, eye_fwd + 0.005, 
        EYE_INNER_W, EYE_INNER_H, EYE_INNER_D, EYE_DARK);

    // Left Eyelid
    if pose.eyelid_left > 0.01 {
        let lid_drop = EYELID_H * pose.eyelid_left;
        add_body_cube(&mut verts, p, head_total_yaw, body_lean, death_pitch, pose.head_pitch,
            -eye_spread, eye_y + EYELID_H * 0.5 - lid_drop * 0.5, eye_fwd + 0.006, 
            EYE_OUTER_W + 0.01, lid_drop, EYELID_D, skin_dark);
    }

    // Right eye
    add_body_cube(&mut verts, p, head_total_yaw, body_lean, death_pitch, pose.head_pitch,
        eye_spread, eye_y, eye_fwd, 
        EYE_OUTER_W, EYE_OUTER_H, EYE_OUTER_D, EYE_WHITE);
        
    add_body_cube(&mut verts, p, head_total_yaw, body_lean, death_pitch, pose.head_pitch,
        eye_spread, eye_y - 0.005, eye_fwd + 0.005, 
        EYE_INNER_W, EYE_INNER_H, EYE_INNER_D, EYE_DARK);

    // Right eyelid
    if pose.eyelid_right > 0.01 {
        let lid_drop = EYELID_H * pose.eyelid_right;
        add_body_cube(&mut verts, p, head_total_yaw, body_lean, death_pitch, pose.head_pitch,
            eye_spread, eye_y + EYELID_H * 0.5 - lid_drop * 0.5, eye_fwd + 0.006, 
            EYE_OUTER_W + 0.01, lid_drop, EYELID_D, skin_dark);
    }

    // Lips & Ears
    add_body_cube(&mut verts, p, head_total_yaw, body_lean, death_pitch, pose.head_pitch,
        0.0, head_cy - HEAD_H * 0.2, eye_fwd - 0.01, LIP_W, LIP_H, LIP_D, lip); // Lip on Z
        
    // Ears on X (Sides)
    let ear_x = HEAD_W * 0.5 + EAR_D * 0.3;
    add_body_cube(&mut verts, p, head_total_yaw, body_lean, death_pitch, pose.head_pitch,
        ear_x, head_cy, 0.0, EAR_D, EAR_H, EAR_W, skin);
    add_body_cube(&mut verts, p, head_total_yaw, body_lean, death_pitch, pose.head_pitch,
        -ear_x, head_cy, 0.0, EAR_D, EAR_H, EAR_W, skin);

    // ─── Arms ───
    let shoulder_y = chest_top - 0.04;
    let shoulder_x = CHEST_W * 0.5 + UPPER_ARM_W * 0.2;

    add_arm(&mut verts, p, torso_yaw, body_lean, death_pitch,
        shoulder_y, shoulder_x,
        pose.left_shoulder, pose.left_arm_spread, pose.left_elbow,
        true, skin);

    add_arm(&mut verts, p, torso_yaw, body_lean, death_pitch,
        shoulder_y, shoulder_x,
        pose.right_shoulder, pose.right_arm_spread, pose.right_elbow,
        false, skin);

    // ─── Hair & Eyebrows ───
    let hair = tint(HAIR_COLOR, flash_t);

    // Eyebrows: split into two halves so inner/outer can be offset for angry tilt
    let brow_y = eye_y + EYE_OUTER_H * 0.5 + EYEBROW_H * 0.5 + 0.008;
    let half_brow_w = EYEBROW_W * 0.5;
    let anger_tilt = pose.brow_anger * 0.018;

    // Left eyebrow (outer half rises, inner half dips)
    add_body_cube(&mut verts, p, head_total_yaw, body_lean, death_pitch, pose.head_pitch,
        -eye_spread - half_brow_w * 0.5, brow_y + anger_tilt, eye_fwd,
        half_brow_w, EYEBROW_H, EYEBROW_D, hair);
    add_body_cube(&mut verts, p, head_total_yaw, body_lean, death_pitch, pose.head_pitch,
        -eye_spread + half_brow_w * 0.5, brow_y - anger_tilt, eye_fwd,
        half_brow_w, EYEBROW_H, EYEBROW_D, hair);

    // Right eyebrow (inner half dips, outer half rises)
    add_body_cube(&mut verts, p, head_total_yaw, body_lean, death_pitch, pose.head_pitch,
        eye_spread - half_brow_w * 0.5, brow_y - anger_tilt, eye_fwd,
        half_brow_w, EYEBROW_H, EYEBROW_D, hair);
    add_body_cube(&mut verts, p, head_total_yaw, body_lean, death_pitch, pose.head_pitch,
        eye_spread + half_brow_w * 0.5, brow_y + anger_tilt, eye_fwd,
        half_brow_w, EYEBROW_H, EYEBROW_D, hair);

    // ─── Feather Headband ───
    // Band sits flush above the eyebrows, wraps around the full head
    let band_y = brow_y + EYEBROW_H * 0.5 + BAND_H * 0.5 + 0.01;
    add_body_cube(&mut verts, p, head_total_yaw, body_lean, death_pitch, pose.head_pitch,
        0.0, band_y, 0.0,
        BAND_W, BAND_H, BAND_D, tint(BAND_COLOR, flash_t));

    // Feather: white vane sitting on front of headband
    let feather_bottom = band_y + BAND_H * 0.5;
    let feather_cy = feather_bottom + FEATHER_H * 0.5;
    let feather_z = HEAD_D * 0.5 + FEATHER_D * 0.5;
    add_body_cube(&mut verts, p, head_total_yaw, body_lean, death_pitch, pose.head_pitch,
        0.0, feather_cy, feather_z,
        FEATHER_W, FEATHER_H, FEATHER_D, tint(FEATHER_COLOR, flash_t));

    // Quill: grey spine from feather bottom up to 75% of its height, on the front face
    let quill_h = FEATHER_H * 0.75;
    let quill_cy = feather_bottom + quill_h * 0.5;
    add_body_cube(&mut verts, p, head_total_yaw, body_lean, death_pitch, pose.head_pitch,
        0.0, quill_cy, feather_z + FEATHER_D * 0.5 + 0.003,
        FEATHER_QUILL_W, quill_h, 0.006, tint(FEATHER_QUILL_COLOR, flash_t));

    // Hair cap (top of head, set back so it doesn't overhang the face)
    let head_top = head_cy + HEAD_H * 0.5;
    add_body_cube(&mut verts, p, head_total_yaw, body_lean, death_pitch, pose.head_pitch,
        0.0, head_top + HAIR_CAP_H * 0.5, -HEAD_D * 0.1,
        HAIR_CAP_W, HAIR_CAP_H, HAIR_CAP_D, hair);

    // Back curtain: wide slab hanging from back of head down to just above shoulders
    let hair_back_top = head_cy + HEAD_H * 0.1;
    let hair_back_bottom = shoulder_y + 0.06;
    let hair_back_h = hair_back_top - hair_back_bottom;
    let hair_back_cy = (hair_back_top + hair_back_bottom) * 0.5;
    add_body_cube(&mut verts, p, torso_yaw, body_lean, death_pitch, 0.0,
        0.0, hair_back_cy, -(HEAD_D * 0.5 + HAIR_BACK_D * 0.5),
        HAIR_BACK_W, hair_back_h, HAIR_BACK_D, hair);

    // Side strands: hang from sides of head down to just above shoulders
    let side_top = head_cy + HEAD_H * 0.1;
    let side_bottom = shoulder_y + 0.06;
    let side_h = side_top - side_bottom;
    let side_cy = (side_top + side_bottom) * 0.5;
    let side_x = HEAD_W * 0.5 + HAIR_SIDE_W * 0.4;
    add_body_cube(&mut verts, p, torso_yaw, body_lean, death_pitch, 0.0,
        -side_x, side_cy, -HEAD_D * 0.15,
        HAIR_SIDE_W, side_h, HAIR_SIDE_D, hair);
    add_body_cube(&mut verts, p, torso_yaw, body_lean, death_pitch, 0.0,
        side_x, side_cy, -HEAD_D * 0.15,
        HAIR_SIDE_W, side_h, HAIR_SIDE_D, hair);

    // ─── Legs ───
    
    let hip_joint_y = upper_leg_top;
    let leg_x = HIPS_W * 0.25;

    add_leg(&mut verts, p, hips_yaw, body_lean, death_pitch,
        hip_joint_y, leg_x,
        pose.left_hip, pose.left_knee, pose.left_foot,
        true, skin);

    add_leg(&mut verts, p, hips_yaw, body_lean, death_pitch,
        hip_joint_y, leg_x,
        pose.right_hip, pose.right_knee, pose.right_foot,
        false, skin);

    // ─── Breechcloth ───
    let cloth = tint(CLOTH_COLOR, flash_t);
    let belt_col = tint(BELT_COLOR, flash_t);

    // Belt: thin band at the waistline (hips_top), rotates with hips
    add_body_cube(&mut verts, p, hips_yaw, body_lean, death_pitch, 0.0,
        0.0, hips_top + BELT_H * 0.5, 0.0,
        BELT_W, BELT_H, BELT_D, belt_col);

    // Front and back flaps: hang from belt down to halfway through the upper leg
    let cloth_h = HIPS_H + UPPER_LEG_H * 0.5;
    let cloth_cy = hips_top - cloth_h * 0.5;
    let cloth_z = HIPS_D * 0.5 + CLOTH_D * 0.5;

    add_body_cube(&mut verts, p, hips_yaw, body_lean, death_pitch, 0.0,
        0.0, cloth_cy, cloth_z,
        CLOTH_W, cloth_h, CLOTH_D, cloth);

    add_body_cube(&mut verts, p, hips_yaw, body_lean, death_pitch, 0.0,
        0.0, cloth_cy, -cloth_z,
        CLOTH_W, cloth_h, CLOTH_D, cloth);

    verts
}

// ── Joint Math Helpers ──────────────────────────────────────

// Rotates (y, z) around X-axis (Pitch)
// Pitch controls forward/backward limb swing
fn rotate_x(y: f32, z: f32, angle: f32) -> (f32, f32) {
    let c = angle.cos();
    let s = angle.sin();
    (y * c - z * s, y * s + z * c)
}

// Rotates (x, y) around Z-axis (Roll/Spread)
// Spread controls arm/leg abduction
fn rotate_z(x: f32, y: f32, angle: f32) -> (f32, f32) {
    let c = angle.cos();
    let s = angle.sin();
    (x * c - y * s, x * s + y * c)
}

fn rotate_yaw(x: f32, y: f32, z: f32, yaw: f32) -> [f32; 3] {
    let c = yaw.cos();
    let s = yaw.sin();
    [x * c - z * s, y, x * s + z * c]
}

// ── Limb Builders ───────────────────────────────────────────

fn add_arm(
    verts: &mut Vec<Vertex>,
    pos: Point3<f32>, yaw: f32, body_lean: f32, death_pitch: f32,
    shoulder_y: f32, shoulder_x_offset: f32,
    shoulder_angle: f32, spread_angle: f32, elbow_angle: f32,
    is_left: bool, skin: [f32; 3],
) {
    // Determine side: Left is -X, Right is +X
    let side = if is_left { -1.0 } else { 1.0 };
    let sx = shoulder_x_offset * side;

    // Joint 1: Shoulder (Body Relative)
    let j1_x = sx;
    let j1_y = shoulder_y - 0.02;
    let j1_z = 0.0; // Centered on Z (Side of body)

    // ─── UPPER ARM ───
    // 1. Pitch (X Rotation) -> Rotates Y and Z (Swings arm Forward/Back)
    let (ua_py, ua_pz) = rotate_x(-UPPER_ARM_H, 0.0, shoulder_angle);
    
    // 2. Spread (Z Rotation) -> Rotates X and Y (Lifts arm Side-to-Side)
    // We rotate the PREVIOUS Y (ua_py) to get the new X and Y
    let (ua_sx, ua_sy) = rotate_z(0.0, ua_py, spread_angle * side);
    
    // Final Vector for Upper Arm
    let ua_vec = [ua_sx, ua_sy, ua_pz];

    // Upper Arm Center (Halfway along vector)
    let ua_cx = j1_x + ua_vec[0] * 0.5;
    let ua_cy = j1_y + ua_vec[1] * 0.5;
    let ua_cz = j1_z + ua_vec[2] * 0.5;

    add_limb_cube(verts, pos, yaw, body_lean, death_pitch,
        ua_cx, ua_cy, ua_cz, 
        UPPER_ARM_W, UPPER_ARM_H, UPPER_ARM_D,
        shoulder_angle, spread_angle * side, skin);

    // ─── LOWER ARM (Forearm) ───
    // Joint 2: Elbow (End of Upper Arm)
    let j2_x = j1_x + ua_vec[0];
    let j2_y = j1_y + ua_vec[1];
    let j2_z = j1_z + ua_vec[2];

    let total_pitch = shoulder_angle + elbow_angle;
    let total_spread = spread_angle * side;

    let (la_py, la_pz) = rotate_x(-LOWER_ARM_H, 0.0, total_pitch);
    let (la_sx, la_sy) = rotate_z(0.0, la_py, total_spread);
    let la_vec = [la_sx, la_sy, la_pz];

    let la_cx = j2_x + la_vec[0] * 0.5;
    let la_cy = j2_y + la_vec[1] * 0.5;
    let la_cz = j2_z + la_vec[2] * 0.5;

    add_limb_cube(verts, pos, yaw, body_lean, death_pitch,
        la_cx, la_cy, la_cz, 
        LOWER_ARM_W, LOWER_ARM_H, LOWER_ARM_D,
        total_pitch, total_spread, skin);

    // ─── HAND ───
    // Joint 3: Wrist (End of Lower Arm)
    let j3_x = j2_x + la_vec[0];
    let j3_y = j2_y + la_vec[1];
    let j3_z = j2_z + la_vec[2];

    let (h_py, h_pz) = rotate_x(-HAND_H * 0.5, 0.0, total_pitch);
    let (h_sx, h_sy) = rotate_z(0.0, h_py, total_spread);

    add_limb_cube(verts, pos, yaw, body_lean, death_pitch,
        j3_x + h_sx, j3_y + h_sy, j3_z + h_pz, 
        HAND_W, HAND_H, HAND_D,
        total_pitch, total_spread, skin);
}

fn add_leg(
    verts: &mut Vec<Vertex>,
    pos: Point3<f32>, yaw: f32, body_lean: f32, death_pitch: f32,
    hip_y: f32, hip_x_offset: f32,
    hip_angle: f32, knee_angle: f32, ankle_angle: f32,
    is_left: bool, skin: [f32; 3],
) {
    let side = if is_left { -1.0 } else { 1.0 };
    let hx = hip_x_offset * side;

    // Joint 1: Hip
    let j1_x = hx; // Offset on X
    let j1_y = hip_y;
    let j1_z = 0.0;

    // ─── UPPER LEG ───
    // Rotate Y (down) and Z (fwd) around X axis
    // Legs generally don't spread much, so we only use Pitch (rotate_x)
    let (ul_py, ul_pz) = rotate_x(-UPPER_LEG_H, 0.0, hip_angle);
    let ul_vec = [0.0, ul_py, ul_pz]; 

    let ul_cx = j1_x + ul_vec[0] * 0.5;
    let ul_cy = j1_y + ul_vec[1] * 0.5;
    let ul_cz = j1_z + ul_vec[2] * 0.5;

    add_limb_cube(verts, pos, yaw, body_lean, death_pitch,
        ul_cx, ul_cy, ul_cz, 
        UPPER_LEG_W, UPPER_LEG_H, UPPER_LEG_D,
        hip_angle, 0.0, skin);

    // ─── LOWER LEG ───
    // Joint 2: Knee
    let j2_x = j1_x + ul_vec[0];
    let j2_y = j1_y + ul_vec[1];
    let j2_z = j1_z + ul_vec[2];

    let total_knee_pitch = hip_angle + knee_angle;
    let (ll_py, ll_pz) = rotate_x(-LOWER_LEG_H, 0.0, total_knee_pitch);
    let ll_vec = [0.0, ll_py, ll_pz];

    let ll_cx = j2_x + ll_vec[0] * 0.5;
    let ll_cy = j2_y + ll_vec[1] * 0.5;
    let ll_cz = j2_z + ll_vec[2] * 0.5;

    add_limb_cube(verts, pos, yaw, body_lean, death_pitch,
        ll_cx, ll_cy, ll_cz, 
        LOWER_LEG_W, LOWER_LEG_H, LOWER_LEG_D,
        total_knee_pitch, 0.0, skin);

    // ─── FOOT / HEEL ───
    // Joint 3: Ankle
    let j3_x = j2_x + ll_vec[0];
    let j3_y = j2_y + ll_vec[1];
    let j3_z = j2_z + ll_vec[2];

    let total_ankle_pitch = total_knee_pitch + ankle_angle;

    // Heel (Center calculation)
    let (h_py, h_pz) = rotate_x(-HEEL_H * 0.5, 0.0, total_ankle_pitch);
    
    add_limb_cube(verts, pos, yaw, body_lean, death_pitch,
        j3_x, j3_y + h_py, j3_z + h_pz, 
        HEEL_W, HEEL_H, HEEL_D,
        total_ankle_pitch, 0.0, skin);

    // Foot Front (Toes)
    // 1. Vector from Ankle to bottom of Heel
    let (hb_py, hb_pz) = rotate_x(-HEEL_H, 0.0, total_ankle_pitch);
    // 2. Vector forward to toes (Forward is +Z in local space before rotation)
    let (toe_py, toe_pz) = rotate_x(0.0, HEEL_D*0.5 + FOOT_FRONT_D*0.5, total_ankle_pitch);
    // 3. Shift up slightly so toes align with heel bottom
    let (lift_py, lift_pz) = rotate_x(FOOT_FRONT_H * 0.5, 0.0, total_ankle_pitch);

    add_limb_cube(verts, pos, yaw, body_lean, death_pitch,
        j3_x, 
        j3_y + hb_py + toe_py + lift_py, 
        j3_z + hb_pz + toe_pz + lift_pz,
        FOOT_FRONT_W, FOOT_FRONT_H, FOOT_FRONT_D,
        total_ankle_pitch, 0.0, skin);
}

// Helper to draw a cube with correct rotations
fn add_body_cube(
    verts: &mut Vec<Vertex>,
    pos: Point3<f32>, yaw: f32, body_lean: f32, death_pitch: f32, extra_pitch: f32,
    lx: f32, ly: f32, lz: f32,
    w: f32, h: f32, d: f32,
    color: [f32; 3],
) {
    let hw = w * 0.5; let hh = h * 0.5; let hd = d * 0.5;
    let local_corners = [
        (-hw, -hh, -hd), ( hw, -hh, -hd), ( hw,  hh, -hd), (-hw,  hh, -hd),
        (-hw, -hh,  hd), ( hw, -hh,  hd), ( hw,  hh,  hd), (-hw,  hh,  hd),
    ];

    let mut corners = Vec::with_capacity(8);
    for (cx, cy, cz) in local_corners {
        // 1. Local Pitch (Head nod)
        let (px_y, px_z) = rotate_x(cy, cz, extra_pitch);
        let bx = lx + cx;
        let mut by = ly + px_y;
        let mut bz = lz + px_z;

        // 2. Body Lean (Pitch entire body part around origin)
        if body_lean.abs() > 0.001 {
            let (l_y, l_z) = rotate_x(by, bz, body_lean); // Lean is pitch
            by = l_y; 
            bz = l_z;
        }

        // 3. Death Pitch
        if death_pitch.abs() > 0.001 {
            let (d_y, d_z) = rotate_x(by, bz, death_pitch);
            by = d_y; 
            bz = d_z;
        }

        let r = rotate_yaw(bx, by, bz, yaw);
        corners.push([pos.x + r[0], pos.y + r[1], pos.z + r[2]]);
    }
    push_cube_verts(verts, corners, color);
}

fn add_limb_cube(
    verts: &mut Vec<Vertex>,
    pos: Point3<f32>, yaw: f32, body_lean: f32, death_pitch: f32,
    lx: f32, ly: f32, lz: f32,
    w: f32, h: f32, d: f32,
    limb_pitch: f32,
    limb_spread: f32,
    color: [f32; 3],
) {
    let hw = w * 0.5; let hh = h * 0.5; let hd = d * 0.5;
    let local_corners = [
        (-hw, -hh, -hd), ( hw, -hh, -hd), ( hw,  hh, -hd), (-hw,  hh, -hd),
        (-hw, -hh,  hd), ( hw, -hh,  hd), ( hw,  hh,  hd), (-hw,  hh,  hd),
    ];

    let mut corners = Vec::with_capacity(8);
    for (cx, cy, cz) in local_corners {
        // 1. Limb Local Rotation
        // Pitch (X-axis) -> Affects Y and Z
        let (px_y, px_z) = rotate_x(cy, cz, limb_pitch);
        // Spread (Z-axis) -> Affects X and Y
        let (sx_x, sx_y) = rotate_z(cx, px_y, limb_spread);
        
        // Add to center position
        let bx = lx + sx_x;
        let mut by = ly + sx_y;
        let mut bz = lz + px_z;

        // 2. Body Lean
        if body_lean.abs() > 0.001 {
            let (l_y, l_z) = rotate_x(by, bz, body_lean);
            by = l_y; bz = l_z;
        }

        // 3. Death
        if death_pitch.abs() > 0.001 {
            let (d_y, d_z) = rotate_x(by, bz, death_pitch);
            by = d_y; bz = d_z;
        }

        let r = rotate_yaw(bx, by, bz, yaw);
        corners.push([pos.x + r[0], pos.y + r[1], pos.z + r[2]]);
    }
    push_cube_verts(verts, corners, color);
}

fn tint(color: [f32; 3], flash_t: f32) -> [f32; 3] {
    [
        color[0] + (1.0 - color[0]) * flash_t,
        color[1] * (1.0 - flash_t * 0.8),
        color[2] * (1.0 - flash_t * 0.8),
    ]
}

fn push_cube_verts(vertices: &mut Vec<Vertex>, corners: Vec<[f32; 3]>, color: [f32; 3]) {
    let faces = [
        [4, 5, 6, 7], [1, 0, 3, 2], [7, 6, 2, 3],
        [0, 1, 5, 4], [5, 1, 2, 6], [0, 4, 7, 3],
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

// ── RNG helpers ─────────────────────────────────────────────
fn simple_hash(mut x: u32) -> u32 {
    x = x.wrapping_mul(73856093);
    x ^= x >> 16;
    x = x.wrapping_mul(2654435761);
    x ^= x >> 13;
    x
}
fn pseudo_rand_m(state: &mut u32) -> f32 {
    *state = simple_hash(*state);
    (*state & 0xFFFF) as f32 / 65535.0
}
fn pseudo_rand_range_m(state: &mut u32, min: f32, max: f32) -> f32 {
    min + pseudo_rand_m(state) * (max - min)
}
fn pseudo_rand_range_s(seed: u32, min: f32, max: f32) -> f32 {
    let v = simple_hash(seed);
    let t = (v & 0xFFFF) as f32 / 65535.0;
    min + t * (max - min)
}