struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_position: vec4<f32>,
    near: f32,
    far: f32,
    _padding: vec2<f32>,
};

// Wave animation parameters passed from Rust
struct WaveUniform {
    time: f32,
    amplitude: f32,
    frequency: f32,
    speed: f32,
    octaves: f32,
    lacunarity: f32,
    persistence: f32,
    _padding: f32,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var depth_texture: texture_depth_2d;

@group(1) @binding(1)
var depth_sampler: sampler;

@group(1) @binding(2)
var<uniform> wave: WaveUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) light_level: f32,
    @location(4) alpha: f32,  // Now used as wave_factor: 0.0 = no wave, 1.0 = full wave
    @location(5) uv: vec2<f32>,
    @location(6) tex_index: u32,
    @location(7) ao: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) frag_pos: vec3<f32>,
    @location(3) light_level: f32,
    @location(4) ao: f32,
    @location(5) wave_tilt_x: f32,  // X component of wave normal for directional shading
};

// ============================================================================
// Noise Functions for Wave Animation
// ============================================================================

// Integer-based hash function - more stable than floating point operations
// Uses prime number multiplication to avoid patterns
fn hash_int(x: i32, y: i32) -> f32 {
    var n = x + y * 57;
    n = (n << 13) ^ n;
    let m = (n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff;
    return f32(m) / 2147483647.0;
}

// 2D value noise with integer-based hashing
fn value_noise(p: vec2<f32>) -> f32 {
    let pi = vec2<i32>(i32(floor(p.x)), i32(floor(p.y)));
    let pf = fract(p);

    // Quintic interpolation for C2 continuity (smoother than cubic)
    let u = pf * pf * pf * (pf * (pf * 6.0 - 15.0) + 10.0);

    // Four corners using integer hash
    let a = hash_int(pi.x, pi.y);
    let b = hash_int(pi.x + 1, pi.y);
    let c = hash_int(pi.x, pi.y + 1);
    let d = hash_int(pi.x + 1, pi.y + 1);

    // Bilinear interpolation
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Layered noise (fractal Brownian motion)
fn fbm_noise(p: vec2<f32>, octaves: i32, lacunarity: f32, persistence: f32) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var total_amplitude = 0.0;
    var pos = p;

    for (var i = 0; i < octaves; i++) {
        value += amplitude * value_noise(pos * frequency);
        total_amplitude += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    return value / total_amplitude;
}

// Calculate wave height at a given world position
fn get_wave_height(world_pos: vec2<f32>) -> f32 {
    // Create moving coordinates based on time
    let time_offset = wave.time * wave.speed;

    // Use two directional waves for more interesting patterns
    let wave_pos1 = world_pos * wave.frequency + vec2<f32>(time_offset * 0.4, time_offset * 0.2);
    let wave_pos2 = world_pos * wave.frequency * 0.6 + vec2<f32>(-time_offset * 0.25, time_offset * 0.35);

    // Calculate layered noise
    let octaves = i32(wave.octaves);
    let noise1 = fbm_noise(wave_pos1, octaves, wave.lacunarity, wave.persistence);
    let noise2 = fbm_noise(wave_pos2, octaves, wave.lacunarity, wave.persistence);

    // Combine waves (subtract 0.5 to center around 0)
    let combined = ((noise1 + noise2) * 0.5) - 0.5;

    // Clamp to prevent extreme values that could cause visual artifacts
    let clamped = clamp(combined, -0.5, 0.5);

    return clamped * wave.amplitude;
}

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Amount to reduce the wave y center
    // 0.0 - No reduction, 1.0 - Full reduction, 0.5 - Half reduction
    let wave_center_reduction = 0.2;

    // Get wave displacement for this vertex position
    let wave_factor = model.alpha; // 0.0 = no wave, 1.0 = full wave
    let wave_height = get_wave_height(model.position.xz) * wave_factor;

    // Apply wave displacement to Y position
    var displaced_position = model.position;
    displaced_position.y += wave_height - (wave_center_reduction);

    let world_position = vec4<f32>(displaced_position, 1.0);
    out.clip_position = camera.view_proj * world_position;
    out.color = model.color;

    // Calculate approximate normal from wave gradient (for better lighting on waves)
    var wave_normal = model.normal;
    var tilt_x = 0.0;

    if (wave_factor > 0.5) {
        // Sample nearby points to estimate surface normal
        let eps = 0.25;  // Larger epsilon for smoother normals
        let h_px = get_wave_height(model.position.xz + vec2<f32>(eps, 0.0));
        let h_nx = get_wave_height(model.position.xz - vec2<f32>(eps, 0.0));
        let h_pz = get_wave_height(model.position.xz + vec2<f32>(0.0, eps));
        let h_nz = get_wave_height(model.position.xz - vec2<f32>(0.0, eps));

        // Gradient gives us slope in X and Z directions
        let dx = (h_px - h_nx) / (2.0 * eps);
        let dz = (h_pz - h_nz) / (2.0 * eps);

        // Clamp gradients to prevent extreme normal tilts
        let dx_clamped = clamp(dx, -0.5, 0.5);
        let dz_clamped = clamp(dz, -0.5, 0.5);

        // Normal is perpendicular to the surface
        wave_normal = normalize(vec3<f32>(-dx_clamped, 1.0, -dz_clamped));

        // Store the X tilt for directional shading in fragment shader
        tilt_x = -dx_clamped;
    }

    out.normal = wave_normal;
    out.wave_tilt_x = tilt_x;
    out.frag_pos = displaced_position;
    out.light_level = model.light_level;
    out.ao = model.ao;
    return out;
}

// Convert depth buffer value to linear view-space depth
fn linearize_depth(depth: f32) -> f32 {
    let z = camera.near * camera.far / (camera.far - depth * (camera.far - camera.near));
    return z;
}

// Calculate linear depth from clip-space position
fn get_linear_depth(clip_z: f32, clip_w: f32) -> f32 {
    let ndc_z = clip_z / clip_w;
    return linearize_depth(ndc_z);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Get screen UV from fragment position
    let screen_size = vec2<f32>(textureDimensions(depth_texture));
    let screen_uv = in.clip_position.xy / screen_size;

    // Sample scene depth behind water (what's already been rendered)
    let scene_depth_raw = textureSample(depth_texture, depth_sampler, screen_uv);
    let scene_depth = linearize_depth(scene_depth_raw);

    // Get water surface depth
    let water_depth = linearize_depth(in.clip_position.z);

    // Calculate distance the view ray travels through water
    let water_distance = max(scene_depth - water_depth, 0.0);

    // Water absorption parameters
    let absorption_coefficient = 0.25;  // How quickly light is absorbed (higher = more opaque faster)
    let min_alpha = 0.4;               // Minimum opacity (shallow water)
    let max_alpha = 1.0;               // Maximum opacity (very deep water)

    // Exponential absorption: Beer-Lambert law - transmittance = e^(-coefficient * distance)
    let transmittance = exp(-absorption_coefficient * water_distance);
    let alpha = mix(max_alpha, min_alpha, transmittance);

    // Lighting calculations
    let min_ambient = 0.05;
    let voxel_light = in.light_level;
    let curved_light = pow(voxel_light, 1.4);
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let directional = max(dot(in.normal, light_dir), 0.0) * 0.15;
    let total_light = min_ambient + curved_light * 0.9 + directional * voxel_light;

    // Apply ambient occlusion
    let final_light = total_light * in.ao;

    // Apply X-axis directional shading for wave visualization
    // Tilt towards -X = darker, tilt towards +X = lighter
    let tilt_shading = clamp(in.wave_tilt_x * 2.0, -0.5, 0.9);  // Scale and clamp the effect
    let tilt_multiplier = 1.0 + tilt_shading;  // Range: 0.5 to 1.9

    let lit_color = in.color * final_light * tilt_multiplier;

    return vec4<f32>(lit_color, alpha);
}
