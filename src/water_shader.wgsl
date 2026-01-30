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
    @location(4) edge_factor: f32,      // Edge proximity for shore foam (0.0 = interior, 1.0 = touching solid)
    @location(5) wave_tilt_x: f32,      // X component of wave normal for directional shading
    @location(6) wave_height: f32,      // Current wave displacement for foam on crests
    @location(7) world_pos_xz: vec2<f32>, // World XZ position for foam noise sampling
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

// ============================================================================
// Foam and Specular Constants
// ============================================================================

const FOAM_COLOR: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);  // Slightly blue-white foam
const FOAM_PIXEL_SIZE: f32 = 0.125;                      // Pixelation size for foam (1/8 of a block = 8 pixels per block)
const FOAM_LINE_COUNT: i32 = 4;                          // Number of foam lines from the edge
const FOAM_BASE_DENSITY: f32 = 0.85;                      // Density of first line (1.0 = solid, 0.0 = empty)
const FOAM_DENSITY_FALLOFF: f32 = 0.50;                  // Multiplier per line (0.5 = each line has half the density of previous)

// Pixelated noise - snaps position to grid before sampling
fn pixel_noise(pos: vec2<f32>, pixel_size: f32, time: f32) -> f32 {
    // Snap to pixel grid
    let snapped = floor(pos / pixel_size) * pixel_size;
    // Add time-based animation (also snapped for consistent pixelation)
    let time_step = floor(time * 3.0) * 0.33;  // Step every ~0.33 seconds
    let animated_pos = snapped + vec2<f32>(time_step * 0.5, time_step * 0.3);
    return hash_int(i32(animated_pos.x * 100.0), i32(animated_pos.y * 100.0));
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
    out.edge_factor = model.ao;  // ao field repurposed as edge factor for water
    out.wave_height = wave_height;
    out.world_pos_xz = model.position.xz;
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
    let max_alpha = 0.9;               // Maximum opacity (very deep water)

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

    // Apply X-axis directional shading for wave visualization
    // Tilt towards -X = darker, tilt towards +X = lighter
    let tilt_shading = clamp(in.wave_tilt_x * 2.0, -0.5, 2.9);  // Scale and clamp the effect
    let tilt_multiplier = 1.0 + tilt_shading;  // Range: 0.5 to 1.9

    var lit_color = in.color * total_light * tilt_multiplier;

    // ========================================================================
    // FOAM CALCULATION (Pixelated)
    // ========================================================================

    // Get fractional position within the block (0.0 to 1.0)
    let frac_pos = fract(in.world_pos_xz);

    // Pixelated noise for foam pattern
    let foam_noise_val = pixel_noise(in.world_pos_xz, FOAM_PIXEL_SIZE, wave.time);

    // ---- EDGE FOAM: Pixelated lines along coastline edges ----
    // edge_factor encodes which edges have solid neighbors as a bitmask:
    // bit 0 (1): neg_x (-X edge), bit 1 (2): pos_x (+X edge)
    // bit 2 (4): neg_z (-Z edge), bit 3 (8): pos_z (+Z edge)
    var edge_foam = 0.0;

    if (in.edge_factor > 0.001) {
        // Decode bitmask (multiply by 16 to get original 0-15 value)
        let flags = i32(in.edge_factor * 16.0 + 0.5);  // +0.5 for rounding
        let has_neg_x = (flags & 1) != 0;
        let has_pos_x = (flags & 2) != 0;
        let has_neg_z = (flags & 4) != 0;
        let has_pos_z = (flags & 8) != 0;

        // Distance from each edge (in blocks, 0.0 = at edge)
        let dist_neg_x = frac_pos.x;           // Distance from -X edge (left)
        let dist_pos_x = 1.0 - frac_pos.x;     // Distance from +X edge (right)
        let dist_neg_z = frac_pos.y;           // Distance from -Z edge (back)
        let dist_pos_z = 1.0 - frac_pos.y;     // Distance from +Z edge (front)

        // Calculate foam for each edge that has a solid neighbor
        // For each edge: determine which "line" we're in based on distance
        let line_width = FOAM_PIXEL_SIZE;
        let max_foam_dist = line_width * f32(FOAM_LINE_COUNT);

        // Check each edge and calculate foam contribution
        var min_dist_to_solid_edge = 999.0;

        if (has_neg_x && dist_neg_x < max_foam_dist) {
            min_dist_to_solid_edge = min(min_dist_to_solid_edge, dist_neg_x);
        }
        if (has_pos_x && dist_pos_x < max_foam_dist) {
            min_dist_to_solid_edge = min(min_dist_to_solid_edge, dist_pos_x);
        }
        if (has_neg_z && dist_neg_z < max_foam_dist) {
            min_dist_to_solid_edge = min(min_dist_to_solid_edge, dist_neg_z);
        }
        if (has_pos_z && dist_pos_z < max_foam_dist) {
            min_dist_to_solid_edge = min(min_dist_to_solid_edge, dist_pos_z);
        }

        // If we're within foam range of a solid edge
        if (min_dist_to_solid_edge < max_foam_dist) {
            // Determine which line we're in (0 = closest to edge)
            let line_index = i32(min_dist_to_solid_edge / line_width);

            // Calculate density for this line (decreases with each line)
            let line_density = FOAM_BASE_DENSITY * pow(FOAM_DENSITY_FALLOFF, f32(line_index));

            // Threshold for this pixel: lower density = higher threshold = fewer pixels
            let noise_threshold = 1.0 - line_density;

            // Pixel is foam if noise exceeds threshold
            edge_foam = step(noise_threshold, foam_noise_val);
        }
    }

    // ---- WAVE FOAM: Appears in FRONT of waves (rising slope) ----
    // wave_tilt_x > 0 means wave is rising in +X direction (front of wave)
    // We want foam on the rising front, plus a bit on the crest
    let wave_slope = in.wave_tilt_x;  // Positive = rising, negative = falling
    let rising_front = max(wave_slope * 2.0, 0.0);  // Only positive slopes
    let on_crest = smoothstep(0.15, 0.25, in.wave_height);  // Small amount on crests

    // Combine front and crest, weighted toward front
    let wave_foam_factor = min(rising_front * 0.8 + on_crest * 0.3, 1.0);

    // Pixelated wave foam
    let wave_noise_threshold = 0.5 - wave_foam_factor * 0.3;
    let wave_foam = step(wave_noise_threshold, foam_noise_val) * wave_foam_factor;

    // Combine foam types (max to avoid over-brightening)
    let total_foam = max(wave_foam, edge_foam);

    // Apply foam to color - foam is bright white overlay
    lit_color = mix(lit_color, FOAM_COLOR * total_light, total_foam * 0.9);

    // Increase alpha slightly where there's foam for better visibility
    let foam_alpha_boost = total_foam * 0.15;

    return vec4<f32>(lit_color, min(alpha + foam_alpha_boost, 0.95));
}
