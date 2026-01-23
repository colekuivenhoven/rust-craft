struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_position: vec4<f32>,
    near: f32,
    far: f32,
    _padding: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var depth_texture: texture_depth_2d;

@group(1) @binding(1)
var depth_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) light_level: f32,
    @location(4) alpha: f32,  // Base alpha - unused as I'm calculating dynamically
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) frag_pos: vec3<f32>,
    @location(3) light_level: f32,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_position = vec4<f32>(model.position, 1.0);
    out.clip_position = camera.view_proj * world_position;
    out.color = model.color;
    out.normal = model.normal;
    out.frag_pos = model.position;
    out.light_level = model.light_level;
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
    let absorption_coefficient = 0.1;  // How quickly light is absorbed (higher = more opaque faster)
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

    let lit_color = in.color * total_light;

    return vec4<f32>(lit_color, alpha);
}