struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_position: vec4<f32>,
    near: f32,
    far: f32,
    _padding: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) light_level: f32,
    @location(4) alpha: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) frag_pos: vec3<f32>,
    @location(3) light_level: f32,
    @location(4) alpha: f32,
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
    out.alpha = model.alpha;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Minimum ambient light (visibility even in complete darkness)
    let min_ambient = 0.05;

    // Voxel lighting from BFS propagation (0.0 to 1.0)
    let voxel_light = in.light_level;

    // Apply curve to make light falloff more pleasing
    let curved_light = pow(voxel_light, 1.4);

    // Slight directional component for depth perception
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let directional = max(dot(in.normal, light_dir), 0.0) * 0.15;

    // Combine lighting: voxel light is primary, directional adds depth
    let total_light = min_ambient + curved_light * 0.9 + directional * voxel_light;

    let lit_color = in.color * total_light;

    return vec4<f32>(lit_color, in.alpha);
}
