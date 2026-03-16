// Shadow map shader — renders depth from the sun's perspective.
// Used to generate the shadow map that the main shader samples.

struct SunCamera {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> sun_camera: SunCamera;

// Texture atlas for alpha testing (leaves, grass, etc.)
@group(1) @binding(0)
var texture_atlas: texture_2d<f32>;

@group(1) @binding(1)
var texture_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) light_level: f32,
    @location(4) alpha: f32,
    @location(5) uv: vec2<f32>,
    @location(6) tex_index: u32,
    @location(7) ao: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) alpha: f32,
    @location(2) @interpolate(flat) tex_index: u32,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = sun_camera.view_proj * vec4<f32>(model.position, 1.0);
    out.uv = model.uv;
    out.alpha = model.alpha;
    out.tex_index = model.tex_index;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) {
    let base_idx = in.tex_index & 0xFFFFu;

    // For textured blocks, alpha-test so leaves/grass don't cast solid shadows
    if (base_idx != 255u) {
        let tex_color = textureSample(texture_atlas, texture_sampler, in.uv);
        let alpha = tex_color.a * in.alpha;
        if (alpha < 0.5) {
            discard;
        }
    }
    // No color output — we only care about the depth buffer
}
