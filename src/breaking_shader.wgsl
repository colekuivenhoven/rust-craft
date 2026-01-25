// Breaking overlay shader - renders destruction animation on block faces

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
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    out.uv = model.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(texture_atlas, texture_sampler, in.uv);

    // Breaking textures typically use black/dark colors for cracks
    // We want to darken the underlying block, so we use the texture's darkness
    // Discard nearly transparent pixels
    if (tex_color.a < 0.1) {
        discard;
    }

    // Darken effect - the crack texture is typically black on transparent
    // We want to show the cracks as dark overlay
    let crack_intensity = 1.0 - tex_color.r; // Darker parts = more cracked
    let overlay_color = vec3<f32>(0.0, 0.0, 0.0); // Black cracks
    let overlay_alpha = tex_color.a * 0.7; // Semi-transparent overlay

    return vec4<f32>(overlay_color, overlay_alpha);
}
