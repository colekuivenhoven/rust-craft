// Camera-rotation motion blur post-processing shader
// Applies directional blur based on camera yaw/pitch velocity

struct Uniforms {
    blur_dir: vec2<f32>,   // screen-space blur direction (yaw -> horizontal, pitch -> vertical)
    blur_strength: f32,    // overall blur intensity (0 = passthrough)
    _padding: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var t_screen: texture_2d<f32>;
@group(0) @binding(2) var s_screen: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Full-screen triangle
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let strength = uniforms.blur_strength;

    // Early out: no blur needed
    if (strength < 0.001) {
        return textureSample(t_screen, s_screen, in.uv);
    }

    let blur_dir = uniforms.blur_dir * strength;

    // 8 samples along the blur direction, centered on the current pixel
    var color = vec4<f32>(0.0);
    color += textureSample(t_screen, s_screen, in.uv + blur_dir * -0.4286);
    color += textureSample(t_screen, s_screen, in.uv + blur_dir * -0.3061);
    color += textureSample(t_screen, s_screen, in.uv + blur_dir * -0.1837);
    color += textureSample(t_screen, s_screen, in.uv + blur_dir * -0.0612);
    color += textureSample(t_screen, s_screen, in.uv + blur_dir *  0.0612);
    color += textureSample(t_screen, s_screen, in.uv + blur_dir *  0.1837);
    color += textureSample(t_screen, s_screen, in.uv + blur_dir *  0.3061);
    color += textureSample(t_screen, s_screen, in.uv + blur_dir *  0.4286);
    color /= 8.0;

    return color;
}
