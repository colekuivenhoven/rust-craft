// Pause background blur shader.
// Renders a full-screen triangle (no vertex buffer), samples scene_texture
// with a 2-pass-style cross-kernel blur and darkens the result.
//
// Tunables (mirrored as Rust consts in modal.rs):
//   BLUR_STEP_PX  – pixel radius of each sample step
//   DARKEN_FACTOR – how much to darken the blurred frame

const BLUR_STEP_PX: f32 = 3.5;
const DARKEN_FACTOR: f32 = 0.45;

@group(0) @binding(0) var t_scene: texture_2d<f32>;
@group(0) @binding(1) var s_scene: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0)       uv:            vec2<f32>,
};

// Full-screen triangle: vertex ID 0/1/2 → covers the entire NDC cube
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var out: VertexOutput;
    out.clip_position = vec4<f32>(positions[vid], 0.0, 1.0);
    out.uv            = uvs[vid];
    return out;
}

// 13-tap cross-pattern Gaussian approximation.
// Two rings: ±1 and ±2 steps in both axes, weighted toward center.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(t_scene));
    let step1 = BLUR_STEP_PX       / tex_size;
    let step2 = BLUR_STEP_PX * 2.0 / tex_size;
    let step3 = BLUR_STEP_PX * 3.0 / tex_size;

    let uv = in.uv;

    // Center (weight 0.20)
    var col = textureSample(t_scene, s_scene, uv) * 0.20;

    // Ring 1 (4 taps, weight 0.14 each)
    col += textureSample(t_scene, s_scene, uv + vec2( step1.x,  0.0    )) * 0.14;
    col += textureSample(t_scene, s_scene, uv + vec2(-step1.x,  0.0    )) * 0.14;
    col += textureSample(t_scene, s_scene, uv + vec2( 0.0,      step1.y)) * 0.14;
    col += textureSample(t_scene, s_scene, uv + vec2( 0.0,     -step1.y)) * 0.14;

    // Ring 2 (4 taps, weight 0.065 each)
    col += textureSample(t_scene, s_scene, uv + vec2( step2.x,  0.0    )) * 0.065;
    col += textureSample(t_scene, s_scene, uv + vec2(-step2.x,  0.0    )) * 0.065;
    col += textureSample(t_scene, s_scene, uv + vec2( 0.0,      step2.y)) * 0.065;
    col += textureSample(t_scene, s_scene, uv + vec2( 0.0,     -step2.y)) * 0.065;

    // Ring 3 (4 taps, weight 0.02 each)
    col += textureSample(t_scene, s_scene, uv + vec2( step3.x,  0.0    )) * 0.02;
    col += textureSample(t_scene, s_scene, uv + vec2(-step3.x,  0.0    )) * 0.02;
    col += textureSample(t_scene, s_scene, uv + vec2( 0.0,      step3.y)) * 0.02;
    col += textureSample(t_scene, s_scene, uv + vec2( 0.0,     -step3.y)) * 0.02;
    // Total weight ≈ 0.20 + 4×0.14 + 4×0.065 + 4×0.02 = 1.00

    col = vec4<f32>(col.rgb * DARKEN_FACTOR, 1.0);
    return col;
}
