// Bloom post-processing shader
// Passes: horizontal blur → vertical blur → additive composite
// (Emissive pixels are rendered directly into bloom texture by the emissive pipeline in shader.wgsl)

@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Full-screen triangle (shared by all passes)
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// --- Pass 0: Downsample (full-res → quarter-res) ---
@fragment
fn fs_downsample(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_input, s_input, in.uv);
}

// --- Pass 1: Horizontal gaussian blur (9-tap) ---
@fragment
fn fs_blur_h(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(t_input));
    let pixel = vec2<f32>(2.5 / tex_size.x, 0.0);

    var color = textureSample(t_input, s_input, in.uv) * 0.227027;
    color += textureSample(t_input, s_input, in.uv + pixel * 1.0) * 0.1945946;
    color += textureSample(t_input, s_input, in.uv - pixel * 1.0) * 0.1945946;
    color += textureSample(t_input, s_input, in.uv + pixel * 2.0) * 0.1216216;
    color += textureSample(t_input, s_input, in.uv - pixel * 2.0) * 0.1216216;
    color += textureSample(t_input, s_input, in.uv + pixel * 3.0) * 0.0540541;
    color += textureSample(t_input, s_input, in.uv - pixel * 3.0) * 0.0540541;
    color += textureSample(t_input, s_input, in.uv + pixel * 4.0) * 0.0162162;
    color += textureSample(t_input, s_input, in.uv - pixel * 4.0) * 0.0162162;

    return color;
}

// --- Pass 2: Vertical gaussian blur (9-tap) ---
@fragment
fn fs_blur_v(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(t_input));
    let pixel = vec2<f32>(0.0, 2.5 / tex_size.y);

    var color = textureSample(t_input, s_input, in.uv) * 0.227027;
    color += textureSample(t_input, s_input, in.uv + pixel * 1.0) * 0.1945946;
    color += textureSample(t_input, s_input, in.uv - pixel * 1.0) * 0.1945946;
    color += textureSample(t_input, s_input, in.uv + pixel * 2.0) * 0.1216216;
    color += textureSample(t_input, s_input, in.uv - pixel * 2.0) * 0.1216216;
    color += textureSample(t_input, s_input, in.uv + pixel * 3.0) * 0.0540541;
    color += textureSample(t_input, s_input, in.uv - pixel * 3.0) * 0.0540541;
    color += textureSample(t_input, s_input, in.uv + pixel * 4.0) * 0.0162162;
    color += textureSample(t_input, s_input, in.uv - pixel * 4.0) * 0.0162162;

    return color;
}

// --- Pass 3: Additive composite ---
// Outputs bloom color; pipeline uses additive blending to overlay onto scene
@fragment
fn fs_composite(in: VertexOutput) -> @location(0) vec4<f32> {
    let bloom = textureSample(t_input, s_input, in.uv);
    let intensity = 0.8;
    return vec4<f32>(bloom.rgb * intensity, 1.0);
}
