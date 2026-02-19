// Modal sand-background shader.
// Renders a textured quad with UV coordinates that tile the sand texture.
// Used as the fill for any modal panel background.

@group(0) @binding(0) var t_sand: texture_2d<f32>;
@group(0) @binding(1) var s_sand: sampler;

struct VertexInput {
    @location(0) position:   vec2<f32>,  // clip-space (-1..1)
    @location(1) tex_coords: vec2<f32>,  // UV (may exceed 1.0 for tiling)
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0)       tex_coords:    vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    out.tex_coords    = in.tex_coords;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sampler is set to Repeat, so UVs > 1 tile automatically
    return textureSample(t_sand, s_sand, in.tex_coords);
}
