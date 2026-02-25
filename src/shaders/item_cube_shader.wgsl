// Item cube shader — renders 3 isometric faces of a block icon in the hotbar.
// Each face uses the block's texture atlas tile; falls back to vertex color when
// use_texture == 0.0 (i.e. the block has TEX_NONE = 255).

struct VertexInput {
    @location(0) position:    vec2<f32>,
    @location(1) uv:          vec2<f32>,
    @location(2) color:       vec4<f32>,
    @location(3) use_texture: f32,
    @location(4) _pad:        f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv:          vec2<f32>,
    @location(1) color:       vec4<f32>,
    @location(2) use_texture: f32,
};

@group(0) @binding(0) var t_atlas: texture_2d<f32>;
@group(0) @binding(1) var s_atlas: sampler;

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(model.position, 0.0, 1.0);
    out.uv            = model.uv;
    out.color         = model.color;
    out.use_texture   = model.use_texture;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if in.use_texture > 0.5 {
        // Sample atlas tile and multiply by shade tint stored in color
        let tex = textureSample(t_atlas, s_atlas, in.uv);
        return vec4<f32>(tex.rgb * in.color.rgb, tex.a);
    } else {
        // No texture: use pre-shaded fallback color directly
        return in.color;
    }
}
