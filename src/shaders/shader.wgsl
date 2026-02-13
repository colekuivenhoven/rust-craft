struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_position: vec4<f32>,
    near: f32,
    far: f32,
    _padding: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Texture atlas bindings
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
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) frag_pos: vec3<f32>,
    @location(3) light_level: f32,
    @location(4) alpha: f32,
    @location(5) uv: vec2<f32>,
    @location(6) @interpolate(flat) tex_index: u32,
    @location(7) ao: f32,
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
    out.uv = model.uv;
    out.tex_index = model.tex_index;
    out.ao = model.ao;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Determine base color and alpha: texture or vertex color
    var base_color: vec3<f32>;
    var alpha: f32 = in.alpha;

    // Extract packed fields from tex_index:
    //   bits 0-15:  base texture index
    //   bits 16-23: overlay texture index + 1 (0 = no overlay)
    //   bits 24-31: overlay tint parameter (0-255 -> 0.0-1.0)
    let base_idx = in.tex_index & 0xFFFFu;
    let overlay_idx_raw = (in.tex_index >> 16u) & 0xFFu;
    let tint_byte = (in.tex_index >> 24u) & 0xFFu;

    if (base_idx == 255u) {
        // No texture - use vertex color (fallback for untextured blocks)
        base_color = in.color;
    } else {
        // Sample base texture from atlas
        let tex_color = textureSample(texture_atlas, texture_sampler, in.uv);

        if (overlay_idx_raw > 0u) {
            // Overlay texture present (e.g., grass_side blended over dirt)
            let overlay_idx = overlay_idx_raw - 1u;
            // Extract local UV within the base tile using known tile position
            let base_col = f32(base_idx % 16u);
            let base_row = f32(base_idx / 16u);
            let local_u = in.uv.x * 16.0 - base_col;
            let local_v = in.uv.y * 16.0 - base_row;
            // Compute overlay atlas UV from overlay tile index + local coords
            let o_col = f32(overlay_idx % 16u);
            let o_row = f32(overlay_idx / 16u);
            let overlay_uv = vec2<f32>(
                (o_col + local_u) / 16.0,
                (o_row + local_v) / 16.0
            );
            let overlay_tex = textureSample(texture_atlas, texture_sampler, overlay_uv);

            // Reconstruct grass tint from packed parameter (green <-> orange spectrum)
            let t = f32(tint_byte) / 255.0;
            let overlay_tint = vec3<f32>(0.3 + t * 0.7, 0.95 - t * 0.3, 0.2 - t * 0.1);

            // Blend: base texture with vertex color where overlay is transparent,
            // overlay with reconstructed tint where overlay is opaque
            base_color = mix(tex_color.rgb * in.color, overlay_tex.rgb * overlay_tint, overlay_tex.a);
            alpha = 1.0; // Combined face is always fully opaque
        } else {
            // Standard single texture
            // Tint texture by vertex color (white = no tint, colored = leaf tint, etc.)
            base_color = tex_color.rgb * in.color;
            // Use texture alpha for transparency (e.g., ice blocks)
            alpha = tex_color.a * in.alpha;

            // Alpha test: discard fully transparent pixels so they don't write to the depth buffer
            // (prevents leaf texture transparent pixels from blocking faces behind them)
            if (alpha < 0.5) {
                discard;
            }
        }
    }

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

    // Apply ambient occlusion
    let final_light = total_light * in.ao;

    let lit_color = base_color * final_light;

    return vec4<f32>(lit_color, alpha);
}
