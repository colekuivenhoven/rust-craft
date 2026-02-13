// Damage flash post-processing shader
// Red tint + strong chromatic aberration that fades out

struct Uniforms {
    intensity: f32,
    time: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var t_screen: texture_2d<f32>;
@group(0) @binding(2) var s_screen: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Full-screen triangle (no vertex buffer needed)
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
    let intensity = uniforms.intensity;
    let time = uniforms.time;

    // Subtle UV distortion for visceral impact
    let distort_strength = intensity * 0.008;
    let distorted_uv = in.uv + vec2<f32>(
        sin(in.uv.y * 20.0 + time * 8.0) * distort_strength,
        cos(in.uv.x * 20.0 + time * 8.0) * distort_strength
    );

    // --- CHROMATIC ABERRATION ---
    // Strong aberration that scales with intensity and distance from center
    let center_dist = length(distorted_uv - vec2<f32>(0.5, 0.5));
    let aber_strength = 0.04 * intensity * (0.5 + center_dist);

    // Offset direction radiates outward from center
    let aber_dir = normalize(distorted_uv - vec2<f32>(0.5, 0.5));
    let aber_offset = aber_dir * aber_strength;

    let r = textureSample(t_screen, s_screen, distorted_uv + aber_offset).r;
    let g = textureSample(t_screen, s_screen, distorted_uv).g;
    let b = textureSample(t_screen, s_screen, distorted_uv - aber_offset).b;

    let scene_color = vec3<f32>(r, g, b);

    // --- RED TINT ---
    // Push colors toward red proportional to intensity
    let red_tint = vec3<f32>(0.9, 0.15, 0.1);
    let tinted = mix(scene_color, scene_color * red_tint + red_tint * 0.15, intensity * 0.7);

    // --- VIGNETTE ---
    // Strong darkened edges
    let vignette = 1.0 - center_dist * intensity * 1.2;
    let final_color = tinted * max(vignette, 0.0);

    return vec4<f32>(final_color, 1.0);
}
