// Underwater post-processing shader
// Renders a full-screen distorted, tinted effect

// WE NEED UNIFORMS FOR ANIMATION
struct Uniforms {
    time: f32,
}

// BINDINGS
// Ensure your Rust code binds the Uniform buffer at 0, 
// the Scene Texture at 1, and a Sampler at 2.
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var t_screen: texture_2d<f32>;
@group(0) @binding(2) var s_screen: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Full-screen triangle (Unchanged)
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
    // 1. WAVINESS (Multi-layered distortion)
    // -----------------------------------------------------------
    let time = uniforms.time;
    var uv = in.uv;

    // Layer 1: Large, slow "swell"
    // We disturb X based on Y, and Y based on X to create a swirling liquid feel
    let swell_x = sin(uv.y * 3.0 + time * 0.5) * 0.005;
    let swell_y = cos(uv.x * 3.0 + time * 0.5) * 0.005;

    // Layer 2: Faster, smaller "ripples"
    let ripple_x = sin(uv.y * 15.0 + time * 1.5) * 0.004;
    let ripple_y = cos(uv.x * 15.0 + time * 1.5) * 0.004;

    // Apply distortion to UVs
    let distorted_uv = uv + vec2<f32>(swell_x + ripple_x, swell_y + ripple_y);


    // 2. CHROMATIC ABERRATION
    // -----------------------------------------------------------
    // Calculate how far we are from the center (stronger effect at edges)
    let center_dist = length(distorted_uv - vec2<f32>(0.5, 0.5));
    
    // Offset strength increases toward edges
    let aber_strength = 0.015 * center_dist; 
    
    // Sample the color channels at slightly different positions
    // Red pulls in one direction, Blue in the opposite
    let r = textureSample(t_screen, s_screen, distorted_uv + vec2<f32>(aber_strength, 0.0)).r;
    let g = textureSample(t_screen, s_screen, distorted_uv).g;
    let b = textureSample(t_screen, s_screen, distorted_uv - vec2<f32>(aber_strength, 0.0)).b;

    let scene_color = vec3<f32>(r, g, b);

    // 3. UNDERWATER TINT & VIGNETTE (Your original logic)
    // -----------------------------------------------------------
    let vignette = 1.0 - center_dist * 0.6; // Increased intensity slightly
    let tint_color = vec3<f32>(0.18, 0.4, 0.75); // Slightly deeper blue
    
    // Mix the scene color with the tint
    // We multiply scene by tint to simulate light absorption, 
    // then mix a bit of solid blue for "fog" density.
    let absorbed_light = scene_color * tint_color * 1.1;
    let final_color = mix(absorbed_light, tint_color * 0.5, 0.3);

    // Apply vignette darkness
    return vec4<f32>(final_color * vignette, 1.0);
}