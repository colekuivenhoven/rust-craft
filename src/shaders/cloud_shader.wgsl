struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_position: vec4<f32>,
    near: f32,
    far: f32,
    _padding: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Fog uniform
struct FogUniform {
    start: f32,
    end: f32,
    enabled: f32,
    use_square_fog: f32,
};

@group(1) @binding(0)
var<uniform> fog: FogUniform;

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
    @location(1) frag_pos: vec3<f32>,
    @location(2) alpha: f32,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_position = vec4<f32>(model.position, 1.0);
    out.clip_position = camera.view_proj * world_position;
    out.color = model.color;
    out.frag_pos = model.position;
    out.alpha = model.alpha;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simple flat cloud rendering - color and alpha are baked into vertices
    let base_color = in.color;

    // Apply distance fog with DOUBLED fog distances for clouds
    var final_alpha = in.alpha;

    if (fog.enabled > 0.5) {
        let offset = in.frag_pos - camera.view_position.xyz;

        // Choose distance calculation based on use_square_fog setting
        var distance: f32;
        if (fog.use_square_fog > 0.5) {
            // Chebyshev distance: square fog pattern
            distance = max(abs(offset.x), abs(offset.z));
        } else {
            // Euclidean distance: circular fog pattern
            distance = length(offset);
        }

        // DOUBLED fog distances so clouds are visible beyond normal fog
        let cloud_fog_start = fog.start * 2.0;
        let cloud_fog_end = fog.end * 2.0;

        // Calculate fog factor
        let fog_factor = clamp((distance - cloud_fog_start) / (cloud_fog_end - cloud_fog_start), 0.0, 1.0);

        // Apply fog to alpha
        final_alpha = in.alpha * (1.0 - fog_factor);
    }

    return vec4<f32>(base_color, final_alpha);
}
