// Sky shader — renders a fullscreen quad behind all geometry.
// Produces: dynamic sky gradient, sun disc with glow, and procedural stars at night.

struct SkyUniforms {
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    sun_dir: vec4<f32>,          // normalised, points toward sun
    sun_color: vec4<f32>,        // pre-multiplied color * brightness
    // [sun_intensity, night_ambient, shadow_strength, time_of_day]
    params: vec4<f32>,
    // [sun_radius_rad, sun_glow_falloff, star_density, star_brightness]
    sky_params: vec4<f32>,
    // [star_twinkle_speed, 0, 0, total_time]
    sky_params2: vec4<f32>,
    // sky colors packed: zenith_day, horizon_day, zenith_night, sunset
    zenith_day: vec4<f32>,       // rgb + pad
    horizon_day: vec4<f32>,
    zenith_night: vec4<f32>,
    sunset_color: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> sky: SkyUniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

// Fullscreen triangle (3 verts, no vertex buffer needed)
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Oversized triangle covering the entire screen
    let x = f32(i32(vertex_index) / 2) * 4.0 - 1.0;
    let y = f32(i32(vertex_index) % 2) * 4.0 - 1.0;
    out.clip_position = vec4<f32>(x, y, 1.0, 1.0); // z=1.0 = far plane
    out.ndc = vec2<f32>(x, y);
    return out;
}

// ============================================================================
// Procedural star field
// ============================================================================

// Integer hash for star placement
fn hash_star(p: vec2<f32>) -> f32 {
    let n = i32(p.x * 127.1 + p.y * 311.7);
    let m = (n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff;
    return f32(m) / 2147483647.0;
}

fn hash2(p: vec2<f32>) -> vec2<f32> {
    let a = hash_star(p);
    let b = hash_star(p + vec2<f32>(37.0, 113.0));
    return vec2<f32>(a, b);
}

// Compute stars on a single cube-map face.
// uv: 2D position on the face, face_id: unique offset per face to avoid repeats.
fn star_face(uv: vec2<f32>, face_id: f32, time: f32, density: f32, brightness: f32, twinkle_speed: f32) -> f32 {
    // Grid resolution per face — sqrt(density/6) cells per unit on each face
    let cells_per_unit = sqrt(density / 6.0);
    let grid = uv * cells_per_unit;
    let cell = floor(grid);
    let frac_pos = fract(grid);

    var star = 0.0;

    // 3x3 neighborhood check for edge continuity
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let neighbor = cell + vec2<f32>(f32(dx), f32(dy));
            // Offset by face_id so each cube face has unique stars
            let h = hash2(neighbor + vec2<f32>(face_id * 100.0, face_id * 73.0));

            // Only some cells have stars
            if (h.x > 0.7) {
                let star_pos = vec2<f32>(h.x, h.y) * 0.8 + 0.1;
                let diff = frac_pos - star_pos - vec2<f32>(f32(dx), f32(dy));

                // Chebyshev distance → square shape, always axis-aligned on the face
                let dist = max(abs(diff.x), abs(diff.y));

                // Hard-edged square pixel
                let size = 0.03 + h.x * 0.02;
                let point = step(dist, size);

                // Twinkle
                let phase = h.y * 6.283;
                let twinkle = 0.5 + 0.5 * sin(time * twinkle_speed + phase);

                star += point * twinkle * brightness * (0.5 + h.x * 0.5);
            }
        }
    }
    return star;
}

// Returns star brightness for a given view direction using cube-map projection.
// Each cube face provides a flat, undistorted 2D grid so stars are always
// perfect squares that face the player regardless of viewing angle.
fn star_field(dir: vec3<f32>, time: f32, density: f32, brightness: f32, twinkle_speed: f32) -> f32 {
    let ad = abs(dir);

    // Determine dominant axis → cube face
    var uv: vec2<f32>;
    var face_id: f32;

    if (ad.x >= ad.y && ad.x >= ad.z) {
        // ±X face
        uv = dir.yz / ad.x;
        face_id = select(1.0, 0.0, dir.x > 0.0);
    } else if (ad.y >= ad.x && ad.y >= ad.z) {
        // ±Y face
        uv = dir.xz / ad.y;
        face_id = select(3.0, 2.0, dir.y > 0.0);
    } else {
        // ±Z face
        uv = dir.xy / ad.z;
        face_id = select(5.0, 4.0, dir.z > 0.0);
    }

    return star_face(uv, face_id, time, density, brightness, twinkle_speed);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Reconstruct world-space ray direction from NDC
    let ndc = vec4<f32>(in.ndc.x, in.ndc.y, 1.0, 1.0);
    let world_pos = sky.inv_view_proj * ndc;
    let ray_dir = normalize(world_pos.xyz / world_pos.w - sky.camera_pos.xyz);

    let sun_dir = normalize(sky.sun_dir.xyz);
    let sun_intensity = sky.params.x;
    let time_of_day = sky.params.w;
    let total_time = sky.sky_params2.w;

    let sun_radius = sky.sky_params.x;
    let sun_glow_falloff = sky.sky_params.y;
    let star_density = sky.sky_params.z;
    let star_brightness = sky.sky_params.w;
    let twinkle_speed = sky.sky_params2.x;

    // ── Sun elevation factor ─────────────────────────────────────────────
    // sun_dir.y > 0 means sun is above horizon
    let sun_elevation = sun_dir.y; // -1 to 1
    let day_factor = smoothstep(-0.1, 0.2, sun_elevation); // 0=night, 1=day

    // ── Sunset/sunrise factor ────────────────────────────────────────────
    // Strongest when sun is near horizon (elevation ~0)
    let sunset_factor = smoothstep(0.3, 0.0, abs(sun_elevation)) * smoothstep(-0.2, 0.0, sun_elevation);

    // ── Sky gradient ─────────────────────────────────────────────────────
    // Vertical gradient: horizon (y~0) to zenith (y~1)
    let up_factor = max(ray_dir.y, 0.0);
    let horizon_factor = 1.0 - up_factor;

    // Day sky
    let day_zenith = sky.zenith_day.rgb;
    let day_horizon = sky.horizon_day.rgb;
    let day_sky = mix(day_horizon, day_zenith, pow(up_factor, 0.5));

    // Night sky
    let night_zenith = sky.zenith_night.rgb;
    let night_horizon = night_zenith * 1.5; // slightly lighter near horizon
    let night_sky = mix(night_horizon, night_zenith, pow(up_factor, 0.3));

    // Blend day/night
    var sky_color = mix(night_sky, day_sky, day_factor);

    // Sunset tint near horizon
    let sunset_tint = sky.sunset_color.rgb;
    // Apply sunset color more strongly near the horizon and in the sun's direction
    let sun_horiz_dir = normalize(vec2<f32>(sun_dir.x, sun_dir.z));
    let ray_horiz_dir = normalize(vec2<f32>(ray_dir.x, ray_dir.z) + vec2<f32>(0.0001, 0.0001));
    let sun_horiz_dot = max(dot(sun_horiz_dir, ray_horiz_dir), 0.0);
    let sunset_mask = sunset_factor * horizon_factor * (0.3 + 0.7 * pow(sun_horiz_dot, 2.0));
    sky_color = mix(sky_color, sunset_tint, sunset_mask * 0.8);

    // Below horizon — darken
    if (ray_dir.y < 0.0) {
        let below = smoothstep(0.0, -0.3, ray_dir.y);
        sky_color = mix(sky_color, sky_color * 0.3, below);
    }

    // ── Sun disc ─────────────────────────────────────────────────────────
    let cos_angle = dot(ray_dir, sun_dir);
    let angle = acos(clamp(cos_angle, -1.0, 1.0));

    // Sharp disc
    let disc = smoothstep(sun_radius * 1.1, sun_radius * 0.9, angle);
    // Soft glow around disc
    let glow = pow(max(1.0 - angle / (sun_radius * sun_glow_falloff), 0.0), 3.0);

    let sun_visual = sky.sun_color.rgb * (disc + glow * 0.3);
    sky_color += sun_visual;

    // ── Stars ────────────────────────────────────────────────────────────
    // Stars visible when sun is below horizon (night)
    let star_visibility = smoothstep(0.1, -0.1, sun_elevation);
    if (star_visibility > 0.01 && ray_dir.y > -0.05) {
        let stars = star_field(ray_dir, total_time, star_density, star_brightness, twinkle_speed);
        sky_color += vec3<f32>(stars * star_visibility);
    }

    return vec4<f32>(sky_color, 1.0);
}
