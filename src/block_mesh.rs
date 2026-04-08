use cgmath::Vector3;
use crate::block::BlockType;

// ============================================================================
// Cross Model Constants (grass tufts, foliage)
// ============================================================================
pub const CROSS_MODEL_SCALE_MIN: f32 = 0.5;   // Minimum scale factor for cross model quads (1.0 = full block size)
pub const CROSS_MODEL_SCALE_MAX: f32 = 1.0;   // Maximum scale factor for cross model quads
pub const CROSS_MODEL_OFFSET_MAX: f32 = 0.5; // Maximum offset applied to quad endpoints for angle variation (in blocks). Higher values make the crossing angle deviate more from 90 degrees.

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub normal: [f32; 3],
    pub light_level: f32,
    pub alpha: f32,  // Transparency (1.0 = opaque, 0.0 = fully transparent)
    pub uv: [f32; 2],  // Texture coordinates
    pub tex_index: u32,  // Texture index in atlas (255 = use color fallback)
    pub ao: f32,  // Ambient occlusion (0.0 = fully occluded, 1.0 = no occlusion)
}

impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position: location 0
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color: location 1
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // normal: location 2
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // light_level: location 3
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 9]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32,
                },
                // alpha: location 4
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 10]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32,
                },
                // uv: location 5
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 11]>() as wgpu::BufferAddress,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // tex_index: location 6
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 13]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Uint32,
                },
                // ao: location 7
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 13]>() as wgpu::BufferAddress
                        + std::mem::size_of::<u32>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

/// AO neighbor offsets for each face and vertex.
/// Format: [face_index][vertex_index] = [[side1], [side2], [corner]]
/// Each offset is [dx, dy, dz] relative to the block position.
/// Vertices ordered: bottom-left, bottom-right, top-right, top-left
///
/// Key insight: For each face, we check neighbors in the direction the face points.
/// E.g., for front face (+Z), all checks are at z+1 (in front of the block).
pub const AO_OFFSETS: [[[[i32; 3]; 3]; 4]; 6] = [
    // Face 0: Front (+Z) - face at z+1, check neighbors at z+1
    [
        [[-1, 0, 1], [0, -1, 1], [-1, -1, 1]],  // BL
        [[1, 0, 1], [0, -1, 1], [1, -1, 1]],    // BR
        [[1, 0, 1], [0, 1, 1], [1, 1, 1]],      // TR
        [[-1, 0, 1], [0, 1, 1], [-1, 1, 1]],    // TL
    ],
    // Face 1: Back (-Z) - face at z=0, check neighbors at z-1
    [
        [[1, 0, -1], [0, -1, -1], [1, -1, -1]],    // BL
        [[-1, 0, -1], [0, -1, -1], [-1, -1, -1]],  // BR
        [[-1, 0, -1], [0, 1, -1], [-1, 1, -1]],    // TR
        [[1, 0, -1], [0, 1, -1], [1, 1, -1]],      // TL
    ],
    // Face 2: Top (+Y) - face at y+1, check neighbors at y+1
    [
        [[-1, 1, 0], [0, 1, 1], [-1, 1, 1]],    // vertex at x, z+1
        [[1, 1, 0], [0, 1, 1], [1, 1, 1]],      // vertex at x+1, z+1
        [[1, 1, 0], [0, 1, -1], [1, 1, -1]],    // vertex at x+1, z
        [[-1, 1, 0], [0, 1, -1], [-1, 1, -1]],  // vertex at x, z
    ],
    // Face 3: Bottom (-Y) - face at y=0, check neighbors at y-1
    [
        [[-1, -1, 0], [0, -1, -1], [-1, -1, -1]],  // vertex at x, z
        [[1, -1, 0], [0, -1, -1], [1, -1, -1]],    // vertex at x+1, z
        [[1, -1, 0], [0, -1, 1], [1, -1, 1]],      // vertex at x+1, z+1
        [[-1, -1, 0], [0, -1, 1], [-1, -1, 1]],    // vertex at x, z+1
    ],
    // Face 4: Right (+X) - face at x+1, check neighbors at x+1
    [
        [[1, 0, 1], [1, -1, 0], [1, -1, 1]],    // z+1, y
        [[1, 0, -1], [1, -1, 0], [1, -1, -1]],  // z, y
        [[1, 0, -1], [1, 1, 0], [1, 1, -1]],    // z, y+1
        [[1, 0, 1], [1, 1, 0], [1, 1, 1]],      // z+1, y+1
    ],
    // Face 5: Left (-X) - face at x=0, check neighbors at x-1
    [
        [[-1, 0, -1], [-1, -1, 0], [-1, -1, -1]],  // z, y
        [[-1, 0, 1], [-1, -1, 0], [-1, -1, 1]],    // z+1, y
        [[-1, 0, 1], [-1, 1, 0], [-1, 1, 1]],      // z+1, y+1
        [[-1, 0, -1], [-1, 1, 0], [-1, 1, -1]],    // z, y+1
    ],
];

/// AO strength: 0.0 = no AO effect, 1.0 = full AO effect.
/// Lower values create softer, more blended shadows.
pub const AO_STRENGTH: f32 = 0.5;

// ============================================================================
// Water Wave Constants
// ============================================================================

/// Maximum wave height in blocks (vertical displacement)
pub const WAVE_AMPLITUDE: f32 = 0.0;

/// Base frequency of waves (lower = larger wavelength)
pub const WAVE_FREQUENCY: f32 = 0.2;

/// Wave movement speed multiplier
pub const WAVE_SPEED: f32 = 1.0;

/// Number of noise octaves for layered waves
pub const WAVE_OCTAVES: u32 = 2;

/// Frequency multiplier between octaves (lacunarity)
pub const WAVE_LACUNARITY: f32 = 2.0;

/// Amplitude multiplier between octaves (persistence)
pub const WAVE_PERSISTENCE: f32 = 0.5;

/// Calculates ambient occlusion value for a vertex based on neighboring blocks.
/// Returns a value from (darkest) to 1.0 (no occlusion), smoothed by AO_STRENGTH.
pub fn calculate_ao(side1_solid: bool, side2_solid: bool, corner_solid: bool) -> f32 {
    let level = if side1_solid && side2_solid {
        0  // Both sides solid = fully occluded (corner is irrelevant)
    } else {
        3 - (side1_solid as u8 + side2_solid as u8 + corner_solid as u8)
    };
    // Map levels 0-3 to base brightness values
    let raw_ao = [0.25, 0.55, 0.8, 1.0][level as usize];

    // Blend toward 1.0 based on AO_STRENGTH (lower strength = softer shadows)
    raw_ao + (1.0 - raw_ao) * (1.0 - AO_STRENGTH)
}

/// Creates vertices for a single face of a cube with per-vertex AO and smooth lighting
pub fn create_face_vertices(pos: Vector3<f32>, block_type: BlockType, face_index: usize, light_values: [f32; 4], tex_index: u32, uvs: [[f32; 2]; 4], ao_values: [f32; 4]) -> [Vertex; 4] {
    create_face_vertices_with_alpha(pos, block_type, face_index, light_values, 1.0, tex_index, uvs, ao_values)
}

/// Creates vertices for a single face with a custom color tint and per-vertex smooth lighting
pub fn create_face_vertices_tinted(pos: Vector3<f32>, face_index: usize, light_values: [f32; 4], tex_index: u32, uvs: [[f32; 2]; 4], ao_values: [f32; 4], tint: [f32; 3]) -> [Vertex; 4] {
    let color = tint;
    let x = pos.x;
    let y = pos.y;
    let z = pos.z;
    let alpha = 1.0;

    match face_index {
        0 => [ // Front face (+Z)
            Vertex { position: [x, y, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level: light_values[0], alpha, uv: uvs[0], tex_index, ao: ao_values[0] },
            Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level: light_values[1], alpha, uv: uvs[1], tex_index, ao: ao_values[1] },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level: light_values[2], alpha, uv: uvs[2], tex_index, ao: ao_values[2] },
            Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level: light_values[3], alpha, uv: uvs[3], tex_index, ao: ao_values[3] },
        ],
        1 => [ // Back face (-Z)
            Vertex { position: [x + 1.0, y, z], color, normal: [0.0, 0.0, -1.0], light_level: light_values[0], alpha, uv: uvs[0], tex_index, ao: ao_values[0] },
            Vertex { position: [x, y, z], color, normal: [0.0, 0.0, -1.0], light_level: light_values[1], alpha, uv: uvs[1], tex_index, ao: ao_values[1] },
            Vertex { position: [x, y + 1.0, z], color, normal: [0.0, 0.0, -1.0], light_level: light_values[2], alpha, uv: uvs[2], tex_index, ao: ao_values[2] },
            Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [0.0, 0.0, -1.0], light_level: light_values[3], alpha, uv: uvs[3], tex_index, ao: ao_values[3] },
        ],
        2 => [ // Top face (+Y)
            Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [0.0, 1.0, 0.0], light_level: light_values[0], alpha, uv: uvs[0], tex_index, ao: ao_values[0] },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [0.0, 1.0, 0.0], light_level: light_values[1], alpha, uv: uvs[1], tex_index, ao: ao_values[1] },
            Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [0.0, 1.0, 0.0], light_level: light_values[2], alpha, uv: uvs[2], tex_index, ao: ao_values[2] },
            Vertex { position: [x, y + 1.0, z], color, normal: [0.0, 1.0, 0.0], light_level: light_values[3], alpha, uv: uvs[3], tex_index, ao: ao_values[3] },
        ],
        3 => [ // Bottom face (-Y)
            Vertex { position: [x, y, z], color, normal: [0.0, -1.0, 0.0], light_level: light_values[0], alpha, uv: uvs[0], tex_index, ao: ao_values[0] },
            Vertex { position: [x + 1.0, y, z], color, normal: [0.0, -1.0, 0.0], light_level: light_values[1], alpha, uv: uvs[1], tex_index, ao: ao_values[1] },
            Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [0.0, -1.0, 0.0], light_level: light_values[2], alpha, uv: uvs[2], tex_index, ao: ao_values[2] },
            Vertex { position: [x, y, z + 1.0], color, normal: [0.0, -1.0, 0.0], light_level: light_values[3], alpha, uv: uvs[3], tex_index, ao: ao_values[3] },
        ],
        4 => [ // Right face (+X)
            Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [1.0, 0.0, 0.0], light_level: light_values[0], alpha, uv: uvs[0], tex_index, ao: ao_values[0] },
            Vertex { position: [x + 1.0, y, z], color, normal: [1.0, 0.0, 0.0], light_level: light_values[1], alpha, uv: uvs[1], tex_index, ao: ao_values[1] },
            Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [1.0, 0.0, 0.0], light_level: light_values[2], alpha, uv: uvs[2], tex_index, ao: ao_values[2] },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [1.0, 0.0, 0.0], light_level: light_values[3], alpha, uv: uvs[3], tex_index, ao: ao_values[3] },
        ],
        _ => [ // Left face (-X)
            Vertex { position: [x, y, z], color, normal: [-1.0, 0.0, 0.0], light_level: light_values[0], alpha, uv: uvs[0], tex_index, ao: ao_values[0] },
            Vertex { position: [x, y, z + 1.0], color, normal: [-1.0, 0.0, 0.0], light_level: light_values[1], alpha, uv: uvs[1], tex_index, ao: ao_values[1] },
            Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [-1.0, 0.0, 0.0], light_level: light_values[2], alpha, uv: uvs[2], tex_index, ao: ao_values[2] },
            Vertex { position: [x, y + 1.0, z], color, normal: [-1.0, 0.0, 0.0], light_level: light_values[3], alpha, uv: uvs[3], tex_index, ao: ao_values[3] },
        ],
    }
}

/// Creates vertices for a single face of a cube with custom alpha, per-vertex AO and smooth lighting
pub fn create_face_vertices_with_alpha(pos: Vector3<f32>, block_type: BlockType, face_index: usize, light_values: [f32; 4], alpha: f32, tex_index: u32, uvs: [[f32; 2]; 4], ao_values: [f32; 4]) -> [Vertex; 4] {
    // White tint for textured blocks (shader multiplies tex_color * vertex_color).
    // Fallback color only used when tex_index == 255 (no texture).
    let color = if tex_index != 255 {
        [1.0, 1.0, 1.0]
    } else {
        block_type.get_color()
    };
    let x = pos.x;
    let y = pos.y;
    let z = pos.z;

    match face_index {
        0 => [ // Front face (+Z)
            Vertex { position: [x, y, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level: light_values[0], alpha, uv: uvs[0], tex_index, ao: ao_values[0] },
            Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level: light_values[1], alpha, uv: uvs[1], tex_index, ao: ao_values[1] },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level: light_values[2], alpha, uv: uvs[2], tex_index, ao: ao_values[2] },
            Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level: light_values[3], alpha, uv: uvs[3], tex_index, ao: ao_values[3] },
        ],
        1 => [ // Back face (-Z)
            Vertex { position: [x + 1.0, y, z], color, normal: [0.0, 0.0, -1.0], light_level: light_values[0], alpha, uv: uvs[0], tex_index, ao: ao_values[0] },
            Vertex { position: [x, y, z], color, normal: [0.0, 0.0, -1.0], light_level: light_values[1], alpha, uv: uvs[1], tex_index, ao: ao_values[1] },
            Vertex { position: [x, y + 1.0, z], color, normal: [0.0, 0.0, -1.0], light_level: light_values[2], alpha, uv: uvs[2], tex_index, ao: ao_values[2] },
            Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [0.0, 0.0, -1.0], light_level: light_values[3], alpha, uv: uvs[3], tex_index, ao: ao_values[3] },
        ],
        2 => [ // Top face (+Y)
            Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [0.0, 1.0, 0.0], light_level: light_values[0], alpha, uv: uvs[0], tex_index, ao: ao_values[0] },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [0.0, 1.0, 0.0], light_level: light_values[1], alpha, uv: uvs[1], tex_index, ao: ao_values[1] },
            Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [0.0, 1.0, 0.0], light_level: light_values[2], alpha, uv: uvs[2], tex_index, ao: ao_values[2] },
            Vertex { position: [x, y + 1.0, z], color, normal: [0.0, 1.0, 0.0], light_level: light_values[3], alpha, uv: uvs[3], tex_index, ao: ao_values[3] },
        ],
        3 => [ // Bottom face (-Y)
            Vertex { position: [x, y, z], color, normal: [0.0, -1.0, 0.0], light_level: light_values[0], alpha, uv: uvs[0], tex_index, ao: ao_values[0] },
            Vertex { position: [x + 1.0, y, z], color, normal: [0.0, -1.0, 0.0], light_level: light_values[1], alpha, uv: uvs[1], tex_index, ao: ao_values[1] },
            Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [0.0, -1.0, 0.0], light_level: light_values[2], alpha, uv: uvs[2], tex_index, ao: ao_values[2] },
            Vertex { position: [x, y, z + 1.0], color, normal: [0.0, -1.0, 0.0], light_level: light_values[3], alpha, uv: uvs[3], tex_index, ao: ao_values[3] },
        ],
        4 => [ // Right face (+X)
            Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [1.0, 0.0, 0.0], light_level: light_values[0], alpha, uv: uvs[0], tex_index, ao: ao_values[0] },
            Vertex { position: [x + 1.0, y, z], color, normal: [1.0, 0.0, 0.0], light_level: light_values[1], alpha, uv: uvs[1], tex_index, ao: ao_values[1] },
            Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [1.0, 0.0, 0.0], light_level: light_values[2], alpha, uv: uvs[2], tex_index, ao: ao_values[2] },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [1.0, 0.0, 0.0], light_level: light_values[3], alpha, uv: uvs[3], tex_index, ao: ao_values[3] },
        ],
        _ => [ // Left face (-X)
            Vertex { position: [x, y, z], color, normal: [-1.0, 0.0, 0.0], light_level: light_values[0], alpha, uv: uvs[0], tex_index, ao: ao_values[0] },
            Vertex { position: [x, y, z + 1.0], color, normal: [-1.0, 0.0, 0.0], light_level: light_values[1], alpha, uv: uvs[1], tex_index, ao: ao_values[1] },
            Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [-1.0, 0.0, 0.0], light_level: light_values[2], alpha, uv: uvs[2], tex_index, ao: ao_values[2] },
            Vertex { position: [x, y + 1.0, z], color, normal: [-1.0, 0.0, 0.0], light_level: light_values[3], alpha, uv: uvs[3], tex_index, ao: ao_values[3] },
        ],
    }
}

/// Creates vertices for a water face with wave factor encoded in the alpha channel.
/// The alpha field stores how much each vertex should be affected by wave animation:
/// - 1.0 = fully wave-affected (surface vertices that should move)
/// - 0.0 = not wave-affected (vertices that should stay fixed)
///
/// The ao field is repurposed for water to store edge_factor for foam rendering:
/// - 1.0 = vertex is adjacent to a solid block (shore foam)
/// - 0.0 = vertex is interior water (no shore foam)
///
/// `corner_heights` contains the absolute Y height for each of the 4 top-face corners:
///   [0] = (-X, +Z), [1] = (+X, +Z), [2] = (+X, -Z), [3] = (-X, -Z)
/// `corner_wave_scales` controls how much each corner is affected by wave animation (0.0-1.0).
///   Scales proportionally with water level so shallow water has smaller waves.
/// For non-surface water, pass y+1.0 for heights and 0.0 for wave scales.
pub fn create_water_face_vertices(
    pos: Vector3<f32>,
    face_index: usize,
    light_values: [f32; 4],
    tex_index: u32,
    uvs: [[f32; 2]; 4],
    edge_factors: [f32; 4],
    is_surface_water: bool,
    corner_heights: [f32; 4],
    corner_wave_scales: [f32; 4],
) -> [Vertex; 4] {
    let color = BlockType::Water.get_color();
    let x = pos.x;
    let y = pos.y;
    let z = pos.z;

    // h0=(-X,+Z), h1=(+X,+Z), h2=(+X,-Z), h3=(-X,-Z)
    let [h0, h1, h2, h3] = corner_heights;
    let [w0, w1, w2, w3] = corner_wave_scales;

    // Map per-corner wave scales to per-vertex wave factors based on face type.
    // Top face: each vertex uses its corner's wave scale.
    // Side faces: bottom vertices = 0.0, top vertices = their corner's wave scale.
    // Bottom face: all 0.0.
    let wave_factors: [f32; 4] = match face_index {
        // Top face: [(-X,+Z), (+X,+Z), (+X,-Z), (-X,-Z)]
        2 => [w0, w1, w2, w3],
        3 => [0.0, 0.0, 0.0, 0.0],
        // Front (+Z): top verts are v2=(+X,+Z)=w1, v3=(-X,+Z)=w0
        0 if is_surface_water => [0.0, 0.0, w1, w0],
        // Back (-Z): top verts are v2=(-X,-Z)=w3, v3=(+X,-Z)=w2
        1 if is_surface_water => [0.0, 0.0, w3, w2],
        // Right (+X): top verts are v2=(+X,-Z)=w2, v3=(+X,+Z)=w1
        4 if is_surface_water => [0.0, 0.0, w2, w1],
        // Left (-X): top verts are v2=(-X,+Z)=w0, v3=(-X,-Z)=w3
        _ if is_surface_water => [0.0, 0.0, w0, w3],
        _ => [0.0, 0.0, 0.0, 0.0],
    };

    match face_index {
        0 => [ // Front face (+Z) - top verts: h1(+X,+Z), h0(-X,+Z)
            Vertex { position: [x, y, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level: light_values[0], alpha: wave_factors[0], uv: uvs[0], tex_index, ao: edge_factors[0] },
            Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level: light_values[1], alpha: wave_factors[1], uv: uvs[1], tex_index, ao: edge_factors[1] },
            Vertex { position: [x + 1.0, h1, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level: light_values[2], alpha: wave_factors[2], uv: uvs[2], tex_index, ao: edge_factors[2] },
            Vertex { position: [x, h0, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level: light_values[3], alpha: wave_factors[3], uv: uvs[3], tex_index, ao: edge_factors[3] },
        ],
        1 => [ // Back face (-Z) - top verts: h3(-X,-Z), h2(+X,-Z)
            Vertex { position: [x + 1.0, y, z], color, normal: [0.0, 0.0, -1.0], light_level: light_values[0], alpha: wave_factors[0], uv: uvs[0], tex_index, ao: edge_factors[0] },
            Vertex { position: [x, y, z], color, normal: [0.0, 0.0, -1.0], light_level: light_values[1], alpha: wave_factors[1], uv: uvs[1], tex_index, ao: edge_factors[1] },
            Vertex { position: [x, h3, z], color, normal: [0.0, 0.0, -1.0], light_level: light_values[2], alpha: wave_factors[2], uv: uvs[2], tex_index, ao: edge_factors[2] },
            Vertex { position: [x + 1.0, h2, z], color, normal: [0.0, 0.0, -1.0], light_level: light_values[3], alpha: wave_factors[3], uv: uvs[3], tex_index, ao: edge_factors[3] },
        ],
        2 => [ // Top face (+Y) - all 4 corner heights
            Vertex { position: [x, h0, z + 1.0], color, normal: [0.0, 1.0, 0.0], light_level: light_values[0], alpha: wave_factors[0], uv: uvs[0], tex_index, ao: edge_factors[0] },
            Vertex { position: [x + 1.0, h1, z + 1.0], color, normal: [0.0, 1.0, 0.0], light_level: light_values[1], alpha: wave_factors[1], uv: uvs[1], tex_index, ao: edge_factors[1] },
            Vertex { position: [x + 1.0, h2, z], color, normal: [0.0, 1.0, 0.0], light_level: light_values[2], alpha: wave_factors[2], uv: uvs[2], tex_index, ao: edge_factors[2] },
            Vertex { position: [x, h3, z], color, normal: [0.0, 1.0, 0.0], light_level: light_values[3], alpha: wave_factors[3], uv: uvs[3], tex_index, ao: edge_factors[3] },
        ],
        3 => [ // Bottom face (-Y)
            Vertex { position: [x, y, z], color, normal: [0.0, -1.0, 0.0], light_level: light_values[0], alpha: wave_factors[0], uv: uvs[0], tex_index, ao: edge_factors[0] },
            Vertex { position: [x + 1.0, y, z], color, normal: [0.0, -1.0, 0.0], light_level: light_values[1], alpha: wave_factors[1], uv: uvs[1], tex_index, ao: edge_factors[1] },
            Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [0.0, -1.0, 0.0], light_level: light_values[2], alpha: wave_factors[2], uv: uvs[2], tex_index, ao: edge_factors[2] },
            Vertex { position: [x, y, z + 1.0], color, normal: [0.0, -1.0, 0.0], light_level: light_values[3], alpha: wave_factors[3], uv: uvs[3], tex_index, ao: edge_factors[3] },
        ],
        4 => [ // Right face (+X) - top verts: h2(+X,-Z), h1(+X,+Z)
            Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [1.0, 0.0, 0.0], light_level: light_values[0], alpha: wave_factors[0], uv: uvs[0], tex_index, ao: edge_factors[0] },
            Vertex { position: [x + 1.0, y, z], color, normal: [1.0, 0.0, 0.0], light_level: light_values[1], alpha: wave_factors[1], uv: uvs[1], tex_index, ao: edge_factors[1] },
            Vertex { position: [x + 1.0, h2, z], color, normal: [1.0, 0.0, 0.0], light_level: light_values[2], alpha: wave_factors[2], uv: uvs[2], tex_index, ao: edge_factors[2] },
            Vertex { position: [x + 1.0, h1, z + 1.0], color, normal: [1.0, 0.0, 0.0], light_level: light_values[3], alpha: wave_factors[3], uv: uvs[3], tex_index, ao: edge_factors[3] },
        ],
        _ => [ // Left face (-X) - top verts: h0(-X,+Z), h3(-X,-Z)
            Vertex { position: [x, y, z], color, normal: [-1.0, 0.0, 0.0], light_level: light_values[0], alpha: wave_factors[0], uv: uvs[0], tex_index, ao: edge_factors[0] },
            Vertex { position: [x, y, z + 1.0], color, normal: [-1.0, 0.0, 0.0], light_level: light_values[1], alpha: wave_factors[1], uv: uvs[1], tex_index, ao: edge_factors[1] },
            Vertex { position: [x, h0, z + 1.0], color, normal: [-1.0, 0.0, 0.0], light_level: light_values[2], alpha: wave_factors[2], uv: uvs[2], tex_index, ao: edge_factors[2] },
            Vertex { position: [x, h3, z], color, normal: [-1.0, 0.0, 0.0], light_level: light_values[3], alpha: wave_factors[3], uv: uvs[3], tex_index, ao: edge_factors[3] },
        ],
    }
}

/// Creates vertices for a cross model (two intersecting quads forming an X shape).
/// Used for grass tufts and similar foliage. Each quad is rendered double-sided.
/// `tint` is the RGB color tint applied to the texture.
/// `hash` is a deterministic seed derived from world position for per-instance variation.
pub fn create_cross_model_vertices(pos: Vector3<f32>, light_level: f32, tex_index: u32, uvs: [[f32; 2]; 4], tint: [f32; 3], hash: u32) -> ([Vertex; 16], [u16; 24]) {
    let x = pos.x;
    let y = pos.y;
    let z = pos.z;
    let ao = 1.0;
    let alpha = 1.0;

    // Derive deterministic random values from hash
    // Scale: varies quad size within CROSS_MODEL_SCALE_MIN..CROSS_MODEL_SCALE_MAX
    let scale_frac = ((hash >> 0) & 0xFF) as f32 / 255.0;
    let scale = CROSS_MODEL_SCALE_MIN + scale_frac * (CROSS_MODEL_SCALE_MAX - CROSS_MODEL_SCALE_MIN);
    // Offsets: shift each quad endpoint for angle variation
    let off1_frac = (((hash >> 8) & 0xFF) as f32 / 255.0) * 2.0 - 1.0;  // -1..1
    let off2_frac = (((hash >> 16) & 0xFF) as f32 / 255.0) * 2.0 - 1.0; // -1..1
    let off1 = off1_frac * CROSS_MODEL_OFFSET_MAX;
    let off2 = off2_frac * CROSS_MODEL_OFFSET_MAX;

    // Center the scaled quad within the block
    let inset = (1.0 - scale) * 0.5;
    let lo = inset;
    let hi = 1.0 - inset;
    let h = scale; // quad height = scale

    // Quad 1 endpoints on XZ plane (diagonal + offset for angle variation)
    // Corner A near (0,0), corner B near (1,1), with offset shifting the z components
    let a1x = x + lo;
    let a1z = z + lo + off1;
    let b1x = x + hi;
    let b1z = z + hi + off1;
    // Quad 2 endpoints (other diagonal + independent offset)
    let a2x = x + hi;
    let a2z = z + lo + off2;
    let b2x = x + lo;
    let b2z = z + hi + off2;

    let y_lo = y;
    let y_hi = y + h;

    // Compute face normals from cross product for correct lighting
    fn face_normal(ax: f32, az: f32, bx: f32, bz: f32, h: f32) -> [f32; 3] {
        // edge1 = B - A along bottom, edge2 = up vector
        let dx = bx - ax;
        let dz = bz - az;
        // normal = edge1 × up = (dx, 0, dz) × (0, h, 0) = (-dz*h, 0, dx*h)
        let len = (dz * dz + dx * dx).sqrt() * h;
        if len < 1e-6 { return [0.0, 1.0, 0.0]; }
        [-dz * h / len, 0.0, dx * h / len]
    }

    let n1 = face_normal(a1x, a1z, b1x, b1z, h);
    let n1_back = [-n1[0], -n1[1], -n1[2]];
    let n2 = face_normal(a2x, a2z, b2x, b2z, h);
    let n2_back = [-n2[0], -n2[1], -n2[2]];

    let vertices = [
        // Quad 1 front
        Vertex { position: [a1x, y_lo, a1z], color: tint, normal: n1, light_level, alpha, uv: uvs[0], tex_index, ao },
        Vertex { position: [b1x, y_lo, b1z], color: tint, normal: n1, light_level, alpha, uv: uvs[1], tex_index, ao },
        Vertex { position: [b1x, y_hi, b1z], color: tint, normal: n1, light_level, alpha, uv: uvs[2], tex_index, ao },
        Vertex { position: [a1x, y_hi, a1z], color: tint, normal: n1, light_level, alpha, uv: uvs[3], tex_index, ao },
        // Quad 1 back (reversed winding)
        Vertex { position: [b1x, y_lo, b1z], color: tint, normal: n1_back, light_level, alpha, uv: uvs[0], tex_index, ao },
        Vertex { position: [a1x, y_lo, a1z], color: tint, normal: n1_back, light_level, alpha, uv: uvs[1], tex_index, ao },
        Vertex { position: [a1x, y_hi, a1z], color: tint, normal: n1_back, light_level, alpha, uv: uvs[2], tex_index, ao },
        Vertex { position: [b1x, y_hi, b1z], color: tint, normal: n1_back, light_level, alpha, uv: uvs[3], tex_index, ao },
        // Quad 2 front
        Vertex { position: [a2x, y_lo, a2z], color: tint, normal: n2, light_level, alpha, uv: uvs[0], tex_index, ao },
        Vertex { position: [b2x, y_lo, b2z], color: tint, normal: n2, light_level, alpha, uv: uvs[1], tex_index, ao },
        Vertex { position: [b2x, y_hi, b2z], color: tint, normal: n2, light_level, alpha, uv: uvs[2], tex_index, ao },
        Vertex { position: [a2x, y_hi, a2z], color: tint, normal: n2, light_level, alpha, uv: uvs[3], tex_index, ao },
        // Quad 2 back (reversed winding)
        Vertex { position: [b2x, y_lo, b2z], color: tint, normal: n2_back, light_level, alpha, uv: uvs[0], tex_index, ao },
        Vertex { position: [a2x, y_lo, a2z], color: tint, normal: n2_back, light_level, alpha, uv: uvs[1], tex_index, ao },
        Vertex { position: [a2x, y_hi, a2z], color: tint, normal: n2_back, light_level, alpha, uv: uvs[2], tex_index, ao },
        Vertex { position: [b2x, y_hi, b2z], color: tint, normal: n2_back, light_level, alpha, uv: uvs[3], tex_index, ao },
    ];

    let indices = [
        0, 1, 2, 2, 3, 0,       // Quad 1 front
        4, 5, 6, 6, 7, 4,       // Quad 1 back
        8, 9, 10, 10, 11, 8,    // Quad 2 front
        12, 13, 14, 14, 15, 12, // Quad 2 back
    ];

    (vertices, indices)
}

/// Creates vertices for a single thin vine panel on one face of a vine block.
/// The panel is 1/16 block thick, pressed against the face adjacent to `wall_dir`:
///   0 = wall to -X (panel faces +X)   1 = wall to +X (panel faces -X)
///   2 = wall to -Z (panel faces +Z)   3 = wall to +Z (panel faces -Z)
/// Only one face is emitted (single-sided) — the wall behind is always solid.
pub fn create_vine_face_vertices(
    pos: Vector3<f32>,
    wall_dir: u8,
    light_level: f32,
    tex_index: u32,
    uvs: [[f32; 2]; 4],
    tint: [f32; 3],
) -> ([Vertex; 4], [u16; 12]) {
    const T: f32 = 0.0625; // 1/16 block
    let x = pos.x;
    let y = pos.y;
    let z = pos.z;
    let color = tint;
    let ao = 1.0;
    let alpha = 1.0;

    let vertices = match wall_dir {
        0 => { // Wall at -X → panel at x+T, facing +X
            let p = x + T;
            [
                Vertex { position: [p, y,     z    ], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv: uvs[0], tex_index, ao },
                Vertex { position: [p, y,     z+1.0], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv: uvs[1], tex_index, ao },
                Vertex { position: [p, y+1.0, z+1.0], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv: uvs[2], tex_index, ao },
                Vertex { position: [p, y+1.0, z    ], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv: uvs[3], tex_index, ao },
            ]
        },
        1 => { // Wall at +X → panel at x+1-T, facing -X
            let p = x + 1.0 - T;
            [
                Vertex { position: [p, y,     z+1.0], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv: uvs[0], tex_index, ao },
                Vertex { position: [p, y,     z    ], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv: uvs[1], tex_index, ao },
                Vertex { position: [p, y+1.0, z    ], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv: uvs[2], tex_index, ao },
                Vertex { position: [p, y+1.0, z+1.0], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv: uvs[3], tex_index, ao },
            ]
        },
        2 => { // Wall at -Z → panel at z+T, facing +Z
            let p = z + T;
            [
                Vertex { position: [x,     y,     p], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv: uvs[0], tex_index, ao },
                Vertex { position: [x+1.0, y,     p], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv: uvs[1], tex_index, ao },
                Vertex { position: [x+1.0, y+1.0, p], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv: uvs[2], tex_index, ao },
                Vertex { position: [x,     y+1.0, p], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv: uvs[3], tex_index, ao },
            ]
        },
        _ => { // Wall at +Z → panel at z+1-T, facing -Z
            let p = z + 1.0 - T;
            [
                Vertex { position: [x+1.0, y,     p], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv: uvs[0], tex_index, ao },
                Vertex { position: [x,     y,     p], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv: uvs[1], tex_index, ao },
                Vertex { position: [x,     y+1.0, p], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv: uvs[2], tex_index, ao },
                Vertex { position: [x+1.0, y+1.0, p], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv: uvs[3], tex_index, ao },
            ]
        },
    };

    (vertices, [
        0, 1, 2, 2, 3, 0, // Front face
        0, 3, 2, 2, 1, 0  // Back face
    ])
}

pub fn create_cube_vertices(pos: Vector3<f32>, block_type: BlockType, light_level: f32) -> Vec<Vertex> {
    use crate::texture::TEX_NONE;
    let color = block_type.get_color();
    let x = pos.x;
    let y = pos.y;
    let z = pos.z;
    let alpha = 1.0;
    let uv = [0.0, 0.0]; // Default UV for non-textured blocks
    let tex_index = TEX_NONE;
    let ao = 1.0; // No AO for standalone cubes

    vec![
        // Front face
        Vertex { position: [x, y, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv, tex_index, ao },
        // Back face
        Vertex { position: [x + 1.0, y, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x, y, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x, y + 1.0, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv, tex_index, ao },
        // Top face
        Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [0.0, 1.0, 0.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [0.0, 1.0, 0.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [0.0, 1.0, 0.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x, y + 1.0, z], color, normal: [0.0, 1.0, 0.0], light_level, alpha, uv, tex_index, ao },
        // Bottom face
        Vertex { position: [x, y, z], color, normal: [0.0, -1.0, 0.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x + 1.0, y, z], color, normal: [0.0, -1.0, 0.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [0.0, -1.0, 0.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x, y, z + 1.0], color, normal: [0.0, -1.0, 0.0], light_level, alpha, uv, tex_index, ao },
        // Right face
        Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x + 1.0, y, z], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv, tex_index, ao },
        // Left face
        Vertex { position: [x, y, z], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x, y, z + 1.0], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x, y + 1.0, z], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv, tex_index, ao },
    ]
}

pub const CUBE_INDICES: &[u16] = &[
    0, 1, 2, 2, 3, 0,       // Front
    4, 5, 6, 6, 7, 4,       // Back
    8, 9, 10, 10, 11, 8,    // Top
    12, 13, 14, 14, 15, 12, // Bottom
    16, 17, 18, 18, 19, 16, // Right
    20, 21, 22, 22, 23, 20, // Left
];

/// Creates vertices for a scaled cube centered at the given position.
/// Used for dropped items (mini-blocks).
/// - `center`: The center point of the cube
/// - `block_type`: The type of block (for color and texture)
/// - `scale`: The size of the cube (0.25 for 1/4 size mini-blocks)
/// - `light_level`: Brightness level (0.0-1.0)
pub fn create_scaled_cube_vertices(
    center: cgmath::Point3<f32>,
    block_type: BlockType,
    scale: f32,
    light_level: f32,
) -> Vec<Vertex> {
    use crate::texture::get_face_uvs;

    let face_textures = block_type.get_face_textures(false); // Dropped items are always "exposed"
    // Use white tint for textured blocks (shader multiplies tex * color).
    // For leaves, this gives a neutral white tint for dropped items/previews.
    let has_texture = face_textures.get_for_face(0) != 255;
    let color = if has_texture { [1.0, 1.0, 1.0] } else { block_type.get_color() };
    let half = scale * 0.5;
    let x = center.x - half;
    let y = center.y - half;
    let z = center.z - half;
    let s = scale; // Size
    let alpha = 1.0;
    let ao = 1.0;

    // Get texture indices for each face
    let tex_front = face_textures.get_for_face(0);  // Front (+Z) - sides
    let tex_back = face_textures.get_for_face(1);   // Back (-Z) - sides
    let tex_top = face_textures.get_for_face(2);    // Top (+Y) - top
    let tex_bottom = face_textures.get_for_face(3); // Bottom (-Y) - bottom
    let tex_right = face_textures.get_for_face(4);  // Right (+X) - sides
    let tex_left = face_textures.get_for_face(5);   // Left (-X) - sides

    // Get UVs for each face
    let uvs_front = get_face_uvs(tex_front);
    let uvs_back = get_face_uvs(tex_back);
    let uvs_top = get_face_uvs(tex_top);
    let uvs_bottom = get_face_uvs(tex_bottom);
    let uvs_right = get_face_uvs(tex_right);
    let uvs_left = get_face_uvs(tex_left);

    vec![
        // Front face (+Z)
        Vertex { position: [x, y, z + s], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv: uvs_front[0], tex_index: tex_front, ao },
        Vertex { position: [x + s, y, z + s], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv: uvs_front[1], tex_index: tex_front, ao },
        Vertex { position: [x + s, y + s, z + s], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv: uvs_front[2], tex_index: tex_front, ao },
        Vertex { position: [x, y + s, z + s], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv: uvs_front[3], tex_index: tex_front, ao },
        // Back face (-Z)
        Vertex { position: [x + s, y, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv: uvs_back[0], tex_index: tex_back, ao },
        Vertex { position: [x, y, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv: uvs_back[1], tex_index: tex_back, ao },
        Vertex { position: [x, y + s, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv: uvs_back[2], tex_index: tex_back, ao },
        Vertex { position: [x + s, y + s, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv: uvs_back[3], tex_index: tex_back, ao },
        // Top face (+Y)
        Vertex { position: [x, y + s, z + s], color, normal: [0.0, 1.0, 0.0], light_level, alpha, uv: uvs_top[0], tex_index: tex_top, ao },
        Vertex { position: [x + s, y + s, z + s], color, normal: [0.0, 1.0, 0.0], light_level, alpha, uv: uvs_top[1], tex_index: tex_top, ao },
        Vertex { position: [x + s, y + s, z], color, normal: [0.0, 1.0, 0.0], light_level, alpha, uv: uvs_top[2], tex_index: tex_top, ao },
        Vertex { position: [x, y + s, z], color, normal: [0.0, 1.0, 0.0], light_level, alpha, uv: uvs_top[3], tex_index: tex_top, ao },
        // Bottom face (-Y)
        Vertex { position: [x, y, z], color, normal: [0.0, -1.0, 0.0], light_level, alpha, uv: uvs_bottom[0], tex_index: tex_bottom, ao },
        Vertex { position: [x + s, y, z], color, normal: [0.0, -1.0, 0.0], light_level, alpha, uv: uvs_bottom[1], tex_index: tex_bottom, ao },
        Vertex { position: [x + s, y, z + s], color, normal: [0.0, -1.0, 0.0], light_level, alpha, uv: uvs_bottom[2], tex_index: tex_bottom, ao },
        Vertex { position: [x, y, z + s], color, normal: [0.0, -1.0, 0.0], light_level, alpha, uv: uvs_bottom[3], tex_index: tex_bottom, ao },
        // Right face (+X)
        Vertex { position: [x + s, y, z + s], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv: uvs_right[0], tex_index: tex_right, ao },
        Vertex { position: [x + s, y, z], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv: uvs_right[1], tex_index: tex_right, ao },
        Vertex { position: [x + s, y + s, z], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv: uvs_right[2], tex_index: tex_right, ao },
        Vertex { position: [x + s, y + s, z + s], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv: uvs_right[3], tex_index: tex_right, ao },
        // Left face (-X)
        Vertex { position: [x, y, z], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv: uvs_left[0], tex_index: tex_left, ao },
        Vertex { position: [x, y, z + s], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv: uvs_left[1], tex_index: tex_left, ao },
        Vertex { position: [x, y + s, z + s], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv: uvs_left[2], tex_index: tex_left, ao },
        Vertex { position: [x, y + s, z], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv: uvs_left[3], tex_index: tex_left, ao },
    ]
}

/// Creates a double-sided flat panel for dropped items that are cross-model or face-panel
/// blocks in the world (GrassTuft, GrassTuftTall, Vines). The panel is oriented at 45° in
/// the XZ plane so it is visible from all four cardinal directions.
/// Returns `(vertices, indices)`.
pub fn create_flat_item_vertices(
    center: cgmath::Point3<f32>,
    block_type: BlockType,
    scale: f32,
    light_level: f32,
) -> (Vec<Vertex>, Vec<u16>) {
    use crate::texture::get_face_uvs;

    let face_textures = block_type.get_face_textures(false);
    let tex_index = face_textures.get_for_face(0);
    let uvs = get_face_uvs(tex_index);
    let color = [1.0f32, 1.0, 1.0];
    let alpha = 1.0f32;
    let ao   = 1.0f32;

    let h = scale * 0.5;
    // Diagonal extent along (1/√2, 0, 1/√2); quad width in world-space equals `scale`.
    let d = h * std::f32::consts::FRAC_1_SQRT_2;

    // Four corners of the quad (diagonal orientation in XZ at 45°).
    let bl = [center.x - d, center.y - h, center.z - d]; // bottom-left
    let br = [center.x + d, center.y - h, center.z + d]; // bottom-right
    let tr = [center.x + d, center.y + h, center.z + d]; // top-right
    let tl = [center.x - d, center.y + h, center.z - d]; // top-left

    // Normals perpendicular to the diagonal plane.
    let nf = [-std::f32::consts::FRAC_1_SQRT_2, 0.0_f32,  std::f32::consts::FRAC_1_SQRT_2]; // front
    let nb = [ std::f32::consts::FRAC_1_SQRT_2, 0.0_f32, -std::f32::consts::FRAC_1_SQRT_2]; // back

    // Front face: CCW winding when viewed from nf direction.
    // Back face: same positions, reversed winding (CCW when viewed from nb direction).
    let verts = vec![
        Vertex { position: bl, color, normal: nf, light_level, alpha, uv: uvs[0], tex_index, ao },
        Vertex { position: br, color, normal: nf, light_level, alpha, uv: uvs[1], tex_index, ao },
        Vertex { position: tr, color, normal: nf, light_level, alpha, uv: uvs[2], tex_index, ao },
        Vertex { position: tl, color, normal: nf, light_level, alpha, uv: uvs[3], tex_index, ao },
        // Back face — bl→tl→tr→br reverses the triangle winding.
        Vertex { position: bl, color, normal: nb, light_level, alpha, uv: uvs[0], tex_index, ao },
        Vertex { position: tl, color, normal: nb, light_level, alpha, uv: uvs[3], tex_index, ao },
        Vertex { position: tr, color, normal: nb, light_level, alpha, uv: uvs[2], tex_index, ao },
        Vertex { position: br, color, normal: nb, light_level, alpha, uv: uvs[1], tex_index, ao },
    ];

    let indices: Vec<u16> = vec![
        0, 1, 2, 2, 3, 0, // front
        4, 5, 6, 6, 7, 4, // back
    ];

    (verts, indices)
}

/// Creates vertices for a small particle (tiny cube) at the given position.
/// - `center`: The center point of the particle
/// - `color`: RGB color
/// - `size`: The size of the particle
/// - `alpha`: Transparency (0.0-1.0)
pub fn create_particle_vertices(
    center: cgmath::Point3<f32>,
    color: [f32; 3],
    size: f32,
    alpha: f32,
) -> Vec<Vertex> {
    use crate::texture::TEX_NONE;
    let half = size * 0.5;
    let x = center.x - half;
    let y = center.y - half;
    let z = center.z - half;
    let s = size;
    let light_level = 1.0; // Full brightness for particles
    let uv = [0.0, 0.0];
    let tex_index = TEX_NONE;
    let ao = 1.0;

    vec![
        // Front face
        Vertex { position: [x, y, z + s], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x + s, y, z + s], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x + s, y + s, z + s], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x, y + s, z + s], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv, tex_index, ao },
        // Back face
        Vertex { position: [x + s, y, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x, y, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x, y + s, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x + s, y + s, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv, tex_index, ao },
        // Top face
        Vertex { position: [x, y + s, z + s], color, normal: [0.0, 1.0, 0.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x + s, y + s, z + s], color, normal: [0.0, 1.0, 0.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x + s, y + s, z], color, normal: [0.0, 1.0, 0.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x, y + s, z], color, normal: [0.0, 1.0, 0.0], light_level, alpha, uv, tex_index, ao },
        // Bottom face
        Vertex { position: [x, y, z], color, normal: [0.0, -1.0, 0.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x + s, y, z], color, normal: [0.0, -1.0, 0.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x + s, y, z + s], color, normal: [0.0, -1.0, 0.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x, y, z + s], color, normal: [0.0, -1.0, 0.0], light_level, alpha, uv, tex_index, ao },
        // Right face
        Vertex { position: [x + s, y, z + s], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x + s, y, z], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x + s, y + s, z], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x + s, y + s, z + s], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv, tex_index, ao },
        // Left face
        Vertex { position: [x, y, z], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x, y, z + s], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x, y + s, z + s], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv, tex_index, ao },
        Vertex { position: [x, y + s, z], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv, tex_index, ao },
    ]
}

/// Creates vertices for a circular shadow (octagon approximation) at the given position.
/// The shadow is a flat shape on the XZ plane, facing upward.
/// - `center`: The center point of the shadow (should be just above ground)
/// - `radius`: The radius of the shadow circle
/// - `alpha`: Transparency (0.0-1.0, typically 0.3-0.5)
/// Returns vertices and indices for the octagon (8 triangles from center).
pub fn create_shadow_vertices(
    center: cgmath::Point3<f32>,
    radius: f32,
    alpha: f32,
) -> (Vec<Vertex>, Vec<u16>) {
    use crate::texture::TEX_NONE;
    use std::f32::consts::PI;

    let color = [0.0, 0.0, 0.0]; // Black shadow
    let light_level = 0.0; // Dark
    let uv = [0.0, 0.0];
    let tex_index = TEX_NONE;
    let ao = 1.0;
    let normal = [0.0, 1.0, 0.0]; // Facing up

    let mut vertices = Vec::with_capacity(9); // Center + 8 outer points
    let mut indices = Vec::with_capacity(24); // 8 triangles * 3 indices

    // Center vertex
    vertices.push(Vertex {
        position: [center.x, center.y, center.z],
        color,
        normal,
        light_level,
        alpha,
        uv,
        tex_index,
        ao,
    });

    // 8 outer vertices (octagon)
    for i in 0..8 {
        let angle = (i as f32) * PI / 4.0; // 45 degrees apart
        let x = center.x + angle.cos() * radius;
        let z = center.z + angle.sin() * radius;
        vertices.push(Vertex {
            position: [x, center.y, z],
            color,
            normal,
            light_level,
            alpha: 0.0, // Outer edges are fully transparent for soft falloff
            uv,
            tex_index,
            ao,
        });
    }

    // Create 8 triangles (center to each pair of adjacent outer vertices)
    // Winding order is counterclockwise when viewed from above (+Y)
    for i in 0..8 {
        let next = (i + 1) % 8;
        indices.push(0); // Center
        indices.push((next + 1) as u16); // Next outer vertex (reversed for CCW winding)
        indices.push((i + 1) as u16); // Current outer vertex
    }

    (vertices, indices)
}

// Simple 2D vertex for UI elements
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UiVertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

impl UiVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<UiVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

// Textured vertex for isometric item cube icons in the hotbar
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ItemCubeVertex {
    pub position:    [f32; 2],  // clip-space 2D position
    pub uv:          [f32; 2],  // atlas UV coordinate
    pub color:       [f32; 4],  // shade tint (textured) or pre-shaded fallback color
    pub use_texture: f32,       // 1.0 = sample atlas, 0.0 = use color directly
    pub _pad:        f32,       // alignment padding
}

impl ItemCubeVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ItemCubeVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: 36,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

// UV-mapped vertex for modal sand-texture background
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModalVertex {
    pub position:   [f32; 2],  // clip space
    pub tex_coords: [f32; 2],  // UV (wraps for tiling)
}

impl ModalVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ModalVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

// Simple 3D position vertex for outlines
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineVertex {
    pub position: [f32; 3],
}

impl LineVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

// Create outline vertices for a block at position (x, y, z)
pub fn create_block_outline(x: i32, y: i32, z: i32) -> Vec<LineVertex> {
    let x = x as f32;
    let y = y as f32;
    let z = z as f32;
    let o = 0.002; // Small offset to prevent z-fighting

    vec![
        // Bottom face edges
        LineVertex { position: [x - o, y - o, z - o] },
        LineVertex { position: [x + 1.0 + o, y - o, z - o] },
        LineVertex { position: [x + 1.0 + o, y - o, z - o] },
        LineVertex { position: [x + 1.0 + o, y - o, z + 1.0 + o] },
        LineVertex { position: [x + 1.0 + o, y - o, z + 1.0 + o] },
        LineVertex { position: [x - o, y - o, z + 1.0 + o] },
        LineVertex { position: [x - o, y - o, z + 1.0 + o] },
        LineVertex { position: [x - o, y - o, z - o] },
        // Top face edges
        LineVertex { position: [x - o, y + 1.0 + o, z - o] },
        LineVertex { position: [x + 1.0 + o, y + 1.0 + o, z - o] },
        LineVertex { position: [x + 1.0 + o, y + 1.0 + o, z - o] },
        LineVertex { position: [x + 1.0 + o, y + 1.0 + o, z + 1.0 + o] },
        LineVertex { position: [x + 1.0 + o, y + 1.0 + o, z + 1.0 + o] },
        LineVertex { position: [x - o, y + 1.0 + o, z + 1.0 + o] },
        LineVertex { position: [x - o, y + 1.0 + o, z + 1.0 + o] },
        LineVertex { position: [x - o, y + 1.0 + o, z - o] },
        // Vertical edges
        LineVertex { position: [x - o, y - o, z - o] },
        LineVertex { position: [x - o, y + 1.0 + o, z - o] },
        LineVertex { position: [x + 1.0 + o, y - o, z - o] },
        LineVertex { position: [x + 1.0 + o, y + 1.0 + o, z - o] },
        LineVertex { position: [x + 1.0 + o, y - o, z + 1.0 + o] },
        LineVertex { position: [x + 1.0 + o, y + 1.0 + o, z + 1.0 + o] },
        LineVertex { position: [x - o, y - o, z + 1.0 + o] },
        LineVertex { position: [x - o, y + 1.0 + o, z + 1.0 + o] },
    ]
}
