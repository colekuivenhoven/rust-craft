use cgmath::Vector3;
use crate::texture::{FaceTextures, TEX_DIRT, TEX_GRASS_TOP, TEX_GRASS_SIDE, TEX_SAND, TEX_ICE, TEX_STONE, TEX_WOOD_TOP, TEX_WOOD_SIDE, TEX_NONE};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockType {
    Air,
    Grass,
    Dirt,
    Stone,
    Wood,
    Leaves,
    Sand,
    Water,
    Cobblestone,
    Planks,
    GlowStone,
    Ice,
    Snow,
    Boundary, // Virtual block type for unloaded chunk boundaries.
}

impl BlockType {
    pub fn is_solid(&self) -> bool {
        !matches!(self, BlockType::Air | BlockType::Water | BlockType::Boundary)
    }

    pub fn is_water(&self) -> bool {
        matches!(self, BlockType::Water)
    }

    pub fn is_transparent(&self) -> bool {
        matches!(self, BlockType::Air | BlockType::Water | BlockType::Leaves | BlockType::Boundary)
    }

    /// Returns true if this block is semi-transparent (rendered with alpha blending after opaque blocks).
    /// These blocks are NOT transparent for face culling - all neighboring faces are always rendered.
    pub fn is_semi_transparent(&self) -> bool {
        matches!(self, BlockType::Ice)
    }

    /// Returns the alpha value for this block (1.0 = fully opaque, 0.0 = fully transparent)
    pub fn get_alpha(&self) -> f32 {
        match self {
            BlockType::Ice => 1.0,
            _ => 1.0,
        }
    }

    /// Returns true if this block is transparent specifically for water face culling.
    /// Boundary blocks are NOT transparent for water to prevent rendering artifacts.
    pub fn is_transparent_for_water(&self) -> bool {
        matches!(self, BlockType::Air | BlockType::Water | BlockType::Leaves)
    }

    pub fn get_color(&self) -> [f32; 3] {
        match self {
            BlockType::Air => [0.0, 0.0, 0.0],
            BlockType::Grass => [0.2, 0.8, 0.2],
            BlockType::Dirt => [0.6, 0.4, 0.2],
            BlockType::Stone => [0.5, 0.5, 0.5],
            BlockType::Wood => [0.4, 0.25, 0.1],
            BlockType::Leaves => [0.1, 0.6, 0.1],
            BlockType::Sand => [0.9, 0.9, 0.6],
            BlockType::Water => [0.2, 0.4, 0.8],
            BlockType::Cobblestone => [0.4, 0.4, 0.4],
            BlockType::Planks => [0.7, 0.5, 0.3],
            BlockType::GlowStone => [1.0, 0.9, 0.5],
            BlockType::Ice => [0.6, 0.8, 0.95],  // Light blue tint
            BlockType::Snow => [0.95, 0.95, 0.98],  // Nearly white
            BlockType::Boundary => [0.0, 0.0, 0.0], // Never rendered
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            BlockType::Air => "Air",
            BlockType::Grass => "Grass",
            BlockType::Dirt => "Dirt",
            BlockType::Stone => "Stone",
            BlockType::Wood => "Wood",
            BlockType::Leaves => "Leaves",
            BlockType::Sand => "Sand",
            BlockType::Water => "Water",
            BlockType::Cobblestone => "Cobblestone",
            BlockType::Planks => "Planks",
            BlockType::GlowStone => "Glowstone",
            BlockType::Ice => "Ice",
            BlockType::Snow => "Snow",
            BlockType::Boundary => "Boundary",
        }
    }

    pub fn get_id(&self) -> u8 {
        match self {
            BlockType::Air => 0,
            BlockType::Grass => 1,
            BlockType::Dirt => 2,
            BlockType::Stone => 3,
            BlockType::Wood => 4,
            BlockType::Leaves => 5,
            BlockType::Sand => 6,
            BlockType::Water => 7,
            BlockType::Cobblestone => 8,
            BlockType::Planks => 9,
            BlockType::GlowStone => 10,
            BlockType::Ice => 11,
            BlockType::Snow => 12,
            BlockType::Boundary => 13,
        }
    }

    pub fn from_id(id: u8) -> Self {
        match id {
            0 => BlockType::Air,
            1 => BlockType::Grass,
            2 => BlockType::Dirt,
            3 => BlockType::Stone,
            4 => BlockType::Wood,
            5 => BlockType::Leaves,
            6 => BlockType::Sand,
            7 => BlockType::Water,
            8 => BlockType::Cobblestone,
            9 => BlockType::Planks,
            10 => BlockType::GlowStone,
            11 => BlockType::Ice,
            12 => BlockType::Snow,
            13 => BlockType::Boundary,
            _ => BlockType::Air,
        }
    }

    /// Returns the light level emitted by this block (0-15)
    pub fn get_light_emission(&self) -> u8 {
        match self {
            BlockType::GlowStone => 15,
            BlockType::Boundary => 0, // Virtual block, no light
            _ => 0,
        }
    }

    /// Returns texture indices for block faces.
    /// `has_block_above`: true if there's a solid block directly above this one
    pub fn get_face_textures(&self, has_block_above: bool) -> FaceTextures {
        match self {
            // Dirt
            BlockType::Dirt | BlockType::Grass => {
                if has_block_above {
                    // Covered dirt: all faces use dirt texture
                    FaceTextures::all(TEX_DIRT)
                } else {
                    // Exposed dirt/grass: grass top, grass sides, dirt bottom
                    FaceTextures {
                        top: TEX_GRASS_TOP,
                        bottom: TEX_DIRT,
                        sides: TEX_GRASS_SIDE,
                    }
                }
            }

            // Wood
            BlockType::Wood => FaceTextures {
                top: TEX_WOOD_TOP,
                bottom: TEX_WOOD_TOP,
                sides: TEX_WOOD_SIDE,
            },

            // Sand
            BlockType::Sand => FaceTextures::all(TEX_SAND),

            // Ice
            BlockType::Ice => FaceTextures::all(TEX_ICE),

            // Stone
            BlockType::Stone => FaceTextures::all(TEX_STONE),

            // All other blocks use color fallback
            _ => FaceTextures::all(TEX_NONE),
        }
    }

    /// Returns the time in seconds required to break this block
    pub fn get_durability(&self) -> f32 {
        match self {
            BlockType::Air => 0.0,
            BlockType::Grass => 0.6,
            BlockType::Dirt => 0.5,
            BlockType::Stone => 1.5,
            BlockType::Wood => 2.0,
            BlockType::Leaves => 0.2,
            BlockType::Sand => 0.5,
            BlockType::Water => 0.0,  // Cannot break water
            BlockType::Cobblestone => 2.0,
            BlockType::Planks => 1.5,
            BlockType::GlowStone => 0.8,
            BlockType::Ice => 0.5,
            BlockType::Snow => 0.2,
            BlockType::Boundary => 0.0,
        }
    }

    /// Returns true if this block can be broken
    pub fn is_breakable(&self) -> bool {
        !matches!(self, BlockType::Air | BlockType::Water | BlockType::Boundary)
    }
}

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
pub const WAVE_AMPLITUDE: f32 = 0.9;

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

/// Creates vertices for a single face of a cube with per-vertex AO
pub fn create_face_vertices(pos: Vector3<f32>, block_type: BlockType, face_index: usize, light_level: f32, tex_index: u32, uvs: [[f32; 2]; 4], ao_values: [f32; 4]) -> [Vertex; 4] {
    create_face_vertices_with_alpha(pos, block_type, face_index, light_level, 1.0, tex_index, uvs, ao_values)
}

/// Creates vertices for a single face of a cube with custom alpha and per-vertex AO
pub fn create_face_vertices_with_alpha(pos: Vector3<f32>, block_type: BlockType, face_index: usize, light_level: f32, alpha: f32, tex_index: u32, uvs: [[f32; 2]; 4], ao_values: [f32; 4]) -> [Vertex; 4] {
    let color = block_type.get_color();
    let x = pos.x;
    let y = pos.y;
    let z = pos.z;

    match face_index {
        0 => [ // Front face (+Z)
            Vertex { position: [x, y, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv: uvs[0], tex_index, ao: ao_values[0] },
            Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv: uvs[1], tex_index, ao: ao_values[1] },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv: uvs[2], tex_index, ao: ao_values[2] },
            Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv: uvs[3], tex_index, ao: ao_values[3] },
        ],
        1 => [ // Back face (-Z)
            Vertex { position: [x + 1.0, y, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv: uvs[0], tex_index, ao: ao_values[0] },
            Vertex { position: [x, y, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv: uvs[1], tex_index, ao: ao_values[1] },
            Vertex { position: [x, y + 1.0, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv: uvs[2], tex_index, ao: ao_values[2] },
            Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv: uvs[3], tex_index, ao: ao_values[3] },
        ],
        2 => [ // Top face (+Y)
            Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [0.0, 1.0, 0.0], light_level, alpha, uv: uvs[0], tex_index, ao: ao_values[0] },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [0.0, 1.0, 0.0], light_level, alpha, uv: uvs[1], tex_index, ao: ao_values[1] },
            Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [0.0, 1.0, 0.0], light_level, alpha, uv: uvs[2], tex_index, ao: ao_values[2] },
            Vertex { position: [x, y + 1.0, z], color, normal: [0.0, 1.0, 0.0], light_level, alpha, uv: uvs[3], tex_index, ao: ao_values[3] },
        ],
        3 => [ // Bottom face (-Y)
            Vertex { position: [x, y, z], color, normal: [0.0, -1.0, 0.0], light_level, alpha, uv: uvs[0], tex_index, ao: ao_values[0] },
            Vertex { position: [x + 1.0, y, z], color, normal: [0.0, -1.0, 0.0], light_level, alpha, uv: uvs[1], tex_index, ao: ao_values[1] },
            Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [0.0, -1.0, 0.0], light_level, alpha, uv: uvs[2], tex_index, ao: ao_values[2] },
            Vertex { position: [x, y, z + 1.0], color, normal: [0.0, -1.0, 0.0], light_level, alpha, uv: uvs[3], tex_index, ao: ao_values[3] },
        ],
        4 => [ // Right face (+X)
            Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv: uvs[0], tex_index, ao: ao_values[0] },
            Vertex { position: [x + 1.0, y, z], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv: uvs[1], tex_index, ao: ao_values[1] },
            Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv: uvs[2], tex_index, ao: ao_values[2] },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv: uvs[3], tex_index, ao: ao_values[3] },
        ],
        _ => [ // Left face (-X)
            Vertex { position: [x, y, z], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv: uvs[0], tex_index, ao: ao_values[0] },
            Vertex { position: [x, y, z + 1.0], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv: uvs[1], tex_index, ao: ao_values[1] },
            Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv: uvs[2], tex_index, ao: ao_values[2] },
            Vertex { position: [x, y + 1.0, z], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv: uvs[3], tex_index, ao: ao_values[3] },
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
/// is_surface_water: true if there's no water block above this one
pub fn create_water_face_vertices(
    pos: Vector3<f32>,
    face_index: usize,
    light_level: f32,
    tex_index: u32,
    uvs: [[f32; 2]; 4],
    edge_factors: [f32; 4],
    is_surface_water: bool,
) -> [Vertex; 4] {
    let color = BlockType::Water.get_color();
    let x = pos.x;
    let y = pos.y;
    let z = pos.z;

    // Determine wave factor for each vertex based on face type and surface status
    // For all faces, vertices 2 and 3 are at the top edge (y+1)
    let wave_factors: [f32; 4] = match face_index {
        2 => [1.0, 1.0, 1.0, 1.0], // Top face: always affected (only rendered on surface)
        3 => [0.0, 0.0, 0.0, 0.0], // Bottom face: never affected
        _ if is_surface_water => [0.0, 0.0, 1.0, 1.0], // Side faces: top vertices (2,3) affected
        _ => [0.0, 0.0, 0.0, 0.0], // Non-surface side faces: no wave
    };

    match face_index {
        0 => [ // Front face (+Z)
            Vertex { position: [x, y, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha: wave_factors[0], uv: uvs[0], tex_index, ao: edge_factors[0] },
            Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha: wave_factors[1], uv: uvs[1], tex_index, ao: edge_factors[1] },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha: wave_factors[2], uv: uvs[2], tex_index, ao: edge_factors[2] },
            Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha: wave_factors[3], uv: uvs[3], tex_index, ao: edge_factors[3] },
        ],
        1 => [ // Back face (-Z)
            Vertex { position: [x + 1.0, y, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha: wave_factors[0], uv: uvs[0], tex_index, ao: edge_factors[0] },
            Vertex { position: [x, y, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha: wave_factors[1], uv: uvs[1], tex_index, ao: edge_factors[1] },
            Vertex { position: [x, y + 1.0, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha: wave_factors[2], uv: uvs[2], tex_index, ao: edge_factors[2] },
            Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha: wave_factors[3], uv: uvs[3], tex_index, ao: edge_factors[3] },
        ],
        2 => [ // Top face (+Y)
            Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [0.0, 1.0, 0.0], light_level, alpha: wave_factors[0], uv: uvs[0], tex_index, ao: edge_factors[0] },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [0.0, 1.0, 0.0], light_level, alpha: wave_factors[1], uv: uvs[1], tex_index, ao: edge_factors[1] },
            Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [0.0, 1.0, 0.0], light_level, alpha: wave_factors[2], uv: uvs[2], tex_index, ao: edge_factors[2] },
            Vertex { position: [x, y + 1.0, z], color, normal: [0.0, 1.0, 0.0], light_level, alpha: wave_factors[3], uv: uvs[3], tex_index, ao: edge_factors[3] },
        ],
        3 => [ // Bottom face (-Y)
            Vertex { position: [x, y, z], color, normal: [0.0, -1.0, 0.0], light_level, alpha: wave_factors[0], uv: uvs[0], tex_index, ao: edge_factors[0] },
            Vertex { position: [x + 1.0, y, z], color, normal: [0.0, -1.0, 0.0], light_level, alpha: wave_factors[1], uv: uvs[1], tex_index, ao: edge_factors[1] },
            Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [0.0, -1.0, 0.0], light_level, alpha: wave_factors[2], uv: uvs[2], tex_index, ao: edge_factors[2] },
            Vertex { position: [x, y, z + 1.0], color, normal: [0.0, -1.0, 0.0], light_level, alpha: wave_factors[3], uv: uvs[3], tex_index, ao: edge_factors[3] },
        ],
        4 => [ // Right face (+X)
            Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [1.0, 0.0, 0.0], light_level, alpha: wave_factors[0], uv: uvs[0], tex_index, ao: edge_factors[0] },
            Vertex { position: [x + 1.0, y, z], color, normal: [1.0, 0.0, 0.0], light_level, alpha: wave_factors[1], uv: uvs[1], tex_index, ao: edge_factors[1] },
            Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [1.0, 0.0, 0.0], light_level, alpha: wave_factors[2], uv: uvs[2], tex_index, ao: edge_factors[2] },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [1.0, 0.0, 0.0], light_level, alpha: wave_factors[3], uv: uvs[3], tex_index, ao: edge_factors[3] },
        ],
        _ => [ // Left face (-X)
            Vertex { position: [x, y, z], color, normal: [-1.0, 0.0, 0.0], light_level, alpha: wave_factors[0], uv: uvs[0], tex_index, ao: edge_factors[0] },
            Vertex { position: [x, y, z + 1.0], color, normal: [-1.0, 0.0, 0.0], light_level, alpha: wave_factors[1], uv: uvs[1], tex_index, ao: edge_factors[1] },
            Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [-1.0, 0.0, 0.0], light_level, alpha: wave_factors[2], uv: uvs[2], tex_index, ao: edge_factors[2] },
            Vertex { position: [x, y + 1.0, z], color, normal: [-1.0, 0.0, 0.0], light_level, alpha: wave_factors[3], uv: uvs[3], tex_index, ao: edge_factors[3] },
        ],
    }
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

    let color = block_type.get_color();
    let face_textures = block_type.get_face_textures(false); // Dropped items are always "exposed"
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
