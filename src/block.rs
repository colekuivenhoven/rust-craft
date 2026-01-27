use cgmath::Vector3;
use crate::texture::{FaceTextures, TEX_DIRT, TEX_GRASS_TOP, TEX_GRASS_SIDE, TEX_NONE};

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
    /// Virtual block type for unloaded chunk boundaries.
    /// Transparent for regular blocks (so faces render) but opaque for water (prevents artifacts).
    Boundary,
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
            BlockType::Boundary => 11,
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
            11 => BlockType::Boundary,
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
            ],
        }
    }
}

/// Creates vertices for a single face of a cube
pub fn create_face_vertices(pos: Vector3<f32>, block_type: BlockType, face_index: usize, light_level: f32, tex_index: u32, uvs: [[f32; 2]; 4]) -> [Vertex; 4] {
    create_face_vertices_with_alpha(pos, block_type, face_index, light_level, 1.0, tex_index, uvs)
}

/// Creates vertices for a single face of a cube with custom alpha
pub fn create_face_vertices_with_alpha(pos: Vector3<f32>, block_type: BlockType, face_index: usize, light_level: f32, alpha: f32, tex_index: u32, uvs: [[f32; 2]; 4]) -> [Vertex; 4] {
    let color = block_type.get_color();
    let x = pos.x;
    let y = pos.y;
    let z = pos.z;

    match face_index {
        0 => [ // Front face (+Z)
            Vertex { position: [x, y, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv: uvs[0], tex_index },
            Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv: uvs[1], tex_index },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv: uvs[2], tex_index },
            Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv: uvs[3], tex_index },
        ],
        1 => [ // Back face (-Z)
            Vertex { position: [x + 1.0, y, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv: uvs[0], tex_index },
            Vertex { position: [x, y, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv: uvs[1], tex_index },
            Vertex { position: [x, y + 1.0, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv: uvs[2], tex_index },
            Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv: uvs[3], tex_index },
        ],
        2 => [ // Top face (+Y)
            Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [0.0, 1.0, 0.0], light_level, alpha, uv: uvs[0], tex_index },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [0.0, 1.0, 0.0], light_level, alpha, uv: uvs[1], tex_index },
            Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [0.0, 1.0, 0.0], light_level, alpha, uv: uvs[2], tex_index },
            Vertex { position: [x, y + 1.0, z], color, normal: [0.0, 1.0, 0.0], light_level, alpha, uv: uvs[3], tex_index },
        ],
        3 => [ // Bottom face (-Y)
            Vertex { position: [x, y, z], color, normal: [0.0, -1.0, 0.0], light_level, alpha, uv: uvs[0], tex_index },
            Vertex { position: [x + 1.0, y, z], color, normal: [0.0, -1.0, 0.0], light_level, alpha, uv: uvs[1], tex_index },
            Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [0.0, -1.0, 0.0], light_level, alpha, uv: uvs[2], tex_index },
            Vertex { position: [x, y, z + 1.0], color, normal: [0.0, -1.0, 0.0], light_level, alpha, uv: uvs[3], tex_index },
        ],
        4 => [ // Right face (+X)
            Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv: uvs[0], tex_index },
            Vertex { position: [x + 1.0, y, z], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv: uvs[1], tex_index },
            Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv: uvs[2], tex_index },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv: uvs[3], tex_index },
        ],
        _ => [ // Left face (-X)
            Vertex { position: [x, y, z], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv: uvs[0], tex_index },
            Vertex { position: [x, y, z + 1.0], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv: uvs[1], tex_index },
            Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv: uvs[2], tex_index },
            Vertex { position: [x, y + 1.0, z], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv: uvs[3], tex_index },
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

    vec![
        // Front face
        Vertex { position: [x, y, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv, tex_index },
        Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv, tex_index },
        Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv, tex_index },
        Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha, uv, tex_index },
        // Back face
        Vertex { position: [x + 1.0, y, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv, tex_index },
        Vertex { position: [x, y, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv, tex_index },
        Vertex { position: [x, y + 1.0, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv, tex_index },
        Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha, uv, tex_index },
        // Top face
        Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [0.0, 1.0, 0.0], light_level, alpha, uv, tex_index },
        Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [0.0, 1.0, 0.0], light_level, alpha, uv, tex_index },
        Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [0.0, 1.0, 0.0], light_level, alpha, uv, tex_index },
        Vertex { position: [x, y + 1.0, z], color, normal: [0.0, 1.0, 0.0], light_level, alpha, uv, tex_index },
        // Bottom face
        Vertex { position: [x, y, z], color, normal: [0.0, -1.0, 0.0], light_level, alpha, uv, tex_index },
        Vertex { position: [x + 1.0, y, z], color, normal: [0.0, -1.0, 0.0], light_level, alpha, uv, tex_index },
        Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [0.0, -1.0, 0.0], light_level, alpha, uv, tex_index },
        Vertex { position: [x, y, z + 1.0], color, normal: [0.0, -1.0, 0.0], light_level, alpha, uv, tex_index },
        // Right face
        Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv, tex_index },
        Vertex { position: [x + 1.0, y, z], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv, tex_index },
        Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv, tex_index },
        Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [1.0, 0.0, 0.0], light_level, alpha, uv, tex_index },
        // Left face
        Vertex { position: [x, y, z], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv, tex_index },
        Vertex { position: [x, y, z + 1.0], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv, tex_index },
        Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv, tex_index },
        Vertex { position: [x, y + 1.0, z], color, normal: [-1.0, 0.0, 0.0], light_level, alpha, uv, tex_index },
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
