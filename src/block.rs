use cgmath::Vector3;

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
}

impl BlockType {
    pub fn is_solid(&self) -> bool {
        !matches!(self, BlockType::Air | BlockType::Water)
    }

    pub fn is_transparent(&self) -> bool {
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
            BlockType::GlowStone => "Glow",
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
            _ => BlockType::Air,
        }
    }

    /// Returns the light level emitted by this block (0-15)
    pub fn get_light_emission(&self) -> u8 {
        match self {
            BlockType::GlowStone => 15,
            _ => 0,
        }
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
}

impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 9]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 10]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

/// Creates vertices for a single face of a cube
pub fn create_face_vertices(pos: Vector3<f32>, block_type: BlockType, face_index: usize, light_level: f32) -> [Vertex; 4] {
    create_face_vertices_with_alpha(pos, block_type, face_index, light_level, 1.0)
}

/// Creates vertices for a single face of a cube with custom alpha
pub fn create_face_vertices_with_alpha(pos: Vector3<f32>, block_type: BlockType, face_index: usize, light_level: f32, alpha: f32) -> [Vertex; 4] {
    let color = block_type.get_color();
    let x = pos.x;
    let y = pos.y;
    let z = pos.z;

    match face_index {
        0 => [ // Front face (+Z)
            Vertex { position: [x, y, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha },
            Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha },
            Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha },
        ],
        1 => [ // Back face (-Z)
            Vertex { position: [x + 1.0, y, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha },
            Vertex { position: [x, y, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha },
            Vertex { position: [x, y + 1.0, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha },
            Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha },
        ],
        2 => [ // Top face (+Y)
            Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [0.0, 1.0, 0.0], light_level, alpha },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [0.0, 1.0, 0.0], light_level, alpha },
            Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [0.0, 1.0, 0.0], light_level, alpha },
            Vertex { position: [x, y + 1.0, z], color, normal: [0.0, 1.0, 0.0], light_level, alpha },
        ],
        3 => [ // Bottom face (-Y)
            Vertex { position: [x, y, z], color, normal: [0.0, -1.0, 0.0], light_level, alpha },
            Vertex { position: [x + 1.0, y, z], color, normal: [0.0, -1.0, 0.0], light_level, alpha },
            Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [0.0, -1.0, 0.0], light_level, alpha },
            Vertex { position: [x, y, z + 1.0], color, normal: [0.0, -1.0, 0.0], light_level, alpha },
        ],
        4 => [ // Right face (+X)
            Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [1.0, 0.0, 0.0], light_level, alpha },
            Vertex { position: [x + 1.0, y, z], color, normal: [1.0, 0.0, 0.0], light_level, alpha },
            Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [1.0, 0.0, 0.0], light_level, alpha },
            Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [1.0, 0.0, 0.0], light_level, alpha },
        ],
        _ => [ // Left face (-X)
            Vertex { position: [x, y, z], color, normal: [-1.0, 0.0, 0.0], light_level, alpha },
            Vertex { position: [x, y, z + 1.0], color, normal: [-1.0, 0.0, 0.0], light_level, alpha },
            Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [-1.0, 0.0, 0.0], light_level, alpha },
            Vertex { position: [x, y + 1.0, z], color, normal: [-1.0, 0.0, 0.0], light_level, alpha },
        ],
    }
}

pub fn create_cube_vertices(pos: Vector3<f32>, block_type: BlockType, light_level: f32) -> Vec<Vertex> {
    let color = block_type.get_color();
    let x = pos.x;
    let y = pos.y;
    let z = pos.z;
    let alpha = 1.0;

    vec![
        // Front face
        Vertex { position: [x, y, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha },
        Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha },
        Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha },
        Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [0.0, 0.0, 1.0], light_level, alpha },
        // Back face
        Vertex { position: [x + 1.0, y, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha },
        Vertex { position: [x, y, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha },
        Vertex { position: [x, y + 1.0, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha },
        Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [0.0, 0.0, -1.0], light_level, alpha },
        // Top face
        Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [0.0, 1.0, 0.0], light_level, alpha },
        Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [0.0, 1.0, 0.0], light_level, alpha },
        Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [0.0, 1.0, 0.0], light_level, alpha },
        Vertex { position: [x, y + 1.0, z], color, normal: [0.0, 1.0, 0.0], light_level, alpha },
        // Bottom face
        Vertex { position: [x, y, z], color, normal: [0.0, -1.0, 0.0], light_level, alpha },
        Vertex { position: [x + 1.0, y, z], color, normal: [0.0, -1.0, 0.0], light_level, alpha },
        Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [0.0, -1.0, 0.0], light_level, alpha },
        Vertex { position: [x, y, z + 1.0], color, normal: [0.0, -1.0, 0.0], light_level, alpha },
        // Right face
        Vertex { position: [x + 1.0, y, z + 1.0], color, normal: [1.0, 0.0, 0.0], light_level, alpha },
        Vertex { position: [x + 1.0, y, z], color, normal: [1.0, 0.0, 0.0], light_level, alpha },
        Vertex { position: [x + 1.0, y + 1.0, z], color, normal: [1.0, 0.0, 0.0], light_level, alpha },
        Vertex { position: [x + 1.0, y + 1.0, z + 1.0], color, normal: [1.0, 0.0, 0.0], light_level, alpha },
        // Left face
        Vertex { position: [x, y, z], color, normal: [-1.0, 0.0, 0.0], light_level, alpha },
        Vertex { position: [x, y, z + 1.0], color, normal: [-1.0, 0.0, 0.0], light_level, alpha },
        Vertex { position: [x, y + 1.0, z + 1.0], color, normal: [-1.0, 0.0, 0.0], light_level, alpha },
        Vertex { position: [x, y + 1.0, z], color, normal: [-1.0, 0.0, 0.0], light_level, alpha },
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
