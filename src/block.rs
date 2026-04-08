use crate::texture::{
    FaceTextures, 
    TEX_DIRT, 
    TEX_SAND, 
    TEX_ICE, 
    TEX_STONE, 
    TEX_WOOD_TOP, 
    TEX_WOOD_SIDE, 
    TEX_LEAVES, 
    TEX_GRAINS, 
    TEX_GRAINS_TALL, 
    TEX_CRAFTING_TABLE, 
    TEX_COBBLESTONE, 
    TEX_VINES, 
    TEX_PLANKS,
    TEX_BEDROCK,
    TEX_FROZEN_STONE,
    TEX_NONE
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockType {
    Air,
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
    GrassTuft,
    GrassTuftTall,
    CraftingTable,
    Vines,
    Bedrock,
    FrozenStone,
    Boundary, // Virtual block type for unloaded chunk boundaries.
}

impl BlockType {
    pub fn name(&self) -> &'static str {
        match self {
            BlockType::Air => "Air",
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
            BlockType::GrassTuft => "Grass",
            BlockType::GrassTuftTall => "Tall Grass",
            BlockType::CraftingTable => "Crafting Table",
            BlockType::Vines => "Vines",
            BlockType::Bedrock => "Bedrock",
            BlockType::FrozenStone => "Frozen Stone",
            BlockType::Boundary => "Boundary",
        }
    }

    pub fn is_solid(&self) -> bool {
        !matches!(self, BlockType::Air | BlockType::Water | BlockType::GrassTuft | BlockType::GrassTuftTall | BlockType::Vines | BlockType::Boundary)
    }

    /// True for any block that the player can point at, select, and break.
    /// Broader than is_solid() — includes non-solid interactable blocks like
    /// grass tufts and vines.
    pub fn is_targetable(&self) -> bool {
        !matches!(self, BlockType::Air | BlockType::Water | BlockType::Boundary)
    }

    pub fn is_water(&self) -> bool {
        matches!(self, BlockType::Water)
    }

    pub fn is_transparent(&self) -> bool {
        matches!(self, BlockType::Air | BlockType::Water | BlockType::Leaves | BlockType::GrassTuft | BlockType::GrassTuftTall | BlockType::Vines | BlockType::Boundary)
    }

    /// Returns true if this block is semi-transparent (rendered with alpha blending after opaque blocks).
    /// These blocks are NOT transparent for face culling - all neighboring faces are always rendered.
    pub fn is_semi_transparent(&self) -> bool {
        matches!(self, BlockType::Ice)
    }

    pub fn no_shadow_casting(&self) -> bool {
        matches!(self, BlockType::Air | BlockType::GrassTuft | BlockType::GrassTuftTall | BlockType::Vines)
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
        matches!(self, BlockType::Air | BlockType::Water | BlockType::Leaves | BlockType::GrassTuft | BlockType::GrassTuftTall | BlockType::Vines)
    }

    pub fn get_color(&self) -> [f32; 3] {
        match self {
            BlockType::Air => [0.0, 0.0, 0.0],
            BlockType::Dirt => [1.0, 1.0, 1.0], // Uses texture
            BlockType::Stone => [1.0, 1.0, 1.0], // Uses texture
            BlockType::Wood => [1.0, 1.0, 1.0], // Uses texture
            BlockType::Leaves => [1.0, 1.0, 1.0], // Uses texture
            BlockType::Sand => [1.0, 1.0, 1.0], // Uses texture
            BlockType::Water => [0.2, 0.4, 0.8], // Uses texture/shader
            BlockType::Cobblestone => [0.4, 0.4, 0.4],
            BlockType::Planks => [0.7, 0.5, 0.3],
            BlockType::GlowStone => [1.0, 0.9, 0.5],
            BlockType::Ice => [0.6, 0.8, 0.95], // Light blue tint
            BlockType::Snow => [0.95, 0.95, 0.98], // Nearly white
            BlockType::GrassTuft => [1.0, 1.0, 1.0], // Uses texture
            BlockType::GrassTuftTall => [1.0, 1.0, 1.0], // Uses texture
            BlockType::CraftingTable => [1.0, 1.0, 1.0], // Uses texture
            BlockType::Vines => [1.0, 1.0, 1.0], // Uses texture
            BlockType::Bedrock => [1.0, 1.0, 1.0], // Uses texture
            BlockType::FrozenStone => [1.0, 1.0, 1.0], // Uses texture
            BlockType::Boundary => [0.0, 0.0, 0.0], // Never rendered
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            BlockType::Air => "Air",
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
            BlockType::GrassTuft => "Grass Tuft",
            BlockType::GrassTuftTall => "Tall Grass Tuft",
            BlockType::CraftingTable => "Crafting Table",
            BlockType::Vines => "Vines",
            BlockType::Bedrock => "Bedrock",
            BlockType::FrozenStone => "Frozen Stone",
            BlockType::Boundary => "Boundary",
        }
    }

    pub fn get_id(&self) -> u8 {
        match self {
            BlockType::Air => 0,
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
            BlockType::GrassTuft => 14,
            BlockType::GrassTuftTall => 16,
            BlockType::CraftingTable => 17,
            BlockType::Vines => 18,
            BlockType::Bedrock => 19,
            BlockType::FrozenStone => 20,
            BlockType::Boundary => 15,
        }
    }

    pub fn from_id(id: u8) -> Self {
        match id {
            0 => BlockType::Air,
            1 => BlockType::Dirt, // Legacy grass ID maps to dirt
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
            14 => BlockType::GrassTuft,
            16 => BlockType::GrassTuftTall,
            15 => BlockType::Boundary,
            17 => BlockType::CraftingTable,
            18 => BlockType::Vines,
            19 => BlockType::Bedrock,
            20 => BlockType::FrozenStone,
            _ => BlockType::Air,
        }
    }

    /// Returns the light level emitted by this block (0-15)
    pub fn get_light_emission(&self) -> u8 {
        match self {
            BlockType::GlowStone => 15,
            BlockType::GrassTuft => 0,
            BlockType::GrassTuftTall => 0,
            BlockType::Boundary => 0, // Virtual block, no light
            _ => 0,
        }
    }

    /// Returns texture indices for block faces.
    /// `has_block_above`: true if there's a solid block directly above this one
    pub fn get_face_textures(&self, _has_block_above: bool) -> FaceTextures {
        match self {
            // Dirt (grass overlay handled in mesh builder for exposed tops)
            BlockType::Dirt => FaceTextures::all(TEX_DIRT),

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

            // Leaves
            BlockType::Leaves => FaceTextures::all(TEX_LEAVES),

            // Grass tuft (cross model uses grains texture)
            BlockType::GrassTuft => FaceTextures::all(TEX_GRAINS),

            // Tall grass tuft (cross model uses grains_tall texture)
            BlockType::GrassTuftTall => FaceTextures::all(TEX_GRAINS_TALL),

            // Crafting table
            BlockType::CraftingTable => FaceTextures::all(TEX_CRAFTING_TABLE),

            // Cobblestone
            BlockType::Cobblestone => FaceTextures::all(TEX_COBBLESTONE),

            // Vines
            BlockType::Vines => FaceTextures::all(TEX_VINES),

            // Planks
            BlockType::Planks => FaceTextures::all(TEX_PLANKS),

            // Bedrock
            BlockType::Bedrock => FaceTextures::all(TEX_BEDROCK),

            // Frozen Stone
            BlockType::FrozenStone => FaceTextures::all(TEX_FROZEN_STONE),

            // All other blocks use color fallback
            _ => FaceTextures::all(TEX_NONE),
        }
    }

    /// Returns the time in seconds required to break this block
    pub fn get_durability(&self) -> f32 {
        match self {
            BlockType::Air => 0.0,
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
            BlockType::GrassTuft => 0.1,
            BlockType::GrassTuftTall => 0.1,
            BlockType::CraftingTable => 2.5,
            BlockType::Vines => 0.1,
            BlockType::Bedrock => 0.0,
            BlockType::FrozenStone => 2.5,
            BlockType::Boundary => 0.0,
        }
    }

    /// Returns true if this block can be broken
    /// Returns true if this block uses a cross model (two intersecting quads) instead of a cube.
    pub fn is_cross_model(&self) -> bool {
        matches!(self, BlockType::GrassTuft | BlockType::GrassTuftTall)
    }

    /// True for blocks whose dropped item should render as a flat panel instead of a mini-cube.
    pub fn is_flat_item(&self) -> bool {
        matches!(self, BlockType::GrassTuft | BlockType::GrassTuftTall | BlockType::Vines)
    }

    pub fn is_breakable(&self) -> bool {
        !matches!(self, BlockType::Air | BlockType::Water | BlockType::Bedrock | BlockType::Boundary)
    }
}

// Re-exports from block_mesh for backward compatibility
pub use crate::block_mesh::{
    Vertex, UiVertex, ItemCubeVertex, ModalVertex, LineVertex,
    AO_OFFSETS, CUBE_INDICES,
    WAVE_AMPLITUDE, WAVE_FREQUENCY, WAVE_SPEED, WAVE_OCTAVES, WAVE_LACUNARITY, WAVE_PERSISTENCE,
    calculate_ao,
    create_face_vertices, create_face_vertices_tinted, create_face_vertices_with_alpha,
    create_water_face_vertices, create_cross_model_vertices, create_vine_face_vertices,
    create_cube_vertices, create_scaled_cube_vertices, create_flat_item_vertices,
    create_particle_vertices, create_block_outline,
};
