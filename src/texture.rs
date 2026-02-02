use image::{GenericImage, RgbaImage};

pub const ATLAS_SIZE: u32 = 256;
pub const TILE_SIZE: u32 = 16;
pub const TILES_PER_ROW: u32 = ATLAS_SIZE / TILE_SIZE; // 16

// Texture indices in the atlas (row 0: block textures)
pub const TEX_DIRT: u32 = 0;
pub const TEX_GRASS_TOP: u32 = 1;
pub const TEX_GRASS_SIDE: u32 = 2;
pub const TEX_SAND: u32 = 3;
pub const TEX_ICE: u32 = 4;
pub const TEX_NONE: u32 = 255; // Sentinel for "use color fallback"

// Breaking textures start at row 1 (index 16)
pub const TEX_DESTROY_BASE: u32 = 16;

/// Texture information for each face of a block
#[derive(Clone, Copy)]
pub struct FaceTextures {
    pub top: u32,
    pub bottom: u32,
    pub sides: u32,
}

impl FaceTextures {
    pub fn all(tex: u32) -> Self {
        Self {
            top: tex,
            bottom: tex,
            sides: tex,
        }
    }

    pub fn get_for_face(&self, face_index: usize) -> u32 {
        match face_index {
            2 => self.top,     // +Y (top)
            3 => self.bottom,  // -Y (bottom)
            _ => self.sides,   // All side faces (0, 1, 4, 5)
        }
    }
}

/// Calculate UV coordinates for a tile in the atlas
/// Returns (u_min, v_min, u_max, v_max)
pub fn get_uv_for_tile(tile_index: u32) -> (f32, f32, f32, f32) {
    let row = tile_index / TILES_PER_ROW;
    let col = tile_index % TILES_PER_ROW;
    let u_min = (col as f32 * TILE_SIZE as f32) / ATLAS_SIZE as f32;
    let v_min = (row as f32 * TILE_SIZE as f32) / ATLAS_SIZE as f32;
    let u_max = u_min + (TILE_SIZE as f32 / ATLAS_SIZE as f32);
    let v_max = v_min + (TILE_SIZE as f32 / ATLAS_SIZE as f32);
    (u_min, v_min, u_max, v_max)
}

/// Get the UV coordinates array for a face's 4 vertices
/// Vertices are ordered: bottom-left, bottom-right, top-right, top-left
pub fn get_face_uvs(tile_index: u32) -> [[f32; 2]; 4] {
    let (u_min, v_min, u_max, v_max) = get_uv_for_tile(tile_index);
    [
        [u_min, v_max],  // bottom-left
        [u_max, v_max],  // bottom-right
        [u_max, v_min],  // top-right
        [u_min, v_min],  // top-left
    ]
}

pub struct TextureAtlas {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

impl TextureAtlas {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        // Create blank RGBA atlas
        let mut atlas = RgbaImage::new(ATLAS_SIZE, ATLAS_SIZE);

        // Fill with a default color (magenta for debugging missing textures)
        for pixel in atlas.pixels_mut() {
            *pixel = image::Rgba([255, 0, 255, 255]);
        }

        // Load block textures (row 0)
        Self::load_texture_into_atlas(&mut atlas, "assets/textures/blocks/dirt.png", TEX_DIRT);
        Self::load_texture_into_atlas(&mut atlas, "assets/textures/blocks/grass_top.png", TEX_GRASS_TOP);
        Self::load_texture_into_atlas(&mut atlas, "assets/textures/blocks/grass_side.png", TEX_GRASS_SIDE);
        Self::load_texture_into_atlas(&mut atlas, "assets/textures/blocks/sand.png", TEX_SAND);
        Self::load_texture_into_atlas(&mut atlas, "assets/textures/blocks/ice.png", TEX_ICE);

        // Load breaking textures (row 1, starting at index 16)
        for i in 0..10 {
            let path = format!("assets/textures/effects/destroy_stage_{}.png", i);
            Self::load_texture_into_atlas(&mut atlas, &path, TEX_DESTROY_BASE + i);
        }

        // Create WGPU texture
        let texture_size = wgpu::Extent3d {
            width: ATLAS_SIZE,
            height: ATLAS_SIZE,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture Atlas"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Write atlas data to texture
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * ATLAS_SIZE),
                rows_per_image: Some(ATLAS_SIZE),
            },
            texture_size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Sampler with nearest filtering for pixel art
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Texture Atlas Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Bind group layout for texture sampling
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Texture Atlas Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture Atlas Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        Self {
            texture,
            view,
            sampler,
            bind_group_layout,
            bind_group,
        }
    }

    fn load_texture_into_atlas(atlas: &mut RgbaImage, path: &str, tile_index: u32) {
        let row = tile_index / TILES_PER_ROW;
        let col = tile_index % TILES_PER_ROW;
        let x = col * TILE_SIZE;
        let y = row * TILE_SIZE;

        match image::open(path) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                // Only copy if the image fits in a tile
                if rgba.width() <= TILE_SIZE && rgba.height() <= TILE_SIZE {
                    if let Err(e) = atlas.copy_from(&rgba, x, y) {
                        eprintln!("Failed to copy texture {} to atlas: {}", path, e);
                    }
                } else {
                    // Resize if needed
                    let resized = image::imageops::resize(&rgba, TILE_SIZE, TILE_SIZE, image::imageops::FilterType::Nearest);
                    if let Err(e) = atlas.copy_from(&resized, x, y) {
                        eprintln!("Failed to copy resized texture {} to atlas: {}", path, e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to load texture {}: {}", path, e);
                // Leave the magenta placeholder
            }
        }
    }
}
