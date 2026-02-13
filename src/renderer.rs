use crate::bird::{BirdManager, create_bird_vertices, generate_bird_indices};
use crate::fish::{FishManager, create_fish_vertices, generate_fish_indices};
use crate::block::{BlockType, Vertex, UiVertex, LineVertex, create_cube_vertices, create_block_outline, create_face_vertices, create_scaled_cube_vertices, create_particle_vertices, create_shadow_vertices, CUBE_INDICES};
use crate::camera::{Camera, CameraController, CameraUniform, Projection, Frustum};
use crate::chunk::{CHUNK_SIZE, CHUNK_HEIGHT};
use crate::crafting::CraftingSystem;
use crate::dropped_item::DroppedItemManager;
use crate::enemy::EnemyManager;
use crate::bitmap_font;
use crate::particle::ParticleManager;
use crate::player::Player;
use crate::texture::{TextureAtlas, get_face_uvs, TEX_DESTROY_BASE};
use crate::water::WaterSimulation;
use crate::world::World;
use cgmath::{Point3, Vector3};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::event::*;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

/// Cached GPU buffers for a chunk - avoids recreating buffers every frame
pub struct ChunkBuffers {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub water_vertex_buffer: Option<wgpu::Buffer>,
    pub water_index_buffer: Option<wgpu::Buffer>,
    pub water_index_count: u32,
    pub transparent_vertex_buffer: Option<wgpu::Buffer>,
    pub transparent_index_buffer: Option<wgpu::Buffer>,
    pub transparent_index_count: u32,
    pub mesh_version: u32,
}

/// Tracks the state of an in-progress block break
pub struct BreakingState {
    /// Block position being broken
    pub block_pos: (i32, i32, i32),
    /// Breaking progress from 0.0 to 1.0
    pub progress: f32,
    /// Block type being broken (for durability lookup)
    pub block_type: BlockType,
}

impl BreakingState {
    pub fn new(pos: (i32, i32, i32), block_type: BlockType) -> Self {
        Self {
            block_pos: pos,
            progress: 0.0,
            block_type,
        }
    }

    /// Returns the destroy stage index (0-9) based on progress
    pub fn get_destroy_stage(&self) -> u32 {
        let stage = (self.progress * 10.0).floor() as u32;
        stage.min(9)
    }
}

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    water_pipeline: wgpu::RenderPipeline,  // Separate pipeline for transparent water with depth sampling
    transparent_pipeline: wgpu::RenderPipeline,  // Separate pipeline for semi-transparent blocks (ice) with no depth write
    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    projection: Projection,
    frustum: Frustum,
    world: World,
    player: Player,
    spawn_point: Point3<f32>,
    water_simulation: WaterSimulation,
    enemy_manager: EnemyManager,
    bird_manager: BirdManager,
    fish_manager: FishManager,
    dropped_item_manager: DroppedItemManager,
    particle_manager: ParticleManager,
    crafting_system: CraftingSystem,
    last_frame: Instant,
    mouse_pressed: bool,
    window: Arc<Window>,
    show_inventory: bool,
    show_crafting: bool,
    // UI rendering
    ui_pipeline: wgpu::RenderPipeline,
    ui_bind_group: wgpu::BindGroup,
    ui_uniform_buffer: wgpu::Buffer,
    crosshair_vertex_buffer: wgpu::Buffer,
    hud_vertex_buffer: wgpu::Buffer,
    hud_vertex_count: u32,
    // Block outline rendering
    outline_pipeline: wgpu::RenderPipeline,
    targeted_block: Option<(i32, i32, i32)>,
    // Chunk outline rendering (debug)
    chunk_outline_pipeline: wgpu::RenderPipeline,
    // Mouse capture state
    mouse_captured: bool,
    // Depth textures: one for rendering, one for sampling in water shader
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    depth_copy_texture: wgpu::Texture,  // Copy of depth buffer for water shader to sample
    depth_copy_view: wgpu::TextureView,
    depth_sampler: wgpu::Sampler,
    water_bind_group_layout: wgpu::BindGroupLayout,
    water_bind_group: wgpu::BindGroup,
    water_time_buffer: wgpu::Buffer,
    // FPS tracking
    fps: f32,
    fps_frame_count: u32,
    fps_timer: f32,
    // Debug mode
    show_chunk_outlines: bool,
    noclip_mode: bool,
    
    // Underwater effect (Post-Processing)
    camera_underwater: bool,
    underwater_pipeline: wgpu::RenderPipeline,
    underwater_bind_group_layout: wgpu::BindGroupLayout,
    underwater_bind_group: wgpu::BindGroup,
    underwater_uniform_buffer: wgpu::Buffer,
    // Off-screen textures for post-processing
    scene_texture: wgpu::Texture,
    scene_texture_view: wgpu::TextureView,
    post_process_texture: wgpu::Texture,
    post_process_texture_view: wgpu::TextureView,
    scene_sampler: wgpu::Sampler,
    start_time: Instant,

    // Motion blur (Post-Processing)
    motion_blur_pipeline: wgpu::RenderPipeline,
    motion_blur_bind_group_layout: wgpu::BindGroupLayout,
    motion_blur_bind_group: wgpu::BindGroup,
    motion_blur_uniform_buffer: wgpu::Buffer,

    // Texture atlas
    texture_atlas: TextureAtlas,
    // Breaking mechanics
    breaking_pipeline: wgpu::RenderPipeline,
    breaking_state: Option<BreakingState>,
    left_mouse_held: bool,
    // Cached GPU buffers for chunks - avoids recreating every frame
    chunk_buffers: HashMap<(i32, i32), ChunkBuffers>,
}

impl State {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let mut camera = Camera::new(Point3::new(0.0, 35.0, 0.0)); // Temporary position, updated after world gen
        let camera_controller = CameraController::new(10.0, 0.003);
        let projection = Projection::new(
            config.width,
            config.height,
            cgmath::Deg(45.0).into(),
            0.1,
            1000.0,
        );

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection);

        // Create initial frustum for culling
        let frustum = Frustum::from_view_proj(&(projection.calc_matrix() * camera.get_view_matrix()));

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shader.wgsl").into()),
        });

        // Create texture atlas
        let texture_atlas = TextureAtlas::new(&device, &queue);

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &texture_atlas.bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Transparent blocks pipeline (same as main but no depth write for proper alpha blending)
        let transparent_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Transparent Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,  // Key difference: don't write depth for transparent blocks
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Depth texture for rendering (write target)
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Depth copy texture for water shader to sample (read only)
        let depth_copy_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Copy Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let depth_copy_view = depth_copy_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Sampler for depth texture
        let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Depth Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: None,
            ..Default::default()
        });

        // Bind group layout for water shader depth sampling and wave animation
        let water_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
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
                // Wave animation uniforms (time + wave parameters)
                // Needs VERTEX for wave displacement and FRAGMENT for foam animation
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Wave animation uniform buffer (time + wave parameters)
        use crate::block::{WAVE_AMPLITUDE, WAVE_FREQUENCY, WAVE_SPEED, WAVE_OCTAVES, WAVE_LACUNARITY, WAVE_PERSISTENCE};
        let wave_uniforms: [f32; 8] = [
            0.0,               // time (updated each frame)
            WAVE_AMPLITUDE,
            WAVE_FREQUENCY,
            WAVE_SPEED,
            WAVE_OCTAVES as f32,
            WAVE_LACUNARITY,
            WAVE_PERSISTENCE,
            0.0,               // padding for alignment
        ];
        let water_time_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Water Wave Uniform Buffer"),
            contents: bytemuck::cast_slice(&wave_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Bind group for water depth sampling and wave animation (uses the copy texture)
        let water_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Water Bind Group"),
            layout: &water_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&depth_copy_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&depth_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: water_time_buffer.as_entire_binding(),
                },
            ],
        });

        // Water shader with depth-based transparency
        let water_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/water_shader.wgsl").into()),
        });

        // Water pipeline layout includes camera bind group and depth texture bind group
        let water_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Water Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout, &water_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Water pipeline with alpha blending for transparency
        let water_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Pipeline"),
            layout: Some(&water_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &water_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &water_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Render both sides of water faces
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false, // Don't write depth for transparent water
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let world = World::new(18); // Sets render_distance

        // Find a safe spawn point: scan from top down at (0, 0) for a solid, non-water
        // block exposed to the sky, then spawn 2 blocks above it
        let spawn_point = {
            let spawn_x = 0;
            let spawn_z = 0;
            let mut spawn_y = 35.0_f32; // fallback
            for y in (1..CHUNK_HEIGHT).rev() {
                let block = world.get_block_world(spawn_x, y as i32, spawn_z);
                let block_above = world.get_block_world(spawn_x, y as i32 + 1, spawn_z);
                if block.is_solid() && block_above == BlockType::Air {
                    spawn_y = y as f32 + 3.0; // 2 blocks above surface + player eye height offset
                    break;
                }
            }
            Point3::new(spawn_x as f32 + 0.5, spawn_y, spawn_z as f32 + 0.5)
        };

        camera.position = spawn_point;
        let player = Player::new(spawn_point);
        let water_simulation = WaterSimulation::new(0.5);
        let enemy_manager = EnemyManager::new(10.0, 10);
        let bird_manager = BirdManager::new();
        let fish_manager = FishManager::new();
        let dropped_item_manager = DroppedItemManager::new();
        let particle_manager = ParticleManager::new();
        let crafting_system = CraftingSystem::new();

        // UI Pipeline for crosshair
        let ui_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("UI Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ui_shader.wgsl").into()),
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct UiUniform {
            aspect_ratio: f32,
            _padding: [f32; 3],
        }

        let aspect_ratio = size.width as f32 / size.height as f32;
        let ui_uniform = UiUniform {
            aspect_ratio,
            _padding: [0.0; 3],
        };

        let ui_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("UI Uniform Buffer"),
            contents: bytemuck::cast_slice(&[ui_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let ui_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("ui_bind_group_layout"),
        });

        let ui_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &ui_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ui_uniform_buffer.as_entire_binding(),
            }],
            label: Some("ui_bind_group"),
        });

        let ui_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("UI Pipeline Layout"),
            bind_group_layouts: &[&ui_bind_group_layout],
            push_constant_ranges: &[],
        });

        let ui_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("UI Pipeline"),
            layout: Some(&ui_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &ui_shader,
                entry_point: Some("vs_main"),
                buffers: &[UiVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &ui_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None, // No depth for UI
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Crosshair vertices (will be updated on resize for aspect ratio correction)
        let crosshair_vertices = Self::build_crosshair_vertices(aspect_ratio);

        let crosshair_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Crosshair Vertex Buffer"),
            contents: bytemuck::cast_slice(&crosshair_vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        // HUD vertex buffer (hotbar + text). We'll rebuild it each frame.
        let hud_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HUD Vertex Buffer"),
            size: (std::mem::size_of::<UiVertex>() as u64) * 100_000,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Outline Pipeline for block highlighting
        let outline_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Outline Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/outline_shader.wgsl").into()),
        });

        let outline_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Outline Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let outline_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Outline Pipeline"),
            layout: Some(&outline_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &outline_shader,
                entry_point: Some("vs_main"),
                buffers: &[LineVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &outline_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: -2, // Push outline toward camera
                    slope_scale: -2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Chunk Outline Pipeline for debug visualization
        let chunk_outline_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Chunk Outline Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/chunk_outline_shader.wgsl").into()),
        });

        let chunk_outline_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Chunk Outline Pipeline"),
            layout: Some(&outline_pipeline_layout),  // Reuse the same layout
            vertex: wgpu::VertexState {
                module: &chunk_outline_shader,
                entry_point: Some("vs_main"),
                buffers: &[LineVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &chunk_outline_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Breaking overlay pipeline
        let breaking_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Breaking Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/breaking_shader.wgsl").into()),
        });

        let breaking_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Breaking Pipeline"),
            layout: Some(&render_pipeline_layout),  // Reuse main pipeline layout (camera + texture)
            vertex: wgpu::VertexState {
                module: &breaking_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &breaking_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: -1,  // Push slightly toward camera
                    slope_scale: -1.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // === POST PROCESSING RESOURCES ===
        
        // 1. Create Scene Texture (the off-screen canvas)
        let scene_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Scene Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let scene_texture_view = scene_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // 2. Create Linear Sampler for waviness
        let scene_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Scene Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // 3. Time Uniform for animation
        let start_time = Instant::now();
        let underwater_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Underwater Uniform Buffer"),
            contents: bytemuck::cast_slice(&[0.0f32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // 4. Underwater Bind Group Layout
        let underwater_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Underwater Bind Group Layout"),
            entries: &[
                // Binding 0: Time
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Scene Texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 2: Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // 5. Underwater Bind Group (created later after motion blur setup, pointing to post_process_texture)

        // Underwater effect pipeline
        let underwater_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Underwater Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/underwater_shader.wgsl").into()),
        });

        let underwater_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Underwater Pipeline Layout"),
            bind_group_layouts: &[&underwater_bind_group_layout], // Now uses the layout
            push_constant_ranges: &[],
        });

        let underwater_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Underwater Pipeline"),
            layout: Some(&underwater_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &underwater_shader,
                entry_point: Some("vs_main"),
                buffers: &[], // No vertex buffers needed - generates full-screen triangle
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &underwater_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None, // No depth testing for post-process overlay
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // === MOTION BLUR POST-PROCESSING ===

        // Second off-screen texture for chaining post-process passes
        let post_process_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Post Process Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let post_process_texture_view = post_process_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let motion_blur_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Motion Blur Uniform Buffer"),
            contents: bytemuck::cast_slice(&[0.0f32, 0.0, 0.0, 0.0]), // blur_dir.xy, strength, padding
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let motion_blur_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Motion Blur Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let motion_blur_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Motion Blur Bind Group"),
            layout: &motion_blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: motion_blur_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&scene_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&scene_sampler),
                },
            ],
        });

        let motion_blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Motion Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/motion_blur_shader.wgsl").into()),
        });

        let motion_blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Motion Blur Pipeline Layout"),
            bind_group_layouts: &[&motion_blur_bind_group_layout],
            push_constant_ranges: &[],
        });

        let motion_blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Motion Blur Pipeline"),
            layout: Some(&motion_blur_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &motion_blur_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &motion_blur_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Update underwater bind group to read from post_process_texture (after motion blur)
        let underwater_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Underwater Bind Group"),
            layout: &underwater_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: underwater_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&post_process_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&scene_sampler),
                },
            ],
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            water_pipeline,
            transparent_pipeline,
            camera,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            projection,
            frustum,
            world,
            player,
            spawn_point,
            water_simulation,
            enemy_manager,
            bird_manager,
            fish_manager,
            dropped_item_manager,
            particle_manager,
            crafting_system,
            last_frame: Instant::now(),
            mouse_pressed: false,
            window,
            show_inventory: false,
            show_crafting: false,
            ui_pipeline,
            ui_bind_group,
            ui_uniform_buffer,
            crosshair_vertex_buffer,
            hud_vertex_buffer,
            hud_vertex_count: 0,
            outline_pipeline,
            targeted_block: None,
            chunk_outline_pipeline,
            mouse_captured: false,
            depth_texture,
            depth_view,
            depth_copy_texture,
            depth_copy_view,
            depth_sampler,
            water_bind_group_layout,
            water_bind_group,
            water_time_buffer,
            // FPS tracking
            fps: 0.0,
            fps_frame_count: 0,
            fps_timer: 0.0,
            // Debug mode
            show_chunk_outlines: false,
            noclip_mode: false,
            // Underwater effect
            camera_underwater: false,
            underwater_pipeline,
            underwater_bind_group_layout,
            underwater_bind_group,
            underwater_uniform_buffer,
            scene_texture,
            scene_texture_view,
            post_process_texture,
            post_process_texture_view,
            scene_sampler,
            start_time,
            // Motion blur
            motion_blur_pipeline,
            motion_blur_bind_group_layout,
            motion_blur_bind_group,
            motion_blur_uniform_buffer,
            // Texture atlas
            texture_atlas,
            // Breaking mechanics
            breaking_pipeline,
            breaking_state: None,
            left_mouse_held: false,
            // Cached GPU buffers
            chunk_buffers: HashMap::new(),
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn respawn(&mut self) {
        self.player.health = self.player.max_health;
        self.camera.position = self.spawn_point;
        self.player.position = self.spawn_point;
        self.camera_controller.velocity = Vector3::new(0.0, 0.0, 0.0);
        self.camera_controller.on_ground = false;
        self.camera_controller.last_fall_velocity = 0.0;
    }

    pub fn capture_mouse(&mut self) {
        let _ = self.window.set_cursor_grab(winit::window::CursorGrabMode::Confined);
        self.window.set_cursor_visible(false);
        self.mouse_captured = true;
    }

    pub fn release_mouse(&mut self) {
        let _ = self.window.set_cursor_grab(winit::window::CursorGrabMode::None);
        self.window.set_cursor_visible(true);
        self.mouse_captured = false;
    }

    pub fn is_mouse_captured(&self) -> bool {
        self.mouse_captured
    }

    /// Saves all modified chunks to disk (call before exiting)
    pub fn save_world(&mut self) {
        self.world.save_all_modified_chunks();
    }

    pub fn process_mouse(&mut self, dx: f64, dy: f64) {
        if self.mouse_captured {
            self.camera_controller.process_mouse(dx as f32, dy as f32);
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.projection.resize(new_size.width, new_size.height);

            // Recreate depth texture for rendering
            self.depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            self.depth_view = self.depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Recreate depth copy texture for water shader sampling
            self.depth_copy_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth Copy Texture"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.depth_copy_view = self.depth_copy_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Recreate water bind group with new depth copy view
            self.water_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Water Bind Group"),
                layout: &self.water_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.depth_copy_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.depth_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.water_time_buffer.as_entire_binding(),
                    },
                ],
            });
            
            // === RESIZE POST-PROCESSING TEXTURES ===
            
            // Recreate Scene Texture
            self.scene_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Scene Texture"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.scene_texture_view = self.scene_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Recreate Post Process Texture
            self.post_process_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Post Process Texture"),
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.post_process_texture_view = self.post_process_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Recreate Motion Blur Bind Group (reads from scene_texture)
            self.motion_blur_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Motion Blur Bind Group"),
                layout: &self.motion_blur_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.motion_blur_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.scene_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.scene_sampler),
                    },
                ],
            });

            // Recreate Underwater Bind Group (reads from post_process_texture after motion blur)
            self.underwater_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Underwater Bind Group"),
                layout: &self.underwater_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.underwater_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.post_process_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.scene_sampler),
                    },
                ],
            });

            // Update UI aspect ratio
            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct UiUniform {
                aspect_ratio: f32,
                _padding: [f32; 3],
            }
            let aspect_ratio = new_size.width as f32 / new_size.height as f32;
            let ui_uniform = UiUniform {
                aspect_ratio,
                _padding: [0.0; 3],
            };
            self.queue.write_buffer(
                &self.ui_uniform_buffer,
                0,
                bytemuck::cast_slice(&[ui_uniform]),
            );

            // Rebuild crosshair vertices with new aspect ratio
            let crosshair_vertices = Self::build_crosshair_vertices(aspect_ratio);
            self.queue.write_buffer(
                &self.crosshair_vertex_buffer,
                0,
                bytemuck::cast_slice(&crosshair_vertices),
            );
        }
    }

    fn build_crosshair_vertices(aspect_ratio: f32) -> Vec<UiVertex> {
        let crosshair_size = 0.06;
        let crosshair_thickness = 0.01;
        let crosshair_color = [1.0, 1.0, 1.0, 0.7];

        // Correct X coordinates for aspect ratio to maintain 1:1 ratio
        let h_size = crosshair_size / aspect_ratio;
        let h_thick = crosshair_thickness / aspect_ratio;

        vec![
            // Horizontal bar (X scaled for aspect ratio)
            UiVertex { position: [-h_size, -crosshair_thickness], color: crosshair_color },
            UiVertex { position: [h_size, -crosshair_thickness], color: crosshair_color },
            UiVertex { position: [h_size, crosshair_thickness], color: crosshair_color },
            UiVertex { position: [-h_size, -crosshair_thickness], color: crosshair_color },
            UiVertex { position: [h_size, crosshair_thickness], color: crosshair_color },
            UiVertex { position: [-h_size, crosshair_thickness], color: crosshair_color },
            // Vertical bar (X scaled for aspect ratio)
            UiVertex { position: [-h_thick, -crosshair_size], color: crosshair_color },
            UiVertex { position: [h_thick, -crosshair_size], color: crosshair_color },
            UiVertex { position: [h_thick, crosshair_size], color: crosshair_color },
            UiVertex { position: [-h_thick, -crosshair_size], color: crosshair_color },
            UiVertex { position: [h_thick, crosshair_size], color: crosshair_color },
            UiVertex { position: [-h_thick, crosshair_size], color: crosshair_color },
        ]
    }

    fn rebuild_hud_vertices(&mut self) {
        let screen_w = self.size.width as f32;
        let screen_h = self.size.height as f32;

        let slots = self.player.inventory.size.max(1);
        let selected = self.player.inventory.selected_slot.min(slots - 1);

        // Layout in pixels
        let slot_size = 80.0;
        let slot_gap = 8.0;
        let margin_bottom = 18.0;
        let total_w = (slots as f32) * slot_size + (slots as f32 - 1.0) * slot_gap;
        let start_x = (screen_w - total_w) * 0.5;
        let start_y = screen_h - margin_bottom - slot_size;

        let mut verts: Vec<UiVertex> = Vec::with_capacity(4096);

        // === FPS Counter (top-left) ===
        let fps_text = format!("{} FPS", self.fps as u32);
        let fps_color = [1.0, 1.0, 1.0, 0.9];
        let fps_bg_color = [0.0, 0.0, 0.0, 0.5];
        let fps_x = 10.0;
        let fps_y = 10.0;
        let fps_scale = 2.0;
        let fps_char_w = 6.0 * fps_scale; // 5 pixels + 1 spacing
        let fps_char_h = 7.0 * fps_scale;
        let fps_text_width = fps_text.len() as f32 * fps_char_w;
        // Background
        bitmap_font::push_rect_px(
            &mut verts,
            fps_x - 4.0,
            fps_y - 4.0,
            fps_text_width + 8.0,
            fps_char_h + 8.0,
            fps_bg_color,
            screen_w,
            screen_h,
        );
        bitmap_font::draw_text_quads(
            &mut verts,
            &fps_text,
            fps_x,
            fps_y,
            fps_scale,
            fps_scale,
            fps_color,
            screen_w,
            screen_h,
        );

        // === Chunk Outline Toggle Indicator (below FPS) ===
        let (debug_text, debug_color, debug_bg_color) = if self.show_chunk_outlines {
            ("F1 - CHUNK OUTLINE: ON", [0.5, 1.0, 0.5, 1.0], [0.0, 0.2, 0.0, 0.6])  // Light green when on
        } else {
            ("F1 - CHUNK OUTLINE: OFF", [1.0, 1.0, 1.0, 0.9], [0.0, 0.0, 0.0, 0.5])  // White when off
        };
        let debug_y = fps_y + fps_char_h + 8.0;
        let debug_text_width = debug_text.len() as f32 * fps_char_w;
        // Background
        bitmap_font::push_rect_px(
            &mut verts,
            fps_x - 4.0,
            debug_y - 4.0,
            debug_text_width + 8.0,
            fps_char_h + 8.0,
            debug_bg_color,
            screen_w,
            screen_h,
        );
        bitmap_font::draw_text_quads(
            &mut verts,
            debug_text,
            fps_x,
            debug_y,
            fps_scale,
            fps_scale,
            debug_color,
            screen_w,
            screen_h,
        );

        // === Noclip Toggle Indicator (below Chunk Outline) ===
        let (noclip_text, noclip_color, noclip_bg_color) = if self.noclip_mode {
            ("F2 - NOCLIP: ON", [0.5, 1.0, 0.5, 1.0], [0.0, 0.2, 0.0, 0.6])  // Light green when on
        } else {
            ("F2 - NOCLIP: OFF", [1.0, 1.0, 1.0, 0.9], [0.0, 0.0, 0.0, 0.5])  // White when off
        };
        let noclip_y = debug_y + fps_char_h + 8.0;
        let noclip_text_width = noclip_text.len() as f32 * fps_char_w;
        // Background
        bitmap_font::push_rect_px(
            &mut verts,
            fps_x - 4.0,
            noclip_y - 4.0,
            noclip_text_width + 8.0,
            fps_char_h + 8.0,
            noclip_bg_color,
            screen_w,
            screen_h,
        );
        bitmap_font::draw_text_quads(
            &mut verts,
            noclip_text,
            fps_x,
            noclip_y,
            fps_scale,
            fps_scale,
            noclip_color,
            screen_w,
            screen_h,
        );

        // === Debug Axes (top-right) ===
        self.build_debug_axes(&mut verts, screen_w, screen_h);

        // === Player Position (below compass) ===
        let pos_text = format!(
            "X:{:.2}, Y:{:.2}, Z:{:.2}",
            self.camera.position.x,
            self.camera.position.y,
            self.camera.position.z
        );
        let pos_color = [1.0, 1.0, 1.0, 0.9];
        let pos_bg_color = [0.0, 0.0, 0.0, 0.5];
        let pos_scale = 2.0;
        let pos_char_w = 6.0 * pos_scale;
        let pos_char_h = 7.0 * pos_scale;
        let pos_text_width = pos_text.len() as f32 * pos_char_w;
        let pos_x = screen_w - pos_text_width - 9.0; // Right-aligned with padding
        let pos_y = 125.0; // Below compass

        // Background
        bitmap_font::push_rect_px(
            &mut verts,
            pos_x - 4.0,
            pos_y - 4.0,
            pos_text_width + 8.0,
            pos_char_h + 8.0,
            pos_bg_color,
            screen_w,
            screen_h,
        );
        bitmap_font::draw_text_quads(
            &mut verts,
            &pos_text,
            pos_x,
            pos_y,
            pos_scale,
            pos_scale,
            pos_color,
            screen_w,
            screen_h,
        );

        // === Health Bar ===
        let health_bar_w = 300.0;
        let health_bar_h = 30.0;
        let health_bar_border = 2.0;
        let health_bar_padding = 6.0; // Right padding for text
        //let health_bar_x = start_x;
        let health_bar_x = (screen_w - health_bar_w) / 2.0;
        let health_bar_y = 24.0;
        let health_pct = self.player.health / self.player.max_health;

        // Dark background
        bitmap_font::push_rect_px(
            &mut verts,
            health_bar_x,
            health_bar_y,
            health_bar_w,
            health_bar_h,
            [0.0, 0.0, 0.0, 0.25],
            screen_w,
            screen_h,
        );

        // White border (4 thin quads)
        let border_color = [1.0, 1.0, 1.0, 0.85];
        // top
        bitmap_font::push_rect_px(&mut verts, health_bar_x, health_bar_y, health_bar_w, health_bar_border, border_color, screen_w, screen_h);
        // bottom
        bitmap_font::push_rect_px(&mut verts, health_bar_x, health_bar_y + health_bar_h - health_bar_border, health_bar_w, health_bar_border, border_color, screen_w, screen_h);
        // left
        bitmap_font::push_rect_px(&mut verts, health_bar_x, health_bar_y, health_bar_border, health_bar_h, border_color, screen_w, screen_h);
        // right
        bitmap_font::push_rect_px(&mut verts, health_bar_x + health_bar_w - health_bar_border, health_bar_y, health_bar_border, health_bar_h, border_color, screen_w, screen_h);

        // Red health fill (inside borders)
        let fill_x = health_bar_x + health_bar_border;
        let fill_y = health_bar_y + health_bar_border;
        let fill_max_w = health_bar_w - health_bar_border * 2.0;
        let fill_h = health_bar_h - health_bar_border * 2.0;
        let fill_w = fill_max_w * health_pct;
        if fill_w > 0.0 {
            bitmap_font::push_rect_px(
                &mut verts,
                fill_x,
                fill_y,
                fill_w,
                fill_h,
                [0.8, 0.1, 0.1, 0.9],
                screen_w,
                screen_h,
            );
        }

        // Percentage text (anchored right inside the bar)
        let health_text = format!("{}%", (health_pct * 100.0).round() as u32);
        let health_text_scale = 2.0;
        let health_char_w = 6.0 * health_text_scale;
        let health_char_h = 7.0 * health_text_scale;
        let health_text_w = health_text.len() as f32 * health_char_w;
        let health_text_x = health_bar_x + health_bar_w - health_bar_border - health_bar_padding - health_text_w;
        let health_text_y = health_bar_y + (health_bar_h - health_char_h) * 0.5;
        bitmap_font::draw_text_quads(
            &mut verts,
            &health_text,
            health_text_x,
            health_text_y,
            health_text_scale,
            health_text_scale,
            [1.0, 1.0, 1.0, 0.95],
            screen_w,
            screen_h,
        );

        let fill = [1.0, 1.0, 1.0, 0.10];
        let outline = [1.0, 1.0, 1.0, 0.85];
        let outline_selected = [1.0, 1.0, 1.0, 1.0];

        // Slot content layout
        let slot_padding = 6.0;
        let name_scale = 2.0;
        let name_char_w = 6.0 * name_scale;
        let name_char_h = 7.0 * name_scale;
        let label_gap = 6.0;

        for i in 0..slots {
            let x = start_x + i as f32 * (slot_size + slot_gap);
            let y = start_y;

            // Fill quad
            bitmap_font::push_rect_px(
                &mut verts,
                x,
                y,
                slot_size,
                slot_size,
                fill,
                screen_w,
                screen_h,
            );

            // Outline as 4 thin quads (thicker for selected)
            let thick = if i == selected { 5.0 } else { 1.5 };
            let c = if i == selected { outline_selected } else { outline };
            // top
            bitmap_font::push_rect_px(&mut verts, x, y, slot_size, thick, c, screen_w, screen_h);
            // bottom
            bitmap_font::push_rect_px(
                &mut verts,
                x,
                y + slot_size - thick,
                slot_size,
                thick,
                c,
                screen_w,
                screen_h,
            );
            // left
            bitmap_font::push_rect_px(&mut verts, x, y, thick, slot_size, c, screen_w, screen_h);
            // right
            bitmap_font::push_rect_px(
                &mut verts,
                x + slot_size - thick,
                y,
                thick,
                slot_size,
                c,
                screen_w,
                screen_h,
            );

            // Slot content
            if let Some(stack) = self.player.inventory.get_slot(i) {
                // Block color fill (with padding)
                let bc = stack.block_type.get_color();
                let block_color = [bc[0], bc[1], bc[2], 1.0];
                bitmap_font::push_rect_px(
                    &mut verts,
                    x + slot_padding,
                    y + slot_padding,
                    slot_size - slot_padding * 2.0,
                    slot_size - slot_padding * 2.0,
                    block_color,
                    screen_w,
                    screen_h,
                );

                // Only show the name for the currently selected slot, positioned just above the slot.
                if i == selected {
                    let name = stack.block_type.display_name();
                    let name_color = [1.0, 1.0, 1.0, 0.95];
                    let name_w = (name.chars().count() as f32) * name_char_w;
                    let name_x = x + (slot_size - name_w) * 0.5;
                    let name_y = (y - name_char_h - label_gap).max(0.0);
                    // Optional background for readability
                    bitmap_font::push_rect_px(
                        &mut verts,
                        name_x - 4.0,
                        name_y - 4.0,
                        name_w + 8.0,
                        name_char_h + 8.0,
                        [0.0, 0.0, 0.0, 0.45],
                        screen_w,
                        screen_h,
                    );
                    bitmap_font::draw_text_quads(
                        &mut verts,
                        name,
                        name_x,
                        name_y,
                        name_scale,
                        name_scale,
                        name_color,
                        screen_w,
                        screen_h,
                    );
                }

                // Count: show in white over the block color in the bottom-left (with padding).
                // Show count if > 0.0
                if stack.count > 0.0 {
                    let count_text = format!("{:.2}", stack.count);
                    let count_color = [1.0, 1.0, 1.0, 0.95];
                    let count_scale = 2.0;
                    let count_w = (count_text.chars().count() as f32) * 6.0 * count_scale;
                    let count_x = x + (slot_size - count_w) * 0.5;
                    //let count_x = x + slot_padding + 7.0;
                    let count_y = y + slot_size - (slot_padding + 6.0) - (7.0 * count_scale) - 1.0;
                    let count_background_padding = 4.0;

                    // Background for count text
                    bitmap_font::push_rect_px(
                        &mut verts,
                        count_x - count_background_padding, // x
                        count_y - count_background_padding, // y
                        (count_text.len() as f32) * 6.0 * count_scale + (count_background_padding * 2.0), // width
                        7.0 * count_scale + (count_background_padding * 2.0), // height
                        [0.0, 0.0, 0.0, 0.55], // color
                        screen_w,
                        screen_h,
                    );

                    bitmap_font::draw_text_quads(
                        &mut verts,
                        &count_text,
                        count_x,
                        count_y,
                        count_scale,
                        count_scale,
                        count_color,
                        screen_w,
                        screen_h,
                    );
                }
            }
        }

        self.hud_vertex_count = verts.len() as u32;
        self.queue
            .write_buffer(&self.hud_vertex_buffer, 0, bytemuck::cast_slice(&verts));
    }

    fn build_debug_axes(&self, verts: &mut Vec<UiVertex>, screen_w: f32, screen_h: f32) {
        // Debug axes gizmo in top-right corner
        // Colors: X = Red, Y = Green, Z = Blue (R-G-B matches X-Y-Z alphabetically)
        let margin = 60.0;
        let center_x = screen_w - margin;
        let center_y = margin;
        let axis_length = 40.0;
        let line_thickness = 5.0;

        // Get camera rotation (yaw and pitch)
        let yaw = self.camera.yaw.0;
        let pitch = self.camera.pitch.0;

        // Create rotation matrix (view rotation)
        // We need to transform world axes by the inverse of the camera's view rotation
        // to show how world axes appear from the camera's perspective
        let cos_yaw = yaw.cos();
        let sin_yaw = yaw.sin();
        let cos_pitch = pitch.cos();
        let sin_pitch = pitch.sin();

        // View rotation matrix (rotates world into view space)
        // First rotate around Y (yaw), then around X (pitch)
        // The result shows world axes as seen from camera
        let rotate_world_to_view = |world_axis: Vector3<f32>| -> (f32, f32) {
            // Apply yaw rotation (around Y axis)
            let x1 = world_axis.x * cos_yaw + world_axis.z * sin_yaw;
            let y1 = world_axis.y;
            let z1 = -world_axis.x * sin_yaw + world_axis.z * cos_yaw;

            // Apply pitch rotation (around X axis)
            let x2 = x1;
            let y2 = y1 * cos_pitch - z1 * sin_pitch;
            let _z2 = y1 * sin_pitch + z1 * cos_pitch;

            // Project to 2D (x2 is right, y2 is up in screen space)
            (x2, -y2) // Negate y because screen Y goes down
        };

        // World axis directions
        let x_axis = Vector3::new(1.0, 0.0, 0.0);
        let y_axis = Vector3::new(0.0, 1.0, 0.0);
        let z_axis = Vector3::new(0.0, 0.0, 1.0);

        // Transform to screen space
        let (x_screen_x, x_screen_y) = rotate_world_to_view(x_axis);
        let (y_screen_x, y_screen_y) = rotate_world_to_view(y_axis);
        let (z_screen_x, z_screen_y) = rotate_world_to_view(z_axis);

        // Helper to draw a line as a quad from center to endpoint
        let draw_axis_line = |verts: &mut Vec<UiVertex>, dx: f32, dy: f32, color: [f32; 4]| {
            let end_x = center_x + dx * axis_length;
            let end_y = center_y + dy * axis_length;

            // Calculate perpendicular for line thickness
            let len = (dx * dx + dy * dy).sqrt().max(0.001);
            let perp_x = -dy / len * line_thickness * 0.5;
            let perp_y = dx / len * line_thickness * 0.5;

            // Create quad vertices (two triangles)
            let p0 = (center_x + perp_x, center_y + perp_y);
            let p1 = (center_x - perp_x, center_y - perp_y);
            let p2 = (end_x - perp_x, end_y - perp_y);
            let p3 = (end_x + perp_x, end_y + perp_y);

            // Convert to clip space
            let to_clip = |px: f32, py: f32| -> [f32; 2] {
                [
                    (px / screen_w) * 2.0 - 1.0,
                    1.0 - (py / screen_h) * 2.0,
                ]
            };

            let v0 = UiVertex { position: to_clip(p0.0, p0.1), color };
            let v1 = UiVertex { position: to_clip(p1.0, p1.1), color };
            let v2 = UiVertex { position: to_clip(p2.0, p2.1), color };
            let v3 = UiVertex { position: to_clip(p3.0, p3.1), color };

            // Two triangles
            verts.push(v0);
            verts.push(v1);
            verts.push(v2);
            verts.push(v0);
            verts.push(v2);
            verts.push(v3);
        };

        // Draw background
        let bg_color = [0.0, 0.0, 0.0, 0.5];
        let bg_radius = axis_length + 15.0;
        bitmap_font::push_rect_px(
            verts,
            center_x - bg_radius,
            center_y - bg_radius,
            bg_radius * 2.0,
            bg_radius * 2.0,
            bg_color,
            screen_w,
            screen_h,
        );

        // Draw axes (R-G-B for X-Y-Z)
        let red = [1.0, 0.2, 0.2, 1.0];
        let green = [0.2, 1.0, 0.2, 1.0];
        let blue = [0.3, 0.5, 1.0, 1.0];

        draw_axis_line(verts, x_screen_x, x_screen_y, red);   // X axis - Red
        draw_axis_line(verts, y_screen_x, y_screen_y, green); // Y axis - Green
        draw_axis_line(verts, z_screen_x, z_screen_y, blue);  // Z axis - Blue

        // Draw axis labels
        let label_offset = axis_length + 8.0;
        let label_scale = 1.5;

        // X label
        let x_label_x = center_x + x_screen_x * label_offset - 4.0;
        let x_label_y = center_y + x_screen_y * label_offset - 5.0;
        bitmap_font::draw_text_quads(verts, "X", x_label_x, x_label_y, label_scale, label_scale, red, screen_w, screen_h);

        // Y label
        let y_label_x = center_x + y_screen_x * label_offset - 4.0;
        let y_label_y = center_y + y_screen_y * label_offset - 5.0;
        bitmap_font::draw_text_quads(verts, "Y", y_label_x, y_label_y, label_scale, label_scale, green, screen_w, screen_h);

        // Z label
        let z_label_x = center_x + z_screen_x * label_offset - 4.0;
        let z_label_y = center_y + z_screen_y * label_offset - 5.0;
        bitmap_font::draw_text_quads(verts, "Z", z_label_x, z_label_y, label_scale, label_scale, blue, screen_w, screen_h);
    }

    /// Generate line vertices for chunk boundary outlines
    fn build_chunk_outline_vertices(&self) -> Vec<LineVertex> {
        let mut vertices = Vec::new();

        for (&(cx, cz), _chunk) in &self.world.chunks {
            // World coordinates of chunk boundaries
            let x0 = (cx * CHUNK_SIZE as i32) as f32;
            let x1 = x0 + CHUNK_SIZE as f32;
            let z0 = (cz * CHUNK_SIZE as i32) as f32;
            let z1 = z0 + CHUNK_SIZE as f32;
            let y0 = 0.0f32;
            let y1 = CHUNK_HEIGHT as f32;

            // Vertical edges (4 corners)
            vertices.extend_from_slice(&[
                LineVertex { position: [x0, y0, z0] }, LineVertex { position: [x0, y1, z0] },
                LineVertex { position: [x1, y0, z0] }, LineVertex { position: [x1, y1, z0] },
                LineVertex { position: [x0, y0, z1] }, LineVertex { position: [x0, y1, z1] },
                LineVertex { position: [x1, y0, z1] }, LineVertex { position: [x1, y1, z1] },
            ]);

            // Bottom edges
            vertices.extend_from_slice(&[
                LineVertex { position: [x0, y0, z0] }, LineVertex { position: [x1, y0, z0] },
                LineVertex { position: [x1, y0, z0] }, LineVertex { position: [x1, y0, z1] },
                LineVertex { position: [x1, y0, z1] }, LineVertex { position: [x0, y0, z1] },
                LineVertex { position: [x0, y0, z1] }, LineVertex { position: [x0, y0, z0] },
            ]);

            // Top edges
            vertices.extend_from_slice(&[
                LineVertex { position: [x0, y1, z0] }, LineVertex { position: [x1, y1, z0] },
                LineVertex { position: [x1, y1, z0] }, LineVertex { position: [x1, y1, z1] },
                LineVertex { position: [x1, y1, z1] }, LineVertex { position: [x0, y1, z1] },
                LineVertex { position: [x0, y1, z1] }, LineVertex { position: [x0, y1, z0] },
            ]);
        }

        vertices
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state,
                        physical_key: PhysicalKey::Code(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    KeyCode::KeyW => {
                        self.camera_controller.forward = is_pressed;
                        true
                    }
                    KeyCode::KeyS => {
                        self.camera_controller.backward = is_pressed;
                        true
                    }
                    KeyCode::KeyA => {
                        self.camera_controller.left = is_pressed;
                        true
                    }
                    KeyCode::KeyD => {
                        self.camera_controller.right = is_pressed;
                        true
                    }
                    KeyCode::Space => {
                        self.camera_controller.jump_held = is_pressed;
                        true
                    }
                    KeyCode::ShiftLeft => {
                        self.camera_controller.shift_held = is_pressed;
                        true
                    }
                    KeyCode::KeyE => {
                        if is_pressed {
                            self.show_inventory = !self.show_inventory;
                            println!("=== INVENTORY ===");
                            for (i, slot) in self.player.inventory.slots.iter().enumerate() {
                                if let Some(stack) = slot {
                                    let marker = if i == self.player.inventory.selected_slot {
                                        ">"
                                    } else {
                                        " "
                                    };
                                    println!(
                                        "{} Slot {}: {:?} x{}",
                                        marker, i, stack.block_type, stack.count
                                    );
                                }
                            }
                        }
                        true
                    }
                    KeyCode::KeyC => {
                        if is_pressed {
                            self.show_crafting = !self.show_crafting;
                            println!("=== CRAFTING ===");
                            let available = self
                                .crafting_system
                                .get_available_recipes(&self.player.inventory);
                            for (i, recipe_idx) in available.iter().enumerate() {
                                let recipe = &self.crafting_system.get_recipes()[*recipe_idx];
                                println!("Recipe {}: {:?}", i, recipe);
                            }
                            println!("Press number keys 1-9 to craft");
                        }
                        true
                    }
                    KeyCode::Digit1
                    | KeyCode::Digit2
                    | KeyCode::Digit3
                    | KeyCode::Digit4
                    | KeyCode::Digit5
                    | KeyCode::Digit6
                    | KeyCode::Digit7
                    | KeyCode::Digit8
                    | KeyCode::Digit9 => {
                        if is_pressed {
                            let num = match keycode {
                                KeyCode::Digit1 => 0,
                                KeyCode::Digit2 => 1,
                                KeyCode::Digit3 => 2,
                                KeyCode::Digit4 => 3,
                                KeyCode::Digit5 => 4,
                                KeyCode::Digit6 => 5,
                                KeyCode::Digit7 => 6,
                                KeyCode::Digit8 => 7,
                                KeyCode::Digit9 => 8,
                                _ => 0,
                            };

                            if self.show_crafting {
                                let available = self
                                    .crafting_system
                                    .get_available_recipes(&self.player.inventory);
                                if num < available.len() {
                                    if self.crafting_system.craft(&mut self.player.inventory, available[num]) {
                                        println!("Crafted successfully!");
                                    }
                                }
                            } else {
                                self.player.inventory.selected_slot = num;
                                println!("Selected slot {}", num);
                            }
                        }
                        true
                    }
                    KeyCode::F1 => {
                        if is_pressed {
                            self.show_chunk_outlines = !self.show_chunk_outlines;
                            println!("Chunk outlines: {}", if self.show_chunk_outlines { "ON" } else { "OFF" });
                        }
                        true
                    }
                    KeyCode::F2 => {
                        if is_pressed {
                            self.noclip_mode = !self.noclip_mode;
                            println!("Noclip: {}", if self.noclip_mode { "ON" } else { "OFF" });
                        }
                        true
                    }
                    _ => false,
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                self.mouse_pressed = is_pressed;
                self.left_mouse_held = is_pressed;

                if !is_pressed {
                    // Mouse released - cancel any breaking in progress
                    self.breaking_state = None;
                }
                true
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Right,
                ..
            } => {
                // Only handle block place if mouse is captured (in-game)
                if self.mouse_captured {
                    self.handle_block_place();
                }
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // Only handle scroll if mouse is captured (in-game) and not in crafting mode
                if self.mouse_captured && !self.show_crafting {
                    let scroll_amount = match delta {
                        MouseScrollDelta::LineDelta(_, y) => *y as i32,
                        MouseScrollDelta::PixelDelta(pos) => {
                            // Convert pixel delta to line units (typical ~40 pixels per line)
                            (pos.y / 40.0) as i32
                        }
                    };

                    if scroll_amount != 0 {
                        let slots = self.player.inventory.size;
                        let current = self.player.inventory.selected_slot as i32;
                        // Scroll up = previous slot, scroll down = next slot
                        let new_slot = (current - scroll_amount).rem_euclid(slots as i32) as usize;
                        self.player.inventory.selected_slot = new_slot;
                    }
                }
                true
            }
            _ => false,
        }
    }

    fn complete_block_break(&mut self, x: i32, y: i32, z: i32, block_type: BlockType) {
        // Spawn dropped items (4 mini-blocks)
        let block_pos = Point3::new(x as f32, y as f32, z as f32);
        self.dropped_item_manager.spawn_drops(block_pos, block_type);

        // Spawn break particles
        self.particle_manager.spawn_block_break(block_pos, block_type);

        // Remove the block from the world
        self.world.set_block_world(x, y, z, BlockType::Air);
        println!("Broke block: {:?}", block_type);

        // Break any cross-model block sitting on top (e.g. grass tufts)
        let above = self.world.get_block_world(x, y + 1, z);
        if above.is_cross_model() {
            self.world.set_block_world(x, y + 1, z, BlockType::Air);
        }
    }

    /// Creates vertices for the breaking overlay on visible faces of a block
    fn create_breaking_overlay_vertices(&self, x: i32, y: i32, z: i32, destroy_stage: u32) -> (Vec<Vertex>, Vec<u16>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let world_pos = Vector3::new(x as f32, y as f32, z as f32);
        let block_type = self.world.get_block_world(x, y, z);

        // Small offset to render in front of block faces (prevents z-fighting)
        let offset = 0.001;

        let face_directions: [(i32, i32, i32); 6] = [
            (0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0),
        ];

        // Get UVs for the destroy texture
        let tex_index = TEX_DESTROY_BASE + destroy_stage;
        let uvs = get_face_uvs(tex_index);

        for (face_idx, &(dx, dy, dz)) in face_directions.iter().enumerate() {
            let neighbor = self.world.get_block_world(x + dx, y + dy, z + dz);

            // Only render overlay on visible faces (where neighbor is transparent)
            if neighbor.is_transparent() {
                // Offset position slightly toward camera based on face normal
                let offset_pos = Vector3::new(
                    world_pos.x + dx as f32 * offset,
                    world_pos.y + dy as f32 * offset,
                    world_pos.z + dz as f32 * offset,
                );

                let face_verts = create_face_vertices(offset_pos, block_type, face_idx, 1.0, tex_index, uvs, [1.0; 4]);

                let base_index = vertices.len() as u16;
                vertices.extend_from_slice(&face_verts);
                indices.extend_from_slice(&[
                    base_index, base_index + 1, base_index + 2,
                    base_index + 2, base_index + 3, base_index,
                ]);
            }
        }

        (vertices, indices)
    }

    fn update_block_breaking(&mut self, dt: f32) {
        if !self.left_mouse_held || !self.mouse_captured {
            // Not holding left mouse or not in game
            self.breaking_state = None;
            return;
        }

        // Get currently targeted block
        let direction = self.camera.get_direction();
        let target = self.player.raycast_block(direction, &self.world);

        match (&mut self.breaking_state, target) {
            (Some(state), Some((x, y, z, _))) => {
                if state.block_pos == (x, y, z) {
                    // Still targeting same block - increment progress
                    let durability = state.block_type.get_durability();
                    if durability > 0.0 {
                        state.progress += dt / durability;

                        if state.progress >= 1.0 {
                            // Block broken!
                            let block_type = state.block_type;
                            let pos = state.block_pos;
                            self.breaking_state = None;
                            self.complete_block_break(pos.0, pos.1, pos.2, block_type);
                        }
                    }
                } else {
                    // Targeting different block - reset
                    let block_type = self.world.get_block_world(x, y, z);
                    if block_type.is_breakable() {
                        self.breaking_state = Some(BreakingState::new((x, y, z), block_type));
                    } else {
                        self.breaking_state = None;
                    }
                }
            }
            (None, Some((x, y, z, _))) => {
                // Start breaking new block
                let block_type = self.world.get_block_world(x, y, z);
                if block_type.is_breakable() {
                    self.breaking_state = Some(BreakingState::new((x, y, z), block_type));
                }
            }
            (_, None) => {
                // Not targeting any block
                self.breaking_state = None;
            }
        }
    }

    fn handle_block_place(&mut self) {
        if let Some(selected) = self.player.inventory.get_selected_item() {
            // Only allow placement if player has at least 1.0 of the block
            if selected.count < 1.0 {
                return;
            }

            let block_type = selected.block_type;
            let direction = self.camera.get_direction();

            if let Some((x, y, z, normal)) = self.player.raycast_block(direction, &self.world) {
                let place_x = x + normal.x;
                let place_y = y + normal.y;
                let place_z = z + normal.z;

                if self.world.get_block_world(place_x, place_y, place_z) == BlockType::Air {
                    // Check if placement would overlap with player's collision capsule
                    let player_pos = self.camera.position;
                    let player_radius = 0.25;
                    let player_height = 1.6;
                    let player_min_x = player_pos.x - player_radius;
                    let player_max_x = player_pos.x + player_radius;
                    let player_min_y = player_pos.y - player_height;
                    let player_max_y = player_pos.y;
                    let player_min_z = player_pos.z - player_radius;
                    let player_max_z = player_pos.z + player_radius;

                    let block_min_x = place_x as f32;
                    let block_max_x = place_x as f32 + 1.0;
                    let block_min_y = place_y as f32;
                    let block_max_y = place_y as f32 + 1.0;
                    let block_min_z = place_z as f32;
                    let block_max_z = place_z as f32 + 1.0;

                    let overlaps = player_max_x > block_min_x && player_min_x < block_max_x
                        && player_max_y > block_min_y && player_min_y < block_max_y
                        && player_max_z > block_min_z && player_min_z < block_max_z;

                    if overlaps {
                        return; // Don't place block inside player
                    }

                    self.world
                        .set_block_world(place_x, place_y, place_z, block_type);
                    self.player
                        .inventory
                        .remove_item(self.player.inventory.selected_slot, 1.0);
                    println!("Placed block: {:?}", block_type);
                }
            }
        }
    }

    pub fn update(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        
        // Update Underwater shader time
        let total_time = (now - self.start_time).as_secs_f32();
        self.queue.write_buffer(
            &self.underwater_uniform_buffer,
            0,
            bytemuck::cast_slice(&[total_time]),
        );

        // Update water wave animation time (only the time component, rest stays constant)
        self.queue.write_buffer(
            &self.water_time_buffer,
            0,
            bytemuck::cast_slice(&[total_time]),
        );

        // Update FPS counter
        self.fps_frame_count += 1;
        self.fps_timer += dt;
        if self.fps_timer >= 0.5 {
            self.fps = self.fps_frame_count as f32 / self.fps_timer;
            self.fps_frame_count = 0;
            self.fps_timer = 0.0;
        }

        // Update camera
        self.camera_underwater = self.camera_controller.update_camera(&mut self.camera, dt, &self.world, self.noclip_mode);
        self.player.position = self.camera.position;

        // Update motion blur uniform from camera rotation velocity
        {
            use crate::camera::{MOTION_BLUR_AMOUNT};
            let yaw_vel = self.camera_controller.yaw_velocity;
            let pitch_vel = self.camera_controller.pitch_velocity;
            // Map rotation velocity to screen-space blur direction
            // yaw -> horizontal, pitch -> vertical; scale by dt-like factor for perceptual sizing
            let blur_scale = 0.015; // converts rad/s to UV-space offset
            let blur_x = yaw_vel * blur_scale;
            let blur_y = pitch_vel * blur_scale;
            let magnitude = (blur_x * blur_x + blur_y * blur_y).sqrt();
            // Normalize direction, strength is magnitude * user amount
            let (dir_x, dir_y) = if magnitude > 0.0001 {
                (blur_x / magnitude, blur_y / magnitude)
            } else {
                (0.0, 0.0)
            };
            let strength = (magnitude * MOTION_BLUR_AMOUNT).min(0.05); // cap to avoid extreme blur
            let uniform_data: [f32; 4] = [dir_x, dir_y, strength, 0.0];
            self.queue.write_buffer(
                &self.motion_blur_uniform_buffer,
                0,
                bytemuck::cast_slice(&uniform_data),
            );
        }

        // Fall damage: velocity threshold of -15.0 (roughly 4+ block fall)
        // Damage scales with how far beyond the threshold
        let fall_vel = self.camera_controller.last_fall_velocity;
        if fall_vel < -15.0 {
            let damage = (fall_vel.abs() - 15.0) * 2.5;
            self.player.take_damage(damage);
            if !self.player.is_alive() {
                self.respawn();
            }
        }
        self.camera_controller.last_fall_velocity = 0.0;

        // Update world
        self.world
            .update_chunks((self.camera.position.x, self.camera.position.z));
        self.world.rebuild_dirty_chunks();

        // Update water simulation
        self.water_simulation.update(&mut self.world, dt);

        // Update enemies
        // TODO: Re-enable enemy spawning after testing
        // self.enemy_manager.update(dt, self.player.position);

        // Update birds
        self.bird_manager.update(dt, self.player.position, &self.world);

        // Update fish
        self.fish_manager.update(dt, self.player.position, &self.world);

        // Update dropped items and collect any that touch the player
        let collected_items = self.dropped_item_manager.update(dt, self.player.position, &self.world);
        for item in collected_items {
            self.player.inventory.add_item(item.block_type, item.value);
        }

        // Update particles
        self.particle_manager.update(dt);

        // Check for enemy damage
        let damage = self.enemy_manager.check_player_damage(self.player.position);
        if damage > 0.0 {
            self.player.take_damage(damage * dt);
            if !self.player.is_alive() {
                self.respawn();
            }
        }

        // Update targeted block (for outline rendering)
        let direction = self.camera.get_direction();
        self.targeted_block = self.player.raycast_block(direction, &self.world)
            .map(|(x, y, z, _)| (x, y, z));

        // Update block breaking
        self.update_block_breaking(dt);

        // Update camera uniform and frustum
        self.camera_uniform
            .update_view_proj(&self.camera, &self.projection);
        self.frustum = Frustum::from_view_proj(
            &(self.projection.calc_matrix() * self.camera.get_view_matrix())
        );
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // Update chunk GPU buffer cache
        self.update_chunk_buffers();
    }

    /// Updates the GPU buffer cache for chunks that have changed
    fn update_chunk_buffers(&mut self) {
        // Collect chunk positions that need buffer updates
        let chunks_to_update: Vec<((i32, i32), u32, bool, bool, bool)> = self.world.chunks.iter()
            .filter_map(|(&pos, chunk)| {
                let cached = self.chunk_buffers.get(&pos);
                let needs_update = match cached {
                    Some(cb) => cb.mesh_version != chunk.mesh_version,
                    None => true,
                };
                if needs_update && !chunk.vertices.is_empty() {
                    Some((pos, chunk.mesh_version, !chunk.vertices.is_empty(), !chunk.water_vertices.is_empty(), !chunk.transparent_vertices.is_empty()))
                } else if needs_update && chunk.vertices.is_empty() {
                    // Remove empty chunks from cache
                    Some((pos, chunk.mesh_version, false, false, false))
                } else {
                    None
                }
            })
            .collect();

        // Update buffers for chunks that need it
        for (pos, mesh_version, has_geometry, has_water, has_transparent) in chunks_to_update {
            if !has_geometry {
                // Remove from cache if chunk has no geometry
                self.chunk_buffers.remove(&pos);
                continue;
            }

            let chunk = match self.world.chunks.get(&pos) {
                Some(c) => c,
                None => continue,
            };

            let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Chunk Vertex Buffer"),
                contents: bytemuck::cast_slice(&chunk.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

            let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Chunk Index Buffer"),
                contents: bytemuck::cast_slice(&chunk.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            let (water_vertex_buffer, water_index_buffer, water_index_count) = if has_water {
                let wvb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Water Vertex Buffer"),
                    contents: bytemuck::cast_slice(&chunk.water_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let wib = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Water Index Buffer"),
                    contents: bytemuck::cast_slice(&chunk.water_indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
                (Some(wvb), Some(wib), chunk.water_indices.len() as u32)
            } else {
                (None, None, 0)
            };

            let (transparent_vertex_buffer, transparent_index_buffer, transparent_index_count) = if has_transparent {
                let tvb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Transparent Vertex Buffer"),
                    contents: bytemuck::cast_slice(&chunk.transparent_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let tib = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Transparent Index Buffer"),
                    contents: bytemuck::cast_slice(&chunk.transparent_indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
                (Some(tvb), Some(tib), chunk.transparent_indices.len() as u32)
            } else {
                (None, None, 0)
            };

            self.chunk_buffers.insert(pos, ChunkBuffers {
                vertex_buffer,
                index_buffer,
                index_count: chunk.indices.len() as u32,
                water_vertex_buffer,
                water_index_buffer,
                water_index_count,
                transparent_vertex_buffer,
                transparent_index_buffer,
                transparent_index_count,
                mesh_version,
            });
        }

        // Remove cached buffers for chunks that no longer exist
        let existing_chunks: std::collections::HashSet<(i32, i32)> =
            self.world.chunks.keys().cloned().collect();
        self.chunk_buffers.retain(|pos, _| existing_chunks.contains(pos));
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.rebuild_hud_vertices();

        let output = self.surface.get_current_texture()?;
        let swap_chain_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
            
        // Always render world to scene_texture for motion blur post-processing.
        // Motion blur then writes to swap_chain (or post_process_texture if underwater).
        let world_render_target = &self.scene_texture_view;

        // --- PASS 1: RENDER WORLD (Opaque chunks & enemies) ---
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("World Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: world_render_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.5,
                            g: 0.7,
                            b: 1.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.texture_atlas.bind_group, &[]);

            // Render chunks using cached GPU buffers with frustum culling
            for (&(cx, cz), buffers) in self.chunk_buffers.iter() {
                // Calculate chunk bounding box
                let min = Vector3::new(
                    (cx * CHUNK_SIZE as i32) as f32,
                    0.0,
                    (cz * CHUNK_SIZE as i32) as f32,
                );
                let max = Vector3::new(
                    min.x + CHUNK_SIZE as f32,
                    CHUNK_HEIGHT as f32,
                    min.z + CHUNK_SIZE as f32,
                );

                // Skip chunks outside the view frustum
                if !self.frustum.is_box_visible(min, max) {
                    continue;
                }

                render_pass.set_vertex_buffer(0, buffers.vertex_buffer.slice(..));
                render_pass.set_index_buffer(buffers.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..buffers.index_count, 0, 0..1);
            }

            // Render enemies
            for enemy in &self.enemy_manager.enemies {
                if enemy.alive {
                    let vertices = create_cube_vertices(
                        cgmath::Vector3::new(
                            enemy.position.x,
                            enemy.position.y,
                            enemy.position.z,
                        ),
                        BlockType::Air,
                        1.0, // Full brightness for enemies
                    );

                    // Override color for enemies
                    let enemy_color = enemy.get_color();
                    let vertices: Vec<Vertex> = vertices
                        .into_iter()
                        .map(|mut v| {
                            v.color = enemy_color;
                            v
                        })
                        .collect();

                    let vertex_buffer =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Enemy Vertex Buffer"),
                                contents: bytemuck::cast_slice(&vertices),
                                usage: wgpu::BufferUsages::VERTEX,
                            });

                    let index_buffer =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Enemy Index Buffer"),
                                contents: bytemuck::cast_slice(CUBE_INDICES),
                                usage: wgpu::BufferUsages::INDEX,
                            });

                    render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                    render_pass.draw_indexed(0..CUBE_INDICES.len() as u32, 0, 0..1);
                }
            }

            // Render dropped item shadows first (so they appear behind items)
            for item in &self.dropped_item_manager.items {
                // Find the ground below the item (search up to 10 blocks down)
                let mut ground_y = None;
                let item_x = item.position.x.floor() as i32;
                let item_z = item.position.z.floor() as i32;
                for check_y in (0..=(item.position.y.floor() as i32)).rev() {
                    if self.world.get_block_world(item_x, check_y, item_z).is_solid() {
                        ground_y = Some(check_y as f32 + 1.0); // Top of the solid block
                        break;
                    }
                }

                if let Some(gy) = ground_y {
                    let height_above_ground = item.position.y - gy;
                    // Only show shadow if within reasonable height (< 5 blocks)
                    if height_above_ground >= 0.0 && height_above_ground < 5.0 {
                        // Alpha decreases with height (0.5 at ground, 0 at 5 blocks up)
                        let alpha = 0.5 * (1.0 - height_above_ground / 5.0);
                        // Shadow radius is slightly larger than item (item is 0.25, shadow is 0.35)
                        let shadow_radius = 0.35;
                        let shadow_center = Point3::new(
                            item.position.x,
                            gy + 0.01, // Just above ground to avoid z-fighting
                            item.position.z,
                        );

                        let (vertices, indices) = create_shadow_vertices(shadow_center, shadow_radius, alpha);

                        let vertex_buffer =
                            self.device
                                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some("Shadow Vertex Buffer"),
                                    contents: bytemuck::cast_slice(&vertices),
                                    usage: wgpu::BufferUsages::VERTEX,
                                });

                        let index_buffer =
                            self.device
                                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some("Shadow Index Buffer"),
                                    contents: bytemuck::cast_slice(&indices),
                                    usage: wgpu::BufferUsages::INDEX,
                                });

                        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                        render_pass
                            .set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                        render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
                    }
                }
            }

            // Render dropped items (mini-blocks)
            for item in &self.dropped_item_manager.items {
                let vertices = create_scaled_cube_vertices(
                    item.position,
                    item.block_type,
                    item.get_size(),
                    1.0, // Full brightness
                );

                let vertex_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Dropped Item Vertex Buffer"),
                            contents: bytemuck::cast_slice(&vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                let index_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Dropped Item Index Buffer"),
                            contents: bytemuck::cast_slice(CUBE_INDICES),
                            usage: wgpu::BufferUsages::INDEX,
                        });

                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..CUBE_INDICES.len() as u32, 0, 0..1);
            }

            // Render particles
            for particle in &self.particle_manager.particles {
                let vertices = create_particle_vertices(
                    particle.position,
                    particle.color,
                    particle.size,
                    particle.get_alpha(),
                );

                let vertex_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Particle Vertex Buffer"),
                            contents: bytemuck::cast_slice(&vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                let index_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Particle Index Buffer"),
                            contents: bytemuck::cast_slice(CUBE_INDICES),
                            usage: wgpu::BufferUsages::INDEX,
                        });

                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..CUBE_INDICES.len() as u32, 0, 0..1);
            }

            // Render birds
            for bird in &self.bird_manager.birds {
                let vertices = create_bird_vertices(bird);
                if vertices.is_empty() {
                    continue;
                }

                // Calculate number of cubes (each cube has 24 vertices)
                let num_cubes = vertices.len() / 24;
                let indices = generate_bird_indices(num_cubes);

                let vertex_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Bird Vertex Buffer"),
                            contents: bytemuck::cast_slice(&vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                let index_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Bird Index Buffer"),
                            contents: bytemuck::cast_slice(&indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });

                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
            }

            // Render fish
            for fish in &self.fish_manager.fish {
                let vertices = create_fish_vertices(fish);
                if vertices.is_empty() {
                    continue;
                }

                // Calculate number of cubes (each cube has 24 vertices)
                let num_cubes = vertices.len() / 24;
                let indices = generate_fish_indices(num_cubes);

                let vertex_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Fish Vertex Buffer"),
                            contents: bytemuck::cast_slice(&vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                let index_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Fish Index Buffer"),
                            contents: bytemuck::cast_slice(&indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });

                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
            }
        }

        // Copy depth buffer to the copy texture for water shader to sample
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: &self.depth_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &self.depth_copy_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
        );

        // --- PASS 2: RENDER WATER (Transparent) ---
        {
            let mut water_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Water Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: world_render_target, // Target same buffer as world
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Keep existing color
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Keep existing depth
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            water_pass.set_pipeline(&self.water_pipeline);
            water_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            water_pass.set_bind_group(1, &self.water_bind_group, &[]);

            // Render water using cached GPU buffers with frustum culling
            for (&(cx, cz), buffers) in self.chunk_buffers.iter() {
                if let (Some(wvb), Some(wib)) = (&buffers.water_vertex_buffer, &buffers.water_index_buffer) {
                    if buffers.water_index_count > 0 {
                        // Calculate chunk bounding box
                        let min = Vector3::new(
                            (cx * CHUNK_SIZE as i32) as f32,
                            0.0,
                            (cz * CHUNK_SIZE as i32) as f32,
                        );
                        let max = Vector3::new(
                            min.x + CHUNK_SIZE as f32,
                            CHUNK_HEIGHT as f32,
                            min.z + CHUNK_SIZE as f32,
                        );

                        // Skip chunks outside the view frustum
                        if !self.frustum.is_box_visible(min, max) {
                            continue;
                        }

                        water_pass.set_vertex_buffer(0, wvb.slice(..));
                        water_pass.set_index_buffer(wib.slice(..), wgpu::IndexFormat::Uint16);
                        water_pass.draw_indexed(0..buffers.water_index_count, 0, 0..1);
                    }
                }
            }
        }

        // --- PASS 3: RENDER SEMI-TRANSPARENT BLOCKS (Ice) ---
        {
            let mut transparent_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Transparent Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: world_render_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Keep existing color
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Keep existing depth
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            transparent_pass.set_pipeline(&self.transparent_pipeline);
            transparent_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            transparent_pass.set_bind_group(1, &self.texture_atlas.bind_group, &[]);

            // Render semi-transparent blocks using cached GPU buffers with frustum culling
            for (&(cx, cz), buffers) in self.chunk_buffers.iter() {
                if let (Some(tvb), Some(tib)) = (&buffers.transparent_vertex_buffer, &buffers.transparent_index_buffer) {
                    if buffers.transparent_index_count > 0 {
                        // Calculate chunk bounding box
                        let min = Vector3::new(
                            (cx * CHUNK_SIZE as i32) as f32,
                            0.0,
                            (cz * CHUNK_SIZE as i32) as f32,
                        );
                        let max = Vector3::new(
                            min.x + CHUNK_SIZE as f32,
                            CHUNK_HEIGHT as f32,
                            min.z + CHUNK_SIZE as f32,
                        );

                        // Skip chunks outside the view frustum
                        if !self.frustum.is_box_visible(min, max) {
                            continue;
                        }

                        transparent_pass.set_vertex_buffer(0, tvb.slice(..));
                        transparent_pass.set_index_buffer(tib.slice(..), wgpu::IndexFormat::Uint16);
                        transparent_pass.draw_indexed(0..buffers.transparent_index_count, 0, 0..1);
                    }
                }
            }
        }

        // --- PASS 4: OVERLAYS (Breaking, Outlines, Debug) ---
        // Render breaking overlay if actively breaking a block
        if let Some(ref breaking_state) = self.breaking_state {
            let (bx, by, bz) = breaking_state.block_pos;
            let destroy_stage = breaking_state.get_destroy_stage();

            // Generate overlay vertices for visible faces
            let (overlay_vertices, overlay_indices) = self.create_breaking_overlay_vertices(bx, by, bz, destroy_stage);

            if !overlay_vertices.is_empty() {
                let mut breaking_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Breaking Overlay Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: world_render_target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Breaking Overlay Vertex Buffer"),
                    contents: bytemuck::cast_slice(&overlay_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });

                let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Breaking Overlay Index Buffer"),
                    contents: bytemuck::cast_slice(&overlay_indices),
                    usage: wgpu::BufferUsages::INDEX,
                });

                breaking_pass.set_pipeline(&self.breaking_pipeline);
                breaking_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                breaking_pass.set_bind_group(1, &self.texture_atlas.bind_group, &[]);
                breaking_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                breaking_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                breaking_pass.draw_indexed(0..overlay_indices.len() as u32, 0, 0..1);
            }
        }

        // Render block outline if targeting a block - uses depth buffer to hide occluded edges
        if let Some((x, y, z)) = self.targeted_block {
            let mut outline_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Outline Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: world_render_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Keep existing depth values
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            let outline_vertices = create_block_outline(x, y, z);
            let outline_vertex_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Outline Vertex Buffer"),
                        contents: bytemuck::cast_slice(&outline_vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    });

            outline_pass.set_pipeline(&self.outline_pipeline);
            outline_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            outline_pass.set_vertex_buffer(0, outline_vertex_buffer.slice(..));
            outline_pass.draw(0..outline_vertices.len() as u32, 0..1);
        }

        // Render chunk outlines if debug mode is enabled
        if self.show_chunk_outlines {
            let mut chunk_outline_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Chunk Outline Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: world_render_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            let chunk_outline_vertices = self.build_chunk_outline_vertices();
            if !chunk_outline_vertices.is_empty() {
                let chunk_outline_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Chunk Outline Vertex Buffer"),
                            contents: bytemuck::cast_slice(&chunk_outline_vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                chunk_outline_pass.set_pipeline(&self.chunk_outline_pipeline);
                chunk_outline_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                chunk_outline_pass.set_vertex_buffer(0, chunk_outline_buffer.slice(..));
                chunk_outline_pass.draw(0..chunk_outline_vertices.len() as u32, 0..1);
            }
        }

        // --- PASS 5: POST-PROCESSING (Motion Blur + Underwater) ---
        // Motion blur always runs: scene_texture -> swap_chain (or post_process_texture if underwater)
        // If underwater, an additional pass applies the underwater effect afterward.
        {
            // If underwater, motion blur writes to post_process_texture; underwater reads it next.
            // If not underwater, motion blur writes directly to swap_chain.
            let blur_target = if self.camera_underwater {
                &self.post_process_texture_view
            } else {
                &swap_chain_view
            };

            let mut blur_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Motion Blur Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: blur_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            blur_pass.set_pipeline(&self.motion_blur_pipeline);
            blur_pass.set_bind_group(0, &self.motion_blur_bind_group, &[]);
            blur_pass.draw(0..3, 0..1); // Full-screen triangle
        }

        // If underwater, apply underwater distortion: post_process_texture -> swap_chain
        if self.camera_underwater {
            let mut underwater_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Underwater Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &swap_chain_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            underwater_pass.set_pipeline(&self.underwater_pipeline);
            underwater_pass.set_bind_group(0, &self.underwater_bind_group, &[]);
            underwater_pass.draw(0..3, 0..1); // Full-screen triangle
        }

        // --- PASS 6: UI (Crosshair & HUD) ---
        // Always render UI directly to Swap Chain so it stays sharp and on top
        {
            let mut ui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("UI Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &swap_chain_view, // Always Screen
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // If underwater, we just drew the background in Pass 4, so Load. 
                        // If NOT underwater, the World pass drew to this view in Pass 1, so Load.
                        load: wgpu::LoadOp::Load, 
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            ui_pass.set_pipeline(&self.ui_pipeline);
            ui_pass.set_bind_group(0, &self.ui_bind_group, &[]);
            ui_pass.set_vertex_buffer(0, self.crosshair_vertex_buffer.slice(..));
            ui_pass.draw(0..12, 0..1); // 12 vertices for crosshair (2 triangles per bar * 2 bars)

            // Hotbar HUD
            ui_pass.set_vertex_buffer(0, self.hud_vertex_buffer.slice(..));
            ui_pass.draw(0..self.hud_vertex_count, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}