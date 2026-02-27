use crate::entities::bird::{BirdManager, create_bird_vertices, generate_bird_indices};
use crate::entities::fish::{FishManager, create_fish_vertices, generate_fish_indices};
use crate::block::{BlockType, Vertex, UiVertex, ItemCubeVertex, ModalVertex, LineVertex, create_cube_vertices, create_block_outline, create_face_vertices, create_scaled_cube_vertices, create_particle_vertices, create_shadow_vertices, CUBE_INDICES};
use crate::modal::{self, Modal};
use crate::audio::AudioManager;
use crate::camera::{Camera, CameraController, CameraUniform, Projection, Frustum};
use crate::chunk::{CHUNK_SIZE, CHUNK_HEIGHT};
use crate::crafting::{CraftingGrid, match_recipe};
use crate::dropped_item::DroppedItemManager;
use crate::entities::enemy::{EnemyManager, create_enemy_vertices, generate_enemy_indices, create_enemy_collision_outlines};
use crate::bitmap_font;
use crate::particle::ParticleManager;
use crate::player::Player;
use crate::texture::{TextureAtlas, get_face_uvs, TEX_DESTROY_BASE, TEX_NONE};
use crate::water::WaterSimulation;
use crate::world::World;
use cgmath::{Point3, Vector3, InnerSpace};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::event::*;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

/// Result of a crafting-UI pixel hit-test
#[derive(Debug, Clone, Copy)]
enum CraftingHit {
    GridSlot(usize, usize),
    OutputSlot,
    InvSlot(usize),
    None,
}

/// All pixel-space coordinates needed to draw / hit-test the crafting UI.
/// Derived from the crafting modal's panel bounds each frame.
struct CraftingLayout {
    ct_slot:  f32,
    ct_gap:   f32,
    grid_w:   f32,
    grid_h:   f32,
    row1_x:   f32,
    row1_y:   f32,
    row2_x:   f32,
    row2_y:   f32,
    out_x:    f32,
    out_y:    f32,
    arrow_x:  f32,
    arrow_y:  f32,
    arrow_scale: f32,
}

/// Compute the crafting UI pixel layout from the modal's panel bounds.
fn crafting_layout(panel_x: f32, panel_y: f32, panel_w: f32, panel_h: f32) -> CraftingLayout {
    use crate::modal::{MODAL_BORDER_PX, MODAL_BEVEL_PX};
    let bevel_pad    = MODAL_BORDER_PX + MODAL_BEVEL_PX; // 12
    let interior_pad = 8.0f32;
    let content_x    = panel_x + bevel_pad + interior_pad;
    let content_w    = panel_w - 2.0 * (bevel_pad + interior_pad);
    let content_bottom_y = panel_y + panel_h - bevel_pad - interior_pad;

    let ct_gap  = 8.0f32;
    // Slot size chosen so 9 slots + 8 gaps exactly fill the content width
    let ct_slot = ((content_w - 8.0 * ct_gap) / 9.0) * 0.75; // 0.75 scale factor

    // Row 1 horizontal layout (centred in content area)
    let arrow_gap   = (content_w * 0.016).max(10.0);
    let arrow_w     = (ct_slot * 0.40).max(24.0);
    let grid_w      = 3.0 * ct_slot + 2.0 * ct_gap;
    let grid_h      = grid_w;
    let row1_w      = grid_w + arrow_gap + arrow_w + arrow_gap + ct_slot;
    let row1_x      = content_x + (content_w - row1_w) * 0.5;

    // Row 1 vertical start: below the crafting modal's title (fixed 20px offset + title height)
    let title_scale  = 10.5f32; // The scale of the AREA used by the title. Naming should probably be updated here!
    let title_h      = 7.0 * title_scale;
    let title_offset = bevel_pad + interior_pad; // align title to top of content area
    let title_end_y  = panel_y + title_offset + title_h;
    let row1_y       = title_end_y + panel_h * 0.12;

    // Row 2 (inventory): anchored to bottom of content area, horizontally centered
    let row2_y = content_bottom_y - ct_slot;
    let inv_w  = 9.0 * ct_slot + 8.0 * ct_gap;
    let row2_x = content_x + (content_w - inv_w) * 0.5;

    // Output slot (centred vertically in grid area)
    let out_x = row1_x + grid_w + arrow_gap + arrow_w + arrow_gap;
    let out_y = row1_y + (grid_h - ct_slot) * 0.5;

    // Arrow text position (centred vertically in grid area, between grid and output)
    let arrow_scale = 7.5f32;
    let arrow_char_h = 7.0 * arrow_scale;
    let arrow_x = row1_x + grid_w + arrow_gap;
    let arrow_y = row1_y + (grid_h - arrow_char_h) * 0.5;

    CraftingLayout { ct_slot, ct_gap, grid_w, grid_h, row1_x, row1_y,
        row2_x, row2_y, out_x, out_y, arrow_x, arrow_y, arrow_scale }
}

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
    // Fog settings
    fog_config: crate::config::FogConfig,
    fog_buffer: wgpu::Buffer,
    fog_bind_group_layout: wgpu::BindGroupLayout,
    fog_bind_group: wgpu::BindGroup,
    world: World,
    player: Player,
    spawn_point: Point3<f32>,
    water_simulation: WaterSimulation,
    enemy_manager: EnemyManager,
    bird_manager: BirdManager,
    fish_manager: FishManager,
    dropped_item_manager: DroppedItemManager,
    particle_manager: ParticleManager,
    last_frame: Instant,
    mouse_pressed: bool,
    window: Arc<Window>,
    show_inventory: bool,
    // UI rendering
    ui_pipeline: wgpu::RenderPipeline,
    ui_bind_group: wgpu::BindGroup,
    ui_uniform_buffer: wgpu::Buffer,
    crosshair_vertex_buffer: wgpu::Buffer,
    hud_vertex_buffer: wgpu::Buffer,
    hud_vertex_count: u32,
    hud_text_vertex_buffer: wgpu::Buffer,
    hud_text_vertex_count: u32,
    item_cube_pipeline: wgpu::RenderPipeline,
    item_cube_bind_group: wgpu::BindGroup,
    item_cube_vertex_buffer: wgpu::Buffer,
    item_cube_vertex_count: u32,
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
    show_enemy_hitboxes: bool,
    smooth_lighting: bool,
    
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

    // Damage flash effect (Post-Processing)
    damage_flash_intensity: f32,
    damage_pipeline: wgpu::RenderPipeline,
    damage_bind_group_layout: wgpu::BindGroupLayout,
    damage_bind_group: wgpu::BindGroup,       // reads post_process_texture
    damage_bind_group_alt: wgpu::BindGroup,   // reads scene_texture (when combined with underwater)
    damage_uniform_buffer: wgpu::Buffer,

    // Texture atlas
    texture_atlas: TextureAtlas,
    // Breaking mechanics
    breaking_pipeline: wgpu::RenderPipeline,
    breaking_state: Option<BreakingState>,
    left_mouse_held: bool,
    // Melee combat
    hit_cooldown: f32,
    hit_indicator_timer: f32,
    crosshair_vertex_count: u32,
    // Cached GPU buffers for chunks - avoids recreating every frame
    chunk_buffers: HashMap<(i32, i32), ChunkBuffers>,
    // Cloud rendering
    cloud_manager: crate::clouds::CloudManager,
    cloud_pipeline: wgpu::RenderPipeline,
    cloud_vertex_buffer: wgpu::Buffer,
    cloud_index_buffer: wgpu::Buffer,
    cloud_index_count: u32,

    // ── Pause / Modal ─────────────────────────────────────────────────────
    pub paused: bool,
    pause_modal: Modal,
    crafting_modal: Modal,
    /// Current cursor position in pixel space (set via CursorMoved events)
    cursor_pos_px: (f32, f32),

    // Pause-background blur pipeline (reads scene_texture)
    pause_blur_pipeline:    wgpu::RenderPipeline,
    pause_blur_bind_group_layout: wgpu::BindGroupLayout,
    pause_blur_bind_group:  wgpu::BindGroup,

    // Modal sand-texture pipeline
    modal_sand_pipeline:    wgpu::RenderPipeline,
    modal_sand_bind_group_layout: wgpu::BindGroupLayout,
    modal_sand_bind_group:  wgpu::BindGroup,
    modal_sand_texture:     wgpu::Texture,
    modal_sand_sampler:     wgpu::Sampler,

    // Modal GPU draw buffers (rebuilt each frame while paused)
    modal_sand_vertex_buffer: wgpu::Buffer,
    modal_ui_vertex_buffer:   wgpu::Buffer,
    modal_ui_vertex_count:    u32,

    // Dropped-item hover (index into dropped_item_manager.items the crosshair is over)
    hovered_dropped_item: Option<usize>,

    // ── Crafting Table UI ─────────────────────────────────────────────────
    crafting_ui_open: bool,
    hovered_crafting_table: bool,
    crafting_grid: CraftingGrid,
    crafting_output: Option<(BlockType, f32)>,
    /// Item currently held on the cursor (picked up from grid or inventory)
    crafting_held: Option<(BlockType, f32)>,
    crafting_hovered_grid: Option<(usize, usize)>,
    crafting_hovered_inv: Option<usize>,
    crafting_hovered_output: bool,
    /// Per-slot selected pickup quantity for the inventory row in the crafting UI
    crafting_inv_qty: [f32; 9],

    // Audio
    audio: Option<AudioManager>,
    /// Looping walking-sound sink; volume is faded in/out each frame.
    walk_sink:   Option<rodio::Sink>,
    walk_volume: f32,
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

        // Fog uniform setup - load from config file
        let config_path = std::path::Path::new("config.toml");
        let fog_config = crate::config::FogConfig::load_or_create(config_path);
        let fog_uniform: crate::config::FogUniform = fog_config.into();

        // Cloud config setup - load from config file
        let cloud_config = crate::config::CloudConfig::load_or_create(config_path);

        // World config setup - load master_seed from config file
        let world_config = crate::config::WorldConfig::load_or_create(config_path);
        log::info!("Using master_seed = {}", world_config.master_seed);

        // Terrain generation config - loaded from config.toml alongside other settings
        let terrain_cfg = std::sync::Arc::new(
            crate::config::TerrainConfig::load_or_create(config_path)
        );

        let fog_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fog Buffer"),
            contents: bytemuck::cast_slice(&[fog_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let fog_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("fog_bind_group_layout"),
        });

        let fog_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &fog_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: fog_buffer.as_entire_binding(),
            }],
            label: Some("fog_bind_group"),
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
                bind_group_layouts: &[&camera_bind_group_layout, &texture_atlas.bind_group_layout, &fog_bind_group_layout],
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

        // Water pipeline layout includes camera bind group, depth texture bind group, and fog bind group
        let water_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Water Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout, &water_bind_group_layout, &fog_bind_group_layout],
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

        let world = World::new(18, world_config.master_seed, terrain_cfg);

        // Restore from a previous session, or find a safe spawn point.
        let player_save = crate::config::PlayerSave::load();
        let spawn_point = if let Some(ref save) = player_save {
            log::info!("Restoring player position ({:.1}, {:.1}, {:.1})", save.x, save.y, save.z);
            Point3::new(save.x, save.y, save.z)
        } else {
            // Find a safe spawn point: scan from top down for a solid block with 2 blocks
            // of non-solid space above it. Try (0,0) first, then nearby positions.
            let mut spawn_y = None;
            let mut spawn_x = 0i32;
            let mut spawn_z = 0i32;

            'search: for &(sx, sz) in &[
                (0, 0), (1, 0), (0, 1), (-1, 0), (0, -1),
                (2, 0), (0, 2), (-2, 0), (0, -2),
                (4, 4), (-4, 4), (4, -4), (-4, -4),
                (8, 0), (0, 8), (-8, 0), (0, -8),
            ] {
                for y in (1..CHUNK_HEIGHT - 2).rev() {
                    let block = world.get_block_world(sx, y as i32, sz);
                    let above1 = world.get_block_world(sx, y as i32 + 1, sz);
                    let above2 = world.get_block_world(sx, y as i32 + 2, sz);
                    if block.is_solid()
                        && !above1.is_solid()
                        && above1 != BlockType::Water
                        && !above2.is_solid()
                        && above2 != BlockType::Water
                    {
                        spawn_x = sx;
                        spawn_z = sz;
                        spawn_y = Some(y as f32 + 3.0);
                        break 'search;
                    }
                }
            }

            let sy = spawn_y.unwrap_or(35.0);
            Point3::new(spawn_x as f32 + 0.5, sy, spawn_z as f32 + 0.5)
        };

        camera.position = spawn_point;
        if let Some(ref save) = player_save {
            camera.yaw = cgmath::Rad(save.yaw);
            camera.pitch = cgmath::Rad(save.pitch);
        }
        let player = Player::new(spawn_point);
        let water_simulation = WaterSimulation::new(0.5);
        let enemy_manager = EnemyManager::new(20.0, 10); // spawn frequency (seconds), max enemies
        let bird_manager = BirdManager::new();
        let fish_manager = FishManager::new();
        let dropped_item_manager = DroppedItemManager::new();
        let particle_manager = ParticleManager::new();

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

        // Item Cube Pipeline — textured isometric block icons using the texture atlas
        let item_cube_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Item Cube Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/item_cube_shader.wgsl").into()),
        });

        let item_cube_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Item Cube Bind Group Layout"),
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

        let item_cube_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Item Cube Bind Group"),
            layout: &item_cube_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_atlas.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture_atlas.sampler),
                },
            ],
        });

        let item_cube_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Item Cube Pipeline Layout"),
                bind_group_layouts: &[&item_cube_bind_group_layout],
                push_constant_ranges: &[],
            });

        let item_cube_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Item Cube Pipeline"),
                layout: Some(&item_cube_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &item_cube_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[ItemCubeVertex::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &item_cube_shader,
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
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            });

        // Crosshair vertices (will be updated on resize for aspect ratio correction)
        // Allocate enough space for crosshair (12) + hit indicator circle (24 segments * 6 = 144)
        let crosshair_vertices = Self::build_crosshair_vertices(aspect_ratio, false);
        let max_crosshair_bytes = 156 * std::mem::size_of::<UiVertex>();

        let crosshair_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Crosshair Vertex Buffer"),
            size: max_crosshair_bytes as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&crosshair_vertex_buffer, 0, bytemuck::cast_slice(&crosshair_vertices));

        // HUD vertex buffer (hotbar + text). We'll rebuild it each frame.
        let hud_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HUD Vertex Buffer"),
            size: (std::mem::size_of::<UiVertex>() as u64) * 100_000,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // HUD text overlay buffer — count text drawn on top of item cubes
        let hud_text_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HUD Text Vertex Buffer"),
            size: (std::mem::size_of::<UiVertex>() as u64) * 10_000,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Item cube vertex buffer — 3-face isometric icons, rebuilt each frame
        let item_cube_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Item Cube Vertex Buffer"),
            size: (std::mem::size_of::<ItemCubeVertex>() as u64) * 1000,
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

        // Cloud Pipeline
        let cloud_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cloud Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cloud_shader.wgsl").into()),
        });

        let cloud_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cloud Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout, &fog_bind_group_layout],
            push_constant_ranges: &[],
        });

        let cloud_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Cloud Pipeline"),
            layout: Some(&cloud_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &cloud_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &cloud_shader,
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
                depth_write_enabled: false,  // Don't write depth for transparent clouds
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

        // Initialize cloud manager and buffers (use same render distance as world)
        let cloud_manager = crate::clouds::CloudManager::new(18, cloud_config);

        // Create empty cloud buffers (will be updated each frame)
        let cloud_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cloud Vertex Buffer"),
            size: 1024 * 1024, // 1MB initial size
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cloud_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cloud Index Buffer"),
            size: 512 * 1024, // 512KB initial size
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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

        // --- Damage flash effect pipeline ---
        let damage_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Damage Uniform Buffer"),
            contents: bytemuck::cast_slice(&[0.0f32, 0.0f32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let damage_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Damage Bind Group Layout"),
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

        let damage_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Damage Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/damage_shader.wgsl").into()),
        });

        let damage_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Damage Pipeline Layout"),
            bind_group_layouts: &[&damage_bind_group_layout],
            push_constant_ranges: &[],
        });

        let damage_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Damage Pipeline"),
            layout: Some(&damage_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &damage_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &damage_shader,
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Damage bind group: reads from post_process_texture (damage-only case)
        let damage_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Damage Bind Group"),
            layout: &damage_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: damage_uniform_buffer.as_entire_binding(),
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

        // Damage bind group alt: reads from scene_texture (underwater+damage case)
        let damage_bind_group_alt = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Damage Bind Group Alt"),
            layout: &damage_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: damage_uniform_buffer.as_entire_binding(),
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

        // ── PAUSE BLUR PIPELINE ───────────────────────────────────────────────
        // Reads scene_texture, applies a Gaussian blur + darkening,
        // then outputs to the swap chain as the pause backdrop.

        let pause_blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pause Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/pause_blur_shader.wgsl").into(),
            ),
        });

        let pause_blur_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Pause Blur BGL"),
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

        let pause_blur_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pause Blur BG"),
            layout: &pause_blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&scene_sampler),
                },
            ],
        });

        let pause_blur_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pause Blur Pipeline Layout"),
                bind_group_layouts: &[&pause_blur_bind_group_layout],
                push_constant_ranges: &[],
            });

        let pause_blur_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Pause Blur Pipeline"),
                layout: Some(&pause_blur_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &pause_blur_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &pause_blur_shader,
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
                multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
                multiview: None,
                cache: None,
            });

        // ── MODAL SAND TEXTURE ────────────────────────────────────────────────
        // Load sand.png as a standalone repeat-sampled texture.

        let sand_img = image::open("assets/textures/blocks/sand.png")
            .expect("Cannot open assets/textures/blocks/sand.png")
            .to_rgba8();
        let (sand_w, sand_h) = sand_img.dimensions();
        let sand_data = sand_img.into_raw();

        let modal_sand_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Modal Sand Texture"),
            size: wgpu::Extent3d { width: sand_w, height: sand_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &modal_sand_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &sand_data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * sand_w),
                rows_per_image: Some(sand_h),
            },
            wgpu::Extent3d { width: sand_w, height: sand_h, depth_or_array_layers: 1 },
        );
        let modal_sand_texture_view =
            modal_sand_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let modal_sand_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Modal Sand Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let modal_sand_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Modal Sand BGL"),
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

        let modal_sand_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Modal Sand BG"),
            layout: &modal_sand_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&modal_sand_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&modal_sand_sampler),
                },
            ],
        });

        let modal_sand_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Modal Sand Pipeline Layout"),
                bind_group_layouts: &[&modal_sand_bind_group_layout],
                push_constant_ranges: &[],
            });

        let modal_sand_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Modal Sand Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/modal_sand_shader.wgsl").into(),
            ),
        });

        let modal_sand_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Modal Sand Pipeline"),
                layout: Some(&modal_sand_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &modal_sand_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[ModalVertex::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &modal_sand_shader,
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
                depth_stencil: None,
                multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
                multiview: None,
                cache: None,
            });

        // ── Modal GPU buffers ─────────────────────────────────────────────────
        // Pre-allocate; rebuilt each frame while paused.
        let modal_sand_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Modal Sand Vertex Buffer"),
            size: (std::mem::size_of::<ModalVertex>() * 6) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let modal_ui_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Modal UI Vertex Buffer"),
            size: (std::mem::size_of::<UiVertex>() as u64) * 8_000,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Pause modal instance ──────────────────────────────────────────────
        let mut pause_modal = Modal::new("PAUSED", &["RESUME", "QUIT"], modal::MODAL_W_RATIO, modal::MODAL_ASPECT);
        pause_modal.update_layout(size.width as f32, size.height as f32);

        // ── Crafting table modal instance (wider + taller than pause modal) ──
        let mut crafting_modal = Modal::new("CRAFTING TABLE", &[], 0.50, 1.80);
        crafting_modal.update_layout(size.width as f32, size.height as f32);

        // ── Audio ─────────────────────────────────────────────────────────────
        // Both the music sink and walk sink must share the same OutputStream so
        // that they remain connected to the audio device for the life of State.
        let audio = AudioManager::new();
        if let Some(a) = &audio {
            a.play_looping("assets/audio/music/forest_1.mp3");
        }
        let walk_sink = audio.as_ref()
            .and_then(|a| a.create_looping_sink("assets/audio/sounds/walking_loop_sand.mp3"));

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
            fog_config,
            fog_buffer,
            fog_bind_group_layout,
            fog_bind_group,
            world,
            player,
            spawn_point,
            water_simulation,
            enemy_manager,
            bird_manager,
            fish_manager,
            dropped_item_manager,
            particle_manager,
            last_frame: Instant::now(),
            mouse_pressed: false,
            window,
            show_inventory: false,
            ui_pipeline,
            ui_bind_group,
            ui_uniform_buffer,
            crosshair_vertex_buffer,
            hud_vertex_buffer,
            hud_vertex_count: 0,
            hud_text_vertex_buffer,
            hud_text_vertex_count: 0,
            item_cube_pipeline,
            item_cube_bind_group,
            item_cube_vertex_buffer,
            item_cube_vertex_count: 0,
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
            // Debug mode (restored from save if available)
            show_chunk_outlines: player_save.as_ref().map_or(false, |s| s.show_chunk_outlines),
            noclip_mode:         player_save.as_ref().map_or(false, |s| s.noclip_mode),
            show_enemy_hitboxes: player_save.as_ref().map_or(false, |s| s.show_enemy_hitboxes),
            smooth_lighting: true,
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
            // Damage flash
            damage_flash_intensity: 0.0,
            damage_pipeline,
            damage_bind_group_layout,
            damage_bind_group,
            damage_bind_group_alt,
            damage_uniform_buffer,
            // Texture atlas
            texture_atlas,
            // Breaking mechanics
            breaking_pipeline,
            breaking_state: None,
            left_mouse_held: false,
            // Melee combat
            hit_cooldown: 0.0,
            hit_indicator_timer: 0.0,
            crosshair_vertex_count: 12,
            // Cached GPU buffers
            chunk_buffers: HashMap::new(),
            // Clouds
            cloud_manager,
            cloud_pipeline,
            cloud_vertex_buffer,
            cloud_index_buffer,
            cloud_index_count: 0,

            // Pause / Modal
            paused: false,
            pause_modal,
            crafting_modal,
            cursor_pos_px: (0.0, 0.0),
            pause_blur_pipeline,
            pause_blur_bind_group_layout,
            pause_blur_bind_group,
            modal_sand_pipeline,
            modal_sand_bind_group_layout,
            modal_sand_bind_group,
            modal_sand_texture,
            modal_sand_sampler,
            modal_sand_vertex_buffer,
            modal_ui_vertex_buffer,
            modal_ui_vertex_count: 0,

            hovered_dropped_item: None,

            // Crafting Table UI
            crafting_ui_open: false,
            hovered_crafting_table: false,
            crafting_grid: CraftingGrid::default(),
            crafting_output: None,
            crafting_held: None,
            crafting_hovered_grid: None,
            crafting_hovered_inv: None,
            crafting_hovered_output: false,
            crafting_inv_qty: [0.0; 9],

            audio,
            walk_sink,
            walk_volume: 0.0,
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

    /// Open the pause menu: freeze simulation, release mouse, show modal.
    pub fn open_pause_menu(&mut self) {
        self.paused = true;
        self.pause_modal.visible = true;
        let sw = self.size.width  as f32;
        let sh = self.size.height as f32;
        self.pause_modal.update_layout(sw, sh);
        self.release_mouse();
        if let Some(a) = &self.audio {
            a.play("assets/audio/sounds/modal_open.mp3");
        }
    }

    /// Close the pause menu: resume simulation, recapture mouse.
    pub fn close_pause_menu(&mut self) {
        self.paused = false;
        self.pause_modal.visible = false;
        self.capture_mouse();
    }

    pub fn is_paused(&self) -> bool { self.paused }

    pub fn is_crafting_open(&self) -> bool { self.crafting_ui_open }

    // ── Crafting Table UI ─────────────────────────────────────────────────

    pub fn open_crafting_ui(&mut self) {
        let sw = self.size.width as f32;
        let sh = self.size.height as f32;
        self.crafting_modal.update_layout(sw, sh);
        self.crafting_modal.title_scale = 4.0;
        self.crafting_modal.title_y_offset = (crate::modal::MODAL_BORDER_PX + crate::modal::MODAL_BEVEL_PX) + 8.0;
        self.crafting_ui_open = true;
        self.crafting_grid = CraftingGrid::default();
        self.crafting_held = None;
        self.crafting_output = None;
        // Initialise per-slot selected quantities to 1.0 (or 0 if slot empty)
        for i in 0..9 {
            self.crafting_inv_qty[i] = self.player.inventory
                .get_slot(i)
                .map(|_| 1.0)
                .unwrap_or(0.0);
        }
        self.release_mouse();
    }

    pub fn close_crafting_ui(&mut self) {
        // Return held item to inventory
        if let Some((bt, qty)) = self.crafting_held.take() {
            self.player.inventory.add_item(bt, qty);
        }
        // Return every grid item to inventory
        for row in 0..3 {
            for col in 0..3 {
                if let Some((bt, qty)) = self.crafting_grid.slots[row][col].take() {
                    self.player.inventory.add_item(bt, qty);
                }
            }
        }
        self.crafting_ui_open = false;
        self.crafting_output = None;
        self.capture_mouse();
    }

    /// Returns which crafting-UI element the pixel coordinate (px, py) is over.
    fn crafting_slot_hit(&self, px: f32, py: f32, _sw: f32, _sh: f32) -> CraftingHit {
        let lo = crafting_layout(
            self.crafting_modal.panel_x, self.crafting_modal.panel_y,
            self.crafting_modal.panel_w, self.crafting_modal.panel_h,
        );
        // 3×3 grid
        for row in 0..3usize {
            for col in 0..3usize {
                let sx = lo.row1_x + col as f32 * (lo.ct_slot + lo.ct_gap);
                let sy = lo.row1_y + row as f32 * (lo.ct_slot + lo.ct_gap);
                if px >= sx && px < sx + lo.ct_slot && py >= sy && py < sy + lo.ct_slot {
                    return CraftingHit::GridSlot(row, col);
                }
            }
        }
        // Output slot
        if px >= lo.out_x && px < lo.out_x + lo.ct_slot
            && py >= lo.out_y && py < lo.out_y + lo.ct_slot {
            return CraftingHit::OutputSlot;
        }
        // Inventory row
        for i in 0..9usize {
            let sx = lo.row2_x + i as f32 * (lo.ct_slot + lo.ct_gap);
            let sy = lo.row2_y;
            if px >= sx && px < sx + lo.ct_slot && py >= sy && py < sy + lo.ct_slot {
                return CraftingHit::InvSlot(i);
            }
        }
        CraftingHit::None
    }

    /// Consume crafting grid inputs for one craft.
    /// For Recipe 1 (Wood → Planks): clears the single occupied grid slot.
    /// For Recipe 2 (2×2 Planks → CraftingTable): clears the four 2×2 slots.
    fn consume_crafting_inputs(&mut self) {
        // Simply clear all non-empty grid slots (recipes use exact slot contents)
        for row in 0..3 {
            for col in 0..3 {
                self.crafting_grid.slots[row][col] = None;
            }
        }
    }

    /// Call with the current cursor pixel position.
    /// Returns the label of a button that was just clicked, or None.
    pub fn handle_modal_cursor_moved(&mut self, px: f32, py: f32) {
        self.cursor_pos_px = (px, py);
        if self.pause_modal.visible {
            self.pause_modal.update_hover(px, py);
        }
    }

    /// Returns the button label that was clicked (if any).
    pub fn handle_modal_click(&mut self) -> Option<&'static str> {
        if self.pause_modal.visible {
            let (px, py) = self.cursor_pos_px;
            let hit = self.pause_modal.hit_button(px, py);
            if hit.is_some() {
                if let Some(a) = &self.audio {
                    a.play("assets/audio/sounds/button_click.mp3");
                }
            }
            hit
        } else {
            None
        }
    }

    /// Saves all modified chunks and enemies to disk (call before exiting)
    pub fn save_world(&mut self) {
        self.world.save_all_modified_chunks();
        self.enemy_manager.save_to_disk();
        crate::config::PlayerSave {
            x: self.camera.position.x,
            y: self.camera.position.y,
            z: self.camera.position.z,
            yaw: self.camera.yaw.0,
            pitch: self.camera.pitch.0,
            show_chunk_outlines: self.show_chunk_outlines,
            noclip_mode: self.noclip_mode,
            show_enemy_hitboxes: self.show_enemy_hitboxes,
        }
        .save();
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

            // Recreate Pause Blur Bind Group (reads from scene_texture)
            self.pause_blur_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Pause Blur BG"),
                layout: &self.pause_blur_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.scene_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.scene_sampler),
                    },
                ],
            });

            // Also recompute modal layout for new screen size
            self.pause_modal.update_layout(new_size.width as f32, new_size.height as f32);

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

            // Recreate Damage Bind Groups
            self.damage_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Damage Bind Group"),
                layout: &self.damage_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.damage_uniform_buffer.as_entire_binding(),
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
            self.damage_bind_group_alt = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Damage Bind Group Alt"),
                layout: &self.damage_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.damage_uniform_buffer.as_entire_binding(),
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
            let hit_active = self.hit_indicator_timer > 0.0;
            let crosshair_vertices = Self::build_crosshair_vertices(aspect_ratio, hit_active);
            self.crosshair_vertex_count = crosshair_vertices.len() as u32;
            self.queue.write_buffer(
                &self.crosshair_vertex_buffer,
                0,
                bytemuck::cast_slice(&crosshair_vertices),
            );
        }
    }

    fn build_crosshair_vertices(aspect_ratio: f32, hit_active: bool) -> Vec<UiVertex> {
        let crosshair_size = 0.03;
        let crosshair_thickness = 0.01;
        let crosshair_color = [1.0, 1.0, 1.0, 0.5];

        // Correct X coordinates for aspect ratio to maintain 1:1 ratio
        let h_size = crosshair_size / aspect_ratio;
        let h_thick = crosshair_thickness / aspect_ratio;

        let mut verts = vec![
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
        ];

        // Red hit indicator circle
        if hit_active {
            let hit_color = [1.0, 0.2, 0.2, 0.9];
            let radius = 0.10;
            let thickness = 0.008;
            let segments = 24;

            for i in 0..segments {
                let angle0 = (i as f32 / segments as f32) * std::f32::consts::TAU;
                let angle1 = ((i + 1) as f32 / segments as f32) * std::f32::consts::TAU;

                let inner_r = radius - thickness * 0.5;
                let outer_r = radius + thickness * 0.5;

                // Inner and outer points for this segment (aspect-corrected X)
                let ix0 = inner_r * angle0.cos() / aspect_ratio;
                let iy0 = inner_r * angle0.sin();
                let ox0 = outer_r * angle0.cos() / aspect_ratio;
                let oy0 = outer_r * angle0.sin();
                let ix1 = inner_r * angle1.cos() / aspect_ratio;
                let iy1 = inner_r * angle1.sin();
                let ox1 = outer_r * angle1.cos() / aspect_ratio;
                let oy1 = outer_r * angle1.sin();

                // Two triangles forming the quad for this segment
                verts.push(UiVertex { position: [ix0, iy0], color: hit_color });
                verts.push(UiVertex { position: [ox0, oy0], color: hit_color });
                verts.push(UiVertex { position: [ox1, oy1], color: hit_color });
                verts.push(UiVertex { position: [ix0, iy0], color: hit_color });
                verts.push(UiVertex { position: [ox1, oy1], color: hit_color });
                verts.push(UiVertex { position: [ix1, iy1], color: hit_color });
            }
        }

        verts
    }

    /// Project a world-space point to screen pixel coordinates.
    /// Returns None if the point is behind the camera.
    fn world_to_screen(&self, world_pos: Point3<f32>) -> Option<(f32, f32)> {
        let view = self.camera.get_view_matrix();
        let proj = self.projection.calc_matrix();
        let vp = proj * view;

        // Transform to clip space
        let p = vp * cgmath::Vector4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);
        if p.w <= 0.0 {
            return None; // Behind camera
        }

        // Perspective divide to NDC
        let ndc_x = p.x / p.w;
        let ndc_y = p.y / p.w;

        // NDC to pixel coordinates
        let screen_w = self.size.width as f32;
        let screen_h = self.size.height as f32;
        let px = (ndc_x + 1.0) * 0.5 * screen_w;
        let py = (1.0 - ndc_y) * 0.5 * screen_h;

        Some((px, py))
    }

    fn rebuild_hud_vertices(&mut self) {
        let screen_w = self.size.width as f32;
        let screen_h = self.size.height as f32;

        let slots = self.player.inventory.size.max(1);
        let selected = self.player.inventory.selected_slot.min(slots - 1);

        // Layout in pixels
        let slot_size = 80.0;
        let slot_gap = 8.0;
        let margin_bottom = 32.0;
        let total_w = (slots as f32) * slot_size + (slots as f32 - 1.0) * slot_gap;
        let start_x = (screen_w - total_w) * 0.5;
        let start_y = screen_h - margin_bottom - slot_size;

        let mut verts: Vec<UiVertex> = Vec::with_capacity(4096);
        // Count text is rendered after item cubes so it appears on top
        let mut text_verts: Vec<UiVertex> = Vec::with_capacity(512);

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

        // === Enemy Hitbox Toggle Indicator (below Noclip) ===
        let (hitbox_text, hitbox_color, hitbox_bg_color) = if self.show_enemy_hitboxes {
            ("F3 - ENEMY HITBOX: ON", [0.5, 1.0, 0.5, 1.0], [0.0, 0.2, 0.0, 0.6])
        } else {
            ("F3 - ENEMY HITBOX: OFF", [1.0, 1.0, 1.0, 0.9], [0.0, 0.0, 0.0, 0.5])
        };
        let hitbox_y = noclip_y + fps_char_h + 8.0;
        let hitbox_text_width = hitbox_text.len() as f32 * fps_char_w;
        bitmap_font::push_rect_px(
            &mut verts,
            fps_x - 4.0,
            hitbox_y - 4.0,
            hitbox_text_width + 8.0,
            fps_char_h + 8.0,
            hitbox_bg_color,
            screen_w,
            screen_h,
        );
        bitmap_font::draw_text_quads(
            &mut verts,
            hitbox_text,
            fps_x,
            hitbox_y,
            fps_scale,
            fps_scale,
            hitbox_color,
            screen_w,
            screen_h,
        );

        // === Smooth Lighting Toggle Indicator (below Enemy Hitbox) ===
        let (smooth_text, smooth_color, smooth_bg_color) = if self.smooth_lighting {
            ("F4 - SMOOTH LIGHTING: ON", [0.5, 1.0, 0.5, 1.0], [0.0, 0.2, 0.0, 0.6])
        } else {
            ("F4 - SMOOTH LIGHTING: OFF", [1.0, 1.0, 1.0, 0.9], [0.0, 0.0, 0.0, 0.5])
        };
        let smooth_y = hitbox_y + fps_char_h + 8.0;
        let smooth_text_width = smooth_text.len() as f32 * fps_char_w;
        bitmap_font::push_rect_px(
            &mut verts,
            fps_x - 4.0,
            smooth_y - 4.0,
            smooth_text_width + 8.0,
            fps_char_h + 8.0,
            smooth_bg_color,
            screen_w,
            screen_h,
        );
        bitmap_font::draw_text_quads(
            &mut verts,
            smooth_text,
            fps_x,
            smooth_y,
            fps_scale,
            fps_scale,
            smooth_color,
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

        //let fill = [1.0, 1.0, 1.0, 0.10]; // white
        let fill = [0.0, 0.0, 0.0, 0.3]; // dark
        let outline = [1.0, 1.0, 1.0, 0.85];
        //let outline_selected = [1.0, 1.0, 1.0, 1.0];  // white
        //let outline_selected = [0.14, 0.64, 0.98, 1.0]; // blue
        let outline_selected = [1.0, 0.6, 0.0, 1.0]; // yellow-orange

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
            let thick = if i == selected { 6.0 } else { 3.0 };
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

                // Block name
                if i == selected {
                    let name = stack.block_type.display_name();
                    let name_color = [1.0, 1.0, 1.0, 0.95];
                    let name_w = (name.chars().count() as f32) * name_char_w;
                    let name_x = x + (slot_size - name_w) * 0.5;
                    let name_y = (y - name_char_h - label_gap).max(0.0);

                    // Background
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

                // Block count
                if stack.count > 0.0 {
                    let count_text = format!("{:.2}", stack.count);
                    let count_color = [1.0, 1.0, 1.0, 0.95];
                    let count_scale = 2.0;
                    let count_w = (count_text.chars().count() as f32) * 6.0 * count_scale;
                    let count_x = x + (slot_size - count_w) * 0.5;
                    let count_y = y + slot_size + (3.0 * count_scale);
                    let count_background_padding = 4.0;

                    // Background
                    bitmap_font::push_rect_px(
                        &mut text_verts,
                        count_x - count_background_padding,
                        count_y - count_background_padding,
                        (count_text.len() as f32) * 6.0 * count_scale + (count_background_padding * 2.0),
                        7.0 * count_scale + (count_background_padding * 2.0),
                        [0.0, 0.0, 0.0, 0.55],
                        screen_w,
                        screen_h,
                    );

                    bitmap_font::draw_text_quads(
                        &mut text_verts,
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

        // === Enemy Floating Health Bars ===
        for enemy in &self.enemy_manager.enemies {
            if !enemy.alive { continue; }
            // Only show health bar if enemy has taken damage
            if enemy.health >= enemy.max_health && enemy.death_timer < 0.0 { continue; }

            // Project position above enemy head
            let bar_world_pos = Point3::new(
                enemy.position.x,
                enemy.position.y + enemy.height + 0.6,
                enemy.position.z,
            );
            if let Some((sx, sy)) = self.world_to_screen(bar_world_pos) {
                // Skip if off-screen
                if sx < -100.0 || sx > screen_w + 100.0 || sy < -50.0 || sy > screen_h + 50.0 {
                    continue;
                }

                // Scale bar size based on distance
                let dist = {
                    let dx = enemy.position.x - self.camera.position.x;
                    let dy = enemy.position.y - self.camera.position.y;
                    let dz = enemy.position.z - self.camera.position.z;
                    (dx * dx + dy * dy + dz * dz).sqrt()
                };
                let scale = (24.0 / dist.max(2.0)).clamp(0.3, 4.5);

                let bar_w = 60.0 * scale;
                let bar_h = 12.0 * scale;
                let border = 1.0 * scale;
                let bar_x = sx - bar_w * 0.5;
                let bar_y = sy - bar_h * 0.5;

                // Dark background
                bitmap_font::push_rect_px(
                    &mut verts, bar_x, bar_y, bar_w, bar_h,
                    [0.0, 0.0, 0.0, 0.5], screen_w, screen_h,
                );

                // White border
                let bc = [1.0, 1.0, 1.0, 0.7];
                bitmap_font::push_rect_px(&mut verts, bar_x, bar_y, bar_w, border, bc, screen_w, screen_h);
                bitmap_font::push_rect_px(&mut verts, bar_x, bar_y + bar_h - border, bar_w, border, bc, screen_w, screen_h);
                bitmap_font::push_rect_px(&mut verts, bar_x, bar_y, border, bar_h, bc, screen_w, screen_h);
                bitmap_font::push_rect_px(&mut verts, bar_x + bar_w - border, bar_y, border, bar_h, bc, screen_w, screen_h);

                // Red health fill
                let health_pct = (enemy.health / enemy.max_health).clamp(0.0, 1.0);
                let fill_x = bar_x + border;
                let fill_y = bar_y + border;
                let fill_max_w = bar_w - border * 2.0;
                let fill_h = bar_h - border * 2.0;
                let fill_w = fill_max_w * health_pct;
                if fill_w > 0.0 {
                    bitmap_font::push_rect_px(
                        &mut verts, fill_x, fill_y, fill_w, fill_h,
                        [0.8, 0.1, 0.1, 0.9], screen_w, screen_h,
                    );
                }

                // Health percentage text (only if bar is large enough)
                if scale > 0.5 {
                    let text_scale = 1.0 * scale;
                    let pct_text = format!("{}%", (health_pct * 100.0).round() as u32);
                    let char_w = 6.0 * text_scale;
                    let char_h = 7.0 * text_scale;
                    let text_w = pct_text.len() as f32 * char_w;
                    let text_x = sx - text_w * 0.5;
                    let text_y = bar_y + (bar_h - char_h) * 0.5;
                    bitmap_font::draw_text_quads(
                        &mut verts, &pct_text, text_x, text_y,
                        text_scale, text_scale,
                        [1.0, 1.0, 1.0, 0.95], screen_w, screen_h,
                    );
                }
            }
        }

        // ── Dropped-item pickup hint ──────────────────────────────────────────
        if let Some(idx) = self.hovered_dropped_item {
            if let Some(item) = self.dropped_item_manager.items.get(idx) {
                let name = item.block_type.name();
                let label = format!("Pickup {} (Right-click)", name);
                let scale = 2.0f32;
                let char_w = 6.0 * scale; // 5px glyph + 1px spacing
                let char_h = 7.0 * scale;
                let text_w = label.len() as f32 * char_w;
                let pad = 10.0f32;

                // Position: 24 px to the right of screen centre, vertically centred
                let tx = screen_w * 0.5 + 48.0;
                let ty = screen_h * 0.5 - char_h * 0.5;

                // Dark background
                modal::push_rect_px(
                    &mut verts,
                    tx - pad, ty - pad,
                    tx + text_w + pad, ty + char_h + pad,
                    [0.0, 0.0, 0.0, 0.4],
                    screen_w, screen_h,
                );
                // White text
                bitmap_font::draw_text_quads(
                    &mut verts, &label,
                    tx, ty, scale, scale,
                    [1.0, 1.0, 1.0, 1.0],
                    screen_w, screen_h,
                );
            }
        }

        // ── Crafting Table hover hint ─────────────────────────────────────────
        if self.hovered_crafting_table && !self.crafting_ui_open && self.hovered_dropped_item.is_none() {
            let label = "Open Crafting Table (Right-click)";
            let scale = 2.0f32;
            let char_w = 6.0 * scale;
            let char_h = 7.0 * scale;
            let text_w = label.len() as f32 * char_w;
            let pad = 10.0f32;
            let tx = screen_w * 0.5 + 48.0;
            let ty = screen_h * 0.5 - char_h * 0.5;
            modal::push_rect_px(&mut verts, tx - pad, ty - pad, tx + text_w + pad, ty + char_h + pad,
                [0.0, 0.0, 0.0, 0.4], screen_w, screen_h);
            bitmap_font::draw_text_quads(&mut verts, label, tx, ty, scale, scale,
                [1.0, 1.0, 1.0, 1.0], screen_w, screen_h);
        }

        // ── Crafting Table UI panel ────────────────────────────────────────
        if self.crafting_ui_open {
            let lo = crafting_layout(
                self.crafting_modal.panel_x, self.crafting_modal.panel_y,
                self.crafting_modal.panel_w, self.crafting_modal.panel_h,
            );
            let ct_slot = lo.ct_slot;
            let ct_gap  = lo.ct_gap;

            // ── Modal chrome (border, bevel, title) from the reusable Modal ──
            let chrome = self.crafting_modal.build_ui_vertices(screen_w, screen_h);
            verts.extend_from_slice(&chrome);

            // ── 3×3 Crafting Grid ──
            let qty_scale = 1.5f32;
            let qty_cw    = 6.0 * qty_scale;
            let qty_ch    = 7.0 * qty_scale;
            for row in 0..3usize {
                for col in 0..3usize {
                    let sx = lo.row1_x + col as f32 * (ct_slot + ct_gap);
                    let sy = lo.row1_y + row as f32 * (ct_slot + ct_gap);
                    modal::push_rect_px(&mut verts, sx, sy, sx + ct_slot, sy + ct_slot,
                        [0.08, 0.07, 0.04, 1.0], screen_w, screen_h);
                    let hovered = self.crafting_hovered_grid == Some((row, col));
                    let border_col = if hovered { [1.0, 0.8, 0.1, 1.0] } else { [0.6, 0.55, 0.38, 0.9] };
                    let bw = if hovered { 3.0f32 } else { 2.0f32 };
                    modal::push_rect_px(&mut verts, sx - bw, sy - bw, sx + ct_slot + bw, sy + bw, border_col, screen_w, screen_h);
                    modal::push_rect_px(&mut verts, sx - bw, sy + ct_slot, sx + ct_slot + bw, sy + ct_slot + bw, border_col, screen_w, screen_h);
                    modal::push_rect_px(&mut verts, sx - bw, sy, sx, sy + ct_slot, border_col, screen_w, screen_h);
                    modal::push_rect_px(&mut verts, sx + ct_slot, sy, sx + ct_slot + bw, sy + ct_slot, border_col, screen_w, screen_h);
                    if let Some((_, qty)) = self.crafting_grid.slots[row][col] {
                        let qty_text = format!("{:.2}", qty);
                        let qty_x = sx + ct_slot - qty_text.len() as f32 * qty_cw - 2.0;
                        let qty_y = sy + ct_slot - qty_ch - 2.0;
                        bitmap_font::draw_text_quads(&mut verts, &qty_text, qty_x, qty_y,
                            qty_scale, qty_scale, [1.0, 1.0, 0.6, 1.0], screen_w, screen_h);
                    }
                }
            }

            // ── Arrow ──
            bitmap_font::draw_text_quads(&mut verts, "→", lo.arrow_x, lo.arrow_y,
                lo.arrow_scale, lo.arrow_scale, [0.0, 0.0, 0.0, 0.55], screen_w, screen_h);

            // ── Output slot ──
            let out_has_item = self.crafting_output.is_some();
            let out_bg = if out_has_item { [0.1, 0.14, 0.06, 1.0] } else { [0.06, 0.05, 0.03, 1.0] };
            modal::push_rect_px(&mut verts, lo.out_x, lo.out_y, lo.out_x + ct_slot, lo.out_y + ct_slot, out_bg, screen_w, screen_h);
            let out_border = if self.crafting_hovered_output && out_has_item {
                [0.5, 1.0, 0.3, 1.0]
            } else if out_has_item {
                [0.4, 0.8, 0.2, 0.9]
            } else {
                [0.4, 0.38, 0.25, 0.8]
            };
            let bw = 2.0f32;
            modal::push_rect_px(&mut verts, lo.out_x - bw, lo.out_y - bw, lo.out_x + ct_slot + bw, lo.out_y + bw, out_border, screen_w, screen_h);
            modal::push_rect_px(&mut verts, lo.out_x - bw, lo.out_y + ct_slot, lo.out_x + ct_slot + bw, lo.out_y + ct_slot + bw, out_border, screen_w, screen_h);
            modal::push_rect_px(&mut verts, lo.out_x - bw, lo.out_y, lo.out_x, lo.out_y + ct_slot, out_border, screen_w, screen_h);
            modal::push_rect_px(&mut verts, lo.out_x + ct_slot, lo.out_y, lo.out_x + ct_slot + bw, lo.out_y + ct_slot, out_border, screen_w, screen_h);
            if let Some((_, qty)) = self.crafting_output {
                let qty_text = format!("{:.2}", qty);
                let qty_x = lo.out_x + ct_slot - qty_text.len() as f32 * qty_cw - 2.0;
                let qty_y = lo.out_y + ct_slot - qty_ch - 2.0;
                bitmap_font::draw_text_quads(&mut verts, &qty_text, qty_x, qty_y,
                    qty_scale, qty_scale, [0.6, 1.0, 0.4, 1.0], screen_w, screen_h);
            }

            // ── Inventory row (Row 2) ──
            for i in 0..9usize {
                let sx = lo.row2_x + i as f32 * (ct_slot + ct_gap);
                let sy = lo.row2_y;
                let inv_slot = self.player.inventory.get_slot(i);
                let has_item = inv_slot.is_some();
                modal::push_rect_px(&mut verts, sx, sy, sx + ct_slot, sy + ct_slot,
                    [0.08, 0.07, 0.04, 1.0], screen_w, screen_h);
                let hovered = self.crafting_hovered_inv == Some(i);
                let border_col = if hovered && has_item { [1.0, 0.8, 0.1, 1.0] } else { [0.6, 0.55, 0.38, 0.9] };
                let bw = if hovered && has_item { 3.0f32 } else { 2.0f32 };
                modal::push_rect_px(&mut verts, sx - bw, sy - bw, sx + ct_slot + bw, sy + bw, border_col, screen_w, screen_h);
                modal::push_rect_px(&mut verts, sx - bw, sy + ct_slot, sx + ct_slot + bw, sy + ct_slot + bw, border_col, screen_w, screen_h);
                modal::push_rect_px(&mut verts, sx - bw, sy, sx, sy + ct_slot, border_col, screen_w, screen_h);
                modal::push_rect_px(&mut verts, sx + ct_slot, sy, sx + ct_slot + bw, sy + ct_slot, border_col, screen_w, screen_h);
                if let Some(slot) = inv_slot {
                    let qty_text = format!("{:.2}", slot.count);
                    let qty_x = sx + ct_slot - qty_text.len() as f32 * qty_cw - 2.0;
                    let qty_y = sy + ct_slot - qty_ch - 2.0;
                    bitmap_font::draw_text_quads(&mut verts, &qty_text, qty_x, qty_y,
                        qty_scale, qty_scale, [1.0, 1.0, 0.6, 1.0], screen_w, screen_h);
                }
            }

            // ── Inventory slot tooltip (name + qty near cursor) ──
            // Pushed into text_verts so it renders AFTER item cubes and stays on top.
            if let Some(i) = self.crafting_hovered_inv {
                if let Some(stack) = self.player.inventory.get_slot(i) {
                    let qty = self.crafting_inv_qty[i];
                    let tooltip = format!("{} ({:.2} / {:.2})", stack.block_type.name(), qty, stack.count);
                    let tt_scale = 2.0f32;
                    let tt_cw = 6.0 * tt_scale;
                    let tt_ch = 7.0 * tt_scale;
                    let tt_w = tooltip.len() as f32 * tt_cw;
                    let pad = 6.0f32;
                    let (cx, cy) = self.cursor_pos_px;
                    let tt_x = cx + 16.0;
                    let tt_y = (cy - tt_ch - 4.0).max(0.0);
                    modal::push_rect_px(&mut text_verts, tt_x - pad, tt_y - pad, tt_x + tt_w + pad, tt_y + tt_ch + pad,
                        [0.0, 0.0, 0.0, 0.7], screen_w, screen_h);
                    bitmap_font::draw_text_quads(&mut text_verts, &tooltip, tt_x, tt_y, tt_scale, tt_scale,
                        [1.0, 1.0, 1.0, 1.0], screen_w, screen_h);
                }
            }

            // ── Held item: ghost slot at cursor ──
            if let Some((_, qty)) = self.crafting_held {
                let (cx, cy) = self.cursor_pos_px;
                let hx = cx - ct_slot * 0.5;
                let hy = cy - ct_slot * 0.5;
                modal::push_rect_px(&mut verts, hx, hy, hx + ct_slot, hy + ct_slot,
                    [0.2, 0.18, 0.10, 0.7], screen_w, screen_h);
                let qty_text = format!("{:.2}", qty);
                let qty_x = hx + ct_slot - qty_text.len() as f32 * qty_cw - 2.0;
                let qty_y = hy + ct_slot - qty_ch - 2.0;
                bitmap_font::draw_text_quads(&mut verts, &qty_text, qty_x, qty_y,
                    qty_scale, qty_scale, [1.0, 1.0, 0.6, 1.0], screen_w, screen_h);
            }

        }

        self.hud_vertex_count = verts.len() as u32;
        self.queue
            .write_buffer(&self.hud_vertex_buffer, 0, bytemuck::cast_slice(&verts));

        self.hud_text_vertex_count = text_verts.len() as u32;
        self.queue
            .write_buffer(&self.hud_text_vertex_buffer, 0, bytemuck::cast_slice(&text_verts));
    }

    fn rebuild_item_cube_vertices(&mut self) {
        let screen_w = self.size.width as f32;
        let screen_h = self.size.height as f32;

        let slots = self.player.inventory.size.max(1);
        let slot_size = 80.0_f32;
        let slot_gap = 8.0_f32;
        let margin_bottom = 32.0_f32;
        let total_w = (slots as f32) * slot_size + (slots as f32 - 1.0) * slot_gap;
        let start_x = (screen_w - total_w) * 0.5;
        let start_y = screen_h - margin_bottom - slot_size;

        let mut verts: Vec<ItemCubeVertex> = Vec::with_capacity(200);

        for i in 0..slots {
            if let Some(stack) = self.player.inventory.get_slot(i) {
                let x = start_x + i as f32 * (slot_size + slot_gap);
                let cx = x + slot_size * 0.5;
                let cy = start_y + slot_size * 0.5;
                let cube_size = slot_size * 0.65; // cube size relative to slot
                push_item_cube(&mut verts, stack.block_type, cx, cy, cube_size, screen_w, screen_h);
            }
        }

        // ── Crafting UI item cubes ────────────────────────────────────────
        if self.crafting_ui_open {
            let lo = crafting_layout(
                self.crafting_modal.panel_x, self.crafting_modal.panel_y,
                self.crafting_modal.panel_w, self.crafting_modal.panel_h,
            );
            let cube_size = lo.ct_slot * 0.65;

            // 3×3 grid slots
            for row in 0..3usize {
                for col in 0..3usize {
                    if let Some((bt, _)) = self.crafting_grid.slots[row][col] {
                        let cx = lo.row1_x + col as f32 * (lo.ct_slot + lo.ct_gap) + lo.ct_slot * 0.5;
                        let cy = lo.row1_y + row as f32 * (lo.ct_slot + lo.ct_gap) + lo.ct_slot * 0.5;
                        push_item_cube(&mut verts, bt, cx, cy, cube_size, screen_w, screen_h);
                    }
                }
            }

            // Output slot
            if let Some((bt, _)) = self.crafting_output {
                push_item_cube(&mut verts, bt,
                    lo.out_x + lo.ct_slot * 0.5, lo.out_y + lo.ct_slot * 0.5,
                    cube_size, screen_w, screen_h);
            }

            // Inventory row (slots 0-8)
            for i in 0..9usize {
                if let Some(stack) = self.player.inventory.get_slot(i) {
                    let cx = lo.row2_x + i as f32 * (lo.ct_slot + lo.ct_gap) + lo.ct_slot * 0.5;
                    let cy = lo.row2_y + lo.ct_slot * 0.5;
                    push_item_cube(&mut verts, stack.block_type, cx, cy, cube_size, screen_w, screen_h);
                }
            }

            // Held item (follows cursor)
            if let Some((bt, _)) = self.crafting_held {
                let (cx, cy) = self.cursor_pos_px;
                push_item_cube(&mut verts, bt, cx, cy, cube_size, screen_w, screen_h);
            }
        }

        self.item_cube_vertex_count = verts.len() as u32;
        self.queue
            .write_buffer(&self.item_cube_vertex_buffer, 0, bytemuck::cast_slice(&verts));
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
                    KeyCode::Digit1
                    | KeyCode::Digit2
                    | KeyCode::Digit3
                    | KeyCode::Digit4
                    | KeyCode::Digit5
                    | KeyCode::Digit6
                    | KeyCode::Digit7
                    | KeyCode::Digit8
                    | KeyCode::Digit9 => {
                        if is_pressed && !self.crafting_ui_open {
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
                            self.player.inventory.selected_slot = num;
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
                    KeyCode::F3 => {
                        if is_pressed {
                            self.show_enemy_hitboxes = !self.show_enemy_hitboxes;
                            println!("Enemy hitboxes: {}", if self.show_enemy_hitboxes { "ON" } else { "OFF" });
                        }
                        true
                    }
                    KeyCode::F4 => {
                        if is_pressed {
                            self.smooth_lighting = !self.smooth_lighting;
                            println!("Smooth lighting: {}", if self.smooth_lighting { "ON" } else { "OFF" });
                            // Mark all loaded chunks dirty so they get re-meshed with the new setting
                            for chunk in self.world.chunks.values_mut() {
                                chunk.dirty = true;
                            }
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
                if self.crafting_ui_open && is_pressed {
                    let (px, py) = self.cursor_pos_px;
                    let sw = self.size.width as f32;
                    let sh = self.size.height as f32;
                    let hit = self.crafting_slot_hit(px, py, sw, sh);
                    match hit {
                        CraftingHit::GridSlot(row, col) => {
                            if let Some(held) = self.crafting_held.take() {
                                let existing = self.crafting_grid.slots[row][col].take();
                                self.crafting_grid.slots[row][col] = Some(held);
                                if let Some(old) = existing {
                                    self.crafting_held = Some(old);
                                }
                            } else if let Some(item) = self.crafting_grid.slots[row][col].take() {
                                self.crafting_held = Some(item);
                            }
                            self.crafting_output = match_recipe(&self.crafting_grid);
                        }
                        CraftingHit::OutputSlot => {
                            if self.crafting_held.is_none() {
                                if let Some((bt, qty)) = self.crafting_output.take() {
                                    self.consume_crafting_inputs();
                                    self.player.inventory.add_item(bt, qty);
                                    self.crafting_output = match_recipe(&self.crafting_grid);
                                }
                            }
                        }
                        CraftingHit::InvSlot(i) => {
                            if let Some(held) = self.crafting_held.take() {
                                self.player.inventory.add_item(held.0, held.1);
                                self.crafting_inv_qty[i] = if self.player.inventory.get_slot(i).is_some() { 1.0 } else { 0.0 };
                            } else {
                                let pick = self.player.inventory.get_slot(i).map(|s| {
                                    (s.block_type, self.crafting_inv_qty[i].min(s.count))
                                });
                                if let Some((bt, qty)) = pick {
                                    if qty > 0.0 {
                                        self.player.inventory.remove_item(i, qty);
                                        self.crafting_held = Some((bt, qty));
                                        self.crafting_inv_qty[i] = if self.player.inventory.get_slot(i).is_some() { 1.0 } else { 0.0 };
                                    }
                                }
                            }
                        }
                        CraftingHit::None => {
                            if let Some((bt, qty)) = self.crafting_held.take() {
                                self.player.inventory.add_item(bt, qty);
                            }
                        }
                    }
                } else {
                    self.mouse_pressed = is_pressed;
                    self.left_mouse_held = is_pressed;
                    if !is_pressed {
                        self.breaking_state = None;
                    }
                }
                true
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Right,
                ..
            } => {
                if self.mouse_captured {
                    // Priority: pick up a hovered dropped item before placing a block
                    if let Some(idx) = self.hovered_dropped_item {
                        if idx < self.dropped_item_manager.items.len() {
                            let collected = self.dropped_item_manager.collect_item(idx);
                            self.player.inventory.add_item(collected.block_type, collected.value);
                            self.hovered_dropped_item = None;
                        }
                    } else if self.hovered_crafting_table {
                        self.open_crafting_ui();
                    } else {
                        self.handle_block_place();
                    }
                }
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_f = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => (pos.y / 40.0) as f32,
                };
                if self.crafting_ui_open {
                    // Adjust selected quantity for hovered inventory slot
                    if let Some(i) = self.crafting_hovered_inv {
                        let max_qty = self.player.inventory
                            .get_slot(i).map(|s| s.count).unwrap_or(0.0);
                        if max_qty > 0.0 {
                            let step = 0.25f32;
                            let new_qty = (self.crafting_inv_qty[i] + scroll_f * step)
                                .clamp(step, max_qty.max(step));
                            // Round to nearest 0.25
                            self.crafting_inv_qty[i] = (new_qty / step).round() * step;
                        }
                    }
                } else if self.mouse_captured {
                    let scroll_amount = scroll_f as i32;
                    if scroll_amount != 0 {
                        let slots = self.player.inventory.size;
                        let current = self.player.inventory.selected_slot as i32;
                        let new_slot = (current - scroll_amount).rem_euclid(slots as i32) as usize;
                        self.player.inventory.selected_slot = new_slot;
                    }
                }
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                let px = position.x as f32;
                let py = position.y as f32;
                self.handle_modal_cursor_moved(px, py);
                if self.crafting_ui_open {
                    let sw = self.size.width as f32;
                    let sh = self.size.height as f32;
                    self.crafting_hovered_grid = None;
                    self.crafting_hovered_inv = None;
                    self.crafting_hovered_output = false;
                    match self.crafting_slot_hit(px, py, sw, sh) {
                        CraftingHit::GridSlot(r, c) => self.crafting_hovered_grid = Some((r, c)),
                        CraftingHit::InvSlot(i) => {
                            self.crafting_hovered_inv = Some(i);
                            // Initialise qty on first hover to 1.0
                            if self.crafting_inv_qty[i] == 0.0 {
                                if self.player.inventory.get_slot(i).is_some() {
                                    self.crafting_inv_qty[i] = 1.0;
                                }
                            }
                        }
                        CraftingHit::OutputSlot => self.crafting_hovered_output = true,
                        CraftingHit::None => {}
                    }
                }
                false // don't consume — other handlers may need it
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

                let face_verts = create_face_vertices(offset_pos, block_type, face_idx, [1.0; 4], tex_index, uvs, [1.0; 4]);

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

        let direction = self.camera.get_direction();

        // Check if cursor is aimed at an enemy (prioritize over block breaking)
        let enemy_hit_idx = self.enemy_manager.raycast_enemy(
            self.camera.position, direction, self.player.reach_distance,
        );

        if let Some(idx) = enemy_hit_idx {
            // Cursor is on an enemy - cancel any block breaking
            self.breaking_state = None;

            // Attempt melee hit if cooldown is ready
            if self.hit_cooldown <= 0.0 {
                // Calculate damage: 1.0 base + block durability if holding an item
                let damage = match self.player.inventory.get_selected_item() {
                    Some(item) => 1.0 + item.block_type.get_durability(),
                    None => 1.0,
                };

                // Calculate knockback direction (horizontal, from player toward enemy)
                let enemy_pos = self.enemy_manager.enemies[idx].position;
                let to_enemy = Vector3::new(
                    enemy_pos.x - self.camera.position.x,
                    0.0,
                    enemy_pos.z - self.camera.position.z,
                );
                let knockback_dir = if to_enemy.magnitude() > 0.001 {
                    to_enemy.normalize()
                } else {
                    Vector3::new(direction.x, 0.0, direction.z).normalize()
                };

                self.enemy_manager.enemies[idx].hit(damage, knockback_dir);
                self.hit_cooldown = 0.4;
                self.hit_indicator_timer = 0.2;
            }
            return;
        }

        // No enemy targeted - check for block breaking
        let target = self.player.raycast_block(direction, &self.world);

        // If no block targeted either, attempt a melee swing into the air
        if target.is_none() && self.hit_cooldown <= 0.0 {
            // Show the hit indicator even when swinging at nothing
            // (no damage applied, just visual feedback of the swing)
        }

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
        // Cap dt to 50 ms (20 fps minimum) so a long first frame or stutter
        // doesn't let gravity teleport the player through the terrain.
        let dt = (now - self.last_frame).as_secs_f32().min(0.05);
        self.last_frame = now;

        // ── Walking sound fade ────────────────────────────────────────────────
        {
            const START_WALKING_SOUND_FADE_SPEED: f32 = 4.0;
            const STOP_WALKING_SOUND_FADE_SPEED: f32 = 15.0; // 20.0 is almost instantaneous

            let horiz_speed = {
                let vx = self.camera_controller.velocity.x;
                let vz = self.camera_controller.velocity.z;
                (vx * vx + vz * vz).sqrt()
            };
            let is_walking = !self.paused
                && self.camera_controller.on_ground
                && horiz_speed > 0.5;
            let target = if is_walking { 1.0f32 } else { 0.0 };
            let fade_speed = if is_walking { START_WALKING_SOUND_FADE_SPEED } else { STOP_WALKING_SOUND_FADE_SPEED };
            self.walk_volume += (target - self.walk_volume) * (fade_speed * dt).min(1.0);
            self.walk_volume = self.walk_volume.clamp(0.0, 1.0);
            if let Some(sink) = &self.walk_sink {
                sink.set_volume(self.walk_volume);
            }
        }

        // Update Underwater shader time
        let total_time = (now - self.start_time).as_secs_f32();
        self.queue.write_buffer(
            &self.underwater_uniform_buffer,
            0,
            bytemuck::cast_slice(&[total_time]),
        );

        // Update damage flash uniforms
        self.queue.write_buffer(
            &self.damage_uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.damage_flash_intensity, total_time]),
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

        // Clear hover when paused; it will be recomputed below when unpaused.
        self.hovered_dropped_item = None;

        // ── Simulation (skipped while paused) ────────────────────────────────
        if !self.paused {

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
            self.damage_flash_intensity = 1.0;
            if !self.player.is_alive() {
                self.respawn();
            }
        }
        self.camera_controller.last_fall_velocity = 0.0;

        // Update world
        self.world
            .update_chunks((self.camera.position.x, self.camera.position.z));
        self.world.rebuild_dirty_chunks(self.smooth_lighting);

        // Update water simulation
        self.water_simulation.update(&mut self.world, dt);

        // Update enemies and spawn death particles
        let death_events = self.enemy_manager.update(dt, self.player.position, &self.world);
        for (pos, color) in death_events {
            self.particle_manager.spawn_enemy_death(pos, color);
        }

        // Update birds
        self.bird_manager.update(dt, self.player.position, &self.world);

        // Update fish
        self.fish_manager.update(dt, self.player.position, &self.world);

        // Update clouds and regenerate buffers
        self.cloud_manager.update(self.player.position, self.world.get_render_distance());
        let (cloud_vertices, cloud_indices) = self.cloud_manager.get_geometry();

        if !cloud_vertices.is_empty() {
            use wgpu::util::DeviceExt;

            let vertex_data = bytemuck::cast_slice(&cloud_vertices);
            let index_data = bytemuck::cast_slice(&cloud_indices);

            // Recreate buffers if they're too small
            if vertex_data.len() as u64 > self.cloud_vertex_buffer.size() {
                self.cloud_vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Cloud Vertex Buffer"),
                    contents: vertex_data,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });
            } else {
                self.queue.write_buffer(&self.cloud_vertex_buffer, 0, vertex_data);
            }

            if index_data.len() as u64 > self.cloud_index_buffer.size() {
                self.cloud_index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Cloud Index Buffer"),
                    contents: index_data,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                });
            } else {
                self.queue.write_buffer(&self.cloud_index_buffer, 0, index_data);
            }

            self.cloud_index_count = cloud_indices.len() as u32;
        } else {
            self.cloud_index_count = 0;
        }

        // Advance dropped-item physics (no auto-collect; player must right-click)
        self.dropped_item_manager.update(dt, &self.world);

        // Raycast to find which dropped item (if any) the crosshair is over
        let look_dir = self.camera.get_direction();
        self.hovered_dropped_item = self.dropped_item_manager
            .raycast_item(self.camera.position, look_dir, 5.0);

        // Update particles
        self.particle_manager.update(dt);

        // Check for enemy damage
        let damage = self.enemy_manager.check_player_damage(self.player.position);
        if damage > 0.0 {
            self.player.take_damage(damage * dt);
            self.damage_flash_intensity = 1.0;
            if !self.player.is_alive() {
                self.respawn();
            }
        }

        // Decay damage flash
        if self.damage_flash_intensity > 0.0 {
            self.damage_flash_intensity = (self.damage_flash_intensity - dt * 3.0).max(0.0);
        }

        // Update targeted block (for outline rendering)
        let direction = self.camera.get_direction();
        self.targeted_block = self.player.raycast_block(direction, &self.world)
            .map(|(x, y, z, _)| (x, y, z));

        // Update crafting table hover hint
        self.hovered_crafting_table = match self.targeted_block {
            Some((x, y, z)) => self.world.get_block_world(x, y, z) == BlockType::CraftingTable,
            None => false,
        };

        // Update block breaking / melee combat
        self.update_block_breaking(dt);

        // Decay hit indicator and cooldown timers
        if self.hit_cooldown > 0.0 {
            self.hit_cooldown = (self.hit_cooldown - dt).max(0.0);
        }
        if self.hit_indicator_timer > 0.0 {
            self.hit_indicator_timer = (self.hit_indicator_timer - dt).max(0.0);
        }

        // Rebuild crosshair with hit indicator state
        {
            let aspect = self.size.width as f32 / self.size.height as f32;
            let hit_active = self.hit_indicator_timer > 0.0;
            let crosshair_verts = Self::build_crosshair_vertices(aspect, hit_active);
            self.crosshair_vertex_count = crosshair_verts.len() as u32;
            self.queue.write_buffer(
                &self.crosshair_vertex_buffer,
                0,
                bytemuck::cast_slice(&crosshair_verts),
            );
        }

        } // end if !self.paused

        // Update camera uniform and frustum (always, so the paused scene renders)
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
        self.rebuild_item_cube_vertices();

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
            render_pass.set_bind_group(2, &self.fog_bind_group, &[]);

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
                if !enemy.alive { continue; }
                let vertices = create_enemy_vertices(enemy);
                if vertices.is_empty() { continue; }

                let num_cubes = vertices.len() / 24;
                let indices = generate_enemy_indices(num_cubes);

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
                            contents: bytemuck::cast_slice(&indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });

                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
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
            water_pass.set_bind_group(2, &self.fog_bind_group, &[]);

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
            transparent_pass.set_bind_group(2, &self.fog_bind_group, &[]);

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

        // --- PASS 4: RENDER CLOUDS ---
        if self.cloud_index_count > 0 {
            let mut cloud_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Cloud Render Pass"),
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

            cloud_pass.set_pipeline(&self.cloud_pipeline);
            cloud_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            cloud_pass.set_bind_group(1, &self.fog_bind_group, &[]);
            cloud_pass.set_vertex_buffer(0, self.cloud_vertex_buffer.slice(..));
            cloud_pass.set_index_buffer(self.cloud_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            cloud_pass.draw_indexed(0..self.cloud_index_count, 0, 0..1);
        }

        // --- PASS 5: OVERLAYS (Breaking, Outlines, Debug) ---
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
                breaking_pass.set_bind_group(2, &self.fog_bind_group, &[]);
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

        // Render enemy collision hitbox outlines if debug mode is enabled
        if self.show_enemy_hitboxes {
            let hitbox_verts = create_enemy_collision_outlines(&self.enemy_manager.enemies);
            if !hitbox_verts.is_empty() {
                let mut hitbox_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Enemy Hitbox Outline Render Pass"),
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

                let hitbox_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Enemy Hitbox Vertex Buffer"),
                            contents: bytemuck::cast_slice(&hitbox_verts),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                hitbox_pass.set_pipeline(&self.chunk_outline_pipeline);
                hitbox_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                hitbox_pass.set_vertex_buffer(0, hitbox_buffer.slice(..));
                hitbox_pass.draw(0..hitbox_verts.len() as u32, 0..1);
            }
        }

        // --- PASS 6: POST-PROCESSING (Motion Blur + Underwater + Damage) ---
        // Texture routing:
        //   Neither:           blur scene_texture -> swap_chain
        //   Underwater only:   blur scene_texture -> post_process, underwater post_process -> swap_chain
        //   Damage only:       blur scene_texture -> post_process, damage post_process -> swap_chain
        //   Both:              blur scene_texture -> post_process, underwater post_process -> scene_texture,
        //                      damage scene_texture -> swap_chain
        let damage_active = self.damage_flash_intensity > 0.01;
        let needs_post = self.camera_underwater || damage_active;
        {
            let blur_target = if needs_post {
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
            blur_pass.draw(0..3, 0..1);
        }

        // Underwater: post_process_texture -> (scene_texture if damage follows, else swap_chain)
        if self.camera_underwater {
            let underwater_target = if damage_active {
                &self.scene_texture_view
            } else {
                &swap_chain_view
            };

            let mut underwater_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Underwater Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: underwater_target,
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
            underwater_pass.draw(0..3, 0..1);
        }

        // Damage flash: reads from post_process (no underwater) or scene_texture (after underwater) -> swap_chain
        if damage_active {
            let damage_bind_group = if self.camera_underwater {
                &self.damage_bind_group_alt
            } else {
                &self.damage_bind_group
            };

            let mut damage_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Damage Flash Render Pass"),
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

            damage_pass.set_pipeline(&self.damage_pipeline);
            damage_pass.set_bind_group(0, damage_bind_group, &[]);
            damage_pass.draw(0..3, 0..1);
        }

        if self.paused {
            // ── PAUSED: blur backdrop + modal overlay ─────────────────────────

            // Pass A: Gaussian-blur + darken scene_texture → swap_chain
            {
                let mut blur_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Pause Blur Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &swap_chain_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load:  wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                blur_pass.set_pipeline(&self.pause_blur_pipeline);
                blur_pass.set_bind_group(0, &self.pause_blur_bind_group, &[]);
                blur_pass.draw(0..3, 0..1);
            }

            // Build modal geometry on the CPU
            let sw = self.size.width  as f32;
            let sh = self.size.height as f32;

            // Sand panel vertices
            let sand_verts = self.pause_modal.build_sand_vertices(sw, sh);
            self.queue.write_buffer(
                &self.modal_sand_vertex_buffer,
                0,
                bytemuck::cast_slice(&sand_verts),
            );

            // UI overlay vertices (border, bevel, buttons, title)
            let ui_verts = self.pause_modal.build_ui_vertices(sw, sh);
            self.modal_ui_vertex_count = ui_verts.len() as u32;
            self.queue.write_buffer(
                &self.modal_ui_vertex_buffer,
                0,
                bytemuck::cast_slice(&ui_verts),
            );

            // Pass B: Sand-textured modal panel
            {
                let mut sand_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Modal Sand Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &swap_chain_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load:  wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                sand_pass.set_pipeline(&self.modal_sand_pipeline);
                sand_pass.set_bind_group(0, &self.modal_sand_bind_group, &[]);
                sand_pass.set_vertex_buffer(0, self.modal_sand_vertex_buffer.slice(..));
                sand_pass.draw(0..sand_verts.len() as u32, 0..1);
            }

            // Pass C: Solid-color modal UI (border, bevel, buttons, text)
            if self.modal_ui_vertex_count > 0 {
                let mut ui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Modal UI Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &swap_chain_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load:  wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                ui_pass.set_pipeline(&self.ui_pipeline);
                ui_pass.set_bind_group(0, &self.ui_bind_group, &[]);
                ui_pass.set_vertex_buffer(0, self.modal_ui_vertex_buffer.slice(..));
                ui_pass.draw(0..self.modal_ui_vertex_count, 0..1);
            }

        } else {
            // ── NORMAL: crosshair + HUD ───────────────────────────────────────

            // If the crafting UI is open, render its modal sand background first
            if self.crafting_ui_open {
                let sw = self.size.width  as f32;
                let sh = self.size.height as f32;
                let sand_verts = self.crafting_modal.build_sand_vertices(sw, sh);
                self.queue.write_buffer(
                    &self.modal_sand_vertex_buffer,
                    0,
                    bytemuck::cast_slice(&sand_verts),
                );
                let mut sand_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Crafting Sand Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &swap_chain_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load:  wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                sand_pass.set_pipeline(&self.modal_sand_pipeline);
                sand_pass.set_bind_group(0, &self.modal_sand_bind_group, &[]);
                sand_pass.set_vertex_buffer(0, self.modal_sand_vertex_buffer.slice(..));
                sand_pass.draw(0..sand_verts.len() as u32, 0..1);
            }

            // --- PASS 7: UI (Crosshair & HUD) ---
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

                // 1. Crosshair + HUD chrome (slot backgrounds, borders, name labels, fps)
                ui_pass.set_pipeline(&self.ui_pipeline);
                ui_pass.set_bind_group(0, &self.ui_bind_group, &[]);
                ui_pass.set_vertex_buffer(0, self.crosshair_vertex_buffer.slice(..));
                ui_pass.draw(0..self.crosshair_vertex_count, 0..1);
                ui_pass.set_vertex_buffer(0, self.hud_vertex_buffer.slice(..));
                ui_pass.draw(0..self.hud_vertex_count, 0..1);

                // 2. Item cubes (drawn on top of slot backgrounds)
                if self.item_cube_vertex_count > 0 {
                    ui_pass.set_pipeline(&self.item_cube_pipeline);
                    ui_pass.set_bind_group(0, &self.item_cube_bind_group, &[]);
                    ui_pass.set_vertex_buffer(0, self.item_cube_vertex_buffer.slice(..));
                    ui_pass.draw(0..self.item_cube_vertex_count, 0..1);
                }

                // 3. Count text overlay (drawn on top of cubes)
                if self.hud_text_vertex_count > 0 {
                    ui_pass.set_pipeline(&self.ui_pipeline);
                    ui_pass.set_bind_group(0, &self.ui_bind_group, &[]);
                    ui_pass.set_vertex_buffer(0, self.hud_text_vertex_buffer.slice(..));
                    ui_pass.draw(0..self.hud_text_vertex_count, 0..1);
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

// Push the three visible isometric faces of a block cube icon into 'verts'
fn push_item_cube(
    verts: &mut Vec<ItemCubeVertex>,
    block_type: BlockType,
    cx: f32,
    cy: f32,
    cube_size: f32,
    screen_w: f32,
    screen_h: f32,
) {
    let face_tex = block_type.get_face_textures(false);
    let bc = block_type.get_color();

    // Horizontal scale: cube_size maps to 40 isometric units wide
    let kx = cube_size / 40.0;
    // Vertical scale: 1.1× taller than the mathematically-correct square so the cube doesn't look squat (the hex waist creates a "short" illusion).
    let ky = kx * 1.1;

    // ── 7 key 2D hex points (pixel space, y increases downward) ──────────
    // Derived from 2:1 isometric with a vertical stretch applied.
    // Width = 40·kx,  Height = 40·ky,  visual center at (cx, cy).
    let tbl = [cx,           cy - 20.0 * ky]; // top vertex
    let tfl = [cx - 20.0*kx, cy - 10.0 * ky]; // upper-left
    let tbr = [cx + 20.0*kx, cy - 10.0 * ky]; // upper-right
    let tfr = [cx,           cy             ]; // hex center (inner divider)
    let fl  = [cx - 20.0*kx, cy + 10.0 * ky]; // lower-left
    let br  = [cx + 20.0*kx, cy + 10.0 * ky]; // lower-right
    let fr  = [cx,           cy + 20.0 * ky]; // bottom vertex

    // Pixel-to-clip-space conversion (y flipped: screen y-down → clip y-up)
    let p2c = |p: [f32; 2]| -> [f32; 2] {
        [2.0 * p[0] / screen_w - 1.0, 1.0 - 2.0 * p[1] / screen_h]
    };

    // Atlas UV for a given tile index
    let tile_uv = |tile: u32| -> (f32, f32, f32, f32) {
        let col = (tile % 16) as f32;
        let row = (tile / 16) as f32;
        let u0 = col / 16.0;
        let v0 = row / 16.0;
        (u0, v0, u0 + 1.0 / 16.0, v0 + 1.0 / 16.0)
    };

    // Build one ItemCubeVertex
    let make_v = |pos: [f32; 2], uv: [f32; 2], shade: f32, tile: u32| -> ItemCubeVertex {
        let use_texture = if tile != TEX_NONE { 1.0_f32 } else { 0.0_f32 };
        let color = if tile != TEX_NONE {
            [shade, shade, shade, 1.0]
        } else {
            [bc[0] * shade, bc[1] * shade, bc[2] * shade, 1.0]
        };
        ItemCubeVertex {
            position: p2c(pos),
            uv,
            color,
            use_texture,
            _pad: 0.0,
        }
    };

    // Push a quad (4 corners CCW) as two triangles: (0,1,2) and (0,2,3)
    let mut push_quad = |v0: ItemCubeVertex, v1: ItemCubeVertex, v2: ItemCubeVertex, v3: ItemCubeVertex| {
        verts.push(v0);
        verts.push(v1);
        verts.push(v2);
        verts.push(v0);
        verts.push(v2);
        verts.push(v3);
    };

    // ── TOP FACE (shade 1.0, uses face_tex.top) ───────────────────────────
    // Corners: tfl, tbl, tbr, tfr
    // UV (looking down from above — x right, z toward viewer):
    //   tfl (front-left)  → (u0, v1)
    //   tbl (back-left)   → (u0, v0)
    //   tbr (back-right)  → (u1, v0)
    //   tfr (front-right) → (u1, v1)
    {
        let ti = face_tex.top;
        let (u0, v0, u1, v1) = tile_uv(ti);
        let shade = 1.0;
        push_quad(
            make_v(tfl, [u0, v1], shade, ti),
            make_v(tbl, [u0, v0], shade, ti),
            make_v(tbr, [u1, v0], shade, ti),
            make_v(tfr, [u1, v1], shade, ti),
        );
    }

    // ── LEFT FACE — front-left (Z+ face, shade 0.75, uses face_tex.sides) ─
    // Corners: tfl, tfr, fr, fl
    // UV (looking at Z+ face from outside — x right, y up):
    //   tfl → (u0, v0)  tfr → (u1, v0)  fr → (u1, v1)  fl → (u0, v1)
    {
        let ti = face_tex.sides;
        let (u0, v0, u1, v1) = tile_uv(ti);
        let shade = 0.75;
        push_quad(
            make_v(tfl, [u0, v0], shade, ti),
            make_v(tfr, [u1, v0], shade, ti),
            make_v(fr,  [u1, v1], shade, ti),
            make_v(fl,  [u0, v1], shade, ti),
        );
    }

    // ── RIGHT FACE (X+ face, shade 0.60, uses face_tex.sides) ────────────
    // Corners: tfr, tbr, br, fr
    // UV (looking at X+ face from outside — z-axis "left", y up):
    //   tfr → (u0, v0)  tbr → (u1, v0)  br → (u1, v1)  fr → (u0, v1)
    {
        let ti = face_tex.sides;
        let (u0, v0, u1, v1) = tile_uv(ti);
        let shade = 0.60;
        push_quad(
            make_v(tfr, [u0, v0], shade, ti),
            make_v(tbr, [u1, v0], shade, ti),
            make_v(br,  [u1, v1], shade, ti),
            make_v(fr,  [u0, v1], shade, ti),
        );
    }
}