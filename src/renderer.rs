use crate::block::{BlockType, Vertex, UiVertex, LineVertex, create_cube_vertices, create_block_outline, CUBE_INDICES};
use crate::camera::{Camera, CameraController, CameraUniform, Projection};
use crate::crafting::CraftingSystem;
use crate::enemy::EnemyManager;
use crate::bitmap_font;
use crate::player::Player;
use crate::water::WaterSimulation;
use crate::world::World;
use cgmath::{Point3, Vector3};
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::event::*;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    water_pipeline: wgpu::RenderPipeline,  // Separate pipeline for transparent water with depth sampling
    camera: Camera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    projection: Projection,
    world: World,
    player: Player,
    water_simulation: WaterSimulation,
    enemy_manager: EnemyManager,
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
    // FPS tracking
    fps: f32,
    fps_frame_count: u32,
    fps_timer: f32,
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

        let camera = Camera::new(Point3::new(0.0, 35.0, 0.0));
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
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout],
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
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
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

        // Bind group layout for water shader depth sampling
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
            ],
        });

        // Bind group for water depth sampling (uses the copy texture)
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
            ],
        });

        // Water shader with depth-based transparency
        let water_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("water_shader.wgsl").into()),
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

        let world = World::new(4); // Sets render distance - 4 seems like a good balance when chunk size is 32
        let player = Player::new(Point3::new(0.0, 35.0, 0.0));
        let water_simulation = WaterSimulation::new(0.5);
        let enemy_manager = EnemyManager::new(10.0, 10);
        let crafting_system = CraftingSystem::new();

        // UI Pipeline for crosshair
        let ui_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("UI Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ui_shader.wgsl").into()),
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
            source: wgpu::ShaderSource::Wgsl(include_str!("outline_shader.wgsl").into()),
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

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            water_pipeline,
            camera,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            projection,
            world,
            player,
            water_simulation,
            enemy_manager,
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
            mouse_captured: false,
            depth_texture,
            depth_view,
            depth_copy_texture,
            depth_copy_view,
            depth_sampler,
            water_bind_group_layout,
            water_bind_group,
            // FPS tracking
            fps: 0.0,
            fps_frame_count: 0,
            fps_timer: 0.0,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
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

    pub fn process_mouse(&mut self, dx: f64, dy: f64) {
        if self.mouse_captured {
            self.camera_controller.process_mouse(&mut self.camera, dx as f32, dy as f32);
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
        let slot_size = 70.0;
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

        // === Debug Axes (top-right) ===
        self.build_debug_axes(&mut verts, screen_w, screen_h);

        let fill = [1.0, 1.0, 1.0, 0.10];
        let outline = [1.0, 1.0, 1.0, 0.85];
        let outline_selected = [1.0, 1.0, 1.0, 1.0];

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
            let thick = if i == selected { 3.0 } else { 1.5 };
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
                // Color square
                let bc = stack.block_type.get_color();
                let block_color = [bc[0], bc[1], bc[2], 1.0];
                bitmap_font::push_rect_px(
                    &mut verts,
                    x + 6.0,
                    y + 6.0,
                    24.0,
                    24.0,
                    block_color,
                    screen_w,
                    screen_h,
                );

                // Block name (uppercased for font coverage)
                let name = stack.block_type.display_name();
                let name_color = [0.08, 0.08, 0.08, 0.95];
                bitmap_font::draw_text_quads(
                    &mut verts,
                    name,
                    x + 6.0,
                    y + 36.0,
                    2.0,
                    2.0,
                    name_color,
                    screen_w,
                    screen_h,
                );

                // Count "xN"
                let count_text = format!("x{}", stack.count);
                let count_color = [0.08, 0.08, 0.08, 0.9];
                bitmap_font::draw_text_quads(
                    &mut verts,
                    &count_text,
                    x + 6.0,
                    y + slot_size - 16.0,
                    2.0,
                    2.0,
                    count_color,
                    screen_w,
                    screen_h,
                );
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
        let line_thickness = 3.0;

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

        // Draw background circle
        let bg_color = [0.0, 0.0, 0.0, 0.4];
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
                    _ => false,
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                // Only handle block break if mouse is captured (in-game)
                if self.mouse_pressed && self.mouse_captured {
                    self.handle_block_break();
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
            _ => false,
        }
    }

    fn handle_block_break(&mut self) {
        let direction = self.camera.get_direction();
        if let Some((x, y, z, _)) = self.player.raycast_block(direction, &self.world) {
            let block = self.world.get_block_world(x, y, z);
            if block != BlockType::Air {
                self.player.inventory.add_item(block, 1);
                self.world.set_block_world(x, y, z, BlockType::Air);
                println!("Broke block: {:?}", block);
            }
        }
    }

    fn handle_block_place(&mut self) {
        if let Some(selected) = self.player.inventory.get_selected_item() {
            let block_type = selected.block_type;
            let direction = self.camera.get_direction();

            if let Some((x, y, z, normal)) = self.player.raycast_block(direction, &self.world) {
                let place_x = x + normal.x;
                let place_y = y + normal.y;
                let place_z = z + normal.z;

                if self.world.get_block_world(place_x, place_y, place_z) == BlockType::Air {
                    self.world
                        .set_block_world(place_x, place_y, place_z, block_type);
                    self.player
                        .inventory
                        .remove_item(self.player.inventory.selected_slot, 1);
                    println!("Placed block: {:?}", block_type);
                }
            }
        }
    }

    pub fn update(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        // Update FPS counter
        self.fps_frame_count += 1;
        self.fps_timer += dt;
        if self.fps_timer >= 0.5 {
            self.fps = self.fps_frame_count as f32 / self.fps_timer;
            self.fps_frame_count = 0;
            self.fps_timer = 0.0;
        }

        // Update camera
        self.camera_controller.update_camera(&mut self.camera, dt, &self.world);
        self.player.position = self.camera.position;

        // Update world
        self.world
            .update_chunks((self.camera.position.x, self.camera.position.z));
        self.world.rebuild_dirty_chunks();

        // Update water simulation
        self.water_simulation.update(&mut self.world, dt);

        // Update enemies
        // TODO: Re-enable enemy spawning after testing
        // self.enemy_manager.update(dt, self.player.position);

        // Check for enemy damage
        let damage = self.enemy_manager.check_player_damage(self.player.position);
        if damage > 0.0 {
            self.player.take_damage(damage * dt);
            if !self.player.is_alive() {
                println!("GAME OVER! You died!");
            }
        }

        // Update targeted block (for outline rendering)
        let direction = self.camera.get_direction();
        self.targeted_block = self.player.raycast_block(direction, &self.world)
            .map(|(x, y, z, _)| (x, y, z));

        // Update camera uniform
        self.camera_uniform
            .update_view_proj(&self.camera, &self.projection);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
    self.rebuild_hud_vertices();

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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

            // Render chunks
            for chunk in self.world.chunks.values() {
                if !chunk.vertices.is_empty() {
                    let vertex_buffer =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Chunk Vertex Buffer"),
                                contents: bytemuck::cast_slice(&chunk.vertices),
                                usage: wgpu::BufferUsages::VERTEX,
                            });

                    let index_buffer =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Chunk Index Buffer"),
                                contents: bytemuck::cast_slice(&chunk.indices),
                                usage: wgpu::BufferUsages::INDEX,
                            });

                    render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                    render_pass.draw_indexed(0..chunk.indices.len() as u32, 0, 0..1);
                }
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

        // Render water in a separate pass (reads depth copy, writes to same depth for depth testing)
        {
            let mut water_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Water Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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

            for chunk in self.world.chunks.values() {
                if !chunk.water_vertices.is_empty() {
                    let water_vertex_buffer =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Water Vertex Buffer"),
                                contents: bytemuck::cast_slice(&chunk.water_vertices),
                                usage: wgpu::BufferUsages::VERTEX,
                            });

                    let water_index_buffer =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Water Index Buffer"),
                                contents: bytemuck::cast_slice(&chunk.water_indices),
                                usage: wgpu::BufferUsages::INDEX,
                            });

                    water_pass.set_vertex_buffer(0, water_vertex_buffer.slice(..));
                    water_pass
                        .set_index_buffer(water_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                    water_pass.draw_indexed(0..chunk.water_indices.len() as u32, 0, 0..1);
                }
            }
        }

        // Render block outline if targeting a block - uses depth buffer to hide occluded edges
        if let Some((x, y, z)) = self.targeted_block {
            let mut outline_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Outline Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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

        // Render UI (crosshair) - separate pass without depth
        {
            let mut ui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("UI Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Keep existing content
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
