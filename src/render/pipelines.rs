use super::*;

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

        // World config setup - load master_seed from the per-world save directory
        let world_config = crate::config::WorldConfig::load_or_create(
            &std::path::Path::new(&crate::save_context::world_config_path()),
        );
        log::info!("Using master_seed = {}", world_config.master_seed);

        // Terrain generation config - loaded from config.toml alongside other settings
        let mut terrain_cfg_val = crate::config::TerrainConfig::load_or_create(config_path);

        // Apply world type overrides
        match world_config.world_type {
            crate::config::WorldType::Normal => {
                terrain_cfg_val.frozen_stone_ceiling_enabled = false;
            }
            crate::config::WorldType::Crust => {
                terrain_cfg_val.frozen_stone_ceiling_enabled = true;
            }
        }

        let terrain_cfg = std::sync::Arc::new(terrain_cfg_val);

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

        // ── Sky config ───────────────────────────────────────────────────────
        let sky_config = crate::config::SkyConfig::load_or_create(config_path);

        // ── Sun uniform + shadow map (bind group 3 for block shaders) ─────────
        let identity_mat: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        ];
        let sun_uniform = crate::config::SunUniform {
            sun_view_proj: identity_mat,
            sun_dir: [0.5, 1.0, 0.3, 0.0],
            sun_color: [sky_config.sun_color_r, sky_config.sun_color_g, sky_config.sun_color_b, 1.0],
            params: [1.0, sky_config.night_ambient, sky_config.shadow_strength, sky_config.shadow_bias],
            params2: [sky_config.shadow_softness, 0.0, 0.0, 0.0],
        };

        let sun_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sun Uniform Buffer"),
            contents: bytemuck::cast_slice(&[sun_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Shadow map depth texture
        let shadow_res = sky_config.shadow_map_resolution;
        let shadow_map_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow Map Texture"),
            size: wgpu::Extent3d {
                width: shadow_res,
                height: shadow_res,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_map_view = shadow_map_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let shadow_map_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Map Sampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let sun_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
            label: Some("sun_bind_group_layout"),
        });

        let sun_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &sun_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sun_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadow_map_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&shadow_map_sampler),
                },
            ],
            label: Some("sun_bind_group"),
        });

        // ── Shadow map render pipeline ───────────────────────────────────────
        let shadow_map_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow Map Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shadow_map_shader.wgsl").into()),
        });

        // Sun camera bind group for shadow pass (just a view-proj matrix)
        let shadow_map_camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Shadow Map Camera Buffer"),
            contents: bytemuck::cast_slice(&identity_mat),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let shadow_map_camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("shadow_map_camera_bind_group_layout"),
        });

        let shadow_map_camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &shadow_map_camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: shadow_map_camera_buffer.as_entire_binding(),
            }],
            label: Some("shadow_map_camera_bind_group"),
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shader.wgsl").into()),
        });

        // Create texture atlas
        let texture_atlas = TextureAtlas::new(&device, &queue);

        // Shadow map pipeline (depth-only pass from sun's perspective)
        let shadow_map_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow Map Pipeline Layout"),
            bind_group_layouts: &[&shadow_map_camera_bind_group_layout, &texture_atlas.bind_group_layout],
            push_constant_ranges: &[],
        });

        let shadow_map_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow Map Pipeline"),
            layout: Some(&shadow_map_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shadow_map_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shadow_map_shader,
                entry_point: Some("fs_main"),
                targets: &[], // depth-only, no color attachment
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

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &texture_atlas.bind_group_layout, &fog_bind_group_layout, &sun_bind_group_layout],
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

        // Bloom emissive pipeline: renders only emissive blocks into the bloom texture.
        // Uses fs_emissive which discards non-emissive fragments.
        // Depth testing (read-only) prevents bloom from leaking through walls.
        let bloom_emissive_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bloom Emissive Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_emissive"),
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
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false, // read-only — reuse main depth buffer
                depth_compare: wgpu::CompareFunction::LessEqual, // equal depth passes (same geometry re-rendered)
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/water_shader.wgsl").into()),
        });

        // Water pipeline layout includes camera bind group, depth texture bind group, fog bind group, and sun bind group
        let water_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Water Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout, &water_bind_group_layout, &fog_bind_group_layout, &sun_bind_group_layout],
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
                cull_mode: Some(wgpu::Face::Back), // Cull back faces; underwater view handled by dedicated underwater pass
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true, // Write depth so only the closest water face renders per pixel; prevents transparency sorting artifacts in waterfalls
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

        let frozen_stone_ceiling = terrain_cfg.frozen_stone_ceiling_enabled;
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ui_shader.wgsl").into()),
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/item_cube_shader.wgsl").into()),
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/outline_shader.wgsl").into()),
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/chunk_outline_shader.wgsl").into()),
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/cloud_shader.wgsl").into()),
        });

        // Drift uniform: [drift_x, drift_z, pad, pad] — 16 bytes.
        // Applied in the vertex shader so the CPU never rewrites vertex positions per frame.
        let cloud_drift_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cloud Drift Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let cloud_drift_buffer = {
            use wgpu::util::DeviceExt;
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Cloud Drift Buffer"),
                contents: bytemuck::bytes_of(&[0.0f32, 0.0, 0.0, 0.0]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };

        let cloud_drift_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cloud Drift Bind Group"),
            layout: &cloud_drift_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: cloud_drift_buffer.as_entire_binding(),
            }],
        });

        let cloud_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cloud Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout, &fog_bind_group_layout, &cloud_drift_bind_group_layout],
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

        // ── Sky Pipeline ──────────────────────────────────────────────────────
        // Sky uniform buffer: holds inverse view-proj, camera pos, sun params, sky colors
        // Total size: 12 * vec4 = 192 bytes
        let sky_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sky Uniform Buffer"),
            size: 224, // 56 floats: mat4x4(16) + 10 * vec4(4) = 56 * 4 bytes
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sky_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            label: Some("sky_bind_group_layout"),
        });

        let sky_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &sky_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sky_uniform_buffer.as_entire_binding(),
            }],
            label: Some("sky_bind_group"),
        });

        let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sky Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sky_shader.wgsl").into()),
        });

        let sky_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sky Pipeline Layout"),
            bind_group_layouts: &[&sky_bind_group_layout],
            push_constant_ranges: &[],
        });

        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sky Pipeline"),
            layout: Some(&sky_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &sky_shader,
                entry_point: Some("vs_main"),
                buffers: &[], // Fullscreen triangle — no vertex buffer needed
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &sky_shader,
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
                cull_mode: None, // Fullscreen triangle — no culling
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None, // Sky has no depth — it's the background
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/breaking_shader.wgsl").into()),
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/underwater_shader.wgsl").into()),
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/motion_blur_shader.wgsl").into()),
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/damage_shader.wgsl").into()),
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

        // ── BLOOM POST-PROCESSING ────────────────────────────────────────────
        // Full-resolution emissive texture (rendered with depth test to prevent bloom through walls)
        let bloom_emissive_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bloom Emissive Texture"),
            size: wgpu::Extent3d { width: config.width, height: config.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let bloom_emissive_texture_view = bloom_emissive_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Quarter-resolution textures for blur ping-pong
        let bloom_w = (config.width / 4).max(1);
        let bloom_h = (config.height / 4).max(1);

        let bloom_texture_a = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bloom Texture A"),
            size: wgpu::Extent3d { width: bloom_w, height: bloom_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let bloom_texture_a_view = bloom_texture_a.create_view(&wgpu::TextureViewDescriptor::default());

        let bloom_texture_b = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bloom Texture B"),
            size: wgpu::Extent3d { width: bloom_w, height: bloom_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let bloom_texture_b_view = bloom_texture_b.create_view(&wgpu::TextureViewDescriptor::default());

        // Bloom bind group layout: texture + sampler (shared by all bloom passes)
        let bloom_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bloom BGL"),
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

        // Bloom bind groups
        let bloom_emissive_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bloom Emissive BG"),
            layout: &bloom_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&bloom_emissive_texture_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&scene_sampler) },
            ],
        });
        let bloom_a_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bloom A BG"),
            layout: &bloom_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&bloom_texture_a_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&scene_sampler) },
            ],
        });
        let bloom_b_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bloom B BG"),
            layout: &bloom_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&bloom_texture_b_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&scene_sampler) },
            ],
        });

        // Bloom shader and pipelines
        let bloom_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/bloom_shader.wgsl").into()),
        });

        let bloom_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bloom Pipeline Layout"),
            bind_group_layouts: &[&bloom_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Helper closure: create a bloom pipeline with a given fragment entry point and blend state
        let create_bloom_pipeline = |label: &str, fs_entry: &str, blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&bloom_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &bloom_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &bloom_shader,
                    entry_point: Some(fs_entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };

        let bloom_downsample_pipeline = create_bloom_pipeline("Bloom Downsample Pipeline", "fs_downsample", None);
        let bloom_blur_h_pipeline = create_bloom_pipeline("Bloom Blur H Pipeline", "fs_blur_h", None);
        let bloom_blur_v_pipeline = create_bloom_pipeline("Bloom Blur V Pipeline", "fs_blur_v", None);
        // Composite uses additive blending to overlay bloom onto scene_texture
        let bloom_composite_pipeline = create_bloom_pipeline(
            "Bloom Composite Pipeline",
            "fs_composite",
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::OVER,
            }),
        );

        // ── PAUSE BLUR PIPELINE ───────────────────────────────────────────────
        // Reads scene_texture, applies a Gaussian blur + darkening,
        // then outputs to the swap chain as the pause backdrop.

        let pause_blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pause Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/pause_blur_shader.wgsl").into(),
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

        // ── MODAL BACKGROUND TEXTURE ────────────────────────────────────────────────
        let sand_img = image::open(MODAL_BG_TEXTURE)
            .expect("Cannot open MODAL_BG_TEXTURE")
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
                include_str!("../shaders/modal_sand_shader.wgsl").into(),
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
        let mut pause_modal = Modal::new("PAUSED", &["RESUME", "SAVE AND QUIT"], modal::MODAL_W_RATIO, modal::MODAL_ASPECT);
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
            hud_enabled: true,
            frozen_stone_ceiling,
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
            // Bloom
            bloom_emissive_texture,
            bloom_emissive_texture_view,
            bloom_texture_a,
            bloom_texture_a_view,
            bloom_texture_b,
            bloom_texture_b_view,
            bloom_emissive_pipeline,
            bloom_downsample_pipeline,
            bloom_blur_h_pipeline,
            bloom_blur_v_pipeline,
            bloom_composite_pipeline,
            bloom_bind_group_layout,
            bloom_emissive_bind_group,
            bloom_a_bind_group,
            bloom_b_bind_group,
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
            cloud_drift_buffer,
            cloud_drift_bind_group,

            // Sky / Day-Night Cycle
            sky_config,
            sky_pipeline,
            sky_bind_group_layout,
            sky_bind_group,
            sky_uniform_buffer,
            sun_buffer,
            sun_bind_group_layout,
            sun_bind_group,
            shadow_map_texture,
            shadow_map_view,
            shadow_map_sampler,
            shadow_map_pipeline,
            shadow_map_camera_buffer,
            shadow_map_camera_bind_group,

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

            // Recreate Bloom Emissive Texture (full resolution)
            self.bloom_emissive_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Bloom Emissive Texture"),
                size: wgpu::Extent3d { width: new_size.width, height: new_size.height, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.bloom_emissive_texture_view = self.bloom_emissive_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Recreate Bloom Textures (quarter resolution)
            let bloom_w = (new_size.width / 4).max(1);
            let bloom_h = (new_size.height / 4).max(1);
            self.bloom_texture_a = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Bloom Texture A"),
                size: wgpu::Extent3d { width: bloom_w, height: bloom_h, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.bloom_texture_a_view = self.bloom_texture_a.create_view(&wgpu::TextureViewDescriptor::default());
            self.bloom_texture_b = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Bloom Texture B"),
                size: wgpu::Extent3d { width: bloom_w, height: bloom_h, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.bloom_texture_b_view = self.bloom_texture_b.create_view(&wgpu::TextureViewDescriptor::default());

            // Recreate Bloom Bind Groups
            self.bloom_emissive_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bloom Emissive BG"),
                layout: &self.bloom_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.bloom_emissive_texture_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.scene_sampler) },
                ],
            });
            self.bloom_a_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bloom A BG"),
                layout: &self.bloom_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.bloom_texture_a_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.scene_sampler) },
                ],
            });
            self.bloom_b_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bloom B BG"),
                layout: &self.bloom_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.bloom_texture_b_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.scene_sampler) },
                ],
            });

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
}
