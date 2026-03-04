//! Main-menu renderer — shown before the game world is loaded.
//!
//! Three views:
//!   Main     – "CRAFT" title with NEW GAME / LOAD GAME / QUIT
//!   NewGame  – World-name text input + CREATE / BACK
//!   LoadGame – Scrollable list of existing worlds + BACK

use std::sync::Arc;
use std::time::Instant;

use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

use crate::bitmap_font;
use crate::block::{ModalVertex, UiVertex};
use crate::modal::{
    push_rect_px, Modal, MODAL_ASPECT, MODAL_BEVEL_LIGHT_COLOR,
    MODAL_BEVEL_PX, MODAL_BEVEL_SHADOW_COLOR, MODAL_BORDER_COLOR, MODAL_BORDER_PX,
    MODAL_BTN_TEXT_SCALE_RATIO, MODAL_BUTTON_BEVEL_COLOR, MODAL_BUTTON_BG_COLOR,
    MODAL_BUTTON_BORDER_COLOR, MODAL_BUTTON_BORDER_PX, MODAL_BUTTON_BEVEL_PX,
    MODAL_BUTTON_HOVER_COLOR, MODAL_BUTTON_SHADOW_COLOR, MODAL_BUTTON_TEXT_COLOR,
    MODAL_BUTTON_W_RATIO, SAND_TILE_PX, MODAL_TEXT_SHADOW_PX,
    MODAL_TITLE_COLOR, MODAL_TITLE_SCALE_RATIO, MODAL_TITLE_SHADOW, MODAL_W_RATIO,
    px_to_clip_x, px_to_clip_y,
};

// ─────────────────────────────────────────────────────────────────────────────
//  Public types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum MenuPhase {
    Main,
    NewGame,
    LoadGame,
}

pub enum MenuAction {
    None,
    StartNewGame { world_name: String },
    LoadGame { world_name: String },
    Quit,
}

// ─────────────────────────────────────────────────────────────────────────────
//  Layout constants for the main-menu panel (3-button variant)
// ─────────────────────────────────────────────────────────────────────────────

/// Fraction of screen width for the main-menu panel.
const MAIN_PANEL_W_RATIO: f32 = MODAL_W_RATIO * 1.2;
/// Height-to-width ratio that gives enough room for title + 3 buttons.
const MAIN_PANEL_ASPECT: f32 = MODAL_ASPECT * 0.72; // ≈ 1.28:1 (shorter panel)
/// Fraction of panel_h where the first button starts.
const MAIN_BTN_Y_RATIO: f32 = 0.28;
/// Button height as fraction of panel_h.
const MAIN_BTN_H_RATIO: f32 = 0.16;
/// Gap between buttons as fraction of panel_h.
const MAIN_BTN_GAP_RATIO: f32 = 0.04;
/// Title Y distance from panel top, as fraction of panel_h.
const MAIN_TITLE_Y_RATIO: f32 = 0.10;

/// Background textures
const MODAL_BG_TEXTURE: &str = "assets/textures/blocks/planks.png";
const VIEW_BG_TEXTURE: &str = "assets/textures/blocks/sand.png";

// ─────────────────────────────────────────────────────────────────────────────
//  A button with a heap-allocated (dynamic) label, used for world slots.
// ─────────────────────────────────────────────────────────────────────────────

struct DynButton {
    label: String,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    hovered: bool,
}

impl DynButton {
    fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.x + self.w && py >= self.y && py <= self.y + self.h
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  MenuState
// ─────────────────────────────────────────────────────────────────────────────

pub struct MenuState {
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface_config: wgpu::SurfaceConfiguration,

    // Pipelines
    ui_pipeline: wgpu::RenderPipeline,
    ui_bind_group: wgpu::BindGroup,
    modal_pipeline: wgpu::RenderPipeline,
    view_bg_bind_group: wgpu::BindGroup,
    modal_bg_bind_group: wgpu::BindGroup,

    // ── State ────────────────────────────────────────────────────────────────
    pub phase: MenuPhase,

    /// World name typed by the player (NewGame phase).
    pub text_input: String,
    /// Validation error shown below the text field.
    pub error_msg: Option<&'static str>,
    /// Cursor blink timer.
    blink_timer: Instant,
    cursor_visible: bool,

    /// Hover state for the 3 main-menu buttons (NEW GAME / LOAD GAME / QUIT).
    main_hovered: [bool; 3],


    /// Saved world names (populated when entering LoadGame phase).
    saved_worlds: Vec<String>,
    /// Dynamic buttons for the world list.
    load_buttons: Vec<DynButton>,
    /// "BACK" button for the Load Game view.
    load_back: DynButton,

    // ── Layout helpers (recomputed on resize) ────────────────────────────────
    /// Main-menu panel bounds in pixel space.
    main_panel_x: f32,
    main_panel_y: f32,
    main_panel_w: f32,
    main_panel_h: f32,

    /// NEW WORLD panel bounds (manually laid out, not from Modal).
    ng_panel_x: f32,
    ng_panel_y: f32,
    ng_panel_w: f32,
    ng_panel_h: f32,
    /// Hover state for the two New-Game buttons (CREATE / BACK).
    ng_create_hovered: bool,
    ng_back_hovered: bool,

    /// LOAD WORLD modal (structure only, no buttons — buttons are dynamic)
    load_game_modal: Modal,

    pub screen_w: f32,
    pub screen_h: f32,
    pub size: PhysicalSize<u32>,
}

impl MenuState {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let screen_w = size.width as f32;
        let screen_h = size.height as f32;

        // ── wgpu setup ───────────────────────────────────────────────────────
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
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // ── UI pipeline (solid-color quads, same as renderer.rs) ─────────────
        let ui_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Menu UI Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ui_shader.wgsl").into()),
        });

        // The ui_shader declares a uniform binding for aspect_ratio even though
        // the vertex shader doesn't actually read it — we must still provide it.
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct UiUniform { aspect_ratio: f32, _pad: [f32; 3] }

        let ui_uniform = UiUniform {
            aspect_ratio: screen_w / screen_h.max(1.0),
            _pad: [0.0; 3],
        };
        let ui_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Menu UI Uniform"),
            contents: bytemuck::cast_slice(&[ui_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let ui_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Menu UI BGL"),
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
        });

        let ui_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Menu UI BG"),
            layout: &ui_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ui_uniform_buffer.as_entire_binding(),
            }],
        });

        let ui_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Menu UI Pipeline Layout"),
            bind_group_layouts: &[&ui_bgl],
            push_constant_ranges: &[],
        });

        let ui_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Menu UI Pipeline"),
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
                    format: surface_format,
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
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // ── View background ─────────────
        let view_img = image::open(VIEW_BG_TEXTURE)
            .expect("Cannot open VIEW_BG_TEXTURE")
            .into_rgba8();
        let (view_w, view_h) = view_img.dimensions();

        let view_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Menu View Texture"),
            size: wgpu::Extent3d { width: view_w, height: view_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &view_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &view_img,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * view_w),
                rows_per_image: Some(view_h),
            },
            wgpu::Extent3d { width: view_w, height: view_h, depth_or_array_layers: 1 },
        );
        let view_view = view_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let view_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Menu View Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let view_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Menu View BGL"),
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

        let view_bg_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Menu View BG"),
            layout: &view_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&view_sampler),
                },
            ],
        });

        // ── Modal background ─────────────────────
        let modal_img = image::open(MODAL_BG_TEXTURE)
            .expect("Cannot open MODAL_BG_TEXTURE")
            .into_rgba8();
        let (modal_w, modal_h) = modal_img.dimensions();
        let modal_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Menu Modal Texture"),
            size: wgpu::Extent3d { width: modal_w, height: modal_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &modal_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &modal_img,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * modal_w),
                rows_per_image: Some(modal_h),
            },
            wgpu::Extent3d { width: modal_w, height: modal_h, depth_or_array_layers: 1 },
        );
        let modal_view = modal_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let modal_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Menu Modal Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let modal_bg_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Menu Modal BG"),
            layout: &view_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&modal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&modal_sampler),
                },
            ],
        });

        let view_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Menu View Pipeline Layout"),
            bind_group_layouts: &[&view_bgl],
            push_constant_ranges: &[],
        });
        let view_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Menu View Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/modal_sand_shader.wgsl").into(),
            ),
        });
        let modal_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Menu View Pipeline"),
                layout: Some(&view_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &view_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[ModalVertex::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &view_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
            });

        // ── Modals ────────────────────────────────────────────────────────────
        let mut load_game_modal =
            Modal::new("LOAD WORLD", &[], MODAL_W_RATIO, MODAL_ASPECT);
        load_game_modal.update_layout(screen_w, screen_h);

        // ── Initial main-panel layout ─────────────────────────────────────────
        let (mpx, mpy, mpw, mph) = main_panel_bounds(screen_w, screen_h);
        let (ngx, ngy, ngw, ngh) = ng_panel_bounds(screen_w, screen_h);

        let back_btn = DynButton {
            label: "BACK".to_string(),
            x: 0.0, y: 0.0, w: 0.0, h: 0.0,
            hovered: false,
        };

        let mut state = Self {
            surface,
            device,
            queue,
            surface_config,
            ui_pipeline,
            ui_bind_group,
            modal_pipeline,
            view_bg_bind_group,
            modal_bg_bind_group,
            phase: MenuPhase::Main,
            text_input: String::new(),
            error_msg: None,
            blink_timer: Instant::now(),
            cursor_visible: true,
            main_hovered: [false; 3],
            saved_worlds: Vec::new(),
            load_buttons: Vec::new(),
            load_back: back_btn,
            main_panel_x: mpx,
            main_panel_y: mpy,
            main_panel_w: mpw,
            main_panel_h: mph,
            ng_panel_x: ngx,
            ng_panel_y: ngy,
            ng_panel_w: ngw,
            ng_panel_h: ngh,
            ng_create_hovered: false,
            ng_back_hovered: false,
            load_game_modal,
            screen_w,
            screen_h,
            size,
        };
        state.rebuild_load_buttons();
        state
    }

    // ─────────────────────────────────────────────────────────────────────────

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);
        self.screen_w = new_size.width as f32;
        self.screen_h = new_size.height as f32;

        let (mpx, mpy, mpw, mph) = main_panel_bounds(self.screen_w, self.screen_h);
        self.main_panel_x = mpx;
        self.main_panel_y = mpy;
        self.main_panel_w = mpw;
        self.main_panel_h = mph;

        let (ngx, ngy, ngw, ngh) = ng_panel_bounds(self.screen_w, self.screen_h);
        self.ng_panel_x = ngx;
        self.ng_panel_y = ngy;
        self.ng_panel_w = ngw;
        self.ng_panel_h = ngh;

        self.load_game_modal.update_layout(self.screen_w, self.screen_h);
        self.rebuild_load_buttons();
    }

    // ─────────────────────────────────────────────────────────────────────────

    pub fn handle_event(&mut self, event: &WindowEvent) -> MenuAction {
        // Cursor blink update
        if self.blink_timer.elapsed().as_secs_f32() >= 0.5 {
            self.cursor_visible = !self.cursor_visible;
            self.blink_timer = Instant::now();
        }

        match event {
            WindowEvent::CursorMoved { position, .. } => {
                let px = position.x as f32;
                let py = position.y as f32;
                self.update_hover(px, py);
                MenuAction::None
            }

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => self.handle_click(),

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key,
                        text,
                        ..
                    },
                ..
            } => {
                if self.phase == MenuPhase::NewGame {
                    match physical_key {
                        PhysicalKey::Code(KeyCode::Backspace) => {
                            self.text_input.pop();
                            self.error_msg = None;
                        }
                        PhysicalKey::Code(KeyCode::Escape) => {
                            self.phase = MenuPhase::Main;
                            self.text_input.clear();
                            self.error_msg = None;
                        }
                        PhysicalKey::Code(KeyCode::Enter)
                        | PhysicalKey::Code(KeyCode::NumpadEnter) => {
                            return self.try_create_world();
                        }
                        _ => {
                            if let Some(t) = text {
                                for ch in t.chars() {
                                    if is_valid_name_char(ch) && self.text_input.len() < 32 {
                                        self.text_input.push(ch);
                                        self.error_msg = None;
                                    }
                                }
                            }
                        }
                    }
                } else if self.phase == MenuPhase::LoadGame || self.phase == MenuPhase::Main {
                    if let PhysicalKey::Code(KeyCode::Escape) = physical_key {
                        if self.phase == MenuPhase::LoadGame {
                            self.phase = MenuPhase::Main;
                        }
                    }
                }
                MenuAction::None
            }

            _ => MenuAction::None,
        }
    }

    // ─────────────────────────────────────────────────────────────────────────

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // Blink tick
        if self.blink_timer.elapsed().as_secs_f32() >= 0.5 {
            self.cursor_visible = !self.cursor_visible;
            self.blink_timer = Instant::now();
        }

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Menu Encoder"),
            });

        // Build vertex data
        let mut view_verts: Vec<ModalVertex> = Vec::new();
        let mut ui_verts: Vec<UiVertex> = Vec::new();
        self.build_vertices(&mut view_verts, &mut ui_verts);

        // Full-screen modal background (clip-space quad, tiled UVs)
        let bg_u = self.screen_w / 128.0;
        let bg_v = self.screen_h / 128.0;
        let bg_verts: [ModalVertex; 6] = [
            ModalVertex { position: [-1.0, -1.0], tex_coords: [0.0,  bg_v] },
            ModalVertex { position: [ 1.0, -1.0], tex_coords: [bg_u, bg_v] },
            ModalVertex { position: [ 1.0,  1.0], tex_coords: [bg_u, 0.0 ] },
            ModalVertex { position: [-1.0, -1.0], tex_coords: [0.0,  bg_v] },
            ModalVertex { position: [ 1.0,  1.0], tex_coords: [bg_u, 0.0 ] },
            ModalVertex { position: [-1.0,  1.0], tex_coords: [0.0,  0.0 ] },
        ];
        let bg_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Menu Modal BG VB"),
            contents: bytemuck::cast_slice(&bg_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Create temporary vertex buffers (menu frame rate is display-limited only)
        let view_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Menu View VB"),
            contents: bytemuck::cast_slice(&view_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ui_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Menu UI VB"),
            contents: bytemuck::cast_slice(&ui_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Menu Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.05,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Full-screen background
            pass.set_pipeline(&self.modal_pipeline);
            pass.set_bind_group(0, &self.view_bg_bind_group, &[]);
            pass.set_vertex_buffer(0, bg_buf.slice(..));
            pass.draw(0..6, 0..1);

            // Modal panel
            if !view_verts.is_empty() {
                pass.set_bind_group(0, &self.modal_bg_bind_group, &[]);
                pass.set_vertex_buffer(0, view_buf.slice(..));
                pass.draw(0..view_verts.len() as u32, 0..1);
            }

            // UI (border, bevel, buttons, text)
            if !ui_verts.is_empty() {
                pass.set_pipeline(&self.ui_pipeline);
                pass.set_bind_group(0, &self.ui_bind_group, &[]);
                pass.set_vertex_buffer(0, ui_buf.slice(..));
                pass.draw(0..ui_verts.len() as u32, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    fn enter_new_game(&mut self) {
        self.phase = MenuPhase::NewGame;
        self.text_input.clear();
        self.error_msg = None;
    }

    fn enter_load_game(&mut self) {
        self.phase = MenuPhase::LoadGame;
        self.saved_worlds = crate::save_context::list_worlds();
        self.rebuild_load_buttons();
    }

    fn try_create_world(&mut self) -> MenuAction {
        let name = self.text_input.trim().to_string();
        if name.is_empty() {
            self.error_msg = Some("NAME CANNOT BE EMPTY");
            return MenuAction::None;
        }
        let world_path = format!("saves/{}", name);
        if std::path::Path::new(&world_path).join("world.toml").exists() {
            self.error_msg = Some("WORLD ALREADY EXISTS");
            return MenuAction::None;
        }
        MenuAction::StartNewGame { world_name: name }
    }

    fn update_hover(&mut self, px: f32, py: f32) {
        match self.phase {
            MenuPhase::Main => {
                for (i, btn) in main_buttons(
                    self.main_panel_x,
                    self.main_panel_y,
                    self.main_panel_w,
                    self.main_panel_h,
                )
                .iter()
                .enumerate()
                {
                    let (bx, by, bw, bh) = *btn;
                    self.main_hovered[i] =
                        px >= bx && px <= bx + bw && py >= by && py <= by + bh;
                }
            }
            MenuPhase::NewGame => {
                let layout = ng_layout(self.ng_panel_x, self.ng_panel_y, self.ng_panel_w, self.ng_panel_h);
                self.ng_create_hovered = hit(px, py, layout.create_x, layout.create_y, layout.btn_w, layout.btn_h);
                self.ng_back_hovered   = hit(px, py, layout.create_x, layout.back_y,   layout.btn_w, layout.btn_h);
            }
            MenuPhase::LoadGame => {
                for btn in &mut self.load_buttons {
                    btn.hovered = btn.contains(px, py);
                }
                self.load_back.hovered = self.load_back.contains(px, py);
            }
        }
    }

    fn handle_click(&mut self) -> MenuAction {
        match self.phase {
            MenuPhase::Main => {
                // Copy hover state to avoid borrow conflict with self.enter_*()
                let hovered = self.main_hovered;
                for (i, &hov) in hovered.iter().enumerate() {
                    if hov {
                        match i {
                            0 => self.enter_new_game(),
                            1 => self.enter_load_game(),
                            2 => return MenuAction::Quit,
                            _ => {}
                        }
                    }
                }
            }
            MenuPhase::NewGame => {
                if self.ng_create_hovered {
                    return self.try_create_world();
                }
                if self.ng_back_hovered {
                    self.phase = MenuPhase::Main;
                    self.text_input.clear();
                    self.error_msg = None;
                }
            }
            MenuPhase::LoadGame => {
                for btn in &self.load_buttons {
                    if btn.hovered {
                        return MenuAction::LoadGame {
                            world_name: btn.label.clone(),
                        };
                    }
                }
                if self.load_back.hovered {
                    self.phase = MenuPhase::Main;
                }
            }
        }
        MenuAction::None
    }

    /// Recompute the load-game dynamic button layout.
    fn rebuild_load_buttons(&mut self) {
        self.load_buttons.clear();

        let m = &self.load_game_modal;
        let btn_w = m.panel_w * MODAL_BUTTON_W_RATIO;
        // Use same proportions as normal modal buttons but scaled to this panel
        let btn_h = m.panel_h * 0.16;
        let btn_gap = m.panel_h * 0.04;
        let btn_x = (m.panel_x + (m.panel_w - btn_w) * 0.5).round();

        // World slots start at 0.28 of panel height
        let start_y = (m.panel_y + m.panel_h * 0.28).round();

        for (i, world) in self.saved_worlds.iter().take(4).enumerate() {
            let by = start_y + i as f32 * (btn_h + btn_gap);
            self.load_buttons.push(DynButton {
                label: world.clone(),
                x: btn_x,
                y: by,
                w: btn_w,
                h: btn_h,
                hovered: false,
            });
        }

        // BACK button sits below the world list (or at the same position if no worlds)
        let n = self.saved_worlds.len().min(4) as f32;
        let back_y = start_y + n * (btn_h + btn_gap) + if n > 0.0 { btn_gap } else { 0.0 };
        self.load_back = DynButton {
            label: "BACK".to_string(),
            x: btn_x,
            y: back_y,
            w: btn_w,
            h: btn_h,
            hovered: false,
        };
    }

    // ── Vertex builders ───────────────────────────────────────────────────────

    fn build_vertices(&self, view: &mut Vec<ModalVertex>, ui: &mut Vec<UiVertex>) {
        let sw = self.screen_w;
        let sh = self.screen_h;

        match self.phase {
            MenuPhase::Main => self.build_main(view, ui, sw, sh),
            MenuPhase::NewGame => self.build_new_game(view, ui, sw, sh),
            MenuPhase::LoadGame => self.build_load_game(view, ui, sw, sh),
        }
    }

    // ── Main menu ─────────────────────────────────────────────────────────────

    fn build_main(&self, view: &mut Vec<ModalVertex>, ui: &mut Vec<UiVertex>, sw: f32, sh: f32) {
        let px = self.main_panel_x;
        let py = self.main_panel_y;
        let pw = self.main_panel_w;
        let ph = self.main_panel_h;

        push_view_quad(view, px, py, pw, ph, sw, sh);
        push_panel_chrome(ui, px, py, pw, ph, sw, sh);

        // Title "CRAFT"
        let title_s = (ph * MODAL_TITLE_SCALE_RATIO).max(1.0);
        let title = "CRAFT";
        let title_w = title.len() as f32 * (5.0 + 1.0) * title_s;
        let tx = (px + (pw - title_w) * 0.5).round();
        let ty = (py + ph * MAIN_TITLE_Y_RATIO).round();
        let so = MODAL_TEXT_SHADOW_PX;
        bitmap_font::draw_text_quads(ui, title, tx + so, ty + so, title_s, title_s, MODAL_TITLE_SHADOW, sw, sh);
        bitmap_font::draw_text_quads(ui, title, tx, ty, title_s, title_s, MODAL_TITLE_COLOR, sw, sh);

        // Buttons
        let labels = ["NEW GAME", "LOAD GAME", "QUIT"];
        let btns = main_buttons(px, py, pw, ph);
        let btn_text_s = (ph * MODAL_BTN_TEXT_SCALE_RATIO).max(1.0);

        for (i, (bx, by, bw, bh)) in btns.iter().enumerate() {
            let hov = self.main_hovered[i];
            push_dyn_button(ui, labels[i], *bx, *by, *bw, *bh, hov, btn_text_s, sw, sh);
        }
    }

    // ── New World view ────────────────────────────────────────────────────────

    fn build_new_game(&self, view: &mut Vec<ModalVertex>, ui: &mut Vec<UiVertex>, sw: f32, sh: f32) {
        let l = ng_layout(self.ng_panel_x, self.ng_panel_y, self.ng_panel_w, self.ng_panel_h);

        push_view_quad(view, l.px, l.py, l.pw, l.ph, sw, sh);
        push_panel_chrome(ui, l.px, l.py, l.pw, l.ph, sw, sh);

        // Title
        let title = "NEW WORLD";
        let title_w = title.len() as f32 * (5.0 + 1.0) * l.title_s;
        let tx = (l.px + (l.pw - title_w) * 0.5).round();
        let so = MODAL_TEXT_SHADOW_PX;
        bitmap_font::draw_text_quads(ui, title, tx + so, l.title_y + so, l.title_s, l.title_s, MODAL_TITLE_SHADOW, sw, sh);
        bitmap_font::draw_text_quads(ui, title, tx, l.title_y, l.title_s, l.title_s, MODAL_TITLE_COLOR, sw, sh);

        let text_s = l.text_s;

        // Input field
        push_input_field(
            ui,
            &self.text_input,
            self.cursor_visible,
            self.error_msg,
            l.create_x, l.field_y, l.btn_w, l.btn_h,
            text_s,
            sw, sh,
        );

        // CREATE button
        push_dyn_button(ui, "CREATE", l.create_x, l.create_y, l.btn_w, l.btn_h, self.ng_create_hovered, text_s, sw, sh);
        // BACK button
        push_dyn_button(ui, "BACK",   l.create_x, l.back_y,   l.btn_w, l.btn_h, self.ng_back_hovered,   text_s, sw, sh);
    }

    // ── Load World view ───────────────────────────────────────────────────────

    fn build_load_game(&self, view: &mut Vec<ModalVertex>, ui: &mut Vec<UiVertex>, sw: f32, sh: f32) {
        let m = &self.load_game_modal;
        let px = m.panel_x;
        let py = m.panel_y;
        let pw = m.panel_w;
        let ph = m.panel_h;

        push_view_quad(view, px, py, pw, ph, sw, sh);
        push_panel_chrome(ui, px, py, pw, ph, sw, sh);

        // Title
        let title_s = m.title_scale;
        let title = "LOAD WORLD";
        let title_w = title.len() as f32 * (5.0 + 1.0) * title_s;
        let tx = (px + (pw - title_w) * 0.5).round();
        let ty = (py + m.title_y_offset).round();
        let so = MODAL_TEXT_SHADOW_PX;
        bitmap_font::draw_text_quads(ui, title, tx + so, ty + so, title_s, title_s, MODAL_TITLE_SHADOW, sw, sh);
        bitmap_font::draw_text_quads(ui, title, tx, ty, title_s, title_s, MODAL_TITLE_COLOR, sw, sh);

        let btn_text_s = (ph * MODAL_BTN_TEXT_SCALE_RATIO).max(1.0);

        if self.saved_worlds.is_empty() {
            // "NO SAVES FOUND" message
            let msg = "NO SAVES FOUND";
            let msg_w = msg.len() as f32 * (5.0 + 1.0) * btn_text_s;
            let mx = (px + (pw - msg_w) * 0.5).round();
            let my = (py + ph * 0.55).round();
            bitmap_font::draw_text_quads(ui, msg, mx + so, my + so, btn_text_s, btn_text_s, MODAL_BUTTON_SHADOW_COLOR, sw, sh);
            bitmap_font::draw_text_quads(ui, msg, mx, my, btn_text_s, btn_text_s, [0.85, 0.78, 0.55, 1.0], sw, sh);
        } else {
            for btn in &self.load_buttons {
                push_dyn_button(ui, &btn.label, btn.x, btn.y, btn.w, btn.h, btn.hovered, btn_text_s, sw, sh);
            }
        }

        // BACK button
        let back = &self.load_back;
        push_dyn_button(ui, &back.label, back.x, back.y, back.w, back.h, back.hovered, btn_text_s, sw, sh);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Free helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Character allowlist for world names.
fn is_valid_name_char(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '-' || ch == '_' || ch == ' '
}

/// Compute the main-menu panel bounds in pixel space.
fn main_panel_bounds(sw: f32, sh: f32) -> (f32, f32, f32, f32) {
    let pw = (sw * MAIN_PANEL_W_RATIO).min(sh * 0.90 * MAIN_PANEL_ASPECT);
    let ph = pw / MAIN_PANEL_ASPECT;
    let px = ((sw - pw) * 0.5).round();
    let py = ((sh - ph) * 0.5).round();
    (px, py, pw, ph)
}

/// Returns (x, y, w, h) for each of the 3 main-menu buttons.
fn main_buttons(px: f32, py: f32, pw: f32, ph: f32) -> [(f32, f32, f32, f32); 3] {
    let bw = pw * MODAL_BUTTON_W_RATIO;
    let bh = ph * MAIN_BTN_H_RATIO;
    let gap = ph * MAIN_BTN_GAP_RATIO;
    let bx = (px + (pw - bw) * 0.5).round();
    let y0 = (py + ph * MAIN_BTN_Y_RATIO).round();
    [
        (bx, y0, bw, bh),
        (bx, y0 + (bh + gap).round(), bw, bh),
        (bx, y0 + 2.0 * (bh + gap).round(), bw, bh),
    ]
}

// ── New-Game layout helpers ───────────────────────────────────────────────────

/// Aspect ratio for the New-Game panel — taller than 16:9 so the title,
/// input field, and two buttons all have comfortable breathing room.
const NG_PANEL_ASPECT: f32 = MODAL_ASPECT * 0.68; // ≈ 1.21:1

/// Compute the New-Game panel bounds.
fn ng_panel_bounds(sw: f32, sh: f32) -> (f32, f32, f32, f32) {
    let pw = (sw * MODAL_W_RATIO).min(sh * 0.90 * NG_PANEL_ASPECT);
    let ph = pw / NG_PANEL_ASPECT;
    let px = ((sw - pw) * 0.5).round();
    let py = ((sh - ph) * 0.5).round();
    (px, py, pw, ph)
}

struct NgLayout {
    px: f32, py: f32, pw: f32, ph: f32,
    title_s: f32,
    title_y: f32,
    text_s: f32,
    btn_w: f32,
    btn_h: f32,
    /// X of input field AND the two buttons (all share same left edge).
    create_x: f32,
    field_y: f32,
    create_y: f32,
    back_y: f32,
}

/// Compute the full vertical layout for the New-Game view.
///
/// Stack (top → bottom):
///   title  (ph * 0.10)
///   gap    (ph * 0.07)
///   field  (btn_h)
///   gap    (ph * 0.07)
///   CREATE (btn_h)
///   gap    (ph * 0.04)
///   BACK   (btn_h)
fn ng_layout(px: f32, py: f32, pw: f32, ph: f32) -> NgLayout {
    let title_s = (ph * MODAL_TITLE_SCALE_RATIO).max(1.0);
    let text_s  = (ph * MODAL_BTN_TEXT_SCALE_RATIO).max(1.0);
    let btn_w   = pw * MODAL_BUTTON_W_RATIO;
    let btn_h   = ph * 0.18;
    let create_x = (px + (pw - btn_w) * 0.5).round();

    let title_y   = (py + ph * 0.10).round();
    let title_bot = title_y + 7.0 * title_s;
    let field_y   = (title_bot + ph * 0.07).round();
    let create_y  = (field_y + btn_h + ph * 0.06).round();
    let back_y    = (create_y + btn_h + ph * 0.06).round();

    NgLayout { px, py, pw, ph, title_s, title_y, text_s, btn_w, btn_h, create_x, field_y, create_y, back_y }
}

/// Simple AABB hit test in pixel space.
fn hit(px: f32, py: f32, bx: f32, by: f32, bw: f32, bh: f32) -> bool {
    px >= bx && px <= bx + bw && py >= by && py <= by + bh
}

/// Push a view-tiled quad for a modal panel background.
fn push_view_quad(
    out: &mut Vec<ModalVertex>,
    px: f32, py: f32, pw: f32, ph: f32,
    sw: f32, sh: f32,
) {
    let u1 = pw / SAND_TILE_PX;
    let v1 = ph / SAND_TILE_PX;
    let cx0 = px_to_clip_x(px, sw);
    let cx1 = px_to_clip_x(px + pw, sw);
    let cy0 = px_to_clip_y(py, sh);
    let cy1 = px_to_clip_y(py + ph, sh);
    out.push(ModalVertex { position: [cx0, cy1], tex_coords: [0.0, v1] });
    out.push(ModalVertex { position: [cx1, cy1], tex_coords: [u1, v1] });
    out.push(ModalVertex { position: [cx1, cy0], tex_coords: [u1, 0.0] });
    out.push(ModalVertex { position: [cx0, cy1], tex_coords: [0.0, v1] });
    out.push(ModalVertex { position: [cx1, cy0], tex_coords: [u1, 0.0] });
    out.push(ModalVertex { position: [cx0, cy0], tex_coords: [0.0, 0.0] });
}

/// Push the border + bevel chrome around a panel.
fn push_panel_chrome(
    out: &mut Vec<UiVertex>,
    px: f32, py: f32, pw: f32, ph: f32,
    sw: f32, sh: f32,
) {
    // Outer border
    let b = MODAL_BORDER_PX;
    push_border_rect(
        out,
        px - b, py - b, px + pw + b, py + ph + b,
        px, py, px + pw, py + ph,
        MODAL_BORDER_COLOR, sw, sh,
    );

    // Bevel (lit top+left, shadow bottom+right)
    let bv = MODAL_BEVEL_PX;
    push_rect_px(out, px, py, px + pw - bv, py + bv, MODAL_BEVEL_LIGHT_COLOR, sw, sh);
    push_rect_px(out, px, py + bv, px + bv, py + ph, MODAL_BEVEL_LIGHT_COLOR, sw, sh);
    push_rect_px(out, px + bv, py + ph - bv, px + pw, py + ph, MODAL_BEVEL_SHADOW_COLOR, sw, sh);
    push_rect_px(out, px + pw - bv, py, px + pw, py + ph - bv, MODAL_BEVEL_SHADOW_COLOR, sw, sh);
}

/// Push a styled button (dynamic label string).
fn push_dyn_button(
    out: &mut Vec<UiVertex>,
    label: &str,
    bx: f32, by: f32, bw: f32, bh: f32,
    hovered: bool,
    text_scale: f32,
    sw: f32, sh: f32,
) {
    let border = MODAL_BUTTON_BORDER_PX;
    let bevel = MODAL_BUTTON_BEVEL_PX;
    let so = MODAL_TEXT_SHADOW_PX;

    // Outer border
    push_border_rect(
        out,
        bx - border, by - border, bx + bw + border, by + bh + border,
        bx, by, bx + bw, by + bh,
        MODAL_BUTTON_BORDER_COLOR, sw, sh,
    );

    // Fill
    let fill = if hovered { MODAL_BUTTON_HOVER_COLOR } else { MODAL_BUTTON_BG_COLOR };
    push_rect_px(out, bx, by, bx + bw, by + bh, fill, sw, sh);

    // Bevel (top + left only)
    push_rect_px(out, bx, by, bx + bw - bevel, by + bevel, MODAL_BUTTON_BEVEL_COLOR, sw, sh);
    push_rect_px(out, bx, by + bevel, bx + bevel, by + bh, MODAL_BUTTON_BEVEL_COLOR, sw, sh);

    // Centered label
    let char_w = (5.0 + 1.0) * text_scale;
    let label_px_w = label.len() as f32 * char_w;
    let tx = (bx + (bw - label_px_w) * 0.5).round();
    let ty = (by + (bh - 7.0 * text_scale) * 0.5).round();
    bitmap_font::draw_text_quads(out, label, tx + so, ty + so, text_scale, text_scale, MODAL_BUTTON_SHADOW_COLOR, sw, sh);
    bitmap_font::draw_text_quads(out, label, tx, ty, text_scale, text_scale, MODAL_BUTTON_TEXT_COLOR, sw, sh);
}

/// Push the text-input field for the New World view.
fn push_input_field(
    out: &mut Vec<UiVertex>,
    text: &str,
    cursor_visible: bool,
    error: Option<&str>,
    bx: f32, by: f32, bw: f32, bh: f32,
    text_scale: f32,
    sw: f32, sh: f32,
) {
    let border = MODAL_BUTTON_BORDER_PX;
    let bevel = MODAL_BUTTON_BEVEL_PX;
    let so = MODAL_TEXT_SHADOW_PX;

    // Border
    push_border_rect(
        out,
        bx - border, by - border, bx + bw + border, by + bh + border,
        bx, by, bx + bw, by + bh,
        MODAL_BUTTON_BORDER_COLOR, sw, sh,
    );

    // Fill (always normal, no hover)
    push_rect_px(out, bx, by, bx + bw, by + bh, MODAL_BUTTON_BG_COLOR, sw, sh);

    // Bevel
    push_rect_px(out, bx, by, bx + bw - bevel, by + bevel, MODAL_BUTTON_BEVEL_COLOR, sw, sh);
    push_rect_px(out, bx, by + bevel, bx + bevel, by + bh, MODAL_BUTTON_BEVEL_COLOR, sw, sh);

    // Build display string with optional blinking cursor
    let display: String = if cursor_visible {
        format!("{}|", text)
    } else {
        text.to_string()
    };

    // Clamp to field width: show last N chars that fit
    let char_w = (5.0 + 1.0) * text_scale;
    let max_chars = ((bw - 24.0) / char_w).floor() as usize;
    let display = if display.len() > max_chars {
        &display[display.len() - max_chars..]
    } else {
        &display
    };

    // Left-align text inside field
    let tx = (bx + 12.0).round();
    let ty = (by + (bh - 7.0 * text_scale) * 0.5).round();
    let text_color = if text.is_empty() {
        [0.60, 0.55, 0.38, 1.0] // dimmed placeholder color
    } else {
        MODAL_BUTTON_TEXT_COLOR
    };

    bitmap_font::draw_text_quads(out, display, tx + so, ty + so, text_scale, text_scale, MODAL_BUTTON_SHADOW_COLOR, sw, sh);
    bitmap_font::draw_text_quads(out, display, tx, ty, text_scale, text_scale, text_color, sw, sh);

    // Error message below the field
    if let Some(msg) = error {
        let err_s = text_scale * 0.8;
        let msg_w = msg.len() as f32 * (5.0 + 1.0) * err_s;
        let ex = (bx + (bw - msg_w) * 0.5).round();
        let ey = (by + bh + 4.0).round();
        bitmap_font::draw_text_quads(
            out, msg,
            ex + so, ey + so, err_s, err_s,
            MODAL_BUTTON_SHADOW_COLOR, sw, sh,
        );
        bitmap_font::draw_text_quads(
            out, msg,
            ex, ey, err_s, err_s,
            [0.95, 0.35, 0.25, 1.0], // red error color
            sw, sh,
        );
    }
}

/// Push a hollow border rectangle (4 edge strips).
fn push_border_rect(
    out: &mut Vec<UiVertex>,
    ox0: f32, oy0: f32, ox1: f32, oy1: f32,
    ix0: f32, iy0: f32, ix1: f32, iy1: f32,
    color: [f32; 4],
    sw: f32, sh: f32,
) {
    push_rect_px(out, ox0, oy0, ox1, iy0, color, sw, sh);
    push_rect_px(out, ox0, iy1, ox1, oy1, color, sw, sh);
    push_rect_px(out, ox0, iy0, ix0, iy1, color, sw, sh);
    push_rect_px(out, ix1, iy0, ox1, iy1, color, sw, sh);
}
