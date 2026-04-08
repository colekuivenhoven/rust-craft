use crate::entities::bird::{BirdManager, create_bird_vertices, generate_bird_indices};
use crate::entities::fish::{FishManager, create_fish_vertices, generate_fish_indices};
use crate::block::{BlockType, Vertex, UiVertex, ItemCubeVertex, ModalVertex, LineVertex, create_cube_vertices, create_block_outline, create_face_vertices, create_scaled_cube_vertices, create_flat_item_vertices, create_particle_vertices, CUBE_INDICES};
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
use cgmath::{Point3, Vector3, InnerSpace, SquareMatrix};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::event::*;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

/// Modal Background Texture
const MODAL_BG_TEXTURE: &str = "assets/textures/blocks/planks.png";

mod crafting_ui;
mod input;
mod debug;
mod game_logic;
mod hud;
mod pipelines;
mod render_pass;

use crafting_ui::crafting_layout;

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
    pub(super) surface: wgpu::Surface<'static>,
    pub(super) device: wgpu::Device,
    pub(super) queue: wgpu::Queue,
    pub(super) config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    pub(super) render_pipeline: wgpu::RenderPipeline,
    pub(super) water_pipeline: wgpu::RenderPipeline,  // Separate pipeline for transparent water with depth sampling
    pub(super) transparent_pipeline: wgpu::RenderPipeline,  // Separate pipeline for semi-transparent blocks (ice) with no depth write
    pub(super) camera: Camera,
    pub(super) camera_controller: CameraController,
    pub(super) camera_uniform: CameraUniform,
    pub(super) camera_buffer: wgpu::Buffer,
    pub(super) camera_bind_group: wgpu::BindGroup,
    pub(super) projection: Projection,
    pub(super) frustum: Frustum,
    // Fog settings
    pub(super) fog_config: crate::config::FogConfig,
    pub(super) fog_buffer: wgpu::Buffer,
    pub(super) fog_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) fog_bind_group: wgpu::BindGroup,
    pub(super) world: World,
    pub(super) player: Player,
    pub(super) spawn_point: Point3<f32>,
    pub(super) water_simulation: WaterSimulation,
    pub(super) enemy_manager: EnemyManager,
    pub(super) bird_manager: BirdManager,
    pub(super) fish_manager: FishManager,
    pub(super) dropped_item_manager: DroppedItemManager,
    pub(super) particle_manager: ParticleManager,
    pub(super) last_frame: Instant,
    pub(super) mouse_pressed: bool,
    pub(super) window: Arc<Window>,
    pub(super) show_inventory: bool,
    // UI rendering
    pub(super) ui_pipeline: wgpu::RenderPipeline,
    pub(super) ui_bind_group: wgpu::BindGroup,
    pub(super) ui_uniform_buffer: wgpu::Buffer,
    pub(super) crosshair_vertex_buffer: wgpu::Buffer,
    pub(super) hud_vertex_buffer: wgpu::Buffer,
    pub(super) hud_vertex_count: u32,
    pub(super) hud_text_vertex_buffer: wgpu::Buffer,
    pub(super) hud_text_vertex_count: u32,
    pub(super) item_cube_pipeline: wgpu::RenderPipeline,
    pub(super) item_cube_bind_group: wgpu::BindGroup,
    pub(super) item_cube_vertex_buffer: wgpu::Buffer,
    pub(super) item_cube_vertex_count: u32,
    // Block outline rendering
    pub(super) outline_pipeline: wgpu::RenderPipeline,
    pub(super) targeted_block: Option<(i32, i32, i32)>,
    // Chunk outline rendering (debug)
    pub(super) chunk_outline_pipeline: wgpu::RenderPipeline,
    // Mouse capture state
    pub(super) mouse_captured: bool,
    // Depth textures: one for rendering, one for sampling in water shader
    pub(super) depth_texture: wgpu::Texture,
    pub(super) depth_view: wgpu::TextureView,
    pub(super) depth_copy_texture: wgpu::Texture,  // Copy of depth buffer for water shader to sample
    pub(super) depth_copy_view: wgpu::TextureView,
    pub(super) depth_sampler: wgpu::Sampler,
    pub(super) water_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) water_bind_group: wgpu::BindGroup,
    pub(super) water_time_buffer: wgpu::Buffer,
    // FPS tracking
    pub(super) fps: f32,
    pub(super) fps_frame_count: u32,
    pub(super) fps_timer: f32,
    // Debug mode
    pub(super) show_chunk_outlines: bool,
    pub(super) noclip_mode: bool,
    pub(super) show_enemy_hitboxes: bool,
    pub(super) smooth_lighting: bool,
    pub(super) hud_enabled: bool,
    pub(super) frozen_stone_ceiling: bool,

    // Underwater effect (Post-Processing)
    pub(super) camera_underwater: bool,
    pub(super) underwater_pipeline: wgpu::RenderPipeline,
    pub(super) underwater_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) underwater_bind_group: wgpu::BindGroup,
    pub(super) underwater_uniform_buffer: wgpu::Buffer,
    // Off-screen textures for post-processing
    pub(super) scene_texture: wgpu::Texture,
    pub(super) scene_texture_view: wgpu::TextureView,
    pub(super) post_process_texture: wgpu::Texture,
    pub(super) post_process_texture_view: wgpu::TextureView,
    pub(super) scene_sampler: wgpu::Sampler,
    pub(super) start_time: Instant,

    // Motion blur (Post-Processing)
    pub(super) motion_blur_pipeline: wgpu::RenderPipeline,
    pub(super) motion_blur_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) motion_blur_bind_group: wgpu::BindGroup,
    pub(super) motion_blur_uniform_buffer: wgpu::Buffer,

    // Bloom (Post-Processing)
    pub(super) bloom_emissive_texture: wgpu::Texture,          // full-res emissive render target (with depth test)
    pub(super) bloom_emissive_texture_view: wgpu::TextureView,
    pub(super) bloom_texture_a: wgpu::Texture,                 // quarter-res blur ping-pong A
    pub(super) bloom_texture_a_view: wgpu::TextureView,
    pub(super) bloom_texture_b: wgpu::Texture,                 // quarter-res blur ping-pong B
    pub(super) bloom_texture_b_view: wgpu::TextureView,
    pub(super) bloom_emissive_pipeline: wgpu::RenderPipeline,  // renders only emissive blocks (with depth test)
    pub(super) bloom_downsample_pipeline: wgpu::RenderPipeline, // full-res → quarter-res downsample
    pub(super) bloom_blur_h_pipeline: wgpu::RenderPipeline,
    pub(super) bloom_blur_v_pipeline: wgpu::RenderPipeline,
    pub(super) bloom_composite_pipeline: wgpu::RenderPipeline,
    pub(super) bloom_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) bloom_emissive_bind_group: wgpu::BindGroup,     // reads bloom_emissive_texture (for downsample)
    pub(super) bloom_a_bind_group: wgpu::BindGroup,            // reads bloom_texture_a
    pub(super) bloom_b_bind_group: wgpu::BindGroup,            // reads bloom_texture_b

    // Damage flash effect (Post-Processing)
    pub(super) damage_flash_intensity: f32,
    pub(super) damage_pipeline: wgpu::RenderPipeline,
    pub(super) damage_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) damage_bind_group: wgpu::BindGroup,       // reads post_process_texture
    pub(super) damage_bind_group_alt: wgpu::BindGroup,   // reads scene_texture (when combined with underwater)
    pub(super) damage_uniform_buffer: wgpu::Buffer,

    // Texture atlas
    pub(super) texture_atlas: TextureAtlas,
    // Breaking mechanics
    pub(super) breaking_pipeline: wgpu::RenderPipeline,
    pub(super) breaking_state: Option<BreakingState>,
    pub(super) left_mouse_held: bool,
    // Melee combat
    pub(super) hit_cooldown: f32,
    pub(super) hit_indicator_timer: f32,
    pub(super) crosshair_vertex_count: u32,
    // Cached GPU buffers for chunks - avoids recreating every frame
    pub(super) chunk_buffers: HashMap<(i32, i32), ChunkBuffers>,
    // Cloud rendering
    pub(super) cloud_manager: crate::clouds::CloudManager,
    pub(super) cloud_pipeline: wgpu::RenderPipeline,
    pub(super) cloud_vertex_buffer: wgpu::Buffer,
    pub(super) cloud_index_buffer: wgpu::Buffer,
    pub(super) cloud_index_count: u32,
    pub(super) cloud_drift_buffer: wgpu::Buffer,
    pub(super) cloud_drift_bind_group: wgpu::BindGroup,

    // ── Sky / Day-Night Cycle ─────────────────────────────────────────────
    pub(super) sky_config: crate::config::SkyConfig,
    pub(super) sky_pipeline: wgpu::RenderPipeline,
    pub(super) sky_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) sky_bind_group: wgpu::BindGroup,
    pub(super) sky_uniform_buffer: wgpu::Buffer,
    // Sun uniform (shared with block shaders via bind group 3)
    pub(super) sun_buffer: wgpu::Buffer,
    pub(super) sun_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) sun_bind_group: wgpu::BindGroup,
    // Shadow map
    pub(super) shadow_map_texture: wgpu::Texture,
    pub(super) shadow_map_view: wgpu::TextureView,
    pub(super) shadow_map_sampler: wgpu::Sampler,
    pub(super) shadow_map_pipeline: wgpu::RenderPipeline,
    pub(super) shadow_map_camera_buffer: wgpu::Buffer,
    pub(super) shadow_map_camera_bind_group: wgpu::BindGroup,

    // ── Pause / Modal ─────────────────────────────────────────────────────
    pub paused: bool,
    pub(super) pause_modal: Modal,
    pub(super) crafting_modal: Modal,
    /// Current cursor position in pixel space (set via CursorMoved events)
    pub(super) cursor_pos_px: (f32, f32),

    // Pause-background blur pipeline (reads scene_texture)
    pub(super) pause_blur_pipeline:    wgpu::RenderPipeline,
    pub(super) pause_blur_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) pause_blur_bind_group:  wgpu::BindGroup,

    // Modal sand-texture pipeline
    pub(super) modal_sand_pipeline:    wgpu::RenderPipeline,
    pub(super) modal_sand_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) modal_sand_bind_group:  wgpu::BindGroup,
    pub(super) modal_sand_texture:     wgpu::Texture,
    pub(super) modal_sand_sampler:     wgpu::Sampler,

    // Modal GPU draw buffers (rebuilt each frame while paused)
    pub(super) modal_sand_vertex_buffer: wgpu::Buffer,
    pub(super) modal_ui_vertex_buffer:   wgpu::Buffer,
    pub(super) modal_ui_vertex_count:    u32,

    // Dropped-item hover (index into dropped_item_manager.items the crosshair is over)
    pub(super) hovered_dropped_item: Option<usize>,

    // ── Crafting Table UI ─────────────────────────────────────────────────
    pub(super) crafting_ui_open: bool,
    pub(super) hovered_crafting_table: bool,
    pub(super) crafting_grid: CraftingGrid,
    pub(super) crafting_output: Option<(BlockType, f32)>,
    /// Item currently held on the cursor (picked up from grid or inventory)
    pub(super) crafting_held: Option<(BlockType, f32)>,
    pub(super) crafting_hovered_grid: Option<(usize, usize)>,
    pub(super) crafting_hovered_inv: Option<usize>,
    pub(super) crafting_hovered_output: bool,
    /// Per-slot selected pickup quantity for the inventory row in the crafting UI
    pub(super) crafting_inv_qty: [f32; 9],

    // Audio
    pub(super) audio: Option<AudioManager>,
    /// Looping walking-sound sink; volume is faded in/out each frame.
    pub(super) walk_sink:   Option<rodio::Sink>,
    pub(super) walk_volume: f32,
}

impl State {

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

}
