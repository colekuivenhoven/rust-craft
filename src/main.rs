mod bird;
mod block;
mod camera;
mod chunk;
mod chunk_loader;
mod chunk_storage;
mod clouds;
mod config;
mod crafting;
mod dropped_item;
mod enemy;
mod fish;
mod bitmap_font;
mod inventory;
mod lighting;
mod particle;
mod player;
mod renderer;
mod texture;
mod water;
mod world;

use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

fn main() {
    // Initialize rayon thread pool with larger stack size for chunk generation
    // Chunks have ~256KB of stack-allocated arrays, so we need larger stacks
    rayon::ThreadPoolBuilder::new()
        .stack_size(4 * 1024 * 1024) // 4MB stack per thread
        .build_global()
        .expect("Failed to initialize rayon thread pool");

    // Initialize logging with info level by default
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info,wgpu=warn"))
        .format_timestamp_millis()
        .init();

    // Set up panic hook to log panics
    std::panic::set_hook(Box::new(|panic_info| {
        log::error!("PANIC: {}", panic_info);
        eprintln!("PANIC: {}", panic_info);
    }));

    log::info!("Starting Craft...");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    if let Err(e) = event_loop.run_app(&mut app) {
        log::error!("Event loop error: {}", e);
    }
}

#[derive(Default)]
struct App {
    state: Option<renderer::State>,
    window: Option<Arc<Window>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("Craft - Minecraft Clone")
                .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720));

            let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
            self.window = Some(window.clone());
            let mut state = pollster::block_on(renderer::State::new(window));
            state.capture_mouse();
            self.state = Some(state);
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(state) = self.state.as_mut() {
            if let DeviceEvent::MouseMotion { delta } = event {
                state.process_mouse(delta.0, delta.1);
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        // Handle mouse recapture before other input processing
        // This must be checked first because state.input() consumes mouse events
        if let WindowEvent::MouseInput {
            state: ElementState::Pressed,
            button: MouseButton::Left,
            ..
        } = &event
        {
            if !state.is_mouse_captured() {
                state.capture_mouse();
                return; // Don't process this click as game input
            }
        }

        if state.input(&event) {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => {
                // Release mouse cursor instead of closing game
                state.release_mouse();
            }
            WindowEvent::Resized(physical_size) => {
                state.resize(physical_size);
            }
            WindowEvent::RedrawRequested => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Save all modified chunks before exiting
        if let Some(state) = &mut self.state {
            log::info!("Saving world...");
            state.save_world();
        }
    }
}
