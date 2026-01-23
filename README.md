# Craft - Minecraft Clone in Rust

A Minecraft-inspired voxel game built from scratch in Rust using WGPU for rendering. Features procedural world generation, destructible blocks, inventory management, water simulation, enemies with AI, and a crafting system.

## Features

### Core Gameplay
- **Procedural World Generation**: Infinite terrain generated using 3D Perlin noise with multiple octaves
- **Destructible Blocks**: Break and place blocks in the world
- **Inventory System**: 9-slot hotbar with item stacking (max 64 per stack)
- **Crafting System**: Craft items using recipes
- **Water Simulation**: Dynamic water that flows downward and spreads horizontally
- **Enemy AI**: Hostile enemies that spawn, wander, chase, and attack the player

### Block Types
- **Grass**: Surface blocks in high-altitude areas
- **Dirt**: Sub-surface blocks below grass
- **Stone**: Deep underground blocks
- **Sand**: Coastal and underwater blocks
- **Water**: Dynamic flowing water (sea level at Y=28)
- **Wood**: Tree trunks from procedurally generated trees
- **Leaves**: Tree foliage
- **Planks**: Crafted from wood
- **Cobblestone**: Crafted from stone

### World Generation
- Multi-octave Perlin noise for realistic terrain
- Automatic tree generation with trunks and leaves
- Water generation at sea level
- Dynamic chunk loading based on player position
- Render distance: 4 chunks in each direction

### Enemy System
- Enemies spawn around the player (20-40 blocks away)
- AI states: Idle, Wandering, Chasing, Attacking
- Enemies chase player when within 15 blocks
- Attack when within 2 blocks
- Color-coded by state (purple=idle, orange=chasing, red=attacking)
- Maximum 10 enemies active at once

## Controls

### Movement
- **W/A/S/D**: Move forward/left/backward/right
- **Space**: Move up (fly)
- **Left Shift**: Move down (fly)
- **Mouse**: Look around

### Interaction
- **Left Click**: Break block (adds to inventory), or recapture mouse if released
- **Right Click**: Place block from selected hotbar slot
- **E**: Toggle inventory display
- **C**: Toggle crafting menu
- **1-9**: Select hotbar slot (or craft recipe when crafting menu is open)
- **Escape**: Release mouse cursor (allows window resizing)

## Installation & Running

### Prerequisites
- Rust 1.70 or higher (install from https://rustup.rs)
- Visual Studio Build Tools with "Desktop development with C++" workload
- Graphics card with Vulkan, DirectX 12, or Metal support

### Build and Run

**Important for Windows users**: You must run the build commands from a **Visual Studio Developer Command Prompt** or **Developer PowerShell for VS** to ensure the correct MSVC linker is used. If you have Git for Windows installed, its `link.exe` may conflict with MSVC's linker.

#### Option 1: Use Visual Studio Developer Command Prompt (Recommended)
1. Open "Developer Command Prompt for VS" or "Developer PowerShell for VS" from the Start menu
2. Navigate to the project directory and build:
```bash
cd Craft
cargo build --release
cargo run --release
```

#### Option 2: Use x64 Native Tools Command Prompt
1. Open "x64 Native Tools Command Prompt for VS" from the Start menu
2. Navigate to the project and build as above

#### Option 3: Fix PATH temporarily
If you must use a regular terminal, temporarily adjust PATH to prioritize MSVC:
```powershell
$env:PATH = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64;" + $env:PATH
cargo build --release
```
(Adjust the path to match your Visual Studio installation)

### Running the Game

After building successfully:
```bash
cargo run --release
```

For development with debug symbols (slower but better error messages):
```bash
cargo run
```

## Gameplay Guide

### Getting Started
1. Launch the game - you'll spawn in a procedurally generated world at Y=35
2. Look around with your mouse to orient yourself
3. You start with 10 wood, 10 stone, and 20 dirt blocks in your inventory

### Breaking Blocks
1. Look at a block within reach (5 blocks)
2. Left-click to break it
3. The block will be added to your inventory automatically

### Placing Blocks
1. Select a hotbar slot (1-9 keys) containing blocks
2. Look at an existing block
3. Right-click to place your block adjacent to the targeted block

### Inventory Management
1. Press **E** to view your inventory in the console
2. The selected slot is marked with **>**
3. Press 1-9 to switch between hotbar slots
4. Each slot can hold up to 64 of the same block type

### Crafting
1. Press **C** to open the crafting menu
2. Available recipes are displayed in the console
3. Press number keys (1-9) to craft the corresponding recipe
4. Ingredients are automatically consumed from your inventory

#### Available Recipes
- **1 Wood → 4 Planks**
- **4 Planks → 1 Wood**
- **1 Stone → 1 Cobblestone**
- **1 Cobblestone → 1 Stone**
- **2 Dirt + 1 Grass → 3 Grass**
- **4 Sand → 1 Stone** (smelting)

### Surviving Enemies
- Enemies (purple cubes) spawn periodically around you
- They wander randomly when idle
- Turn orange when chasing you (within 15 blocks)
- Turn red when attacking (within 2 blocks)
- You have 100 health points
- Game over message appears when health reaches 0

### Water Mechanics
- Water exists at sea level (Y=28 and below)
- Breaking blocks near water may cause it to flow
- Water flows downward with gravity
- Water spreads horizontally to adjacent air blocks
- Water blocks are semi-transparent blue

## Technical Architecture

### Project Structure
```
src/
├── main.rs           # Entry point and event loop
├── renderer.rs       # WGPU rendering and game state
├── shader.wgsl       # Vertex and fragment shaders
├── block.rs          # Block types and vertex generation
├── chunk.rs          # Chunk management and mesh building
├── world.rs          # World management and chunk loading
├── camera.rs         # Camera and projection systems
├── player.rs         # Player state and raycasting
├── inventory.rs      # Inventory management
├── crafting.rs       # Crafting system and recipes
├── water.rs          # Water simulation
├── enemy.rs          # Enemy AI and management
└── texture.rs        # Texture system (placeholder)
```

### Key Systems

#### Chunk System
- World divided into 16×64×16 chunks
- Chunks generated using Perlin noise
- Greedy meshing for visible faces only
- Automatic chunk loading/unloading

#### Rendering Pipeline
- WGPU-based rendering (supports Vulkan/DX12/Metal)
- Vertex colors for blocks (no textures)
- Directional lighting with ambient and diffuse components
- Distance-based fog for atmosphere
- Depth testing and back-face culling

#### Performance Optimizations
- Only visible block faces are rendered (no internal geometry)
- Chunks only rebuild meshes when modified
- Far chunks automatically unloaded
- Release builds use LTO and optimization level 3

## Performance Notes

- **Debug builds**: May run slowly due to lack of optimizations
- **Release builds**: Significantly faster, recommended for gameplay
- **Render distance**: Adjustable in code (`World::new(4)` in renderer.rs)
- **Enemy cap**: Limited to 10 for performance (adjustable in code)

## Troubleshooting

### Game runs slowly
- Make sure you're using release mode: `cargo run --release`
- Reduce render distance in [renderer.rs:279](src/renderer.rs#L279)
- Reduce max enemies in [renderer.rs:280](src/renderer.rs#L280)

### Graphics errors
- Ensure your GPU drivers are up to date
- Check that your system supports Vulkan, DirectX 12, or Metal
- Try updating WGPU dependency in Cargo.toml

### Build errors
- Update Rust: `rustup update`
- Clean build directory: `cargo clean`
- Rebuild: `cargo build --release`

### Linker errors on Windows (`link: extra operand`)
This error occurs when Git for Windows' `link.exe` is found before MSVC's `link.exe` in your PATH.

**Solutions:**
1. **Use the included build script**: Double-click `build.bat` which sets up the proper MSVC environment
2. **Use Developer Command Prompt**: Open "Developer Command Prompt for VS 2022" from Start menu, then navigate to the project and run `cargo build --release`
3. **Temporarily modify PATH** in PowerShell before building:
   ```powershell
   $env:PATH = ($env:PATH -split ';' | Where-Object { $_ -notlike '*Git*' }) -join ';'
   cargo build --release
   ```

## Future Enhancements

Potential features for expansion:
- Texture mapping for blocks
- Sound effects and music
- Multiplayer support
- More complex crafting recipes
- Different biomes
- Day/night cycle
- Mining tools with durability
- Different enemy types
- Saving/loading worlds
- GUI for inventory and crafting

## Dependencies

- **wgpu** (23): Modern graphics API abstraction
- **winit** (0.30): Window creation and event handling
- **cgmath** (0.18): 3D math library
- **noise** (0.9): Perlin noise generation
- **rand** (0.8): Random number generation
- **bytemuck** (1.14): Safe byte casting
- **pollster** (0.4): Async runtime for wgpu initialization
- **env_logger** (0.11): Logging for debug output

## License

This project is created for educational purposes.

## Credits

Built with Rust and WGPU. Inspired by Minecraft.
