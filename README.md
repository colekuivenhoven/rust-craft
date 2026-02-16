# RustCraft - Minecraft Clone in Rust

A Minecraft-inspired voxel game built in Rust. Features procedural world generation, destructible blocks, and inventory management.

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