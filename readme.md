# Vulkan terrain renderer (C)

A deliberately small Vulkan terrain renderer with a free-fly camera,
camera-relative large-world coordinates, and infinite reversed-Z depth.

## Dependencies

- A C11 compiler and CMake 3.20+
- Vulkan loader and development headers
- SDL 3 development files
- `glslc` (from the Vulkan SDK or shaderc)

On Arch Linux these are provided by `base-devel cmake vulkan-devel sdl3 shaderc`.
On Ubuntu/Debian, SDL3 may need to come from a newer release or be built locally.

## Build and run

```sh
cmake -S . -B build
cmake --build build
./build/terrain_renderer
```

## Controls

- Mouse: look around
- W/A/S/D: move relative to the direction you are looking
- Left Shift: move faster
- F6: toggle logarithmic linear-depth debug view
- F5: reload shaders
- Escape: quit
