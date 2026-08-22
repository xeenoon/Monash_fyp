# Vulkan terrain renderer (C)

A deliberately small Vulkan terrain renderer with a free-fly camera,
camera-relative large-world coordinates, and infinite reversed-Z depth.

## Dependencies

- A C11 compiler and CMake 3.20+
- Vulkan loader and development headers
- SDL 3 development files
- zlib development files
- `glslc` (from the Vulkan SDK or shaderc)

On Arch Linux these are provided by `base-devel cmake vulkan-devel sdl3 shaderc zlib`.
On Ubuntu/Debian, SDL3 may need to come from a newer release or be built locally.

## Build and run

```sh
cmake -S . -B build
cmake --build build
./build/terrain_renderer
```

## Offline terrain tiles

Phase 3 provides a deterministic, versioned terrain-tile pipeline. Inputs must
already cover the same extent in the named profile:

```sh
python3 tools/terrain_tiles.py build \
  --dem dem.tif --imagery imagery.tif --output trn \
  --extent WEST SOUTH EAST NORTH --profile EPSG:2056
python3 tools/terrain_tiles.py validate trn
python3 tools/terrain_tiles.py inspect trn --output trn/tile_atlas.png
```

The binary format and validation contract are documented in
[`docs/offline_terrain_tile_format.md`](docs/offline_terrain_tile_format.md).

The renderer now loads `trn/tiles/0/0/0.trn` at startup, generates its inherited
`Mesh`, resolves its external imagery beneath `trn/`, and uploads both. Generate
the dataset before running the renderer. The previous C-array implementation is
preserved only in the deprecated `src/terrain.c` and `src/terrain.h` files. They
are excluded from every CMake target, and no active source includes or calls
them; they can be deleted together in the commit after deprecation.

## Controls

- Mouse: look around
- W/A/S/D: move relative to the direction you are looking
- Left Shift: move faster
- F6: toggle logarithmic linear-depth debug view
- F5: reload shaders
- Escape: quit
