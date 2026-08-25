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

The no-argument scene is the streamed 1 km terrain. The benchmark scenes remain
available with `TERRAIN_SCENE=coastal_cliff`, `quarry`, `phase_d_demo`, or
`gltf` (with `TERRAIN_GLTF_PATH`). Use `TERRAIN_DATASET=/path/to/trn` to point
the terrain scene at a different tile pyramid.

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

The renderer makes `trn/tiles/0/0/0.trn` its always-resident fallback, then pages
the remaining quadtree using screen-space error, hysteresis, frustum culling,
explicit lifecycle/memory budgets, and complete-child-quad replacement. Every
tile inherits `Mesh`, but projected tiles share one grid and displace it from
per-tile elevation textures. Generate the dataset before running the renderer.
The previous C-array implementation was removed after its deprecation commit;
`.trn` is the only terrain path.

### Macro colour map

The texture-mapping worktree contains a full patch-indexing and Alpine-exemplar
synthesis pipeline. This checkout intentionally keeps only its useful output
contract: a very low-frequency colour field beneath a future authored
stone/grass material. Build the 64 x 64 map from the checked-in Swiss imagery,
then use it as the `.trn` imagery source:

```sh
python3 tools/build_terrain_macro.py
python3 tools/terrain_tiles.py build \
  --dem alps-data/swissalti3d_2025_2647-1160_0.5m.tif \
  --imagery assets/terrain_macro.png --output trn \
  --extent 2647000 1160000 2648000 1161000 --profile EPSG:2056 \
  --levels 3 --samples 65 --imagery-size 256 --gutter 1 \
  --height-encoding f32 --source "Swiss Alps 64px macro colour"
python3 tools/terrain_tiles.py validate trn
```

At this scale one source texel covers about 15.6 metres; a 1.5-texel blur removes
photographic stone/grass detail while retaining the broad rock, vegetation,
water, and snow colours. `terrain_base_color()` in `shaders/terrain.frag` is the
deliberately small insertion point for multiplying that macro tint over the
eventual close-range material blend.

The Phase 4 architecture, Rocky provenance, and runtime workflow are documented
in [`docs/phase4_terrain_quadtree.md`](docs/phase4_terrain_quadtree.md).
Terrain surface normals/detail and the HDR lighting/shadow frame are documented
in [`docs/phase5_surface_quality.md`](docs/phase5_surface_quality.md) and
[`docs/phase6_hdr_lighting_shadows.md`](docs/phase6_hdr_lighting_shadows.md).
The physical sky, atmosphere LUTs, and depth-aware aerial composite are covered
in [`docs/phase7_physical_sky.md`](docs/phase7_physical_sky.md).
Motion vectors, temporal AA, history invalidation, and adaptive exposure are in
[`docs/phase8_temporal_aa_exposure.md`](docs/phase8_temporal_aa_exposure.md).

For a large Swiss-Alps source cache, use
[`tools/download_swiss_alps.py`](tools/download_swiss_alps.py). It obtains 2 m
swissALTI3D/SWISSIMAGE source COGs; its selection policy, natural-terrain
limits, and the required conversion to `.trn` are in
[`docs/swiss_alps_bulk_dataset.md`](docs/swiss_alps_bulk_dataset.md).
Use [`tools/mask_swiss_infrastructure.py`](tools/mask_swiss_infrastructure.py)
with official road, rail, building, and optional infrastructure vectors before
building any natural-only `.trn` dataset.

## Controls

- Mouse: look around
- W/A/S/D: move relative to the direction you are looking
- Left Shift: move faster
- F3: cycle Quarry shading — legacy PBR, Phase A Default Lit, Phase B1 diffuse IBL, Phase B2 diffuse+specular IBL, and Phase C cascaded shadows
- F4: cycle TAA history-weight, rejection, motion, and clamp views
- F6: toggle logarithmic linear-depth debug view
- F7: toggle quadtree LOD colours
- F8: cycle unlit, subtle relight, and full material lighting
- F9: cycle height-field normal, placeholder roughness, and macro colour for
  terrain; static-mesh material diagnostics in benchmark scenes
- F10: cycle cascade, shadow-coordinate, visibility, bias, and raw-map views

- F11: cycle atmosphere LUT and aerial-volume debug views
- `[` / `]`: select the aerial-volume debug slice
- F12 (hold): rotate the visible sun and opposing moon through the sky
- F5: reload shaders
- Escape: quit

Shadows default to stable eight-tap rotated PCF on four 2048² cascades. Set
`TERRAIN_SHADOW_FILTER=hard`, `pcf`, or `pcss` (PCSS is opt-in), and use
`TERRAIN_SHADOW_RESOLUTION=4096` only for benchmarks; other resolutions are rejected.
