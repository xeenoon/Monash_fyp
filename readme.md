# Vulkan terrain and dungeon renderer (C)

A deliberately small Vulkan terrain renderer with a free-fly camera,
camera-relative large-world coordinates, and infinite reversed-Z depth.
It also contains a playable top-down dungeon scene using the same PBR, HDR,
shadow, and temporal rendering path.

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

Run the dungeon explorer with:

```sh
TERRAIN_SCENE=dungeon ./build/terrain_renderer
```

By default this generates a procedural cave (winding corridors, chambers,
puddles) from `DUNGEON_SEED` (a `uint32_t`, default 1) -- try a few different
seeds:

```sh
TERRAIN_SCENE=dungeon DUNGEON_SEED=7 ./build/terrain_renderer
```

`DUNGEON_MAP=<path>` switches to the legacy ASCII-grid frontend instead (e.g.
the checked-in `assets/dungeons/example.map`), for hand-authored layouts or
debugging:

```sh
TERRAIN_SCENE=dungeon DUNGEON_MAP=assets/dungeons/example.map ./build/terrain_renderer
```

The player follows a temporary cube with a fixed top-down perspective camera,
sliding along swept-circle-vs-wall collision. Both frontends rasterize into
the same occupancy field and marching-squares pipeline; nothing past that
point knows or cares which one produced the level. See
[`docs/dungeon_architecture.md`](docs/dungeon_architecture.md).

The three CC0 dungeon PBR materials are checked in for offline use. Their
source pages, physical scales, and checksums are recorded in
[`textures/dungeon/manifest.json`](textures/dungeon/manifest.json). Reproduce
them with `python3 tools/download_dungeon_materials.py`.

The old example-synthesis algorithm running through the modern renderer is
preserved in `trn-golden/` and is the default. Its RGB is the exact
`trn.golden-albedo` dataset from commit `145a86b` on branch
`modern-render-old-texture-sampling`: the tagged `golden/91746cf` albedo at 1:1
coverage with procedural synthesis only for overflow. The PNG alpha channel is
the newer Alpine rock/grass classification and does not alter golden colour.
Build it with:

```sh
cmake -S . -B build -DTERRAIN_GOLDEN=ON
cmake --build build
./build/terrain_renderer
```

Configure with `-DTERRAIN_GOLDEN=OFF` to return to `trn/`. An explicit
`-DTERRAIN_DATASET_DIR=/path/to/trn` takes precedence over either choice, and
the runtime `TERRAIN_DATASET` environment variable still overrides the build.

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

The renderer makes the selected dataset's `tiles/0/0/0.trn` its always-resident fallback, then pages
the remaining quadtree using screen-space error, hysteresis, frustum culling,
explicit lifecycle/memory budgets, and complete-child-quad replacement. Every
tile inherits `Mesh`, but projected tiles share one grid and displace it from
per-tile elevation textures. Generate the dataset before running the renderer.
The previous C-array implementation was removed after its deprecation commit;
`.trn` is the only terrain path.

### Macro colour map

The golden dataset owns runtime RGB. The 64 x 64 Swiss macro builder remains as
the source of its labelled rock/grass alpha and as the non-golden fallback.
Rebuild that classifier from the checked-in Swiss imagery with:

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

At this scale one source texel covers about 15.6 metres. The builder labels
grass and rock first, blurs those two sample populations independently, and
fills water/snow/unclassified holes from the nearest accepted material texel.
`assets/terrain_macro_labels.png` records the decision as R=rock, G=grass,
B=rejected, so water can be audited without allowing it to tint the material.
The macro PNG's alpha channel carries the resulting blurred grass coverage.
`trn-golden/imagery` combines that alpha with untouched golden RGB, preventing
the blurred classifier colour from entering the rendered material.

### Terrain micro materials

Nine CC0 rock scans are represented in `textures/manifest.json`: Rock 01,
Rock 06, Rock 2, Rock055, Rock040, Marble Cliff 01/03, Rocky Mountain Cliff
Face, and Layered Cliff Rock. Six Alpine-matched grass/vegetated-ground sets
are represented in `textures/grass_manifest.json`, and six snow sets in
`textures/snow_manifest.json`. Fetch all three source banks and generate the
compact runtime maps with:

```sh
python3 tools/download_terrain_materials.py
python3 tools/download_terrain_materials.py \
  --manifest textures/grass_manifest.json \
  --source textures/grass-source
python3 tools/download_terrain_materials.py \
  --manifest textures/snow_manifest.json \
  --source textures/snow-source
python3 tools/build_terrain_micro_atlas.py
```

The 5 x 5 runtime atlases preserve nine rock, six grass, and six snow
materials and add wrap gutters. Linear material albedo supplies normalized
achromatic structure for the coarse layer, while fine-scale luminance is
high-pass filtered and packed separately. Original terrain imagery remains the
sole RGB source.
Within each bank
the terrain shader stochastically blends nearby samples across 32 m regions;
the macro alpha then blends the material banks at Alpine scale. A colour and
slope test separates bright neutral snow from rock/grass. Neutral material
structure is layered at a coarse macro scale and true micro scale. A
restrained coarse normal breaks up formations; roughness and AO remain
true-scale. Mip filtering and a far-horizon fade suppress sub-pixel detail.
Steep terrain uses a world-continuous metric triplanar projection of the same
authored scans; their metre scale and phase remain continuous across tile and
LOD boundaries. Top-down imagery is filtered for classification, then removed
from side-facing colour instead of being magnified into cliff-long streaks.
Terrain mode starts at fixed exposure so the surrounding dark sky does not
meter the material to white; `E` still toggles adaptation.

For an arbitrary non-periodic source, `tools/make_texture_tileable.py` moves the
old boundary to the centre and repairs it with a multiband cross blend. Run it
with identical options for each registered PBR channel (and `--normal-map` for
the normal map) before atlas generation.

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

Dungeon:

- W/A/S/D: move the cube relative to the top-down camera
- Escape: quit

Terrain and inspection scenes:

- Mouse: look around
- W/A/S/D: move relative to the direction you are looking
- Left Shift: move faster
- F3: cycle Quarry shading — legacy PBR, Phase A Default Lit, Phase B1 diffuse IBL, Phase B2 diffuse+specular IBL, and Phase C cascaded shadows
- F4: cycle TAA history-weight, rejection, motion, and clamp views
- F6: toggle logarithmic linear-depth debug view
- F7: toggle quadtree LOD colours
- F8: cycle unlit, subtle relight, and full material lighting
- F9: cycle material normal, authored roughness, macro colour, and the
  red=rock/green=grass Alpine blend for terrain; static-mesh material
  diagnostics in benchmark scenes
- F10: cycle cascade, shadow-coordinate, visibility, bias, and raw-map views

- F11: cycle atmosphere LUT and aerial-volume debug views
- `[` / `]`: select the aerial-volume debug slice
- F12 (hold): rotate the visible sun and opposing moon through the sky
- F5: reload shaders
- E: toggle auto-exposure (eye adaptation) on/off
- Escape: quit

Set `TERRAIN_FREEZE_CAMERA=1` when capturing a replayed dump viewpoint so
window focus and pointer motion cannot move the camera between comparisons.
Regular builds include shader-dump support: press `X` to append a shader capture
and `C` to clear it. Memory-constrained builds can opt out with
`-DDEBUG_SHADER_DUMP=OFF`. Inspect captures as a fullscreen, local pixel view with:

```sh
python3 tools/shader_dump_viewer.py debug_dumps/shader_dump.txt
```

The viewer maps any dumped fields to RGB, offers shader-specific presets and
black/white single-channel inspection, and labels the selected channel meanings
beside the image. It is served only on `127.0.0.1` and has no Python package
dependencies.
`TERRAIN_START_DEBUG_VIEW=22` boots into the final colour-preserving macro layer;
`23` shows the X-side/top/Z-side triplanar weights.

Shadows default to stable eight-tap rotated PCF on four 2048² cascades. Set
`TERRAIN_SHADOW_FILTER=hard`, `pcf`, or `pcss` (PCSS is opt-in), and use
`TERRAIN_SHADOW_RESOLUTION=4096` only for benchmarks; other resolutions are rejected.

Normal builds do not contain the red texture-stretch highlight. To compile the
diagnostic explicitly, configure a separate build and launch it with the
runtime switch:

```sh
cmake -S . -B build-stretch -DTERRAIN_STRETCH_OVERLAY=ON
cmake --build build-stretch -j
./build-stretch/terrain_renderer --show-texture-stretch
```

The overlay marks terrain where top-down projection stretches by 1.5x or more,
matching `tools/detect_texture_stretch.py`. Tune it with
`TERRAIN_STRETCH_THRESHOLD` (default `1.5`) and `TERRAIN_STRETCH_OPACITY`
(default `0.25`). Passing `--show-texture-stretch` to an ordinary build prints
an explanatory warning and cannot alter terrain colour.
