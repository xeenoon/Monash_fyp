# Phase 5: terrain normals and surface quality

Phase 5 derives terrain normals from the elevation raster in the fragment
shader. Geometry LOD therefore changes triangle density without changing the
normal source. At a visible tile edge, the central-difference taps enter the
one-texel `.trn` gutter, so neighbouring tiles use matching samples.

The runtime uses three imagery policies through one continuous
`relight_strength` value:

- unlit map (`0.0`): the source imagery is displayed without DEM lighting;
- subtle relight (`0.35`, the default): low-strength DEM lighting is mixed over
  the imagery;
- material (`1.0`): full relighting plus close surface detail.

Press F8 to cycle those policies. Press F9 to cycle normal, slope, curvature,
height-gradient, and off. F6 still toggles depth and F7 still toggles LOD
colour. F5 reloads shaders, so surface constants can be tuned while the program
is running.

## Surface path

The order in `shaders/terrain.frag` follows the Phase 5 specification:

1. sample the sRGB imagery with its generated mip chain;
2. reconstruct a local normal from gutter-aware elevation differences;
3. fade a shared high-frequency normal map by camera distance;
4. switch steep slopes to an octant-snapped cliff projection;
5. apply two low-frequency macro-colour channels;
6. randomise detail rotation/offset per cell with explicit `textureGrad`;
7. blend to a lower detail scale in the distance.

The packed linear detail texture is generated once by `surface_detail.c`: RG is
a tangent-space normal and BA contains smooth macro noise. It is deliberately
shared by every tile. Tile origins are reduced modulo 4096 metres in double
precision on the CPU before reaching the shader. Every shader scale is a
power-of-two fraction of that period, which keeps detail continuous without
putting large projected coordinates in floats.

Height-aware multi-material blending, painted control maps, wetness, and snow
are intentionally absent: there is only one procedural surface material, so
adding those systems now would add lookup and maintenance cost without a source
data requirement.

## Terrain3D reuse boundary

Terrain3D is the primary donor. Adjacent source comments identify every adapted
shader function and its original file/range:

- `main.glsl:263-269` for the compact rotation/hash helpers;
- `main.glsl:321-345` for derivative-preserving random detiling;
- `main.glsl:439-445` for the derivative-controlled distant normal path;
- `projection.glsl` for cheap cliff projection;
- `dual_scaling.glsl:15-75` for distance blending between detail scales;
- `macro_variation.glsl:6-24` for two-channel macro colour breakup.

The engine-specific material arrays, Godot uniforms, painted-region controls,
and PBR accumulator were not copied. The deterministic packed texture generator
is local code because it replaces an external binary material asset with a
small reproducible resource. The full Terrain3D MIT notice is in
`docs/third_party_notices.md`.

## Build and test

From the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/terrain_renderer
```

`phase5_terrain_surface_quality` checks that the packed detail texture is
deterministic, non-flat in every channel, and that large positive and negative
coordinates reduce to a stable periodic phase. Building the `shaders` target
also compiles the full terrain surface path with `glslc`.
