# Phase 6: HDR lighting and terrain shadows

Phase 6 keeps terrain lighting linear and HDR until a dedicated display pass.
Each frame records these passes in order:

1. four depth-only sun-shadow cascades;
2. terrain into `R16G16B16A16_SFLOAT` plus reversed-Z main depth;
3. fixed-exposure ACES tone mapping into the sRGB swapchain.

The swapchain conversion supplies the linear-to-sRGB transfer exactly once.
The terrain fragment shader never tone-maps, and the display shader bypasses
ACES for diagnostic colours. Motion and adaptive exposure remain deferred to
their scheduled phases.

## Terrain lighting

`shaders/terrain.frag` contains the intentionally small dielectric subset:

- GGX normal distribution;
- correlated Smith visibility with a numeric denominator guard;
- Schlick Fresnel with `F0 = 0.04`;
- zero metalness and a high terrain roughness;
- direct sun radiance plus temporary hemisphere irradiance;
- ambient occlusion input, currently `1.0` until material/control data exists.

F8 still drives one continuous `relight_strength`. Unlit mode displays map
imagery, subtle mode mixes low-strength HDR lighting over it, and material mode
uses the full BRDF plus Phase 5 surface detail. Phase 7 replaces the temporary
hemisphere term and fixed sun colour with physical sky irradiance and
transmittance.

## Cascaded shadows

`shadow_cascade.c` builds four camera-relative matrices with 120, 350, 1000,
and 3000 metre splits. Each frustum slice is enclosed by a light-space sphere,
the radius is quantised, and its XY centre is snapped to the 2048-pixel shadow
grid. The snap is calculated from the absolute double-precision camera position
before being reduced back to a small float matrix. This keeps the grid fixed in
the world without sending projected-world floats to the GPU.

The shadow target is a four-layer filterable depth array. `shadow.vert`
repeats the exact elevation displacement used by the colour pass. The terrain
shader selects the smallest cascade containing the receiver, applies a
slope-aware normal offset, filters an eight-point rotated Vogel disk, and
blends across the inner 10% of a cascade into its successor. Raster depth bias
and receiver normal bias are separate controls in `FrameUniforms`.

The shadow pass currently reuses the main terrain selection. This guarantees
matching receiver/caster geometry and avoids drawing overlapping resident
parents. A dedicated shadow-caster quadtree traversal should only be added when
the streaming system can select off-camera casters without publishing
parent/child overlap.

## Controls and diagnostics

- F10 cycles cascade index, shadow coordinates, filtered visibility, receiver
  bias, raw depth, and off.
- F8 cycles the imagery/relighting policy.
- F9 retains the Phase 5 surface diagnostics.
- F5 reloads the terrain, shadow, and tone-map pipelines together.

The cascade view uses red, green, blue, and yellow from near to far. The
visibility view is white when lit and black when shadowed. Raw depth uses the
ordinary (non-reversed) shadow convention: zero is near the light and one is
far. The main camera remains infinite reversed-Z.

## Wicked Engine reuse boundary

Adjacent source comments identify the adapted donor code and why it was kept:

- `brdf.hlsli:8-64`: isotropic GGX and correlated Smith visibility;
- `lightingHF.hlsli:41-55`: direct/indirect lighting separation;
- `lightingHF.hlsli:87-126`: smallest-cascade selection and 10% edge blend;
- `shadowHF.hlsli`: the eight-point subset of its Vogel disk;
- `wiRenderer.cpp:2936-3060`: frustum-sphere fitting and texel snapping;
- `tonemapCS.hlsl:8-43`: the fitted ACES matrices and rational curve.

Wicked's `Surface`, bindless tables, half types, transparent shadows, PCSS,
extra material lobes, bloom, colour grading, and adaptive luminance were not
copied. The full Wicked Engine MIT notice is in `third_party_notices.md`.

## Build and validation

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation ./build/terrain_renderer
```

`phase6_hdr_lighting_and_shadows` checks finite nested cascade matrices,
view-axis coverage, invalid split rejection, and world-grid stability at large
coordinates. Building the shader target compiles the terrain BRDF, shadow-map,
and final display passes with `glslc`.
