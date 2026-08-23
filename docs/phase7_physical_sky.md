# Phase 7: physical sky and aerial perspective

Phase 7 replaces the scene clear colour and Phase 6 hemisphere constant with a
single physical atmosphere model shared by the sky, sun, terrain lighting, and
distance haze. Atmosphere distances and extinction coefficients are in
kilometres; terrain and camera-relative geometry remain in metres.

## Frame workflow

1. The transmittance LUT (256x64) is generated on first use and when shaders or
   atmosphere parameters are invalidated.
2. The unit-sun multiple-scattering LUT (32x32) is generated after
   transmittance. Because it is a transfer function, moving the sun does not
   invalidate it.
3. The sun/camera-dependent sky-view LUT (192x108) is generated each frame.
4. RGB scattering and RGB transmittance aerial volumes (32x32x32 each) are
   generated each frame with squared depth slices out to 128 km.
5. Terrain samples the transmittance LUT for direct sunlight and the sky-view
   LUT for diffuse sky light, then renders into Phase 6's HDR target. Diffuse
   fill uses sky-facing horizon/zenith samples rather than treating one normal
   ray as irradiance; this keeps steep shadowed slopes readable at sunset.
6. The final pass samples reversed-Z scene depth. Empty pixels receive sky plus
   an atmosphere-attenuated sun disk. Opaque pixels receive
   `surface_hdr * transmittance + scattering` from the depth-clamped volume.
7. ACES runs once, after terrain and atmosphere are combined.

All storage images remain in `GENERAL`; explicit compute/fragment barriers make
writes visible without layout churn. Compute dispatches round up and every
entry point checks its target bounds.

## Donor map

The adjacent source comments identify every material port. The main donor is
Wicked Engine's compact integration of Sebastien Hillaire's Unreal Sky
Atmosphere reference:

| Local code | Adapted from | Retained |
|---|---|---|
| `shaders/atmosphere_common.glsl` | Wicked `skyAtmosphere.hlsli`; Unreal `SkyAtmosphereCommon.hlsl` and `RenderSkyRayMarching.hlsl` | LUT mappings, medium densities, sphere tests, phase functions, finite-segment integral |
| `atmosphere_transmittance.comp` | Wicked/Unreal transmittance pass | Bruneton mapping and 40-sample optical depth |
| `atmosphere_multiscattering.comp` | Wicked/Unreal multi-scattering pass | sphere estimator and geometric-series closure; 4x4 directions replace Wicked's 8x8 group reduction |
| `atmosphere_skyview.comp` | Wicked/Unreal sky-view pass | non-linear view mapping and 30-sample integration |
| `atmosphere_aerial.comp` | Wicked camera-volume pass | camera-ray froxels, squared depth distribution, depth-limited integration |
| `shaders/tonemap.frag` | Wicked `GetSunLuminance`, camera-volume sampling, and aerial composite | finite sun disk, depth reconstruction, atmosphere-before-tone-map ordering |

Engine-specific bindless resources, cloud/opaque volumetric shadows, weather
objects, stars, and the high-quality per-pixel fallback were intentionally not
copied. The local implementation stores coloured transmittance separately from
scattering instead of Wicked's scalar opacity.

## Controls and validation

- F11 cycles transmittance, multiple scattering, sky view, aerial scattering,
  and aerial transmittance debug views.
- `[` and `]` select any of the 32 aerial debug slices.
- F12 cycles afternoon, sunset, and high-sun directions. Shadows, terrain sun
  colour, sky, haze, and the sun disk all use that same direction.
- F5 rebuilds all graphics/compute pipelines and invalidates cached static LUTs.

Subtle relight treats the source imagery as its baseline and applies shadow
visibility as a separate modulation. Occluded terrain therefore becomes darker
instead of reverting to the unlit map. Full material mode continues to use the
physical direct-plus-sky-light result.

`phase7_atmosphere_tests` verifies the Earth preset, the transmittance mapping
round trip, and the squared aerial-slice mapping. `spirv-val` validates all
atmosphere and integration shaders as part of the manual verification workflow.
