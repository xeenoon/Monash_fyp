# terrain_bake — headless Vulkan compute baker

Cross-platform (Vulkan → Windows/Linux native, macOS/iOS via MoltenVK, Android;
NVIDIA/AMD/Intel) GPU port of the offline terrain-synthesis pipeline currently
in `tools/*.py`. Opt-in: configure with `-DTERRAIN_BAKE=ON`.

    cmake -S . -B build-bake -DTERRAIN_BAKE=ON
    cmake --build build-bake --target terrain_bake
    ctest --test-dir build-bake -R bake_          # per-stage GPU unit tests
    cmake --build build-bake --target bake         # run the baker to produce tiles

## Why headless Vulkan (not CUDA / not "one big kernel")

- The project is already Vulkan with a working compute path, so we reuse the
  SPIR-V build and device model and stay portable — no vendor lock-in.
- It is **not** one monolithic kernel. The pipeline mixes embarrassingly-parallel
  array math (great on GPU) with sequential greedy loops. Tier-3 speed comes from
  swapping the *sequential* algorithms for parallel ones, not porting loops:
  quilt → PatchMatch / texture-optimization; SDF → jump-flooding (JFA).

## Harness and pipeline status

- `bake_gpu.{h,c}` — surface-free init (instance + compute device + queue +
  command pool), a host-visible storage-buffer helper, SPIR-V pipeline creation,
  and a blocking `bake_dispatch`. Independent of `renderer.c`.
- `shaders/bake_scale.comp` + `tests/bake_gpu_tests.c` — smoke kernel and an
  exact-match GPU round-trip test (skips with exit 77 if no device).
- The Vulkan harness and low-level compute primitives are implemented. The
  current end-to-end baker is a faceted prototype and is **not** a port of
  pass17: it loads three fixed material images and substitutes a parallel
  Voronoi/Jacobi synthesis for pass17's ordered, multilevel minimum-cut quilt.
  Its PNGs must not be treated as pass17-equivalent output.

## Frozen pass17 compatibility contract

The immutable visual oracle is the five-image output produced by production
Python commit `1c7f6a9ff96e3717d996cc0fa881983078860f37`. An accepted Vulkan
implementation must preserve pass17's operation order: source discovery and
cleaning, macro/micro labels, low-frequency composition, ordered 96/48/24 px
patch placement, candidate choice, minimum-cut ownership, then the material
microtexture relayer. Candidate scores may be evaluated in parallel, but the
winning patch must be committed before the next placement is scored.

The required PNG byte-stream SHA-256 targets are:

- `full_snow`: `15bf24f0e7afcc5782cfd4900d180abcd053e1477e0989f682248ed99efb1102`
- `snow_rock`: `7fece61c85e157766aded4d7399733457ce850424b4de64b9c6b62fda090da30`
- `full_rock`: `e0a4c23c7e8ed0d66b39c6d37eb5e2dfe403cd4925debf522b4ac8107bf4ca22`
- `rock_grass`: `c5c4d44d22d09ce1147e8860c96aaec0c1e701c9705d409b17f36b2fb337ee89`
- `full_grass`: `53b69bee36a41063db1c2b7a0c4fa021eef6c8e2b6cbc43249cce723ba5c502f`

For these showcase presets edge snapping is disabled. The final PNG equals the
unblended synthesis result; the saved level-4 diagnostic differs only because
it draws a red one-pixel audit border.

## Validation discipline

The `bake_*` unit tests currently prove C/GLSL agreement for individual
primitives. They do not establish pass17 compatibility: several C primitives
are deliberate replacement algorithms. End-to-end acceptance requires at
least 99.0% normalized RGB byte similarity for every frozen pass17 output.
Exact-pixel percentage remains visible as a diagnostic, but tiny floating-point
drift does not veto a visually indistinguishable result.

> **Height sampling note:** the current Option-A contract bilinearly resamples
> each 129×129 `.trn` interior to the 256×256 output using CPU-generated weights
> and identical C/GLSL operation order. A future faster path may have the GPU
> sample the closest 129×129 height point and round out the edge discrepancy.
> Keep bilinear 1:1 for now; the nearest-point edge treatment is deliberately
> deferred.

## Step ladder (each = one compute shader + one GPU-vs-oracle unit test)

1. **[done]** harness + round-trip test.
2. **[done]** tile I/O (`bake_image.{h,c}`, stb load + PPM out) + GPU buffer
   residency via `bake_copy.comp`. Tests: `bake_image_roundtrip` (synthetic tile,
   exact); `terrain_bake` round-trips a real 258² dataset tile byte-identically.
3. **[done]** per-pixel `classify_materials` (+ snow rule) — `bake_classify.comp`,
   C oracle `bake_reference.h`. Test `bake_classify_oracle`: bit-for-bit over the
   full RGB cube (4096 combos); C oracle verified == Python `colour_labels`.
4. **[done]** `clean_source`: anomaly masks, JFA inpaint, confidence and hard
   masks. Test `bake_clean_bitexact` compares every output field exactly.
5. **[done]** separable Gaussian pyramid + mid/meso/fine band extraction.
   Tests `bake_gaussian_bitexact` and `bake_bands_bitexact` compare all floats.
6. **[prototype only]** the parallel Voronoi quilt and Jacobi refinement match
   their C counterparts, but replace pass17's ordered minimum-cut quilt and
   therefore cannot satisfy the compatibility contract.
7. **[prototype only]** the current relayer matches its simplified C
   counterpart, not pass17's three-level residual quilt plus material-exemplar
   energy top-up.
8. **[prototype only]** dual-JFA transitions do not reproduce pass17's
   macro/micro shape synthesis.
9. **[not matched]** the five-preset command writes PNGs, but its outputs remain
   measurably and visibly different from pass17. The separate 129→256
   `bake_resample_bitexact` test remains valid for the height conversion.
