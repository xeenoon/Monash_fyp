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
- Stages 1–9 are implemented. `terrain_bake` loads three real material
  exemplars plus a `.trn` heightmap, isolates each material, runs all five
  presets, and writes RGBA PNG files.

## Validation discipline

Under Option A, the matched C implementation defines every GPU primitive. Each
`bake_*` unit test dispatches the shader and compares the complete output to C
bit-for-bit, including floating-point Gaussian, resampling, bands, relayer and
the five-preset pipeline. The Python pipeline remains the visual/metric baseline
for purity and band energy, not the byte-level oracle.

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
6. **[done]** parallel material-aware exemplar quilt. Test
   `bake_quilt_bitexact` compares source owners and RGB exactly.
7. **[done]** relayer: broad-colour re-anchor, band top-up and per-material
   chroma retention (snow 0.10). Test `bake_relayer_bitexact` compares RGBA.
8. **[done]** dual-JFA signed-distance transitions plus four Jacobi PatchMatch
   iterations. Test `bake_transition_bitexact` compares labels, SDF, owners and
   RGB exactly.
9. **[done]** all five presets, PNG output and real dataset command. Test
   `bake_pipeline_bitexact` compares every final pixel; the separate 129→256
   `bake_resample_bitexact` test covers the real `.trn` dimension conversion.
