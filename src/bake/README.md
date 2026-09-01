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

## Harness (step 1 — DONE, verified)

- `bake_gpu.{h,c}` — surface-free init (instance + compute device + queue +
  command pool), a host-visible storage-buffer helper, SPIR-V pipeline creation,
  and a blocking `bake_dispatch`. Independent of `renderer.c`.
- `shaders/bake_scale.comp` + `tests/bake_gpu_tests.c` — smoke kernel and an
  exact-match GPU round-trip test (skips with exit 77 if no device).

## Validation discipline

Every stage keeps the Python `tools/` pipeline as the **reference oracle**. Each
`bake_*` unit test dispatches the kernel and diffs GPU output against the CPU
result — exact for integer stages, tolerance-based for float, plus purity /
band-energy metrics for the synthesis stages (GPU float order and the parallel
algorithms are not byte-identical to the CPU quilt). Never eyeball.

## Step ladder (each = one compute shader + one GPU-vs-oracle unit test)

1. **[done]** harness + round-trip test.
2. tile I/O + upload all tiles into a resident 2D image array; readback + PPM/PNG
   out. Test: round-trip a real tile unchanged.
3. per-pixel `classify_materials` (+ snow rule). Test: label agreement vs CPU.
4. `clean_source` via JFA inpaint. Test: cleaned-pixel deltas vs CPU.
5. separable Gaussian pyramid + band extraction (mid/meso/fine). Test: band RMS.
6. per-material exemplar quilt (min-cut → parallel variant). Test: purity/energy.
7. relayer (colour re-anchor + band top-up + per-material desaturation). Test: band RMS + G−R.
8. transition presets: macro/micro SDF via JFA + PatchMatch quilt. Test: mass error + seams.
9. orchestrate all five presets end-to-end; write final PNG tiles.
