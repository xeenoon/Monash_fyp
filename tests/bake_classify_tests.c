// Stage-3 unit test: GPU material classification matches the CPU oracle
// bit-for-bit across the RGB cube.  The 64x64 tile enumerates every
// (r,g,b) at step 17 (16 levels each = 4096 combos), so all grass/rock/snow/
// water threshold branches are exercised.  Skips (77) with no device.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bake_gpu.h"
#include "bake_reference.h"

#ifndef BAKE_SHADER_DIR
#error "BAKE_SHADER_DIR must be defined"
#endif

int main(void) {
    BakeGpu gpu;
    if (!bake_gpu_init(&gpu)) {
        fprintf(stderr, "bake_classify_tests: no compute device; skipping\n");
        return 77;
    }

    const int width = 64, height = 64;
    const int count = width * height;  // 4096
    uint32_t *pixels = malloc(count * sizeof(uint32_t));
    uint8_t *expected = malloc(count);
    for (int i = 0; i < count; ++i) {
        uint8_t r = (uint8_t)((i & 15) * 17);
        uint8_t g = (uint8_t)(((i >> 4) & 15) * 17);
        uint8_t b = (uint8_t)(((i >> 8) & 15) * 17);
        pixels[i] = (uint32_t)r | ((uint32_t)g << 8) | ((uint32_t)b << 16) | (255u << 24);
        expected[i] = bake_ref_classify(r, g, b);
    }

    BakeBuffer in = bake_buffer_host(&gpu, count * sizeof(uint32_t));
    BakeBuffer out = bake_buffer_host(&gpu, count * sizeof(uint32_t));
    memcpy(in.mapped, pixels, count * sizeof(uint32_t));

    BakePipeline pipeline = bake_pipeline_create(
        &gpu, BAKE_SHADER_DIR "/bake_classify.comp.spv", 2, 2 * sizeof(uint32_t));
    struct { uint32_t w, h; } push = {(uint32_t)width, (uint32_t)height};
    BakeBuffer buffers[2] = {in, out};
    bake_dispatch(&gpu, &pipeline, buffers, 2, &push, sizeof(push),
                  (width + 7) / 8, (height + 7) / 8, 1);

    const uint32_t *labels = out.mapped;
    int mismatches = 0, hist[3] = {0, 0, 0};
    for (int i = 0; i < count; ++i) {
        hist[expected[i]]++;
        if (labels[i] != expected[i]) {
            if (mismatches < 8) {
                uint8_t r = pixels[i] & 0xff, g = (pixels[i] >> 8) & 0xff, b = (pixels[i] >> 16) & 0xff;
                fprintf(stderr, "  (r=%u g=%u b=%u): gpu=%u cpu=%u\n", r, g, b,
                        labels[i], expected[i]);
            }
            ++mismatches;
        }
    }

    bake_pipeline_destroy(&gpu, &pipeline);
    bake_buffer_destroy(&gpu, &out);
    bake_buffer_destroy(&gpu, &in);
    bake_gpu_destroy(&gpu);
    free(pixels);
    free(expected);

    printf("bake_classify: %d combos, oracle rock=%d grass=%d snow=%d\n",
           count, hist[0], hist[1], hist[2]);
    if (mismatches) {
        fprintf(stderr, "bake_classify_tests: FAIL (%d/%d mismatches)\n", mismatches, count);
        return 1;
    }
    printf("bake_classify_tests: OK (bit-for-bit vs CPU oracle over full RGB cube)\n");
    return 0;
}
