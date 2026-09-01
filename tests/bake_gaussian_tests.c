// Stage-4 primitive test: the GPU separable Gaussian equals the C reference
// bit-for-bit (Option A contract).  A non-trivial float image + a real sigma
// exercise many taps and both boundary clamps.  Exact uint32-of-float compare.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bake_filter.h"
#include "bake_gpu.h"

#ifndef BAKE_SHADER_DIR
#error "BAKE_SHADER_DIR must be defined"
#endif

int main(void) {
    BakeGpu gpu;
    if (!bake_gpu_init(&gpu)) {
        fprintf(stderr, "bake_gaussian_tests: no compute device; skipping\n");
        return 77;
    }

    const int width = 197, height = 143, channels = 3;  // odd sizes stress clamp
    const float sigma = 3.7f, truncate = 4.0f;
    const int radius = bake_gaussian_radius(sigma, truncate);
    const int taps = 2 * radius + 1;
    const size_t n = (size_t)width * height * channels;

    float *input = malloc(n * sizeof(float));
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < channels; ++c)
                input[(y * width + x) * channels + c] =
                    (float)(((x * 13 + y * 7 + c * 101) ^ (x * y + c)) & 255);

    float *weights = malloc(taps * sizeof(float));
    bake_gaussian_weights(sigma, radius, weights);

    float *cpu = malloc(n * sizeof(float));
    float *scratch = malloc(n * sizeof(float));
    bake_gaussian_cpu(input, cpu, scratch, width, height, channels, weights, radius);

    // GPU: horizontal (in -> mid) then vertical (mid -> out), same as the C ref.
    BakeBuffer in = bake_buffer_host(&gpu, n * sizeof(float));
    BakeBuffer mid = bake_buffer_host(&gpu, n * sizeof(float));
    BakeBuffer out = bake_buffer_host(&gpu, n * sizeof(float));
    BakeBuffer wbuf = bake_buffer_host(&gpu, taps * sizeof(float));
    memcpy(in.mapped, input, n * sizeof(float));
    memcpy(wbuf.mapped, weights, taps * sizeof(float));

    BakePipeline pipeline = bake_pipeline_create(
        &gpu, BAKE_SHADER_DIR "/bake_gaussian.comp.spv", 3, 5 * sizeof(uint32_t));
    struct { uint32_t w, h, c, r, axis; } push = {
        (uint32_t)width, (uint32_t)height, (uint32_t)channels, (uint32_t)radius, 0};
    BakeBuffer h_pass[3] = {in, mid, wbuf};
    bake_dispatch(&gpu, &pipeline, h_pass, 3, &push, sizeof(push),
                  (width + 7) / 8, (height + 7) / 8, 1);
    push.axis = 1;
    BakeBuffer v_pass[3] = {mid, out, wbuf};
    bake_dispatch(&gpu, &pipeline, v_pass, 3, &push, sizeof(push),
                  (width + 7) / 8, (height + 7) / 8, 1);

    const float *result = out.mapped;
    int mismatches = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        // Bit-exact compare via the raw float payload.
        uint32_t a, b;
        memcpy(&a, &result[i], 4);
        memcpy(&b, &cpu[i], 4);
        if (a != b) {
            float d = result[i] - cpu[i];
            if (d < 0) d = -d;
            if (d > max_abs) max_abs = d;
            if (mismatches < 5)
                fprintf(stderr, "  [%zu] gpu=%.9g cpu=%.9g bits %08x/%08x\n", i,
                        result[i], cpu[i], a, b);
            ++mismatches;
        }
    }

    bake_pipeline_destroy(&gpu, &pipeline);
    bake_buffer_destroy(&gpu, &wbuf);
    bake_buffer_destroy(&gpu, &out);
    bake_buffer_destroy(&gpu, &mid);
    bake_buffer_destroy(&gpu, &in);
    bake_gpu_destroy(&gpu);
    free(input); free(weights); free(cpu); free(scratch);

    if (mismatches) {
        fprintf(stderr, "bake_gaussian_tests: FAIL (%d/%zu differ, max|Δ|=%.3g)\n",
                mismatches, n, max_abs);
        return 1;
    }
    printf("bake_gaussian_tests: OK (%dx%dx%d, sigma=%.2f, %d taps, bit-exact)\n",
           width, height, channels, sigma, taps);
    return 0;
}
