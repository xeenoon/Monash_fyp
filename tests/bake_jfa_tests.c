// Stage-4 primitive test: GPU jump-flooding nearest-seed field equals the C
// reference exactly (integer owner indices).  Sparse deterministic seeds force
// many propagation passes; ping-pong parity is tracked so we compare the buffer
// that actually holds the final result.  Skips (77) with no device.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bake_gpu.h"
#include "bake_jfa.h"

#ifndef BAKE_SHADER_DIR
#error "BAKE_SHADER_DIR must be defined"
#endif

int main(void) {
    BakeGpu gpu;
    if (!bake_gpu_init(&gpu)) {
        fprintf(stderr, "bake_jfa_tests: no compute device; skipping\n");
        return 77;
    }

    const int width = 200, height = 150;
    const int count = width * height;
    uint8_t *mask = malloc(count);
    int seeds = 0;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            int seed = ((x * 31 + y * 17) % 97 == 0) || (x == 3 && y == 140);
            mask[y * width + x] = seed ? 0 : 1;  // 0 = seed, 1 = hole
            seeds += seed;
        }

    int32_t *owner_cpu = malloc(count * sizeof(int32_t));
    int32_t *scratch = malloc(count * sizeof(int32_t));
    bake_jfa_cpu(mask, owner_cpu, scratch, width, height);

    // GPU: init owner buffer from the mask, then ping-pong the same step run.
    BakeBuffer a = bake_buffer_host(&gpu, count * sizeof(int32_t));
    BakeBuffer b = bake_buffer_host(&gpu, count * sizeof(int32_t));
    int32_t *a_data = a.mapped;
    for (int i = 0; i < count; ++i) a_data[i] = mask[i] ? -1 : i;

    BakePipeline pipeline = bake_pipeline_create(
        &gpu, BAKE_SHADER_DIR "/bake_jfa.comp.spv", 2, 3 * sizeof(uint32_t));

    int longest = width > height ? width : height;
    int start = 1;
    while (start < longest) start <<= 1;
    start >>= 1;
    BakeBuffer cur = a, other = b;
    for (int step = start; step >= 1; step >>= 1) {
        struct { uint32_t w, h; int32_t step; } push = {
            (uint32_t)width, (uint32_t)height, step};
        BakeBuffer pass[2] = {cur, other};
        bake_dispatch(&gpu, &pipeline, pass, 2, &push, sizeof(push),
                      (width + 7) / 8, (height + 7) / 8, 1);
        BakeBuffer tmp = cur;
        cur = other;
        other = tmp;
    }

    const int32_t *owner_gpu = cur.mapped;
    int mismatches = 0;
    for (int i = 0; i < count; ++i) {
        if (owner_gpu[i] != owner_cpu[i]) {
            if (mismatches < 8)
                fprintf(stderr, "  pixel %d: gpu owner=%d cpu owner=%d\n", i,
                        owner_gpu[i], owner_cpu[i]);
            ++mismatches;
        }
    }

    bake_pipeline_destroy(&gpu, &pipeline);
    bake_buffer_destroy(&gpu, &b);
    bake_buffer_destroy(&gpu, &a);
    bake_gpu_destroy(&gpu);
    free(mask); free(owner_cpu); free(scratch);

    if (mismatches) {
        fprintf(stderr, "bake_jfa_tests: FAIL (%d/%d owners differ)\n", mismatches, count);
        return 1;
    }
    printf("bake_jfa_tests: OK (%dx%d, %d seeds, nearest-seed field bit-exact)\n",
           width, height, seeds);
    return 0;
}
