// Stage-2 unit test: a real image-sized buffer survives a GPU round trip
// byte-for-byte.  Uses a deterministic synthetic tile so the test needs no
// dataset and can assert an exact match (the copy kernel must not touch a
// single pixel).  Skips (exit 77) when no compute device is present.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bake_gpu.h"
#include "bake_image.h"

#ifndef BAKE_SHADER_DIR
#error "BAKE_SHADER_DIR must be defined"
#endif

int main(void) {
    BakeGpu gpu;
    if (!bake_gpu_init(&gpu)) {
        fprintf(stderr, "bake_image_tests: no compute device; skipping\n");
        return 77;
    }

    const int width = 254, height = 254;  // matches the alps tile interior
    BakeImage source;
    if (!bake_image_alloc(width, height, &source)) return 1;
    // Deterministic pattern: gradients per channel plus a hash-ish wobble, so a
    // dropped/duplicated pixel or row/column swap would change the checksum.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t r = (uint8_t)(x);
            uint8_t g = (uint8_t)(y);
            uint8_t b = (uint8_t)((x * 7 + y * 13) ^ (x * y));
            uint8_t a = 255;
            source.pixels[y * width + x] =
                (uint32_t)r | ((uint32_t)g << 8) | ((uint32_t)b << 16) | ((uint32_t)a << 24);
        }
    }

    size_t count = (size_t)width * height;
    BakeBuffer in = bake_buffer_host(&gpu, count * sizeof(uint32_t));
    BakeBuffer out = bake_buffer_host(&gpu, count * sizeof(uint32_t));
    memcpy(in.mapped, source.pixels, count * sizeof(uint32_t));
    for (size_t i = 0; i < count; ++i) ((uint32_t *)out.mapped)[i] = 0xDEADBEEFu;

    BakePipeline pipeline = bake_pipeline_create(
        &gpu, BAKE_SHADER_DIR "/bake_copy.comp.spv", 2, 2 * sizeof(uint32_t));
    struct { uint32_t w; uint32_t h; } push = {(uint32_t)width, (uint32_t)height};
    BakeBuffer buffers[2] = {in, out};
    bake_dispatch(&gpu, &pipeline, buffers, 2, &push, sizeof(push),
                  (width + 7) / 8, (height + 7) / 8, 1);

    int failures = 0;
    const uint32_t *result = out.mapped;
    for (size_t i = 0; i < count; ++i) {
        if (result[i] != source.pixels[i]) {
            if (failures < 5)
                fprintf(stderr, "  pixel %zu: got %08x expected %08x\n", i,
                        result[i], source.pixels[i]);
            ++failures;
        }
    }

    // Prove the image-write path too, so stage 2 produces a real artifact.
    BakeImage baked;
    bake_image_alloc(width, height, &baked);
    memcpy(baked.pixels, out.mapped, count * sizeof(uint32_t));
    bake_image_write_ppm("bake_stage2_roundtrip.ppm", &baked);

    bake_pipeline_destroy(&gpu, &pipeline);
    bake_buffer_destroy(&gpu, &out);
    bake_buffer_destroy(&gpu, &in);
    bake_image_free(&baked);
    bake_image_free(&source);
    bake_gpu_destroy(&gpu);

    if (failures) {
        fprintf(stderr, "bake_image_tests: FAIL (%d of %zu pixels changed)\n",
                failures, count);
        return 1;
    }
    printf("bake_image_tests: OK (%dx%d tile round-tripped unchanged)\n", width, height);
    return 0;
}
