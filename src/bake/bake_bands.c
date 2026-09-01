#include "bake_bands.h"

#include <stdlib.h>

#include "bake_filter.h"

static void cpu_blur(const float *in, float *out, float *scratch,
                     int width, int height, float sigma) {
    int radius = bake_gaussian_radius(sigma, 4.0f);
    float *weights = malloc((size_t)(2 * radius + 1) * sizeof(float));
    bake_gaussian_weights(sigma, radius, weights);
    bake_gaussian_cpu(in, out, scratch, width, height, 3, weights, radius);
    free(weights);
}

void bake_bands_cpu(const uint32_t *input, BakeBandsCpu bands,
                    int width, int height) {
    size_t count = (size_t)width * height;
    float *blur2 = malloc(count * 3 * sizeof(float));
    float *blur12 = malloc(count * 3 * sizeof(float));
    float *scratch = malloc(count * 3 * sizeof(float));
    for (size_t i = 0; i < count; ++i) {
        uint32_t p = input[i];
        bands.rgb[i * 3 + 0] = (float)(p & 0xffu);
        bands.rgb[i * 3 + 1] = (float)((p >> 8) & 0xffu);
        bands.rgb[i * 3 + 2] = (float)((p >> 16) & 0xffu);
    }
    cpu_blur(bands.rgb, blur2, scratch, width, height, 2.0f);
    cpu_blur(bands.rgb, blur12, scratch, width, height, 12.0f);
    cpu_blur(bands.rgb, bands.low32, scratch, width, height, 32.0f);
    for (size_t i = 0; i < count * 3; ++i) {
        bands.mid[i] = blur2[i] - blur12[i];
        bands.meso[i] = blur12[i] - bands.low32[i];
        bands.fine[i] = bands.rgb[i] - blur2[i];
    }
    free(blur2); free(blur12); free(scratch);
}

void bake_bands_gpu(BakeGpu *gpu, const BakeBuffer *input,
                    const BakeBuffer *rgb, const BakeBuffer *low32,
                    const BakeBuffer *mid, const BakeBuffer *meso,
                    const BakeBuffer *fine, int width, int height) {
    size_t bytes = (size_t)width * height * 3 * sizeof(float);
    BakeBuffer blur2 = bake_buffer_host(gpu, bytes);
    BakeBuffer blur12 = bake_buffer_host(gpu, bytes);
    BakeBuffer scratch = bake_buffer_host(gpu, bytes);
    struct { uint32_t width, height; } size = {(uint32_t)width, (uint32_t)height};
    uint32_t gx = (uint32_t)(width + 7) / 8, gy = (uint32_t)(height + 7) / 8;
    BakePipeline unpack = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_unpack.comp.spv", 2, sizeof(size));
    BakeBuffer unpack_bind[2] = {*input, *rgb};
    bake_dispatch(gpu, &unpack, unpack_bind, 2, &size, sizeof(size), gx, gy, 1);
    bake_pipeline_destroy(gpu, &unpack);
    bake_gaussian_gpu(gpu, rgb, &blur2, &scratch, width, height, 3, 2.0f, 4.0f);
    bake_gaussian_gpu(gpu, rgb, &blur12, &scratch, width, height, 3, 12.0f, 4.0f);
    bake_gaussian_gpu(gpu, rgb, low32, &scratch, width, height, 3, 32.0f, 4.0f);
    BakePipeline extract = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_bands.comp.spv", 6, sizeof(size));
    BakeBuffer band_bind[6] = {*rgb, blur2, blur12, *low32, *mid, *meso};
    // Fine shares no output binding in this six-buffer dispatch, so use a
    // second compact subtraction kernel for rgb-blur2.
    bake_dispatch(gpu, &extract, band_bind, 6, &size, sizeof(size), gx, gy, 1);
    bake_pipeline_destroy(gpu, &extract);
    BakePipeline subtract = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_subtract.comp.spv", 3, 3 * sizeof(uint32_t));
    struct { uint32_t width, height, channels; } subpush = {
        (uint32_t)width, (uint32_t)height, 3};
    BakeBuffer sub_bind[3] = {*rgb, blur2, *fine};
    bake_dispatch(gpu, &subtract, sub_bind, 3, &subpush, sizeof(subpush), gx, gy, 1);
    bake_pipeline_destroy(gpu, &subtract);
    bake_buffer_destroy(gpu, &scratch); bake_buffer_destroy(gpu, &blur12);
    bake_buffer_destroy(gpu, &blur2);
}
