#include "bake_relayer.h"

#include <math.h>

static float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

void bake_relayer_cpu(const uint32_t *base, const uint32_t *labels,
                      const uint32_t *owners, const float *low,
                      const float *mid, const float *meso, const float *fine,
                      uint32_t *output, int width, int height) {
    (void)base;
    static const float anchor[3][3] = {
        {112.0f, 108.0f, 101.0f}, {82.0f, 112.0f, 65.0f},
        {218.0f, 221.0f, 224.0f}};
    static const float chroma[3] = {0.65f, 1.0f, 0.10f};
    int count = width * height;
    for (int i = 0; i < count; ++i) {
        uint32_t kind = labels[i] < 3u ? labels[i] : 0u;
        uint32_t owner = owners[i] < (uint32_t)(count * 3) ? owners[i] : kind * count;
        size_t first = (size_t)owner * 3;
        float broad[3];
        for (int c = 0; c < 3; ++c)
            broad[c] = low[first + c] * 0.35f + anchor[kind][c] * 0.65f;
        float sum_band[3];
        for (int c = 0; c < 3; ++c)
            sum_band[c] = mid[first + c] + meso[first + c] + fine[first + c];
        float luma = sum_band[0] * 0.2126f + sum_band[1] * 0.7152f +
                     sum_band[2] * 0.0722f;
        for (int c = 0; c < 3; ++c)
            sum_band[c] = luma + chroma[kind] * (sum_band[c] - luma);
        uint32_t r = (uint32_t)clampf(broad[0] + sum_band[0], 0.0f, 255.0f);
        uint32_t g = (uint32_t)clampf(broad[1] + sum_band[1], 0.0f, 255.0f);
        uint32_t b = (uint32_t)clampf(broad[2] + sum_band[2], 0.0f, 255.0f);
        output[i] = r | (g << 8) | (b << 16) | 0xff000000u;
    }
}

void bake_relayer_gpu(BakeGpu *gpu, const BakeBuffer *base,
                      const BakeBuffer *labels, const BakeBuffer *owners,
                      const BakeBuffer *low, const BakeBuffer *mid,
                      const BakeBuffer *meso, const BakeBuffer *fine,
                      const BakeBuffer *output, int width, int height) {
    BakePipeline pipeline = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_relayer.comp.spv", 8, 2 * sizeof(uint32_t));
    struct { uint32_t width, height; } push = {(uint32_t)width, (uint32_t)height};
    BakeBuffer buffers[8] = {*base, *labels, *owners, *low, *mid, *meso, *fine, *output};
    bake_dispatch(gpu, &pipeline, buffers, 8, &push, sizeof(push),
                  (uint32_t)(width + 7) / 8, (uint32_t)(height + 7) / 8, 1);
    bake_pipeline_destroy(gpu, &pipeline);
}
