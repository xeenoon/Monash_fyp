// Compiled with -ffp-contract=off (see CMakeLists) so no multiply-add fuses;
// this is what lets the GPU match bit-for-bit.
#include "bake_filter.h"

#include <math.h>
#include <stdlib.h>

int bake_gaussian_radius(float sigma, float truncate) {
    int radius = (int)(truncate * sigma + 0.5f);
    return radius < 1 ? 1 : radius;
}

void bake_gaussian_weights(float sigma, int radius, float *weights) {
    float sum = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
        float value = expf(-0.5f * (float)k * (float)k / (sigma * sigma));
        weights[k + radius] = value;
        sum += value;
    }
    for (int i = 0; i < 2 * radius + 1; ++i) weights[i] /= sum;
}

static int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

void bake_gaussian_cpu(const float *in, float *out, float *scratch,
                       int width, int height, int channels,
                       const float *weights, int radius) {
    // Horizontal pass: in -> scratch.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float acc = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    int xx = clampi(x + k, 0, width - 1);
                    acc = acc + weights[k + radius] * in[(y * width + xx) * channels + c];
                }
                scratch[(y * width + x) * channels + c] = acc;
            }
        }
    }
    // Vertical pass: scratch -> out.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float acc = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    int yy = clampi(y + k, 0, height - 1);
                    acc = acc + weights[k + radius] * scratch[(yy * width + x) * channels + c];
                }
                out[(y * width + x) * channels + c] = acc;
            }
        }
    }
}

void bake_gaussian_gpu(BakeGpu *gpu, const BakeBuffer *input,
                       const BakeBuffer *output, const BakeBuffer *scratch,
                       int width, int height, int channels, float sigma,
                       float truncate) {
    int radius = bake_gaussian_radius(sigma, truncate);
    size_t taps = (size_t)(2 * radius + 1);
    BakeBuffer weights = bake_buffer_host(gpu, taps * sizeof(float));
    bake_gaussian_weights(sigma, radius, weights.mapped);
    BakePipeline pipeline = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_gaussian.comp.spv", 3,
        5 * sizeof(uint32_t));
    struct {
        uint32_t width, height, channels, radius, axis;
    } push = {(uint32_t)width, (uint32_t)height, (uint32_t)channels,
              (uint32_t)radius, 0};
    BakeBuffer horizontal[3] = {*input, *scratch, weights};
    bake_dispatch(gpu, &pipeline, horizontal, 3, &push, sizeof(push),
                  (uint32_t)(width + 7) / 8, (uint32_t)(height + 7) / 8, 1);
    push.axis = 1;
    BakeBuffer vertical[3] = {*scratch, *output, weights};
    bake_dispatch(gpu, &pipeline, vertical, 3, &push, sizeof(push),
                  (uint32_t)(width + 7) / 8, (uint32_t)(height + 7) / 8, 1);
    bake_pipeline_destroy(gpu, &pipeline);
    bake_buffer_destroy(gpu, &weights);
}
