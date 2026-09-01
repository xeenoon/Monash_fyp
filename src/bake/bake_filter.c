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

static int reflecti(int v, int extent) {
    while (v < 0 || v >= extent)
        v = v < 0 ? -v - 1 : 2 * extent - v - 1;
    return v;
}

static void gaussian_cpu_mode(const float *in, float *out, float *scratch,
                              int width, int height, int channels,
                              const float *weights, int radius, int reflect) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float acc = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    int xx = reflect ? reflecti(x + k, width) :
                                       clampi(x + k, 0, width - 1);
                    acc = acc + weights[k + radius] *
                          in[(y * width + xx) * channels + c];
                }
                scratch[(y * width + x) * channels + c] = acc;
            }
        }
    }
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                float acc = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    int yy = reflect ? reflecti(y + k, height) :
                                       clampi(y + k, 0, height - 1);
                    acc = acc + weights[k + radius] *
                          scratch[(yy * width + x) * channels + c];
                }
                out[(y * width + x) * channels + c] = acc;
            }
        }
    }
}

void bake_gaussian_cpu(const float *in, float *out, float *scratch,
                       int width, int height, int channels,
                       const float *weights, int radius) {
    gaussian_cpu_mode(in, out, scratch, width, height, channels,
                      weights, radius, 0);
}

void bake_gaussian_cpu_reflect(const float *in, float *out, float *scratch,
                               int width, int height, int channels,
                               const float *weights, int radius) {
    gaussian_cpu_mode(in, out, scratch, width, height, channels,
                      weights, radius, 1);
}

static void gaussian_gpu_mode(BakeGpu *gpu, const BakeBuffer *input,
                              const BakeBuffer *output, const BakeBuffer *scratch,
                              int width, int height, int channels, float sigma,
                              float truncate, uint32_t reflect) {
    int radius = bake_gaussian_radius(sigma, truncate);
    size_t taps = (size_t)(2 * radius + 1);
    BakeBuffer weights = bake_buffer_host(gpu, taps * sizeof(float));
    bake_gaussian_weights(sigma, radius, weights.mapped);
    BakePipeline pipeline = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_gaussian.comp.spv", 3,
        6 * sizeof(uint32_t));
    struct {
        uint32_t width, height, channels, radius, axis, reflect;
    } push = {(uint32_t)width, (uint32_t)height, (uint32_t)channels,
              (uint32_t)radius, 0, reflect};
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

void bake_gaussian_gpu(BakeGpu *gpu, const BakeBuffer *input,
                       const BakeBuffer *output, const BakeBuffer *scratch,
                       int width, int height, int channels, float sigma,
                       float truncate) {
    gaussian_gpu_mode(gpu, input, output, scratch, width, height, channels,
                      sigma, truncate, 0);
}

void bake_gaussian_gpu_reflect(BakeGpu *gpu, const BakeBuffer *input,
                               const BakeBuffer *output,
                               const BakeBuffer *scratch, int width, int height,
                               int channels, float sigma, float truncate) {
    gaussian_gpu_mode(gpu, input, output, scratch, width, height, channels,
                      sigma, truncate, 1);
}

void bake_gaussian_batch_gpu(BakeGpu *gpu, const BakeBuffer *input,
                             const BakeBuffer *output,
                             const BakeBuffer *scratch, int width, int height,
                             int channels, int layers, float sigma,
                             float truncate, int reflect) {
    int radius = bake_gaussian_radius(sigma, truncate);
    size_t taps = (size_t)(2 * radius + 1);
    BakeBuffer weights = bake_buffer_host(gpu, taps * sizeof(float));
    bake_gaussian_weights(sigma, radius, weights.mapped);
    BakePipeline pipeline = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_gaussian_batch.comp.spv", 3,
        7 * sizeof(uint32_t));
    struct {
        uint32_t width, height, channels, radius, axis, layers, reflect_mode;
    } push = {(uint32_t)width, (uint32_t)height, (uint32_t)channels,
              (uint32_t)radius, 0u, (uint32_t)layers,
              reflect ? 1u : 0u};
    BakeBuffer horizontal[3] = {*input, *scratch, weights};
    bake_dispatch(gpu, &pipeline, horizontal, 3, &push, sizeof(push),
                  (uint32_t)(width + 7) / 8,
                  (uint32_t)(height + 7) / 8, (uint32_t)layers);
    push.axis = 1;
    BakeBuffer vertical[3] = {*scratch, *output, weights};
    bake_dispatch(gpu, &pipeline, vertical, 3, &push, sizeof(push),
                  (uint32_t)(width + 7) / 8,
                  (uint32_t)(height + 7) / 8, (uint32_t)layers);
    bake_pipeline_destroy(gpu, &pipeline);
    bake_buffer_destroy(gpu, &weights);
}
