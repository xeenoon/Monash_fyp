// Compiled with -ffp-contract=off. Axis indices and interpolation weights are
// generated once on the CPU and uploaded, while CPU/GLSL use the same explicit
// top-then-bottom-then-vertical operation order.
#include "bake_resample.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static void build_axis(int source_size, int output_size,
                       uint32_t *indices, float *weights) {
    for (int out = 0; out < output_size; ++out) {
        // Pixel-centre mapping used by image bilinear resamplers. Clamp the
        // footprint at the source edges rather than extrapolating it.
        float source = ((float)out + 0.5f) * (float)source_size /
                           (float)output_size - 0.5f;
        int lo = (int)floorf(source);
        float weight = source - (float)lo;
        if (lo < 0) { lo = 0; weight = 0.0f; }
        if (lo >= source_size - 1) { lo = source_size - 1; weight = 0.0f; }
        int hi = lo < source_size - 1 ? lo + 1 : lo;
        indices[out * 2 + 0] = (uint32_t)lo;
        indices[out * 2 + 1] = (uint32_t)hi;
        weights[out] = weight;
    }
}

void bake_resample_bilinear_cpu(const float *input, float *output,
                                int source_width, int source_height,
                                int output_width, int output_height) {
    uint32_t *x_indices = malloc((size_t)output_width * 2 * sizeof(uint32_t));
    uint32_t *y_indices = malloc((size_t)output_height * 2 * sizeof(uint32_t));
    float *x_weights = malloc((size_t)output_width * sizeof(float));
    float *y_weights = malloc((size_t)output_height * sizeof(float));
    build_axis(source_width, output_width, x_indices, x_weights);
    build_axis(source_height, output_height, y_indices, y_weights);
    for (int y = 0; y < output_height; ++y) {
        uint32_t y0 = y_indices[y * 2], y1 = y_indices[y * 2 + 1];
        float ty = y_weights[y];
        for (int x = 0; x < output_width; ++x) {
            uint32_t x0 = x_indices[x * 2], x1 = x_indices[x * 2 + 1];
            float tx = x_weights[x];
            float a = input[(size_t)y0 * source_width + x0];
            float b = input[(size_t)y0 * source_width + x1];
            float c = input[(size_t)y1 * source_width + x0];
            float d = input[(size_t)y1 * source_width + x1];
            float top = a + tx * (b - a);
            float bottom = c + tx * (d - c);
            output[(size_t)y * output_width + x] = top + ty * (bottom - top);
        }
    }
    free(y_weights); free(x_weights); free(y_indices); free(x_indices);
}

void bake_resample_bilinear_gpu(BakeGpu *gpu, const BakeBuffer *input,
                                const BakeBuffer *output,
                                int source_width, int source_height,
                                int output_width, int output_height) {
    BakeBuffer x_indices = bake_buffer_host(
        gpu, (size_t)output_width * 2 * sizeof(uint32_t));
    BakeBuffer y_indices = bake_buffer_host(
        gpu, (size_t)output_height * 2 * sizeof(uint32_t));
    BakeBuffer x_weights = bake_buffer_host(
        gpu, (size_t)output_width * sizeof(float));
    BakeBuffer y_weights = bake_buffer_host(
        gpu, (size_t)output_height * sizeof(float));
    build_axis(source_width, output_width, x_indices.mapped, x_weights.mapped);
    build_axis(source_height, output_height, y_indices.mapped, y_weights.mapped);
    BakePipeline pipeline = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_resample.comp.spv", 6,
        4 * sizeof(uint32_t));
    struct { uint32_t source_width, source_height, output_width, output_height; } push = {
        (uint32_t)source_width, (uint32_t)source_height,
        (uint32_t)output_width, (uint32_t)output_height};
    BakeBuffer buffers[6] = {*input, *output, x_indices, x_weights,
                             y_indices, y_weights};
    bake_dispatch(gpu, &pipeline, buffers, 6, &push, sizeof(push),
                  (uint32_t)(output_width + 7) / 8,
                  (uint32_t)(output_height + 7) / 8, 1);
    bake_pipeline_destroy(gpu, &pipeline);
    bake_buffer_destroy(gpu, &y_weights); bake_buffer_destroy(gpu, &y_indices);
    bake_buffer_destroy(gpu, &x_weights); bake_buffer_destroy(gpu, &x_indices);
}
