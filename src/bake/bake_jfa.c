#include "bake_jfa.h"

#include <limits.h>
#include <stddef.h>
#include <stdlib.h>

void bake_edt_cpu(const uint8_t *mask, int32_t *owner, int width, int height) {
    int count = width * height;
    int32_t *horizontal = malloc((size_t)count * sizeof(*horizontal));
    int32_t *distance = malloc((size_t)count * sizeof(*distance));
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        int best_owner = -1, best_distance = INT_MAX;
        for (int sx = 0; sx < width; ++sx) {
            int candidate = y * width + sx;
            if (mask[candidate]) continue;
            int dx = x - sx, value = dx * dx;
            if (value < best_distance ||
                (value == best_distance && candidate < best_owner)) {
                best_distance = value;
                best_owner = candidate;
            }
        }
        horizontal[y * width + x] = best_owner;
        distance[y * width + x] = best_distance;
    }
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        int best_owner = -1, best_distance = INT_MAX;
        for (int sy = 0; sy < height; ++sy) {
            int i = sy * width + x;
            if (horizontal[i] < 0) continue;
            int dy = y - sy, value = distance[i] + dy * dy;
            // scipy.ndimage's separable EDT resolves equal-distance owners by
            // the last transformed axis first: x before y for a 2-D image.
            if (value < best_distance ||
                (value == best_distance &&
                 horizontal[i] % width < best_owner % width)) {
                best_distance = value;
                best_owner = horizontal[i];
            }
        }
        owner[y * width + x] = best_owner;
    }
    free(distance);
    free(horizontal);
}

void bake_edt_gpu(BakeGpu *gpu, const BakeBuffer *mask, const BakeBuffer *owner,
                  int width, int height) {
    size_t bytes = (size_t)width * height * sizeof(uint32_t);
    BakeBuffer horizontal = bake_buffer_host(gpu, bytes);
    BakeBuffer distance = bake_buffer_host(gpu, bytes);
    struct { uint32_t width, height; } push = {
        (uint32_t)width, (uint32_t)height};
    uint32_t gx = (uint32_t)(width + 7) / 8;
    uint32_t gy = (uint32_t)(height + 7) / 8;
    BakePipeline first = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_edt_horizontal.comp.spv", 3, sizeof(push));
    BakeBuffer first_bindings[3] = {*mask, horizontal, distance};
    bake_dispatch(gpu, &first, first_bindings, 3, &push, sizeof(push), gx, gy, 1);
    bake_pipeline_destroy(gpu, &first);
    BakePipeline second = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_edt_vertical.comp.spv", 3, sizeof(push));
    BakeBuffer second_bindings[3] = {horizontal, distance, *owner};
    bake_dispatch(gpu, &second, second_bindings, 3, &push, sizeof(push), gx, gy, 1);
    bake_pipeline_destroy(gpu, &second);
    bake_buffer_destroy(gpu, &distance);
    bake_buffer_destroy(gpu, &horizontal);
}

int bake_jfa_pass_count(int width, int height) {
    int longest = width > height ? width : height;
    int step = 1;
    int passes = 0;
    while (step < longest) {
        step <<= 1;
        ++passes;
    }
    return passes < 1 ? 1 : passes;  // steps: 2^(passes-1) .. 1
}

// One JFA pass at the given step, reading `in`, writing `out`.  Fixed neighbour
// order (dy then dx over {-step,0,step}) and strict `<` on integer squared
// distance so ties resolve identically here and in the shader.
static void jfa_pass(const int32_t *in, int32_t *out, int width, int height, int step) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int32_t best = in[y * width + x];
            long best_d = -1;
            if (best >= 0) {
                int bx = best % width, by = best / width;
                long dx = x - bx, dy = y - by;
                best_d = dx * dx + dy * dy;
            }
            for (int oy = -step; oy <= step; oy += step) {
                for (int ox = -step; ox <= step; ox += step) {
                    int nx = x + ox, ny = y + oy;
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                    int32_t seed = in[ny * width + nx];
                    if (seed < 0) continue;
                    int sx = seed % width, sy = seed / width;
                    long dx = x - sx, dy = y - sy;
                    long d = dx * dx + dy * dy;
                    if (best_d < 0 || d < best_d) {
                        best_d = d;
                        best = seed;
                    }
                }
            }
            out[y * width + x] = best;
        }
    }
}

void bake_jfa_cpu(const uint8_t *mask, int32_t *owner, int32_t *scratch,
                  int width, int height) {
    for (int i = 0; i < width * height; ++i) owner[i] = mask[i] ? -1 : i;

    int longest = width > height ? width : height;
    int start = 1;
    while (start < longest) start <<= 1;
    start >>= 1;

    int32_t *src = owner, *dst = scratch;
    for (int step = start; step >= 1; step >>= 1) {
        jfa_pass(src, dst, width, height, step);
        int32_t *tmp = src;
        src = dst;
        dst = tmp;
    }
    if (src != owner) {
        for (int i = 0; i < width * height; ++i) owner[i] = src[i];
    }
}

void bake_jfa_gpu(BakeGpu *gpu, const BakeBuffer *mask, const BakeBuffer *owner,
                  int width, int height) {
    size_t count = (size_t)width * height;
    BakeBuffer scratch = bake_buffer_host(gpu, count * sizeof(int32_t));
    struct { uint32_t width, height; } size = {(uint32_t)width, (uint32_t)height};
    uint32_t gx = (uint32_t)(width + 7) / 8, gy = (uint32_t)(height + 7) / 8;
    BakePipeline init = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_jfa_init.comp.spv", 2, sizeof(size));
    BakeBuffer init_buffers[2] = {*mask, *owner};
    bake_dispatch(gpu, &init, init_buffers, 2, &size, sizeof(size), gx, gy, 1);
    bake_pipeline_destroy(gpu, &init);
    BakePipeline jfa = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_jfa.comp.spv", 2, 3 * sizeof(uint32_t));
    int longest = width > height ? width : height, start = 1;
    while (start < longest) start <<= 1;
    start >>= 1;
    BakeBuffer cur = *owner, other = scratch;
    for (int step = start; step >= 1; step >>= 1) {
        struct { uint32_t width, height; int32_t step; } push = {
            (uint32_t)width, (uint32_t)height, step};
        BakeBuffer pass[2] = {cur, other};
        bake_dispatch(gpu, &jfa, pass, 2, &push, sizeof(push), gx, gy, 1);
        BakeBuffer tmp = cur; cur = other; other = tmp;
    }
    bake_pipeline_destroy(gpu, &jfa);
    if (cur.buffer != owner->buffer) {
        int32_t *dst = owner->mapped, *src = cur.mapped;
        for (size_t i = 0; i < count; ++i) dst[i] = src[i];
    }
    bake_buffer_destroy(gpu, &scratch);
}
