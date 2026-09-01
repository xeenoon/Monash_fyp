#include "bake_quilt.h"

#include <limits.h>

static uint32_t mix32(uint32_t v) {
    v ^= v >> 16; v *= 0x7feb352du;
    v ^= v >> 15; v *= 0x846ca68bu;
    return v ^ (v >> 16);
}

static int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

void bake_quilt_cpu(const uint32_t *atlas, const uint32_t *labels,
                    uint32_t *output, uint32_t *owners,
                    int width, int height, uint32_t seed) {
    const int patch = BAKE_QUILT_PATCH, stride = BAKE_QUILT_STRIDE;
    int room_x = width > patch ? width - patch : 0;
    int room_y = height > patch ? height - patch : 0;
    int plane = width * height;
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        int base_x = x / stride, base_y = y / stride;
        uint32_t material = labels[y * width + x] < 3 ? labels[y * width + x] : 0;
        uint32_t best_score = UINT_MAX, best_owner = material * (uint32_t)plane;
        // The nearest four jittered patch cells form an independently-evaluable
        // Voronoi seam. This is the parallel replacement for greedy min-cut.
        for (int cy = base_y - 1; cy <= base_y + 1; ++cy) {
            for (int cx = base_x - 1; cx <= base_x + 1; ++cx) {
                if (cx < 0 || cy < 0) continue;
                int tx = cx * stride, ty = cy * stride;
                if (x < tx || y < ty || x >= tx + patch || y >= ty + patch) continue;
                uint32_t h = mix32(seed ^ (uint32_t)cx * 0x9e3779b9u ^
                                   (uint32_t)cy * 0x85ebca6bu ^ material * 0xc2b2ae35u);
                int sx0 = room_x ? (int)(h % (uint32_t)(room_x + 1)) : 0;
                int sy0 = room_y ? (int)(mix32(h + 0x27d4eb2du) %
                                          (uint32_t)(room_y + 1)) : 0;
                int local_x = x - tx, local_y = y - ty;
                int jitter_x = (int)((h >> 24) & 7u) - 3;
                int jitter_y = (int)((h >> 20) & 7u) - 3;
                int center_x = tx + patch / 2 + jitter_x;
                int center_y = ty + patch / 2 + jitter_y;
                int dx = x - center_x, dy = y - center_y;
                uint32_t score = (uint32_t)(dx * dx + dy * dy);
                // Stable tiny tie-break removes any device-dependent equal-cost
                // choice without being visually strong enough to form a grid.
                score = score * 16u + ((h >> 8) & 15u);
                uint32_t owner = material * (uint32_t)plane +
                                 (uint32_t)clampi(sy0 + local_y, 0, height - 1) *
                                     (uint32_t)width +
                                 (uint32_t)clampi(sx0 + local_x, 0, width - 1);
                if (score < best_score || (score == best_score && owner < best_owner)) {
                    best_score = score; best_owner = owner;
                }
            }
        }
        owners[y * width + x] = best_owner;
        output[y * width + x] = atlas[best_owner];
    }
}

void bake_quilt_gpu(BakeGpu *gpu, const BakeBuffer *atlas,
                    const BakeBuffer *labels, const BakeBuffer *output,
                    const BakeBuffer *owners, int width, int height,
                    uint32_t seed) {
    BakePipeline pipeline = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_quilt.comp.spv", 4, 3 * sizeof(uint32_t));
    struct { uint32_t width, height, seed; } push = {
        (uint32_t)width, (uint32_t)height, seed};
    BakeBuffer buffers[4] = {*atlas, *labels, *output, *owners};
    bake_dispatch(gpu, &pipeline, buffers, 4, &push, sizeof(push),
                  (uint32_t)(width + 7) / 8, (uint32_t)(height + 7) / 8, 1);
    bake_pipeline_destroy(gpu, &pipeline);
}
