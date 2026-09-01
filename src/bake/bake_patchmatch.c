#include "bake_patchmatch.h"

#include <limits.h>

static uint32_t mix32(uint32_t v) {
    v ^= v >> 16; v *= 0x7feb352du;
    v ^= v >> 15; v *= 0x846ca68bu;
    return v ^ (v >> 16);
}
static int absi(int v) { return v < 0 ? -v : v; }
static int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

static uint32_t adjust(uint32_t owner, int dx, int dy, int width, int height) {
    uint32_t plane_size = (uint32_t)(width * height), plane = owner / plane_size;
    uint32_t local = owner % plane_size;
    int x = (int)(local % (uint32_t)width), y = (int)(local / (uint32_t)width);
    x = clampi(x + dx, 0, width - 1); y = clampi(y + dy, 0, height - 1);
    return plane * plane_size + (uint32_t)(y * width + x);
}

static uint32_t candidate_cost(const uint32_t *atlas, const uint32_t *labels,
                               const uint32_t *owners, int x, int y,
                               uint32_t candidate, int width, int height) {
    uint32_t plane_size = (uint32_t)(width * height);
    uint32_t material = labels[y * width + x] < 3u ? labels[y * width + x] : 0u;
    if (candidate / plane_size != material) return UINT_MAX;
    uint32_t sum[3] = {0, 0, 0}, neighbours = 0, continuity = UINT_MAX;
    static const int delta[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
    uint32_t c_local = candidate % plane_size;
    int cx = (int)(c_local % (uint32_t)width), cy = (int)(c_local / (uint32_t)width);
    for (int n = 0; n < 4; ++n) {
        int nx = x + delta[n][0], ny = y + delta[n][1];
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        if (labels[ny * width + nx] != material) continue;
        uint32_t no = owners[ny * width + nx];
        if (no / plane_size != material) continue;
        uint32_t p = atlas[no];
        sum[0] += p & 255u; sum[1] += (p >> 8) & 255u; sum[2] += (p >> 16) & 255u;
        ++neighbours;
        uint32_t expected = adjust(no, -delta[n][0], -delta[n][1], width, height);
        uint32_t e_local = expected % plane_size;
        int ex = (int)(e_local % (uint32_t)width), ey = (int)(e_local / (uint32_t)width);
        uint32_t d = (uint32_t)(absi(cx - ex) + absi(cy - ey));
        if (d < continuity) continuity = d;
    }
    uint32_t p = atlas[candidate];
    uint32_t r = p & 255u, g = (p >> 8) & 255u, b = (p >> 16) & 255u;
    if (!neighbours) return continuity == UINT_MAX ? 0u : continuity * 16u;
    uint32_t ar = sum[0] / neighbours, ag = sum[1] / neighbours, ab = sum[2] / neighbours;
    int dr = (int)r - (int)ar, dg = (int)g - (int)ag, db = (int)b - (int)ab;
    if (continuity == UINT_MAX) continuity = 0;
    return (uint32_t)(dr * dr + dg * dg + db * db) + continuity * 16u;
}

void bake_patchmatch_cpu(const uint32_t *atlas, const uint32_t *labels,
                         uint32_t *owners, uint32_t *scratch,
                         int width, int height, uint32_t seed, int iterations) {
    uint32_t plane_size = (uint32_t)(width * height), *src = owners, *dst = scratch;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
            int i = y * width + x;
            uint32_t material = labels[i] < 3u ? labels[i] : 0u;
            uint32_t candidates[6]; int n = 0;
            candidates[n++] = src[i];
            if (x > 0) candidates[n++] = adjust(src[i - 1], 1, 0, width, height);
            if (x + 1 < width) candidates[n++] = adjust(src[i + 1], -1, 0, width, height);
            if (y > 0) candidates[n++] = adjust(src[i - width], 0, 1, width, height);
            if (y + 1 < height) candidates[n++] = adjust(src[i + width], 0, -1, width, height);
            uint32_t h = mix32(seed ^ (uint32_t)i * 0x9e3779b9u ^
                               (uint32_t)iteration * 0x85ebca6bu);
            candidates[n++] = material * plane_size + h % plane_size;
            uint32_t best = candidates[0], best_cost = UINT_MAX;
            for (int k = 0; k < n; ++k) {
                uint32_t cost = candidate_cost(atlas, labels, src, x, y,
                                               candidates[k], width, height);
                if (cost < best_cost || (cost == best_cost && candidates[k] < best)) {
                    best = candidates[k]; best_cost = cost;
                }
            }
            dst[i] = best;
        }
        uint32_t *tmp = src; src = dst; dst = tmp;
    }
    if (src != owners)
        for (uint32_t i = 0; i < plane_size; ++i) owners[i] = src[i];
}

void bake_patchmatch_gpu(BakeGpu *gpu, const BakeBuffer *atlas,
                         const BakeBuffer *labels, const BakeBuffer *owners,
                         const BakeBuffer *output, int width, int height,
                         uint32_t seed, int iterations) {
    size_t bytes = (size_t)width * height * sizeof(uint32_t);
    BakeBuffer scratch = bake_buffer_host(gpu, bytes);
    BakeBuffer cur = *owners, other = scratch;
    BakePipeline pipeline = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_patchmatch.comp.spv", 4, 4 * sizeof(uint32_t));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        struct { uint32_t width, height, seed, iteration; } push = {
            (uint32_t)width, (uint32_t)height, seed, (uint32_t)iteration};
        BakeBuffer buffers[4] = {*atlas, *labels, cur, other};
        bake_dispatch(gpu, &pipeline, buffers, 4, &push, sizeof(push),
                      (uint32_t)(width + 7) / 8, (uint32_t)(height + 7) / 8, 1);
        BakeBuffer tmp = cur; cur = other; other = tmp;
    }
    bake_pipeline_destroy(gpu, &pipeline);
    BakePipeline gather = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_gather.comp.spv", 3, 2 * sizeof(uint32_t));
    struct { uint32_t width, height; } size = {(uint32_t)width, (uint32_t)height};
    BakeBuffer gather_buffers[3] = {*atlas, cur, *output};
    bake_dispatch(gpu, &gather, gather_buffers, 3, &size, sizeof(size),
                  (uint32_t)(width + 7) / 8, (uint32_t)(height + 7) / 8, 1);
    bake_pipeline_destroy(gpu, &gather);
    // Keep the public owner buffer final for downstream band sampling.
    if (cur.buffer != owners->buffer) {
        uint32_t *dst = owners->mapped, *src = cur.mapped;
        for (size_t i = 0; i < (size_t)width * height; ++i) dst[i] = src[i];
    }
    bake_buffer_destroy(gpu, &scratch);
}
