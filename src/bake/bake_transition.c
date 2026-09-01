#include "bake_transition.h"

#include <stdlib.h>

#include "bake_jfa.h"
#include "bake_patchmatch.h"
#include "bake_quilt.h"

static int distance2(int index, int owner, int width) {
    int x = index % width, y = index / width;
    int source_x = owner % width, source_y = owner / width;
    int dx = x - source_x, dy = y - source_y;
    return dx * dx + dy * dy;
}

void bake_transition_cpu(const float *heightmap, const uint32_t *atlas,
                         uint32_t *labels, int32_t *signed_distance2,
                         uint32_t *owners, uint32_t *output,
                         int width, int height, float threshold,
                         uint32_t low_material, uint32_t high_material,
                         uint32_t seed) {
    int count = width * height;
    uint8_t *to_high = calloc((size_t)count, sizeof(*to_high));
    uint8_t *to_low = calloc((size_t)count, sizeof(*to_low));
    int32_t *high = calloc((size_t)count, sizeof(*high));
    int32_t *low = calloc((size_t)count, sizeof(*low));
    int32_t *scratch = calloc((size_t)count, sizeof(*scratch));
    uint32_t *owner_scratch = calloc((size_t)count, sizeof(*owner_scratch));
    if (!to_high || !to_low || !high || !low || !scratch || !owner_scratch)
        abort();
    for (int i = 0; i < count; ++i) {
        int inside = heightmap[i] >= threshold;
        labels[i] = inside ? high_material : low_material;
        to_high[i] = (uint8_t)!inside;
        to_low[i] = (uint8_t)inside;
    }
    bake_jfa_cpu(to_high, high, scratch, width, height);
    bake_jfa_cpu(to_low, low, scratch, width, height);
    for (int i = 0; i < count; ++i) {
        signed_distance2[i] = heightmap[i] >= threshold
            ? distance2(i, low[i], width)
            : -distance2(i, high[i], width);
    }
    bake_quilt_cpu(atlas, labels, output, owners, width, height, seed);
    bake_patchmatch_cpu(atlas, labels, owners, owner_scratch, width, height,
                        seed ^ 0xa511e9b3u, 4);
    for (int i = 0; i < count; ++i) output[i] = atlas[owners[i]];
    free(owner_scratch); free(scratch); free(low); free(high);
    free(to_low); free(to_high);
}

void bake_transition_gpu(BakeGpu *gpu, const BakeBuffer *heightmap,
                         const BakeBuffer *atlas, const BakeBuffer *labels,
                         const BakeBuffer *signed_distance2,
                         const BakeBuffer *owners, const BakeBuffer *output,
                         int width, int height, float threshold,
                         uint32_t low_material, uint32_t high_material,
                         uint32_t seed) {
    size_t bytes = (size_t)width * height * sizeof(uint32_t);
    BakeBuffer mask_high = bake_buffer_host(gpu, bytes);
    BakeBuffer mask_low = bake_buffer_host(gpu, bytes);
    BakeBuffer high_owner = bake_buffer_host(gpu, bytes);
    BakeBuffer low_owner = bake_buffer_host(gpu, bytes);
    struct {
        uint32_t width, height;
        float threshold;
        uint32_t low_material, high_material;
    } push = {(uint32_t)width, (uint32_t)height, threshold,
              low_material, high_material};
    uint32_t gx = (uint32_t)(width + 7) / 8;
    uint32_t gy = (uint32_t)(height + 7) / 8;
    BakePipeline seed_pipeline = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_transition_seed.comp.spv", 4, sizeof(push));
    BakeBuffer seed_buffers[4] = {*heightmap, *labels, mask_high, mask_low};
    bake_dispatch(gpu, &seed_pipeline, seed_buffers, 4, &push, sizeof(push),
                  gx, gy, 1);
    bake_pipeline_destroy(gpu, &seed_pipeline);
    bake_jfa_gpu(gpu, &mask_high, &high_owner, width, height);
    bake_jfa_gpu(gpu, &mask_low, &low_owner, width, height);

    BakePipeline finish = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_transition_finish.comp.spv", 4,
        3 * sizeof(uint32_t));
    struct { uint32_t width, height; float threshold; } finish_push = {
        (uint32_t)width, (uint32_t)height, threshold};
    BakeBuffer finish_buffers[4] = {
        *heightmap, high_owner, low_owner, *signed_distance2};
    bake_dispatch(gpu, &finish, finish_buffers, 4, &finish_push,
                  sizeof(finish_push), gx, gy, 1);
    bake_pipeline_destroy(gpu, &finish);

    bake_quilt_gpu(gpu, atlas, labels, output, owners, width, height, seed);
    bake_patchmatch_gpu(gpu, atlas, labels, owners, output, width, height,
                        seed ^ 0xa511e9b3u, 4);
    bake_buffer_destroy(gpu, &low_owner);
    bake_buffer_destroy(gpu, &high_owner);
    bake_buffer_destroy(gpu, &mask_low);
    bake_buffer_destroy(gpu, &mask_high);
}
