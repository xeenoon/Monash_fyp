// Stage 9: orchestrate all five terrain material presets end-to-end.
#ifndef BAKE_PIPELINE_H
#define BAKE_PIPELINE_H

#include <stdint.h>
#include "bake_gpu.h"

enum {
    BAKE_FULL_SNOW = 0,
    BAKE_SNOW_ROCK = 1,
    BAKE_FULL_ROCK = 2,
    BAKE_ROCK_GRASS = 3,
    BAKE_FULL_GRASS = 4,
    BAKE_PRESET_COUNT = 5
};
extern const char *const bake_preset_names[BAKE_PRESET_COUNT];

// Atlas layout is three packed-RGBA planes: rock, grass, snow. Outputs are
// five consecutive width*height packed-RGBA planes in preset-name order.
void bake_pipeline_cpu(const uint32_t *atlas, const float *heightmap,
                       uint32_t *outputs, int width, int height,
                       float threshold, uint32_t seed);
void bake_pipeline_gpu(BakeGpu *gpu, const BakeBuffer *atlas,
                       const BakeBuffer *heightmap, BakeBuffer outputs[BAKE_PRESET_COUNT],
                       int width, int height, float threshold, uint32_t seed);

#endif
