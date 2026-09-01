// Pass17 terrain sample loader: cropped imagery, cleaned labels and DEM fields.
#ifndef BAKE_SAMPLE_H
#define BAKE_SAMPLE_H

#include <stdbool.h>
#include <stdint.h>

#include "bake_gpu.h"

typedef struct {
    int tile_x, tile_y;
    int width, height;
    uint32_t *rgb;
    uint32_t *labels;
    float *elevation;
    float *slope;
    float *gradient_x;
    float *gradient_y;
    float repair_fraction;
    float hard_fraction;
} BakeTerrainSample;

bool bake_sample_load(BakeGpu *gpu, const char *dataset, int tile_x, int tile_y,
                      bool clean_rgb, BakeTerrainSample *sample);
void bake_sample_free(BakeTerrainSample *sample);

#endif
