// Pass17 intact-donor patch database shared by coarse retrieval and shortlist scoring.
#ifndef BAKE_TEXTURE_DATABASE_H
#define BAKE_TEXTURE_DATABASE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "bake_residual.h"
#include "bake_sample.h"

enum { BAKE_TEXTURE_THUMB = 8, BAKE_TEXTURE_CONTEXT = 4 };

typedef struct {
    int patch, source_stride, donor_count, count;
    BakeResidualCandidate *records;
    uint8_t *thumbnail;       // count*8*8 labels
    float *density;           // count*3
    float *terrain;           // mean height, height stddev, mean slope
    float *terrain_angle;     // axial angle, anisotropy
    float *mean_lab;          // count*3
    float *source_position;   // count*2, in tile coordinates
    uint8_t *north_context;   // count*4*8*3
    uint8_t *east_context;    // count*8*4*3
    uint8_t *north_valid;
    uint8_t *east_valid;
    uint8_t *top_edge;        // count*8*3
    int16_t *top_gradient;    // count*8*3
    uint8_t *right_edge;      // count*8*3
    int16_t *right_gradient;  // count*8*3
} BakeTextureDatabase;

bool bake_texture_database_build(const BakeTerrainSample *donors,
                                 int donor_count, int patch, int source_stride,
                                 BakeTextureDatabase *database);
void bake_texture_database_free(BakeTextureDatabase *database);

#endif
