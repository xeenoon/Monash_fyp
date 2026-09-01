// Pass17 CPU reference for state-dependent texture retrieval scores.
// The expensive residual extraction/comparison remains GPU-batched; this
// module establishes exact candidate semantics before the coarse pass moves
// into a compute shader.
#ifndef BAKE_TEXTURE_SCORE_H
#define BAKE_TEXTURE_SCORE_H

#include <stdint.h>

#include "bake_texture_database.h"

typedef struct {
    const BakeTerrainSample *target;
    const BakeTerrainSample *north;
    const BakeTerrainSample *east;
    const BakeTerrainSample *donors;
    int donor_count;
    const uint32_t *target_labels;
    const uint32_t *generated;
    const uint32_t *known;
    const int32_t *donor_map;
    const int32_t *source_x;
    const int32_t *source_y;
    const int32_t *source_usage;
    int width;
    int height;
} BakeTextureScoreInput;

// Fills one value per database record and returns the target class density.
void bake_texture_descriptor_scores(const BakeTextureDatabase *database,
                                    const BakeTextureScoreInput *input,
                                    int target_x, int target_y, int level,
                                    float *descriptor, float *coherence,
                                    float *room, float target_density[3]);

// Pass17 uses global 48 candidates at level zero, then unique(local 32,
// global 16). Returns the candidate count (at most 48).
int bake_texture_shortlist(const float *descriptor, const float *coherence,
                           const float *room, int database_count, int level,
                           int *shortlist);

// Produces candidate records and every final score term except residual RGB
// overlap/gradient/scale, which bake_residual_score_gpu adds in parallel.
void bake_texture_base_scores(const BakeTextureDatabase *database,
                              const BakeTextureScoreInput *input,
                              const float *coherence, const float *room,
                              const float target_density[3], int target_x,
                              int target_y, int level, const int *shortlist,
                              int shortlist_count,
                              BakeResidualCandidate *candidates,
                              float *base_scores);

#endif
