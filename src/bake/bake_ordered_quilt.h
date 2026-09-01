// Ordered pass17 residual-quilt scheduler over GPU-parallel shortlist work.
#ifndef BAKE_ORDERED_QUILT_H
#define BAKE_ORDERED_QUILT_H

#include <stdbool.h>
#include <stdint.h>

#include "bake_gpu.h"
#include "bake_random.h"
#include "bake_residual.h"

typedef struct {
    int target_x, target_y;
    int candidate_offset, candidate_count;
} BakeOrderedPlacement;

typedef struct {
    BakeBuffer level_band;   // output_width*output_height*3 floats
    BakeBuffer level_known;  // uint32 per output pixel
    BakeBuffer donor_map;    // int32 per output pixel
    BakeBuffer source_x;     // int32 per output pixel
    BakeBuffer source_y;     // int32 per output pixel
    BakeBuffer source_usage; // donor_count*donor_width*donor_height int32
} BakeOrderedState;

// Small normalized cumulative-probability dead zone used to keep equivalent
// CPU/GPU scores from crossing a stochastic boundary. Default: 0.00025.
void bake_ordered_quilt_set_choice_bias(double bias);

// Evaluate and commit one placement. This is the primitive used by the real
// pass17 driver because shortlist/base scores depend on the state left by the
// immediately preceding placement.
bool bake_ordered_residual_placement_gpu(
    BakeGpu *gpu, const BakeBuffer *donor_atlas, int donor_width,
    int donor_height, int target_x, int target_y,
    const BakeResidualCandidate *candidates, const float *base_scores,
    int candidate_count, int patch, int stride, int level, BakePcg64 *rng,
    BakeOrderedState *state, int output_width, int output_height,
    int *selected, float *selected_rms, float *final_scores);

// Candidate arrays are concatenated placement shortlists. base_scores already
// contains every non-RGB pass17 term. One winner is fully committed before the
// next placement is evaluated. `selected` may be NULL.
bool bake_ordered_residual_level_gpu(
    BakeGpu *gpu, const BakeBuffer *donor_atlas, int donor_width,
    int donor_height, const BakeOrderedPlacement *placements,
    int placement_count, const BakeResidualCandidate *candidates,
    const float *base_scores, int patch, int stride, int level,
    BakePcg64 *rng, BakeOrderedState *state, int output_width,
    int output_height, int *selected);

#endif
