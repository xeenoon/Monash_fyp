// GPU-batched native-frequency residual extraction for pass17 shortlist patches.
#ifndef BAKE_RESIDUAL_H
#define BAKE_RESIDUAL_H

#include <stdint.h>

#include "bake_gpu.h"

typedef struct {
    int32_t donor, source_x, source_y, database_index;
} BakeResidualCandidate;

// `donor_atlas` stores donor_count packed RGBA 256x256 planes. `candidates`
// stores BakeResidualCandidate records. `output` receives candidate_count
// float RGB patch planes. Level is 0/1/2 for pass17's 96/48/24 px bands.
void bake_residual_batch_gpu(BakeGpu *gpu, const BakeBuffer *donor_atlas,
                             const BakeBuffer *candidates,
                             const BakeBuffer *output, int donor_width,
                             int donor_height, int patch, int candidate_count,
                             int level, int phase_seed);

// Adds pass17 rgb_patch_error terms to `base_scores` for a shortlist. Existing
// and level-known are full output-tile fields; bands contains the batch above.
void bake_residual_score_gpu(BakeGpu *gpu, const BakeBuffer *bands,
                             const BakeBuffer *existing,
                             const BakeBuffer *level_known,
                             const BakeBuffer *base_scores,
                             const BakeBuffer *output_scores, int output_width,
                             int output_height, int target_x, int target_y,
                             int patch, int overlap, int candidate_count);

void bake_residual_seam_cost_gpu(BakeGpu *gpu, const BakeBuffer *bands,
                                 const BakeBuffer *existing,
                                 const BakeBuffer *cost, int output_width,
                                 int output_height, int target_x, int target_y,
                                 int patch, int selected_candidate);

void bake_residual_commit_gpu(BakeGpu *gpu, const BakeBuffer *bands,
                              const BakeBuffer *take_mask,
                              const BakeBuffer *candidates,
                              const BakeBuffer *existing,
                              const BakeBuffer *level_known,
                              const BakeBuffer *donor_map,
                              const BakeBuffer *source_x,
                              const BakeBuffer *source_y,
                              const BakeBuffer *source_usage, int output_width,
                              int output_height, int donor_width,
                              int donor_height, int target_x, int target_y,
                              int patch, int selected_candidate);

#endif
