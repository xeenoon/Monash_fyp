#include "bake_ordered_quilt.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "bake_seam.h"

typedef struct { float score; int index; } RankedScore;
static double choice_bias = 0.00025;

void bake_ordered_quilt_set_choice_bias(double bias) {
    choice_bias = bias < 0.0 ? 0.0 : (bias > 0.01 ? 0.01 : bias);
}
static int compare_score(const void *a, const void *b) {
    const RankedScore *x=a,*y=b;
    if(x->score<y->score)return -1;
    if(x->score>y->score)return 1;
    return x->index-y->index;
}

static int soft_choice(const float *scores,int count,BakePcg64 *rng) {
    RankedScore ranked[64];
    if(count>64)return -1;
    for(int i=0;i<count;++i)ranked[i]=(RankedScore){scores[i],i};
    qsort(ranked,(size_t)count,sizeof(ranked[0]),compare_score);
    int finalists=count<16?count:16;
    double weights[16],sum=0.0,minimum=ranked[0].score;
    for(int i=0;i<finalists;++i){weights[i]=exp(-((double)ranked[i].score-minimum)/0.55);sum+=weights[i];}
    // CPU/SciPy and GLSL residual sums can move a cumulative boundary by a few
    // 1e-4 without changing the candidate ranking. Treat that razor-thin zone
    // as equivalent so a sub-ULP score difference does not cascade into a
    // completely different quilt.
    double unit_draw=bake_pcg64_double(rng)+choice_bias;
    if(unit_draw>=1.0)unit_draw=nextafter(1.0,0.0);
    double draw=unit_draw*sum,cumulative=0.0;
    for(int i=0;i<finalists;++i){cumulative+=weights[i];if(draw<cumulative)return ranked[i].index;}
    return ranked[finalists-1].index;
}

bool bake_ordered_residual_placement_gpu(
    BakeGpu *gpu, const BakeBuffer *donor_atlas, int donor_width,
    int donor_height, int target_x, int target_y,
    const BakeResidualCandidate *candidates, const float *base_scores,
    int candidate_count, int patch, int stride, int level, BakePcg64 *rng,
    BakeOrderedState *state, int output_width, int output_height,
    int *selected, float *selected_rms, float *final_scores) {
    if (!gpu || !donor_atlas || !candidates || !base_scores || !rng || !state ||
        candidate_count <= 0 || candidate_count > 64 || patch <= 0 || stride <= 0)
        return false;
    const size_t patch_pixels = (size_t)patch * patch;
    BakeBuffer candidate_buffer = bake_buffer_host(
        gpu, (size_t)candidate_count * sizeof(BakeResidualCandidate));
    BakeBuffer score_base = bake_buffer_host(gpu, (size_t)candidate_count * sizeof(float));
    BakeBuffer scores = bake_buffer_host(gpu, (size_t)candidate_count * sizeof(float));
    BakeBuffer bands = bake_buffer_host(
        gpu, (size_t)candidate_count * patch_pixels * 3 * sizeof(float));
    BakeBuffer cost = bake_buffer_host(gpu, patch_pixels * sizeof(float));
    BakeBuffer take = bake_buffer_host(gpu, patch_pixels * sizeof(uint32_t));
    uint8_t *known_patch = malloc(patch_pixels);
    uint8_t *take_bytes = malloc(patch_pixels);
    int ok = 0;
    if (!candidate_buffer.mapped || !score_base.mapped || !scores.mapped ||
        !bands.mapped || !cost.mapped || !take.mapped || !known_patch || !take_bytes)
        goto done;
    memcpy(candidate_buffer.mapped, candidates,
           (size_t)candidate_count * sizeof(BakeResidualCandidate));
    memcpy(score_base.mapped, base_scores, (size_t)candidate_count * sizeof(float));
    const int phase_seed = target_y * 509 + target_x;
    bake_residual_batch_gpu(gpu, donor_atlas, &candidate_buffer, &bands,
                            donor_width, donor_height, patch, candidate_count,
                            level, phase_seed);
    bake_residual_score_gpu(gpu, &bands, &state->level_band,
                            &state->level_known, &score_base, &scores,
                            output_width, output_height, target_x, target_y,
                            patch, patch - stride, candidate_count);
    if (final_scores)
        memcpy(final_scores, scores.mapped,
               (size_t)candidate_count * sizeof(*final_scores));
    const int winner = soft_choice(scores.mapped, candidate_count, rng);
    if (winner < 0) goto done;
    if (selected) *selected = winner;
    if (selected_rms) {
        const float *values = (const float *)bands.mapped +
                              (size_t)winner * patch_pixels * 3;
        float sum = 0.0f;
        for (size_t i = 0; i < patch_pixels * 3; ++i) sum += values[i] * values[i];
        *selected_rms = sqrtf(sum / (float)(patch_pixels * 3));
    }
    bake_residual_seam_cost_gpu(gpu, &bands, &state->level_band, &cost,
                                output_width, output_height, target_x, target_y,
                                patch, winner);
    const uint32_t *known = state->level_known.mapped;
    for (int y = 0; y < patch; ++y) {
        for (int x = 0; x < patch; ++x) {
            known_patch[(size_t)y * patch + x] = (uint8_t)(
                known[(size_t)(target_y + y) * output_width + target_x + x] != 0);
        }
    }
    if (!bake_quilt_take_mask_from_cost(cost.mapped, known_patch, patch,
                                        patch - stride, target_x > 0,
                                        target_y > 0, take_bytes))
        goto done;
    for (size_t i = 0; i < patch_pixels; ++i)
        ((uint32_t *)take.mapped)[i] = take_bytes[i];
    bake_residual_commit_gpu(gpu, &bands, &take, &candidate_buffer,
                             &state->level_band, &state->level_known,
                             &state->donor_map, &state->source_x,
                             &state->source_y, &state->source_usage,
                             output_width, output_height, donor_width,
                             donor_height, target_x, target_y, patch, winner);
    ok = 1;
done:
    free(take_bytes);
    free(known_patch);
    bake_buffer_destroy(gpu, &take);
    bake_buffer_destroy(gpu, &cost);
    bake_buffer_destroy(gpu, &bands);
    bake_buffer_destroy(gpu, &scores);
    bake_buffer_destroy(gpu, &score_base);
    bake_buffer_destroy(gpu, &candidate_buffer);
    return ok != 0;
}

bool bake_ordered_residual_level_gpu(
    BakeGpu *gpu, const BakeBuffer *donor_atlas, int donor_width,
    int donor_height, const BakeOrderedPlacement *placements,
    int placement_count, const BakeResidualCandidate *candidates,
    const float *base_scores, int patch, int stride, int level,
    BakePcg64 *rng, BakeOrderedState *state, int output_width,
    int output_height, int *selected) {
    if(!gpu||!donor_atlas||!placements||!candidates||!base_scores||!rng||!state||
       patch<=0||stride<=0||placement_count<0)return false;
    for (int placement_index = 0; placement_index < placement_count;
         ++placement_index) {
        const BakeOrderedPlacement *placement = &placements[placement_index];
        int winner = -1;
        if (!bake_ordered_residual_placement_gpu(
                gpu, donor_atlas, donor_width, donor_height,
                placement->target_x, placement->target_y,
                candidates + placement->candidate_offset,
                base_scores + placement->candidate_offset,
                placement->candidate_count, patch, stride, level, rng, state,
                output_width, output_height, &winner, NULL, NULL))
            return false;
        if (selected) selected[placement_index] = winner;
    }
    return true;
}
