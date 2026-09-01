#include "bake_residual.h"

#include <math.h>
#include <stdint.h>

#include "bake_filter.h"

void bake_residual_batch_gpu(BakeGpu *gpu, const BakeBuffer *donor_atlas,
                             const BakeBuffer *candidates,
                             const BakeBuffer *output, int donor_width,
                             int donor_height, int patch, int candidate_count,
                             int level, int phase_seed) {
    size_t values = (size_t)candidate_count * patch * patch * 3;
    BakeBuffer shifted = bake_buffer_host(gpu, values * sizeof(float));
    BakeBuffer small = bake_buffer_host(gpu, values * sizeof(float));
    BakeBuffer large = bake_buffer_host(gpu, values * sizeof(float));
    BakeBuffer scratch = bake_buffer_host(gpu, values * sizeof(float));

    struct {
        uint32_t donor_width, donor_height, patch, candidate_count;
        int32_t level, phase_seed;
    } extract_push = {(uint32_t)donor_width, (uint32_t)donor_height,
                      (uint32_t)patch, (uint32_t)candidate_count,
                      level, phase_seed};
    BakePipeline extract = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_residual_extract.comp.spv", 3,
        sizeof(extract_push));
    BakeBuffer extract_buffers[3] = {*donor_atlas, *candidates, shifted};
    bake_dispatch(gpu, &extract, extract_buffers, 3, &extract_push,
                  sizeof(extract_push), (uint32_t)(patch + 7) / 8,
                  (uint32_t)(patch + 7) / 8, (uint32_t)candidate_count);
    bake_pipeline_destroy(gpu, &extract);

    float small_sigma = level == 1 ? 1.2f : 2.0f;
    bake_gaussian_batch_gpu(gpu, &shifted, &small, &scratch, patch, patch, 3,
                            candidate_count, small_sigma, 4.0f, 1);
    if (level < 2) {
        float large_sigma = level == 0 ? 16.0f : 6.0f;
        bake_gaussian_batch_gpu(gpu, &shifted, &large, &scratch, patch, patch,
                                3, candidate_count, large_sigma, 4.0f, 1);
    }

    struct { uint32_t patch, candidate_count, level; } finish_push = {
        (uint32_t)patch, (uint32_t)candidate_count, (uint32_t)level};
    BakePipeline finish = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_residual_finish.comp.spv", 4,
        sizeof(finish_push));
    BakeBuffer finish_buffers[4] = {shifted, small, large, *output};
    bake_dispatch(gpu, &finish, finish_buffers, 4, &finish_push,
                  sizeof(finish_push), (uint32_t)candidate_count, 1, 1);
    bake_pipeline_destroy(gpu, &finish);

    bake_buffer_destroy(gpu, &scratch);
    bake_buffer_destroy(gpu, &large);
    bake_buffer_destroy(gpu, &small);
    bake_buffer_destroy(gpu, &shifted);
}

void bake_residual_score_gpu(BakeGpu *gpu, const BakeBuffer *bands,
                             const BakeBuffer *existing,
                             const BakeBuffer *level_known,
                             const BakeBuffer *base_scores,
                             const BakeBuffer *output_scores, int output_width,
                             int output_height, int target_x, int target_y,
                             int patch, int overlap, int candidate_count) {
    struct {
        uint32_t output_width, output_height, target_x, target_y;
        uint32_t patch, overlap, candidate_count;
    } push = {(uint32_t)output_width, (uint32_t)output_height,
              (uint32_t)target_x, (uint32_t)target_y, (uint32_t)patch,
              (uint32_t)overlap, (uint32_t)candidate_count};
    BakePipeline pipeline = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_residual_score.comp.spv", 5, sizeof(push));
    BakeBuffer buffers[5] = {*bands, *existing, *level_known,
                             *base_scores, *output_scores};
    bake_dispatch(gpu, &pipeline, buffers, 5, &push, sizeof(push),
                  (uint32_t)candidate_count, 1, 1);
    bake_pipeline_destroy(gpu, &pipeline);
}

void bake_residual_seam_cost_gpu(BakeGpu *gpu, const BakeBuffer *bands,
                                 const BakeBuffer *existing,
                                 const BakeBuffer *cost, int output_width,
                                 int output_height, int target_x, int target_y,
                                 int patch, int selected_candidate) {
    struct {
        uint32_t output_width, output_height, target_x, target_y;
        uint32_t patch, selected_candidate;
    } push = {(uint32_t)output_width, (uint32_t)output_height,
              (uint32_t)target_x, (uint32_t)target_y, (uint32_t)patch,
              (uint32_t)selected_candidate};
    BakePipeline pipeline = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_residual_seam_cost.comp.spv", 3, sizeof(push));
    BakeBuffer buffers[3] = {*bands, *existing, *cost};
    bake_dispatch(gpu, &pipeline, buffers, 3, &push, sizeof(push),
                  (uint32_t)(patch + 7) / 8, (uint32_t)(patch + 7) / 8, 1);
    bake_pipeline_destroy(gpu, &pipeline);
}

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
                              int patch, int selected_candidate) {
    struct {
        uint32_t output_width, output_height, donor_width, donor_height;
        uint32_t target_x, target_y, patch, selected_candidate;
    } push = {(uint32_t)output_width, (uint32_t)output_height,
              (uint32_t)donor_width, (uint32_t)donor_height,
              (uint32_t)target_x, (uint32_t)target_y, (uint32_t)patch,
              (uint32_t)selected_candidate};
    BakePipeline pipeline = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_residual_commit.comp.spv", 9, sizeof(push));
    BakeBuffer buffers[9] = {*bands, *take_mask, *candidates, *existing,
                             *level_known, *donor_map, *source_x, *source_y,
                             *source_usage};
    bake_dispatch(gpu, &pipeline, buffers, 9, &push, sizeof(push),
                  (uint32_t)(patch + 7) / 8, (uint32_t)(patch + 7) / 8, 1);
    bake_pipeline_destroy(gpu, &pipeline);
}
