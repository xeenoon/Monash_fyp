// Stage 8: height-driven two-material SDF and PatchMatch transition quilt.
#ifndef BAKE_TRANSITION_H
#define BAKE_TRANSITION_H
#include <stdint.h>
#include "bake_gpu.h"
void bake_transition_cpu(const float *heightmap, const uint32_t *atlas,
                         uint32_t *labels, int32_t *signed_distance2,
                         uint32_t *owners, uint32_t *output,
                         int width, int height, float threshold,
                         uint32_t low_material, uint32_t high_material,
                         uint32_t seed);
void bake_transition_gpu(BakeGpu *gpu, const BakeBuffer *heightmap,
                         const BakeBuffer *atlas, const BakeBuffer *labels,
                         const BakeBuffer *signed_distance2,
                         const BakeBuffer *owners, const BakeBuffer *output,
                         int width, int height, float threshold,
                         uint32_t low_material, uint32_t high_material,
                         uint32_t seed);
#endif
