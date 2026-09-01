// Integer, Jacobi-style PatchMatch refinement used by transition quilts.
#ifndef BAKE_PATCHMATCH_H
#define BAKE_PATCHMATCH_H

#include <stdint.h>

#include "bake_gpu.h"

void bake_patchmatch_cpu(const uint32_t *atlas, const uint32_t *labels,
                         uint32_t *owners, uint32_t *scratch,
                         int width, int height, uint32_t seed, int iterations);
void bake_patchmatch_gpu(BakeGpu *gpu, const BakeBuffer *atlas,
                         const BakeBuffer *labels, const BakeBuffer *owners,
                         const BakeBuffer *output, int width, int height,
                         uint32_t seed, int iterations);

#endif
