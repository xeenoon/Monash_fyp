// Stage 6: parallel, material-aware exemplar quilt.
#ifndef BAKE_QUILT_H
#define BAKE_QUILT_H

#include <stdint.h>

#include "bake_gpu.h"

enum { BAKE_QUILT_PATCH = 32, BAKE_QUILT_STRIDE = 24 };

// `atlas` stores three width*height RGBA planes in ROCK,GRASS,SNOW order.
// `labels` are uint32 material IDs. `owners` receives the chosen atlas index,
// making seams and later band sampling reproducible without copying metadata.
void bake_quilt_cpu(const uint32_t *atlas, const uint32_t *labels,
                    uint32_t *output, uint32_t *owners,
                    int width, int height, uint32_t seed);
void bake_quilt_gpu(BakeGpu *gpu, const BakeBuffer *atlas,
                    const BakeBuffer *labels, const BakeBuffer *output,
                    const BakeBuffer *owners, int width, int height,
                    uint32_t seed);

#endif
