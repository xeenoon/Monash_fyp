// Stage 5: exact Gaussian pyramid and mid/meso/fine residual bands.
#ifndef BAKE_BANDS_H
#define BAKE_BANDS_H

#include <stdint.h>

#include "bake_gpu.h"

typedef struct {
    float *rgb;
    float *low32;
    float *mid;
    float *meso;
    float *fine;
} BakeBandsCpu;

void bake_bands_cpu(const uint32_t *input, BakeBandsCpu bands,
                    int width, int height);

// All output buffers contain width*height*3 floats.
void bake_bands_gpu(BakeGpu *gpu, const BakeBuffer *input,
                    const BakeBuffer *rgb, const BakeBuffer *low32,
                    const BakeBuffer *mid, const BakeBuffer *meso,
                    const BakeBuffer *fine, int width, int height);

#endif
