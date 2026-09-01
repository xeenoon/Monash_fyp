// Stage 7: broad-colour re-anchor plus material-aware texture-band relayer.
#ifndef BAKE_RELAYER_H
#define BAKE_RELAYER_H

#include <stdint.h>

#include "bake_gpu.h"

// Each atlas band stores 3 material planes * width*height * RGB floats.
void bake_relayer_cpu(const uint32_t *base, const uint32_t *labels,
                      const uint32_t *owners, const float *low,
                      const float *mid, const float *meso, const float *fine,
                      uint32_t *output, int width, int height);
void bake_relayer_gpu(BakeGpu *gpu, const BakeBuffer *base,
                      const BakeBuffer *labels, const BakeBuffer *owners,
                      const BakeBuffer *low, const BakeBuffer *mid,
                      const BakeBuffer *meso, const BakeBuffer *fine,
                      const BakeBuffer *output, int width, int height);

#endif
