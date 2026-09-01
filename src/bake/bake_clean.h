// Stage 4: anomaly removal and nearest-valid-pixel inpainting.
#ifndef BAKE_CLEAN_H
#define BAKE_CLEAN_H

#include <stdint.h>

#include "bake_gpu.h"

// CPU definition used as the Option-A reference. RGBA pixels use the baker's
// byte packing. All output arrays contain width*height elements.
void bake_clean_cpu(const uint32_t *input, uint32_t *cleaned,
                    uint32_t *confidence, uint32_t *hard_mask,
                    int width, int height);

// Complete GPU-resident Stage-4 implementation. `input` and all outputs are
// host-visible BakeBuffers so tests/orchestration can inspect or chain them.
void bake_clean_gpu(BakeGpu *gpu, const BakeBuffer *input,
                    const BakeBuffer *cleaned, const BakeBuffer *confidence,
                    const BakeBuffer *hard_mask, int width, int height);

#endif
