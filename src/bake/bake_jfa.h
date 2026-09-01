// Jump-flooding nearest-seed transform (Option A definition, replacing scipy's
// exact distance_transform_edt).  For every pixel it finds the index of the
// nearest *seed* pixel (a seed = a pixel where mask==0, i.e. "keep"); mask==1
// pixels ("holes") are filled from their nearest seed.  Integer squared
// distance and a fixed neighbour order make it bit-identical CPU vs GPU.
#ifndef BAKE_JFA_H
#define BAKE_JFA_H

#include <stdint.h>
#include "bake_gpu.h"

// Number of ping-pong passes for a width×height field (log2 of the start step,
// plus one).  The host uses this for both the C reference and the GPU dispatch
// sequence so they step identically.
int bake_jfa_pass_count(int width, int height);

// Reference JFA.  `mask` is width*height (non-zero = hole to fill).  Writes the
// nearest seed's linear index into `owner` (seed pixels own themselves).
void bake_jfa_cpu(const uint8_t *mask, int32_t *owner, int32_t *scratch,
                  int width, int height);

// GPU orchestration for a uint32 mask buffer (non-zero = hole). The final
// owner field is always returned in `owner`, independent of ping-pong parity.
void bake_jfa_gpu(BakeGpu *gpu, const BakeBuffer *mask, const BakeBuffer *owner,
                  int width, int height);

#endif  // BAKE_JFA_H
