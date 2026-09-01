// Bit-exact bilinear height-field resampling shared by the C and GPU paths.
#ifndef BAKE_RESAMPLE_H
#define BAKE_RESAMPLE_H

#include "bake_gpu.h"

void bake_resample_bilinear_cpu(const float *input, float *output,
                                int source_width, int source_height,
                                int output_width, int output_height);

void bake_resample_bilinear_gpu(BakeGpu *gpu, const BakeBuffer *input,
                                const BakeBuffer *output,
                                int source_width, int source_height,
                                int output_width, int output_height);

#endif
