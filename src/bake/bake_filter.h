// CPU reference for the GPU filter primitives (Option A: the C reference is the
// spec; the GLSL kernels reproduce it bit-for-bit).  Not a clone of scipy —
// a fresh, GPU-friendly definition: fixed truncated-Gaussian radius, float32,
// clamp ("nearest") boundary, separable H-then-V.
//
// Bit-exactness contract:
//   * weights are computed here once and *uploaded* to the GPU, so exp() never
//     runs on the device (libm vs GLSL exp would diverge);
//   * this file is compiled with -ffp-contract=off and the shader marks the
//     accumulator `precise`, so both do the same IEEE-754 float32 mul-then-add
//     in the same tap order — the result is identical on every conformant GPU.
#ifndef BAKE_FILTER_H
#define BAKE_FILTER_H

#include "bake_gpu.h"

#include <stdint.h>

// radius = round(truncate * sigma); kernel length is 2*radius + 1.
int bake_gaussian_radius(float sigma, float truncate);

// Fill `weights` (length 2*radius+1) with a normalised float32 Gaussian.
void bake_gaussian_weights(float sigma, int radius, float *weights);

// Separable Gaussian on interleaved float data (channels per pixel), clamp
// boundary.  `scratch` must hold width*height*channels floats.
void bake_gaussian_cpu(const float *in, float *out, float *scratch,
                       int width, int height, int channels,
                       const float *weights, int radius);
void bake_gaussian_cpu_reflect(const float *in, float *out, float *scratch,
                               int width, int height, int channels,
                               const float *weights, int radius);

// Dispatch the exact same two passes on the GPU. The caller owns three
// distinct, correctly-sized buffers; input and output remain GPU-resident.
void bake_gaussian_gpu(BakeGpu *gpu, const BakeBuffer *input,
                       const BakeBuffer *output, const BakeBuffer *scratch,
                       int width, int height, int channels, float sigma,
                       float truncate);
void bake_gaussian_gpu_reflect(BakeGpu *gpu, const BakeBuffer *input,
                               const BakeBuffer *output,
                               const BakeBuffer *scratch, int width, int height,
                               int channels, float sigma, float truncate);

// Same operation over `layers` tightly packed image planes. The existing
// pass17 residual and material-exemplar paths use this to amortize shortlist
// and donor-pool filtering across the GPU.
void bake_gaussian_batch_gpu(BakeGpu *gpu, const BakeBuffer *input,
                             const BakeBuffer *output,
                             const BakeBuffer *scratch, int width, int height,
                             int channels, int layers, float sigma,
                             float truncate, int reflect);

#endif  // BAKE_FILTER_H
