// Ordered minimum-error seam primitives used by the pass17-compatible quilt.
#ifndef BAKE_SEAM_H
#define BAKE_SEAM_H

#include <stdbool.h>
#include <stdint.h>

// Exact operation order of procedural_gap_demo.minimum_vertical_seam.
// `cost` is row-major height*width. `seam[y]` receives one column per row.
bool bake_minimum_vertical_seam(const float *cost, int width, int height,
                                int16_t *seam);

// Pass17 residual/RGB seam ownership once the per-pixel cost is known.
// The output is one byte per patch pixel: non-zero means the candidate owns it.
bool bake_quilt_take_mask_from_cost(const float *cost, const uint8_t *known,
                                    int patch, int overlap, bool has_left,
                                    bool has_top, uint8_t *take);

#endif
