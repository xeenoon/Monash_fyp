// Pass17 material-exemplar microtexture relayer.
#ifndef BAKE_PASS17_RELAYER_H
#define BAKE_PASS17_RELAYER_H

#include <stdbool.h>
#include <stdint.h>

#include "bake_gpu.h"
#include "bake_sample.h"

bool bake_pass17_material_relayer_gpu(
    BakeGpu *gpu, const uint32_t *base, const uint32_t *labels,
    const BakeTerrainSample *texture_donors, int texture_donor_count,
    const BakeTerrainSample *material_sources[3],
    const int material_source_counts[3], uint32_t *output,
    int width, int height);

#endif
