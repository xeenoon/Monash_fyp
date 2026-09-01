#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "bake_sample.h"

#ifndef BAKE_DATASET_DIR
#error BAKE_DATASET_DIR is required
#endif

int main(void) {
    BakeGpu gpu;
    if (!bake_gpu_init(&gpu)) return 77;
    BakeTerrainSample sample;
    if (!bake_sample_load(&gpu, BAKE_DATASET_DIR, 25, 30, true, &sample)) return 1;
    size_t counts[3] = {0};
    double height_sum = 0.0, slope_sum = 0.0;
    size_t count = (size_t)sample.width * sample.height;
    for (size_t i = 0; i < count; ++i) {
        if (sample.labels[i] < 3) ++counts[sample.labels[i]];
        height_sum += sample.elevation[i];
        slope_sum += sample.slope[i];
    }
    printf("sample 25/30 repair=%.9f hard=%.9f labels=%zu,%zu,%zu "
           "height_mean=%.9f slope_mean=%.9f\n",
           sample.repair_fraction, sample.hard_fraction,
           counts[0], counts[1], counts[2], height_sum / count, slope_sum / count);
    // Mask fractions and labels are discrete pass17 contracts. Float DEM
    // summaries allow a small GPU resampling tolerance.
    size_t reference_counts[3] = {62474, 2979, 83};
    size_t label_count_error = 0;
    for (int kind = 0; kind < 3; ++kind)
        label_count_error += counts[kind] > reference_counts[kind] ?
            counts[kind] - reference_counts[kind] : reference_counts[kind] - counts[kind];
    int failed = sample.width != 256 || sample.height != 256 ||
        fabsf(sample.repair_fraction - 0.038116455078125f) > 1e-8f ||
        fabsf(sample.hard_fraction - 0.0330810546875f) > 1e-8f ||
        label_count_error != 0 ||
        fabs(height_sum / count - 3095.813232421875) > 0.01;
    bake_sample_free(&sample);
    bake_gpu_destroy(&gpu);
    return failed;
}
