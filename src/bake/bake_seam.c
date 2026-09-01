#include "bake_seam.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>

bool bake_minimum_vertical_seam(const float *cost, int width, int height,
                                int16_t *seam) {
    if (!cost || !seam || width <= 0 || height <= 0 || width > INT16_MAX)
        return false;
    size_t count = (size_t)width * (size_t)height;
    float *energy = malloc(count * sizeof(*energy));
    int16_t *parents = calloc(count, sizeof(*parents));
    if (!energy || !parents) {
        free(parents);
        free(energy);
        return false;
    }
    memcpy(energy, cost, (size_t)width * sizeof(*energy));
    for (int y = 1; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int x0 = x > 0 ? x - 1 : 0;
            int x1 = x + 2 < width ? x + 2 : width;
            int predecessor = x0;
            float best = energy[(size_t)(y - 1) * width + x0];
            // np.argmin keeps the first equal minimum.
            for (int xx = x0 + 1; xx < x1; ++xx) {
                float value = energy[(size_t)(y - 1) * width + xx];
                if (value < best) {
                    best = value;
                    predecessor = xx;
                }
            }
            size_t i = (size_t)y * width + x;
            energy[i] = cost[i] + best;
            parents[i] = (int16_t)predecessor;
        }
    }
    int last = 0;
    float best = energy[(size_t)(height - 1) * width];
    for (int x = 1; x < width; ++x) {
        float value = energy[(size_t)(height - 1) * width + x];
        if (value < best) {
            best = value;
            last = x;
        }
    }
    seam[height - 1] = (int16_t)last;
    for (int y = height - 1; y > 0; --y)
        seam[y - 1] = parents[(size_t)y * width + seam[y]];
    free(parents);
    free(energy);
    return true;
}

bool bake_quilt_take_mask_from_cost(const float *cost, const uint8_t *known,
                                    int patch, int overlap, bool has_left,
                                    bool has_top, uint8_t *take) {
    if (!cost || !known || !take || patch <= 0 || overlap <= 0 || overlap > patch)
        return false;
    size_t count = (size_t)patch * patch;
    float *strip = malloc((size_t)patch * overlap * sizeof(*strip));
    int16_t *seam = malloc((size_t)patch * sizeof(*seam));
    if (!strip || !seam) {
        free(seam);
        free(strip);
        return false;
    }
    memset(take, 1, count);
    if (has_left) {
        for (int y = 0; y < patch; ++y)
            for (int x = 0; x < overlap; ++x) {
                float q = ((float)x - 0.5f * (float)(overlap - 1)) /
                          (float)overlap;
                strip[(size_t)y * overlap + x] =
                    cost[(size_t)y * patch + x] + 1.0e-5f * q * q;
            }
        if (!bake_minimum_vertical_seam(strip, overlap, patch, seam)) goto fail;
        for (int y = 0; y < patch; ++y)
            for (int x = 0; x < overlap; ++x)
                take[(size_t)y * patch + x] &= (uint8_t)(x >= seam[y]);
    }
    if (has_top) {
        // Python transposes the top strip before finding a vertical seam.
        for (int x = 0; x < patch; ++x)
            for (int y = 0; y < overlap; ++y) {
                float q = ((float)y - 0.5f * (float)(overlap - 1)) /
                          (float)overlap;
                strip[(size_t)x * overlap + y] =
                    cost[(size_t)y * patch + x] + 1.0e-5f * q * q;
            }
        if (!bake_minimum_vertical_seam(strip, overlap, patch, seam)) goto fail;
        for (int y = 0; y < overlap; ++y)
            for (int x = 0; x < patch; ++x)
                take[(size_t)y * patch + x] &= (uint8_t)(y >= seam[x]);
    }
    for (size_t i = 0; i < count; ++i) take[i] |= (uint8_t)!known[i];
    free(seam);
    free(strip);
    return true;
fail:
    free(seam);
    free(strip);
    return false;
}
