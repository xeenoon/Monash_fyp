// Header-only C reference implementations of the Python CPU oracle
// (tools/procedural_gap_demo.py + build_terrain_macro.py).  The GPU compute
// shaders are validated bit-for-bit against these; the C reference in turn
// mirrors the Python line by line (verified offline), so it stands in for the
// Python oracle in a portable, dataset-free unit test.
//
// Float math is written in the same order as the shaders (float32, sequential)
// so GPU and CPU agree to the last bit outside genuine threshold ties.
#ifndef BAKE_REFERENCE_H
#define BAKE_REFERENCE_H

#include <stdint.h>

enum { BAKE_ROCK = 0, BAKE_GRASS = 1, BAKE_SNOW = 2 };

// Mirror of colour_labels(): ROCK by default, GRASS where the grass rule holds,
// SNOW (which overrides grass) where the snow rule holds.
static inline uint8_t bake_ref_classify(uint8_t r8, uint8_t g8, uint8_t b8) {
    float r = (float)r8 / 255.0f;
    float g = (float)g8 / 255.0f;
    float b = (float)b8 / 255.0f;
    float mx = r > g ? (r > b ? r : b) : (g > b ? g : b);
    float mn = r < g ? (r < b ? r : b) : (g < b ? g : b);
    float saturation = (mx - mn) / (mx > 1.0e-5f ? mx : 1.0e-5f);
    float luminance = r * 0.2126f + g * 0.7152f + b * 0.0722f;

    int water = (b > r * 1.08f) && (g > r * 1.04f) && (b > g * 1.035f) &&
                ((b + g) > (r * 2.18f)) && (luminance > 0.08f) && (luminance < 0.78f);
    int snow = (luminance > 0.64f) && (saturation < 0.28f);
    int grass = (g > r * 1.035f) && (g > b * 1.10f) && (luminance > 0.08f) &&
                (luminance < 0.64f) && !water;

    uint8_t label = BAKE_ROCK;
    if (grass) label = BAKE_GRASS;
    if (snow) label = BAKE_SNOW;
    return label;
}

#endif  // BAKE_REFERENCE_H
