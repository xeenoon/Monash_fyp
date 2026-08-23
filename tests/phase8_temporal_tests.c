#include "temporal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void require(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        exit(EXIT_FAILURE);
    }
}

static int close_enough(float a, float b, float epsilon) {
    return fabsf(a - b) <= epsilon;
}

int main(void) {
    require(close_enough(temporal_halton(1, 2), 0.5f, 1e-6f),
            "Halton base-2 first sample");
    require(close_enough(temporal_halton(2, 2), 0.25f, 1e-6f),
            "Halton base-2 second sample");
    require(close_enough(temporal_halton(3, 3), 1.0f / 9.0f, 1e-6f),
            "Halton base-3 digit reversal");

    vec2s jitter = temporal_jitter_ndc(0, 100, 50);
    require(fabsf(jitter.x) <= 1.0f / 100.0f &&
            fabsf(jitter.y) <= 1.0f / 50.0f,
            "jitter stays within a half pixel");
    mat4s projection = GLMS_MAT4_ZERO_INIT;
    projection.raw[2][3] = -1.0f;
    mat4s shifted = temporal_jitter_projection(projection, jitter);
    require(close_enough(shifted.raw[2][0], -jitter.x, 1e-7f) &&
            close_enough(shifted.raw[2][1], -jitter.y, 1e-7f),
            "projection receives the NDC offset in its z column");

    WorldPosition origin = {1000.0, 2000.0, 3000.0};
    require(!temporal_camera_cut(origin, (WorldPosition){1010.0, 2000.0, 3000.0},
                                 359.0f, 1.0f, 0.0f, 1.0f, 100.0),
            "ordinary motion and wrapped yaw preserve history");
    require(temporal_camera_cut(origin, (WorldPosition){1200.0, 2000.0, 3000.0},
                                0.0f, 0.0f, 0.0f, 0.0f, 100.0),
            "teleport invalidates history");
    require(temporal_camera_cut(origin, origin, 0.0f, 60.0f, 0.0f, 0.0f, 100.0),
            "large orientation discontinuity invalidates history");

    require(close_enough(temporal_exposure_target(0.18f, 0.18f, 0.05f, 16.0f),
                         1.0f, 1e-6f),
            "middle grey maps to unit exposure");
    require(close_enough(temporal_exposure_target(100.0f, 0.18f, 0.05f, 16.0f),
                         0.05f, 1e-6f),
            "exposure target is clamped");
    float brighten = temporal_adapt_exposure(1.0f, 4.0f, 0.1f, 1.0f, 4.0f);
    float darken = temporal_adapt_exposure(4.0f, 1.0f, 0.1f, 1.0f, 4.0f);
    require(brighten > 1.0f && brighten < 4.0f,
            "brightening adapts smoothly");
    require(darken < 4.0f && darken > 1.0f && (4.0f - darken) > (brighten - 1.0f),
            "darkening uses its independent faster speed");

    puts("phase8 temporal tests passed");
    return EXIT_SUCCESS;
}
