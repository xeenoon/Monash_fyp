#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bake_texture_database.h"

enum { SIZE = 256 };

static int close_float(float actual, float expected, float epsilon) {
    return fabsf(actual - expected) <= epsilon;
}

int main(void) {
    const size_t pixels = (size_t)SIZE * SIZE;
    BakeTerrainSample donor;
    memset(&donor, 0, sizeof(donor));
    donor.tile_x = 7;
    donor.tile_y = 9;
    donor.width = SIZE;
    donor.height = SIZE;
    donor.rgb = calloc(pixels, sizeof(*donor.rgb));
    donor.labels = calloc(pixels, sizeof(*donor.labels));
    donor.elevation = calloc(pixels, sizeof(*donor.elevation));
    donor.slope = calloc(pixels, sizeof(*donor.slope));
    donor.gradient_x = calloc(pixels, sizeof(*donor.gradient_x));
    donor.gradient_y = calloc(pixels, sizeof(*donor.gradient_y));
    if (!donor.rgb || !donor.labels || !donor.elevation || !donor.slope ||
        !donor.gradient_x || !donor.gradient_y) return 1;

    for (int y = 0; y < SIZE; ++y) {
        for (int x = 0; x < SIZE; ++x) {
            const size_t p = (size_t)y * SIZE + x;
            const uint8_t red = (uint8_t)x;
            const uint8_t green = (uint8_t)y;
            donor.rgb[p] = red | ((uint32_t)green << 8) | (32u << 16) |
                           (255u << 24);
            donor.labels[p] = (uint32_t)((x / 32) % 3);
            donor.elevation[p] = (float)x;
            donor.slope[p] = 0.25f;
            donor.gradient_x[p] = 2.0f;
            donor.gradient_y[p] = 0.0f;
        }
    }

    BakeTextureDatabase database;
    if (!bake_texture_database_build(&donor, 1, 96, 8, &database)) return 1;
    if (database.count != 441 || database.patch != 96 ||
        database.source_stride != 8 || database.donor_count != 1) return 1;
    if (database.records[0].donor != 0 || database.records[0].source_x != 0 ||
        database.records[0].source_y != 0 ||
        database.records[0].database_index != 0) return 1;
    const BakeResidualCandidate last = database.records[440];
    if (last.donor != 0 || last.source_x != 160 || last.source_y != 160 ||
        last.database_index != 440) return 1;

    if (!close_float(database.density[0] + database.density[1] +
                     database.density[2], 1.0f, 1.0e-6f)) return 1;
    if (!close_float(database.terrain[0], 47.5f, 1.0e-5f) ||
        !close_float(database.terrain[2], 0.25f, 1.0e-6f)) return 1;
    if (!close_float(database.terrain_angle[0], 0.0f, 1.0e-6f) ||
        !close_float(database.terrain_angle[1], 1.0f, 1.0e-6f)) return 1;
    if (database.thumbnail[0] != 0 || database.thumbnail[7] != 2) return 1;
    if (database.north_valid[0] || !database.east_valid[0]) return 1;
    if (!database.north_valid[440] || database.east_valid[440]) return 1;
    if (database.top_edge[0] != 0 || database.top_edge[1] != 0 ||
        database.top_edge[2] != 32 || database.top_gradient[1] != 1) return 1;
    if (database.right_edge[0] != 95 || database.right_gradient[0] != 1) return 1;
    if (!close_float(database.source_position[0], 7.1875f, 1.0e-6f) ||
        !close_float(database.source_position[1], 9.1875f, 1.0e-6f)) return 1;

    bake_texture_database_free(&database);
    free(donor.gradient_y);
    free(donor.gradient_x);
    free(donor.slope);
    free(donor.elevation);
    free(donor.labels);
    free(donor.rgb);
    puts("bake_texture_database_tests: OK (pass17 origin order and descriptors)");
    return 0;
}
