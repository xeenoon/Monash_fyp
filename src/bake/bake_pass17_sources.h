// Frozen donor coordinates selected by pass17 for target level-5 tile 23/8.
#ifndef BAKE_PASS17_SOURCES_H
#define BAKE_PASS17_SOURCES_H

#include <stddef.h>

typedef struct { int x, y; } BakeTileCoordinate;

enum {
    BAKE_PASS17_FULL_SNOW = 0,
    BAKE_PASS17_SNOW_ROCK = 1,
    BAKE_PASS17_FULL_ROCK = 2,
    BAKE_PASS17_ROCK_GRASS = 3,
    BAKE_PASS17_FULL_GRASS = 4,
};

const BakeTileCoordinate *bake_pass17_texture_sources(int preset,
                                                       size_t *count);
const BakeTileCoordinate *bake_pass17_material_sources(int material,
                                                        size_t *count);

#endif
