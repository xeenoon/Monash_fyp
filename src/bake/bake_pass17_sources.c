#include "bake_pass17_sources.h"

// Pure presets share the same macro-selected style family in pass17.
static const BakeTileCoordinate pure[] = {
    {25,30}, {4,20}, {3,0}, {4,22}, {1,24}, {4,0}, {1,25},
    {7,3}, {30,8}, {28,8}, {28,7}, {31,8}, {20,11}, {22,26},
};
static const BakeTileCoordinate snow_rock[] = {
    {4,20}, {3,0}, {4,22}, {4,21}, {0,18}, {1,24}, {4,0},
    {1,25}, {28,8}, {28,7}, {27,12}, {26,12}, {25,8}, {24,7},
};
static const BakeTileCoordinate rock_grass[] = {
    {3,0}, {4,20}, {31,23}, {31,22}, {18,1}, {18,0}, {17,0},
    {0,18}, {26,12}, {27,12}, {28,8}, {28,7}, {4,0}, {7,3},
};
// full_grass drops the four near-solid snow donors (30/8, 28/8, 28/7, 31/8),
// which belong to full_snow, and takes the green ground/grass tiles instead so
// the quilt can never pull white snow patches into grass.
static const BakeTileCoordinate grass_texture[] = {
    {25,30}, {4,20}, {3,0}, {4,22}, {1,24}, {4,0}, {1,25},
    {7,3}, {20,11}, {22,26}, {0,18}, {17,0}, {18,0}, {18,1},
};
static const BakeTileCoordinate rock[] = {
    {23,13}, {20,12}, {0,26}, {5,23}, {4,21}, {0,25},
};
static const BakeTileCoordinate grass[] = {
    {0,18}, {17,0}, {18,0}, {18,1},
};
static const BakeTileCoordinate snow[] = {
    {24,12}, {26,6}, {13,25}, {24,6}, {13,24}, {26,7},
};

const BakeTileCoordinate *bake_pass17_texture_sources(int preset,
                                                       size_t *count) {
    if (count) *count = 14;
    if (preset == BAKE_PASS17_SNOW_ROCK) return snow_rock;
    if (preset == BAKE_PASS17_ROCK_GRASS) return rock_grass;
    if (preset == BAKE_PASS17_FULL_GRASS) return grass_texture;
    if (preset >= BAKE_PASS17_FULL_SNOW && preset <= BAKE_PASS17_FULL_GRASS)
        return pure;
    if (count) *count = 0;
    return 0;
}

const BakeTileCoordinate *bake_pass17_material_sources(int material,
                                                        size_t *count) {
    if (material == 0) {
        if (count) *count = sizeof(rock) / sizeof(rock[0]);
        return rock;
    }
    if (material == 1) {
        if (count) *count = sizeof(grass) / sizeof(grass[0]);
        return grass;
    }
    if (material == 2) {
        if (count) *count = sizeof(snow) / sizeof(snow[0]);
        return snow;
    }
    if (count) *count = 0;
    return 0;
}
