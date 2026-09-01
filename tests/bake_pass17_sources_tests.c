#include <stdio.h>

#include "bake_pass17_sources.h"

int main(void) {
    size_t count = 0;
    const BakeTileCoordinate *p = bake_pass17_texture_sources(
        BAKE_PASS17_FULL_GRASS, &count);
    if (!p || count != 14 || p[0].x != 25 || p[0].y != 30 ||
        p[13].x != 22 || p[13].y != 26) return 1;
    p = bake_pass17_texture_sources(BAKE_PASS17_SNOW_ROCK, &count);
    if (!p || count != 14 || p[0].x != 4 || p[0].y != 20 ||
        p[13].x != 24 || p[13].y != 7) return 1;
    p = bake_pass17_texture_sources(BAKE_PASS17_ROCK_GRASS, &count);
    if (!p || count != 14 || p[0].x != 3 || p[0].y != 0 ||
        p[13].x != 7 || p[13].y != 3) return 1;
    p = bake_pass17_material_sources(1, &count);
    if (!p || count != 4 || p[0].x != 0 || p[0].y != 18) return 1;
    if (bake_pass17_texture_sources(-1, &count) || count) return 1;
    puts("bake_pass17_sources_tests: OK (frozen cleaned donor pools)");
    return 0;
}
