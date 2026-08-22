#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "terrain_tile.h"

static void require(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "phase3 loader: %s\n", message);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
    require(argc == 2 || argc == 3,
            "usage: phase3_tile_loader_tests TILE [checksum|allow-all-valid]");
    TerrainTile tile;
    TerrainTileResult result = terrain_tile_load(argv[1], &tile);
    if (argc == 3 && strcmp(argv[2], "checksum") == 0) {
        require(result == TERRAIN_TILE_CHECKSUM_MISMATCH,
                "corrupt tile did not report checksum mismatch");
        return EXIT_SUCCESS;
    }
    require(argc == 2 || strcmp(argv[2], "allow-all-valid") == 0,
            "unknown expected result");
    require(result == TERRAIN_TILE_OK, terrain_tile_result_string(result));
    require(offsetof(TerrainTile, base) == 0, "Mesh is not the first member");
    require((Mesh *)&tile == &tile.base, "TerrainTile does not cast to Mesh");
    require(tile.header.level == 0 && tile.header.x == 0 && tile.header.y == 0,
            "unexpected root key");
    require(tile.header.parent_level == TERRAIN_TILE_ROOT_KEY,
            "root has a parent");
    require(tile.header.sample_width == 7 && tile.header.sample_height == 7 &&
            tile.header.gutter == 1, "unexpected raster layout");
    require(strcmp(tile.profile, "EPSG:2056") == 0, "profile was not loaded");
    require(tile.base.texture_path == tile.imagery_uri,
            "Mesh did not inherit the external imagery path");
    require(isfinite(terrain_tile_height(&tile, 1, 1)),
            "valid height did not decode");
    if (argc != 3 || strcmp(argv[2], "allow-all-valid") != 0) {
        require(tile.header.flags & TERRAIN_TILE_HAS_NODATA,
                "explicit no-data flag was not loaded");
        require(isnan(terrain_tile_height(&tile, 3, 3)),
                "masked height did not decode as no-data");
    }
    require(isnan(terrain_tile_height(&tile, 99, 99)),
            "out-of-range height should be no-data");
    terrain_tile_unload(&tile);
    require(tile.base.vertices == NULL && tile.height_data == NULL,
            "unload did not clear ownership");
    puts("phase 3 runtime tile loader tests passed");
    return EXIT_SUCCESS;
}
