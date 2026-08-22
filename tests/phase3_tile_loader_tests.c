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
    require(argc >= 2 && argc <= 4,
            "usage: phase3_tile_loader_tests TILE [checksum|allow-all-valid] [ROOT]");
    TerrainTile tile;
    TerrainTileResult result = terrain_tile_load(argv[1], &tile);
    if (argc == 3 && strcmp(argv[2], "checksum") == 0) {
        require(result == TERRAIN_TILE_CHECKSUM_MISMATCH,
                "corrupt tile did not report checksum mismatch");
        return EXIT_SUCCESS;
    }
    require(argc == 2 || strcmp(argv[2], "allow-all-valid") == 0,
            "unknown expected result");
    int allow_all_valid = argc >= 3 && strcmp(argv[2], "allow-all-valid") == 0;
    require(result == TERRAIN_TILE_OK, terrain_tile_result_string(result));
    require(offsetof(TerrainTile, base) == 0, "Mesh is not the first member");
    require((Mesh *)&tile == &tile.base, "TerrainTile does not cast to Mesh");
    require(tile.header.level == 0 && tile.header.x == 0 && tile.header.y == 0,
            "unexpected root key");
    require(tile.header.parent_level == TERRAIN_TILE_ROOT_KEY,
            "root has a parent");
    require(tile.header.gutter >= 1, "height gutter is missing");
    if (!allow_all_valid)
        require(tile.header.sample_width == 7 && tile.header.sample_height == 7,
                "unexpected fixture raster layout");
    require(strcmp(tile.profile, "EPSG:2056") == 0, "profile was not loaded");
    require(tile.base.texture_path == tile.imagery_uri,
            "Mesh did not inherit the external imagery path");
    require(isfinite(terrain_tile_height(&tile, 1, 1)),
            "valid height did not decode");
    if (!allow_all_valid) {
        require(tile.header.flags & TERRAIN_TILE_HAS_NODATA,
                "explicit no-data flag was not loaded");
        require(isnan(terrain_tile_height(&tile, 3, 3)),
                "masked height did not decode as no-data");
    } else {
        if (argc == 4) {
            require(terrain_tile_resolve_imagery(&tile, argv[3]),
                    "could not resolve external imagery");
            require(tile.base.texture_path == tile.resolved_imagery_path,
                    "resolved imagery was not attached to Mesh");
        }
        require(terrain_tile_build_mesh(&tile), "could not generate inherited Mesh");
        require(tile.base.vertices == tile.owned_vertices &&
                tile.base.indices == tile.owned_indices,
                "generated geometry was not attached to Mesh");
        uint32_t width = tile.header.sample_width - tile.header.gutter * 2u;
        uint32_t height = tile.header.sample_height - tile.header.gutter * 2u;
        uint32_t cells_x = width - 1u, cells_y = height - 1u;
        uint32_t side_segments = 2u * (cells_x + cells_y);
        uint32_t expected_vertices = width * height + side_segments * 4u + 4u;
        uint32_t expected_indices = cells_x * cells_y * 6u +
                                    side_segments * 6u + 6u;
        require(tile.base.vertex_count == expected_vertices &&
                tile.base.index_count == expected_indices,
                "generated Mesh has unexpected dimensions");
        require(isfinite(tile.base.vertices[0].normal[0]) &&
                isfinite(tile.base.vertices[0].normal[1]) &&
                isfinite(tile.base.vertices[0].normal[2]),
                "generated normal is not finite");
    }
    require(isnan(terrain_tile_height(&tile, 99, 99)),
            "out-of-range height should be no-data");
    terrain_tile_unload(&tile);
    require(tile.base.vertices == NULL && tile.height_data == NULL,
            "unload did not clear ownership");
    puts("phase 3 runtime tile loader tests passed");
    return EXIT_SUCCESS;
}
