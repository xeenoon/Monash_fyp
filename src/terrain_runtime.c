#include "terrain_runtime.h"

#include "path_utils.h"

#include <stdio.h>
#include <stdlib.h>

Terrain terrain_create(struct Renderer *renderer) {
    char *tile_path = path_join(TRN_DIR, "tiles/0/0/0.trn");
    if (!tile_path) {
        fprintf(stderr, "Could not allocate the default terrain tile path\n");
        exit(EXIT_FAILURE);
    }

    Terrain terrain;
    TerrainTileResult result = terrain_tile_load(tile_path, &terrain);
    if (result != TERRAIN_TILE_OK) {
        fprintf(stderr, "Could not load terrain tile %s: %s\n", tile_path,
                terrain_tile_result_string(result));
        free(tile_path);
        exit(EXIT_FAILURE);
    }
    free(tile_path);

    if (!terrain_tile_resolve_imagery(&terrain, TRN_DIR)) {
        fprintf(stderr, "Could not resolve terrain imagery %s\n",
                terrain.imagery_uri ? terrain.imagery_uri : "(missing)");
        terrain_tile_unload(&terrain);
        exit(EXIT_FAILURE);
    }
    if (!terrain_tile_build_mesh(&terrain)) {
        fprintf(stderr, "Could not build a mesh from terrain tile %u/%u/%u; "
                        "interior no-data is not renderable yet\n",
                terrain.header.level, terrain.header.x, terrain.header.y);
        terrain_tile_unload(&terrain);
        exit(EXIT_FAILURE);
    }

    mesh_upload(renderer, &terrain.base);
    return terrain;
}

void terrain_destroy(struct Renderer *renderer, Terrain *terrain) {
    if (!terrain) return;
    mesh_destroy(renderer, &terrain->base);
    terrain_tile_unload(terrain);
}
