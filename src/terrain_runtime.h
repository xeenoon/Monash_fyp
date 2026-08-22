#pragma once

#include "terrain_tile.h"

struct Renderer;

/* Renderer-facing terrain loaded from an offline .trn tile. TerrainTile keeps
   Mesh as its first member, so Terrain retains the existing Mesh inheritance. */
typedef TerrainTile Terrain;

Terrain terrain_create(struct Renderer *renderer);
void terrain_destroy(struct Renderer *renderer, Terrain *terrain);
