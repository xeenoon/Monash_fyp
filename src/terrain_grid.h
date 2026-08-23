#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "mesh.h"

struct Renderer;

/* One normalized projected-terrain grid shared by every quadtree tile. Height,
   horizontal span, and skirt depth are supplied per draw in the vertex shader. */
typedef struct
{
	Mesh mesh;
	Vertex *vertices;
	uint32_t *indices;
	uint32_t size;
} TerrainGrid;

bool terrain_grid_create(TerrainGrid *grid, uint32_t size);
void terrain_grid_upload(struct Renderer *renderer, TerrainGrid *grid);
void terrain_grid_destroy(struct Renderer *renderer, TerrainGrid *grid);
