#pragma once

#include "mesh.h"

struct Renderer;

/* Narrow loader for the downloaded Fab-converted Quarry Cliff GLB. It validates
   the GLB container and fixed accessor ranges, then converts centimetre source
   coordinates to this renderer's metre convention. */
typedef struct
{
	Mesh base;
	Vertex *owned_vertices;
	uint32_t *owned_indices;
	/* Scalar declared by the GLB material; multiplied by ORM blue in the
	   forward shader instead of treating the texture as metallic by itself. */
	float metallic_factor;
} Quarry;

bool quarry_create(struct Renderer *renderer, Quarry *out, const char *directory);
void quarry_destroy(struct Renderer *renderer, Quarry *quarry);
