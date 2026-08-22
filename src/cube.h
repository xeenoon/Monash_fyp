#pragma once

#include "mesh.h"

struct Renderer;

/* Cube "derives" from Mesh: base first, so (Mesh*)&cube is valid.
   Cube-only fields (transform, extents, ...) go AFTER base later
   without breaking the cast. */
typedef struct {
    Mesh base;
} Cube;

Cube cube_create(struct Renderer *r);
