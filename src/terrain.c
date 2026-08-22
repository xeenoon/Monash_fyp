#include "terrain.h"

#include <math.h>

#define GRID_N     TERRAIN_GRID_N
#define SPACING    TERRAIN_SPACING_M
#define CELLS      (GRID_N - 1)
#define TERRAIN_VERTEX_COUNT (GRID_N * GRID_N)
#define TERRAIN_INDEX_COUNT  (CELLS * CELLS * 6)

/* Static geometry for the (single) terrain instance, mirroring cube.c: the grid
   is stored as GRID_N x GRID_N unique vertices plus an index buffer that stitches
   them into two triangles per cell. mesh_upload copies both to the GPU. */
static Vertex   TERRAIN_VERTICES[TERRAIN_VERTEX_COUNT];
static uint32_t TERRAIN_INDICES[TERRAIN_INDEX_COUNT];

static float height_at(int i, int j) {
    if (i < 0) i = 0; else if (i >= GRID_N) i = GRID_N - 1;
    if (j < 0) j = 0; else if (j >= GRID_N) j = GRID_N - 1;
    return TERRAIN_HEIGHTMAP[j * GRID_N + i];
}

/* Smooth per-grid-point normal from the height gradient (central difference).
   Surface y = h(x,z), so the normal is (-dh/dx, 1, -dh/dz) normalised. */
static void normal_at(int i, int j, float out[3]) {
    float dhdx = (height_at(i + 1, j) - height_at(i - 1, j)) / (2.0f * SPACING);
    float dhdz = (height_at(i, j + 1) - height_at(i, j - 1)) / (2.0f * SPACING);
    float nx = -dhdx, ny = 1.0f, nz = -dhdz;
    float len = sqrtf(nx * nx + ny * ny + nz * nz);
    out[0] = nx / len; out[1] = ny / len; out[2] = nz / len;
}

Terrain terrain_create(struct Renderer *r) {
    /* Centre the field on the origin and drop its lowest point to y = 0. */
    float min_h = TERRAIN_HEIGHTMAP[0];
    for (int k = 1; k < GRID_N * GRID_N; ++k)
        if (TERRAIN_HEIGHTMAP[k] < min_h) min_h = TERRAIN_HEIGHTMAP[k];
    const float half = (GRID_N - 1) * SPACING * 0.5f;

    /* One unique vertex per grid point: position + smooth normal. */
    for (int j = 0; j < GRID_N; ++j) {
        for (int i = 0; i < GRID_N; ++i) {
            Vertex *v = &TERRAIN_VERTICES[j * GRID_N + i];
            v->position[0] = i * SPACING - half;
            v->position[1] = height_at(i, j) - min_h;
            v->position[2] = j * SPACING - half;
            normal_at(i, j, v->normal);
        }
    }

    /* Two CCW-from-above triangles per cell (renderer culls back faces,
       front = CCW): (00,11,10) and (00,01,11). */
    uint32_t *idx = TERRAIN_INDICES;
    for (int j = 0; j < CELLS; ++j) {
        for (int i = 0; i < CELLS; ++i) {
            uint32_t v00 = (uint32_t)(j * GRID_N + i);
            uint32_t v10 = v00 + 1;
            uint32_t v01 = v00 + GRID_N;
            uint32_t v11 = v01 + 1;
            *idx++ = v00; *idx++ = v11; *idx++ = v10;
            *idx++ = v00; *idx++ = v01; *idx++ = v11;
        }
    }

    Terrain terrain = { .base = {
        .vertices = TERRAIN_VERTICES,
        .vertex_count = (uint32_t)TERRAIN_VERTEX_COUNT,
        .indices = TERRAIN_INDICES,
        .index_count = (uint32_t)TERRAIN_INDEX_COUNT,
    }};
    mesh_upload(r, &terrain.base);
    return terrain;
}
