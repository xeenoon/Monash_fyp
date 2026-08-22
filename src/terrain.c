#include "terrain.h"

#include <math.h>

#define GRID_N     TERRAIN_GRID_N
#define SPACING    TERRAIN_SPACING_M
#define CELLS      (GRID_N - 1)
#define TERRAIN_VERTEX_COUNT (CELLS * CELLS * 6)

/* One flat vertex buffer for the (single) terrain instance. Keeping it static
   mirrors cube.c's static geometry: mesh->vertices can point straight at it and
   mesh_destroy only has to release the GPU copy. */
static Vertex TERRAIN_VERTICES[TERRAIN_VERTEX_COUNT];

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

    /* Emit each grid point's position + smooth normal, indexed on the fly. */
    #define EMIT(dst, gi, gj) do {                                      \
        Vertex *v = (dst);                                             \
        v->position[0] = (gi) * SPACING - half;                       \
        v->position[1] = height_at((gi), (gj)) - min_h;               \
        v->position[2] = (gj) * SPACING - half;                       \
        normal_at((gi), (gj), v->normal);                             \
    } while (0)

    Vertex *out = TERRAIN_VERTICES;
    for (int j = 0; j < CELLS; ++j) {
        for (int i = 0; i < CELLS; ++i) {
            /* Two CCW-from-above triangles per cell (renderer culls back faces,
               front = CCW): (00,11,10) and (00,01,11). */
            EMIT(out++, i,     j);
            EMIT(out++, i + 1, j + 1);
            EMIT(out++, i + 1, j);
            EMIT(out++, i,     j);
            EMIT(out++, i,     j + 1);
            EMIT(out++, i + 1, j + 1);
        }
    }
    #undef EMIT

    Terrain terrain = { .base = {
        .vertices = TERRAIN_VERTICES,
        .vertex_count = (uint32_t)TERRAIN_VERTEX_COUNT,
    }};
    mesh_upload(r, &terrain.base);
    return terrain;
}
