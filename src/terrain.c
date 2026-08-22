#include "terrain.h"

#include <math.h>

#define GRID_N     TERRAIN_GRID_N
#define SPACING    TERRAIN_SPACING_M
#define CELLS      (GRID_N - 1)
#define TOP_VERTEX_COUNT     (GRID_N * GRID_N)
#define TOP_INDEX_COUNT      (CELLS * CELLS * 6)
#define SIDE_SEGMENT_COUNT   (CELLS * 4)
#define SIDE_VERTEX_COUNT    (SIDE_SEGMENT_COUNT * 4)
#define SIDE_INDEX_COUNT     (SIDE_SEGMENT_COUNT * 6)
#define BASE_VERTEX_COUNT    4
#define BASE_INDEX_COUNT     6
#define TERRAIN_VERTEX_COUNT (TOP_VERTEX_COUNT + SIDE_VERTEX_COUNT + BASE_VERTEX_COUNT)
#define TERRAIN_INDEX_COUNT  (TOP_INDEX_COUNT + SIDE_INDEX_COUNT + BASE_INDEX_COUNT)
#define TERRAIN_BASE_Y       (-100.0f)

/* Static geometry for the (single) terrain instance, mirroring cube.c: the grid
   is stored as GRID_N x GRID_N unique vertices plus an index buffer that stitches
   them into two triangles per cell. mesh_upload copies both to the GPU. */
static Vertex   TERRAIN_VERTICES[TERRAIN_VERTEX_COUNT];
static uint32_t TERRAIN_INDICES[TERRAIN_INDEX_COUNT];

/* Add one outward-facing wall quad. A -> B follows the top surface's outer
   boundary winding, so (top A, bottom A, bottom B) faces away from the block. */
static void append_wall_segment(Vertex **vertex, uint32_t **index,
                                float ax, float ay, float az,
                                float bx, float by, float bz,
                                float nx, float nz) {
    Vertex *v = *vertex;
    uint32_t first = (uint32_t)(v - TERRAIN_VERTICES);
    v[0] = (Vertex){{ax, ay, az}, {nx, 0.0f, nz}, {0.0f, 0.0f}, 1.0f};
    v[1] = (Vertex){{ax, TERRAIN_BASE_Y, az}, {nx, 0.0f, nz}, {0.0f, 0.0f}, 1.0f};
    v[2] = (Vertex){{bx, TERRAIN_BASE_Y, bz}, {nx, 0.0f, nz}, {0.0f, 0.0f}, 1.0f};
    v[3] = (Vertex){{bx, by, bz}, {nx, 0.0f, nz}, {0.0f, 0.0f}, 1.0f};
    *vertex += 4;

    uint32_t *idx = *index;
    *idx++ = first; *idx++ = first + 1; *idx++ = first + 2;
    *idx++ = first; *idx++ = first + 2; *idx++ = first + 3;
    *index = idx;
}

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
            /* Texture spans the field 1:1; grid row 0 = north = image row 0. */
            v->texcoord[0] = (float)i / (GRID_N - 1);
            v->texcoord[1] = (float)j / (GRID_N - 1);
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

    /* Close the heightfield into a solid block. Each wall segment owns its
       vertices so its flat outward normal and grey material do not bleed into
       the textured, smoothly-normalled top. */
    Vertex *wall = TERRAIN_VERTICES + TOP_VERTEX_COUNT;

    /* North (-Z), following the top boundary from east to west. */
    for (int i = CELLS; i > 0; --i)
        append_wall_segment(&wall, &idx,
            i * SPACING - half, height_at(i, 0) - min_h, -half,
            (i - 1) * SPACING - half, height_at(i - 1, 0) - min_h, -half,
            0.0f, -1.0f);

    /* West (-X), north to south. */
    for (int j = 0; j < CELLS; ++j)
        append_wall_segment(&wall, &idx,
            -half, height_at(0, j) - min_h, j * SPACING - half,
            -half, height_at(0, j + 1) - min_h, (j + 1) * SPACING - half,
            -1.0f, 0.0f);

    /* South (+Z), west to east. */
    for (int i = 0; i < CELLS; ++i)
        append_wall_segment(&wall, &idx,
            i * SPACING - half, height_at(i, CELLS) - min_h, half,
            (i + 1) * SPACING - half, height_at(i + 1, CELLS) - min_h, half,
            0.0f, 1.0f);

    /* East (+X), south to north. */
    for (int j = CELLS; j > 0; --j)
        append_wall_segment(&wall, &idx,
            half, height_at(CELLS, j) - min_h, j * SPACING - half,
            half, height_at(CELLS, j - 1) - min_h, (j - 1) * SPACING - half,
            1.0f, 0.0f);

    /* Flat underside, outward-facing down. */
    uint32_t base = (uint32_t)(wall - TERRAIN_VERTICES);
    wall[0] = (Vertex){{-half, TERRAIN_BASE_Y, -half}, {0, -1, 0}, {0, 0}, 1.0f};
    wall[1] = (Vertex){{ half, TERRAIN_BASE_Y, -half}, {0, -1, 0}, {0, 0}, 1.0f};
    wall[2] = (Vertex){{ half, TERRAIN_BASE_Y,  half}, {0, -1, 0}, {0, 0}, 1.0f};
    wall[3] = (Vertex){{-half, TERRAIN_BASE_Y,  half}, {0, -1, 0}, {0, 0}, 1.0f};
    idx[0] = base; idx[1] = base + 1; idx[2] = base + 2;
    idx[3] = base; idx[4] = base + 2; idx[5] = base + 3;

    Terrain terrain = { .base = {
        .vertices = TERRAIN_VERTICES,
        .vertex_count = (uint32_t)TERRAIN_VERTEX_COUNT,
        .indices = TERRAIN_INDICES,
        .index_count = (uint32_t)TERRAIN_INDEX_COUNT,
        .texture_path = ASSET_DIR "/terrain_albedo.png",
    }};
    mesh_upload(r, &terrain.base);
    return terrain;
}
