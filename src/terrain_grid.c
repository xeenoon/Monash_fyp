#include "terrain_grid.h"

#include <stddef.h>
#include <stdlib.h>

static void set_skirt_pair(Vertex *vertices, uint32_t *cursor,
                           const Vertex *source) {
    vertices[*cursor] = *source;
    vertices[*cursor].untextured = 1.0f; /* skirt top */
    (*cursor)++;
    vertices[*cursor] = *source;
    vertices[*cursor].untextured = 2.0f; /* vertex shader extrudes this copy */
    (*cursor)++;
}

static void add_skirt_triangles(uint32_t *indices, uint32_t *cursor,
                                uint32_t first, uint32_t next) {
    indices[(*cursor)++] = first;
    indices[(*cursor)++] = first + 1u;
    indices[(*cursor)++] = next;
    indices[(*cursor)++] = next;
    indices[(*cursor)++] = first + 1u;
    indices[(*cursor)++] = next + 1u;
}

bool terrain_grid_create(TerrainGrid *grid, uint32_t size) {
    if (!grid || size < 2u) return false;
    uint64_t surface_vertices = (uint64_t)size * size;
    uint64_t skirt_vertices = (uint64_t)(size - 1u) * 8u;
    uint64_t surface_indices = (uint64_t)(size - 1u) * (size - 1u) * 6u;
    uint64_t skirt_indices = (uint64_t)(size - 1u) * 24u;
    uint64_t vertex_count = surface_vertices + skirt_vertices;
    uint64_t index_count = surface_indices + skirt_indices;
    if (vertex_count > UINT32_MAX || index_count > UINT32_MAX ||
        vertex_count > SIZE_MAX / sizeof(Vertex) ||
        index_count > SIZE_MAX / sizeof(uint32_t))
        return false;

    Vertex *vertices = calloc((size_t)vertex_count, sizeof(*vertices));
    uint32_t *indices = malloc((size_t)index_count * sizeof(*indices));
    if (!vertices || !indices) {
        free(vertices);
        free(indices);
        return false;
    }

    for (uint32_t row = 0; row < size; ++row) {
        float v = (float)row / (float)(size - 1u);
        for (uint32_t column = 0; column < size; ++column) {
            float u = (float)column / (float)(size - 1u);
            vertices[(size_t)row * size + column] = (Vertex){
                .position = {u - 0.5f, 0.0f, v - 0.5f},
                .normal = {0.0f, 1.0f, 0.0f},
                .texcoord = {u, v},
            };
        }
    }

    /* Adapted from rocky/GeometryPool.cpp::createIndices and its skirt macros
       (Pelican Mapping, MIT). We retain Rocky's shared perimeter topology, but
       store an extrusion marker because displacement happens in our shader. */
    uint32_t out = 0;
    for (uint32_t row = 0; row < size - 1u; ++row) {
        for (uint32_t column = 0; column < size - 1u; ++column) {
            uint32_t i00 = row * size + column;
            uint32_t i01 = i00 + size;
            uint32_t i10 = i00 + 1u;
            uint32_t i11 = i01 + 1u;
            indices[out++] = i01; indices[out++] = i00; indices[out++] = i11;
            indices[out++] = i00; indices[out++] = i10; indices[out++] = i11;
        }
    }

    uint32_t skirt_begin = (uint32_t)surface_vertices;
    uint32_t vertex = skirt_begin;
    for (uint32_t column = 0; column < size - 1u; ++column)
        set_skirt_pair(vertices, &vertex, &vertices[column]);
    for (uint32_t row = 0; row < size - 1u; ++row)
        set_skirt_pair(vertices, &vertex, &vertices[(size_t)row * size + size - 1u]);
    for (uint32_t column = size - 1u; column > 0; --column)
        set_skirt_pair(vertices, &vertex,
                       &vertices[(size_t)(size - 1u) * size + column]);
    for (uint32_t row = size - 1u; row > 0; --row)
        set_skirt_pair(vertices, &vertex, &vertices[(size_t)row * size]);

    uint32_t skirt_end = (uint32_t)vertex_count;
    uint32_t first;
    for (first = skirt_begin; first < skirt_end - 2u; first += 2u)
        add_skirt_triangles(indices, &out, first, first + 2u);
    add_skirt_triangles(indices, &out, first, skirt_begin);

    *grid = (TerrainGrid){
        .mesh = {
            .vertices = vertices,
            .vertex_count = (uint32_t)vertex_count,
            .indices = indices,
            .index_count = (uint32_t)index_count,
        },
        .vertices = vertices,
        .indices = indices,
        .size = size,
    };
    return true;
}

void terrain_grid_upload(struct Renderer *renderer, TerrainGrid *grid) {
    if (!renderer || !grid) return;
    mesh_upload(renderer, &grid->mesh);
}

void terrain_grid_destroy(struct Renderer *renderer, TerrainGrid *grid) {
    if (!grid) return;
    if (renderer) mesh_destroy(renderer, &grid->mesh);
    free(grid->vertices);
    free(grid->indices);
    *grid = (TerrainGrid){0};
}
