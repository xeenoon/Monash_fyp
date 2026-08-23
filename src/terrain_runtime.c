#include "terrain_runtime.h"

#include "str_utils.h"
#include "surface_detail.h"
#include "terrain_grid.h"
#include "terrain_tile.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    TerrainTile tile;
    Texture elevation;
    UploadTicket upload;
} RuntimeTile;

struct TerrainRuntime {
    Renderer *renderer;
    char *dataset_root;
    TerrainRuntimeSettings settings;
    TerrainQuadtree tree;
    TerrainGrid grid;
    RendererDraw *draws;
    uint32_t draw_count;
    RendererDraw *shadow_draws;
    uint32_t shadow_draw_count;
};

TerrainRuntimeSettings terrain_runtime_default_settings(void) {
    return (TerrainRuntimeSettings){
        .quadtree = terrain_quadtree_default_settings(),
        .skirt_ratio = 0.05f,
    };
}

static char *tile_path(const char *root, TerrainTileKey key) {
    int length = snprintf(NULL, 0, "%s/tiles/%u/%u/%u.trn",
                          root, key.level, key.x, key.y);
    if (length < 0) return NULL;
    char *path = malloc((size_t)length + 1u);
    if (path)
        snprintf(path, (size_t)length + 1u, "%s/tiles/%u/%u/%u.trn",
                 root, key.level, key.x, key.y);
    return path;
}

static uint64_t tile_cpu_bytes(const TerrainTile *tile) {
    return sizeof(*tile) + tile->header.height_bytes + tile->header.validity_bytes +
        tile->header.imagery_bytes + tile->header.profile_bytes + 1u +
        tile->header.source_bytes + 1u + tile->header.imagery_uri_bytes + 1u +
        (tile->resolved_imagery_path ? strlen(tile->resolved_imagery_path) + 1u : 0u);
}

static void tile_bounds(const TerrainTile *tile, float skirt_ratio,
                        TerrainTileBounds *out) {
    double span_x = tile->header.extent[2] - tile->header.extent[0];
    double span_z = tile->header.extent[3] - tile->header.extent[1];
    double skirt = fmax(span_x, span_z) * skirt_ratio;
    const double minimum[3] = {-0.5 * span_x, -skirt, -0.5 * span_z};
    const double maximum[3] = { 0.5 * span_x, tile->header.height_range_m,
                                0.5 * span_z};
    WorldPosition sum = {0};
    for (uint32_t corner = 0; corner < 8; ++corner) {
        TileLocalPosition local = {
            (float)((corner & 1u) ? maximum[0] : minimum[0]),
            (float)((corner & 2u) ? maximum[1] : minimum[1]),
            (float)((corner & 4u) ? maximum[2] : minimum[2]),
        };
        WorldPosition world = coordinate_local_to_world(&tile->header.local_to_world,
                                                         local);
        out->corners[corner][0] = world.x;
        out->corners[corner][1] = world.y;
        out->corners[corner][2] = world.z;
        sum.x += world.x; sum.y += world.y; sum.z += world.z;
    }
    out->center = (WorldPosition){sum.x / 8.0, sum.y / 8.0, sum.z / 8.0};
    out->radius_m = 0.0;
    for (uint32_t corner = 0; corner < 8; ++corner) {
        double dx = out->corners[corner][0] - out->center.x;
        double dy = out->corners[corner][1] - out->center.y;
        double dz = out->corners[corner][2] - out->center.z;
        double radius = sqrt(dx*dx + dy*dy + dz*dz);
        if (radius > out->radius_m) out->radius_m = radius;
    }
    out->geometric_error_m = tile->header.geometric_error_m;
}

static bool interior_valid(const TerrainTile *tile) {
    uint32_t gutter = tile->header.gutter;
    uint32_t width = tile->header.sample_width - 2u * gutter;
    uint32_t height = tile->header.sample_height - 2u * gutter;
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x)
            if (!terrain_tile_sample_valid(tile, x + gutter, y + gutter))
                return false;
    return true;
}

/* This is the plain-C equivalent of Rocky TerrainTilePager::requestLoadData:
   load immutable CPU data first; the owner publishes it only if the generation
   still matches. I/O remains synchronous until a job system is introduced. */
static bool load_node(TerrainRuntime *terrain, uint32_t index) {
    TerrainQuadNode *node = &terrain->tree.nodes[index];
    uint32_t generation = node->generation;
    char *path = tile_path(terrain->dataset_root, node->key);
    RuntimeTile *runtime_tile = calloc(1, sizeof(*runtime_tile));
    if (!path || !runtime_tile) {
        free(path); free(runtime_tile);
        return false;
    }
    TerrainTileResult result = terrain_tile_load(path, &runtime_tile->tile);
    free(path);
    if (result == TERRAIN_TILE_IO_ERROR) {
        free(runtime_tile);
        terrain_quadtree_unavailable(&terrain->tree, index, generation);
        return true;
    }
    if (result != TERRAIN_TILE_OK ||
        !terrain_tile_resolve_imagery(&runtime_tile->tile, terrain->dataset_root) ||
        !interior_valid(&runtime_tile->tile)) {
        fprintf(stderr, "Terrain tile %u/%u/%u is unusable: %s\n",
                node->key.level, node->key.x, node->key.y,
                terrain_tile_result_string(result));
        terrain_tile_unload(&runtime_tile->tile);
        free(runtime_tile);
        terrain_quadtree_unavailable(&terrain->tree, index, generation);
        return true;
    }
    TerrainTileBounds bounds = {0};
    tile_bounds(&runtime_tile->tile, terrain->settings.skirt_ratio, &bounds);
    uint64_t bytes = tile_cpu_bytes(&runtime_tile->tile) + sizeof(*runtime_tile);
    if (!terrain_quadtree_cpu_ready(&terrain->tree, index, generation,
                                    &bounds, bytes, runtime_tile)) {
        terrain_tile_unload(&runtime_tile->tile);
        free(runtime_tile);
        return false;
    }
    return true;
}

static uint16_t *encode_elevation_r16(const TerrainTile *tile) {
    size_t count = (size_t)tile->header.sample_width * tile->header.sample_height;
    uint16_t *encoded = malloc(count * sizeof(*encoded));
    if (!encoded) return NULL;
    float range = tile->header.height_range_m;
    for (uint32_t y = 0; y < tile->header.sample_height; ++y) {
        for (uint32_t x = 0; x < tile->header.sample_width; ++x) {
            size_t index = (size_t)y * tile->header.sample_width + x;
            float height = terrain_tile_height(tile, x, y);
            float unit = isfinite(height) && range > 0.0f
                ? (height - tile->header.min_height_m) / range : 0.0f;
            if (unit < 0.0f) unit = 0.0f;
            if (unit > 1.0f) unit = 1.0f;
            encoded[index] = (uint16_t)lroundf(unit * 65535.0f);
        }
    }
    return encoded;
}

static uint64_t texture_bytes(const Texture *texture, uint32_t bytes_per_pixel) {
    uint64_t width = texture->extent.width;
    uint64_t height = texture->extent.height;
    uint64_t total = 0;
    for (uint32_t mip = 0; mip < texture->mip_levels; ++mip) {
        total += width * height * bytes_per_pixel;
        if (width > 1) width /= 2;
        if (height > 1) height /= 2;
    }
    return total;
}

static void destroy_runtime_tile(TerrainRuntime *terrain, RuntimeTile *tile);

static bool reject_node(TerrainRuntime *terrain, uint32_t index,
                        RuntimeTile *tile, uint32_t generation) {
    bool rejected = terrain_quadtree_unavailable(&terrain->tree, index, generation);
    destroy_runtime_tile(terrain, tile);
    return rejected;
}

/* Rocky's GeometryPool shares topology while per-tile descriptors provide
   elevation and colour. This function performs the same merge for our Vulkan
   objects: copy shared buffer handles, then publish two tile textures. */
static bool upload_node(TerrainRuntime *terrain, uint32_t index) {
    TerrainQuadNode *node = &terrain->tree.nodes[index];
    RuntimeTile *runtime_tile = node->payload;
    if (!runtime_tile) return false;
    uint32_t generation = node->generation;
    TerrainTile *tile = &runtime_tile->tile;
    uint32_t interior = tile->header.sample_width - 2u * tile->header.gutter;
    if (interior != terrain->grid.size ||
        tile->header.sample_height - 2u * tile->header.gutter != terrain->grid.size)
        return reject_node(terrain, index, runtime_tile, generation);
    uint16_t *elevation = encode_elevation_r16(tile);
    if (!elevation || !terrain_quadtree_upload_begin(&terrain->tree, index, generation)) {
        free(elevation);
        return reject_node(terrain, index, runtime_tile, generation);
    }

    tile->base.vertices = terrain->grid.mesh.vertices;
    tile->base.vertex_count = terrain->grid.mesh.vertex_count;
    tile->base.indices = terrain->grid.mesh.indices;
    tile->base.index_count = terrain->grid.mesh.index_count;
    tile->base.vertex_buffer = terrain->grid.mesh.vertex_buffer;
    tile->base.index_buffer = terrain->grid.mesh.index_buffer;
    texture_load(terrain->renderer->device, terrain->renderer->allocator,
                 terrain->renderer->upload, &tile->base.texture,
                 tile->resolved_imagery_path, terrain->renderer->max_anisotropy);
    texture_create_elevation(terrain->renderer->device, terrain->renderer->allocator,
        terrain->renderer->upload, &runtime_tile->elevation, elevation,
        tile->header.sample_width, tile->header.sample_height);
    free(elevation);
    tile->base.material_set = renderer_allocate_terrain_set(terrain->renderer,
        tile->base.texture.view, tile->base.texture.sampler,
        runtime_tile->elevation.view, runtime_tile->elevation.sampler);
    runtime_tile->upload = upload_last_ticket(terrain->renderer->upload);
    return true;
}

static void complete_uploads(TerrainRuntime *terrain) {
    for (uint32_t i = 0; i < terrain->tree.node_count; ++i) {
        TerrainQuadNode *node = &terrain->tree.nodes[i];
        if (node->state != TERRAIN_TILE_UPLOAD_PENDING) continue;
        RuntimeTile *tile = node->payload;
        if (!tile || !upload_complete(terrain->renderer->upload, tile->upload)) continue;
        uint64_t gpu_bytes = texture_bytes(&tile->tile.base.texture, 4u) +
                             texture_bytes(&tile->elevation, 2u);
        terrain_quadtree_resident(&terrain->tree, i, node->generation, gpu_bytes);
    }
}

static uint32_t next_cpu_ready(const TerrainQuadtree *tree) {
    uint32_t best = TERRAIN_QUADTREE_INVALID_NODE;
    double priority = -INFINITY;
    for (uint32_t i = 0; i < tree->node_count; ++i) {
        if (tree->nodes[i].state == TERRAIN_TILE_CPU_READY &&
            tree->nodes[i].request_priority > priority) {
            best = i;
            priority = tree->nodes[i].request_priority;
        }
    }
    return best;
}

static void raster_uv(uint32_t size, uint32_t gutter, vec4s *out) {
    uint32_t interior = size - 2u * gutter;
    out->x = (float)(interior - 1u) / (float)size;
    out->y = out->x;
    out->z = ((float)gutter + 0.5f) / (float)size;
    out->w = out->z;
}

static bool fill_draw(TerrainRuntime *terrain, const TerrainQuadtreeView *view,
                      uint32_t node_index, RendererDraw *draw) {
    TerrainQuadNode *node = &terrain->tree.nodes[node_index];
    RuntimeTile *runtime_tile = node->payload;
    if (!runtime_tile) return false;
    TerrainTile *tile = &runtime_tile->tile;
    draw->mesh = &tile->base;
    draw->push.local_to_camera_relative = coordinate_local_to_camera_relative(
        &tile->header.local_to_world, view->camera_world);
    float span_x = (float)(tile->header.extent[2] - tile->header.extent[0]);
    float span_z = (float)(tile->header.extent[3] - tile->header.extent[1]);
    draw->push.geometry = (vec4s){{span_x, span_z,
        tile->header.height_range_m,
        fmaxf(span_x, span_z) * terrain->settings.skirt_ratio}};
    raster_uv(tile->header.sample_width, tile->header.gutter,
              &draw->push.elevation_uv);
    raster_uv(tile->base.texture.extent.width, tile->header.gutter,
              &draw->push.imagery_uv);
    /* A 4096 m phase is exactly periodic for every power-of-two surface
       scale used by terrain.frag. Reducing in double on the CPU preserves
       close-detail continuity at Earth-sized projected coordinates. */
    const double phase_period_m = 4096.0;
    draw->push.debug = (vec4s){{
        (float)node->key.level,
        surface_detail_phase(tile->header.local_to_world.translation.x,
                             phase_period_m),
        surface_detail_phase(tile->header.local_to_world.translation.y,
                             phase_period_m),
        surface_detail_phase(tile->header.local_to_world.translation.z,
                             phase_period_m),
    }};
    return true;
}

static void build_draws(TerrainRuntime *terrain,
                        const TerrainQuadtreeView *view) {
    terrain->draw_count = 0;
    for (uint32_t i = 0; i < terrain->tree.draw_count; ++i)
        if (fill_draw(terrain, view, terrain->tree.draw_nodes[i],
                      &terrain->draws[terrain->draw_count]))
            terrain->draw_count++;

    terrain->shadow_draw_count = 0;
    for (uint32_t i = 0; i < terrain->tree.shadow_count; ++i)
        if (fill_draw(terrain, view, terrain->tree.shadow_nodes[i],
                      &terrain->shadow_draws[terrain->shadow_draw_count]))
            terrain->shadow_draw_count++;
}

TerrainRuntime *terrain_runtime_create(Renderer *renderer,
                                       const char *dataset_root,
                                       const TerrainRuntimeSettings *settings) {
    if (!renderer || !dataset_root) return NULL;
    TerrainRuntimeSettings resolved = settings ? *settings
                                               : terrain_runtime_default_settings();
    if (!(resolved.skirt_ratio >= 0.0f)) return NULL;
    TerrainRuntime *terrain = calloc(1, sizeof(*terrain));
    if (!terrain) return NULL;
    terrain->renderer = renderer;
    terrain->settings = resolved;
    terrain->dataset_root = str_dup_n(dataset_root, strlen(dataset_root));
    if (!terrain->dataset_root ||
        !terrain_quadtree_init(&terrain->tree, &resolved.quadtree)) {
        terrain_runtime_destroy(terrain);
        return NULL;
    }
    terrain->draws = calloc(resolved.quadtree.max_nodes, sizeof(*terrain->draws));
    terrain->shadow_draws =
        calloc(resolved.quadtree.max_nodes, sizeof(*terrain->shadow_draws));
    if (!terrain->draws || !terrain->shadow_draws ||
        !terrain_quadtree_request_root(&terrain->tree) ||
        !load_node(terrain, 0)) {
        terrain_runtime_destroy(terrain);
        return NULL;
    }
    RuntimeTile *root = terrain->tree.nodes[0].payload;
    uint32_t grid_size = root->tile.header.sample_width -
                         2u * root->tile.header.gutter;
    if (!terrain_grid_create(&terrain->grid, grid_size)) {
        terrain_runtime_destroy(terrain);
        return NULL;
    }
    terrain_grid_upload(renderer, &terrain->grid);
    if (!upload_node(terrain, 0)) {
        terrain_runtime_destroy(terrain);
        return NULL;
    }
    upload_wait_idle(renderer->upload);
    complete_uploads(terrain);
    if (terrain->tree.nodes[0].state != TERRAIN_TILE_RESIDENT) {
        terrain_runtime_destroy(terrain);
        return NULL;
    }
    return terrain;
}

static void destroy_runtime_tile(TerrainRuntime *terrain, RuntimeTile *tile) {
    if (!tile) return;
    if (tile->tile.base.material_set)
        renderer_free_material_set(terrain->renderer, tile->tile.base.material_set);
    if (tile->tile.base.texture.image)
        texture_destroy(terrain->renderer->device, terrain->renderer->allocator,
                        &tile->tile.base.texture);
    if (tile->elevation.image)
        texture_destroy(terrain->renderer->device, terrain->renderer->allocator,
                        &tile->elevation);
    terrain_tile_unload(&tile->tile);
    free(tile);
}

void terrain_runtime_destroy(TerrainRuntime *terrain) {
    if (!terrain) return;
    if (terrain->renderer) renderer_wait_idle(terrain->renderer);
    if (terrain->tree.nodes) {
        for (uint32_t i = 0; i < terrain->tree.node_count; ++i)
            destroy_runtime_tile(terrain, terrain->tree.nodes[i].payload);
    }
    if (terrain->grid.vertices)
        terrain_grid_destroy(terrain->renderer, &terrain->grid);
    terrain_quadtree_destroy(&terrain->tree);
    free(terrain->draws);
    free(terrain->shadow_draws);
    free(terrain->dataset_root);
    free(terrain);
}

void terrain_runtime_update(TerrainRuntime *terrain,
                            const TerrainQuadtreeView *view, uint64_t frame) {
    if (!terrain || !view) return;
    complete_uploads(terrain);
    terrain_quadtree_select(&terrain->tree, view, frame);

    for (uint32_t loaded = 0;
         loaded < terrain->settings.quadtree.max_new_requests_per_frame;
         ++loaded) {
        uint32_t index = terrain_quadtree_next_request(&terrain->tree);
        if (index == TERRAIN_QUADTREE_INVALID_NODE || !load_node(terrain, index)) break;
    }
    for (uint32_t uploaded = 0;
         uploaded < terrain->settings.quadtree.max_uploads_per_frame;
         ++uploaded) {
        uint32_t index = next_cpu_ready(&terrain->tree);
        if (index == TERRAIN_QUADTREE_INVALID_NODE || !upload_node(terrain, index)) break;
    }
    complete_uploads(terrain);
    /* Casters are collected after streaming so newly-resident tiles are included;
       the collection ignores camera orientation to keep shadows view-stable. */
    terrain_quadtree_collect_casters(&terrain->tree, view);
    build_draws(terrain, view);
    terrain_quadtree_schedule_evictions(&terrain->tree);
}

const RendererDraw *terrain_runtime_draws(const TerrainRuntime *terrain,
                                          uint32_t *count) {
    if (count) *count = terrain ? terrain->draw_count : 0;
    return terrain ? terrain->draws : NULL;
}

const RendererDraw *terrain_runtime_shadow_draws(const TerrainRuntime *terrain,
                                                 uint32_t *count) {
    if (count) *count = terrain ? terrain->shadow_draw_count : 0;
    return terrain ? terrain->shadow_draws : NULL;
}

LocalToWorldTransform terrain_runtime_root_transform(const TerrainRuntime *terrain) {
    if (!terrain || !terrain->tree.nodes || !terrain->tree.nodes[0].payload)
        return coordinate_identity_transform((WorldPosition){0});
    RuntimeTile *root = terrain->tree.nodes[0].payload;
    return root->tile.header.local_to_world;
}

TerrainRuntimeStats terrain_runtime_stats(const TerrainRuntime *terrain) {
    if (!terrain) return (TerrainRuntimeStats){0};
    return (TerrainRuntimeStats){
        .known_tiles = terrain->tree.node_count,
        .resident_tiles = terrain->tree.resident_tiles,
        .drawn_tiles = terrain->draw_count,
        .cpu_bytes = terrain->tree.cpu_bytes,
        .gpu_bytes = terrain->tree.gpu_bytes,
    };
}

void terrain_runtime_collect_evictions(TerrainRuntime *terrain) {
    if (!terrain) return;
    for (;;) {
        uint32_t index = terrain_quadtree_next_eviction(&terrain->tree);
        if (index == TERRAIN_QUADTREE_INVALID_NODE) break;
        RuntimeTile *tile = terrain->tree.nodes[index].payload;
        destroy_runtime_tile(terrain, tile);
        terrain_quadtree_evicted(&terrain->tree, index);
    }
}
