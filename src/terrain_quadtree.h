#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "coordinate.h"

#define TERRAIN_QUADTREE_INVALID_NODE UINT32_MAX

typedef struct {
    uint32_t level;
    uint32_t x;
    uint32_t y;
} TerrainTileKey;

typedef enum {
    TERRAIN_TILE_UNLOADED,
    TERRAIN_TILE_REQUESTED,
    TERRAIN_TILE_CPU_READY,
    TERRAIN_TILE_UPLOAD_PENDING,
    TERRAIN_TILE_RESIDENT,
    TERRAIN_TILE_EVICT_PENDING,
    /* A missing file is remembered so a dataset leaf is not requested every frame. */
    TERRAIN_TILE_UNAVAILABLE,
} TerrainTileLifecycleState;

/* Eight double-precision world corners are retained for Rocky-style tight
   frustum tests. The sphere supports conservative distance/SSE calculations. */
typedef struct {
    double corners[8][3];
    WorldPosition center;
    double radius_m;
    float geometric_error_m;
} TerrainTileBounds;

typedef struct {
    WorldPosition camera_world;
    double forward[3];
    double up[3];
    double vertical_fov_radians;
    double aspect;
    double near_plane_m;
    double viewport_height_px;
} TerrainQuadtreeView;

typedef struct {
    float split_threshold_px;
    float merge_threshold_px;
    uint32_t max_nodes;
    uint32_t max_new_requests_per_frame;
    uint32_t max_uploads_per_frame;
    uint32_t max_resident_tiles;
    uint64_t max_cpu_bytes;
    uint64_t max_gpu_bytes;
    uint64_t eviction_frames;
} TerrainQuadtreeSettings;

typedef struct {
    TerrainTileKey key;
    TerrainTileLifecycleState state;
    uint32_t generation;
    uint32_t parent;
    uint32_t children[4];
    float parent_uv_scale_bias[4]; /* scale U,V; bias U,V */
    TerrainTileBounds bounds;
    double error_px;
    double request_priority;
    uint64_t last_visible_frame;
    uint64_t cpu_bytes;
    uint64_t gpu_bytes;
    void *payload;                 /* owned by the runtime/factory, not this core */
    bool split_active;
    bool visible;
} TerrainQuadNode;

typedef struct {
    TerrainQuadtreeSettings settings;
    TerrainQuadNode *nodes;
    uint32_t node_count;
    uint32_t *draw_nodes;
    uint32_t draw_count;
    /* Shadow casters: the same LOD set as draw_nodes but without camera-frustum
       culling, so shadows depend only on camera position, never orientation. */
    uint32_t *shadow_nodes;
    uint32_t shadow_count;
    uint64_t frame;
    uint64_t cpu_bytes;
    uint64_t gpu_bytes;
    uint32_t resident_tiles;
    uint32_t requests_issued_this_frame;
} TerrainQuadtree;

TerrainQuadtreeSettings terrain_quadtree_default_settings(void);
bool terrain_quadtree_init(TerrainQuadtree *tree,
                           const TerrainQuadtreeSettings *settings);
void terrain_quadtree_destroy(TerrainQuadtree *tree);

TerrainTileKey terrain_tile_key_child(TerrainTileKey parent, uint32_t quadrant);
uint32_t terrain_tile_key_quadrant(TerrainTileKey key);
void terrain_tile_parent_uv_scale_bias(TerrainTileKey child, float out[4]);

/* Root and child requests follow unloaded -> requested. A request completion
   must carry the generation observed when work began; stale work is rejected. */
bool terrain_quadtree_request_root(TerrainQuadtree *tree);
uint32_t terrain_quadtree_next_request(const TerrainQuadtree *tree);
bool terrain_quadtree_cpu_ready(TerrainQuadtree *tree, uint32_t node,
                                uint32_t generation,
                                const TerrainTileBounds *bounds,
                                uint64_t cpu_bytes, void *payload);
bool terrain_quadtree_upload_begin(TerrainQuadtree *tree, uint32_t node,
                                   uint32_t generation);
bool terrain_quadtree_resident(TerrainQuadtree *tree, uint32_t node,
                               uint32_t generation, uint64_t gpu_bytes);
bool terrain_quadtree_unavailable(TerrainQuadtree *tree, uint32_t node,
                                  uint32_t generation);
bool terrain_quadtree_cancel(TerrainQuadtree *tree, uint32_t node);

/* Selects renderable nodes, issuing a frame-limited set of child requests.
   A parent remains selected until its complete four-child replacement is resident. */
void terrain_quadtree_select(TerrainQuadtree *tree,
                             const TerrainQuadtreeView *view, uint64_t frame);

/* Fills shadow_nodes with the shadow-caster set: the same distance-based LOD as
   the last terrain_quadtree_select, but with no frustum culling and no new tile
   requests, so the caster set (and therefore the shadows) is independent of where
   the camera looks. Must be called after terrain_quadtree_select. */
void terrain_quadtree_collect_casters(TerrainQuadtree *tree,
                                      const TerrainQuadtreeView *view);

/* Marks expired/LRU sibling quads for eviction. The runtime destroys their
   resources only after the renderer's prior-frame fence is known complete. */
void terrain_quadtree_schedule_evictions(TerrainQuadtree *tree);
uint32_t terrain_quadtree_next_eviction(const TerrainQuadtree *tree);
bool terrain_quadtree_evicted(TerrainQuadtree *tree, uint32_t node);

const char *terrain_tile_state_string(TerrainTileLifecycleState state);
