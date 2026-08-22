#include "terrain_quadtree.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

TerrainQuadtreeSettings terrain_quadtree_default_settings(void) {
    return (TerrainQuadtreeSettings){
        .split_threshold_px = 3.0f,
        .merge_threshold_px = 2.0f,
        .max_nodes = 4096,
        .max_new_requests_per_frame = 4,
        .max_uploads_per_frame = 2,
        .max_resident_tiles = 256,
        .max_cpu_bytes = UINT64_C(256) * 1024u * 1024u,
        .max_gpu_bytes = UINT64_C(512) * 1024u * 1024u,
        /* This is a cache expiry, not the budget enforcement path. A generous
           default avoids churn on uncapped render loops; hard budgets still
           evict immediately when crossed. */
        .eviction_frames = 3600,
    };
}

static TerrainQuadNode empty_node(void) {
    TerrainQuadNode node = {
        .state = TERRAIN_TILE_UNLOADED,
        .generation = 1,
        .parent = TERRAIN_QUADTREE_INVALID_NODE,
    };
    for (uint32_t i = 0; i < 4; ++i)
        node.children[i] = TERRAIN_QUADTREE_INVALID_NODE;
    return node;
}

bool terrain_quadtree_init(TerrainQuadtree *tree,
                           const TerrainQuadtreeSettings *settings) {
    if (!tree) return false;
    TerrainQuadtreeSettings resolved = settings ? *settings
                                                : terrain_quadtree_default_settings();
    if (resolved.max_nodes < 5 || resolved.merge_threshold_px < 0.0f ||
        resolved.split_threshold_px <= resolved.merge_threshold_px ||
        resolved.max_new_requests_per_frame == 0 ||
        resolved.max_uploads_per_frame == 0)
        return false;

    TerrainQuadNode *nodes = calloc(resolved.max_nodes, sizeof(*nodes));
    uint32_t *draw_nodes = malloc((size_t)resolved.max_nodes * sizeof(*draw_nodes));
    if (!nodes || !draw_nodes) {
        free(nodes);
        free(draw_nodes);
        return false;
    }
    *tree = (TerrainQuadtree){
        .settings = resolved,
        .nodes = nodes,
        .node_count = 1,
        .draw_nodes = draw_nodes,
    };
    tree->nodes[0] = empty_node();
    tree->nodes[0].key = (TerrainTileKey){0, 0, 0};
    return true;
}

void terrain_quadtree_destroy(TerrainQuadtree *tree) {
    if (!tree) return;
    free(tree->nodes);
    free(tree->draw_nodes);
    *tree = (TerrainQuadtree){0};
}

/* Adapted from rocky/TileKey.cpp::createChildKey and ::getQuadrant
   (Pelican Mapping, MIT). Keeping Rocky's quadrant numbering makes the pager
   rules directly comparable; our y axis is the same top-left/south-positive XYZ. */
TerrainTileKey terrain_tile_key_child(TerrainTileKey parent, uint32_t quadrant) {
    TerrainTileKey child = {parent.level + 1u, parent.x * 2u, parent.y * 2u};
    if (quadrant == 1u || quadrant == 3u) child.x++;
    if (quadrant == 2u || quadrant == 3u) child.y++;
    return child;
}

uint32_t terrain_tile_key_quadrant(TerrainTileKey key) {
    if (key.level == 0) return 0;
    return (key.x & 1u) | ((key.y & 1u) << 1u);
}

/* Adapted from rocky/TerrainTileNode.cpp::scaleBias (Pelican Mapping, MIT).
   Rocky's matrix is reduced to the four values our shader consumes. Bias V is
   top-left based here because .trn XYZ and decoded imagery both use y=0 north. */
void terrain_tile_parent_uv_scale_bias(TerrainTileKey child, float out[4]) {
    uint32_t quadrant = terrain_tile_key_quadrant(child);
    out[0] = 0.5f;
    out[1] = 0.5f;
    out[2] = (quadrant & 1u) ? 0.5f : 0.0f;
    out[3] = (quadrant & 2u) ? 0.5f : 0.0f;
}

bool terrain_quadtree_request_root(TerrainQuadtree *tree) {
    if (!tree || !tree->nodes || tree->nodes[0].state != TERRAIN_TILE_UNLOADED)
        return false;
    tree->nodes[0].state = TERRAIN_TILE_REQUESTED;
    tree->nodes[0].request_priority = DBL_MAX;
    return true;
}

uint32_t terrain_quadtree_next_request(const TerrainQuadtree *tree) {
    if (!tree) return TERRAIN_QUADTREE_INVALID_NODE;
    uint32_t best = TERRAIN_QUADTREE_INVALID_NODE;
    double priority = -DBL_MAX;
    for (uint32_t i = 0; i < tree->node_count; ++i) {
        if (tree->nodes[i].state == TERRAIN_TILE_REQUESTED &&
            tree->nodes[i].request_priority > priority) {
            best = i;
            priority = tree->nodes[i].request_priority;
        }
    }
    return best;
}

bool terrain_quadtree_cpu_ready(TerrainQuadtree *tree, uint32_t index,
                                uint32_t generation,
                                const TerrainTileBounds *bounds,
                                uint64_t cpu_bytes, void *payload) {
    if (!tree || index >= tree->node_count || !bounds) return false;
    TerrainQuadNode *node = &tree->nodes[index];
    if (node->state != TERRAIN_TILE_REQUESTED || node->generation != generation)
        return false;
    node->bounds = *bounds;
    node->cpu_bytes = cpu_bytes;
    node->payload = payload;
    node->state = TERRAIN_TILE_CPU_READY;
    tree->cpu_bytes += cpu_bytes;
    return true;
}

bool terrain_quadtree_upload_begin(TerrainQuadtree *tree, uint32_t index,
                                   uint32_t generation) {
    if (!tree || index >= tree->node_count) return false;
    TerrainQuadNode *node = &tree->nodes[index];
    if (node->state != TERRAIN_TILE_CPU_READY || node->generation != generation)
        return false;
    node->state = TERRAIN_TILE_UPLOAD_PENDING;
    return true;
}

bool terrain_quadtree_resident(TerrainQuadtree *tree, uint32_t index,
                               uint32_t generation, uint64_t gpu_bytes) {
    if (!tree || index >= tree->node_count) return false;
    TerrainQuadNode *node = &tree->nodes[index];
    if (node->state != TERRAIN_TILE_UPLOAD_PENDING || node->generation != generation)
        return false;
    node->gpu_bytes = gpu_bytes;
    node->state = TERRAIN_TILE_RESIDENT;
    tree->gpu_bytes += gpu_bytes;
    tree->resident_tiles++;
    return true;
}

bool terrain_quadtree_unavailable(TerrainQuadtree *tree, uint32_t index,
                                  uint32_t generation) {
    if (!tree || index >= tree->node_count) return false;
    TerrainQuadNode *node = &tree->nodes[index];
    bool in_progress = node->state == TERRAIN_TILE_REQUESTED ||
                       node->state == TERRAIN_TILE_CPU_READY ||
                       node->state == TERRAIN_TILE_UPLOAD_PENDING;
    if (!in_progress || node->generation != generation)
        return false;
    if (node->cpu_bytes <= tree->cpu_bytes) tree->cpu_bytes -= node->cpu_bytes;
    node->cpu_bytes = 0;
    node->payload = NULL;
    node->state = TERRAIN_TILE_UNAVAILABLE;
    node->generation++;
    return true;
}

bool terrain_quadtree_cancel(TerrainQuadtree *tree, uint32_t index) {
    if (!tree || index >= tree->node_count) return false;
    TerrainQuadNode *node = &tree->nodes[index];
    if (node->state == TERRAIN_TILE_RESIDENT ||
        node->state == TERRAIN_TILE_EVICT_PENDING ||
        node->state == TERRAIN_TILE_UNAVAILABLE ||
        node->state == TERRAIN_TILE_UNLOADED)
        return false;
    if (node->cpu_bytes <= tree->cpu_bytes) tree->cpu_bytes -= node->cpu_bytes;
    node->cpu_bytes = 0;
    node->payload = NULL;
    node->state = TERRAIN_TILE_UNLOADED;
    node->generation++;
    return true;
}

static double dot3(const double a[3], const double b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static void cross3(const double a[3], const double b[3], double out[3]) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static bool normalize3(double value[3]) {
    double length = sqrt(dot3(value, value));
    if (!(length > 0.0) || !isfinite(length)) return false;
    value[0] /= length; value[1] /= length; value[2] /= length;
    return true;
}

/* Adapted from rocky/SurfaceNode.h::isVisible (Pelican Mapping, MIT).
   Rocky tests all eight elevation-aware corners against each frustum face. We
   do the same in camera coordinates to avoid importing a scene-graph frustum. */
static bool tile_visible(const TerrainTileBounds *bounds,
                         const TerrainQuadtreeView *view) {
    double forward[3] = {view->forward[0], view->forward[1], view->forward[2]};
    double up_hint[3] = {view->up[0], view->up[1], view->up[2]};
    double right[3], up[3];
    if (!normalize3(forward)) return true;
    cross3(forward, up_hint, right);
    if (!normalize3(right)) return true;
    cross3(right, forward, up);
    if (!normalize3(up)) return true;

    double tan_v = tan(view->vertical_fov_radians * 0.5);
    double tan_h = tan_v * view->aspect;
    if (!(tan_v > 0.0) || !(tan_h > 0.0)) return true;

    unsigned outside[5] = {0};
    for (uint32_t i = 0; i < 8; ++i) {
        double relative[3] = {
            bounds->corners[i][0] - view->camera_world.x,
            bounds->corners[i][1] - view->camera_world.y,
            bounds->corners[i][2] - view->camera_world.z,
        };
        double x = dot3(relative, right);
        double y = dot3(relative, up);
        double z = dot3(relative, forward);
        outside[0] += z < view->near_plane_m;
        outside[1] += x + z * tan_h < 0.0;
        outside[2] += -x + z * tan_h < 0.0;
        outside[3] += y + z * tan_v < 0.0;
        outside[4] += -y + z * tan_v < 0.0;
    }
    for (uint32_t plane = 0; plane < 5; ++plane)
        if (outside[plane] == 8) return false;
    return true;
}

static double screen_error(const TerrainTileBounds *bounds,
                           const TerrainQuadtreeView *view) {
    double dx = bounds->center.x - view->camera_world.x;
    double dy = bounds->center.y - view->camera_world.y;
    double dz = bounds->center.z - view->camera_world.z;
    double distance = sqrt(dx*dx + dy*dy + dz*dz) - bounds->radius_m;
    if (distance < 1.0) distance = 1.0;
    double focal_px = 0.5 * view->viewport_height_px /
                      tan(0.5 * view->vertical_fov_radians);
    return (double)bounds->geometric_error_m * focal_px / distance;
}

static bool ensure_children(TerrainQuadtree *tree, uint32_t parent_index,
                            double priority) {
    TerrainQuadNode *parent = &tree->nodes[parent_index];
    if (parent->children[0] == TERRAIN_QUADTREE_INVALID_NODE) {
        if (tree->node_count > tree->settings.max_nodes - 4u) return false;
        uint32_t first = tree->node_count;
        tree->node_count += 4u;
        for (uint32_t quadrant = 0; quadrant < 4; ++quadrant) {
            uint32_t index = first + quadrant;
            tree->nodes[index] = empty_node();
            tree->nodes[index].key = terrain_tile_key_child(parent->key, quadrant);
            tree->nodes[index].parent = parent_index;
            terrain_tile_parent_uv_scale_bias(tree->nodes[index].key,
                                               tree->nodes[index].parent_uv_scale_bias);
            parent->children[quadrant] = index;
        }
    }

    for (uint32_t quadrant = 0; quadrant < 4; ++quadrant) {
        TerrainQuadNode *child = &tree->nodes[parent->children[quadrant]];
        if (child->state == TERRAIN_TILE_UNLOADED &&
            tree->requests_issued_this_frame <
                tree->settings.max_new_requests_per_frame) {
            child->state = TERRAIN_TILE_REQUESTED;
            child->request_priority = priority;
            tree->requests_issued_this_frame++;
        } else if (child->state == TERRAIN_TILE_REQUESTED &&
                   priority > child->request_priority) {
            child->request_priority = priority;
        }
    }
    return true;
}

static bool all_children_resident(const TerrainQuadtree *tree,
                                  const TerrainQuadNode *node) {
    if (node->children[0] == TERRAIN_QUADTREE_INVALID_NODE) return false;
    for (uint32_t i = 0; i < 4; ++i)
        if (tree->nodes[node->children[i]].state != TERRAIN_TILE_RESIDENT)
            return false;
    return true;
}

/* Adapted from rocky/TerrainTileNode.cpp::accept (Pelican Mapping, MIT).
   This preserves its most important invariant: traverse children only as one
   complete quad; otherwise render the parent and keep requesting the quad. */
static void select_node(TerrainQuadtree *tree, uint32_t index,
                        const TerrainQuadtreeView *view) {
    TerrainQuadNode *node = &tree->nodes[index];
    if (node->state != TERRAIN_TILE_RESIDENT) return;
    node->visible = tile_visible(&node->bounds, view);
    if (!node->visible) return;

    node->last_visible_frame = tree->frame;
    node->error_px = screen_error(&node->bounds, view);
    bool wants_children = node->error_px > tree->settings.split_threshold_px ||
        (node->split_active && node->error_px >= tree->settings.merge_threshold_px);
    node->split_active = wants_children;

    if (wants_children && ensure_children(tree, index, node->error_px) &&
        all_children_resident(tree, node)) {
        for (uint32_t i = 0; i < 4; ++i)
            select_node(tree, node->children[i], view);
        return;
    }

    if (tree->draw_count < tree->settings.max_nodes)
        tree->draw_nodes[tree->draw_count++] = index;
}

void terrain_quadtree_select(TerrainQuadtree *tree,
                             const TerrainQuadtreeView *view, uint64_t frame) {
    if (!tree || !view || !tree->nodes) return;
    tree->frame = frame;
    tree->draw_count = 0;
    tree->requests_issued_this_frame = 0;
    for (uint32_t i = 0; i < tree->node_count; ++i)
        tree->nodes[i].visible = false;
    select_node(tree, 0, view);
}

static bool node_drawn(const TerrainQuadtree *tree, uint32_t index) {
    for (uint32_t i = 0; i < tree->draw_count; ++i)
        if (tree->draw_nodes[i] == index) return true;
    return false;
}

static void mark_sibling_quad(TerrainQuadtree *tree, TerrainQuadNode *parent) {
    for (uint32_t q = 0; q < 4; ++q) {
        TerrainQuadNode *child = &tree->nodes[parent->children[q]];
        child->state = TERRAIN_TILE_EVICT_PENDING;
    }
}

static bool has_live_descendants(const TerrainQuadtree *tree,
                                 const TerrainQuadNode *node) {
    if (node->children[0] == TERRAIN_QUADTREE_INVALID_NODE) return false;
    for (uint32_t q = 0; q < 4; ++q) {
        const TerrainQuadNode *child = &tree->nodes[node->children[q]];
        if (child->state != TERRAIN_TILE_UNLOADED &&
            child->state != TERRAIN_TILE_UNAVAILABLE)
            return true;
        if (has_live_descendants(tree, child)) return true;
    }
    return false;
}

/* Adapted from TerrainTilePager::update's tracker flush (Pelican Mapping, MIT).
   Rocky expires complete child quads so a parent never loses one arbitrary
   sibling. Here the same rule is driven by frame age and explicit byte budgets. */
void terrain_quadtree_schedule_evictions(TerrainQuadtree *tree) {
    if (!tree) return;
    uint64_t projected_cpu = tree->cpu_bytes;
    uint64_t projected_gpu = tree->gpu_bytes;
    uint32_t projected_resident = tree->resident_tiles;

    for (;;) {
        uint32_t best_parent = TERRAIN_QUADTREE_INVALID_NODE;
        uint64_t oldest = UINT64_MAX;
        bool over_budget = projected_cpu > tree->settings.max_cpu_bytes ||
                           projected_gpu > tree->settings.max_gpu_bytes ||
                           projected_resident > tree->settings.max_resident_tiles;
        for (uint32_t p = 0; p < tree->node_count; ++p) {
            TerrainQuadNode *parent = &tree->nodes[p];
            if (parent->children[0] == TERRAIN_QUADTREE_INVALID_NODE) continue;
            bool eligible = true;
            uint64_t newest = 0;
            for (uint32_t q = 0; q < 4; ++q) {
                uint32_t child_index = parent->children[q];
                TerrainQuadNode *child = &tree->nodes[child_index];
                if (child->state != TERRAIN_TILE_RESIDENT ||
                    node_drawn(tree, child_index) ||
                    has_live_descendants(tree, child)) {
                    eligible = false;
                    break;
                }
                if (child->last_visible_frame > newest)
                    newest = child->last_visible_frame;
            }
            bool expired = tree->frame > newest &&
                tree->frame - newest > tree->settings.eviction_frames;
            if (eligible && (over_budget || expired) && newest < oldest) {
                oldest = newest;
                best_parent = p;
            }
        }
        if (best_parent == TERRAIN_QUADTREE_INVALID_NODE) break;
        TerrainQuadNode *parent = &tree->nodes[best_parent];
        for (uint32_t q = 0; q < 4; ++q) {
            TerrainQuadNode *child = &tree->nodes[parent->children[q]];
            projected_cpu -= child->cpu_bytes;
            projected_gpu -= child->gpu_bytes;
            projected_resident--;
        }
        mark_sibling_quad(tree, parent);
    }
}

uint32_t terrain_quadtree_next_eviction(const TerrainQuadtree *tree) {
    if (!tree) return TERRAIN_QUADTREE_INVALID_NODE;
    for (uint32_t i = 1; i < tree->node_count; ++i)
        if (tree->nodes[i].state == TERRAIN_TILE_EVICT_PENDING) return i;
    return TERRAIN_QUADTREE_INVALID_NODE;
}

bool terrain_quadtree_evicted(TerrainQuadtree *tree, uint32_t index) {
    if (!tree || index >= tree->node_count) return false;
    TerrainQuadNode *node = &tree->nodes[index];
    if (node->state != TERRAIN_TILE_EVICT_PENDING) return false;
    if (node->cpu_bytes <= tree->cpu_bytes) tree->cpu_bytes -= node->cpu_bytes;
    if (node->gpu_bytes <= tree->gpu_bytes) tree->gpu_bytes -= node->gpu_bytes;
    if (tree->resident_tiles) tree->resident_tiles--;
    node->cpu_bytes = 0;
    node->gpu_bytes = 0;
    node->payload = NULL;
    node->state = TERRAIN_TILE_UNLOADED;
    node->generation++;
    node->split_active = false;
    return true;
}

const char *terrain_tile_state_string(TerrainTileLifecycleState state) {
    switch (state) {
    case TERRAIN_TILE_UNLOADED: return "unloaded";
    case TERRAIN_TILE_REQUESTED: return "requested";
    case TERRAIN_TILE_CPU_READY: return "CPU ready";
    case TERRAIN_TILE_UPLOAD_PENDING: return "upload pending";
    case TERRAIN_TILE_RESIDENT: return "GPU resident";
    case TERRAIN_TILE_EVICT_PENDING: return "evict pending";
    case TERRAIN_TILE_UNAVAILABLE: return "unavailable";
    }
    return "unknown";
}
