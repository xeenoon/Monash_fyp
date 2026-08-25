#pragma once

#include <stddef.h>
#include <stdint.h>

#include "renderer.h"
#include "terrain_quadtree.h"

typedef struct TerrainRuntime TerrainRuntime;

typedef struct
{
	TerrainQuadtreeSettings quadtree;
	float skirt_ratio;
} TerrainRuntimeSettings;

typedef struct
{
	uint32_t known_tiles;
	uint32_t resident_tiles;
	uint32_t drawn_tiles;
	uint64_t cpu_bytes;
	uint64_t gpu_bytes;
} TerrainRuntimeStats;

TerrainRuntimeSettings terrain_runtime_default_settings(void);

/* Creates and synchronously makes the root resident so there is always a valid
   fallback. Descendants are paged incrementally by terrain_runtime_update. */
TerrainRuntime *terrain_runtime_create(Renderer *renderer, const char *dataset_root,
									   const TerrainRuntimeSettings *settings);
void terrain_runtime_destroy(TerrainRuntime *terrain);

void terrain_runtime_update(TerrainRuntime *terrain, const TerrainQuadtreeView *view,
							uint64_t frame);

const RendererDraw *terrain_runtime_draws(const TerrainRuntime *terrain, uint32_t *count);
/* Shadow-caster draws: the same tiles the shadow pass should render, selected
   without camera-frustum culling so shadows stay stable as the camera turns. */
const RendererDraw *terrain_runtime_shadow_draws(const TerrainRuntime *terrain, uint32_t *count);
LocalToWorldTransform terrain_runtime_root_transform(const TerrainRuntime *terrain);
/* Largest horizontal side of the root footprint, in profile metres. */
float terrain_runtime_root_span(const TerrainRuntime *terrain);
TerrainRuntimeStats terrain_runtime_stats(const TerrainRuntime *terrain);

/* Call after renderer_draw_frame. Its fence wait made resources absent from the
   new draw list safe to destroy, while the new submission does not reference them. */
void terrain_runtime_collect_evictions(TerrainRuntime *terrain);
