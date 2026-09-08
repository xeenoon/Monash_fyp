#pragma once
#include "mesh.h"
#include <stdbool.h>

/* Preorder, stackless BVH. Internal nodes skip to end on a miss;
 * leaves contain one triangle. All coordinates are dungeon world
 * metres. */
typedef struct
{
	float minimum[3];
	uint32_t end;
	float maximum[3];
	uint32_t leaf;
	float a[4], b[4], c[4];
} DungeonShadowNode;
typedef struct
{
	float a[3], b[3], c[3];
} DungeonShadowTriangle;
typedef struct
{
	DungeonShadowNode *nodes;
	uint32_t count;
} DungeonShadow;
bool dungeon_shadow_build(const DungeonShadowTriangle *triangles,
						  uint32_t count, DungeonShadow *out);
void dungeon_shadow_destroy(DungeonShadow *shadow);
/* Nearest two-sided hit, used by fixture placement and regression
 * tests. */
bool dungeon_shadow_trace(const DungeonShadow *shadow,
						  const float origin[3],
						  const float direction[3], float *distance,
						  float normal[3]);
