#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct
{
	float x, z;
} DungeonPoint;

typedef struct
{
	DungeonPoint min, max;
} DungeonRect;

typedef struct
{
	DungeonRect footprint;
	float elevation;
	uint32_t material;
} DungeonSurface;

typedef struct
{
	DungeonRect footprint;
	float base_y, height;
	uint32_t material;
} DungeonSolid;

typedef enum
{
	DUNGEON_COLLIDER_AABB
} DungeonColliderType;

typedef struct
{
	DungeonColliderType type;
	DungeonRect bounds;
} DungeonCollider;

typedef struct
{
	DungeonSurface *surfaces;
	uint32_t surface_count;
	DungeonSolid *solids;
	uint32_t solid_count;
	DungeonCollider *colliders;
	uint32_t collider_count;
	DungeonPoint spawn;
	DungeonPoint exit;
	float cell_size;
	float floor_y;
} DungeonLevel;

typedef struct
{
	char message[256];
	size_t line;
	size_t column;
} DungeonLevelError;

void dungeon_level_destroy(DungeonLevel *level);

