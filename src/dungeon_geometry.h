#pragma once

/* The shared value types every dungeon module builds on: a 2D point in the
 * XZ ground plane, an axis-aligned rectangle, a wall-footprint segment, and a
 * puddle placement. Deliberately dependency-free so dungeon_field.h,
 * dungeon_contour.h, dungeon_cave.h, and dungeon_level.h can each include it
 * without forming a cycle (dungeon_level.h needs field/contour types, which
 * need DungeonPoint -- it cannot be the one to define it). */

typedef struct
{
	float x, z;
} DungeonPoint;

typedef struct
{
	DungeonPoint min, max;
} DungeonRect;

/* A wall footprint segment: a piece of the 0.5-isoline contour. Used both as
 * a swept-circle collider and, decimated further, as a 2D light-occlusion
 * blocker. Both ends are always corners produced by the same marching-squares
 * pass as the visual geometry, so collision/occlusion never disagree with
 * what is actually drawn. */
typedef struct
{
	DungeonPoint a, b;
} DungeonSegment;

typedef enum
{
	DUNGEON_COLLIDER_SEGMENT
} DungeonColliderType;

typedef struct
{
	DungeonColliderType type;
	DungeonSegment segment;
} DungeonCollider;

typedef struct
{
	DungeonPoint center;
	float radius;
} DungeonPuddle;
