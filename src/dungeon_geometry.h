#pragma once

#include <stdint.h>

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

/* Which minigame a doorway's lock presents. NONE is a plain door: it still
 * swings and still blocks light and movement until opened, it just needs no
 * puzzle solved first. */
typedef enum
{
	DUNGEON_LOCK_NONE,
	DUNGEON_LOCK_PIN_TUMBLER,
	DUNGEON_LOCK_SAFE_PINS,
	DUNGEON_LOCK_PRISM
} DungeonLockKind;

/* A door standing in a hallway aperture. Deliberately NOT carved into the
 * occupancy field: the field stays open through the doorway, and the door is a
 * dynamic object contributing its own collider, light occluder, and draw. That
 * is what lets it open without recompiling any level geometry.
 *
 * `blocker` must stay a segment, never an axis-aligned rect, to match what
 * dungeon_light_segment_blocked and segment_crosses_blocker in mesh.frag both
 * expect of a light blocker. */
typedef struct
{
	DungeonPoint center;
	float yaw;				/* door plane normal, taken from the hallway axis */
	float half_width;		/* half the hallway width the door spans */
	DungeonSegment blocker; /* collider + light occluder while closed */
	DungeonLockKind lock;
	uint32_t seed; /* per-door, so a lock's combination follows DUNGEON_SEED */
} DungeonDoorway;
