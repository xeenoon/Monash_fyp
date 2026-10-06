#pragma once

#include "dungeon_contour.h"
#include "dungeon_field.h"
#include "dungeon_geometry.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct
{
	/* The field and its raw (full-resolution) contour loops, kept around
	 * because lighting placement and future tooling need to resample the
	 * cave, not just draw it. */
	DungeonField field;
	DungeonContourSet contours;

	/* Flat XZ triangle fills sharing the contours' exact boundary -- lifted
	 * to floor_y / (floor_y + wall_height) by dungeon_mesh_build. */
	DungeonTriangleMesh floor_triangles;
	DungeonTriangleMesh plateau_triangles;

	DungeonCollider *colliders; /* contour decimated to ~0.4 m, for the player */
	uint32_t collider_count;
	DungeonSegment *occluders; /* contour decimated to ~1.0 m, for light blockers */
	uint32_t occluder_count;

	DungeonPuddle *puddles;
	uint32_t puddle_count;

	/* Doorways standing in hallway apertures. Not part of the field, the
	 * contours, or the collider/occluder arrays above: a door is a dynamic
	 * object whose collider and light occluder the scene adds and drops as it
	 * opens, which is what lets it open without recompiling any geometry. */
	DungeonDoorway *doors;
	uint32_t door_count;

	DungeonPoint spawn;
	DungeonPoint exit;
	DungeonPoint monk_spawn;
	bool has_monk_spawn;
	float floor_y;
	float wall_height;
} DungeonLevel;

typedef struct
{
	char message[256];
	size_t line;
	size_t column;
} DungeonLevelError;

/* On-disk procedural-level cache. The caller supplies a content version tied
 * to the generator: changing generation semantics must change that version,
 * while representation changes are covered by the cache format itself. */
typedef enum
{
	DUNGEON_LEVEL_CACHE_LOADED,
	DUNGEON_LEVEL_CACHE_MISSING,
	DUNGEON_LEVEL_CACHE_STALE,
	DUNGEON_LEVEL_CACHE_INVALID,
	DUNGEON_LEVEL_CACHE_IO_ERROR
} DungeonLevelCacheResult;

/* Shared compile path: turns a rasterized occupancy field into every piece of
 * derived geometry a frontend needs (contours, floor/plateau fills, collider
 * and occluder segments). Both dungeon_grid (ASCII maps) and dungeon_cave
 * (procedural caves) rasterize into a DungeonField and call this -- nothing
 * past this point knows or cares which frontend produced the field. Takes
 * ownership of `field` (moved into the output on success; destroyed on
 * failure) and copies `puddles` and `doors`. */
bool dungeon_level_compile_field(DungeonField *field, DungeonPoint spawn, DungeonPoint exit,
								 const DungeonPuddle *puddles, uint32_t puddle_count,
								 const DungeonDoorway *doors, uint32_t door_count, float floor_y,
								 float wall_height, DungeonLevel *out, DungeonLevelError *error);

void dungeon_level_destroy(DungeonLevel *level);

/* Reads only a cache header. This lets the game restore a persistent
 * dungeon's seed and display name without constructing its level at startup. */
DungeonLevelCacheResult dungeon_level_cache_probe(const char *path, uint32_t dungeon_id,
										   uint32_t content_version, uint32_t *out_seed);
/* Loads/saves the complete compiled DungeonLevel. Files are versioned,
 * checksummed and written through a temporary file followed by an atomic
 * rename, so an interrupted write is treated as a miss rather than a level. */
DungeonLevelCacheResult dungeon_level_cache_load(const char *path, uint32_t dungeon_id,
										  uint32_t seed, uint32_t content_version,
										  DungeonLevel *out);
bool dungeon_level_cache_save(const char *path, uint32_t dungeon_id, uint32_t seed,
							  uint32_t content_version, const DungeonLevel *level);

/* Print dungeon layout in human-readable text format: spawn, exit, doors with types, puddles. */
void dungeon_level_print(const DungeonLevel *level);
