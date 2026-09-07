#pragma once

#include "dungeon_field.h"
#include "dungeon_geometry.h"
#include "dungeon_level.h"

#include <stdbool.h>
#include <stdint.h>

/* Tunable knobs for the procedural cave frontend. All distances are metres,
 * angles radians. dungeon_cave_default_params gives a level roughly the size
 * of the old hand-authored example.map. */
typedef struct
{
	uint32_t seed;
	float extent_m;   /* field spans [-extent/2, extent/2] on both axes */
	float cell_size;  /* metres between field corners; 0.25 is the tuned value */
	uint32_t worm_count;
	uint32_t worm_steps;
	float step_length;
	float turn_rate; /* max heading change per step */
	float min_radius, max_radius;
	uint32_t chamber_count;
	float chamber_radius;
	uint32_t blur_iterations;
	uint32_t puddle_max;
	float puddle_min_radius, puddle_max_radius;
} DungeonCaveParams;

DungeonCaveParams dungeon_cave_default_params(uint32_t seed);

typedef struct
{
	DungeonField field;
	DungeonPoint spawn, exit;
	DungeonPuddle *puddles;
	uint32_t puddle_count;
} DungeonCaveResult;

/* Deterministic: identical params (including seed) always produce a
 * byte-identical field, spawn, exit, and puddle set. Returns false only on
 * allocation failure or a param combination too tight to carve safely (e.g.
 * extent_m too small for max_radius). */
bool dungeon_cave_generate(const DungeonCaveParams *params, DungeonCaveResult *out);
void dungeon_cave_destroy(DungeonCaveResult *result);

/* Generates a cave and immediately compiles it into a DungeonLevel via
 * dungeon_level_compile_field -- the cave-frontend counterpart of
 * dungeon_grid_compile_text/_file. floor_y is 0, wall_height a fixed 2.4 m,
 * matching the old grid frontend's wall height. */
bool dungeon_cave_compile(const DungeonCaveParams *params, DungeonLevel *out,
						  DungeonLevelError *error);
