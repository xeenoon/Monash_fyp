#pragma once

#include "dungeon_field.h"
#include "dungeon_geometry.h"
#include "dungeon_level.h"

#include <stdbool.h>
#include <stdint.h>

/* Tunable knobs for the procedural frontend. All distances are metres, angles
 * radians. dungeon_cave_default_params gives a level roughly the size of the
 * old hand-authored example.map.
 *
 * The layout is a hybrid: rectilinear rooms joined by axis-aligned hallways
 * (hard rect stamps, so their corners survive marching squares square), with
 * organic cave pockets grown off room walls (disc chains, selectively blurred
 * so they alone read as rounded rock). The two halves are kept apart by
 * construction -- every pocket is a validated dead end -- which is what lets a
 * door in a hallway be a real chokepoint rather than something the cave half
 * can quietly tunnel around. */
typedef struct
{
	uint32_t seed;
	float extent_m;  /* field spans [-extent/2, extent/2] on both axes */
	float cell_size; /* metres between field corners; 0.25 is the tuned value */

	/* Rooms are laid out on a jittered grid. room_count is a request: the grid
	 * is squared up, so the level ends with ceil(sqrt(n))^2-ish rooms. */
	uint32_t room_count;
	float room_min_m, room_max_m;
	float hall_width_m;	 /* fixed: the door mesh is built to span exactly this */
	uint32_t extra_loops; /* non-spanning-tree hallways, for non-linearity */

	/* Organic pockets. worm_* drives dead-end grottos grown off room walls;
	 * chamber_* drives single-disc alcoves straddling a room wall. */
	uint32_t worm_count;
	uint32_t worm_steps;
	float step_length;
	float turn_rate; /* max heading change per step */
	float min_radius, max_radius;
	uint32_t chamber_count;
	float chamber_radius;
	uint32_t blur_iterations; /* applied to organic corners only */

	uint32_t door_max; /* upper bound on locked doors actually placed */

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
	/* Every doorway here is verified: solidifying its footprint disconnects
	 * spawn from exit. Candidates that failed that test are dropped, never
	 * emitted as decoration, so a lock is always worth picking. */
	DungeonDoorway *doors;
	uint32_t door_count;
} DungeonCaveResult;

/* Deterministic: identical params (including seed) always produce a
 * byte-identical field, spawn, exit, puddle set, and door set. Returns false
 * only on allocation failure or a param combination too tight to lay out
 * safely (e.g. extent_m too small for the requested room grid). */
bool dungeon_cave_generate(const DungeonCaveParams *params, DungeonCaveResult *out);
void dungeon_cave_destroy(DungeonCaveResult *result);

/* Generates a level and immediately compiles it into a DungeonLevel via
 * dungeon_level_compile_field -- the procedural counterpart of
 * dungeon_grid_compile_text/_file. floor_y is 0, wall_height a fixed 2.4 m,
 * matching the grid frontend's wall height. */
bool dungeon_cave_compile(const DungeonCaveParams *params, DungeonLevel *out,
						  DungeonLevelError *error);
