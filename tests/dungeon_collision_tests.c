#include "dungeon_collision.h"
#include "dungeon_player.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

/* Expresses an axis-aligned rectangle as its four boundary segments -- the
 * shape every real collider is decimated from, but easy to hand-write for a
 * test fixture. */
static void rect_segments(DungeonRect rect, DungeonCollider out[4])
{
	DungeonPoint bl = {rect.min.x, rect.min.z}, br = {rect.max.x, rect.min.z};
	DungeonPoint tr = {rect.max.x, rect.max.z}, tl = {rect.min.x, rect.max.z};
	out[0] = (DungeonCollider){DUNGEON_COLLIDER_SEGMENT, {bl, tl}}; /* left */
	out[1] = (DungeonCollider){DUNGEON_COLLIDER_SEGMENT, {br, tr}}; /* right */
	out[2] = (DungeonCollider){DUNGEON_COLLIDER_SEGMENT, {bl, br}}; /* bottom */
	out[3] = (DungeonCollider){DUNGEON_COLLIDER_SEGMENT, {tl, tr}}; /* top */
}

static void wall_stops_and_slides(void)
{
	DungeonCollider wall[4];
	rect_segments((DungeonRect){{1.0f, -2.0f}, {2.0f, 2.0f}}, wall);
	DungeonPoint stopped =
		dungeon_collision_move(wall, 4, (DungeonPoint){0, 0}, (DungeonPoint){5, 0}, 0.35f);
	assert(fabsf(stopped.x - 0.65f) < 1e-3f);
	DungeonPoint slid = dungeon_collision_move(wall, 4, (DungeonPoint){0, 0},
										  (DungeonPoint){2, 1}, 0.35f);
	assert(slid.x <= 0.66f);
	assert(slid.z > 0.9f);
}

static void parallel_motion_does_not_snap_to_a_wall_corner(void)
{
	DungeonCollider wall[4];
	rect_segments((DungeonRect){{1.0f, -2.0f}, {2.0f, 2.0f}}, wall);
	DungeonPoint moved = dungeon_collision_move(wall, 4, (DungeonPoint){1.5f, 2.34f},
										  (DungeonPoint){0.5f, 0.0f}, 0.35f);
	assert(moved.x > 1.9f);
	assert(moved.z > 2.3f);

	DungeonPoint other_side = dungeon_collision_move(
		wall, 4, (DungeonPoint){1.5f, -2.34f}, (DungeonPoint){-0.5f, 0.0f}, 0.35f);
	assert(other_side.x < 1.1f);
	assert(other_side.z < -2.3f);
}

static void player_movement_is_normalized_and_reaches_exit(void)
{
	DungeonLevel level = {.exit = {2.0f, 2.0f}};
	DungeonPlayer straight, diagonal;
	dungeon_player_init(&straight, (DungeonPoint){0});
	dungeon_player_init(&diagonal, (DungeonPoint){0});
	dungeon_player_update(&straight, &level, 1, 0, 90, 1.0f);
	dungeon_player_update(&diagonal, &level, 1, 1, 90, 1.0f);
	float straight_distance = hypotf(straight.position.x, straight.position.z);
	float diagonal_distance = hypotf(diagonal.position.x, diagonal.position.z);
	assert(fabsf(straight_distance - diagonal_distance) < 1e-5f);
	diagonal.position = (DungeonPoint){1.9f, 2.0f};
	assert(dungeon_player_update(&diagonal, &level, 0, 0, 90, 0.016f));
	assert(!dungeon_player_update(&diagonal, &level, 0, 0, 90, 0.016f));
}

int main(void)
{
	wall_stops_and_slides();
	parallel_motion_does_not_snap_to_a_wall_corner();
	player_movement_is_normalized_and_reaches_exit();
	puts("dungeon collision tests passed");
	return 0;
}
