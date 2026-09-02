#include "dungeon_collision.h"
#include "dungeon_player.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static void wall_stops_and_slides(void)
{
	DungeonCollider wall = {.type = DUNGEON_COLLIDER_AABB,
						  .bounds = {{1.0f, -2.0f}, {2.0f, 2.0f}}};
	DungeonPoint stopped =
		dungeon_collision_move(&wall, 1, (DungeonPoint){0, 0}, (DungeonPoint){5, 0}, 0.35f);
	assert(fabsf(stopped.x - 0.65f) < 1e-4f);
	DungeonPoint slid = dungeon_collision_move(&wall, 1, (DungeonPoint){0, 0},
										  (DungeonPoint){2, 1}, 0.35f);
	assert(slid.x <= 0.6501f);
	assert(slid.z > 0.9f);
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
	player_movement_is_normalized_and_reaches_exit();
	puts("dungeon collision tests passed");
	return 0;
}
