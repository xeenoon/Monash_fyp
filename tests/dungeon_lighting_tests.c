#include "dungeon_grid.h"
#include "dungeon_lighting.h"

#include <assert.h>
#include <stdio.h>

int main(void)
{
	assert(dungeon_light_attenuation(0.0f, 5.0f) >
		   dungeon_light_attenuation(1.0f, 5.0f));
	assert(dungeon_light_attenuation(5.0f, 5.0f) == 0.0f);
	assert(dungeon_light_attenuation(6.0f, 5.0f) == 0.0f);
	DungeonCollider blocker = {
		.type = DUNGEON_COLLIDER_AABB, .bounds = {{1.0f, -1.0f}, {2.0f, 1.0f}}};
	assert(dungeon_light_segment_blocked((DungeonPoint){0.0f, 0.0f},
									   (DungeonPoint){3.0f, 0.0f}, &blocker, 1));
	assert(!dungeon_light_segment_blocked((DungeonPoint){0.0f, 2.0f},
										(DungeonPoint){3.0f, 2.0f}, &blocker, 1));
	assert(!dungeon_light_segment_blocked((DungeonPoint){0.0f, 0.0f},
										(DungeonPoint){1.0f, 0.0f}, &blocker, 1));
	DungeonLevel level = {0};
	DungeonLevelError error = {0};
	assert(dungeon_grid_compile_text("#####\n#S.E#\n#####\n", 2.0f, &level, &error));
	DungeonLight lights[DUNGEON_MAX_LIGHTS];
	uint32_t count = dungeon_lighting_build(&level, lights, DUNGEON_MAX_LIGHTS);
	assert(count >= 2u && count <= DUNGEON_MAX_LIGHTS);
	assert(lights[0].position.x == level.spawn.x);
	assert(lights[count - 1].position.x == level.exit.x);
	dungeon_level_destroy(&level);
	puts("dungeon lighting tests passed");
	return 0;
}
