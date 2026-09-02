#include "dungeon_grid.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static void valid_map_compiles_to_generic_level(void)
{
	const char *map = "#####\n#S..#\n###.#\n#..E#\n#####\n";
	DungeonLevel level = {0};
	DungeonLevelError error = {0};
	assert(dungeon_grid_compile_text(map, 2.0f, &level, &error));
	assert(level.surface_count == 3);
	assert(level.solid_count > 0);
	assert(level.collider_count == level.solid_count);
	assert(level.spawn.x == -2.0f && level.spawn.z == -2.0f);
	assert(level.exit.x == 2.0f && level.exit.z == 2.0f);
	dungeon_level_destroy(&level);
}

static void malformed_maps_are_rejected(void)
{
	DungeonLevel level = {0};
	DungeonLevelError error = {0};
	assert(!dungeon_grid_compile_text("###\n#S#\n##\n", 2.0f, &level, &error));
	assert(strstr(error.message, "length"));
	assert(!dungeon_grid_compile_text("#####\n#S@E#\n#####\n", 2.0f, &level, &error));
	assert(strstr(error.message, "unknown"));
	assert(!dungeon_grid_compile_text("#####\n#S#E#\n#####\n", 2.0f, &level, &error));
	assert(strstr(error.message, "unreachable"));
	assert(!dungeon_grid_compile_text("#####\n#S.E#\n#S..#\n#####\n", 2.0f, &level, &error));
	assert(strstr(error.message, "exactly one"));
}

int main(void)
{
	valid_map_compiles_to_generic_level();
	malformed_maps_are_rejected();
	puts("dungeon grid tests passed");
	return 0;
}
