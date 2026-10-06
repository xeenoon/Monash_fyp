#include "dungeon_cave.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
int main(void)
{
	for (unsigned seed = 1; seed <= 40; ++seed)
	{
		DungeonCaveParams params = dungeon_cave_default_params(seed);
		DungeonCaveResult c = {0};
		assert(dungeon_cave_generate(&params, &c));
		assert(c.monk_spawn.x < c.spawn.x && c.monk_spawn.z < c.spawn.z);
		for (unsigned i = 0; i < 32; ++i)
		{
			float a = i * 6.2831853f / 32;
			DungeonPoint p = {c.monk_spawn.x + .85f * cosf(a), c.monk_spawn.z + .85f * sinf(a)};
			assert(dungeon_field_sample(&c.field, p) >= .5f);
		}
		assert(dungeon_field_reachable(&c.field, .5f, c.spawn, c.monk_spawn, NULL));
		for (unsigned i = 0; i < c.puddle_count; ++i)
			assert(hypotf(c.puddles[i].center.x - c.monk_spawn.x,
						  c.puddles[i].center.z - c.monk_spawn.z) >= c.puddles[i].radius + 1.1f);
		for (unsigned i = 0; i < c.door_count; ++i)
			assert(hypotf(c.doors[i].center.x - c.monk_spawn.x,
						  c.doors[i].center.z - c.monk_spawn.z) > 2);
		dungeon_cave_destroy(&c);
	}
	DungeonLevel a = {0}, b = {0};
	DungeonLevelError error = {0};
	DungeonCaveParams params = dungeon_cave_default_params(91);
	assert(dungeon_cave_compile(&params, &a, &error));
	char path[128];
	snprintf(path, sizeof(path), "/tmp/gameport-monk-cache-%ld", (long)getpid());
	assert(dungeon_level_cache_save(path, 1, 91, DUNGEON_CAVE_CACHE_VERSION, &a));
	assert(dungeon_level_cache_load(path, 1, 91, DUNGEON_CAVE_CACHE_VERSION, &b) ==
		   DUNGEON_LEVEL_CACHE_LOADED);
	assert(a.has_monk_spawn && b.has_monk_spawn);
	assert(a.monk_spawn.x == b.monk_spawn.x && a.monk_spawn.z == b.monk_spawn.z);
	unlink(path);
	dungeon_level_destroy(&a);
	dungeon_level_destroy(&b);
	puts("Monk clearance and reachability passed for 40 seeds; cached placement roundtrip passed");
}
