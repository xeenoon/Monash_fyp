#include "dungeon_cave.h"
#include "dungeon_field.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* Re-runs keep_largest_component on a copy and requires it to be a no-op:
 * the only way that holds is if the generated field already has exactly one
 * open component, i.e. no disconnected pocket survived generation. */
static void assert_single_component(const DungeonField *field, DungeonPoint spawn)
{
	DungeonField copy = {0};
	assert(dungeon_field_create(field->width, field->height, field->cell_size, field->origin,
								&copy));
	size_t count = (size_t)field->width * field->height;
	memcpy(copy.values, field->values, count * sizeof(*copy.values));
	dungeon_field_keep_largest_component(&copy, 0.5f, spawn);
	for (size_t i = 0; i < count; ++i)
		assert(copy.values[i] == field->values[i]);
	dungeon_field_destroy(&copy);
}

static float brute_force_distance_to_nearest_solid(const DungeonField *field, DungeonPoint point)
{
	float best = 1e9f;
	for (uint32_t z = 0; z < field->height; ++z)
		for (uint32_t x = 0; x < field->width; ++x)
		{
			if (dungeon_field_get(field, x, z) >= 0.5f)
				continue;
			DungeonPoint corner = dungeon_field_corner_world(field, x, z);
			float dx = corner.x - point.x, dz = corner.z - point.z;
			float distance = sqrtf(dx * dx + dz * dz);
			if (distance < best)
				best = distance;
		}
	return best;
}

static void identical_seed_reproduces_byte_identical_field(void)
{
	DungeonCaveParams params = dungeon_cave_default_params(12345u);
	params.extent_m = 20.0f;
	params.worm_steps = 60u;
	DungeonCaveResult a = {0}, b = {0};
	assert(dungeon_cave_generate(&params, &a));
	assert(dungeon_cave_generate(&params, &b));
	assert(a.field.width == b.field.width && a.field.height == b.field.height);
	size_t count = (size_t)a.field.width * a.field.height;
	for (size_t i = 0; i < count; ++i)
		assert(a.field.values[i] == b.field.values[i]);
	assert(a.spawn.x == b.spawn.x && a.spawn.z == b.spawn.z);
	assert(a.exit.x == b.exit.x && a.exit.z == b.exit.z);
	assert(a.puddle_count == b.puddle_count);
	for (uint32_t i = 0; i < a.puddle_count; ++i)
	{
		assert(a.puddles[i].center.x == b.puddles[i].center.x);
		assert(a.puddles[i].center.z == b.puddles[i].center.z);
		assert(a.puddles[i].radius == b.puddles[i].radius);
	}
	dungeon_cave_destroy(&a);
	dungeon_cave_destroy(&b);
}

static void different_seeds_produce_different_fields(void)
{
	DungeonCaveParams pa = dungeon_cave_default_params(1u);
	DungeonCaveParams pb = dungeon_cave_default_params(2u);
	pa.extent_m = pb.extent_m = 20.0f;
	pa.worm_steps = pb.worm_steps = 60u;
	DungeonCaveResult a = {0}, b = {0};
	assert(dungeon_cave_generate(&pa, &a));
	assert(dungeon_cave_generate(&pb, &b));
	bool differs = false;
	size_t count = (size_t)a.field.width * a.field.height;
	for (size_t i = 0; i < count && !differs; ++i)
		if (a.field.values[i] != b.field.values[i])
			differs = true;
	assert(differs);
	dungeon_cave_destroy(&a);
	dungeon_cave_destroy(&b);
}

static void generated_caves_are_connected_and_puddles_are_clear(void)
{
	for (uint32_t seed = 1; seed <= 5u; ++seed)
	{
		DungeonCaveParams params = dungeon_cave_default_params(seed);
		DungeonCaveResult cave = {0};
		assert(dungeon_cave_generate(&params, &cave));

		assert(dungeon_field_sample(&cave.field, cave.spawn) >= 0.5f);
		assert(dungeon_field_sample(&cave.field, cave.exit) >= 0.5f);
		float dx = cave.exit.x - cave.spawn.x, dz = cave.exit.z - cave.spawn.z;
		assert(sqrtf(dx * dx + dz * dz) > 3.0f); /* not a trivially short hop */

		assert_single_component(&cave.field, cave.spawn);

		for (uint32_t i = 0; i < cave.puddle_count; ++i)
		{
			assert(dungeon_field_sample(&cave.field, cave.puddles[i].center) >= 0.5f);
			float clearance =
				brute_force_distance_to_nearest_solid(&cave.field, cave.puddles[i].center);
			assert(clearance >= cave.puddles[i].radius - params.cell_size * 2.0f);
		}

		dungeon_cave_destroy(&cave);
	}
}

int main(void)
{
	identical_seed_reproduces_byte_identical_field();
	different_seeds_produce_different_fields();
	generated_caves_are_connected_and_puddles_are_clear();
	puts("dungeon cave tests passed");
	return 0;
}
