#include "dungeon_cave.h"
#include "dungeon_field.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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

/* The invariant the whole hybrid layout rests on. A door that does not
 * actually gate the route is worse than no door: the player picks a lock for
 * nothing, or walks around it and never sees the puzzle at all. The generator
 * is supposed to have dropped every such candidate already, so re-asking the
 * question here is a check on the generator, not on the player. */
static void every_locked_door_is_a_real_chokepoint(void)
{
	for (uint32_t seed = 1; seed <= 30u; ++seed)
	{
		DungeonCaveParams params = dungeon_cave_default_params(seed);
		DungeonCaveResult cave = {0};
		assert(dungeon_cave_generate(&params, &cave));
		size_t count = (size_t)cave.field.width * cave.field.height;
		uint8_t *overlay = calloc(count, sizeof(*overlay));
		assert(overlay);

		/* With every door open, the exit is reachable -- otherwise the level
		 * is unfinishable no matter how well the locks are picked. */
		assert(dungeon_field_reachable(&cave.field, 0.5f, cave.spawn, cave.exit, NULL));
		assert(cave.door_count <= params.door_max);

		for (uint32_t i = 0; i < cave.door_count; ++i)
		{
			const DungeonDoorway *door = &cave.doors[i];
			assert(door->lock != DUNGEON_LOCK_NONE);
			/* The doorway stands in open floor: it is a dynamic object, never
			 * carved into the field. */
			assert(dungeon_field_sample(&cave.field, door->center) >= 0.5f);
			assert(door->half_width > 0.0f);

			memset(overlay, 0, count);
			float normal_x = cosf(door->yaw), normal_z = sinf(door->yaw);
			float across_x = -normal_z, across_z = normal_x;
			float half_thickness = cave.field.cell_size * 2.0f;
			float half_span = door->half_width + cave.field.cell_size * 2.0f;
			for (uint32_t z = 0; z < cave.field.height; ++z)
				for (uint32_t x = 0; x < cave.field.width; ++x)
				{
					DungeonPoint world = dungeon_field_corner_world(&cave.field, x, z);
					float dx = world.x - door->center.x, dz = world.z - door->center.z;
					float along = dx * normal_x + dz * normal_z;
					float lateral = dx * across_x + dz * across_z;
					if (fabsf(along) <= half_thickness && fabsf(lateral) <= half_span)
						overlay[(size_t)z * cave.field.width + x] = 1u;
				}
			assert(!dungeon_field_reachable(&cave.field, 0.5f, cave.spawn, cave.exit, overlay));

			/* The blocker segment must span the aperture and stay a segment:
			 * dungeon_light_segment_blocked and segment_crosses_blocker in
			 * mesh.frag both read it as one. */
			float bx = door->blocker.b.x - door->blocker.a.x;
			float bz = door->blocker.b.z - door->blocker.a.z;
			assert(fabsf(sqrtf(bx * bx + bz * bz) - door->half_width * 2.0f) < 1e-3f);
		}
		free(overlay);
		dungeon_cave_destroy(&cave);
	}
}

/* A hard rect stamp has to survive as a square corner, otherwise hallways read
 * as cave and the layout change is invisible. Checked on the field directly:
 * along a room's straight wall the occupancy step is exact, with no partial
 * blur values in between. */
static void hard_rect_stamps_keep_square_corners(void)
{
	DungeonField field = {0};
	assert(dungeon_field_create(81u, 81u, 0.25f, (DungeonPoint){-10.0f, -10.0f}, &field));
	dungeon_field_stamp_rect(&field, (DungeonRect){{-4.0f, -3.0f}, {4.0f, 3.0f}}, 0.0f);
	for (uint32_t z = 0; z < field.height; ++z)
		for (uint32_t x = 0; x < field.width; ++x)
		{
			float value = dungeon_field_get(&field, x, z);
			assert(value == 0.0f || value == 1.0f); /* no feathered band at all */
		}
	assert(dungeon_field_sample(&field, (DungeonPoint){0.0f, 0.0f}) >= 0.5f);
	assert(dungeon_field_sample(&field, (DungeonPoint){-9.0f, -9.0f}) < 0.5f);

	/* And a masked blur must leave unmasked corners bit-identical, which is
	 * what lets rooms and pockets share one field. */
	size_t count = (size_t)field.width * field.height;
	float *before = malloc(count * sizeof(*before));
	uint8_t *mask = calloc(count, sizeof(*mask));
	assert(before && mask);
	memcpy(before, field.values, count * sizeof(*before));
	dungeon_field_blur_masked(&field, 3u, mask); /* mask all-zero: nothing may move */
	for (size_t i = 0; i < count; ++i)
		assert(field.values[i] == before[i]);
	free(before);
	free(mask);
	dungeon_field_destroy(&field);
}

int main(void)
{
	identical_seed_reproduces_byte_identical_field();
	different_seeds_produce_different_fields();
	generated_caves_are_connected_and_puddles_are_clear();
	every_locked_door_is_a_real_chokepoint();
	hard_rect_stamps_keep_square_corners();
	puts("dungeon cave tests passed");
	return 0;
}
