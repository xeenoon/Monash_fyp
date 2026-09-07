#include "dungeon_cave.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define DUNGEON_CAVE_TAU 6.28318530718f

/* Small local splitmix64, seeded from the public uint32_t seed. Deliberately
 * not libc rand(): its state is explicit, so identical params always produce
 * an identical cave regardless of what else in the process has called
 * rand() first. */
typedef struct
{
	uint64_t state;
} CaveRng;

static CaveRng cave_rng_create(uint32_t seed)
{
	CaveRng rng = {.state = (uint64_t)seed * 0x9E3779B97F4A7C15ULL + 0xA24BAED4963EE407ULL};
	return rng;
}

static uint64_t cave_rng_next(CaveRng *rng)
{
	uint64_t z = (rng->state += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

static float cave_rng_01(CaveRng *rng) { return (float)((cave_rng_next(rng) >> 40) * (1.0 / 16777216.0)); }

static float cave_rng_range(CaveRng *rng, float low, float high)
{
	return low + (high - low) * cave_rng_01(rng);
}

static uint32_t cave_rng_index(CaveRng *rng, uint32_t count)
{
	if (!count)
		return 0u;
	uint32_t index = (uint32_t)(cave_rng_01(rng) * (float)count);
	return index < count ? index : count - 1u;
}

DungeonCaveParams dungeon_cave_default_params(uint32_t seed)
{
	return (DungeonCaveParams){
		.seed = seed,
		.extent_m = 44.0f,
		.cell_size = 0.25f,
		.worm_count = 6u,
		.worm_steps = 110u,
		.step_length = 0.55f,
		.turn_rate = 0.35f,
		.min_radius = 1.2f,
		.max_radius = 2.4f,
		.chamber_count = 4u,
		.chamber_radius = 4.0f,
		.blur_iterations = 3u,
		.puddle_max = 14u,
		.puddle_min_radius = 0.6f,
		.puddle_max_radius = 1.8f,
	};
}

static float clampf(float value, float low, float high)
{
	return value < low ? low : (value > high ? high : value);
}

bool dungeon_cave_generate(const DungeonCaveParams *params, DungeonCaveResult *out)
{
	if (!params || !out || !(params->cell_size > 0.0f) || !params->worm_count)
		return false;
	*out = (DungeonCaveResult){0};
	float half_extent = params->extent_m * 0.5f;
	/* Carving and blurring never reach the field border, so every contour
	 * loop dungeon_contour_extract finds is guaranteed closed. */
	float margin = fmaxf(params->max_radius, params->chamber_radius) +
				  (float)params->blur_iterations * params->cell_size * 2.0f + 1.0f;
	float safe_half = half_extent - margin;
	if (!(safe_half > 1.0f))
		return false;
	uint32_t corners = (uint32_t)ceilf(params->extent_m / params->cell_size) + 1u;
	DungeonPoint origin = {-half_extent, -half_extent};
	if (!dungeon_field_create(corners, corners, params->cell_size, origin, &out->field))
		return false;

	CaveRng rng = cave_rng_create(params->seed);
	DungeonPoint spawn_point = {0.0f, 0.0f};

	uint32_t waypoints_per_worm = params->worm_steps / 5u + 1u;
	uint32_t waypoint_capacity = 1u + params->worm_count * waypoints_per_worm;
	DungeonPoint *waypoints = malloc(waypoint_capacity * sizeof(*waypoints));
	if (!waypoints)
	{
		dungeon_field_destroy(&out->field);
		return false;
	}
	uint32_t waypoint_count = 0;
	waypoints[waypoint_count++] = spawn_point;

	for (uint32_t worm = 0; worm < params->worm_count; ++worm)
	{
		DungeonPoint position =
			worm == 0u ? spawn_point : waypoints[cave_rng_index(&rng, waypoint_count)];
		float heading = cave_rng_range(&rng, -DUNGEON_CAVE_TAU * 0.5f, DUNGEON_CAVE_TAU * 0.5f);
		float radius = cave_rng_range(&rng, params->min_radius, params->max_radius);
		for (uint32_t step = 0; step < params->worm_steps; ++step)
		{
			heading += cave_rng_range(&rng, -params->turn_rate, params->turn_rate);
			radius = clampf(radius + cave_rng_range(&rng, -0.08f, 0.08f), params->min_radius,
							params->max_radius);
			DungeonPoint next = {position.x + cosf(heading) * params->step_length,
								 position.z + sinf(heading) * params->step_length};
			if (next.x < -safe_half || next.x > safe_half || next.z < -safe_half ||
				next.z > safe_half)
			{
				/* Turn back toward the centre instead of carving into the
				 * guaranteed-solid border. */
				heading = atan2f(-position.z, -position.x) + cave_rng_range(&rng, -0.5f, 0.5f);
				next = (DungeonPoint){position.x + cosf(heading) * params->step_length,
									  position.z + sinf(heading) * params->step_length};
				next.x = clampf(next.x, -safe_half, safe_half);
				next.z = clampf(next.z, -safe_half, safe_half);
			}
			position = next;
			dungeon_field_stamp_disc(&out->field, position, radius, params->cell_size * 1.5f);
			if (step % 5u == 0u)
				waypoints[waypoint_count++] = position;
		}
	}

	for (uint32_t chamber = 0; chamber < params->chamber_count; ++chamber)
	{
		DungeonPoint center = waypoints[cave_rng_index(&rng, waypoint_count)];
		for (uint32_t lobe = 0; lobe < 3u; ++lobe)
		{
			float angle = cave_rng_range(&rng, -DUNGEON_CAVE_TAU * 0.5f, DUNGEON_CAVE_TAU * 0.5f);
			float offset = cave_rng_range(&rng, 0.0f, params->chamber_radius * 0.4f);
			DungeonPoint lobe_center = {clampf(center.x + cosf(angle) * offset, -safe_half, safe_half),
									   clampf(center.z + sinf(angle) * offset, -safe_half, safe_half)};
			float lobe_radius = params->chamber_radius * cave_rng_range(&rng, 0.65f, 1.0f);
			dungeon_field_stamp_disc(&out->field, lobe_center, lobe_radius, params->cell_size * 1.5f);
		}
	}
	free(waypoints);

	dungeon_field_blur(&out->field, params->blur_iterations);
	dungeon_field_keep_largest_component(&out->field, 0.5f, spawn_point);

	out->spawn = spawn_point;
	if (!dungeon_field_bfs_farthest(&out->field, 0.5f, spawn_point, &out->exit))
	{
		dungeon_field_destroy(&out->field);
		return false;
	}

	if (!params->puddle_max)
		return true;
	size_t cell_total = (size_t)out->field.width * out->field.height;
	float *distance = malloc(cell_total * sizeof(*distance));
	out->puddles = malloc(params->puddle_max * sizeof(*out->puddles));
	if (!distance || !out->puddles)
	{
		free(distance);
		free(out->puddles);
		out->puddles = NULL;
		dungeon_field_destroy(&out->field);
		return false;
	}
	dungeon_field_distance_to_solid(&out->field, 0.5f, distance);

	uint32_t attempts = params->puddle_max * 20u + 40u;
	for (uint32_t attempt = 0; attempt < attempts && out->puddle_count < params->puddle_max;
		++attempt)
	{
		uint32_t x = cave_rng_index(&rng, out->field.width);
		uint32_t z = cave_rng_index(&rng, out->field.height);
		if (dungeon_field_get(&out->field, x, z) < 0.5f)
			continue;
		float clearance = distance[(size_t)z * out->field.width + x];
		float radius = cave_rng_range(&rng, params->puddle_min_radius, params->puddle_max_radius);
		if (clearance < radius + 0.3f)
		{
			radius = clearance - 0.3f;
			if (radius < params->puddle_min_radius * 0.5f)
				continue;
		}
		DungeonPoint center = dungeon_field_corner_world(&out->field, x, z);
		bool too_close = false;
		for (uint32_t i = 0; i < out->puddle_count; ++i)
		{
			float dx = center.x - out->puddles[i].center.x, dz = center.z - out->puddles[i].center.z;
			float min_gap = radius + out->puddles[i].radius + 0.5f;
			if (dx * dx + dz * dz < min_gap * min_gap)
			{
				too_close = true;
				break;
			}
		}
		if (too_close)
			continue;
		out->puddles[out->puddle_count++] = (DungeonPuddle){.center = center, .radius = radius};
	}
	free(distance);
	return true;
}

bool dungeon_cave_compile(const DungeonCaveParams *params, DungeonLevel *out,
						  DungeonLevelError *error)
{
	DungeonCaveResult cave = {0};
	if (!dungeon_cave_generate(params, &cave))
	{
		if (error)
			snprintf(error->message, sizeof(error->message), "cave generation failed for seed %u",
					params ? params->seed : 0u);
		return false;
	}
	/* dungeon_level_compile_field takes ownership of cave.field (moves it on
	 * success, destroys it on failure) and copies the puddles, so only the
	 * puddle array is still ours to free here. */
	bool ok = dungeon_level_compile_field(&cave.field, cave.spawn, cave.exit, cave.puddles,
										  cave.puddle_count, 0.0f, 2.4f, out, error);
	free(cave.puddles);
	return ok;
}

void dungeon_cave_destroy(DungeonCaveResult *result)
{
	if (!result)
		return;
	dungeon_field_destroy(&result->field);
	free(result->puddles);
	*result = (DungeonCaveResult){0};
}
