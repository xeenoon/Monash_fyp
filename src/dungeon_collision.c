#include "dungeon_collision.h"

#include <math.h>

static float clampf(float value, float low, float high)
{
	return value < low ? low : value > high ? high : value;
}

static bool overlaps(DungeonPoint point, float radius, DungeonRect bounds)
{
	float closest_x = clampf(point.x, bounds.min.x, bounds.max.x);
	float closest_z = clampf(point.z, bounds.min.z, bounds.max.z);
	float dx = point.x - closest_x, dz = point.z - closest_z;
	return dx * dx + dz * dz < radius * radius;
}

static DungeonPoint move_axis(const DungeonCollider *colliders, uint32_t count, DungeonPoint start,
							  float amount, float radius, bool x_axis)
{
	DungeonPoint result = start;
	if (x_axis)
		result.x += amount;
	else
		result.z += amount;
	for (uint32_t i = 0; i < count; ++i)
	{
		if (colliders[i].type != DUNGEON_COLLIDER_AABB || !overlaps(result, radius,
															colliders[i].bounds))
			continue;
		DungeonRect bounds = colliders[i].bounds;
		if (x_axis)
		{
			float dz = result.z - clampf(result.z, bounds.min.z, bounds.max.z);
			float reach = sqrtf(fmaxf(radius * radius - dz * dz, 0.0f));
			result.x = amount > 0.0f ? bounds.min.x - reach : bounds.max.x + reach;
		}
		else
		{
			float dx = result.x - clampf(result.x, bounds.min.x, bounds.max.x);
			float reach = sqrtf(fmaxf(radius * radius - dx * dx, 0.0f));
			result.z = amount > 0.0f ? bounds.min.z - reach : bounds.max.z + reach;
		}
	}
	return result;
}

DungeonPoint dungeon_collision_move(const DungeonCollider *colliders, uint32_t collider_count,
									DungeonPoint start, DungeonPoint displacement, float radius)
{
	if (!colliders || !collider_count || radius <= 0.0f)
		return (DungeonPoint){start.x + displacement.x, start.z + displacement.z};
	float distance = sqrtf(displacement.x * displacement.x + displacement.z * displacement.z);
	uint32_t steps = (uint32_t)ceilf(distance / fmaxf(radius * 0.5f, 0.01f));
	if (steps < 1u)
		steps = 1u;
	DungeonPoint step = {displacement.x / (float)steps, displacement.z / (float)steps};
	DungeonPoint result = start;
	for (uint32_t i = 0; i < steps; ++i)
	{
		result = move_axis(colliders, collider_count, result, step.x, radius, true);
		result = move_axis(colliders, collider_count, result, step.z, radius, false);
	}
	return result;
}

