#include "dungeon_collision.h"

#include <math.h>

static DungeonPoint closest_point_on_segment(DungeonPoint point, DungeonSegment segment)
{
	float dx = segment.b.x - segment.a.x, dz = segment.b.z - segment.a.z;
	float length2 = dx * dx + dz * dz;
	float t = length2 > 1e-12f
				 ? ((point.x - segment.a.x) * dx + (point.z - segment.a.z) * dz) / length2
				 : 0.0f;
	t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
	return (DungeonPoint){segment.a.x + dx * t, segment.a.z + dz * t};
}

/* Pushes `point` outside every colliding segment by exactly `radius`, one
 * pass over the full collider list. Called several times per substep so
 * corners -- where resolving one wall can push into its neighbour -- settle
 * instead of leaving the player embedded a little in one of them. */
static DungeonPoint resolve_once(const DungeonCollider *colliders, uint32_t count, DungeonPoint point,
								 float radius)
{
	for (uint32_t i = 0; i < count; ++i)
	{
		if (colliders[i].type != DUNGEON_COLLIDER_SEGMENT)
			continue;
		DungeonSegment segment = colliders[i].segment;
		DungeonPoint closest = closest_point_on_segment(point, segment);
		float dx = point.x - closest.x, dz = point.z - closest.z;
		float distance = sqrtf(dx * dx + dz * dz);
		if (distance >= radius)
			continue;
		float push_x, push_z;
		if (distance > 1e-6f)
		{
			push_x = dx / distance;
			push_z = dz / distance;
		}
		else
		{
			/* Degenerate: centre sits exactly on the wall line. Push toward
			 * open floor using the contour's winding (open floor is on the
			 * left of a->b) rather than an arbitrary direction. */
			float ex = segment.b.x - segment.a.x, ez = segment.b.z - segment.a.z;
			float length = sqrtf(ex * ex + ez * ez);
			if (length < 1e-6f)
			{
				push_x = 0.0f;
				push_z = 1.0f;
			}
			else
			{
				push_x = -ez / length;
				push_z = ex / length;
			}
		}
		point.x = closest.x + push_x * radius;
		point.z = closest.z + push_z * radius;
	}
	return point;
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
		result.x += step.x;
		result.z += step.z;
		for (uint32_t iteration = 0; iteration < 4u; ++iteration)
			result = resolve_once(colliders, collider_count, result, radius);
	}
	return result;
}
