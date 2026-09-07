#include "dungeon_lighting.h"

#include <math.h>
#include <stdlib.h>

static bool segments_intersect(DungeonPoint p1, DungeonPoint p2, DungeonPoint p3, DungeonPoint p4)
{
	float d1x = p2.x - p1.x, d1z = p2.z - p1.z;
	float d2x = p4.x - p3.x, d2z = p4.z - p3.z;
	float denom = d1x * d2z - d1z * d2x;
	if (fabsf(denom) < 1e-9f)
		return false; /* parallel (or collinear); a coincident wall never occludes itself */
	float ex = p3.x - p1.x, ez = p3.z - p1.z;
	float t = (ex * d2z - ez * d2x) / denom;
	float u = (ex * d1z - ez * d1x) / denom;
	/* t excludes both endpoints so a wall carrying the light source (or the
	 * lit point sitting exactly on a wall) never shadows itself. */
	return t > 0.002f && t < 0.998f && u >= 0.0f && u <= 1.0f;
}

bool dungeon_light_segment_blocked(DungeonPoint from, DungeonPoint to,
								   const DungeonCollider *colliders, uint32_t collider_count)
{
	if (!colliders)
		return false;
	for (uint32_t i = 0; i < collider_count; ++i)
	{
		if (colliders[i].type != DUNGEON_COLLIDER_SEGMENT)
			continue;
		if (segments_intersect(from, to, colliders[i].segment.a, colliders[i].segment.b))
			return true;
	}
	return false;
}

float dungeon_light_attenuation(float distance_m, float radius_m)
{
	if (radius_m <= 0.0f || distance_m >= radius_m)
		return 0.0f;
	float normalized = distance_m / radius_m;
	float window = fmaxf(1.0f - normalized * normalized * normalized * normalized, 0.0f);
	return window * window / fmaxf(distance_m * distance_m, 0.01f);
}

uint32_t dungeon_lighting_build(const DungeonLevel *level, DungeonLight *out, uint32_t capacity)
{
	if (!level || !out || capacity < 2u)
		return 0;
	uint32_t count = 0;
	out[count++] = (DungeonLight){.position = level->spawn,
								.height = 1.65f,
								.radius = 6.5f,
								.color = {1.0f, 0.38f, 0.12f},
								.intensity = 24.0f};
	uint32_t available = capacity - 1u;
	uint32_t desired_interior = available > 5u ? 5u : available;

	/* Interior torches are spread evenly over open floor corners rather than
	 * one per rectangular surface run -- the cave frontend has no such runs. */
	size_t field_count = (size_t)level->field.width * level->field.height;
	DungeonPoint *open_points = field_count ? malloc(field_count * sizeof(*open_points)) : NULL;
	uint32_t open_count = 0;
	if (open_points)
		for (uint32_t z = 0; z < level->field.height; ++z)
			for (uint32_t x = 0; x < level->field.width; ++x)
				if (dungeon_field_get(&level->field, x, z) >= 0.5f)
					open_points[open_count++] = dungeon_field_corner_world(&level->field, x, z);

	for (uint32_t i = 0; i < desired_interior && open_count; ++i)
	{
		uint32_t index = ((i + 1u) * open_count) / (desired_interior + 1u);
		if (index >= open_count)
			index = open_count - 1u;
		out[count++] = (DungeonLight){
			.position = open_points[index],
			.height = 1.7f,
			.radius = 6.0f,
			.color = {1.0f, 0.46f, 0.18f},
			.intensity = 20.0f,
		};
	}
	free(open_points);

	out[count++] = (DungeonLight){.position = level->exit,
								.height = 1.5f,
								.radius = 7.5f,
								.color = {0.18f, 0.48f, 1.0f},
								.intensity = 34.0f};
	return count;
}
