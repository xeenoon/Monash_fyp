#include "dungeon_lighting.h"

#include <math.h>

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
	for (uint32_t i = 0; i < desired_interior && level->surface_count; ++i)
	{
		uint32_t surface_index = ((i + 1u) * level->surface_count) / (desired_interior + 1u);
		DungeonRect footprint = level->surfaces[surface_index].footprint;
		out[count++] = (DungeonLight){
			.position = {(footprint.min.x + footprint.max.x) * 0.5f,
						 (footprint.min.z + footprint.max.z) * 0.5f},
			.height = 1.7f,
			.radius = 6.0f,
			.color = {1.0f, 0.46f, 0.18f},
			.intensity = 20.0f,
		};
	}
	out[count++] = (DungeonLight){.position = level->exit,
								.height = 1.5f,
								.radius = 7.5f,
								.color = {0.18f, 0.48f, 1.0f},
								.intensity = 34.0f};
	return count;
}

