#pragma once

#include "dungeon_level.h"

#include <stdint.h>

#define DUNGEON_MAX_LIGHTS 8u

typedef struct
{
	DungeonPoint position;
	float height;
	float radius;
	float color[3];
	float intensity;
} DungeonLight;

uint32_t dungeon_lighting_build(const DungeonLevel *level, DungeonLight *out, uint32_t capacity);
float dungeon_light_attenuation(float distance_m, float radius_m);
bool dungeon_light_segment_blocked(DungeonPoint from, DungeonPoint to,
								   const DungeonCollider *colliders, uint32_t collider_count);
