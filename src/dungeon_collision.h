#pragma once

#include "dungeon_level.h"

#include <stdint.h>

DungeonPoint dungeon_collision_move(const DungeonCollider *colliders, uint32_t collider_count,
									DungeonPoint start, DungeonPoint displacement, float radius);

