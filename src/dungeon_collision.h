#pragma once

#include "dungeon_geometry.h"

#include <stdint.h>

/* Swept-circle-vs-segment collision for the player. Displacement is
 * substepped to at most radius/2 per step so a large frame delta can't tunnel
 * through a thin wall, and each substep resolves against every collider with
 * a few relaxation iterations so corners (including acute ones) settle
 * instead of jittering. */
DungeonPoint dungeon_collision_move(const DungeonCollider *colliders, uint32_t collider_count,
									DungeonPoint start, DungeonPoint displacement, float radius);
