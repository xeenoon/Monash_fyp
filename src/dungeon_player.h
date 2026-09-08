#pragma once

#include "dungeon_level.h"

typedef struct
{
	DungeonPoint position;
	float radius;
	float speed;
	bool reached_exit;
} DungeonPlayer;

void dungeon_player_init(DungeonPlayer *player, DungeonPoint spawn);

/* Colliders are passed in rather than read off the level because doors are
 * dynamic: the session's array is the level's plus one segment per door still
 * shut, and it changes as locks are picked. Pass level->colliders directly for
 * a level with no doors in it. */
bool dungeon_player_update(DungeonPlayer *player, const DungeonLevel *level,
						   const DungeonCollider *colliders, uint32_t collider_count,
						   float move_forward, float move_right, float camera_yaw_degrees,
						   float dt);

