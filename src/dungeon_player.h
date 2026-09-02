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
bool dungeon_player_update(DungeonPlayer *player, const DungeonLevel *level, float move_forward,
						   float move_right, float camera_yaw_degrees, float dt);

