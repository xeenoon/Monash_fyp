#include "dungeon_player.h"

#include "dungeon_collision.h"

#include <cglm/struct.h>
#include <math.h>

void dungeon_player_init(DungeonPlayer *player, DungeonPoint spawn)
{
	*player = (DungeonPlayer){.position = spawn, .radius = 0.35f, .speed = 3.5f};
}

bool dungeon_player_update(DungeonPlayer *player, const DungeonLevel *level, float move_forward,
						   float move_right, float camera_yaw_degrees, float dt)
{
	if (!player || !level || dt <= 0.0f)
		return false;
	float yaw = glm_rad(camera_yaw_degrees);
	DungeonPoint forward = {cosf(yaw), sinf(yaw)};
	DungeonPoint right = {-forward.z, forward.x};
	DungeonPoint direction = {forward.x * move_forward + right.x * move_right,
							  forward.z * move_forward + right.z * move_right};
	float length = sqrtf(direction.x * direction.x + direction.z * direction.z);
	if (length > 1.0f)
	{
		direction.x /= length;
		direction.z /= length;
	}
	DungeonPoint displacement = {direction.x * player->speed * dt,
								 direction.z * player->speed * dt};
	player->position = dungeon_collision_move(level->colliders, level->collider_count,
											player->position, displacement, player->radius);
	float exit_dx = player->position.x - level->exit.x;
	float exit_dz = player->position.z - level->exit.z;
	bool was_reached = player->reached_exit;
	player->reached_exit = exit_dx * exit_dx + exit_dz * exit_dz < 0.55f * 0.55f;
	return player->reached_exit && !was_reached;
}

