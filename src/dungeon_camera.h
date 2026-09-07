#pragma once

#include "camera.h"
#include "dungeon_level.h"

typedef struct
{
	Camera camera;
	float height;
	float trailing_distance;
	float follow_sharpness;
	float vertical_fov_degrees;
} DungeonCamera;

void dungeon_camera_init(DungeonCamera *camera, DungeonPoint target);
void dungeon_camera_orbit(DungeonCamera *camera, float yaw_input, float pitch_input, float dt);
void dungeon_camera_update(DungeonCamera *camera, DungeonPoint target, float dt);
mat4s dungeon_camera_projection(const DungeonCamera *camera, float aspect);

