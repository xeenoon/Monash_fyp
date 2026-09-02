#include "dungeon_camera.h"

#include <math.h>

static WorldPosition desired_position(const DungeonCamera *camera, DungeonPoint target)
{
	float yaw = glm_rad(camera->camera.yaw);
	return (WorldPosition){
		target.x - cosf(yaw) * camera->trailing_distance,
		camera->height,
		target.z - sinf(yaw) * camera->trailing_distance,
	};
}

void dungeon_camera_init(DungeonCamera *camera, DungeonPoint target)
{
	*camera = (DungeonCamera){
		.camera = {.yaw = 90.0f, .pitch = -58.0f},
		.height = 12.0f,
		.trailing_distance = 7.5f,
		.follow_sharpness = 10.0f,
		.vertical_fov_degrees = 48.0f,
	};
	camera->camera.position = desired_position(camera, target);
}

void dungeon_camera_update(DungeonCamera *camera, DungeonPoint target, float dt)
{
	if (!camera || dt <= 0.0f)
		return;
	WorldPosition desired = desired_position(camera, target);
	float alpha = 1.0f - expf(-camera->follow_sharpness * dt);
	camera->camera.position.x += (desired.x - camera->camera.position.x) * alpha;
	camera->camera.position.y += (desired.y - camera->camera.position.y) * alpha;
	camera->camera.position.z += (desired.z - camera->camera.position.z) * alpha;
}

mat4s dungeon_camera_projection(const DungeonCamera *camera, float aspect)
{
	return camera_projection_fov(&camera->camera, aspect, camera->vertical_fov_degrees);
}

