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
		.camera = {.yaw = 90.0f, .pitch = -76.0f},
		.height = 14.0f,
		.trailing_distance = 3.5f,
		.follow_sharpness = 10.0f,
		.vertical_fov_degrees = 48.0f,
	};
	camera->camera.position = desired_position(camera, target);
}

/* Rotate the existing orbit offset immediately; following still smooths
 * target movement without letting rotation pull the cube off-centre. */
void dungeon_camera_orbit(DungeonCamera *camera, float yaw_input, float pitch_input, float dt)
{
	if (!camera || dt <= 0.0f || !isfinite(dt)) return;
	float old_yaw = glm_rad(camera->camera.yaw);
	double target_x = camera->camera.position.x + cosf(old_yaw) * camera->trailing_distance;
	double target_z = camera->camera.position.z + sinf(old_yaw) * camera->trailing_distance;
	camera->camera.yaw = fmodf(camera->camera.yaw + yaw_input * 85.0f * dt + 360.0f, 360.0f);
	if (pitch_input != 0.0f)
	{
		float radius = hypotf(camera->height, camera->trailing_distance);
		camera->camera.pitch = fminf(-35.0f, fmaxf(-82.0f, camera->camera.pitch + pitch_input * 45.0f * dt));
		camera->height = -sinf(glm_rad(camera->camera.pitch)) * radius;
		camera->trailing_distance = cosf(glm_rad(camera->camera.pitch)) * radius;
	}
	float yaw = glm_rad(camera->camera.yaw);
	camera->camera.position.x = target_x - cosf(yaw) * camera->trailing_distance;
	camera->camera.position.z = target_z - sinf(yaw) * camera->trailing_distance;
	camera->camera.position.y = camera->height;
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
