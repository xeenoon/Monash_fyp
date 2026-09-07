#include "dungeon_camera.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

int main(void)
{
	DungeonCamera camera;
	dungeon_camera_init(&camera, (DungeonPoint){2.0f, 3.0f});
	assert(fabs(camera.camera.position.x - 2.0) < 1e-6);
	assert(fabs(camera.camera.position.y - 14.0) < 1e-6);
	assert(fabs(camera.camera.position.z - -0.5) < 1e-6);
	assert(camera.camera.pitch < -70.0f && camera.camera.pitch > -85.0f);
	double before = camera.camera.position.x;
	dungeon_camera_update(&camera, (DungeonPoint){12.0f, 3.0f}, 1.0f / 60.0f);
	assert(camera.camera.position.x > before);
	assert(camera.camera.position.x < 12.0);
	for (int i = 0; i < 600; ++i)
		dungeon_camera_update(&camera, (DungeonPoint){12.0f, 3.0f}, 1.0f / 60.0f);
	assert(fabs(camera.camera.position.x - 12.0) < 1e-4);
	/* Arrow orbit preserves radius and keeps the follow target centred. */
	float radius = hypotf(camera.height, camera.trailing_distance);
	double target_x = camera.camera.position.x + cosf(glm_rad(camera.camera.yaw)) * camera.trailing_distance;
	double target_z = camera.camera.position.z + sinf(glm_rad(camera.camera.yaw)) * camera.trailing_distance;
	dungeon_camera_orbit(&camera, 1.0f, 1.0f, 0.5f);
	assert(fabsf(camera.camera.yaw - 132.5f) < 1e-4f);
	assert(fabsf(hypotf(camera.height, camera.trailing_distance) - radius) < 1e-4f);
	assert(fabs(camera.camera.position.x + cosf(glm_rad(camera.camera.yaw)) * camera.trailing_distance - target_x) < 1e-4);
	assert(fabs(camera.camera.position.z + sinf(glm_rad(camera.camera.yaw)) * camera.trailing_distance - target_z) < 1e-4);
	dungeon_camera_orbit(&camera, 0, 1, 10);
	assert(camera.camera.pitch == -35.0f);
	dungeon_camera_orbit(&camera, 0, -1, 10);
	assert(camera.camera.pitch == -82.0f);
	float yaw = camera.camera.yaw;
	dungeon_camera_orbit(&camera, 1, 1, 0);
	assert(camera.camera.yaw == yaw);
	mat4s projection = dungeon_camera_projection(&camera, 16.0f / 9.0f);
	assert(isfinite(projection.raw[0][0]));
	assert(projection.raw[2][3] == -1.0f);
	puts("dungeon camera tests passed");
	return 0;
}
