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
	mat4s projection = dungeon_camera_projection(&camera, 16.0f / 9.0f);
	assert(isfinite(projection.raw[0][0]));
	assert(projection.raw[2][3] == -1.0f);
	puts("dungeon camera tests passed");
	return 0;
}
