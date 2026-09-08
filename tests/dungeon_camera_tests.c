#include "dungeon_camera.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>


/* Picking a lock eases the camera in close and then all the way back out. The
 * two things that must hold: pitch stays consistent with height and trailing
 * distance (otherwise the view stops pointing at what the follow code frames),
 * and whatever the player had orbited to is what they get back. */
static void focusing_on_a_lock_returns_to_the_exploring_framing(void)
{
	DungeonCamera camera = {0};
	dungeon_camera_init(&camera, (DungeonPoint){0.0f, 0.0f});
	dungeon_camera_orbit(&camera, 0.4f, 0.3f, 0.5f);
	dungeon_camera_focus(&camera, false, 0.0f, 0.0f, 1.0f / 60.0f); /* captures the baseline */
	float explore_height = camera.height;
	float explore_trailing = camera.trailing_distance;
	float explore_fov = camera.vertical_fov_degrees;
	float explore_yaw = camera.camera.yaw;
	/* A facing deliberately on the far side of the 0/360 wrap from where the
	 * camera starts, so the shortest-path turn is exercised rather than a
	 * numeric interpolation that would spin the long way. */
	float facing = fmodf(explore_yaw + 200.0f, 360.0f);
	const float lock_height = 0.92f; /* the lock sits up the door, not on the floor */
	assert(dungeon_camera_focus_blend(&camera) == 0.0f);

	for (int frame = 0; frame < 600; ++frame)
		dungeon_camera_focus(&camera, true, facing, lock_height, 1.0f / 60.0f);
	assert(dungeon_camera_focus_blend(&camera) > 0.999f);
	assert(camera.height < explore_height);			  /* closer */
	assert(camera.trailing_distance < explore_trailing);
	assert(camera.vertical_fov_degrees < explore_fov); /* and tighter */
	/* Pitch aims at the lock's height on the door, not at the floor beneath it,
	 * and the framing is near-horizontal rather than the top-down explore view. */
	float expected_pitch =
		-glm_deg(atan2f(camera.height - lock_height, camera.trailing_distance));
	assert(fabsf(camera.camera.pitch - expected_pitch) < 1e-3f);
	assert(camera.camera.pitch > -45.0f);
	/* And it came round to face the lock, the short way. */
	assert(fabsf(fmodf(camera.camera.yaw - facing + 540.0f, 360.0f) - 180.0f) < 1e-2f);

	for (int frame = 0; frame < 600; ++frame)
		dungeon_camera_focus(&camera, false, facing, lock_height, 1.0f / 60.0f);
	/* Exactly back, not an epsilon short: a residual would become the next
	 * baseline and every lock picked would leave the camera further adrift. */
	assert(dungeon_camera_focus_blend(&camera) == 0.0f);
	assert(camera.height == explore_height);
	assert(camera.trailing_distance == explore_trailing);
	assert(camera.vertical_fov_degrees == explore_fov);
	assert(camera.camera.yaw == explore_yaw);
}

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
	focusing_on_a_lock_returns_to_the_exploring_framing();
	puts("dungeon camera tests passed");
	return 0;
}
