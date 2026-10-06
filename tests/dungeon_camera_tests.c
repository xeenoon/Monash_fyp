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

	/* Mouse-look is retained instead of the focus ease snapping it back. */
	dungeon_camera_focus_look(&camera, 250.0f, -125.0f);
	float adjusted_yaw = camera.camera.yaw;
	float adjusted_pitch = camera.camera.pitch;
	for (int frame = 0; frame < 120; ++frame)
		dungeon_camera_focus(&camera, true, facing, lock_height, 1.0f / 60.0f);
	assert(fabsf(camera.camera.yaw - adjusted_yaw) < 1e-2f);
	assert(fabsf(camera.camera.pitch - adjusted_pitch) < 1e-2f);
	/* Large deltas are bounded to a minor inspection adjustment. */
	dungeon_camera_focus_look(&camera, 100000.0f, 100000.0f);
	assert(fabsf(camera.focus_yaw_offset) <= 24.0f);
	assert(fabsf(camera.focus_pitch_offset) <= 15.0f);

	for (int frame = 0; frame < 600; ++frame)
		dungeon_camera_focus(&camera, false, facing, lock_height, 1.0f / 60.0f);
	/* Exactly back, not an epsilon short: a residual would become the next
	 * baseline and every lock picked would leave the camera further adrift. */
	assert(dungeon_camera_focus_blend(&camera) == 0.0f);
	assert(camera.height == explore_height);
	assert(camera.trailing_distance == explore_trailing);
	assert(camera.vertical_fov_degrees == explore_fov);
	assert(camera.camera.yaw == explore_yaw);
	assert(camera.focus_yaw_offset == 0.0f);
	assert(camera.focus_pitch_offset == 0.0f);
}

/* The movement keys nudge the inspection view while a lock is up, through the
 * same clamped offsets the mouse writes -- so the two cannot fight each other
 * and neither can leave the framing. A held key has no magnitude of its own, so
 * the pan is a rate per second; what this checks is that the conversion lands
 * in the same place, facing the same way, and stops at the same limits. */
static void the_movement_keys_pan_the_same_bounded_view(void)
{
	DungeonCamera camera = {0};
	dungeon_camera_init(&camera, (DungeonPoint){0.0f, 0.0f});
	for (int frame = 0; frame < 600; ++frame)
		dungeon_camera_focus(&camera, true, 0.0f, 0.92f, 1.0f / 60.0f);
	assert(dungeon_camera_focus_blend(&camera) > 0.999f);

	/* Right and up mean what they mean for the mouse. 0.4 s at the pan rate. */
	dungeon_camera_focus_pan(&camera, 1.0f, 1.0f, 0.4f);
	assert(fabsf(camera.focus_yaw_offset - 12.0f) < 1e-3f);
	assert(fabsf(camera.focus_pitch_offset - 12.0f) < 1e-3f);

	/* Back the same distance lands on centre, not near it: a residual left in
	 * the offsets is a view that drifts a little further with every lock. */
	dungeon_camera_focus_pan(&camera, -1.0f, -1.0f, 0.4f);
	assert(fabsf(camera.focus_yaw_offset) < 1e-3f);
	assert(fabsf(camera.focus_pitch_offset) < 1e-3f);

	/* Held to the stops it gets the mouse's limits and not a degree more. */
	for (int frame = 0; frame < 600; ++frame)
		dungeon_camera_focus_pan(&camera, 1.0f, 1.0f, 1.0f / 60.0f);
	assert(fabsf(camera.focus_yaw_offset - 24.0f) < 1e-3f);
	assert(fabsf(camera.focus_pitch_offset - 15.0f) < 1e-3f);

	/* No key, or no elapsed time, moves nothing. */
	float held = camera.focus_yaw_offset;
	dungeon_camera_focus_pan(&camera, 1.0f, 1.0f, 0.0f);
	dungeon_camera_focus_pan(&camera, 0.0f, 0.0f, 1.0f);
	assert(camera.focus_yaw_offset == held);
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
	the_movement_keys_pan_the_same_bounded_view();
	puts("dungeon camera tests passed");
	return 0;
}
