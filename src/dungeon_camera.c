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
	camera->explore_height = camera->height;
	camera->explore_trailing = camera->trailing_distance;
	camera->explore_fov = camera->vertical_fov_degrees;
	camera->explore_yaw = camera->camera.yaw;
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


/* The framing a lock is read at. The exploring camera looks almost straight
 * down, and from there a lock mounted on a door face is edge-on and invisible
 * -- so focusing drops the camera to about eye height and turns it nearly
 * horizontal, until the door face fills the frame. That is the whole reason
 * this is a framing change rather than just a zoom.
 *
 * It then IS also a zoom, because what has to be legible is millimetres. A pin
 * moves about 6 mm between one slot and the next -- that is the lock's own
 * authored travel, not a number chosen here -- and at the old framing (1.45 m
 * back, 42 degrees) a metre of frame spanned a thousand pixels, which put a
 * whole key press inside two of them. At 1.0 m and 38 degrees the frame spans
 * 0.69 m and the same press moves nine pixels. The height drops with it, so
 * the camera looks ACROSS the mechanism rather than down onto it: pin height is
 * the thing being read, and a steep angle foreshortens exactly that. */
#define DUNGEON_CAMERA_FOCUS_HEIGHT 1.20f
#define DUNGEON_CAMERA_FOCUS_TRAILING 1.00f
#define DUNGEON_CAMERA_FOCUS_FOV 38.0f
#define DUNGEON_CAMERA_FOCUS_YAW_LIMIT 24.0f
#define DUNGEON_CAMERA_FOCUS_PITCH_LIMIT 15.0f
#define DUNGEON_CAMERA_MOUSE_SENSITIVITY 0.04f

float dungeon_camera_focus_blend(const DungeonCamera *camera)
{
	if (!camera)
		return 0.0f;
	float t = camera->focus;
	return t * t * (3.0f - 2.0f * t);
}

/* Shortest signed way round from `from` to `to`, in degrees. Interpolating yaw
 * numerically would spin the long way whenever a turn crosses 0/360. */
static float shortest_yaw_delta(float from, float to)
{
	float delta = fmodf(to - from + 540.0f, 360.0f) - 180.0f;
	return delta;
}

void dungeon_camera_focus(DungeonCamera *camera, bool focusing, float facing_yaw_degrees,
						  float target_height_m, float dt)
{
	if (!camera || dt <= 0.0f || !isfinite(dt))
		return;
	if (!focusing && camera->focus <= 0.0f)
	{
		/* Settled and exploring: whatever the player has orbited to now IS the
		 * framing to come back to after the next lock. */
		camera->explore_height = camera->height;
		camera->explore_trailing = camera->trailing_distance;
		camera->explore_fov = camera->vertical_fov_degrees;
		camera->explore_yaw = camera->camera.yaw;
		return;
	}
	/* Brisk enough that the framing has settled before a player has read the
	 * lock -- a slower ease leaves the puzzle visibly off-centre while they
	 * are already pressing keys at it. */
	float alpha = 1.0f - expf(-11.0f * dt);
	camera->focus += ((focusing ? 1.0f : 0.0f) - camera->focus) * alpha;
	if (!focusing && camera->focus < 1e-4f)
	{
		/* Land exactly on the exploring framing instead of an epsilon short of
		 * it. Otherwise the residual becomes the new baseline on the next call
		 * and every lock picked leaves the camera slightly further adrift. */
		camera->focus = 0.0f;
		camera->height = camera->explore_height;
		camera->trailing_distance = camera->explore_trailing;
		camera->vertical_fov_degrees = camera->explore_fov;
		camera->camera.yaw = camera->explore_yaw;
		camera->focus_yaw_offset = 0.0f;
		camera->focus_pitch_offset = 0.0f;
		camera->camera.pitch =
			-glm_deg(atan2f(camera->height, fmaxf(camera->trailing_distance, 1e-3f)));
		return;
	}
	float t = dungeon_camera_focus_blend(camera);
	float wanted_yaw =
		focusing ? facing_yaw_degrees + camera->focus_yaw_offset : camera->explore_yaw;
	camera->camera.yaw =
		fmodf(camera->camera.yaw + shortest_yaw_delta(camera->camera.yaw, wanted_yaw) * alpha +
				  360.0f,
			  360.0f);
	camera->height = camera->explore_height +
					 (DUNGEON_CAMERA_FOCUS_HEIGHT - camera->explore_height) * t;
	camera->trailing_distance = camera->explore_trailing +
								(DUNGEON_CAMERA_FOCUS_TRAILING - camera->explore_trailing) * t;
	camera->vertical_fov_degrees =
		camera->explore_fov + (DUNGEON_CAMERA_FOCUS_FOV - camera->explore_fov) * t;
	/* Height, trailing distance and how far up the target sits define the look
	 * angle; the view matrix reads pitch. Leaving pitch behind would aim the
	 * camera past the door -- and ignoring the target height would aim it at
	 * the floor under a lock that is a metre up the leaf. */
	camera->camera.pitch =
		-glm_deg(atan2f(camera->height - target_height_m * t,
						 fmaxf(camera->trailing_distance, 1e-3f))) +
		camera->focus_pitch_offset * t;
}

void dungeon_camera_focus_look(DungeonCamera *camera, float look_dx, float look_dy)
{
	if (!camera || camera->focus <= 0.0f || !isfinite(look_dx) || !isfinite(look_dy))
		return;
	float old_yaw = camera->focus_yaw_offset;
	float old_pitch = camera->focus_pitch_offset;
	camera->focus_yaw_offset =
		fminf(DUNGEON_CAMERA_FOCUS_YAW_LIMIT,
			  fmaxf(-DUNGEON_CAMERA_FOCUS_YAW_LIMIT,
					camera->focus_yaw_offset + look_dx * DUNGEON_CAMERA_MOUSE_SENSITIVITY));
	camera->focus_pitch_offset =
		fminf(DUNGEON_CAMERA_FOCUS_PITCH_LIMIT,
			  fmaxf(-DUNGEON_CAMERA_FOCUS_PITCH_LIMIT,
					camera->focus_pitch_offset - look_dy * DUNGEON_CAMERA_MOUSE_SENSITIVITY));
	/* Apply immediately; dungeon_camera_focus uses the stored offsets on every
	 * later frame, so its centring ease no longer snaps this movement back. */
	camera->camera.yaw = fmodf(camera->camera.yaw + camera->focus_yaw_offset - old_yaw + 360.0f,
							 360.0f);
	camera->camera.pitch += camera->focus_pitch_offset - old_pitch;
}

/* How fast the keys sweep the inspection view. The offsets are clamped to
 * +/-24 degrees of yaw, so this crosses the whole range in about a second and a
 * half: fast enough to be worth pressing, slow enough that a tap is the small
 * adjustment it is meant to be. */
#define DUNGEON_CAMERA_FOCUS_PAN_DEGREES_PER_SECOND 30.0f

void dungeon_camera_focus_pan(DungeonCamera *camera, float pan_right, float pan_up, float dt)
{
	if (!camera || dt <= 0.0f || !isfinite(dt))
		return;
	if (pan_right == 0.0f && pan_up == 0.0f)
		return;
	/* Expressed as the mouse movement that would do the same thing, so there is
	 * one clamp, one sign convention and one place the limits live. */
	float pixels = DUNGEON_CAMERA_FOCUS_PAN_DEGREES_PER_SECOND * dt /
				   DUNGEON_CAMERA_MOUSE_SENSITIVITY;
	dungeon_camera_focus_look(camera, pan_right * pixels, -pan_up * pixels);
}

mat4s dungeon_camera_projection(const DungeonCamera *camera, float aspect)
{
	return camera_projection_fov(&camera->camera, aspect, camera->vertical_fov_degrees);
}
