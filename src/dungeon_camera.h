#pragma once

#include "camera.h"
#include "dungeon_level.h"

#include <stdbool.h>

typedef struct
{
	Camera camera;
	float height;
	float trailing_distance;
	float follow_sharpness;
	float vertical_fov_degrees;
	/* Lock-picking framing. `focus` eases 0 -> 1 as the player enters a lock;
	 * the explore_* fields remember the framing to return to, so orbiting
	 * before a lock is not undone by picking one. */
	float focus;
	float explore_height, explore_trailing, explore_fov, explore_yaw;
} DungeonCamera;

void dungeon_camera_init(DungeonCamera *camera, DungeonPoint target);
void dungeon_camera_orbit(DungeonCamera *camera, float yaw_input, float pitch_input, float dt);
void dungeon_camera_update(DungeonCamera *camera, DungeonPoint target, float dt);

/* Eases the camera between the exploring framing and a close, shallower one
 * suited to reading a lock: height, trailing distance, pitch, FOV -- and yaw,
 * so the camera comes round to actually FACE the lock instead of reading it
 * from whatever angle exploring happened to leave. `facing_yaw_degrees` is the
 * direction to look along, from dungeon_session_focus_facing_degrees; it is
 * ignored while not focusing. Pitch is kept consistent with height/trailing,
 * otherwise the view stops pointing at what the follow code is framing. */
void dungeon_camera_focus(DungeonCamera *camera, bool focusing, float facing_yaw_degrees,
						  float target_height_m, float dt);

/* The eased 0..1 focus weight, smoothstepped -- use it to blend the follow
 * target from the player toward the lock. */
float dungeon_camera_focus_blend(const DungeonCamera *camera);
mat4s dungeon_camera_projection(const DungeonCamera *camera, float aspect);

