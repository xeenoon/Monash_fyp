#pragma once

#include "coordinate.h"
#include <cglm/struct.h>
#include <stdbool.h>

#define CAMERA_NEAR_PLANE 0.5f

typedef struct
{
	WorldPosition position;
	float yaw, pitch;
	/* Seconds shift has been held while moving; ramps sprint speed. Reset on
	   release. Zero-initialized by the designated initializers in main. */
	float sprint_charge;
	/* Multiplies every movement speed below. The defaults are sized for a
	   16 km terrain map, where 60 m/s is a slow pan; in an eight-metre room
	   the same speed crosses the whole space in a tenth of a second. Zero or
	   negative reads as 1.0, so every existing zero-initialized Camera keeps
	   the terrain speeds. */
	float speed_scale;
} Camera;

vec3s camera_forward(const Camera *cam);
/* Rotation-only view matrix. Absolute translation is applied per tile after a
   double-precision tile-origin minus camera-position subtraction. */
mat4s camera_view(const Camera *cam);
/* Right-handed, Vulkan [0,1] infinite reversed-Z projection. */
mat4s camera_projection(const Camera *cam, float aspect);
mat4s camera_projection_fov(const Camera *cam, float aspect, float vertical_fov_degrees);

/* Raw values only — the camera knows nothing about the input module. */
void camera_update(Camera *cam, float move_forward, float move_right, float look_dx, float look_dy,
				   bool sprint, float dt);
