#pragma once

#include <stdbool.h>
#include <cglm/struct.h>

typedef struct { vec3s position; float yaw, pitch; } Camera;

vec3s camera_forward   (const Camera *cam);
mat4s camera_view      (const Camera *cam);
mat4s camera_projection(const Camera *cam, float aspect);

/* Raw values only — the camera knows nothing about the input module. */
void camera_update(Camera *cam, float move_forward, float move_right,
                   float look_dx, float look_dy, bool sprint, float dt);
