#pragma once

#include <stdbool.h>
#include <cglm/struct.h>
#include "coordinate.h"

#define CAMERA_NEAR_PLANE 0.5f

typedef struct {
    WorldPosition position;
    float yaw, pitch;
} Camera;

vec3s camera_forward   (const Camera *cam);
/* Rotation-only view matrix. Absolute translation is applied per tile after a
   double-precision tile-origin minus camera-position subtraction. */
mat4s camera_view      (const Camera *cam);
/* Right-handed, Vulkan [0,1] infinite reversed-Z projection. */
mat4s camera_projection(const Camera *cam, float aspect);

/* Raw values only — the camera knows nothing about the input module. */
void camera_update(Camera *cam, float move_forward, float move_right,
                   float look_dx, float look_dy, bool sprint, float dt);
