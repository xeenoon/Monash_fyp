#pragma once

#include <stdbool.h>
#include <stdint.h>

#include <cglm/struct.h>

#include "coordinate.h"

#define SHADOW_CASCADE_COUNT 4u
#define SHADOW_MAP_RESOLUTION 2048u

typedef struct
{
	float near_plane_m;
	float split_m[SHADOW_CASCADE_COUNT];
	float vertical_fov_radians;
	float aspect;
	uint32_t resolution;
} ShadowCascadeConfig;

typedef struct
{
	mat4s view_projection[SHADOW_CASCADE_COUNT];
	float radius_m[SHADOW_CASCADE_COUNT];
} ShadowCascadeSet;

ShadowCascadeConfig shadow_cascade_default_config(float aspect);

/* Builds world-grid-anchored, camera-relative cascade matrices. The bounding
   sphere and texel snapping follow Wicked Engine's CreateDirLightShadowCams,
   reduced to one directional light and a fixed four-cascade array. */
bool shadow_cascade_build(const ShadowCascadeConfig *config, WorldPosition camera_world,
						  vec3s camera_forward, vec3s camera_up, vec3s light_direction,
						  ShadowCascadeSet *out);
