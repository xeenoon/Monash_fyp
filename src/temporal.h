#pragma once

#include <stdbool.h>
#include <stdint.h>

#include <cglm/struct.h>
#include "coordinate.h"

/* Phase-8 temporal helpers kept independent from Vulkan so the jitter and
   exposure policy can be tested without a GPU. Jitter is returned in NDC. */
float temporal_halton(uint32_t index, uint32_t base);
vec2s temporal_jitter_ndc(uint64_t frame_index, uint32_t width, uint32_t height);
mat4s temporal_jitter_projection(mat4s projection, vec2s jitter_ndc);

bool temporal_camera_cut(WorldPosition previous, WorldPosition current,
                         float previous_yaw, float current_yaw,
                         float previous_pitch, float current_pitch,
                         double teleport_distance_m);

float temporal_exposure_target(float average_luminance, float middle_grey,
                               float minimum_exposure, float maximum_exposure);
float temporal_adapt_exposure(float current_exposure, float target_exposure,
                              float delta_seconds, float brighten_speed,
                              float darken_speed);
