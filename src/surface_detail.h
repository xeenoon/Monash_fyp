#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Generates one periodic linear RGBA texture for terrain surface treatment.
   RG stores a tangent-space detail normal, while B/A store two smooth noise
   octaves used for macro colour variation. Keeping this generated asset small
   avoids a per-tile material texture and makes its colour-space intent explicit. */
bool surface_detail_generate_rgba8(uint8_t *pixels, uint32_t width, uint32_t height);

/* Reduces a large double-precision world coordinate to a shader-friendly phase.
   `period_m` must be shared by every surface sampling scale. */
float surface_detail_phase(double world_m, double period_m);
