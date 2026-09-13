#pragma once

#include "dungeon_level.h"

#include <stdint.h>

#define DUNGEON_MAX_LIGHTS 8u

typedef struct
{
	DungeonPoint position;
	float height;
	float radius;
	float color[3];
	/* Steady-state intensity. A torch's live output is this scaled by
	 * dungeon_light_flicker(); nothing should read it as the final value. */
	float intensity;
	/* Per-fixture offset into the flicker noise, in seconds. Spread so no two
	 * torches in a room breathe in step. */
	float phase;
	/* Multiplies the flame shader's own temperature ramp. NOT derivable from
	 * `color` above: that colour is orange BECAUSE the ramp is, and folding it
	 * in again would double-count the hue and take the white-hot core with it.
	 * An ordinary torch is neutral here; a fixture that burns in some other
	 * colour says so. */
	float flame_tint[3];
} DungeonLight;

/* One fixture's flame at one moment. `intensity_scale` multiplies the steady
 * intensity above, `warmth` runs -1 (guttering, deep and saturated) to +1
 * (flaring, washed toward white), and the sway is where the burning gas above
 * the head has been pushed to, in metres. */
typedef struct
{
	float intensity_scale;
	float warmth;
	float sway_x, sway_y;
} DungeonFlicker;

/* Deterministic in (phase, seconds): the same frame time always produces the
 * same flame, so a fixed-timestep harness run reproduces exactly and the CPU
 * light and the GPU flame agree on how bright this instant is. */
DungeonFlicker dungeon_light_flicker(float phase, float seconds);

/* Biases `color` by `warmth` about its own peak channel, so a warm torch goes
 * yellow-white when it flares and blood red when it gutters -- and a coloured
 * fixture (the exit's blue) shifts along its own hue rather than toward
 * orange. Writes three floats. */
void dungeon_light_warmth_color(const float color[3], float warmth, float *out_rgb);

uint32_t dungeon_lighting_build(const DungeonLevel *level, DungeonLight *out, uint32_t capacity);
float dungeon_light_attenuation(float distance_m, float radius_m);
bool dungeon_light_segment_blocked(DungeonPoint from, DungeonPoint to,
								   const DungeonCollider *colliders, uint32_t collider_count);
