#include "dungeon_lighting.h"

#include <math.h>
#include <stdlib.h>

static bool segments_intersect(DungeonPoint p1, DungeonPoint p2, DungeonPoint p3, DungeonPoint p4)
{
	float d1x = p2.x - p1.x, d1z = p2.z - p1.z;
	float d2x = p4.x - p3.x, d2z = p4.z - p3.z;
	float denom = d1x * d2z - d1z * d2x;
	if (fabsf(denom) < 1e-9f)
		return false; /* parallel (or collinear); a coincident wall never occludes itself */
	float ex = p3.x - p1.x, ez = p3.z - p1.z;
	float t = (ex * d2z - ez * d2x) / denom;
	float u = (ex * d1z - ez * d1x) / denom;
	/* t excludes both endpoints so a wall carrying the light source (or the
	 * lit point sitting exactly on a wall) never shadows itself. */
	return t > 0.002f && t < 0.998f && u >= 0.0f && u <= 1.0f;
}

bool dungeon_light_segment_blocked(DungeonPoint from, DungeonPoint to,
								   const DungeonCollider *colliders, uint32_t collider_count)
{
	if (!colliders)
		return false;
	for (uint32_t i = 0; i < collider_count; ++i)
	{
		if (colliders[i].type != DUNGEON_COLLIDER_SEGMENT)
			continue;
		if (segments_intersect(from, to, colliders[i].segment.a, colliders[i].segment.b))
			return true;
	}
	return false;
}

/* --- Flame flicker ------------------------------------------------------
 *
 * A torch is not a constant emitter. Its output is broadband noise: a slow
 * breathing drift as the fuel bed burns unevenly, the 5-10 Hz flutter that is
 * what actually reads as "fire" to the eye, a fast top octave that keeps the
 * flutter from looking like a sine, and the occasional deeper gutter when a
 * draught catches it. Amplitude falls with frequency, the way a real flame's
 * spectrum does; the bands are correlated noise rather than a fresh random
 * number per frame, which is what white-noise flicker looks wrong doing.
 *
 * This lives on the CPU rather than in the flame shader on purpose: the point
 * light and the drawn flame have to agree about how bright this instant is,
 * or the room brightens a frame out of step with the fire the player is
 * looking at. The shader gets the same number through a push constant. */

static float flicker_hash(float x)
{
	float s = sinf(x * 127.1f) * 43758.5453f;
	return s - floorf(s);
}

/* Smooth 1D value noise, range [-1, 1]. */
static float flicker_noise(float t)
{
	float cell = floorf(t);
	float f = t - cell;
	float smooth = f * f * (3.0f - 2.0f * f);
	float a = flicker_hash(cell), b = flicker_hash(cell + 1.0f);
	return (a + (b - a) * smooth) * 2.0f - 1.0f;
}

static float clampf(float value, float low, float high)
{
	return value < low ? low : (value > high ? high : value);
}

DungeonFlicker dungeon_light_flicker(float phase, float seconds)
{
	/* The phase multiplier is large and irrational-ish so two fixtures a
	 * tenth of a second apart in phase still land in unrelated parts of the
	 * noise rather than in the same wave a moment later. */
	float t = seconds + phase * 37.0f;
	float body = 0.11f * flicker_noise(t * 1.7f) +
				 0.14f * flicker_noise(t * 6.3f + 11.0f) +
				 0.06f * flicker_noise(t * 14.9f + 23.0f);
	/* Guttering. Thresholding a slow octave makes the dips rare; squaring the
	 * envelope keeps them from reading as a periodic pulse. */
	float draught = flicker_noise(t * 0.83f + 5.0f);
	float gutter = draught > 0.55f ? (draught - 0.55f) / 0.45f : 0.0f;
	DungeonFlicker out;
	out.intensity_scale = clampf(1.0f + body - 0.42f * gutter * gutter, 0.45f, 1.35f);
	out.warmth = clampf((out.intensity_scale - 1.0f) * 2.6f, -1.0f, 1.0f);
	/* Where the burning gas has been pushed to. Small -- a few centimetres --
	 * but enough that the lit patch on the wall behind is never quite still. */
	out.sway_x = 0.035f * flicker_noise(t * 2.6f + 3.0f);
	out.sway_y = 0.020f * flicker_noise(t * 3.4f + 17.0f);
	return out;
}

void dungeon_light_warmth_color(const float color[3], float warmth, float *out_rgb)
{
	if (!color || !out_rgb)
		return;
	float peak = fmaxf(fmaxf(color[0], color[1]), color[2]);
	/* Positive warmth pulls every channel toward the peak (desaturating
	 * toward white); negative pushes them away from it (deepening the hue).
	 * The asymmetry is deliberate: a flare washes out faster than a gutter
	 * saturates. */
	float amount = warmth >= 0.0f ? 0.30f * warmth : 0.45f * warmth;
	for (int channel = 0; channel < 3; ++channel)
		out_rgb[channel] =
			clampf(peak + (color[channel] - peak) * (1.0f - amount), 0.0f, 1.0f);
}

float dungeon_light_attenuation(float distance_m, float radius_m)
{
	if (radius_m <= 0.0f || distance_m >= radius_m)
		return 0.0f;
	float normalized = distance_m / radius_m;
	float window = fmaxf(1.0f - normalized * normalized * normalized * normalized, 0.0f);
	return window * window / fmaxf(distance_m * distance_m, 0.01f);
}

uint32_t dungeon_lighting_build(const DungeonLevel *level, DungeonLight *out, uint32_t capacity)
{
	if (!level || !out || capacity < 2u)
		return 0;
	uint32_t count = 0;
	out[count++] = (DungeonLight){.position = level->spawn,
								.height = 1.65f,
								.radius = 6.5f,
								.color = {1.0f, 0.38f, 0.12f},
								.intensity = 12.0f,
								.flame_tint = {1.0f, 1.0f, 1.0f}};
	uint32_t available = capacity - 1u;
	uint32_t desired_interior = available > 5u ? 5u : available;

	/* Interior torches are spread evenly over open floor corners rather than
	 * one per rectangular surface run -- the cave frontend has no such runs. */
	size_t field_count = (size_t)level->field.width * level->field.height;
	DungeonPoint *open_points = field_count ? malloc(field_count * sizeof(*open_points)) : NULL;
	uint32_t open_count = 0;
	if (open_points)
		for (uint32_t z = 0; z < level->field.height; ++z)
			for (uint32_t x = 0; x < level->field.width; ++x)
				if (dungeon_field_get(&level->field, x, z) >= 0.5f)
					open_points[open_count++] = dungeon_field_corner_world(&level->field, x, z);

	for (uint32_t i = 0; i < desired_interior && open_count; ++i)
	{
		uint32_t index = ((i + 1u) * open_count) / (desired_interior + 1u);
		if (index >= open_count)
			index = open_count - 1u;
		out[count++] = (DungeonLight){
			.position = open_points[index],
			.height = 1.7f,
			.radius = 6.0f,
			.color = {1.0f, 0.46f, 0.18f},
			.intensity = 10.0f,
			.flame_tint = {1.0f, 1.0f, 1.0f},
		};
	}
	free(open_points);

	out[count++] = (DungeonLight){.position = level->exit,
								.height = 1.5f,
								.radius = 7.5f,
								.color = {0.18f, 0.48f, 1.0f},
								/* The exit burns cold: same fire, other end of
								 * the spectrum, so it reads as a way out rather
								 * than as one more torch. */
								.intensity = 17.0f,
								.flame_tint = {0.30f, 0.62f, 1.0f}};
	/* Golden-ratio spacing: consecutive fixtures land far apart in the flicker
	 * noise, and so does every other pair, however many are placed. Two
	 * torches breathing in step across a room is the one thing that gives an
	 * animated flame away as a shader. */
	for (uint32_t i = 0; i < count; ++i)
		out[i].phase = (float)i * 0.61803399f;
	return count;
}
