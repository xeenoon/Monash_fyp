#include "dungeon_grid.h"
#include "dungeon_lighting.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

/* The torch flicker is what the point light and the flame billboard BOTH read
 * to decide how bright this instant is, so it is worth pinning numerically:
 * an animation is exactly the kind of thing that gets "checked" by looking at
 * it once and never again. */
static void flicker_tests(void)
{
	/* Deterministic in (phase, seconds). The fixed-timestep harness and the
	 * agreement between the CPU light and the GPU flame both rest on this. */
	for (int step = 0; step < 64; ++step)
	{
		float seconds = (float)step * 0.137f;
		DungeonFlicker a = dungeon_light_flicker(0.618f, seconds);
		DungeonFlicker b = dungeon_light_flicker(0.618f, seconds);
		assert(a.intensity_scale == b.intensity_scale);
		assert(a.warmth == b.warmth);
		assert(a.sway_x == b.sway_x && a.sway_y == b.sway_y);
	}

	/* Bounded, and actually moving. A flame pinned near 1.0 is a static light
	 * with extra arithmetic; one that leaves these bounds either blows the
	 * exposure out or drops the room to black. */
	float minimum = 2.0f, maximum = 0.0f, sum = 0.0f, sum_squares = 0.0f;
	float sway_peak = 0.0f;
	const int samples = 4000;
	for (int step = 0; step < samples; ++step)
	{
		float seconds = (float)step / 60.0f; /* a bit over a minute at 60 Hz */
		DungeonFlicker flicker = dungeon_light_flicker(1.236f, seconds);
		assert(flicker.intensity_scale >= 0.45f && flicker.intensity_scale <= 1.35f);
		assert(flicker.warmth >= -1.0f && flicker.warmth <= 1.0f);
		minimum = fminf(minimum, flicker.intensity_scale);
		maximum = fmaxf(maximum, flicker.intensity_scale);
		sum += flicker.intensity_scale;
		sum_squares += flicker.intensity_scale * flicker.intensity_scale;
		sway_peak = fmaxf(sway_peak, fmaxf(fabsf(flicker.sway_x), fabsf(flicker.sway_y)));
	}
	float mean = sum / (float)samples;
	float deviation = sqrtf(sum_squares / (float)samples - mean * mean);
	assert(minimum < 0.80f);			  /* it gutters */
	assert(maximum > 1.15f);			  /* it flares */
	assert(deviation > 0.05f);			  /* and does so continually, not once */
	assert(mean > 0.85f && mean < 1.05f); /* without moving the room's average */
	assert(sway_peak > 0.005f && sway_peak <= 0.036f); /* centimetres, not a shake */

	/* Consecutive frames must be CORRELATED -- flicker is low-frequency noise,
	 * not a fresh random number per frame. White noise at 60 Hz reads as a
	 * strobing bug, and is the usual way this effect is got wrong. */
	float frame_step_sum = 0.0f;
	for (int step = 1; step < samples; ++step)
	{
		float a = dungeon_light_flicker(1.236f, (float)(step - 1) / 60.0f).intensity_scale;
		float b = dungeon_light_flicker(1.236f, (float)step / 60.0f).intensity_scale;
		frame_step_sum += fabsf(b - a);
	}
	float mean_frame_step = frame_step_sum / (float)(samples - 1);
	assert(mean_frame_step < deviation * 0.5f);

	/* Warmth tracks brightness: a flare washes toward white, a gutter deepens.
	 * The flame shader assumes this sign convention. */
	for (int step = 0; step < 500; ++step)
	{
		DungeonFlicker flicker = dungeon_light_flicker(0.0f, (float)step / 30.0f);
		assert((flicker.intensity_scale > 1.0f) == (flicker.warmth > 0.0f) ||
			   flicker.intensity_scale == 1.0f);
	}

	/* Two fixtures must not breathe in step -- that is the single tell that
	 * gives an animated flame away as one shader driving every torch. Compare
	 * the phases dungeon_lighting_build actually hands out. */
	float mean_a = 0.0f, mean_b = 0.0f;
	float series_a[600], series_b[600];
	for (int step = 0; step < 600; ++step)
	{
		float seconds = (float)step / 60.0f;
		series_a[step] = dungeon_light_flicker(0.0f, seconds).intensity_scale;
		series_b[step] = dungeon_light_flicker(0.61803399f, seconds).intensity_scale;
		mean_a += series_a[step] / 600.0f;
		mean_b += series_b[step] / 600.0f;
	}
	float covariance = 0.0f, variance_a = 0.0f, variance_b = 0.0f;
	for (int step = 0; step < 600; ++step)
	{
		float da = series_a[step] - mean_a, db = series_b[step] - mean_b;
		covariance += da * db;
		variance_a += da * da;
		variance_b += db * db;
	}
	float correlation = covariance / sqrtf(variance_a * variance_b);
	assert(fabsf(correlation) < 0.25f);
}

static void warmth_color_tests(void)
{
	const float torch[3] = {1.0f, 0.38f, 0.12f};
	float hot[3], cold[3], neutral[3];
	dungeon_light_warmth_color(torch, 1.0f, hot);
	dungeon_light_warmth_color(torch, -1.0f, cold);
	dungeon_light_warmth_color(torch, 0.0f, neutral);
	/* Zero warmth is the fixture's own colour, untouched. */
	for (int channel = 0; channel < 3; ++channel)
		assert(fabsf(neutral[channel] - torch[channel]) < 1e-5f);
	/* Flaring desaturates toward the peak channel; guttering deepens away
	 * from it. The peak itself never moves, so the light keeps its hue. */
	assert(hot[0] == 1.0f && cold[0] == 1.0f);
	assert(hot[1] > torch[1] && hot[2] > torch[2]);
	assert(cold[1] < torch[1] && cold[2] < torch[2]);
	assert(cold[1] >= 0.0f && cold[2] >= 0.0f);

	/* A coloured fixture shifts along its OWN hue rather than toward orange:
	 * the exit's blue stays blue-dominant at both extremes. */
	const float exit_light[3] = {0.18f, 0.48f, 1.0f};
	float exit_hot[3], exit_cold[3];
	dungeon_light_warmth_color(exit_light, 1.0f, exit_hot);
	dungeon_light_warmth_color(exit_light, -1.0f, exit_cold);
	assert(exit_hot[2] >= exit_hot[1] && exit_hot[1] >= exit_hot[0]);
	assert(exit_cold[2] >= exit_cold[1] && exit_cold[1] >= exit_cold[0]);
	for (int channel = 0; channel < 3; ++channel)
	{
		assert(exit_hot[channel] >= 0.0f && exit_hot[channel] <= 1.0f);
		assert(exit_cold[channel] >= 0.0f && exit_cold[channel] <= 1.0f);
	}
}

int main(void)
{
	assert(dungeon_light_attenuation(0.0f, 5.0f) >
		   dungeon_light_attenuation(1.0f, 5.0f));
	assert(dungeon_light_attenuation(5.0f, 5.0f) == 0.0f);
	assert(dungeon_light_attenuation(6.0f, 5.0f) == 0.0f);
	DungeonCollider blocker = {
		.type = DUNGEON_COLLIDER_SEGMENT, .segment = {{1.0f, -1.0f}, {1.0f, 1.0f}}};
	assert(dungeon_light_segment_blocked((DungeonPoint){0.0f, 0.0f},
									   (DungeonPoint){3.0f, 0.0f}, &blocker, 1));
	assert(!dungeon_light_segment_blocked((DungeonPoint){0.0f, 2.0f},
										(DungeonPoint){3.0f, 2.0f}, &blocker, 1));
	assert(!dungeon_light_segment_blocked((DungeonPoint){0.0f, 0.0f},
										(DungeonPoint){1.0f, 0.0f}, &blocker, 1));
	DungeonLevel level = {0};
	DungeonLevelError error = {0};
	assert(dungeon_grid_compile_text("#####\n#S.E#\n#####\n", 2.0f, &level, &error));
	DungeonLight lights[DUNGEON_MAX_LIGHTS];
	uint32_t count = dungeon_lighting_build(&level, lights, DUNGEON_MAX_LIGHTS);
	assert(count >= 2u && count <= DUNGEON_MAX_LIGHTS);
	assert(lights[0].position.x == level.spawn.x);
	assert(lights[count - 1].position.x == level.exit.x);
	/* Every fixture gets its own place in the flicker noise, and the exit
	 * burns in its own colour rather than as one more orange torch. */
	for (uint32_t i = 0; i < count; ++i)
		for (uint32_t j = i + 1u; j < count; ++j)
			assert(lights[i].phase != lights[j].phase);
	assert(lights[count - 1].flame_tint[2] > lights[count - 1].flame_tint[0]);
	assert(lights[0].flame_tint[0] == 1.0f && lights[0].flame_tint[1] == 1.0f &&
		   lights[0].flame_tint[2] == 1.0f);
	dungeon_level_destroy(&level);
	flicker_tests();
	warmth_color_tests();
	puts("dungeon lighting tests passed");
	return 0;
}
