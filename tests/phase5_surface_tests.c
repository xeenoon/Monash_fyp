#include "surface_detail.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static void generated_texture_is_deterministic_and_nonflat(void)
{
	enum
	{
		SIZE = 128,
		BYTES = SIZE * SIZE * 4
	};
	uint8_t *a = malloc(BYTES);
	uint8_t *b = malloc(BYTES);
	assert(a && b);
	assert(surface_detail_generate_rgba8(a, SIZE, SIZE));
	assert(surface_detail_generate_rgba8(b, SIZE, SIZE));

	uint32_t checksum_a = 2166136261u;
	uint32_t checksum_b = 2166136261u;
	uint8_t low[4] = {255, 255, 255, 255};
	uint8_t high[4] = {0, 0, 0, 0};
	for (size_t i = 0; i < BYTES; ++i)
	{
		checksum_a = (checksum_a ^ a[i]) * 16777619u;
		checksum_b = (checksum_b ^ b[i]) * 16777619u;
		unsigned channel = (unsigned)(i & 3u);
		if (a[i] < low[channel])
			low[channel] = a[i];
		if (a[i] > high[channel])
			high[channel] = a[i];
	}
	assert(checksum_a == checksum_b);
	for (unsigned channel = 0; channel < 4; ++channel)
		assert((unsigned)high[channel] - (unsigned)low[channel] > 24u);
	free(a);
	free(b);
}

static void large_coordinates_reduce_to_a_continuous_phase(void)
{
	const double period = 4096.0;
	const double origin = 2654321.125;
	float phase = surface_detail_phase(origin, period);
	assert(phase >= 0.0f && phase < (float)period);
	assert(fabsf(surface_detail_phase(origin + period, period) - phase) < 0.001f);
	assert(fabsf(surface_detail_phase(-1.25, period) - 4094.75f) < 0.001f);
	assert(surface_detail_phase(origin, 0.0) == 0.0f);
}

static void side_projection_phase_is_independent_of_tile_origin(void)
{
	const double period = 4096.0;
	const double camera_y = 2651.622096;
	const double surface_world_y = 2438.375;
	const double fine_tile_origin_y = 707.602;
	const double coarse_tile_origin_y = 512.0;
	float camera_phase = surface_detail_phase(camera_y, period);
	float fine_projected_y = (float)(surface_world_y - fine_tile_origin_y) +
		(float)(fine_tile_origin_y - camera_y) + camera_phase;
	float coarse_projected_y = (float)(surface_world_y - coarse_tile_origin_y) +
		(float)(coarse_tile_origin_y - camera_y) + camera_phase;
	assert(fabsf(fine_projected_y - coarse_projected_y) < 0.001f);
	assert(fabsf(surface_detail_phase(fine_projected_y, period) -
				 surface_detail_phase(surface_world_y, period)) < 0.001f);
}

int main(void)
{
	generated_texture_is_deterministic_and_nonflat();
	large_coordinates_reduce_to_a_continuous_phase();
	side_projection_phase_is_independent_of_tile_origin();
	return 0;
}
