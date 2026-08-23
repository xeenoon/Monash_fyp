#include "surface_detail.h"

#include <math.h>
#include <stdbool.h>

#define TAU 6.28318530717958647692

static float periodic_height(double x, double y, double width, double height)
{
	double u = x / width;
	double v = y / height;
	return (float)(0.48 * sin(TAU * (7.0 * u + 5.0 * v)) + 0.31 * sin(TAU * (17.0 * u - 11.0 * v)) +
				   0.21 * cos(TAU * (29.0 * u + 13.0 * v)));
}

static uint8_t encode_unorm(float value)
{
	value = fmaxf(0.0f, fminf(1.0f, value));
	return (uint8_t)lroundf(value * 255.0f);
}

bool surface_detail_generate_rgba8(uint8_t *pixels, uint32_t width, uint32_t height)
{
	if (!pixels || width < 2u || height < 2u)
		return false;

	for (uint32_t y = 0; y < height; ++y)
	{
		for (uint32_t x = 0; x < width; ++x)
		{
			float left = periodic_height((double)x - 1.0, y, width, height);
			float right = periodic_height((double)x + 1.0, y, width, height);
			float down = periodic_height(x, (double)y - 1.0, width, height);
			float up = periodic_height(x, (double)y + 1.0, width, height);
			float nx = (left - right) * 1.8f;
			float ny = (down - up) * 1.8f;
			float nz = 1.0f / sqrtf(nx * nx + ny * ny + 1.0f);
			nx *= nz;
			ny *= nz;

			double u = (double)x / width;
			double v = (double)y / height;
			float macro_a =
				(float)(0.5 + 0.25 * sin(TAU * (u + 2.0 * v)) + 0.15 * cos(TAU * (3.0 * u - v)));
			float macro_b =
				(float)(0.5 + 0.27 * cos(TAU * (2.0 * u + v)) + 0.13 * sin(TAU * (u - 3.0 * v)));

			size_t i = ((size_t)y * width + x) * 4u;
			pixels[i + 0u] = encode_unorm(nx * 0.5f + 0.5f);
			pixels[i + 1u] = encode_unorm(ny * 0.5f + 0.5f);
			pixels[i + 2u] = encode_unorm(macro_a);
			pixels[i + 3u] = encode_unorm(macro_b);
		}
	}
	return true;
}

float surface_detail_phase(double world_m, double period_m)
{
	if (!isfinite(world_m) || !isfinite(period_m) || period_m <= 0.0)
		return 0.0f;
	double phase = fmod(world_m, period_m);
	if (phase < 0.0)
		phase += period_m;
	return (float)phase;
}
