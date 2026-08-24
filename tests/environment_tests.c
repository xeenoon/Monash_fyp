#include "environment.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>

/* Mirrors the GLSL environment_irradiance(vec3 N) reconstruction in
   shaders/common.glsl -- same normalized SH basis (matching
   environment_project_sh9's projection basis) and coefficient order, so this
   test exercises exactly what the shader will compute. */
static void reconstruct(const EnvironmentSH *sh, double x, double y, double z, double out[3])
{
	double basis[9] = {
		0.282095,
		0.488603 * y,
		0.488603 * z,
		0.488603 * x,
		1.092548 * x * y,
		1.092548 * y * z,
		0.315392 * (3.0 * z * z - 1.0),
		1.092548 * x * z,
		0.546274 * (x * x - y * y),
	};
	for (int c = 0; c < 3; ++c)
	{
		double v = 0.0;
		for (int t = 0; t < 9; ++t)
			v += sh->coeffs[t][c] * basis[t];
		out[c] = v;
	}
}

static void uniform_environment_yields_flat_irradiance(void)
{
	const int width = 32, height = 16;
	const float value = 0.6f;
	float *pixels = malloc((size_t)width * (size_t)height * 4 * sizeof(float));
	for (int i = 0; i < width * height * 4; ++i)
		pixels[i] = value;

	EnvironmentSH sh;
	environment_project_sh9(pixels, width, height, &sh);
	free(pixels);

	/* A handful of directions spanning the sphere; every one should reconstruct
	   to ~value, since a spatially-constant environment has no directionality. */
	const double dirs[6][3] = {
		{0, 1, 0}, {0, -1, 0}, {1, 0, 0}, {-1, 0, 0}, {0, 0, 1}, {0, 0, -1},
	};
	for (int i = 0; i < 6; ++i)
	{
		double out[3];
		reconstruct(&sh, dirs[i][0], dirs[i][1], dirs[i][2], out);
		for (int c = 0; c < 3; ++c)
			assert(fabs(out[c] - value) < 0.01);
	}
}

static void bright_direction_peaks_reconstruction_along_it(void)
{
	const int width = 128, height = 64;
	float *pixels = calloc((size_t)width * (size_t)height * 4, sizeof(float));

	/* A small bright patch centred where the projection's own convention
	   places +Z (u=0.25 -> phi=pi/2, v=0.5 -> theta=pi/2): x=sin(theta)cos(phi)=0,
	   y=cos(theta)=0, z=sin(theta)sin(phi)=1. */
	int cx = width / 4, cy = height / 2;
	for (int dy = -1; dy <= 1; ++dy)
		for (int dx = -1; dx <= 1; ++dx)
		{
			int x = cx + dx, y = cy + dy;
			size_t idx = ((size_t)y * (size_t)width + (size_t)x) * 4;
			pixels[idx + 0] = pixels[idx + 1] = pixels[idx + 2] = 500.0f;
			pixels[idx + 3] = 1.0f;
		}

	EnvironmentSH sh;
	environment_project_sh9(pixels, width, height, &sh);
	free(pixels);

	double along[3], opposite[3], orthogonal[3];
	reconstruct(&sh, 0.0, 0.0, 1.0, along);	   /* +Z: the bright direction */
	reconstruct(&sh, 0.0, 0.0, -1.0, opposite);  /* -Z: opposite */
	reconstruct(&sh, 1.0, 0.0, 0.0, orthogonal); /* +X: perpendicular */

	assert(along[0] > opposite[0]);
	assert(along[0] > orthogonal[0]);
	assert(along[0] > 0.0);
}

/* CPU mirror of environment_reflection_visibility in
   shaders/environment_lighting.glsl. */
static double reflection_visibility(double nov, double ao, double roughness)
{
	double exponent = exp2(-16.0 * roughness - 1.0);
	double base = fmin(fmax(nov + ao, 0.0), 2.0);
	double value = pow(base, exponent) - 1.0 + ao;
	return fmin(fmax(value, 0.0), 1.0);
}

static void reflection_visibility_contract(void)
{
	const double values[] = {0.0, 0.05, 0.25, 0.5, 0.75, 1.0};
	for (size_t r = 0; r < sizeof(values) / sizeof(values[0]); ++r)
		for (size_t n = 0; n < sizeof(values) / sizeof(values[0]); ++n)
			for (size_t a = 0; a < sizeof(values) / sizeof(values[0]); ++a)
			{
				double v = reflection_visibility(values[n], values[a], values[r]);
				assert(isfinite(v));
				assert(v >= 0.0 && v <= 1.0);
				assert(reflection_visibility(values[n], 0.0, values[r]) == 0.0);
				assert(fabs(reflection_visibility(values[n], 1.0, values[r]) - 1.0) < 1e-12);
				if (a + 1 < sizeof(values) / sizeof(values[0]))
					assert(v <= reflection_visibility(values[n], values[a + 1], values[r]) + 1e-12);
				if (n + 1 < sizeof(values) / sizeof(values[0]))
					assert(v <= reflection_visibility(values[n + 1], values[a], values[r]) + 1e-12);
			}
	for (size_t n = 0; n < sizeof(values) / sizeof(values[0]); ++n)
		for (size_t a = 0; a < sizeof(values) / sizeof(values[0]); ++a)
			assert(reflection_visibility(values[n], values[a], 1.0) <= values[a] + 0.02);
}

int main(void)
{
	uniform_environment_yields_flat_irradiance();
	bright_direction_peaks_reconstruction_along_it();
	reflection_visibility_contract();
	return 0;
}
