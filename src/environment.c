#include "environment.h"
#include "stb_image.h"

#include <math.h>
#include <stddef.h>

bool environment_load_hdr(const char *path, float **pixels, int *width, int *height)
{
	int channels = 0;
	float *data = stbi_loadf(path, width, height, &channels, 3);
	if (!data)
		return false;
	*pixels = data;
	return true;
}

void environment_free_hdr(float *pixels)
{
	stbi_image_free(pixels);
}

void environment_project_sh9(const float *pixels, int width, int height, EnvironmentSH *out)
{
	const double pi = 3.14159265358979323846;
	/* A_l/pi (Ramamoorthi & Hanrahan cosine-lobe convolution, pre-divided by
	   the Lambertian 1/pi) per SH band: l=0 -> 1, l=1 -> 2/3, l=2 -> 1/4. */
	static const double band_scale[9] = {1.0,  2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.25,
										 0.25, 0.25,	  0.25,		 0.25};
	double sums[9][3] = {{0}};
	double dtheta = pi / (double)height;
	double dphi = 2.0 * pi / (double)width;
	for (int j = 0; j < height; ++j)
	{
		double v = ((double)j + 0.5) / (double)height;
		double theta = v * pi;
		double sin_theta = sin(theta), cos_theta = cos(theta);
		double domega = sin_theta * dtheta * dphi;
		for (int i = 0; i < width; ++i)
		{
			double u = ((double)i + 0.5) / (double)width;
			double phi = u * 2.0 * pi;
			double x = sin_theta * cos(phi);
			double y = cos_theta;
			double z = sin_theta * sin(phi);
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
			const float *texel = &pixels[((size_t)j * (size_t)width + (size_t)i) * 3];
			for (int c = 0; c < 3; ++c)
			{
				double L = texel[c] * domega;
				for (int t = 0; t < 9; ++t)
					sums[t][c] += L * basis[t];
			}
		}
	}
	for (int t = 0; t < 9; ++t)
		for (int c = 0; c < 3; ++c)
			out->coeffs[t][c] = (float)(sums[t][c] * band_scale[t]);
}
