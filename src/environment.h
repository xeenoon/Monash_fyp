#pragma once

#include <stdbool.h>

/* CPU-side spherical-harmonic approximation of a captured environment's
   diffuse response. 3rd-order (9-term) real SH, RGB per term. Two things are
   baked into these coefficients at projection time so a shader can reconstruct
   them as a flat 9-term polynomial in the surface normal with no extra
   constants (see environment_irradiance in shaders/environment_lighting.glsl):
	 - the l=1/l=2 cosine-lobe convolution (Ramamoorthi & Hanrahan 2001, "An
	   Efficient Representation for Irradiance Environment Maps", the A_l
	   bands: A0=pi, A1=2pi/3, A2=pi/4);
	 - the 1/pi Lambertian normalization, so the result is ready to multiply
	   directly by a diffuse albedo (matching how the hemispheric-ambient
	   fallback constant it replaces was already used unscaled). */
typedef struct
{
	float coeffs[9][3];
} EnvironmentSH;

/* Decode a Radiance .hdr equirectangular environment as RGBA (via stbi_loadf,
   alpha forced to 1) so the same buffer uploads directly to a 4-component GPU
   format for the B2 specular prefilter with no repacking. *pixels is
   allocated by stb_image; free with environment_free_hdr. Returns false
   (leaving *pixels, *width, *height untouched) if `path` is missing or fails
   to decode -- the caller falls back to the hemispheric ambient term, per the
   "absent-able without changing draw submission" invariant. */
bool environment_load_hdr(const char *path, float **pixels, int *width, int *height);
void environment_free_hdr(float *pixels);

/* Solid-angle-weighted projection of an equirectangular HDR image (RGBA,
   row-major, `width`x`height`, u in [0,1) -> longitude, v=0 at +Y/up -> v=1 at
   -Y/down; alpha ignored) onto 3rd-order SH. Pure function, no I/O --
   unit-testable independent of environment_load_hdr. */
void environment_project_sh9(const float *pixels, int width, int height, EnvironmentSH *out);
