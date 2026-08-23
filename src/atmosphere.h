#pragma once

#include <stdbool.h>

/* The one CPU-side atmosphere definition copied into FrameUniforms. Distances
   and coefficients use kilometres, matching the Unreal/Wicked model and
   avoiding loss of precision at an Earth-sized radius. */
typedef struct
{
	float bottom_radius_km;
	float top_radius_km;
	float rayleigh_scattering[3];
	float rayleigh_density_exp_scale;
	float mie_scattering[3];
	float mie_density_exp_scale;
	float mie_extinction[3];
	float mie_phase_g;
	float absorption_extinction[3];
	float ground_albedo[3];
	float multiple_scattering_factor;
	float aerial_max_distance_km;
	float sun_angular_radius_rad;
} AtmosphereParameters;

AtmosphereParameters atmosphere_earth(void);
bool atmosphere_parameters_valid(const AtmosphereParameters *atmosphere);

/* Bruneton 2017 transmittance parameterisation, ported through Wicked's
   skyAtmosphere.hlsli so CPU tests exercise the shader's coordinate contract. */
void atmosphere_transmittance_to_uv(const AtmosphereParameters *atmosphere, float height_km,
									float zenith_cos, float uv[2]);
void atmosphere_uv_to_transmittance(const AtmosphereParameters *atmosphere, const float uv[2],
									float *height_km, float *zenith_cos);

float atmosphere_aerial_slice_to_depth(float slice, float max_distance_km);
float atmosphere_aerial_depth_to_w(float depth_km, float max_distance_km);
