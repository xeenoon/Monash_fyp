#include "atmosphere.h"

#include <math.h>

AtmosphereParameters atmosphere_earth(void) {
    /* Values are the Earth preset from Unreal Sky Atmosphere's
       FAtmosphereParameters constructor (MIT). Keeping them here prevents four
       shader passes from quietly acquiring different copies. */
    return (AtmosphereParameters){
        .bottom_radius_km = 6360.0f,
        .top_radius_km = 6460.0f,
        .rayleigh_scattering = {0.005802f, 0.013558f, 0.033100f},
        .rayleigh_density_exp_scale = -1.0f / 8.0f,
        .mie_scattering = {0.003996f, 0.003996f, 0.003996f},
        .mie_density_exp_scale = -1.0f / 1.2f,
        .mie_extinction = {0.004440f, 0.004440f, 0.004440f},
        .mie_phase_g = 0.8f,
        .absorption_extinction = {0.000650f, 0.001881f, 0.000085f},
        .ground_albedo = {0.30f, 0.30f, 0.30f},
        .multiple_scattering_factor = 1.0f,
        .aerial_max_distance_km = 128.0f,
        .sun_angular_radius_rad = 0.004675f,
    };
}

bool atmosphere_parameters_valid(const AtmosphereParameters *a) {
    if (!a || !isfinite(a->bottom_radius_km) || !isfinite(a->top_radius_km) ||
        a->bottom_radius_km <= 0.0f || a->top_radius_km <= a->bottom_radius_km ||
        a->aerial_max_distance_km <= 0.0f || a->sun_angular_radius_rad <= 0.0f ||
        fabsf(a->mie_phase_g) >= 1.0f)
        return false;
    for (unsigned i = 0; i < 3; ++i)
        if (a->rayleigh_scattering[i] < 0.0f || a->mie_scattering[i] < 0.0f ||
            a->mie_extinction[i] < a->mie_scattering[i] ||
            a->absorption_extinction[i] < 0.0f)
            return false;
    return true;
}

void atmosphere_transmittance_to_uv(const AtmosphereParameters *a,
                                    float height, float mu, float uv[2]) {
    /* Direct C port of Wicked skyAtmosphere.hlsli:
       LutTransmittanceParamsToUv (MIT), itself Bruneton 2017. */
    float h = sqrtf(fmaxf(0.0f, a->top_radius_km * a->top_radius_km -
                                a->bottom_radius_km * a->bottom_radius_km));
    float rho = sqrtf(fmaxf(0.0f, height * height -
                                  a->bottom_radius_km * a->bottom_radius_km));
    float discriminant = height * height * (mu * mu - 1.0f) +
                         a->top_radius_km * a->top_radius_km;
    float d = fmaxf(0.0f, -height * mu + sqrtf(fmaxf(discriminant, 0.0f)));
    float d_min = a->top_radius_km - height;
    float d_max = rho + h;
    uv[0] = (d - d_min) / fmaxf(d_max - d_min, 1e-6f);
    uv[1] = rho / h;
}

void atmosphere_uv_to_transmittance(const AtmosphereParameters *a,
                                    const float uv[2], float *height,
                                    float *mu) {
    /* Direct C port of Wicked skyAtmosphere.hlsli:
       UvToLutTransmittanceParams (MIT). */
    float h = sqrtf(a->top_radius_km * a->top_radius_km -
                    a->bottom_radius_km * a->bottom_radius_km);
    float rho = h * uv[1];
    *height = sqrtf(rho * rho + a->bottom_radius_km * a->bottom_radius_km);
    float d_min = a->top_radius_km - *height;
    float d_max = rho + h;
    float d = d_min + uv[0] * (d_max - d_min);
    *mu = d == 0.0f ? 1.0f :
        (h * h - rho * rho - d * d) / (2.0f * *height * d);
    *mu = fmaxf(-1.0f, fminf(1.0f, *mu));
}

float atmosphere_aerial_slice_to_depth(float slice, float max_distance_km) {
    /* Wicked camera-volume LUT uses the same squared distribution. Here slice
       is normalised, which keeps the mapping valid if the volume resolution
       changes from its initial 32 layers. */
    float w = fmaxf(0.0f, fminf(1.0f, slice));
    return w * w * max_distance_km;
}

float atmosphere_aerial_depth_to_w(float depth_km, float max_distance_km) {
    return sqrtf(fmaxf(0.0f, fminf(1.0f, depth_km / max_distance_km)));
}
