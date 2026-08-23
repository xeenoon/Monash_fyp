/* GLSL subset of Wicked Engine skyAtmosphere.hlsli (MIT), which implements
   Sebastien Hillaire's Unreal Sky Atmosphere reference (MIT). We retain the
   proven LUT coordinates, density model, phases, ray/sphere tests, and exact
   finite-segment integral; engine-specific bindless/shadow code is omitted. */
#ifndef ATMOSPHERE_COMMON_GLSL
#define ATMOSPHERE_COMMON_GLSL

const float ATM_PI = 3.14159265358979323846;
const vec2 ATM_TRANSMITTANCE_SIZE = vec2(256.0, 64.0);
const vec2 ATM_MULTISCATTER_SIZE = vec2(32.0);
const vec2 ATM_SKYVIEW_SIZE = vec2(192.0, 108.0);
const vec3 ATM_AERIAL_SIZE = vec3(32.0);
const float ATM_PLANET_RADIUS_OFFSET = 0.001; /* 1 metre in sky kilometres */

struct AtmosphereMedium {
    vec3 scattering;
    vec3 extinction;
    vec3 scattering_mie;
    vec3 scattering_rayleigh;
};

struct AtmosphereIntegral {
    vec3 luminance;
    vec3 transmittance;
    vec3 multi_scatter_as_one;
};

float atmosphere_bottom_radius() { return frame.atmosphere_radii.x; }
float atmosphere_top_radius() { return frame.atmosphere_radii.y; }

vec3 atmosphere_camera_position() {
    return vec3(0.0, atmosphere_bottom_radius() +
                     max(frame.atmosphere_radii.z, ATM_PLANET_RADIUS_OFFSET), 0.0);
}

vec3 atmosphere_position_from_camera_relative(vec3 relative_metres) {
    return atmosphere_camera_position() + relative_metres * 0.001;
}

float from_unit_to_sub_uv(float u, float resolution) {
    return (u + 0.5 / resolution) * (resolution / (resolution + 1.0));
}

float from_sub_uv_to_unit(float u, float resolution) {
    return (u - 0.5 / resolution) * (resolution / (resolution - 1.0));
}

/* Direct port of Wicked UvToLutTransmittanceParams/Lut...ToUv (MIT), itself
   the Bruneton 2017 transmittance parameterisation. */
void uv_to_transmittance_params(vec2 uv, out float height, out float mu) {
    float H = sqrt(atmosphere_top_radius() * atmosphere_top_radius() -
                   atmosphere_bottom_radius() * atmosphere_bottom_radius());
    float rho = H * uv.y;
    height = sqrt(rho * rho + atmosphere_bottom_radius() * atmosphere_bottom_radius());
    float d_min = atmosphere_top_radius() - height;
    float d_max = rho + H;
    float d = d_min + uv.x * (d_max - d_min);
    mu = d == 0.0 ? 1.0 :
         (H * H - rho * rho - d * d) / (2.0 * height * d);
    mu = clamp(mu, -1.0, 1.0);
}

vec2 transmittance_params_to_uv(float height, float mu) {
    float H = sqrt(max(0.0, atmosphere_top_radius() * atmosphere_top_radius() -
                            atmosphere_bottom_radius() * atmosphere_bottom_radius()));
    float rho = sqrt(max(0.0, height * height -
                              atmosphere_bottom_radius() * atmosphere_bottom_radius()));
    float discriminant = height * height * (mu * mu - 1.0) +
                         atmosphere_top_radius() * atmosphere_top_radius();
    float d = max(0.0, -height * mu + sqrt(max(discriminant, 0.0)));
    float d_min = atmosphere_top_radius() - height;
    float d_max = rho + H;
    return vec2((d - d_min) / max(d_max - d_min, 1e-6), rho / H);
}

vec2 ray_sphere_intersections(vec3 origin, vec3 direction, float radius) {
    /* Copied from Wicked RaySphereIntersect (MIT); the planet is centred at 0. */
    float b = 2.0 * dot(direction, origin);
    float c = dot(origin, origin) - radius * radius;
    float delta = b * b - 4.0 * dot(direction, direction) * c;
    if (delta < 0.0) return vec2(-1.0);
    return (-b + vec2(-1.0, 1.0) * sqrt(delta)) /
           (2.0 * dot(direction, direction));
}

float ray_sphere_nearest(vec3 origin, vec3 direction, float radius) {
    vec2 solution = ray_sphere_intersections(origin, direction, radius);
    if (solution.x < 0.0 && solution.y < 0.0) return -1.0;
    if (solution.x < 0.0) return max(0.0, solution.y);
    if (solution.y < 0.0) return max(0.0, solution.x);
    return max(0.0, min(solution.x, solution.y));
}

AtmosphereMedium sample_atmosphere_medium(vec3 position) {
    /* Unreal/Wicked's exponential Rayleigh/Mie and two-piece ozone profile.
       The two-piece profile reduces to a triangle centred at 25 km here. */
    float altitude = max(length(position) - atmosphere_bottom_radius(), 0.0);
    float mie_density = exp(frame.atmosphere_mie_scatter.w * altitude);
    float rayleigh_density = exp(frame.atmosphere_rayleigh.w * altitude);
    float ozone_density = clamp(1.0 - abs(altitude - 25.0) / 15.0, 0.0, 1.0);
    AtmosphereMedium medium;
    medium.scattering_mie = mie_density * frame.atmosphere_mie_scatter.rgb;
    medium.scattering_rayleigh = rayleigh_density * frame.atmosphere_rayleigh.rgb;
    medium.scattering = medium.scattering_mie + medium.scattering_rayleigh;
    medium.extinction = mie_density * frame.atmosphere_mie_extinct.rgb +
                        medium.scattering_rayleigh +
                        ozone_density * frame.atmosphere_absorption.rgb;
    return medium;
}

float rayleigh_phase(float cosine) {
    return 3.0 / (16.0 * ATM_PI) * (1.0 + cosine * cosine);
}

float mie_phase(float g, float cosine) {
    /* Wicked's Cornette-Shanks phase (MIT), retained for its sharper but
       energy-normalised forward lobe. */
    float k = 3.0 / (8.0 * ATM_PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + cosine * cosine) /
           pow(abs(1.0 + g * g - 2.0 * g * -cosine), 1.5);
}

vec3 atmosphere_transmittance_to_sun(sampler2D transmittance_lut,
                                     vec3 position, vec3 to_sun) {
    if (any(greaterThan(ray_sphere_intersections(
            position, to_sun, atmosphere_bottom_radius()), vec2(0.0))))
        return vec3(0.0);
    float height = length(position);
    float mu = dot(to_sun, position / height);
    return textureLod(transmittance_lut,
        clamp(transmittance_params_to_uv(height, mu), vec2(0.0), vec2(1.0)), 0.0).rgb;
}

vec3 atmosphere_multiple_scattering(sampler2D multiple_lut,
                                    vec3 position, float sun_zenith_cos) {
    vec2 uv = clamp(vec2(sun_zenith_cos * 0.5 + 0.5,
        (length(position) - atmosphere_bottom_radius()) /
        (atmosphere_top_radius() - atmosphere_bottom_radius())), 0.0, 1.0);
    uv = vec2(from_unit_to_sub_uv(uv.x, ATM_MULTISCATTER_SIZE.x),
              from_unit_to_sub_uv(uv.y, ATM_MULTISCATTER_SIZE.y));
    return textureLod(multiple_lut, uv, 0.0).rgb;
}

float atmosphere_ray_limit(vec3 origin, vec3 direction) {
    float bottom = ray_sphere_nearest(origin, direction, atmosphere_bottom_radius());
    float top = ray_sphere_nearest(origin, direction, atmosphere_top_radius());
    if (bottom < 0.0) return top;
    if (top < 0.0) return bottom;
    return min(bottom, top);
}

vec3 integrate_optical_depth(vec3 origin, vec3 direction, float limit,
                             int sample_count) {
    vec3 optical_depth = vec3(0.0);
    float dt = limit / float(sample_count);
    for (int i = 0; i < sample_count; ++i) {
        float t = (float(i) + 0.5) * dt;
        optical_depth += sample_atmosphere_medium(origin + direction * t).extinction * dt;
    }
    return optical_depth;
}

AtmosphereIntegral integrate_atmosphere(
    vec3 origin, vec3 direction, vec3 to_sun, vec3 sun_illuminance,
    float distance_limit,
    int sample_count, bool directional_phase, bool use_multiple_scattering,
    sampler2D transmittance_lut, sampler2D multiple_lut) {
    AtmosphereIntegral result = AtmosphereIntegral(vec3(0.0), vec3(1.0), vec3(0.0));
    float ray_limit = atmosphere_ray_limit(origin, direction);
    if (ray_limit <= 0.0) return result;
    float limit = distance_limit > 0.0 ? min(ray_limit, distance_limit) : ray_limit;
    float dt = limit / float(sample_count);
    float cosine = dot(to_sun, direction);
    float phase_r = directional_phase ? rayleigh_phase(cosine) : 1.0 / (4.0 * ATM_PI);
    float phase_m = directional_phase ? mie_phase(frame.atmosphere_mie_extinct.w, -cosine)
                                      : 1.0 / (4.0 * ATM_PI);
    vec3 throughput = vec3(1.0);
    for (int i = 0; i < sample_count; ++i) {
        /* Wicked/Unreal bias toward the segment start (0.3) to retain the
           horizon's dense lower atmosphere with modest sample counts. */
        float t = (float(i) + 0.3) * dt;
        vec3 position = origin + direction * t;
        AtmosphereMedium medium = sample_atmosphere_medium(position);
        vec3 sample_transmittance = exp(-medium.extinction * dt);
        vec3 up = position / length(position);
        float sun_mu = dot(to_sun, up);
        vec3 sun_transmittance = atmosphere_transmittance_to_sun(
            transmittance_lut, position, to_sun);
        vec3 phase_scattering = medium.scattering_rayleigh * phase_r +
                                medium.scattering_mie * phase_m;
        vec3 multiple = use_multiple_scattering
            ? atmosphere_multiple_scattering(multiple_lut, position, sun_mu) : vec3(0.0);
        vec3 source = sun_illuminance *
            (sun_transmittance * phase_scattering + multiple * medium.scattering);

        /* Copied exactly from Unreal RenderSkyRayMarching.hlsl:231-234 and
           Wicked IntegrateScatteredLuminance (MIT): analytic finite-segment
           integration is stable when extinction or step length is small. */
        vec3 safe_extinction = max(medium.extinction, vec3(1e-7));
        vec3 segment = (source - source * sample_transmittance) / safe_extinction;
        result.luminance += throughput * segment;
        vec3 transfer_source = medium.scattering;
        result.multi_scatter_as_one += throughput *
            (transfer_source - transfer_source * sample_transmittance) /
            safe_extinction;
        throughput *= sample_transmittance;
    }
    result.transmittance = throughput;
    return result;
}

/* Direct ports of Wicked's non-linear sky-view coordinate maps (MIT). */
void uv_to_skyview_params(vec2 uv, float view_height,
                          out float view_zenith_cos, out float light_view_cos) {
    uv = vec2(from_sub_uv_to_unit(uv.x, ATM_SKYVIEW_SIZE.x),
              from_sub_uv_to_unit(uv.y, ATM_SKYVIEW_SIZE.y));
    float horizon = sqrt(max(view_height * view_height -
                       atmosphere_bottom_radius() * atmosphere_bottom_radius(), 0.0));
    float beta = acos(clamp(horizon / view_height, -1.0, 1.0));
    float zenith_horizon = ATM_PI - beta;
    if (uv.y < 0.5) {
        float coordinate = 1.0 - 2.0 * uv.y;
        coordinate *= coordinate;
        view_zenith_cos = cos(zenith_horizon * (1.0 - coordinate));
    } else {
        float coordinate = uv.y * 2.0 - 1.0;
        coordinate *= coordinate;
        view_zenith_cos = cos(zenith_horizon + beta * coordinate);
    }
    light_view_cos = -(uv.x * uv.x * 2.0 - 1.0);
}

vec2 skyview_params_to_uv(bool intersects_ground, float view_zenith_cos,
                          float light_view_cos, float view_height) {
    float horizon = sqrt(max(view_height * view_height -
                       atmosphere_bottom_radius() * atmosphere_bottom_radius(), 0.0));
    float beta = acos(clamp(horizon / view_height, -1.0, 1.0));
    float zenith_horizon = ATM_PI - beta;
    vec2 uv;
    if (!intersects_ground) {
        float coordinate = 1.0 - acos(clamp(view_zenith_cos, -1.0, 1.0)) /
                                      zenith_horizon;
        uv.y = (1.0 - sqrt(abs(coordinate))) * 0.5;
    } else {
        float coordinate = (acos(clamp(view_zenith_cos, -1.0, 1.0)) -
                            zenith_horizon) / beta;
        uv.y = sqrt(abs(coordinate)) * 0.5 + 0.5;
    }
    uv.x = sqrt(clamp(-light_view_cos * 0.5 + 0.5, 0.0, 1.0));
    return vec2(from_unit_to_sub_uv(uv.x, ATM_SKYVIEW_SIZE.x),
                from_unit_to_sub_uv(uv.y, ATM_SKYVIEW_SIZE.y));
}

vec2 atmosphere_skyview_uv(vec3 direction, vec3 to_sun) {
    vec3 origin = atmosphere_camera_position();
    vec3 up = origin / length(origin);
    float view_cos = dot(direction, up);
    vec3 view_horizontal = direction - up * view_cos;
    vec3 sun_horizontal = to_sun - up * dot(to_sun, up);
    float light_cos = length(view_horizontal) > 1e-5 && length(sun_horizontal) > 1e-5
        ? dot(normalize(view_horizontal), normalize(sun_horizontal)) : 1.0;
    bool ground = ray_sphere_nearest(origin, direction,
                                     atmosphere_bottom_radius()) >= 0.0;
    return clamp(skyview_params_to_uv(ground, view_cos, light_cos,
                                      length(origin)), vec2(0.0), vec2(1.0));
}

#endif
