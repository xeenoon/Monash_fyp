#version 450
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"
#include "atmosphere_common.glsl"

layout(location = 0) in vec2 texcoord;
layout(location = 0) out vec4 out_color;
layout(set = 1, binding = 0) uniform sampler2D hdr_scene;
layout(set = 1, binding = 1) uniform sampler2D scene_depth;
layout(set = 2, binding = 0) uniform sampler2D atmosphere_transmittance_lut;
layout(set = 2, binding = 1) uniform sampler2D atmosphere_multiscattering_lut;
layout(set = 2, binding = 2) uniform sampler2D atmosphere_skyview_lut;
layout(set = 2, binding = 3) uniform sampler2DArray atmosphere_aerial_scattering;
layout(set = 2, binding = 4) uniform sampler2DArray atmosphere_aerial_transmittance;

/* Direct GLSL port of Wicked Engine tonemapCS.hlsl::RRTAndODTFit/ACESFitted
   (Wicked Engine contributors, MIT). We use its fixed-exposure ACES path but
   leave bloom, grading, and adaptive luminance for their owning phases. */
vec3 rrt_and_odt_fit(vec3 value) {
    vec3 a = value * (value + 0.0245786) - 0.000090537;
    vec3 b = value * (0.983729 * value + 0.4329510) + 0.238081;
    return a / b;
}

vec3 aces_fitted(vec3 color) {
    const mat3 input_matrix = mat3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777);
    const mat3 output_matrix = mat3(
         1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602);
    color = input_matrix * color;
    color = rrt_and_odt_fit(clamp(color, 0.0, 100.0));
    return clamp(output_matrix * color, 0.0, 1.0);
}

vec3 sky_with_sun(vec3 view_direction) {
    vec3 to_sun = normalize(-frame.sun_direction.xyz);
    vec3 sky = textureLod(atmosphere_skyview_lut,
        atmosphere_skyview_uv(view_direction, to_sun), 0.0).rgb;

    /* Adapted from Wicked GetSunLuminance (MIT): a finite angular disk with a
       soft outer quarter, attenuated by this same atmosphere's transmittance. */
    float cosine = dot(view_direction, to_sun);
    float outer = cos(frame.atmosphere_radii.w);
    float inner = outer + (1.0 - outer) * 0.25;
    float disk = smoothstep(outer, inner, cosine);
    vec3 disk_transmittance = atmosphere_transmittance_to_sun(
        atmosphere_transmittance_lut, atmosphere_camera_position(), view_direction);
    return sky + disk * frame.sun_radiance.rgb * disk_transmittance;
}

vec3 sample_aerial_volume(sampler2DArray volume, vec2 uv, float w) {
    /* A 2D array is used as portable 3D storage. This is ordinary manual
       trilinear interpolation across Wicked's squared depth slices. */
    float layer = clamp(w * ATM_AERIAL_SIZE.z - 0.5,
                        0.0, ATM_AERIAL_SIZE.z - 1.0);
    float lower = floor(layer);
    float upper = min(lower + 1.0, ATM_AERIAL_SIZE.z - 1.0);
    return mix(textureLod(volume, vec3(uv, lower), 0.0).rgb,
               textureLod(volume, vec3(uv, upper), 0.0).rgb,
               fract(layer));
}

vec3 debug_atmosphere(float mode) {
    if (mode < 12.5)
        return textureLod(atmosphere_transmittance_lut, texcoord, 0.0).rgb;
    if (mode < 13.5)
        return textureLod(atmosphere_multiscattering_lut, texcoord, 0.0).rgb;
    if (mode < 14.5)
        return textureLod(atmosphere_skyview_lut, texcoord, 0.0).rgb;
    float layer = clamp(frame.atmosphere_options.y, 0.0, 31.0);
    if (mode < 15.5)
        return textureLod(atmosphere_aerial_scattering,
                          vec3(texcoord, layer), 0.0).rgb;
    return textureLod(atmosphere_aerial_transmittance,
                      vec3(texcoord, layer), 0.0).rgb;
}

void main() {
    vec3 source = texture(hdr_scene, texcoord).rgb;
    if (frame.debug_view > 11.5) {
        out_color = vec4(aces_fitted(debug_atmosphere(frame.debug_view) *
                                    frame.shadow_parameters.z), 1.0);
        return;
    }
    if (frame.debug_view > 0.5) {
        out_color = vec4(clamp(source, 0.0, 1.0), 1.0);
        return;
    }

    float depth = texture(scene_depth, texcoord).r;
    vec4 ray_h = frame.inverse_view_projection *
                 vec4(texcoord * 2.0 - 1.0, 0.0, 1.0);
    vec3 view_direction = normalize(ray_h.xyz);
    vec3 atmosphere_composite;
    if (depth <= 1e-8) {
        atmosphere_composite = sky_with_sun(view_direction);
    } else {
        vec3 surface_position = reconstruct_camera_relative(texcoord, depth);
        float distance_km = length(surface_position) * 0.001;
        float w = sqrt(clamp(distance_km /
                       frame.atmosphere_options.x, 0.0, 1.0));
        vec3 scattering = sample_aerial_volume(
            atmosphere_aerial_scattering, texcoord, w);
        vec3 transmittance = sample_aerial_volume(
            atmosphere_aerial_transmittance, texcoord, w);
        /* Wicked GetAerialPerspectiveTransmittance fades the first half of one
           4 km linear slice (2 km) from the camera to avoid a near-volume seam. */
        float near_weight = clamp(distance_km /
            (0.5 * frame.atmosphere_options.x / ATM_AERIAL_SIZE.z), 0.0, 1.0);
        scattering *= near_weight;
        transmittance = mix(vec3(1.0), transmittance, near_weight);
        /* Opaque compositing happens after all terrain lighting and before the
           one tone map: surface * coloured transmittance + in-scattering. */
        atmosphere_composite = source * transmittance + scattering;
    }
    vec3 hdr = atmosphere_composite * frame.shadow_parameters.z;
    /* The swapchain is B8G8R8A8_SRGB, so Vulkan applies the output transfer
       curve exactly once after this linear tone-mapped value is written. */
    out_color = vec4(aces_fitted(hdr), 1.0);
}
