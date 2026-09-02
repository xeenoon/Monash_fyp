#version 450
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"
#include "atmosphere_common.glsl"
#include "shader_dump.glsl"

layout(location = 0) in vec2 texcoord;
layout(location = 0) out vec4 out_color;
layout(set = 1, binding = 0) uniform sampler2D hdr_scene;
layout(set = 1, binding = 1) uniform sampler2D scene_depth;
layout(set = 2, binding = 0) uniform sampler2D atmosphere_transmittance_lut;
layout(set = 2, binding = 1) uniform sampler2D atmosphere_multiscattering_lut;
layout(set = 2, binding = 2) uniform sampler2D atmosphere_skyview_lut;
layout(set = 2, binding = 3) uniform sampler2DArray atmosphere_aerial_scattering;
layout(set = 2, binding = 4) uniform sampler2DArray atmosphere_aerial_transmittance;

float celestial_disk(vec3 view_direction, vec3 direction, float radius) {
    float outer = cos(radius);
    float inner = outer + (1.0 - outer) * 0.25;
    return smoothstep(outer, inner, dot(view_direction, direction));
}

/* There is no bloom pass in this renderer, so a physically-sized HDR sun disk
   (~0.27 deg radius) reads as a near-invisible dot once tonemapped. This term
   fakes the halation a bloom pass would add, tightly clamped so it stays a
   small glow rather than washing out the sky. */
float celestial_glow(vec3 view_direction, vec3 direction) {
    float cosine = max(dot(view_direction, direction), 0.0);
    return pow(cosine, 5000.0);
}

vec3 sky_with_celestials(vec3 view_direction) {
    vec3 to_sun = normalize(-frame.sun_direction.xyz);
    vec3 sky = textureLod(atmosphere_skyview_lut,
        atmosphere_skyview_uv(view_direction, to_sun), 0.0).rgb;
    vec3 transmittance = atmosphere_transmittance_to_sun(
        atmosphere_transmittance_lut, atmosphere_camera_position(), view_direction);
    float sun_disk = celestial_disk(
        view_direction, to_sun, frame.atmosphere_radii.w * 2.7);
    float sun_glow = celestial_glow(view_direction, to_sun);
    vec3 to_moon = -to_sun;
    float moon_disk = celestial_disk(
        view_direction, to_moon, frame.atmosphere_radii.w * 1.25);
    vec3 sun = (sun_disk + sun_glow * 0.25) * frame.sun_radiance.rgb * 8.0 * transmittance;
    vec3 moon = moon_disk * vec3(0.10, 0.12, 0.16) * transmittance;
    return sky + sun + moon;
}

vec3 sample_aerial_volume(sampler2DArray volume, vec2 uv, float w) {
    float layer = clamp(w * ATM_AERIAL_SIZE.z - 0.5,
                        0.0, ATM_AERIAL_SIZE.z - 1.0);
    float lower = floor(layer);
    float upper = min(lower + 1.0, ATM_AERIAL_SIZE.z - 1.0);
    return mix(textureLod(volume, vec3(uv, lower), 0.0).rgb,
               textureLod(volume, vec3(uv, upper), 0.0).rgb, fract(layer));
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

/* See SHADER_DUMP_LEGEND["atmosphere_composite"] in renderer.c for the f0..f19
   layout; f3 (branch) identifies which of the four exits below produced the
   record: 1=LUT debug view, 2=debug passthrough, 3=sky (no depth), 4=aerial
   perspective composite. */
void main() {
    vec3 source = texture(hdr_scene, texcoord).rgb;
    if (frame.atmosphere_options.z < 0.5) {
        out_color = vec4(source, 1.0);
        return;
    }
    if (frame.debug_view > 11.5 && frame.debug_view < 16.5) {
        out_color = vec4(debug_atmosphere(frame.debug_view), 1.0);
        shader_dump(DUMP_SHADER_ATMOSPHERE_COMPOSITE,
                    vec4(texcoord, 0.0, 1.0), vec4(out_color.rgb, 0.0),
                    vec4(0.0), vec4(0.0), vec4(0.0));
        return;
    }
    if (frame.debug_view > 0.5 && frame.debug_view < 11.5) {
        out_color = vec4(source, 1.0);
        shader_dump(DUMP_SHADER_ATMOSPHERE_COMPOSITE,
                    vec4(texcoord, 0.0, 2.0), vec4(out_color.rgb, 0.0),
                    vec4(0.0), vec4(0.0), vec4(0.0));
        return;
    }

    float depth = texture(scene_depth, texcoord).r;
    vec4 ray_h = frame.inverse_view_projection *
                 vec4(texcoord * 2.0 - 1.0, 0.0, 1.0);
    vec3 view_direction = normalize(ray_h.xyz);
    if (depth <= 1e-8) {
        out_color = vec4(sky_with_celestials(view_direction), 1.0);
        shader_dump(DUMP_SHADER_ATMOSPHERE_COMPOSITE,
                    vec4(texcoord, depth, 3.0), vec4(out_color.rgb, 0.0),
                    vec4(view_direction, 0.0), vec4(0.0), vec4(0.0));
        return;
    }

    vec3 surface_position = reconstruct_camera_relative(texcoord, depth);
    float distance_km = length(surface_position) * 0.001;
    float w = sqrt(clamp(distance_km / frame.atmosphere_options.x, 0.0, 1.0));
    vec3 scattering = sample_aerial_volume(
        atmosphere_aerial_scattering, texcoord, w);
    vec3 transmittance = sample_aerial_volume(
        atmosphere_aerial_transmittance, texcoord, w);
    float near_weight = clamp(distance_km /
        (0.5 * frame.atmosphere_options.x / ATM_AERIAL_SIZE.z), 0.0, 1.0);
    scattering *= near_weight;
    transmittance = mix(vec3(1.0), transmittance, near_weight);
    out_color = vec4(source * transmittance + scattering, 1.0);
    shader_dump(DUMP_SHADER_ATMOSPHERE_COMPOSITE,
                vec4(texcoord, depth, 4.0), vec4(out_color.rgb, 0.0),
                vec4(view_direction, distance_km), vec4(scattering, near_weight),
                vec4(transmittance, w));
}
