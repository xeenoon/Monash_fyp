#version 450
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) in vec2 texcoord;
layout(location = 0) out vec4 out_history;
layout(location = 1) out float out_history_depth;

layout(set = 1, binding = 0) uniform sampler2D current_hdr;
layout(set = 1, binding = 1) uniform sampler2D current_depth;
layout(set = 1, binding = 2) uniform sampler2D motion_vectors;
layout(set = 1, binding = 3) uniform sampler2D previous_history;
layout(set = 1, binding = 4) uniform sampler2D previous_depth;

vec3 hdr_compress(vec3 color) {
    float luminance = dot(max(color, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
    return color / (1.0 + luminance);
}

vec3 hdr_expand(vec3 color) {
    float luminance = dot(max(color, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
    return color / max(1.0 - luminance, 1e-4);
}

vec2 sky_motion(vec2 uv) {
    vec4 ray_h = frame.inverse_view_projection * vec4(uv * 2.0 - 1.0, 0.0, 1.0);
    vec3 direction = normalize(ray_h.xyz);
    vec4 previous_clip = frame.previous_projection * frame.previous_view *
                         vec4(direction, 0.0);
    if (previous_clip.w <= 0.0) return vec2(2.0);
    vec2 previous_uv = previous_clip.xy / previous_clip.w * 0.5 + 0.5;
    return previous_uv - uv;
}

void main() {
    ivec2 size = textureSize(current_hdr, 0);
    ivec2 pixel = clamp(ivec2(gl_FragCoord.xy), ivec2(0), size - 1);
    float depth = texelFetch(current_depth, pixel, 0).r;
    out_history_depth = depth;

    vec3 neighbourhood_min = vec3(1e30);
    vec3 neighbourhood_max = vec3(-1e30);
    float nearest_depth = -1.0;
    ivec2 velocity_pixel = pixel;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            ivec2 sample_pixel = clamp(pixel + ivec2(x, y), ivec2(0), size - 1);
            vec3 sample_color = hdr_compress(
                texelFetch(current_hdr, sample_pixel, 0).rgb);
            neighbourhood_min = min(neighbourhood_min, sample_color);
            neighbourhood_max = max(neighbourhood_max, sample_color);
            float sample_depth = texelFetch(current_depth, sample_pixel, 0).r;
            if (sample_depth > nearest_depth) {
                nearest_depth = sample_depth;
                velocity_pixel = sample_pixel;
            }
        }
    }

    vec2 velocity = nearest_depth > 1e-8
        ? texelFetch(motion_vectors, velocity_pixel, 0).rg
        : sky_motion(texcoord);
    vec2 own_velocity = texelFetch(motion_vectors, pixel, 0).rg;
    bool replacement_pixel = depth > 1e-8 &&
                             any(greaterThanEqual(abs(own_velocity), vec2(1.0)));
    vec2 previous_uv = texcoord + velocity;
    bool valid = frame.temporal_parameters.x > 0.5 && !replacement_pixel &&
                 all(greaterThan(previous_uv, vec2(0.0))) &&
                 all(lessThan(previous_uv, vec2(1.0))) &&
                 all(lessThan(abs(velocity), vec2(1.0)));

    float old_depth = valid ? texture(previous_depth, previous_uv).r : 0.0;
    if (depth <= 1e-8)
        valid = valid && old_depth <= 1e-8;
    else {
        vec3 current_position = reconstruct_camera_relative(texcoord, depth);
        vec4 previous_position = frame.previous_local_to_camera_relative *
                                 vec4(current_position, 1.0);
        vec4 predicted_clip = frame.previous_view_projection * previous_position;
        float predicted_depth = predicted_clip.z / predicted_clip.w;
        float predicted_metres = linear_view_depth(predicted_depth);
        float previous_metres = linear_view_depth(old_depth);
        float tolerance = max(1.0, predicted_metres * 0.025);
        valid = valid && old_depth > 1e-8 &&
                predicted_depth > 0.0 &&
                abs(previous_metres - predicted_metres) <= tolerance;
    }

    vec3 current = hdr_compress(texelFetch(current_hdr, pixel, 0).rgb);
    vec3 history = valid ? hdr_compress(texture(previous_history, previous_uv).rgb)
                         : current;
    vec3 unclamped_history = history;
    history = clamp(history, neighbourhood_min, neighbourhood_max);
    float motion_pixels = length(velocity * vec2(size));
    float current_weight = mix(0.08, 0.35, clamp(motion_pixels / 24.0, 0.0, 1.0));
    vec3 resolved = valid ? mix(history, current, current_weight) : current;

    if (frame.debug_view > 16.5 && frame.debug_view < 17.5)
        resolved = vec3(valid ? 1.0 - current_weight : 0.0);
    else if (frame.debug_view < 18.5 && frame.debug_view > 17.5)
        resolved = valid ? vec3(0.1, 0.9, 0.2) : vec3(0.95, 0.1, 0.05);
    else if (frame.debug_view < 19.5 && frame.debug_view > 18.5)
        resolved = vec3(velocity * 20.0 + 0.5, 0.5);
    else if (frame.debug_view < 20.5 && frame.debug_view > 19.5)
        resolved = clamp(abs(unclamped_history - history) * 12.0, 0.0, 1.0);

    bool temporal_debug = frame.debug_view > 16.5 && frame.debug_view < 20.5;
    out_history = vec4(temporal_debug ? resolved : hdr_expand(resolved), 1.0);
}
