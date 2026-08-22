#version 450
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) in vec2 texcoord;
layout(location = 1) in float untextured;
layout(location = 0) out vec4 out_color;

layout(set = 1, binding = 0) uniform sampler2D albedo;

layout(push_constant) uniform DrawData {
    mat4 local_to_camera_relative;
    vec4 geometry;
    vec4 elevation_uv;
    vec4 imagery_uv;
    vec4 debug;
} draw;

vec3 lod_color(float level) {
    const vec3 colors[6] = vec3[6](
        vec3(0.90, 0.20, 0.20), vec3(0.20, 0.75, 0.25),
        vec3(0.20, 0.45, 0.95), vec3(0.95, 0.75, 0.15),
        vec3(0.75, 0.25, 0.90), vec3(0.15, 0.85, 0.85));
    return colors[int(level) % 6];
}

void main() {
    if (frame.depth_debug > 0.5 && frame.depth_debug < 1.5) {
        /* Logarithmic display spans the near plane through planetary distance;
           black is near and white is 10,000 km. */
        float metres = linear_view_depth(gl_FragCoord.z);
        float value = clamp(log2(1.0 + metres) / log2(1.0 + 1.0e7), 0.0, 1.0);
        out_color = vec4(vec3(value), 1.0);
        return;
    }

    const vec3 shell_gray = vec3(0.18);
    vec3 color = mix(texture(albedo, texcoord).rgb, shell_gray,
                     step(0.5, untextured));
    if (frame.depth_debug > 1.5)
        color = mix(color, lod_color(draw.debug.x), 0.72);
    out_color = vec4(display_transform(color), 1.0);
}
