#version 450
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) in vec2 texcoord;
layout(location = 1) in float untextured;
layout(location = 0) out vec4 out_color;

layout(set = 1, binding = 0) uniform sampler2D albedo;

void main() {
    if (frame.depth_debug > 0.5) {
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
    out_color = vec4(display_transform(color), 1.0);
}
