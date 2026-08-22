#version 450
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) in vec2 texcoord;
layout(location = 1) in float untextured;
layout(location = 0) out vec4 out_color;

layout(set = 1, binding = 0) uniform sampler2D albedo;

void main() {
    const vec3 shell_gray = vec3(0.18);
    vec3 color = mix(texture(albedo, texcoord).rgb, shell_gray,
                     step(0.5, untextured));
    out_color = vec4(display_transform(color), 1.0);
}
