#version 450
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texcoord;
layout(location = 3) in float in_untextured;

layout(location = 0) out vec2 texcoord;
layout(location = 1) out float untextured;

void main() {
    vec4 camera_relative = frame.local_to_camera_relative * vec4(in_position, 1.0);
    gl_Position = frame.view_projection * camera_relative;
    texcoord = in_texcoord;
    untextured = in_untextured;
}
