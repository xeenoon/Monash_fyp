#version 450
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec3 normal;

void main() {
    vec4 camera_relative = frame.local_to_camera_relative * vec4(in_position, 1.0);
    gl_Position = frame.view_projection * camera_relative;
    normal = in_normal;
}
