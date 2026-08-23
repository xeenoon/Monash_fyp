#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
layout(location = 0) in vec3 in_position;
layout(push_constant) uniform DrawData {
    mat4 local_to_camera_relative;
    vec4 geometry;
    vec4 elevation_uv;
    vec4 imagery_uv;
    vec4 debug;
} draw;
void main() {
    vec3 camera_relative = (draw.local_to_camera_relative * vec4(in_position, 1.0)).xyz;
    gl_Position = frame.shadow_view_projection[uint(draw.debug.x + 0.5)] *
                  vec4(camera_relative, 1.0);
}
