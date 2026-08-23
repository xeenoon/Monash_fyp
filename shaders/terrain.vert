#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texcoord;
layout(location = 3) in vec4 in_tangent;
layout(location = 0) out vec2 uv;
layout(location = 1) out vec3 normal;
layout(location = 2) out vec4 tangent;
layout(location = 3) out vec3 camera_relative_position;
layout(location = 4) out vec4 current_clip;
layout(location = 5) out vec4 previous_clip;

layout(push_constant) uniform DrawData {
    mat4 local_to_camera_relative;
    vec4 geometry;
    vec4 elevation_uv;
    vec4 imagery_uv;
    vec4 debug;
} draw;

void main() {
    vec4 camera_relative = draw.local_to_camera_relative * vec4(in_position, 1.0);
    mat3 rotation = mat3(draw.local_to_camera_relative);
    uv = in_texcoord;
    normal = normalize(rotation * in_normal);
    tangent = vec4(normalize(rotation * in_tangent.xyz), in_tangent.w);
    camera_relative_position = camera_relative.xyz;
    current_clip = frame.view_projection * camera_relative;
    previous_clip = frame.previous_view_projection *
                    (frame.previous_local_to_camera_relative * camera_relative);
    gl_Position = current_clip;
}
