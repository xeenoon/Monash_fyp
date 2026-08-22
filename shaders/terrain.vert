#version 450
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texcoord;
layout(location = 3) in float in_untextured;

layout(location = 0) out vec2 texcoord;
layout(location = 1) out float untextured;

layout(set = 1, binding = 1) uniform sampler2D elevation;

layout(push_constant) uniform DrawData {
    mat4 local_to_camera_relative;
    vec4 geometry;
    vec4 elevation_uv;
    vec4 imagery_uv;
    vec4 debug;
} draw;

void main() {
    vec2 height_uv = in_texcoord * draw.elevation_uv.xy + draw.elevation_uv.zw;
    float height_m = textureLod(elevation, height_uv, 0.0).r * draw.geometry.z;
    if (in_untextured > 1.5)
        height_m -= draw.geometry.w;
    vec3 local = vec3(in_position.x * draw.geometry.x,
                      height_m,
                      in_position.z * draw.geometry.y);
    vec4 camera_relative = draw.local_to_camera_relative * vec4(local, 1.0);
    gl_Position = frame.view_projection * camera_relative;
    texcoord = in_texcoord * draw.imagery_uv.xy + draw.imagery_uv.zw;
    untextured = in_untextured;
}
