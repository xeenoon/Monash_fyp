#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location = 0) in vec3 in_position;
layout(location = 2) in vec2 in_texcoord;
layout(set = 1, binding = 1) uniform sampler2D elevation_map;

layout(push_constant) uniform DrawData {
    mat4 local_to_camera_relative;
    vec4 geometry;
    vec4 elevation_uv;
    vec4 imagery_uv;
    vec4 debug;
} draw;

void main() {
    vec3 local = in_position;
    /* Static materials store colour factors (0..1) in geometry.xy. Terrain
       stores metre spans, so this single pipeline can displace only terrain. */
    if (draw.geometry.x > 2.0 || draw.geometry.y > 2.0) {
        vec2 uv = in_texcoord * draw.elevation_uv.xy + draw.elevation_uv.zw;
        local = vec3(in_position.x * draw.geometry.x,
                     textureLod(elevation_map, uv, 0.0).r * draw.geometry.z,
                     in_position.z * draw.geometry.y);
    }
    vec3 camera_relative =
        (draw.local_to_camera_relative * vec4(local, 1.0)).xyz;
    gl_Position = frame.shadow_view_projection[uint(draw.debug.x + 0.5)] *
                  vec4(camera_relative, 1.0);
}
