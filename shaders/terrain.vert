#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location = 0) in vec3 in_position;
layout(location = 2) in vec2 in_texcoord;
layout(location = 4) in float in_untextured;

layout(location = 0) out vec2 imagery_uv;
layout(location = 1) out float untextured;
layout(location = 2) out vec2 tile_uv;
layout(location = 3) out vec3 local_position;
layout(location = 4) out vec3 camera_relative_position;
layout(location = 5) out vec4 current_clip;
layout(location = 6) out vec4 previous_clip;

layout(set = 1, binding = 1) uniform sampler2D elevation_map;

layout(push_constant) uniform DrawData {
    mat4 local_to_camera_relative;
    vec4 geometry;      /* tile span X/Z, elevation range, skirt depth */
    vec4 elevation_uv;  /* scale and bias into the guttered height raster */
    vec4 imagery_uv;
    vec4 debug;
} draw;

void main() {
    vec2 height_uv = in_texcoord * draw.elevation_uv.xy + draw.elevation_uv.zw;
    float height_m = textureLod(elevation_map, height_uv, 0.0).r * draw.geometry.z;
    if (in_untextured > 1.5)
        height_m -= draw.geometry.w;

    vec3 local = vec3(in_position.x * draw.geometry.x,
                      height_m,
                      in_position.z * draw.geometry.y);
    vec4 camera_relative = draw.local_to_camera_relative * vec4(local, 1.0);

    imagery_uv = in_texcoord * draw.imagery_uv.xy + draw.imagery_uv.zw;
    untextured = in_untextured;
    tile_uv = in_texcoord;
    local_position = local;
    camera_relative_position = camera_relative.xyz;
    current_clip = frame.view_projection * camera_relative;
    previous_clip = frame.previous_view_projection *
                    (frame.previous_local_to_camera_relative * camera_relative);
    gl_Position = current_clip;
}
