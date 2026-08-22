/* Shared shader definitions. Every shader that includes this file is rebuilt
   when it changes (tracked via glslc -MD depfiles in CMakeLists.txt). */
#ifndef COMMON_GLSL
#define COMMON_GLSL

/* Descriptor set 0: per-frame data. Matches FrameUniforms in renderer.h
   (std140: every member vec4-aligned, tail padded). */
layout(set = 0, binding = 0) uniform FrameUniforms {
    mat4  projection;
    mat4  view;
    mat4  view_projection;
    mat4  inverse_view_projection;
    mat4  previous_projection;
    mat4  previous_view;
    mat4  previous_view_projection;
    mat4  local_to_camera_relative;
    mat4  previous_local_to_camera_relative;
    vec4  sun_direction;
    float time;
    float near_plane;
    float debug_view;
    float relight_strength;
} frame;

/* Infinite reversed-Z helpers. UV follows Vulkan framebuffer orientation, so
   the projection's single Y flip is not repeated here. The reconstructed
   position is camera-relative by design; absolute world positions stay double
   precision on the CPU. */
float linear_view_depth(float depth) {
    return frame.near_plane / max(depth, 1e-20);
}

vec3 reconstruct_camera_relative(vec2 uv, float depth) {
    vec4 clip = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 world = frame.inverse_view_projection * clip;
    return world.xyz / world.w;
}

/* Placeholder display transform. The real tone-map/display conversion arrives
   in a later phase; kept here so colour handling has one shared home. */
vec3 display_transform(vec3 color) {
    return color;
}

#endif
