/* Shared shader definitions. Every shader that includes this file is rebuilt
   when it changes (tracked via glslc -MD depfiles in CMakeLists.txt). */
#ifndef COMMON_GLSL
#define COMMON_GLSL

/* Descriptor set 0: per-frame data. Matches FrameUniforms in renderer.h
   (std140: every member vec4-aligned, tail padded). */
layout(set = 0, binding = 0) uniform FrameUniforms {
    mat4  view_projection;
    vec4  camera_position;
    vec4  sun_direction;
    float time;
} frame;

/* Placeholder display transform. The real tone-map/display conversion arrives
   in a later phase; kept here so colour handling has one shared home. */
vec3 display_transform(vec3 color) {
    return color;
}

#endif
