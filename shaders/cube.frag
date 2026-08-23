#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "shader_dump.glsl"

/* Opaque, single-pass shading with no discard/gl_FragDepth write: safe to
   resolve the depth test before running the fragment shader, which also
   collapses shader_dump() records toward ~1/pixel instead of counting every
   overdrawn fragment. */
layout(early_fragment_tests) in;

layout(location = 0) in vec3 normal;
layout(location = 0) out vec4 out_color;

void main() {
    vec3 light_direction = normalize(vec3(0.6, 1.0, 0.8));
    float diffuse = max(dot(normalize(normal), light_direction), 0.0);
    float lighting = 0.28 + 0.72 * diffuse;
    out_color = vec4(vec3(0.68) * lighting, 1.0);

    /* See SHADER_DUMP_LEGEND["cube"] in renderer.c for the f0..f19 layout. */
    shader_dump(DUMP_SHADER_CUBE,
                vec4(normalize(normal), diffuse),
                vec4(lighting, out_color.rgb),
                vec4(0.0), vec4(0.0), vec4(0.0));
}
