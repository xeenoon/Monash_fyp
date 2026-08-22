#version 450
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) in vec2 texcoord;
layout(location = 0) out vec4 out_color;
layout(set = 1, binding = 0) uniform sampler2D hdr_scene;

/* Direct GLSL port of Wicked Engine tonemapCS.hlsl::RRTAndODTFit/ACESFitted
   (Wicked Engine contributors, MIT). We use its fixed-exposure ACES path but
   leave bloom, grading, and adaptive luminance for their owning phases. */
vec3 rrt_and_odt_fit(vec3 value) {
    vec3 a = value * (value + 0.0245786) - 0.000090537;
    vec3 b = value * (0.983729 * value + 0.4329510) + 0.238081;
    return a / b;
}

vec3 aces_fitted(vec3 color) {
    const mat3 input_matrix = mat3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777);
    const mat3 output_matrix = mat3(
         1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602);
    color = input_matrix * color;
    color = rrt_and_odt_fit(clamp(color, 0.0, 100.0));
    return clamp(output_matrix * color, 0.0, 1.0);
}

void main() {
    vec3 source = texture(hdr_scene, texcoord).rgb;
    if (frame.debug_view > 0.5) {
        out_color = vec4(clamp(source, 0.0, 1.0), 1.0);
        return;
    }
    vec3 hdr = source * frame.shadow_parameters.z;
    /* The swapchain is B8G8R8A8_SRGB, so Vulkan applies the output transfer
       curve exactly once after this linear tone-mapped value is written. */
    out_color = vec4(aces_fitted(hdr), 1.0);
}
