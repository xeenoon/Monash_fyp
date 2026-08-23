#version 450
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) in vec2 texcoord;
layout(location = 0) out vec4 out_color;
layout(set = 1, binding = 5) uniform sampler2D resolved_hdr;
layout(std430, set = 1, binding = 6) readonly buffer ExposureState {
    float exposure;
    float average_luminance;
    uint sample_count;
    uint padding;
    uint bins[256];
} exposure_state;

/* Wicked Engine's compact RRT/ODT fit (MIT). This is deliberately named
   ACES fitted: it is a useful filmic approximation, not the full ACES system. */
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

float centred_dither(vec2 pixel, float frame_number) {
    float noise = fract(52.9829189 * fract(dot(pixel + frame_number,
                                               vec2(0.06711056, 0.00583715))));
    return (noise - 0.5) / 255.0;
}

vec3 linear_to_srgb(vec3 value) {
    vec3 low = value * 12.92;
    vec3 high = 1.055 * pow(max(value, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(low, high, greaterThan(value, vec3(0.0031308)));
}

vec3 srgb_to_linear(vec3 value) {
    vec3 low = value / 12.92;
    vec3 high = pow((max(value, vec3(0.0)) + 0.055) / 1.055, vec3(2.4));
    return mix(low, high, greaterThan(value, vec3(0.04045)));
}

void main() {
    vec3 resolved = texture(resolved_hdr, texcoord).rgb;
    bool debug_view = frame.debug_view > 0.5;
    vec3 display_linear = debug_view
        ? clamp(resolved, 0.0, 1.0)
        : aces_fitted(resolved * exposure_state.exposure);
    float dither = centred_dither(gl_FragCoord.xy, frame.time * 60.0);
    vec3 encoded = clamp(linear_to_srgb(display_linear) + dither, 0.0, 1.0);
    /* For an sRGB attachment, invert back to linear so the attachment's one
       hardware transfer recreates `encoded`. UNORM fallbacks receive it directly. */
    vec3 attachment_value = frame.temporal_parameters.z > 0.5
        ? srgb_to_linear(encoded) : encoded;
    out_color = vec4(attachment_value, 1.0);
}
