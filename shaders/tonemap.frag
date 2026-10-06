#version 450
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"
#include "shader_dump.glsl"

layout(location = 0) in vec2 texcoord;
layout(location = 0) out vec4 out_color;
layout(set = 1, binding = 5) uniform sampler2D resolved_hdr;
layout(set = 1, binding = 7) uniform sampler2D bloom_pyramid;
/* CPU-painted UI canvas, straight alpha, already display-encoded. */
layout(set = 1, binding = 8) uniform sampler2D ui_overlay;
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

/* Rec. 709 luma of the bloom this pixel received, for the dump's f19. */
float bloom_luminance_dump(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

/* See SHADER_DUMP_LEGEND["tonemap"] in renderer.c for the f0..f19 layout. */
void main() {
    vec3 resolved = texture(resolved_hdr, texcoord).rgb;
    /* The bloom pyramid holds only the frame's above-threshold energy, so it
       is ADDED to the scene rather than blended with it: this is light that
       spread, not a wash over the top. Added before exposure because it was
       gathered in scene radiance, and the tone mapper is what decides how
       much of it survives -- which is why a bright flame's halo blows out and
       a dim one's stays a glow, with no extra logic here. */
    vec3 bloom = textureLod(bloom_pyramid, texcoord, 0.0).rgb * frame.bloom_parameters.z;
    resolved += bloom;
    bool debug_view = frame.debug_view > 0.5;
    vec3 display_linear = debug_view
        ? clamp(resolved, 0.0, 1.0)
        : aces_fitted(resolved * exposure_state.exposure);
    float dither = centred_dither(gl_FragCoord.xy, frame.time * 60.0);
    vec3 encoded = clamp(linear_to_srgb(display_linear) + dither, 0.0, 1.0);
    /* Fit the 16:9 canvas inside the window without stretching it. Past its
       edges, use the bottom-right texel: only full-screen fills (menu dims)
       ever cover it, so those carry out to the window edge and HUD panels
       anchored to the canvas edges do not smear into the bars. */
    vec2 screen_size = vec2(textureSize(resolved_hdr, 0));
    vec2 ui_size = vec2(textureSize(ui_overlay, 0));
    float fit = (screen_size.x / screen_size.y) / (ui_size.x / ui_size.y);
    vec2 ui_uv = texcoord - 0.5;
    if (fit > 1.0)
        ui_uv.x *= fit;
    else
        ui_uv.y /= fit;
    ui_uv += 0.5;
    bool inside_ui = all(greaterThanEqual(ui_uv, vec2(0.0))) && all(lessThanEqual(ui_uv, vec2(1.0)));
    vec4 ui = texture(ui_overlay, inside_ui ? ui_uv : vec2(1.0) - 0.5 / ui_size);
    encoded = mix(encoded, ui.rgb, ui.a);
    /* For an sRGB attachment, invert back to linear so the attachment's one
       hardware transfer recreates `encoded`. UNORM fallbacks receive it directly. */
    vec3 attachment_value = frame.temporal_parameters.z > 0.5
        ? srgb_to_linear(encoded) : encoded;
    out_color = vec4(attachment_value, 1.0);

    shader_dump(DUMP_SHADER_TONEMAP,
                vec4(texcoord, exposure_state.exposure, dither),
                vec4(resolved, exposure_state.average_luminance),
                vec4(display_linear, debug_view ? 1.0 : 0.0),
                vec4(encoded, 0.0),
                vec4(out_color.rgb, bloom_luminance_dump(bloom)));
}
