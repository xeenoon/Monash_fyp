#ifndef BLOOM_COMMON_GLSL
#define BLOOM_COMMON_GLSL

/* Shared by bloom_downsample.comp and bloom_upsample.comp.
 *
 * These two are the only shaders in the renderer that bind nothing but their
 * own set: one source image, one destination mip, and the exposure the tone
 * mapper is going to use. They deliberately do NOT include common.glsl -- they
 * need no camera, no lights and no shadows, and keeping the frame set out of
 * the layout is what lets the same two pipelines run over every mip with one
 * descriptor set each. */

layout(set = 0, binding = 0) uniform sampler2D bloom_source;
layout(set = 0, binding = 1, rgba16f) uniform image2D bloom_destination;
layout(std430, set = 0, binding = 2) readonly buffer ExposureState {
    float exposure;
    float average_luminance;
    uint sample_count;
    uint padding;
    uint bins[256];
} exposure_state;

layout(push_constant) uniform BloomData {
    /* x: mip to sample from the source image, y: 1 on the prefilter level,
       z: threshold in exposed units, w: soft-knee width in the same units. */
    vec4 filter_parameters;
    /* x: upsample tent radius in source texels, y: how much of the coarser
       level to add into the finer one, zw: reserved. */
    vec4 blend_parameters;
} bloom_push;

#define bloom_source_lod   bloom_push.filter_parameters.x
#define bloom_is_prefilter (bloom_push.filter_parameters.y > 0.5)
#define bloom_threshold    bloom_push.filter_parameters.z
#define bloom_knee         bloom_push.filter_parameters.w
#define bloom_radius       bloom_push.blend_parameters.x
#define bloom_weight       bloom_push.blend_parameters.y

/* Rec. 709 luma. Used for both the Karis weighting and the threshold, so both
   judge "bright" the way the eye does rather than by the largest channel --
   which would let a saturated red flame through at a third of its apparent
   brightness. */
float bloom_luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

/* Quadratic soft knee (the Unreal/Unity curve): below `threshold - knee`
   nothing blooms, above `threshold + knee` the full excess does, and between
   the two the response is smooth. A hard step there makes the bloom boundary
   crawl over surfaces as the auto-exposure adapts.

   The threshold is in EXPOSED units, not scene radiance, which is why this
   reads the exposure buffer. A fixed radiance threshold cannot work in a
   renderer whose exposure ranges from open sky to a torch-lit dungeon: it
   would bloom everything outdoors and nothing indoors. */
vec3 bloom_prefilter(vec3 color) {
    float luma = bloom_luminance(color) * exposure_state.exposure;
    float knee = max(bloom_knee, 1e-4);
    float soft = clamp(luma - bloom_threshold + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee);
    float contribution = max(soft, luma - bloom_threshold) / max(luma, 1e-4);
    return color * contribution;
}

#endif
