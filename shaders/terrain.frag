#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "shadow_filter.glsl"
#include "pbr_common.glsl"
#include "environment_lighting.glsl"
#include "shader_dump.glsl"

/* Minimal terrain material path: the streamed image is deliberately only a
   low-frequency Alpine colour field. A later stone/grass layer should produce
   detail_albedo first, then multiply it by macro_tint in terrain_base_color(). */
layout(early_fragment_tests) in;

layout(location = 0) in vec2 imagery_uv;
layout(location = 1) in float untextured;
layout(location = 2) in vec2 tile_uv;
layout(location = 3) in vec3 local_position;
layout(location = 4) in vec3 camera_relative_position;
layout(location = 5) in vec4 current_clip;
layout(location = 6) in vec4 previous_clip;
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 out_motion;

layout(set = 1, binding = 0) uniform sampler2D macro_color_map;
layout(set = 1, binding = 1) uniform sampler2D elevation_map;
/* Binding 2 remains the shared terrain-detail slot for the future close-range
   stone/grass material. It is intentionally not sampled by this macro pass. */
layout(set = 1, binding = 2) uniform sampler2D material_detail_map;

layout(push_constant) uniform DrawData {
    mat4 local_to_camera_relative;
    vec4 geometry;
    vec4 elevation_uv;
    vec4 imagery_uv_transform;
    vec4 debug;
} draw;

struct HeightSurface {
    vec3 local_normal;
    vec2 gradient;
};

float height_at(vec2 uv) {
    return textureLod(elevation_map, uv, 0.0).r * draw.geometry.z;
}

HeightSurface height_surface() {
    vec2 size = vec2(textureSize(elevation_map, 0));
    vec2 uv = tile_uv * draw.elevation_uv.xy + draw.elevation_uv.zw;
    vec2 texel = 1.0 / size;
    float left  = height_at(uv - vec2(texel.x, 0.0));
    float right = height_at(uv + vec2(texel.x, 0.0));
    float down  = height_at(uv - vec2(0.0, texel.y));
    float up    = height_at(uv + vec2(0.0, texel.y));
    vec2 metres = draw.geometry.xy / (draw.elevation_uv.xy * size);

    HeightSurface result;
    result.gradient = vec2((right - left) / max(2.0 * metres.x, 1e-5),
                           (up - down) / max(2.0 * metres.y, 1e-5));
    result.local_normal = normalize(vec3(-result.gradient.x, 1.0,
                                         -result.gradient.y));

    /* Once one fragment spans many height samples, the rasterized geometric
       derivative is both cheaper and more representative than a fine central
       difference. */
    float footprint = max(length(dFdxCoarse(uv) * size),
                          length(dFdyCoarse(uv) * size));
    if (footprint > 8.0) {
        result.local_normal = normalize(cross(dFdyCoarse(local_position),
                                              dFdxCoarse(local_position)));
        if (result.local_normal.y < 0.0)
            result.local_normal = -result.local_normal;
    }
    return result;
}

vec3 lod_color(float level) {
    const vec3 colors[6] = vec3[6](
        vec3(0.90, 0.20, 0.20), vec3(0.20, 0.75, 0.25),
        vec3(0.20, 0.45, 0.95), vec3(0.95, 0.75, 0.15),
        vec3(0.75, 0.25, 0.90), vec3(0.15, 0.85, 0.85));
    return colors[int(abs(level)) % 6];
}

vec3 terrain_base_color(vec3 macro_tint) {
    /* Placeholder for the close-range material layer:
         vec3 detail_albedo = blend_stone_and_grass(...);
         return detail_albedo * macro_tint / neutral_macro;
       Until those authored textures arrive, show the macro field directly. */
    return macro_tint;
}

void main() {
    vec2 current_uv = current_clip.xy / current_clip.w * 0.5 + 0.5;
    vec2 previous_uv = previous_clip.xy / previous_clip.w * 0.5 + 0.5;
    out_motion = previous_uv - current_uv;

    if (frame.debug_view > 0.5 && frame.debug_view < 1.5) {
        float metres = linear_view_depth(gl_FragCoord.z);
        float value = clamp(log2(1.0 + metres) / log2(1.0 + 1.0e7), 0.0, 1.0);
        out_color = vec4(vec3(value), 1.0);
        return;
    }

    HeightSurface surface = height_surface();
    vec3 geometric_normal = normalize(mat3(draw.local_to_camera_relative) *
                                      surface.local_normal);
    vec3 V = normalize(-camera_relative_position);
    if (dot(geometric_normal, V) < 0.0)
        geometric_normal = -geometric_normal;

    vec3 macro_tint = textureGrad(macro_color_map, imagery_uv,
                                  dFdxCoarse(imagery_uv),
                                  dFdyCoarse(imagery_uv)).rgb;
    vec3 base_color = terrain_base_color(macro_tint);
    base_color = mix(base_color, vec3(0.18), step(0.5, untextured));

    const float roughness = 0.82;
    const float metallic = 0.0;
    const float ao = 1.0;
    vec3 L = normalize(-frame.sun_direction.xyz);
    float NoL = max(dot(geometric_normal, L), 0.0);
    vec3 F0 = vec3(0.04);
    UeDefaultLit bxdf = ue_default_lit_bxdf(
        base_color, F0, roughness, geometric_normal, V, L);
    ShadowResult shadow = shadow_evaluate(camera_relative_position,
                                          geometric_normal);

    if (frame.debug_view > 1.5 && frame.debug_view < 2.5) {
        out_color = vec4(mix(base_color, lod_color(draw.debug.x), 0.72), 1.0);
        return;
    }
    if (frame.debug_view > 2.5 && frame.debug_view < 3.5) {
        out_color = vec4(geometric_normal * 0.5 + 0.5, 1.0);
        return;
    }
    if (frame.debug_view > 3.5 && frame.debug_view < 4.5) {
        out_color = vec4(vec3(roughness), 1.0);
        return;
    }
    if (frame.debug_view > 4.5 && frame.debug_view < 5.5) {
        out_color = vec4(macro_tint, 1.0);
        return;
    }
    if (frame.debug_view > 6.5 && frame.debug_view < 7.5) {
        const vec3 colors[5] = vec3[5](vec3(.95,.18,.12), vec3(.18,.82,.25),
            vec3(.15,.45,1), vec3(.95,.75,.1), vec3(.1));
        out_color = vec4(colors[shadow.cascade], 1.0);
        return;
    }
    if (frame.debug_view > 7.5 && frame.debug_view < 8.5) {
        out_color = vec4(shadow.coordinate, 1.0); return;
    }
    if (frame.debug_view > 8.5 && frame.debug_view < 9.5) {
        out_color = vec4(vec3(shadow.visibility), 1.0); return;
    }
    if (frame.debug_view > 9.5 && frame.debug_view < 10.5) {
        out_color = vec4(vec3(clamp(shadow.receiver_bias /
            max(frame.shadow_parameters.x, 1e-5), 0.0, 1.0)), 1.0); return;
    }
    if (frame.debug_view > 10.5 && frame.debug_view < 11.5) {
        float depth = shadow.cascade < 4u ? texture(shadow_map_raw,
            vec3(shadow.coordinate.xy, float(shadow.cascade))).r : 1.0;
        out_color = vec4(vec3(depth), 1.0); return;
    }

    vec3 direct = (bxdf.diffuse + bxdf.specular) * frame.sun_radiance.rgb *
                  NoL * shadow.visibility;
    EnvironmentLightingResult environment = environment_evaluate(
        camera_relative_position, geometric_normal, V, roughness, F0, ao,
        true, true);
    vec3 lit_color = direct + base_color * environment.irradiance * ao +
                     environment.final_specular;
    vec3 final_color = mix(base_color, lit_color,
                           clamp(frame.relight_strength, 0.0, 1.0));
    out_color = vec4(final_color, 1.0);

    shader_dump(DUMP_SHADER_TERRAIN,
                vec4(imagery_uv, roughness, NoL),
                vec4(base_color, shadow.visibility),
                vec4(geometric_normal, metallic),
                vec4(camera_relative_position, frame.relight_strength),
                vec4(environment.irradiance, ao));
    shader_dump(DUMP_SHADER_ENVIRONMENT_IBL,
                vec4(environment.reflection_direction, environment.mip),
                vec4(environment.sampled_reflection_radiance, environment.NoV),
                vec4(environment.ggx_specular_energy, environment.reflection_visibility),
                vec4(environment.unoccluded_specular, ao),
                vec4(environment.final_specular, roughness));
}
