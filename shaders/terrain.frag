#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "shadow_filter.glsl"
#include "pbr_common.glsl"
#include "environment_lighting.glsl"
#include "shader_dump.glsl"

/* The streamed image is a low-frequency, water-free Alpine material field:
   RGB is cross-sampled macro colour and alpha is its grass-vs-rock coverage.
   Fifteen authored scans supply sub-metre detail without replacing RGB. */
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
layout(set = 1, binding = 2) uniform sampler2D micro_albedo_atlas;
layout(set = 1, binding = 3) uniform sampler2D micro_normal_atlas;
layout(set = 1, binding = 4) uniform sampler2D micro_ormh_atlas;
layout(set = 1, binding = 5) uniform sampler2D parent_macro_color_map;

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

uint material_hash(ivec2 cell) {
    uvec2 value = uvec2(cell);
    uint hash = value.x * 0x8da6b343u ^ value.y * 0xd8163841u;
    hash ^= hash >> 13u;
    hash *= 0xcb1ab31fu;
    return hash ^ (hash >> 16u);
}

uint material_index(ivec2 cell, uint first, uint count) {
    /* The detail coordinate repeats every phase_period_m (terrain_runtime.c),
       currently 4096 m / 32 m-per-cell = 128 cells. Wrap the hashed cell to
       match, otherwise the selected material jumps when that coordinate
       wraps. Kept small (not the exact LCM of every scan width) so the CPU
       phase stays float32-precise near the camera -- see the comment on
       phase_period_m. */
    const int material_region_period = 128;
    cell = ivec2((cell.x % material_region_period + material_region_period) % material_region_period,
                 (cell.y % material_region_period + material_region_period) % material_region_period);
    return first + material_hash(cell) % count;
}

float material_width_decimetres(uint index) {
const float widths[15] = float[15](
    60.0, 60.0, 60.0, 80.0, 80.0,
    172.0, 228.0, 160.0, 200.0,
    56.0, 56.0, 56.0, 80.0, 40.0,
    80.0
);
    return widths[index];
}

vec2 material_atlas_uv(uint index, vec2 tiled_uv) {
    vec2 atlas_size = vec2(textureSize(micro_albedo_atlas, 0));
    vec2 cell_size = atlas_size / 4.0;
    vec2 gutter = vec2(8.0) / cell_size;
    vec2 interior = mix(gutter, vec2(1.0) - gutter, fract(tiled_uv));
    vec2 cell = vec2(float(index % 4u), float(index / 4u));
    return (cell + interior) / 4.0;
}

struct MicroMaterial {
    float luminance_factor;
    vec3 tangent_normal;
    float ao;
    float roughness;
    float height;
};

MicroMaterial sample_micro(uint index, vec2 world_xz) {
    float width_dm = material_width_decimetres(index);
    /* Keep the texture coordinate derivative unwrapped, but form the sampled
       fraction in decimetres. DETAIL_PHASE_PERIOD_M is an exact common repeat
       of every scan width, so this avoids a float rounding seam at tile edges. */
    vec2 tiled_uv = world_xz * (10.0 / width_dm);
    vec2 tile_fraction = mod(world_xz * 10.0, width_dm) / width_dm;
    vec2 atlas_uv = material_atlas_uv(index, tile_fraction);
    /* The per-cell wrap gutter occupies 8/528 of a cell. Scale derivatives to
       the atlas interior so implicit filtering never sees a 3x3 cell jump. */
    float derivative_scale = (512.0 / 528.0) / 4.0;
    vec2 dx = dFdxCoarse(tiled_uv) * derivative_scale;
    vec2 dy = dFdyCoarse(tiled_uv) * derivative_scale;
    /* derivative_scale only maps up to one tile repeat per fragment into the
       cell interior. At grazing/distant view a fragment's footprint can span
       several repeats of a narrow (down to 4 m) scan, pushing dx/dy past the
       cell boundary; textureGrad's implicit mip then bleeds neighbouring
       atlas cells' unrelated materials into the sample -- occasional
       fragments read a wildly different normal/roughness than their
       neighbours, seen as a dark crosshatch. Clamp the footprint to the cell
       interior so mip selection never leaves this material's cell; beyond
       that distance the pattern aliases instead of blending in other scans. */
    float dx_len = length(dx);
    float dy_len = length(dy);
    /* footprint_ratio = tile repeats covered by one fragment. Below 1 the sample
       is trustworthy; above 1 we clamp the derivative to keep mip selection
       inside this cell, but the atlas only has 4 mip levels and 16x anisotropy,
       so the clamped-yet-cell-wide footprint still lets aniso taps reach across
       the 8px gutter into the neighbouring (or empty black 16th) cell. That
       bleed reads a steep, unrelated tangent normal on scattered fragments and
       shades them nearly sunless -- the black speckle/crosshatch. */
    float footprint_ratio = max(dx_len, dy_len) / derivative_scale;
    if (dx_len > derivative_scale)
        dx *= derivative_scale / dx_len;
    if (dy_len > derivative_scale)
        dy *= derivative_scale / dy_len;
    vec3 detail = textureGrad(micro_albedo_atlas, atlas_uv, dx, dy).rgb;
    vec3 normal = textureGrad(micro_normal_atlas, atlas_uv, dx, dy).xyz * 2.0 - 1.0;
    vec4 ormh = textureGrad(micro_ormh_atlas, atlas_uv, dx, dy);
    /* Once the footprint exceeds one repeat the tangent normal is unreliable
       (possibly bled from another cell), so fade its lateral tilt toward flat.
       The sample then contributes no spurious dark facet; the surface keeps its
       smooth geometric normal where the scan can no longer be resolved. */
    float detail_confidence = 1.0 - smoothstep(0.7, 1.6, footprint_ratio);
    normal.xy *= detail_confidence;
    MicroMaterial result;
    /* Bled cells also drag albedo dark; fade the high-pass luminance to neutral
       (1.0, an identity multiplier on macro_tint) with the same confidence. */
    result.luminance_factor = mix(1.0, detail.r * 2.0, detail_confidence);
    result.tangent_normal = normalize(normal);
    result.ao = ormh.r;
    result.roughness = ormh.g;
    result.height = ormh.a;
    return result;
}

MicroMaterial blend_micro_bank(vec2 world_xz, uint first, uint count) {
    const float region_m = 32.0;
    vec2 region = world_xz / region_m;
    ivec2 base = ivec2(floor(region));
    vec2 blend = smoothstep(vec2(0.20), vec2(0.80), fract(region));
    float weights[4] = float[4]((1.0-blend.x)*(1.0-blend.y),
                                blend.x*(1.0-blend.y),
                                (1.0-blend.x)*blend.y, blend.x*blend.y);
    ivec2 offsets[4] = ivec2[4](ivec2(0,0), ivec2(1,0),
                                ivec2(0,1), ivec2(1,1));
    MicroMaterial result = MicroMaterial(0.0, vec3(0.0), 0.0, 0.0, 0.0);
    for (int i = 0; i < 4; ++i) {
        MicroMaterial sample_value = sample_micro(
            material_index(base + offsets[i], first, count), world_xz);
        result.luminance_factor += sample_value.luminance_factor * weights[i];
        result.tangent_normal += sample_value.tangent_normal * weights[i];
        result.ao += sample_value.ao * weights[i];
        result.roughness += sample_value.roughness * weights[i];
        result.height += sample_value.height * weights[i];
    }
    result.tangent_normal = normalize(result.tangent_normal);
    return result;
}

MicroMaterial mix_micro(MicroMaterial rock, MicroMaterial grass,
                        float grass_weight);

MicroMaterial classified_micro(vec2 world_xz, float grass_weight) {
    if (grass_weight < 0.015)
        return blend_micro_bank(world_xz, 0u, 9u);
    if (grass_weight > 0.985 && false)
        return blend_micro_bank(world_xz, 9u, 6u);
    MicroMaterial rock = blend_micro_bank(world_xz, 0u, 9u);
    MicroMaterial grass = blend_micro_bank(world_xz, 9u, 6u);
    return mix_micro(rock, grass, grass_weight);
}

MicroMaterial mix_micro(MicroMaterial rock, MicroMaterial grass,
                        float grass_weight) {
    MicroMaterial result;
    result.luminance_factor = mix(rock.luminance_factor,
                                  grass.luminance_factor, grass_weight);
    result.tangent_normal = normalize(mix(rock.tangent_normal,
                                          grass.tangent_normal, grass_weight));
    result.ao = mix(rock.ao, grass.ao, grass_weight);
    result.roughness = mix(rock.roughness, grass.roughness, grass_weight);
    result.height = mix(rock.height, grass.height, grass_weight);
    return result;
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
    vec4 child_macro = textureGrad(macro_color_map, imagery_uv,
                                   dFdxCoarse(imagery_uv),
                                   dFdyCoarse(imagery_uv));
    float parent_quadrant = floor(draw.debug.z);
    vec2 parent_tile_uv = tile_uv * 0.5 + 0.5 * vec2(
        mod(parent_quadrant, 2.0), floor(parent_quadrant * 0.5));
    vec2 parent_imagery_uv = parent_tile_uv * draw.imagery_uv_transform.xy +
                             draw.imagery_uv_transform.zw;
    vec4 parent_macro = textureGrad(parent_macro_color_map, parent_imagery_uv,
                                    dFdxCoarse(parent_imagery_uv),
                                    dFdyCoarse(parent_imagery_uv));
    float lod_fade = clamp(fract(draw.debug.z) * 8.0, 0.0, 1.0);
    vec4 macro_sample = mix(parent_macro, child_macro, lod_fade);
    vec3 macro_tint = macro_sample.rgb;
    /* Golden synthesis owns RGB. Alpha is the separately composed Alpine
       rock/grass classification, so material selection remains water-free
       without feeding the blurred classifier colour into the result. */
    float grass_weight = smoothstep(0.08, 0.55, macro_sample.a);
    float steepness = smoothstep(0.42, 0.78, 1.0 - surface.local_normal.y);
    grass_weight *= 1.0 - 0.40 * steepness;
    float rock_weight = 1.0 - grass_weight;

    /* draw.debug.yw is the tile's world-space origin reduced modulo
       phase_period_m (4096 m, terrain_runtime.c) in double on the CPU, so it
       stays small enough here to keep full float32 precision once added to
       the small tile-local coordinate below. */
    vec2 world_phase = local_position.xz + draw.debug.yw;
    /* Golden already supplies macro and meso structure. The authored scans are
       a deliberately subordinate, true-scale PBR layer. */
    MicroMaterial micro = classified_micro(world_phase, grass_weight);
    float view_distance = length(camera_relative_position);
    float detail_fade = 1.0 - smoothstep(4000.0, 12000.0, view_distance);
    const float detail_weight = 0.65;
    float weighted_detail = detail_weight * detail_fade;
    /* Scan normals are tangent-space data. Reorient them around the terrain
       normal; treating their Z component as world-up made steep close-range
       cells falsely face the sun. Keep the original flat-ground orientation
       (tangent +X, bitangent +Z) for visual continuity. */
    vec3 tangent_reference = abs(surface.local_normal.x) < 0.95
        ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 0.0, 1.0);
    vec3 tangent = normalize(tangent_reference - surface.local_normal *
                             dot(tangent_reference, surface.local_normal));
    vec3 bitangent = normalize(cross(tangent, surface.local_normal));
    vec3 micro_local_normal = normalize(tangent * micro.tangent_normal.x +
                                        bitangent * micro.tangent_normal.y +
                                        surface.local_normal * micro.tangent_normal.z);
    vec3 material_local_normal = normalize(mix(surface.local_normal,
                                                micro_local_normal,
                                                weighted_detail));
    vec3 geometric_normal = normalize(mat3(draw.local_to_camera_relative) *
                                      material_local_normal);
    vec3 V = normalize(-camera_relative_position);
    if (dot(geometric_normal, V) < 0.0)
        geometric_normal = -geometric_normal;

    /* Neutral high-pass luminance cannot replace golden chroma and is capped
       at the same 20% priority as the other authored scan channels. */
    vec3 base_color = macro_tint * mix(1.0, micro.luminance_factor,
                                       weighted_detail);
    /* Skirts (untextured > 0.5) are crack-fillers extruded straight down from a
       tile edge. Their vertices are copies of the edge surface vertex, so they
       carry the same imagery/tile UVs and world phase and therefore sample the
       exact edge material above -- letting them texture normally makes the wall
       blend into the terrain. Previously they were forced to flat grey (0.18),
       which painted a dark untextured stripe wherever a skirt showed through at
       an LOD crack. Do NOT reintroduce that override. */

    float roughness = mix(0.82, micro.roughness, weighted_detail);
    const float metallic = 0.0;
    float ao = mix(1.0, micro.ao, weighted_detail);
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
    if (frame.debug_view > 5.5 && frame.debug_view < 6.5) {
        out_color = vec4(rock_weight, grass_weight, 0.0, 1.0);
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
                vec4(macro_tint, micro.luminance_factor),
                vec4(surface.local_normal, weighted_detail),
                vec4(micro.tangent_normal, grass_weight),
                vec4(base_color, roughness),
                vec4(final_color, NoL));
    shader_dump(DUMP_SHADER_ENVIRONMENT_IBL,
                vec4(environment.reflection_direction, environment.mip),
                vec4(environment.sampled_reflection_radiance, environment.NoV),
                vec4(environment.ggx_specular_energy, environment.reflection_visibility),
                vec4(environment.unoccluded_specular, ao),
                vec4(environment.final_specular, roughness));
}
