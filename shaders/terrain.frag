#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "shadow_filter.glsl"
#include "pbr_common.glsl"
#include "environment_lighting.glsl"
#include "shader_dump.glsl"

/* The streamed image is a low-frequency, water-free Alpine material field:
   RGB is cross-sampled macro colour and alpha is its grass-vs-rock coverage.
   Twenty-one authored rock/grass/snow scans supply material-matched macro and
   sub-metre detail without replacing RGB. */
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
    /* Pure DEM central-difference normal, before the geometric-derivative
       blend below mixes in a screen-space estimate for distant/grazing
       fragments. Slope-driven effects (top-down stretch compensation/overlay)
       key off this one: the geometric blend is stable for shading a huge
       rasterized triangle, but it is a screen-space artifact, not the actual
       terrain slope, and goes unreliable exactly at the distant/grazing
       viewing angles where it dominates -- keying stretch detection off it
       painted large stretches of ordinary distant terrain solid red. */
    vec3 analytic_normal;
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
    result.analytic_normal = result.local_normal;

    /* Once one fragment spans many height samples, the rasterized geometric
       derivative is both cheaper and more representative than a fine central
       difference. Blend between the two across a footprint range instead of a
       hard switch: a hard threshold on this per-quad footprint drew a dashed
       normal-discontinuity line along the iso-footprint contour (grazing
       ridges). */
    float footprint = max(length(dFdxCoarse(uv) * size),
                          length(dFdyCoarse(uv) * size));
    float geo_blend = smoothstep(6.0, 12.0, footprint);
    if (geo_blend > 0.0) {
        vec3 geo = normalize(cross(dFdyCoarse(local_position),
                                   dFdxCoarse(local_position)));
        if (geo.y < 0.0)
            geo = -geo;
        result.local_normal = normalize(mix(result.local_normal, geo, geo_blend));
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
const float widths[21] = float[21](
    60.0, 60.0, 60.0, 80.0, 80.0,
    172.0, 228.0, 160.0, 200.0,
    56.0, 56.0, 56.0, 80.0, 40.0,
    80.0,
    20.0, 20.0, 20.0, 20.0, 20.0, 20.0
);
    return widths[index];
}

vec2 material_atlas_uv(uint index, vec2 tiled_uv) {
    const float atlas_grid = 5.0;
    vec2 atlas_size = vec2(textureSize(micro_albedo_atlas, 0));
    vec2 cell_size = atlas_size / atlas_grid;
    vec2 gutter = vec2(8.0) / cell_size;
    vec2 interior = mix(gutter, vec2(1.0) - gutter, fract(tiled_uv));
    vec2 cell = vec2(float(index % 5u), float(index / 5u));
    return (cell + interior) / atlas_grid;
}

struct MicroMaterial {
    float luminance_factor;
    vec3 tangent_normal;
    float ao;
    float roughness;
    float height;
};

MicroMaterial sample_micro(uint index, vec2 projected_position) {
    float width_dm = material_width_decimetres(index);
    /* Keep the texture coordinate derivative unwrapped, but form the sampled
       fraction in decimetres. DETAIL_PHASE_PERIOD_M is an exact common repeat
       of every scan width, so this avoids a float rounding seam at tile edges. */
    vec2 tiled_uv = projected_position * (10.0 / width_dm);
    vec2 tile_fraction = mod(projected_position * 10.0, width_dm) / width_dm;
    vec2 atlas_uv = material_atlas_uv(index, tile_fraction);
    /* The per-cell wrap gutter occupies 8/528 of a cell. Scale derivatives to
       the atlas interior so implicit filtering never sees a 3x3 cell jump. */
    float derivative_scale = (512.0 / 528.0) / 5.0;
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
    /* Clamp the footprint so mip selection stays inside this cell (never bleeds
       the neighbouring or empty black 16th cell). The confidence fade that hides
       residual bleed/aliasing is applied ONCE in main() from a world-space
       footprint -- NOT per-material here: each material has a different width_dm,
       so a per-material footprint fade crosses its threshold in a patchy
       per-32m-cell way that reads as dashed lines along the iso-footprint
       contour. */
    if (dx_len > derivative_scale)
        dx *= derivative_scale / dx_len;
    if (dy_len > derivative_scale)
        dy *= derivative_scale / dy_len;
    vec3 normal = textureGrad(micro_normal_atlas, atlas_uv, dx, dy).xyz * 2.0 - 1.0;
    vec4 ormh = textureGrad(micro_ormh_atlas, atlas_uv, dx, dy);
    /* Cap the lateral tilt of the tangent normal. Residual atlas bleed at cell
       edges yields steep off-axis normals that shade isolated fragments sunless
       (the last speckle at grazing silhouettes); genuine subordinate scan detail
       stays well under this, so the cap is invisible on normal terrain. Applied
       per-fragment on the normal value, not on position/footprint, so it never
       forms a spatial contour/dashed line. */
    float lateral = length(normal.xy);
    normal.xy *= min(1.0, 0.5 / max(lateral, 1e-4));
    MicroMaterial result;
    result.luminance_factor = ormh.b * 2.0;
    result.tangent_normal = normalize(normal);
    result.ao = ormh.r;
    result.roughness = ormh.g;
    result.height = ormh.a;
    return result;
}

MicroMaterial blend_micro_bank(vec2 material_position_xz, vec2 projected_position,
                               uint first, uint count) {
    const float region_m = 32.0;
    vec2 region = material_position_xz / region_m;
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
            material_index(base + offsets[i], first, count), projected_position);
        result.luminance_factor += sample_value.luminance_factor * weights[i];
        result.tangent_normal += sample_value.tangent_normal * weights[i];
        result.ao += sample_value.ao * weights[i];
        result.roughness += sample_value.roughness * weights[i];
        result.height += sample_value.height * weights[i];
    }
    result.tangent_normal = normalize(result.tangent_normal);
    return result;
}

/* A second material read at twenty times the authored width supplies
   mesoscopic breakup above the true-scale PBR scan. Its albedo contributes
   only normalized luminance structure: geographic terrain RGB remains the
   sole colour source. A weak normal contribution creates formation-scale
   structure; AO and roughness stay true-scale so enlarged pores do not read
   as literal geometry. */
struct MaterialMacro {
    vec3 color;
    vec3 tangent_normal;
    float reference_luminance;
};

float material_macro_reference_luminance(uint index) {
    /* Raw linear-atlas means for the fixed neutral macro representatives used
       by classified_macro(). Other indices are retained for diagnostic/helper
       completeness but do not currently reach the final macro layer. */
    if (index == 7u) return 0.19218;
    if (index == 10u) return 0.06208;
    if (index == 16u) return 0.80287;
    return 0.18;
}

MaterialMacro sample_material_macro(uint index, vec2 projected_position) {
    const float macro_scale = 20.0;
    float width_dm = material_width_decimetres(index) * macro_scale;
    vec2 tiled_uv = projected_position * (10.0 / width_dm);
    vec2 tile_fraction = mod(projected_position * 10.0, width_dm) / width_dm;
    vec2 atlas_uv = material_atlas_uv(index, tile_fraction);
    float derivative_scale = (512.0 / 528.0) / 5.0;
    vec2 dx = dFdxCoarse(tiled_uv) * derivative_scale;
    vec2 dy = dFdyCoarse(tiled_uv) * derivative_scale;
    float dx_len = length(dx);
    float dy_len = length(dy);
    if (dx_len > derivative_scale)
        dx *= derivative_scale / dx_len;
    if (dy_len > derivative_scale)
        dy *= derivative_scale / dy_len;
    MaterialMacro result;
    result.color = textureGrad(micro_albedo_atlas, atlas_uv, dx, dy).rgb;
    result.tangent_normal = normalize(
        textureGrad(micro_normal_atlas, atlas_uv, dx, dy).xyz * 2.0 - 1.0);
    result.reference_luminance = material_macro_reference_luminance(index);
    return result;
}

MaterialMacro blend_macro_bank(vec2 material_position_xz,
                               vec2 projected_position,
                               uint first, uint count) {
    const float region_m = 128.0;
    vec2 region = material_position_xz / region_m;
    ivec2 base = ivec2(floor(region));
    vec2 blend = smoothstep(vec2(0.20), vec2(0.80), fract(region));
    float weights[4] = float[4]((1.0-blend.x)*(1.0-blend.y),
                                blend.x*(1.0-blend.y),
                                (1.0-blend.x)*blend.y, blend.x*blend.y);
    ivec2 offsets[4] = ivec2[4](ivec2(0,0), ivec2(1,0),
                                ivec2(0,1), ivec2(1,1));
    MaterialMacro result = MaterialMacro(vec3(0.0), vec3(0.0), 0.0);
    for (int i = 0; i < 4; ++i) {
        MaterialMacro sample_value = sample_material_macro(
            material_index(base + offsets[i], first, count),
            projected_position);
        result.color += sample_value.color * weights[i];
        result.tangent_normal += sample_value.tangent_normal * weights[i];
        result.reference_luminance += sample_value.reference_luminance * weights[i];
    }
    result.tangent_normal = normalize(result.tangent_normal);
    return result;
}

MaterialMacro mix_macro(MaterialMacro first, MaterialMacro second,
                        float weight) {
    return MaterialMacro(
        mix(first.color, second.color, weight),
        normalize(mix(first.tangent_normal, second.tangent_normal, weight)),
        mix(first.reference_luminance, second.reference_luminance, weight));
}

MicroMaterial mix_micro(MicroMaterial rock, MicroMaterial grass,
                        float grass_weight);

MicroMaterial classified_micro(vec2 material_position_xz,
                               vec2 projected_position,
                               float grass_weight,
                               float snow_weight) {
    if (snow_weight < 0.015) {
        if (grass_weight < 0.015)
            return blend_micro_bank(material_position_xz, projected_position, 0u, 9u);
        if (grass_weight > 0.985)
            return blend_micro_bank(material_position_xz, projected_position, 9u, 6u);
        MicroMaterial rock = blend_micro_bank(
            material_position_xz, projected_position, 0u, 9u);
        MicroMaterial grass = blend_micro_bank(
            material_position_xz, projected_position, 9u, 6u);
        return mix_micro(rock, grass, grass_weight);
    }
    if (grass_weight < 0.015) {
        if (snow_weight > 0.985)
            return blend_micro_bank(material_position_xz, projected_position, 15u, 6u);
        MicroMaterial rock = blend_micro_bank(
            material_position_xz, projected_position, 0u, 9u);
        MicroMaterial snow = blend_micro_bank(
            material_position_xz, projected_position, 15u, 6u);
        return mix_micro(rock, snow, snow_weight);
    }
    MicroMaterial rock = blend_micro_bank(material_position_xz, projected_position, 0u, 9u);
    MicroMaterial grass = blend_micro_bank(material_position_xz, projected_position, 9u, 6u);
    MicroMaterial snow = blend_micro_bank(material_position_xz, projected_position, 15u, 6u);
    float non_snow = max(1.0 - snow_weight, 1e-4);
    MicroMaterial ground = mix_micro(rock, grass,
                                     clamp(grass_weight / non_snow, 0.0, 1.0));
    return mix_micro(ground, snow, snow_weight);
}

MaterialMacro classified_macro(vec2 material_position_xz,
                               vec2 projected_position,
                               float grass_weight,
                               float snow_weight) {
    /* Region-randomized macro scans made one cliff alternate between cracked,
       layered, and mottled formations in large rectangular sections. Keep
       that variation in the true-scale micro layer, where it is useful and
       subtle, but give each macro material class one neutral representative.
       Index 7 is deliberately the non-directional mountain rock scan: giant
       directional cracks can otherwise look like a return of UV stretching. */
    MaterialMacro rock = sample_material_macro(7u, projected_position);
    if (snow_weight < 0.015) {
        if (grass_weight < 0.015)
            return rock;
        MaterialMacro grass = sample_material_macro(10u, projected_position);
        return mix_macro(rock, grass, grass_weight);
    }
    MaterialMacro snow = sample_material_macro(16u, projected_position);
    if (grass_weight < 0.015)
        return mix_macro(rock, snow, snow_weight);
    MaterialMacro grass = sample_material_macro(10u, projected_position);
    float non_snow = max(1.0 - snow_weight, 1e-4);
    MaterialMacro ground = mix_macro(
        rock, grass, clamp(grass_weight / non_snow, 0.0, 1.0));
    return mix_macro(ground, snow, snow_weight);
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

/* A top-down orthophoto cannot provide new cliff colour, but it can be
   prevented from injecting vertically magnified detail. Expand its texture
   derivatives in the DEM downslope direction by exactly the surface-area
   stretch, 1/normal.y. This selects a correspondingly coarser/anisotropic mip
   on a cliff while leaving flat ground bit-identical. */
vec2 slope_corrected_derivative(vec2 derivative, vec2 slope_direction,
                                float stretch) {
    float along_slope = dot(derivative, slope_direction);
    return derivative + slope_direction * along_slope * (stretch - 1.0);
}

/* Metric box projection: every coordinate is measured in metres, so the scan
   has the same physical scale on horizontal ground and on either cliff axis.
   Fourth-power weights keep most fragments to one dominant projection while
   retaining a broad, continuous cross-fade at axis boundaries. Unlike the old
   snapped contour axis, this cannot switch UV frames abruptly between adjacent
   DEM sections. Components are X-side, top, Z-side. */
vec3 material_projection_weights(vec3 normal) {
    vec3 weights = pow(abs(normal), vec3(4.0));
    return weights / max(weights.x + weights.y + weights.z, 1e-5);
}

void material_projection_coordinates(vec3 position, vec3 normal, uint axis,
                                     out vec2 projected_position,
                                     out vec3 tangent_hint) {
    if (axis == 0u) {
        /* YZ plane. cross(+Z, normal) follows the signed vertical UV axis. */
        float facing_sign = normal.x < 0.0 ? -1.0 : 1.0;
        projected_position = vec2(position.z, facing_sign * position.y);
        tangent_hint = vec3(0.0, 0.0, 1.0);
    } else if (axis == 1u) {
        projected_position = position.xz;
        tangent_hint = vec3(1.0, 0.0, 0.0);
    } else {
        /* XY plane. cross(+X, normal) follows the signed vertical UV axis. */
        float facing_sign = normal.z < 0.0 ? -1.0 : 1.0;
        projected_position = vec2(position.x, -facing_sign * position.y);
        tangent_hint = vec3(1.0, 0.0, 0.0);
    }
}

vec3 reorient_micro_normal(vec3 tangent_normal, vec3 terrain_normal,
                           vec3 tangent_hint) {
    vec3 tangent = normalize(tangent_hint - terrain_normal *
                             dot(tangent_hint, terrain_normal));
    vec3 bitangent = normalize(cross(tangent, terrain_normal));
    return normalize(tangent * tangent_normal.x +
                     bitangent * tangent_normal.y +
                     terrain_normal * tangent_normal.z);
}

void accumulate_projected_material(
    vec2 material_position_xz, vec3 material_position, vec3 analytic_normal,
    vec3 shading_normal, uint axis, float weight,
    float grass_weight, float snow_weight,
    inout MicroMaterial micro, inout vec3 micro_local_normal,
    inout MaterialMacro material_macro,
    inout vec3 material_macro_local_normal) {
    if (weight <= 0.0005)
        return;
    vec2 projected_position;
    vec3 tangent_hint;
    material_projection_coordinates(material_position, analytic_normal, axis,
                                    projected_position, tangent_hint);
    MicroMaterial projected_micro = classified_micro(
        material_position_xz, projected_position, grass_weight, snow_weight);
    MaterialMacro projected_macro = classified_macro(
        material_position_xz, projected_position, grass_weight, snow_weight);
    micro.luminance_factor += projected_micro.luminance_factor * weight;
    micro.tangent_normal += projected_micro.tangent_normal * weight;
    micro.ao += projected_micro.ao * weight;
    micro.roughness += projected_micro.roughness * weight;
    micro.height += projected_micro.height * weight;
    micro_local_normal += reorient_micro_normal(
        projected_micro.tangent_normal, shading_normal, tangent_hint) * weight;
    material_macro.color += projected_macro.color * weight;
    material_macro.tangent_normal += projected_macro.tangent_normal * weight;
    material_macro.reference_luminance +=
        projected_macro.reference_luminance * weight;
    material_macro_local_normal += reorient_micro_normal(
        projected_macro.tangent_normal, shading_normal, tangent_hint) * weight;
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
    float top_down_stretch = 1.0 / max(surface.analytic_normal.y, 0.05);
    vec3 projection_weights = material_projection_weights(
        surface.analytic_normal);
    float side_projection_weight = 1.0 - projection_weights.y;
    vec2 slope_direction = length(surface.gradient) > 1e-5
        ? normalize(surface.gradient) : vec2(1.0, 0.0);
    vec2 child_dx = slope_corrected_derivative(
        dFdxCoarse(imagery_uv), slope_direction, top_down_stretch);
    vec2 child_dy = slope_corrected_derivative(
        dFdyCoarse(imagery_uv), slope_direction, top_down_stretch);
    vec4 child_macro = textureGrad(macro_color_map, imagery_uv,
                                   child_dx, child_dy);
    /* At the height-field limit the top-down derivative tends to zero, so no
       finite derivative multiplier can select the needed mip. Supply a
       slope-derived minimum LOD there. The two extra cliff mips deliberately
       leave only broad geographic tint for the projected material to detail. */
    float macro_unstretch_lod = clamp(log2(top_down_stretch) +
                                      2.0 * side_projection_weight, 0.0, 6.0);
    vec4 child_macro_lowpass = textureLod(macro_color_map, imagery_uv,
                                          macro_unstretch_lod);
    child_macro = mix(child_macro, child_macro_lowpass,
                      side_projection_weight);
    float parent_quadrant = floor(draw.debug.z);
    vec2 parent_tile_uv = tile_uv * 0.5 + 0.5 * vec2(
        mod(parent_quadrant, 2.0), floor(parent_quadrant * 0.5));
    vec2 parent_imagery_uv = parent_tile_uv * draw.imagery_uv_transform.xy +
                             draw.imagery_uv_transform.zw;
    vec2 parent_dx = slope_corrected_derivative(
        dFdxCoarse(parent_imagery_uv), slope_direction, top_down_stretch);
    vec2 parent_dy = slope_corrected_derivative(
        dFdyCoarse(parent_imagery_uv), slope_direction, top_down_stretch);
    vec4 parent_macro = textureGrad(parent_macro_color_map, parent_imagery_uv,
                                    parent_dx, parent_dy);
    vec4 parent_macro_lowpass = textureLod(parent_macro_color_map,
                                           parent_imagery_uv,
                                           macro_unstretch_lod);
    parent_macro = mix(parent_macro, parent_macro_lowpass,
                       side_projection_weight);
    float lod_fade = clamp(fract(draw.debug.z) * 8.0, 0.0, 1.0);
    vec4 macro_sample = mix(parent_macro, child_macro, lod_fade);
    vec3 macro_tint = macro_sample.rgb;
    /* Golden synthesis owns RGB. Alpha supplies the older Alpine rock/grass
       classification; bright neutral RGB adds snow below without feeding a
       separately blurred classifier colour into the result. */
    float grass_weight = smoothstep(0.08, 0.55, macro_sample.a);
    float steepness = smoothstep(0.42, 0.78, 1.0 - surface.local_normal.y);
    /* Orthophoto grass coverage is top-down land cover, not proof that grass
       coats a near-vertical wall. Keep ledges green but converge cliffs to the
       single continuous rock macro material. */
    grass_weight *= 1.0 - 0.90 * steepness;
    float macro_luminance = dot(macro_tint, vec3(0.2126, 0.7152, 0.0722));
    float macro_chroma = max(macro_tint.r, max(macro_tint.g, macro_tint.b)) -
                         min(macro_tint.r, min(macro_tint.g, macro_tint.b));
    /* Runtime RGB retains the actual orthophoto snow that the older binary
       grass alpha could not represent. In linear space snow is both bright and
       neutral; the upward-facing term rejects bright vertical rock and keeps
       the three material weights mutually exclusive. */
    float snow_weight = smoothstep(0.42, 0.68, macro_luminance) *
                        (1.0 - smoothstep(0.07, 0.18, macro_chroma)) *
                        smoothstep(0.18, 0.58, surface.analytic_normal.y);
    grass_weight *= 1.0 - snow_weight;
    float rock_weight = max(1.0 - grass_weight - snow_weight, 0.0);

    /* draw.debug.yw is the tile's world-space origin reduced modulo
       phase_period_m (4096 m, terrain_runtime.c) in double on the CPU, so it
       stays small enough here to keep full float32 precision once added to
       the small tile-local coordinate below. */
    vec2 world_phase = local_position.xz + draw.debug.yw;
    /* Golden already supplies macro and meso structure. The authored scans are
       a deliberately subordinate, true-scale PBR layer. */
    /* local_position.y resets at every tile transform, which made side
       projection jump phase in rectangular LOD sections. Tile translation is
       camera-relative; adding the camera's reduced world-Y phase reconstructs
       one continuous vertical material coordinate without sending a large,
       imprecise absolute float to the GPU. */
    float world_phase_y = local_position.y +
                          draw.local_to_camera_relative[3].y +
                          frame.shader_dump.z;
    vec3 material_position = vec3(world_phase.x, world_phase_y, world_phase.y);
    /* Blend the same matching material through all three metric projections.
       Material selection remains keyed to X/Z, so the projection blend cannot
       introduce material-cell seams. */
    MicroMaterial micro = MicroMaterial(0.0, vec3(0.0), 0.0, 0.0, 0.0);
    vec3 micro_local_normal = vec3(0.0);
    MaterialMacro material_macro = MaterialMacro(vec3(0.0), vec3(0.0), 0.0);
    vec3 material_macro_local_normal = vec3(0.0);
    for (uint axis = 0u; axis < 3u; ++axis) {
        accumulate_projected_material(
            world_phase, material_position, surface.analytic_normal,
            surface.local_normal, axis, projection_weights[axis],
            grass_weight, snow_weight, micro, micro_local_normal,
            material_macro, material_macro_local_normal);
    }
    micro.tangent_normal = normalize(micro.tangent_normal);
    micro_local_normal = normalize(micro_local_normal);
    material_macro.tangent_normal = normalize(material_macro.tangent_normal);
    material_macro_local_normal = normalize(material_macro_local_normal);
    float view_distance = length(camera_relative_position);
    float detail_fade = 1.0 - smoothstep(4000.0, 12000.0, view_distance);
    const float detail_weight = 0.65;
    /* Fade the whole micro layer out as one pixel comes to cover more ground
       than the scans can resolve. Driven by the WORLD-space footprint (metres of
       terrain per pixel), which is material-independent, so the fade is a smooth
       gradient rather than the per-material dashed contour a per-scan footprint
       produces. This is where residual atlas bleed and grazing-angle aliasing
       (the black speckle) get cleaned -- without it, distant/grazing micro
       normals shade scattered fragments sunless. */
    float world_footprint_m = max(length(dFdx(local_position.xz)),
                                  length(dFdy(local_position.xz)));
    float footprint_confidence = 1.0 - smoothstep(0.8, 4.0, world_footprint_m);
    float weighted_detail = detail_weight * detail_fade * footprint_confidence;
    /* Each scan normal was reoriented through the same top/cliff basis as its
       albedo sample above; treating tangent Z as world-up would make steep
       close-range cells falsely face the sun. */
    vec3 macro_local_normal = normalize(mix(surface.local_normal,
                                             material_macro_local_normal,
                                             weighted_detail * 0.45));
    vec3 material_local_normal = normalize(mix(macro_local_normal,
                                                micro_local_normal,
                                                weighted_detail * 0.22));
    vec3 geometric_normal = normalize(mat3(draw.local_to_camera_relative) *
                                      material_local_normal);
    vec3 V = normalize(-camera_relative_position);
    if (dot(geometric_normal, V) < 0.0)
        geometric_normal = -geometric_normal;

    /* Fine high-pass luminance remains subordinate to the geographic tint and
       the material-matched coarse structure layer. */
    float micro_layer = mix(1.0, micro.luminance_factor,
                            weighted_detail * 0.40);
    /* Keep the original terrain imagery as the only RGB source. On slopes it
       has already been reduced to a broad, slope-corrected colour guide above;
       the metric material projection contributes achromatic structure only.
       Normalizing by each representative scan's linear mean removes the brown
       rock / vivid-green grass / blue-white snow cast that previously looked
       like a red diagnostic overlay baked into the terrain. */
    float top_imagery_retention = pow(projection_weights.y, 8.0);
    float material_color_weight = 1.0 -
        (1.0 - weighted_detail * 0.70) *
        top_imagery_retention;
    float sampled_macro_luminance = dot(
        material_macro.color, vec3(0.2126, 0.7152, 0.0722));
    float macro_structure = clamp(
        sampled_macro_luminance /
        max(material_macro.reference_luminance, 0.02), 0.65, 1.35);
    vec3 matched_macro_tint = macro_tint *
        mix(1.0, macro_structure, material_color_weight);
    vec3 base_color = matched_macro_tint * micro_layer;
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
        out_color = vec4(rock_weight, grass_weight, snow_weight, 1.0);
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
    if (frame.debug_view > 21.5 && frame.debug_view < 22.5) {
        out_color = vec4(material_macro.color, 1.0); return;
    }
    if (frame.debug_view > 22.5 && frame.debug_view < 23.5) {
        out_color = vec4(projection_weights, 1.0); return;
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
#ifdef TERRAIN_STRETCH_OVERLAY
    /* Diagnostic builds only: --show-texture-stretch (main.c) is the live
       equivalent of tools/detect_texture_stretch.py's red overlay, driven by
       the same 1/|normal.y| ratio computed above. Normal builds do not contain
       this colour path at all. */
    if (frame.stretch_overlay.x > 0.5 && top_down_stretch >= frame.stretch_overlay.y)
        final_color = mix(final_color, vec3(1.0, 0.0, 0.0), frame.stretch_overlay.z);
#endif
    out_color = vec4(final_color, 1.0);

    shader_dump(DUMP_SHADER_TERRAIN,
                vec4(macro_tint, micro.luminance_factor),
                vec4(surface.local_normal, weighted_detail),
                vec4(projection_weights, side_projection_weight),
                vec4(base_color, roughness),
                vec4(final_color, NoL));
    shader_dump(DUMP_SHADER_ENVIRONMENT_IBL,
                vec4(environment.reflection_direction, environment.mip),
                vec4(environment.sampled_reflection_radiance, environment.NoV),
                vec4(environment.ggx_specular_energy, environment.reflection_visibility),
                vec4(environment.unoccluded_specular, ao),
                vec4(environment.final_specular, roughness));
}
