#version 450
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) in vec2 texcoord;
layout(location = 1) in float untextured;
layout(location = 2) in vec2 tile_uv;
layout(location = 3) in vec3 local_position;
layout(location = 4) in vec3 camera_relative_position;
layout(location = 0) out vec4 out_color;

layout(set = 1, binding = 0) uniform sampler2D albedo;
layout(set = 1, binding = 1) uniform sampler2D elevation;
layout(set = 1, binding = 2) uniform sampler2D surface_detail;

layout(push_constant) uniform DrawData {
    mat4 local_to_camera_relative;
    vec4 geometry;
    vec4 elevation_uv;
    vec4 imagery_uv;
    vec4 debug;
} draw;

const float TAU = 6.28318530717958647692;

struct HeightSurface {
    vec3 local_normal;
    vec2 gradient;
    float curvature;
};

vec3 lod_color(float level) {
    const vec3 colors[6] = vec3[6](
        vec3(0.90, 0.20, 0.20), vec3(0.20, 0.75, 0.25),
        vec3(0.20, 0.45, 0.95), vec3(0.95, 0.75, 0.15),
        vec3(0.75, 0.25, 0.90), vec3(0.15, 0.85, 0.85));
    return colors[int(level) % 6];
}

float height_at(vec2 uv) {
    return textureLod(elevation, uv, 0.0).r * draw.geometry.z;
}

HeightSurface height_surface() {
    ivec2 size_i = textureSize(elevation, 0);
    vec2 size = vec2(size_i);
    vec2 height_uv = tile_uv * draw.elevation_uv.xy + draw.elevation_uv.zw;
    vec2 texel = 1.0 / size;
    float left  = height_at(height_uv - vec2(texel.x, 0.0));
    float right = height_at(height_uv + vec2(texel.x, 0.0));
    float down  = height_at(height_uv - vec2(0.0, texel.y));
    float up    = height_at(height_uv + vec2(0.0, texel.y));
    float centre = height_at(height_uv);

    /* elevation_uv.xy * textureSize is the number of interior intervals.
       Sampling one texel outside the visible edge lands in the .trn gutter,
       so adjacent tiles produce the same central difference. */
    vec2 metres = draw.geometry.xy / (draw.elevation_uv.xy * size);
    vec2 gradient = vec2((right - left) / (2.0 * metres.x),
                         (up - down) / (2.0 * metres.y));
    vec3 normal = normalize(vec3(-gradient.x, 1.0, -gradient.y));

    /* Terrain3D main.glsl:439-445 uses derivative-estimated mip to skip work
       once extra interpolation cannot affect more than a pixel. Our compact
       equivalent uses the already-rasterised geometric derivative only beyond
       an eight-height-texel footprint; the transition is sub-pixel by design. */
    float texel_footprint = max(length(dFdxCoarse(height_uv) * size),
                                length(dFdyCoarse(height_uv) * size));
    if (texel_footprint > 8.0) {
        normal = normalize(cross(dFdyCoarse(local_position),
                                 dFdxCoarse(local_position)));
        if (normal.y < 0.0) normal = -normal;
    }

    HeightSurface result;
    result.local_normal = normal;
    result.gradient = gradient;
    result.curvature = (left + right + down + up - 4.0 * centre) /
                       max(metres.x + metres.y, 1e-5);
    return result;
}

/* Adapted from Terrain3D main.glsl:263-269 (Petkovsek, Palmroos et al., MIT).
   Keeping its two-component sin/cos representation avoids constructing a mat2
   for each detail basis and rotated shadow-filter kernel. */
float random_cell(vec2 xy) {
    return fract(sin(dot(xy, vec2(12.9898, 78.233))) * 43758.5453);
}

vec2 rotate_vec2(vec2 v, vec2 cs) {
    return vec2(fma(cs.x, v.x, cs.y * v.y),
                fma(cs.x, v.y, -cs.y * v.x));
}

vec4 sample_rotated_detail(vec2 uv, vec2 ddx_uv, vec2 ddy_uv,
                           vec2 rotation, vec2 offset) {
    return textureGrad(surface_detail, rotate_vec2(uv, rotation) + offset,
                       rotate_vec2(ddx_uv, rotation),
                       rotate_vec2(ddy_uv, rotation));
}

/* Adapted from Terrain3D projection.glsl (MIT). Its octant-snapped cliff axis
   gives a much cheaper stable projection than full triplanar sampling. */
void projected_coordinates(vec3 position, vec3 normal,
                           out vec2 uv, out vec2 ddx_uv, out vec2 ddy_uv,
                           out vec3 tangent_hint) {
    vec3 ddx_position = dFdxCoarse(position);
    vec3 ddy_position = dFdyCoarse(position);
    uv = position.xz;
    ddx_uv = ddx_position.xz;
    ddy_uv = ddy_position.xz;
    tangent_hint = vec3(1.0, 0.0, 0.0);
    if (normal.y <= 0.7071067811865475) {
        vec2 axis = round(normalize(-normal.xz) * 1.3065629648763765);
        axis *= abs(axis.x) + abs(axis.y) > 1.5
            ? 0.7071067811865475 : 1.0;
        axis = vec2(-axis.y, axis.x);
        uv = vec2(dot(position.xz, axis), -position.y);
        ddx_uv = vec2(dot(ddx_position.xz, axis), -ddx_position.y);
        ddy_uv = vec2(dot(ddy_position.xz, axis), -ddy_position.y);
        tangent_hint = vec3(axis.x, 0.0, axis.y);
    }
}

vec3 unpack_detail_normal(vec4 packed, vec2 rotation, vec3 terrain_normal,
                          vec3 tangent_hint) {
    vec2 xy = packed.rg * 2.0 - 1.0;
    xy = rotate_vec2(xy, rotation);
    /* The generated map contains unit normals so its RG range is suitable for
       storage, but applying that full amplitude overwhelms aerial imagery.
       Treat it as micro-normal perturbation, not replacement geometry. */
    xy *= 0.22;
    float z = sqrt(max(1.0 - dot(xy, xy), 0.02));
    vec3 tangent = normalize(tangent_hint -
                             terrain_normal * dot(terrain_normal, tangent_hint));
    vec3 bitangent = normalize(cross(tangent, terrain_normal));
    return normalize(tangent * xy.x + bitangent * xy.y + terrain_normal * z);
}

/* Terrain3D main.glsl:321-345 (MIT) supplies the derivative-rotation pattern.
   Its hard random cells assume authored material textures whose rotated edges
   match. Our generated periodic map does not have that property—the old direct
   port exposed every cell as a rectangle. Instead we blend two continuous,
   derivative-correct bases with a low-frequency mask. Quarter-turn rotation is
   deliberate: it preserves the 4096 m CPU phase lattice across tile seams. */
vec3 sample_detail_scale(vec2 projected_uv, vec2 projected_ddx,
                         vec2 projected_ddy, float scale,
                         vec3 terrain_normal, vec3 tangent_hint) {
    vec2 uv = projected_uv * scale;
    vec2 ddx_uv = projected_ddx * scale;
    vec2 ddy_uv = projected_ddy * scale;
    const vec2 rotation_a = vec2(1.0, 0.0);
    const vec2 rotation_b = vec2(0.0, 1.0); /* 90 degrees */
    vec4 packed_a = sample_rotated_detail(uv, ddx_uv, ddy_uv,
                                           rotation_a, vec2(0.13, 0.37));
    vec4 packed_b = sample_rotated_detail(uv, ddx_uv, ddy_uv,
                                           rotation_b, vec2(0.61, 0.19));
    vec3 normal_a = unpack_detail_normal(packed_a, rotation_a,
                                          terrain_normal, tangent_hint);
    vec3 normal_b = unpack_detail_normal(packed_b, rotation_b,
                                          terrain_normal, tangent_hint);

    vec2 mask_uv = projected_uv * (1.0 / 128.0) + vec2(0.31, 0.73);
    float mask = textureGrad(surface_detail, mask_uv,
                             projected_ddx * (1.0 / 128.0),
                             projected_ddy * (1.0 / 128.0)).b;
    return normalize(mix(normal_a, normal_b,
                         smoothstep(0.25, 0.75, mask)));
}

vec3 detailed_normal(vec3 terrain_normal, vec3 surface_position, float distance_m) {
    vec2 projected_uv, projected_ddx, projected_ddy;
    vec3 tangent_hint;
    projected_coordinates(surface_position, terrain_normal,
                          projected_uv, projected_ddx, projected_ddy, tangent_hint);

    const float near_scale = 0.0625;   /* one packed texture per 16 metres */
    const float far_scale = 0.015625;  /* one packed texture per 64 metres */
    vec3 near_normal = sample_detail_scale(projected_uv, projected_ddx,
        projected_ddy, near_scale, terrain_normal, tangent_hint);

    /* Adapted from Terrain3D dual_scaling.glsl:15-75 (MIT): blend a reduced
       scale over distance while preserving each scale's own derivatives. */
    float far_factor = smoothstep(90.0, 260.0, distance_m);
    if (far_factor <= 0.0) return near_normal;
    vec3 far_normal = sample_detail_scale(projected_uv, projected_ddx,
        projected_ddy, far_scale, terrain_normal, tangent_hint);
    return normalize(mix(near_normal, far_normal, far_factor));
}

/* Reduced from Terrain3D macro_variation.glsl:6-24 (MIT). Two low-frequency
   channels break up repetition; slope attenuation avoids tinting cliff imagery
   whose source appearance is already strongly directional. */
vec3 apply_macro_variation(vec3 color, vec3 surface_position, vec3 normal) {
    vec2 uv1 = surface_position.xz * (1.0 / 256.0);
    vec2 uv2 = surface_position.xz * (1.0 / 128.0);
    vec2 cs = vec2(0.819152, 0.573576); /* 35 degrees */
    vec2 uv1_rotated = rotate_vec2(uv1, cs);
    float noise1 = textureGrad(surface_detail, uv1_rotated,
        rotate_vec2(dFdxCoarse(uv1), cs), rotate_vec2(dFdyCoarse(uv1), cs)).b;
    float noise2 = textureGrad(surface_detail, uv2,
                               dFdxCoarse(uv2), dFdyCoarse(uv2)).a;
    float variation = mix(0.90, 1.08, noise1) * mix(0.94, 1.05, noise2);
    return color * mix(1.0, variation, smoothstep(0.15, 0.70, normal.y));
}

struct ShadowResult {
    float visibility;
    float receiver_bias;
    uint cascade;
    vec3 coordinate;
};

/* Eight of Wicked shadowHF.hlsli's Vogel disk points (MIT). Rotating this
   compact kernel per pixel avoids axis-aligned PCF banding without importing
   Wicked's bindless atlas, transparent-shadow, or PCSS machinery. */
const vec2 vogel_points[8] = vec2[8](
    vec2( 0.25000000,  0.00000000), vec2(-0.31930089,  0.29248416),
    vec2( 0.04891348, -0.55687296), vec2( 0.40238643,  0.52496207),
    vec2(-0.73851585, -0.13074535), vec2( 0.69968677, -0.44490278),
    vec2(-0.23419666,  0.87043202), vec2(-0.44604915, -0.85938364));

float sample_shadow_pcf(uint cascade, vec3 coordinate) {
    float angle = random_cell(floor(gl_FragCoord.xy * 0.5)) * TAU;
    vec2 cs = vec2(cos(angle), sin(angle));
    vec2 texel_radius = vec2(frame.shadow_parameters.y /
                             float(textureSize(shadow_map, 0).x));
    float visibility = 0.0;
    for (uint i = 0u; i < 8u; ++i) {
        vec2 offset = rotate_vec2(vogel_points[i], cs) * texel_radius;
        visibility += texture(shadow_map,
            vec4(coordinate.xy + offset, float(cascade), coordinate.z));
    }
    return visibility * (1.0 / 8.0);
}

vec3 shadow_coordinate(uint cascade, vec3 receiver) {
    vec4 clip = frame.shadow_view_projection[cascade] * vec4(receiver, 1.0);
    clip.xyz /= clip.w;
    return vec3(clip.xy * 0.5 + 0.5, clip.z);
}

/* Adapted from Wicked lightingHF.hlsli:87-126 (MIT): choose the smallest
   containing cascade, then blend across its inner 10% edge into the next one. */
ShadowResult terrain_shadow(vec3 position, vec3 normal) {
    vec3 to_sun = normalize(-frame.sun_direction.xyz);
    float grazing = 1.0 - max(dot(normal, to_sun), 0.0);
    float receiver_bias = frame.shadow_parameters.x * grazing;
    vec3 receiver = position + normal * receiver_bias;
    ShadowResult result = ShadowResult(1.0, receiver_bias, 4u, vec3(0.0));
    vec3 clip_coordinate = vec3(0.0);
    for (uint cascade = 0u; cascade < 4u; ++cascade) {
        vec4 clip = frame.shadow_view_projection[cascade] * vec4(receiver, 1.0);
        clip.xyz /= clip.w;
        vec3 uv_depth = vec3(clip.xy * 0.5 + 0.5, clip.z);
        if (all(greaterThanEqual(uv_depth, vec3(0.0))) &&
            all(lessThanEqual(uv_depth, vec3(1.0)))) {
            result.cascade = cascade;
            result.coordinate = uv_depth;
            clip_coordinate = clip.xyz;
            break;
        }
    }
    if (result.cascade >= 4u) return result;

    result.visibility = sample_shadow_pcf(result.cascade, result.coordinate);
    if (result.cascade < 3u) {
        vec3 edge = clamp((abs(vec3(clip_coordinate.xy,
                                    clip_coordinate.z * 2.0 - 1.0)) - 0.8) * 5.0,
                          0.0, 1.0);
        float blend = max(edge.x, max(edge.y, edge.z));
        if (blend > 0.0) {
            uint fallback = result.cascade + 1u;
            float fallback_visibility = sample_shadow_pcf(
                fallback, shadow_coordinate(fallback, receiver));
            result.visibility = mix(result.visibility, fallback_visibility, blend);
        }
    }
    return result;
}

/* Direct GLSL subset of Wicked brdf.hlsli:8-64 (MIT), itself based on
   Filament. Only the isotropic dielectric GGX path is retained. */
float D_GGX(float NoH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float denominator = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / max(3.141592653589793 * denominator * denominator, 1e-6);
}

float V_SmithGGXCorrelated(float roughness, float NoV, float NoL) {
    float a2 = roughness * roughness;
    float lambda_v = NoL * sqrt(max((NoV - a2 * NoV) * NoV + a2, 0.0));
    float lambda_l = NoV * sqrt(max((NoL - a2 * NoL) * NoL + a2, 0.0));
    return 0.5 / max(lambda_v + lambda_l, 1e-5);
}

vec3 F_Schlick(float VoH, vec3 f0) {
    float factor = pow(1.0 - VoH, 5.0);
    return f0 + (1.0 - f0) * factor;
}

vec3 evaluate_terrain_lighting(vec3 base_color, vec3 normal, float roughness,
                               float occlusion, float shadow_visibility) {
    vec3 view_direction = normalize(-camera_relative_position);
    vec3 light_direction = normalize(-frame.sun_direction.xyz);
    vec3 half_direction = normalize(view_direction + light_direction);
    float NoV = max(dot(normal, view_direction), 1e-4);
    float NoL = max(dot(normal, light_direction), 0.0);
    float NoH = max(dot(normal, half_direction), 0.0);
    float VoH = max(dot(view_direction, half_direction), 0.0);
    vec3 fresnel = F_Schlick(VoH, vec3(0.04));
    float distribution = D_GGX(NoH, roughness);
    float visibility = V_SmithGGXCorrelated(roughness * roughness, NoV, NoL);
    vec3 specular = distribution * visibility * fresnel;
    vec3 diffuse = (1.0 - fresnel) * base_color / 3.141592653589793;

    /* Mirrors Wicked ApplyLighting's direct/indirect separation. The temporary
       hemisphere term is replaced by physical sky irradiance in Phase 7. */
    vec3 direct = (diffuse + specular) * frame.sun_radiance.rgb *
                  NoL * shadow_visibility;
    float hemisphere = mix(0.08, 0.22,
        clamp(normal.y * 0.5 + 0.5, 0.0, 1.0));
    vec3 indirect = base_color * (1.0 - fresnel) * hemisphere * occlusion;
    return direct + indirect;
}

void main() {
    if (frame.debug_view > 0.5 && frame.debug_view < 1.5) {
        float metres = linear_view_depth(gl_FragCoord.z);
        float value = clamp(log2(1.0 + metres) / log2(1.0 + 1.0e7), 0.0, 1.0);
        out_color = vec4(vec3(value), 1.0);
        return;
    }

    HeightSurface surface = height_surface();
    mat3 tile_rotation = mat3(draw.local_to_camera_relative);
    vec3 geometric_normal = normalize(tile_rotation * surface.local_normal);
    float distance_m = length(camera_relative_position);
    ShadowResult shadow = terrain_shadow(camera_relative_position,
                                          geometric_normal);

    if (frame.debug_view > 2.5 && frame.debug_view < 3.5) {
        out_color = vec4(geometric_normal * 0.5 + 0.5, 1.0);
        return;
    }
    if (frame.debug_view > 3.5 && frame.debug_view < 4.5) {
        float slope = 1.0 - clamp(surface.local_normal.y, 0.0, 1.0);
        out_color = vec4(mix(vec3(0.08, 0.25, 0.85), vec3(0.95, 0.18, 0.05),
                             smoothstep(0.0, 0.75, slope)), 1.0);
        return;
    }
    if (frame.debug_view > 4.5 && frame.debug_view < 5.5) {
        float curve = clamp(surface.curvature * 0.08, -1.0, 1.0);
        out_color = vec4(curve < 0.0 ? vec3(0.1, 0.3, 1.0) * -curve
                                    : vec3(1.0, 0.25, 0.08) * curve,
                         1.0);
        return;
    }
    if (frame.debug_view > 5.5 && frame.debug_view < 6.5) {
        float magnitude = length(surface.gradient);
        vec2 direction = magnitude > 1e-6 ? surface.gradient / magnitude : vec2(0.0);
        out_color = vec4(direction * 0.5 + 0.5,
                         clamp(log2(1.0 + magnitude) / 5.0, 0.0, 1.0), 1.0);
        return;
    }
    if (frame.debug_view > 6.5 && frame.debug_view < 7.5) {
        const vec3 cascade_colors[5] = vec3[5](
            vec3(0.95, 0.18, 0.12), vec3(0.18, 0.82, 0.25),
            vec3(0.15, 0.45, 1.0), vec3(0.95, 0.75, 0.10), vec3(0.1));
        out_color = vec4(cascade_colors[shadow.cascade], 1.0);
        return;
    }
    if (frame.debug_view > 7.5 && frame.debug_view < 8.5) {
        out_color = vec4(shadow.coordinate, 1.0);
        return;
    }
    if (frame.debug_view > 8.5 && frame.debug_view < 9.5) {
        out_color = vec4(vec3(shadow.visibility), 1.0);
        return;
    }
    if (frame.debug_view > 9.5 && frame.debug_view < 10.5) {
        out_color = vec4(vec3(clamp(shadow.receiver_bias /
                                    max(frame.shadow_parameters.x, 1e-5), 0.0, 1.0)),
                         1.0);
        return;
    }
    if (frame.debug_view > 10.5 && frame.debug_view < 11.5) {
        float raw_depth = shadow.cascade < 4u
            ? texture(shadow_map_raw,
                      vec3(shadow.coordinate.xy, float(shadow.cascade))).r : 1.0;
        out_color = vec4(vec3(raw_depth), 1.0);
        return;
    }

    vec3 map_color = textureGrad(albedo, texcoord,
                                 dFdxCoarse(texcoord), dFdyCoarse(texcoord)).rgb;
    map_color = mix(map_color, vec3(0.18), step(0.5, untextured));

    float material_strength = smoothstep(0.45, 1.0, frame.relight_strength);
    vec3 surface_position = local_position + draw.debug.yzw;
    vec3 local_lit_normal = surface.local_normal;
    if (material_strength > 0.0) {
        vec3 detail_normal = detailed_normal(surface.local_normal,
                                             surface_position, distance_m);
        float detail_fade = 1.0 - smoothstep(120.0, 400.0, distance_m);
        local_lit_normal = normalize(mix(surface.local_normal, detail_normal,
                                         material_strength * detail_fade));
        vec3 varied = apply_macro_variation(map_color, surface_position,
                                             surface.local_normal);
        map_color = mix(map_color, varied, material_strength);
    }

    vec3 lit_normal = normalize(tile_rotation * local_lit_normal);
    float roughness = mix(0.88, 0.68, material_strength);
    vec3 relit_color = evaluate_terrain_lighting(
        map_color, lit_normal, roughness, 1.0, shadow.visibility);
    vec3 color = mix(map_color, relit_color,
                     clamp(frame.relight_strength, 0.0, 1.0));
    if (frame.debug_view > 1.5)
        color = mix(color, lod_color(draw.debug.x), 0.72);
    out_color = vec4(display_transform(color), 1.0);
}
