#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "shadow_filter.glsl"
#include "pbr_common.glsl"
#include "environment_lighting.glsl"
#include "material_detail.glsl"
#include "dungeon_noise.glsl"
#include "shader_dump.glsl"

/* Dungeon floor/wall PBR permutation. Moss uses the user-provided JPEG at
   binding 3; height above the floor excludes the plateau cap. */
layout(early_fragment_tests) in;

layout(location = 0) in vec2 uv;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 tangent;
layout(location = 3) in vec3 camera_relative_position;
layout(location = 4) in vec4 current_clip;
layout(location = 5) in vec4 previous_clip;
layout(location = 6) in vec3 local_position;
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 out_motion;

layout(set = 1, binding = 0) uniform sampler2D albedo_map;
layout(set = 1, binding = 1) uniform sampler2D orm_map;
layout(set = 1, binding = 2) uniform sampler2D normal_map;
layout(set = 1, binding = 3) uniform sampler2D moss_albedo_map;


layout(push_constant) uniform DrawData {
    mat4 local_to_camera_relative;
    vec4 geometry;
    vec4 elevation_uv;
    vec4 material_factors; /* metallic, Default Lit, diffuse IBL, specular IBL */
    vec4 debug; /* cavity strength, displacement (reserved), shadows, curvature */
} draw;

vec3 fallback_tangent(vec3 N) {
    vec3 axis = abs(N.y) < 0.999 ? vec3(0.0, 1.0, 0.0)
                                 : vec3(1.0, 0.0, 0.0);
    return normalize(cross(axis, N));
}

mat3 tangent_frame(vec3 interpolated_normal, vec4 interpolated_tangent) {
    float normal_length2 = dot(interpolated_normal, interpolated_normal);
    vec3 N = normal_length2 > 1e-10
        ? interpolated_normal * inversesqrt(normal_length2)
        : vec3(0.0, 1.0, 0.0);
    vec3 projected_tangent = interpolated_tangent.xyz -
                             N * dot(N, interpolated_tangent.xyz);
    float tangent_length2 = dot(projected_tangent, projected_tangent);
    bool invalid_tangent = tangent_length2 <= 1e-10 ||
        any(isnan(projected_tangent)) || any(isinf(projected_tangent));
    vec3 T = invalid_tangent
        ? fallback_tangent(N)
        : projected_tangent * inversesqrt(tangent_length2);
    float handedness = interpolated_tangent.w < 0.0 ? -1.0 : 1.0;
    vec3 B = normalize(cross(N, T)) * handedness;
    T = normalize(cross(B, N)) * handedness;
    return mat3(T, B, N);
}

#include "dungeon_shadow.glsl"

float crack_segment_distance(vec2 p, vec2 a, vec2 b) {
    vec2 edge = b - a;
    return length(p - a - edge * clamp(dot(p-a, edge) / dot(edge, edge), 0.0, 1.0));
}

// Sparse broken branches, fixed in the brick UVs. Keep endpoints inside
// each cell so the discontinuous cell hash never creates a square seam.
float brick_crack_distance(vec2 p) {
    vec2 cell = floor(p);
    vec2 seed = dungeon_hash2(cell + 53.7) * 0.5 + 0.5;
    if (seed.x < 0.64) return 1.0;
    vec2 f = fract(p);
    if (seed.y > 0.5) f = f.yx;
    vec2 a = vec2(0.15, 0.20 + seed.y * 0.25);
    vec2 b = vec2(0.43 + seed.y * 0.12, 0.43);
    vec2 c = vec2(0.82, 0.68 + seed.y * 0.15);
    float d = min(crack_segment_distance(f,a,b), crack_segment_distance(f,b,c));
    if (seed.x > 0.82)
        d = min(d, crack_segment_distance(f,b,vec2(0.29,0.84)));
    return d;
}

float brick_crack_height(vec2 p) {
    return -0.018 * (1.0 - smoothstep(0.0, 0.022, brick_crack_distance(p)));
}

void main() {
    vec4 base_color_sample = texture(albedo_map, uv);
    vec3 base_color = base_color_sample.rgb * draw.geometry.rgb;
    vec3 orm = texture(orm_map, uv).rgb;
    vec3 filtered_normal = texture(normal_map, uv).xyz * 2.0 - 1.0;
    float normal_scale = draw.elevation_uv.y;
    float filtered_normal_length = clamp(length(filtered_normal),
                                         frame.material_normal_filter.z, 1.0);
    vec3 tangent_normal = filtered_normal / filtered_normal_length;
    tangent_normal.xy *= normal_scale;
    tangent_normal = normalize(tangent_normal);

    // Millimetre-wide fissures with bevel normals and a dark recessed core.
    float brick_width = abs(normalize(normal).y) > 0.5 ? 2.4 : 1.8;
    vec2 crack_uv = uv * brick_width / 0.7;
    float crack_distance = brick_crack_distance(crack_uv);
    float crack_aa = max(fwidth(crack_distance), 0.001);
    float crack_mask = 1.0 - smoothstep(0.004, 0.010 + crack_aa, crack_distance);
    base_color *= 1.0 - crack_mask * 0.72;
    orm.r *= 1.0 - crack_mask * 0.45;
    orm.g = mix(orm.g, 0.96, crack_mask);
    const float crack_step = 0.006;
    vec2 crack_slope = vec2(
        brick_crack_height(crack_uv + vec2(crack_step,0)) - brick_crack_height(crack_uv - vec2(crack_step,0)),
        brick_crack_height(crack_uv + vec2(0,crack_step)) - brick_crack_height(crack_uv - vec2(0,crack_step))) / (2.0 * crack_step);
    tangent_normal = normalize(vec3(tangent_normal.xy - crack_slope * 0.4, tangent_normal.z));

    vec3 world_position = local_position; // dungeon meshes are authored in world metres
    float height_above_floor = local_position.y - draw.geometry.w;
    // The cap shares the wall batch. Fade out before its upper edge; downward
    // facing surfaces never support moss, even if ceilings are added later.
    float surface_mask = (1.0 - smoothstep(0.75, 1.65, height_above_floor)) *
                         smoothstep(-0.1, 0.2, normalize(normal).y + 0.5);
    // Same fixed cluster field as the CPU foliage scatter; never camera phased.
    float cluster = sin(world_position.x * 0.83 + sin(world_position.z * 0.57)) *
                  sin(world_position.z * 0.91 + sin(world_position.x * 0.43));
    float moss_noise = cluster * 0.5 + 0.5;
    float moss_ao_term = 1.0 - orm.r;
    float recess_mask = smoothstep(0.035, 0.14, moss_ao_term);
    float base_width = abs(normalize(normal).y) > 0.5 ? 2.4 : 1.8;
    // Double the old physical scale. Blend independently offset/rotated
    // samples with a slow, fixed mask to break the repeated square image.
    vec2 moss_uv = uv * base_width * 0.5;
    vec3 moss_a = texture(moss_albedo_map, moss_uv).rgb;
    vec3 moss_b = texture(moss_albedo_map, mat2(0.6, 0.8, -0.8, 0.6) * moss_uv * 0.83 + 0.37).rgb;
    float variation = smoothstep(-0.16, 0.16, dungeon_fbm(world_position.xz * 0.37));
    vec3 moss_color = mix(moss_a, moss_b, variation);
    float moss_mask = surface_mask * 0.65 * smoothstep(0.28, 0.65, cluster + recess_mask * 0.06);
    base_color = mix(base_color, moss_color, moss_mask);
    // Keep the underlying stone relief, softened by the moss cover.
    tangent_normal = normalize(mix(tangent_normal,
        normalize(vec3(tangent_normal.xy * 0.4, tangent_normal.z)), moss_mask));

    bool default_lit = draw.material_factors.y > 0.5;
    vec3 geometric_normal = normalize(normal);
    vec3 mapped_normal;
    if (default_lit) {
        mat3 tbn = tangent_frame(normal, tangent);
        mapped_normal = normalize(tbn * tangent_normal);
    } else {
        vec3 legacy_normal = normalize(normal);
        vec3 T = normalize(tangent.xyz -
                           legacy_normal * dot(legacy_normal, tangent.xyz));
        vec3 B = normalize(cross(legacy_normal, T)) * tangent.w;
        geometric_normal = legacy_normal;
        mapped_normal = normalize(mat3(T, B, legacy_normal) * tangent_normal);
    }
    vec3 L = normalize(-frame.sun_direction.xyz);
    vec3 V = normalize(-camera_relative_position);
    /* Keep the normal in the surface hemisphere. Flipping toward the view
       reverses the wall's light response abruptly at grazing camera angles. */
    if (default_lit)
        mapped_normal = normalize(mapped_normal + geometric_normal *
            max(0.0, 0.001 - dot(mapped_normal, geometric_normal)));
    vec3 N = mapped_normal;
    float NoL = max(dot(N, L), 0.0);
    float authored_roughness = material_authored_roughness(orm.g, draw.elevation_uv.x,
                                                            draw.elevation_uv.w);
    float authored_roughness_before_moss = authored_roughness;
    authored_roughness = mix(authored_roughness, 0.92, moss_mask); /* moss reads matte */
    MaterialDetailResult detail = material_detail_evaluate(geometric_normal,
        filtered_normal_length, normal_scale, authored_roughness, draw.debug.w > 0.0);
    float roughness = detail.effective_roughness;
    float metallic;
    vec3 indirect_diffuse;
    vec3 F0;
    UeDefaultLit bxdf;
    if (default_lit) {
        metallic = clamp(orm.b * draw.material_factors.x, 0.0, 1.0);
        F0 = mix(vec3(0.04), base_color, metallic);
        vec3 diffuse_color = base_color * (1.0 - metallic);
        bxdf = ue_default_lit_bxdf(
            diffuse_color, F0, roughness, N, V, L);
        indirect_diffuse = diffuse_color;
    } else {
        metallic = orm.b;
        F0 = mix(vec3(0.04), base_color, metallic);
        bxdf = legacy_quarry_bxdf(
            base_color, metallic, roughness, N, V, L);
        indirect_diffuse = base_color;
    }
    ShadowResult shadow = shadow_evaluate(camera_relative_position, normalize(normal));
    float visibility = mix(1.0, shadow.visibility, draw.debug.z);
    vec3 direct = (bxdf.diffuse + bxdf.specular) *
                  frame.sun_radiance.rgb * NoL * visibility;
    int point_light_count = clamp(int(frame.point_light_options.x + 0.5), 0, 16);
    for (int light_index = 0; light_index < point_light_count; ++light_index) {
        vec4 position_radius = frame.point_light_position_radius[light_index];
        vec3 to_light = position_radius.xyz - camera_relative_position;
        float distance2 = dot(to_light, to_light);
        float distance_to_light = sqrt(max(distance2, 1e-8));
        float normalized_distance = distance_to_light / max(position_radius.w, 1e-4);
        float attenuation_window = max(1.0 - pow(normalized_distance, 4.0), 0.0);
        float attenuation = attenuation_window * attenuation_window / max(distance2, 0.01);
        vec3 local_L = to_light / distance_to_light;
        float local_NoL = max(dot(N, local_L), 0.0);
        UeDefaultLit local_bxdf;
        if (default_lit)
            local_bxdf = ue_default_lit_bxdf(indirect_diffuse, F0, roughness, N, V, local_L);
        else
            local_bxdf = legacy_quarry_bxdf(base_color, metallic, roughness, N, V, local_L);
        vec4 color_intensity = frame.point_light_color_intensity[light_index];
        vec3 contribution = (local_bxdf.diffuse + local_bxdf.specular) * color_intensity.rgb *
                            color_intensity.w * local_NoL * attenuation;
        if (attenuation > 0.0 && local_NoL > 0.0)
            direct += contribution * point_light_visibility(camera_relative_position,
                normal, position_radius.xyz, light_index, contribution);
    }
    bool diffuse_ibl_enabled = draw.material_factors.z > 0.5;
    bool specular_ibl_enabled = draw.material_factors.w > 0.5;
    float ao_strength = draw.elevation_uv.z;
    float ao = 1.0 + ao_strength * (orm.r - 1.0);
    EnvironmentLightingResult environment = environment_evaluate(
        camera_relative_position, N, V, roughness, F0, ao,
        diffuse_ibl_enabled, specular_ibl_enabled);
    float cavity_sample = 1.0;
    float cavity_visibility = material_visibility(cavity_sample, draw.debug.x);
    vec3 irradiance = environment.irradiance * frame.point_light_options.y;
    vec3 ambient = indirect_diffuse * irradiance * ao * cavity_visibility;
    vec3 specular_ibl = environment.final_specular * frame.point_light_options.z;
    vec3 final_hdr = direct + ambient + specular_ibl;
    out_color = vec4(final_hdr, 1.0);
    vec2 current_uv = current_clip.xy / current_clip.w * 0.5 + 0.5;
    vec2 previous_uv = previous_clip.xy / previous_clip.w * 0.5 + 0.5;
    out_motion = draw.debug.y > 0.5 ? vec2(2.0) : previous_uv - current_uv;

    /* See SHADER_DUMP_LEGEND["dungeon_surface"] in renderer.c for the f0..f19 layout. */
    shader_dump(DUMP_SHADER_DUNGEON_SURFACE,
                vec4(moss_mask, moss_noise, recess_mask, moss_ao_term),
                vec4(base_color, crack_mask),
                vec4(world_position, metallic),
                vec4(N, roughness),
                vec4(final_hdr, authored_roughness_before_moss));
}
