#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "shadow_filter.glsl"
#include "pbr_common.glsl"
#include "environment_lighting.glsl"
#include "material_detail.glsl"
#include "shader_dump.glsl"

/* Opaque, single-pass forward shading with no discard/gl_FragDepth write:
   safe to resolve the depth test before running the fragment shader, which
   also collapses shader_dump() records toward ~1/pixel instead of counting
   every overdrawn fragment. */
layout(early_fragment_tests) in;

layout(location = 0) in vec2 uv;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 tangent;
layout(location = 3) in vec3 camera_relative_position;
layout(location = 4) in vec4 current_clip;
layout(location = 5) in vec4 previous_clip;
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 out_motion;

layout(set = 1, binding = 0) uniform sampler2D albedo_map;
layout(set = 1, binding = 1) uniform sampler2D orm_map;
layout(set = 1, binding = 2) uniform sampler2D normal_map;
layout(set = 1, binding = 3) uniform sampler2D occlusion_map;
layout(set = 1, binding = 4) uniform sampler2D cavity_map;

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

/* glTF tangents may be absent or collapse while being interpolated. Rebuild a
   stable orthonormal frame in that case, and always renormalize it per pixel. */
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
    /* Recompute T so all three interpolated axes are mutually orthogonal. */
    T = normalize(cross(B, N)) * handedness;
    return mat3(T, B, N);
}

#include "dungeon_shadow.glsl"


/* Static-mesh shading: photogrammetry albedo/ORM/normal sampled directly
   through the asset's own unwrapped UVs. Lighting matches the terrain forward
   pass so the two stay visually consistent. */
void main() {
    vec4 base_color_sample = texture(albedo_map, uv);
    vec3 base_color = base_color_sample.rgb * draw.geometry.rgb;
    vec3 orm = vec3(1.0, 0.92, 0.0); // dry foliage: matte, dielectric
    vec3 filtered_normal = vec3(0.0, 0.0, 1.0); // real bent leaf normals
    float normal_scale = draw.elevation_uv.y;
    float filtered_normal_length = clamp(length(filtered_normal),
                                         frame.material_normal_filter.z, 1.0);
    vec3 tangent_normal = filtered_normal / filtered_normal_length;
    tangent_normal.xy *= normal_scale;
    tangent_normal = normalize(tangent_normal);
    bool default_lit = draw.material_factors.y > 0.5;
    vec3 geometric_normal = normalize(normal);
    vec3 mapped_normal;
    if (default_lit) {
        mat3 tbn = tangent_frame(normal, tangent);
        mapped_normal = normalize(tbn * tangent_normal);
    } else {
        /* Exact legacy tangent construction, including its unguarded inputs. */
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
    // Microfiber edge response, kept subtle; all energy still comes from
    // the scene lights. The geometry supplies the actual silhouette/relief.
    base_color *= 0.90 + 0.18 * pow(1.0 - abs(dot(N, V)), 3.0);
    float NoL = max(dot(N, L), 0.0);
    float authored_roughness = material_authored_roughness(orm.g, draw.elevation_uv.x,
                                                            draw.elevation_uv.w);
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
    if (frame.debug_view > 6.5 && frame.debug_view < 7.5) {
        const vec3 colors[5] = vec3[5](vec3(.95,.18,.12),vec3(.18,.82,.25),vec3(.15,.45,1),vec3(.95,.75,.1),vec3(.1));
        out_color = vec4(colors[shadow.cascade], 1); return;
    }
    if (frame.debug_view > 7.5 && frame.debug_view < 8.5) { out_color = vec4(shadow.coordinate,1); return; }
    if (frame.debug_view > 8.5 && frame.debug_view < 9.5) { out_color = vec4(vec3(visibility),1); return; }
    if (frame.debug_view > 9.5 && frame.debug_view < 10.5) { out_color = vec4(vec3(clamp(shadow.receiver_bias/max(frame.shadow_parameters.x,1e-5),0,1)),1); return; }
    if (frame.debug_view > 10.5 && frame.debug_view < 11.5) { float d=shadow.cascade<4u?texture(shadow_map_raw,vec3(shadow.coordinate.xy,float(shadow.cascade))).r:1.; out_color=vec4(vec3(d),1); return; }
    vec3 direct = (bxdf.diffuse + bxdf.specular) *
                  frame.sun_radiance.rgb * NoL * visibility;
    direct += base_color * (0.12 / 3.14159265) * max(dot(-N, L), 0.0) *
              frame.sun_radiance.rgb * visibility;
    int point_light_count = clamp(int(frame.point_light_options.x + 0.5), 0, 16);
    for (int light_index = 0; light_index < point_light_count; ++light_index) {
        vec4 position_radius = frame.point_light_position_radius[light_index];
        vec3 to_light = position_radius.xyz - camera_relative_position;
        float distance2 = dot(to_light, to_light);
        float distance_to_light = sqrt(max(distance2, 1e-8));
        float attenuation = local_light_attenuation(distance_to_light, position_radius.w,
                                                    frame.light_shape.x);
        if (attenuation > 0.0)
            attenuation *= point_light_visibility(camera_relative_position, normal,
                position_radius.xyz, light_index, vec3(attenuation));
        vec3 local_L = to_light / distance_to_light;
        float local_NoL = max(dot(N, local_L), 0.0);
        UeDefaultLit local_bxdf;
        if (default_lit)
            local_bxdf = ue_default_lit_bxdf(indirect_diffuse, F0, roughness, N, V, local_L);
        else
            local_bxdf = legacy_quarry_bxdf(base_color, metallic, roughness, N, V, local_L);
        vec4 color_intensity = frame.point_light_color_intensity[light_index];
        direct += (local_bxdf.diffuse + local_bxdf.specular) * color_intensity.rgb *
                  color_intensity.w * local_NoL * attenuation;
        // Thin leaves transmit some backlight. Blocker attenuation applies
        // to both sides, so foliage cannot glow through dungeon walls.
        direct += base_color * (0.12 / 3.14159265) * max(dot(-N, local_L), 0.0) *
                  color_intensity.rgb * color_intensity.w * attenuation;
    }
    /* Sky diffuse IBL (Phase B1). Falls back to the original hemispheric
       constant when no HDR was loaded, or when the F3 cycle has it switched
       off (material_factors.z), so the render is byte-identical to pre-
       Phase-B in either case. AO (orm.r) applies only here, never to direct
       or specular. */
    bool diffuse_ibl_enabled = draw.material_factors.z > 0.5;
    bool specular_ibl_enabled = draw.material_factors.w > 0.5;
    float ao_strength = draw.elevation_uv.z;
    float ao = 1.0 + ao_strength * (0.9 - 1.0);
    EnvironmentLightingResult environment = environment_evaluate(
        camera_relative_position, N, V, roughness, F0, ao,
        diffuse_ibl_enabled, specular_ibl_enabled);
    float cavity_sample = texture(cavity_map, uv).r;
    float cavity_visibility = material_visibility(cavity_sample, draw.debug.x);
    vec3 irradiance = environment.irradiance * frame.point_light_options.y;
    vec3 ambient = indirect_diffuse * irradiance * ao * cavity_visibility;
    /* Sky specular IBL (Phase B2). Its own F3 step (material_factors.w),
       one past diffuse-only, so B1 and B2 can be compared independently.
       Reuses the analytic split-sum energy terms already computed for the
       direct BRDF instead of a baked LUT. Its dedicated, view/roughness-aware
       reflection visibility consumes AO; direct light remains unaffected. */
    vec3 specular_ibl = environment.final_specular * frame.point_light_options.z;
    if (frame.debug_view > 2.5 && frame.debug_view < 3.5) { out_color = vec4(N * .5 + .5, 1); return; }
    if (frame.debug_view > 3.5 && frame.debug_view < 4.5) { out_color = vec4(vec3(authored_roughness), 1); return; }
    if (frame.debug_view > 4.5 && frame.debug_view < 5.5) { out_color = vec4(vec3(detail.geometric_floor), 1); return; }
    /* Effective material-detail diagnostic: authored (R), effective (G),
       curvature floor (B).  Green therefore appears only where D raises it. */
    if (frame.debug_view > 5.5 && frame.debug_view < 6.5) { out_color = vec4(authored_roughness, roughness, detail.geometric_floor, 1); return; }
    if (frame.debug_view > 20.5 && frame.debug_view < 21.5) { out_color = vec4(vec3(clamp((roughness - authored_roughness) * 20.0, 0.0, 1.0)), 1); return; }
    vec3 final_hdr = direct + ambient + specular_ibl;
    out_color = vec4(final_hdr, 1.0);
    vec2 current_uv = current_clip.xy / current_clip.w * 0.5 + 0.5;
    vec2 previous_uv = previous_clip.xy / previous_clip.w * 0.5 + 0.5;
    out_motion = draw.debug.y > 0.5 ? vec2(2.0) : previous_uv - current_uv;

    /* See SHADER_DUMP_LEGEND["terrain/mesh"] in renderer.c for the f0..f19 layout. */
    shader_dump(DUMP_SHADER_MESH,
                vec4(uv, roughness, NoL),
                vec4(base_color, visibility),
                vec4(N, metallic),
                vec4(camera_relative_position, default_lit ? 1.0 : 0.0),
                vec4(irradiance, ao));
    shader_dump(DUMP_SHADER_ENVIRONMENT_IBL,
                vec4(environment.reflection_direction, environment.mip),
                vec4(environment.sampled_reflection_radiance, environment.NoV),
                vec4(environment.ggx_specular_energy, environment.reflection_visibility),
                vec4(environment.unoccluded_specular, orm.r),
                vec4(environment.final_specular, roughness));
    shader_dump(DUMP_SHADER_MATERIAL_DETAIL,
                vec4(detail.authored_roughness, detail.effective_roughness,
                     detail.geometric_floor, detail.geometric_variance),
                vec4(detail.filtered_normal_length, detail.mip_variance,
                     detail.mip_kernel, detail.mip_roughness),
                vec4(ao * cavity_visibility, cavity_visibility, draw.debug.w > 0.0 ? 1.0 : 0.0,
                     normal_scale),
                vec4(final_hdr, textureQueryLod(normal_map, uv).x), vec4(0.0));
}
