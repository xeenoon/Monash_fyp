#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "shadow_filter.glsl"
#include "pbr_common.glsl"
#include "environment_lighting.glsl"
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

layout(push_constant) uniform DrawData {
    mat4 local_to_camera_relative;
    vec4 geometry;
    vec4 elevation_uv;
    vec4 material_factors; /* x: glTF metallicFactor, y: Default Lit enabled,
                               z: Phase B diffuse sky IBL enabled (F3 cycle) */
    vec4 debug;
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


/* Static-mesh shading: photogrammetry albedo/ORM/normal sampled directly
   through the asset's own unwrapped UVs. Lighting matches the terrain forward
   pass so the two stay visually consistent. */
void main() {
    vec4 base_color_sample = texture(albedo_map, uv);
    vec3 base_color = base_color_sample.rgb * draw.geometry.rgb;
    vec3 orm = texture(orm_map, uv).rgb;
    vec3 ntex = texture(normal_map, uv).xyz * 2.0 - 1.0;
    float normal_scale = draw.elevation_uv.y;
    ntex.xy *= normal_scale;
    bool default_lit = draw.material_factors.y > 0.5;
    vec3 N;
    if (default_lit) {
        mat3 tbn = tangent_frame(normal, tangent);
        float mapped_normal_length2 = dot(ntex, ntex);
        ntex = mapped_normal_length2 > 1e-10
            ? ntex * inversesqrt(mapped_normal_length2)
            : vec3(0.0, 0.0, 1.0);
        N = normalize(tbn * ntex);
    } else {
        /* Exact legacy tangent construction, including its unguarded inputs. */
        vec3 legacy_normal = normalize(normal);
        vec3 T = normalize(tangent.xyz -
                           legacy_normal * dot(legacy_normal, tangent.xyz));
        vec3 B = normalize(cross(legacy_normal, T)) * tangent.w;
        N = normalize(mat3(T, B, legacy_normal) * ntex);
    }
    vec3 L = normalize(-frame.sun_direction.xyz);
    vec3 V = normalize(-camera_relative_position);
    /* Default Lit normal maps must shade the visible hemisphere. Preserve the
       legacy Quarry path verbatim for its F3 comparison. */
    if (default_lit && dot(N, V) < 0.0)
        N = -N;
    float NoL = max(dot(N, L), 0.0);
    float roughness = clamp(orm.g * draw.elevation_uv.x, 0.045, 1.0);
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
    /* geometry.a is the Quarry F3 Phase-C gate; ordinary glTF materials use
       their opaque base-colour alpha and therefore keep shadows enabled. */
    float visibility = mix(1.0, shadow.visibility, draw.geometry.a);
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
    /* Sky diffuse IBL (Phase B1). Falls back to the original hemispheric
       constant when no HDR was loaded, or when the F3 cycle has it switched
       off (material_factors.z), so the render is byte-identical to pre-
       Phase-B in either case. AO (orm.r) applies only here, never to direct
       or specular. */
    bool diffuse_ibl_enabled = draw.material_factors.z > 0.5;
    bool specular_ibl_enabled = draw.material_factors.w > 0.5;
    float ao_strength = draw.elevation_uv.z;
    float ao = 1.0 + ao_strength * (texture(occlusion_map, uv).r - 1.0);
    EnvironmentLightingResult environment = environment_evaluate(
        camera_relative_position, N, V, roughness, F0, ao,
        diffuse_ibl_enabled, specular_ibl_enabled);
    vec3 irradiance = environment.irradiance;
    vec3 ambient = indirect_diffuse * irradiance * ao;
    /* Sky specular IBL (Phase B2). Its own F3 step (material_factors.w),
       one past diffuse-only, so B1 and B2 can be compared independently.
       Reuses the analytic split-sum energy terms already computed for the
       direct BRDF instead of a baked LUT. Its dedicated, view/roughness-aware
       reflection visibility consumes AO; direct light remains unaffected. */
    vec3 specular_ibl = environment.final_specular;
    out_color = vec4(direct + ambient + specular_ibl, 1.0);
    vec2 current_uv = current_clip.xy / current_clip.w * 0.5 + 0.5;
    vec2 previous_uv = previous_clip.xy / previous_clip.w * 0.5 + 0.5;
    out_motion = previous_uv - current_uv;

    /* See SHADER_DUMP_LEGEND["terrain/mesh"] in renderer.c for the f0..f19 layout. */
    shader_dump(DUMP_SHADER_MESH,
                vec4(uv, roughness, NoL),
                vec4(base_color, visibility),
                vec4(N, metallic),
                vec4(camera_relative_position, default_lit ? 1.0 : 0.0),
                vec4(irradiance, orm.r));
    shader_dump(DUMP_SHADER_ENVIRONMENT_IBL,
                vec4(environment.reflection_direction, environment.mip),
                vec4(environment.sampled_reflection_radiance, environment.NoV),
                vec4(environment.ggx_specular_energy, environment.reflection_visibility),
                vec4(environment.unoccluded_specular, orm.r),
                vec4(environment.final_specular, roughness));
}
