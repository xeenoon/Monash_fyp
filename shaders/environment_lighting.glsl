/* Runtime environment-lighting contract. Binding 5 is deliberately the
   permanent global-cube fallback: B2.2 may choose local sources or visibility
   data behind these helpers without changing terrain or material call sites. */
#ifndef ENVIRONMENT_LIGHTING_GLSL
#define ENVIRONMENT_LIGHTING_GLSL

layout(set = 0, binding = 4) uniform EnvironmentUniforms {
    vec4 sh[9];
    vec4 env_params; /* x: enabled, y: diffuse intensity, z: specular intensity,
                         w: global specular-cube maximum mip index */
} env;

layout(set = 0, binding = 5) uniform samplerCube env_specular;

struct EnvironmentLightingResult {
    vec3 irradiance;
    vec3 sampled_reflection_radiance;
    float mip;
    vec3 ggx_specular_energy;
    vec3 unoccluded_specular;
    float reflection_visibility;
    vec3 final_specular;
    vec3 reflection_direction;
    float NoV;
};

vec3 environment_irradiance(vec3 N) {
    vec3 result =
        env.sh[0].xyz * 0.282095 + env.sh[1].xyz * (0.488603 * N.y) +
        env.sh[2].xyz * (0.488603 * N.z) + env.sh[3].xyz * (0.488603 * N.x) +
        env.sh[4].xyz * (1.092548 * N.x * N.y) + env.sh[5].xyz * (1.092548 * N.y * N.z) +
        env.sh[6].xyz * (0.315392 * (3.0 * N.z * N.z - 1.0)) +
        env.sh[7].xyz * (1.092548 * N.x * N.z) + env.sh[8].xyz * (0.546274 * (N.x * N.x - N.y * N.y));
    return max(result, vec3(0.0));
}

/* B2.2 seam: position, direction, and roughness are all supplied even though
   the B2.1 global fallback only consumes direction and roughness. */
vec3 environment_reflection_source(vec3 camera_relative_position,
                                   vec3 reflection_direction, float roughness,
                                   out float mip) {
    mip = roughness * env.env_params.w;
    return textureLod(env_specular, reflection_direction, mip).rgb;
}

float environment_reflection_visibility(float NoV, float material_ao, float roughness) {
    float exponent = exp2(-16.0 * roughness - 1.0);
    return clamp(pow(clamp(NoV + material_ao, 0.0, 2.0), exponent) - 1.0 + material_ao,
                 0.0, 1.0);
}

/* Kept distinct from source selection and visibility so later environment
   systems can replace any one policy without touching material call sites. */
vec3 environment_reflection_brdf_response(float roughness, float NoV, vec3 F0) {
    return ue_compute_ggx_spec_energy_terms(roughness, NoV, F0).specular_energy;
}

EnvironmentLightingResult environment_evaluate(vec3 camera_relative_position,
                                                vec3 N, vec3 V, float roughness,
                                                vec3 F0, float material_ao,
                                                bool diffuse_enabled, bool specular_enabled) {
    EnvironmentLightingResult result;
    bool loaded = env.env_params.x > 0.5;
    result.irradiance = loaded && diffuse_enabled
        ? environment_irradiance(N) * env.env_params.y
        : vec3(0.045 + 0.10 * max(N.y, 0.0));
    result.NoV = clamp(dot(N, V), 0.0, 1.0);
    result.reflection_direction = reflect(-V, N);
    result.mip = roughness * env.env_params.w;
    result.sampled_reflection_radiance = vec3(0.0);
    result.ggx_specular_energy = vec3(0.0);
    result.unoccluded_specular = vec3(0.0);
    result.reflection_visibility = 0.0;
    result.final_specular = vec3(0.0);
    if (loaded && specular_enabled) {
        result.sampled_reflection_radiance = environment_reflection_source(
            camera_relative_position, result.reflection_direction, roughness, result.mip);
        result.ggx_specular_energy = environment_reflection_brdf_response(
            roughness, result.NoV, F0);
        result.unoccluded_specular = result.sampled_reflection_radiance *
            env.env_params.z * result.ggx_specular_energy;
        result.reflection_visibility = environment_reflection_visibility(
            result.NoV, material_ao, roughness);
        result.final_specular = result.unoccluded_specular * result.reflection_visibility;
    }
    return result;
}

#endif
