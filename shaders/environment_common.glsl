/* Shared face-direction/equirect mapping and GGX importance sampling for the
   B2 specular-IBL prefilter compute shaders (environment_equirect_to_cube.comp,
   environment_prefilter.comp). */
#ifndef ENVIRONMENT_COMMON_GLSL
#define ENVIRONMENT_COMMON_GLSL

const float ENV_PI = 3.14159265358979323846;

/* Vulkan cube image layer order: +X,-X,+Y,-Y,+Z,-Z. uv is a texel's face-local
   coordinate in [-1, 1] (x right, y down in face space). */
vec3 environment_face_direction(uint face, vec2 uv) {
    if (face == 0u) return normalize(vec3(1.0, -uv.y, -uv.x));
    if (face == 1u) return normalize(vec3(-1.0, -uv.y, uv.x));
    if (face == 2u) return normalize(vec3(uv.x, 1.0, uv.y));
    if (face == 3u) return normalize(vec3(uv.x, -1.0, -uv.y));
    if (face == 4u) return normalize(vec3(uv.x, -uv.y, 1.0));
    return normalize(vec3(-uv.x, -uv.y, -1.0));
}

/* Must match environment_project_sh9's basis exactly (environment.c): v=0 at
   +Y, u=0 toward +X, u increasing toward +Z. Keeping the two in lockstep means
   the specular reflection and the diffuse SH agree on which way the loaded
   HDR is facing. */
vec2 environment_direction_to_equirect_uv(vec3 dir) {
    float phi = atan(dir.z, dir.x);
    float u = phi / (2.0 * ENV_PI);
    if (u < 0.0) u += 1.0;
    float v = acos(clamp(dir.y, -1.0, 1.0)) / ENV_PI;
    return vec2(u, v);
}

/* Karis "Real Shading in Unreal Engine 4": Hammersley low-discrepancy set. */
float environment_radical_inverse_vdc(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

vec2 environment_hammersley(uint i, uint n) {
    return vec2(float(i) / float(n), environment_radical_inverse_vdc(i));
}

/* GGX importance sample of the halfway vector around N, in world space. */
vec3 environment_importance_sample_ggx(vec2 xi, float roughness, vec3 N) {
    float a = roughness * roughness;
    float phi = 2.0 * ENV_PI * xi.x;
    float cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    vec3 h = vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);
    return normalize(tangent * h.x + bitangent * h.y + N * h.z);
}

#endif
