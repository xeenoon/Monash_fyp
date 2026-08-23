#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

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

float shadow_visibility(vec3 position, vec3 N) {
    float distance_m = length(position);
    uint cascade = distance_m < frame.shadow_splits.x ? 0u :
                   distance_m < frame.shadow_splits.y ? 1u :
                   distance_m < frame.shadow_splits.z ? 2u : 3u;
    vec3 receiver = position + N * frame.shadow_parameters.x;
    vec4 clip = frame.shadow_view_projection[cascade] * vec4(receiver, 1.0);
    vec3 c = clip.xyz / clip.w;
    vec3 coord = vec3(c.xy * 0.5 + 0.5, c.z);
    if (any(lessThan(coord.xy, vec2(0.0))) || any(greaterThan(coord.xy, vec2(1.0)))) return 1.0;
    return texture(shadow_map, vec4(coord.xy, float(cascade), coord.z));
}

void main() {
    vec3 base = texture(albedo_map, uv).rgb;
    vec3 orm = texture(orm_map, uv).rgb;
    vec3 ntex = texture(normal_map, uv).xyz * 2.0 - 1.0;
    vec3 T = normalize(tangent.xyz - normal * dot(normal, tangent.xyz));
    vec3 B = normalize(cross(normal, T)) * tangent.w;
    vec3 N = normalize(mat3(T, B, normal) * ntex);
    vec3 L = normalize(-frame.sun_direction.xyz);
    vec3 V = normalize(-camera_relative_position);
    vec3 H = normalize(L + V);
    float NoL = max(dot(N, L), 0.0);
    float NoV = max(dot(N, V), 0.0);
    float roughness = clamp(orm.g, 0.045, 1.0);
    float metallic = orm.b;
    float alpha = roughness * roughness;
    float a2 = alpha * alpha;
    float NoH = max(dot(N, H), 0.0);
    float denom = max(3.14159265 * pow(NoH * NoH * (a2 - 1.0) + 1.0, 2.0), 1e-5);
    float D = a2 / denom;
    float k = (roughness + 1.0); k = k * k / 8.0;
    float G = NoL / mix(NoL, 1.0, k) * NoV / mix(NoV, 1.0, k);
    vec3 F0 = mix(vec3(0.04), base, metallic);
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);
    vec3 spec = D * G * F / max(4.0 * NoL * NoV, 1e-5);
    vec3 diffuse = (1.0 - metallic) * base / 3.14159265;
    float visibility = shadow_visibility(camera_relative_position, N);
    vec3 direct = (diffuse + spec) * frame.sun_radiance.rgb * NoL * visibility;
    vec3 ambient = base * (0.045 + 0.10 * max(N.y, 0.0)) * orm.r;
    out_color = vec4(direct + ambient, 1.0);
    vec2 current_uv = current_clip.xy / current_clip.w * 0.5 + 0.5;
    vec2 previous_uv = previous_clip.xy / previous_clip.w * 0.5 + 0.5;
    out_motion = previous_uv - current_uv;
}
