#include "static_material.h"

#include <math.h>

static float clamp01(float x) { return x < 0.f ? 0.f : x > 1.f ? 1.f : x; }

float static_material_authored_roughness(float orm_roughness, float gltf_factor,
                                         float roughness_bias)
{
    float r = orm_roughness * gltf_factor + roughness_bias;
    return r < .045f ? .045f : r > 1.f ? 1.f : r;
}
float static_material_curvature_floor(float variance, float curvature_strength)
{
    if (!isfinite(variance) || !isfinite(curvature_strength) || variance <= 0.f || curvature_strength <= 0.f)
        return 0.f;
    /* CPU mirror of material_detail.glsl's initial Unreal defaults. */
    return clamp01(powf(variance, .333f) * curvature_strength);
}
float static_material_normal_mip_variance(float filtered_length, float normal_strength)
{
    if (!isfinite(filtered_length) || !isfinite(normal_strength) || normal_strength <= 0.f) return 0.f;
    float len = filtered_length < 1e-4f ? 1e-4f : filtered_length > 1.f ? 1.f : filtered_length;
    return (1.f - len) / len * normal_strength * normal_strength;
}
float static_material_mip_roughness(float authored, float variance, float scale, float threshold)
{
    if (!isfinite(variance) || !isfinite(scale) || !isfinite(threshold)) return authored;
    float alpha = authored * authored, kernel = fminf(fmaxf(variance * scale, 0.f), fmaxf(threshold, 0.f));
    return sqrtf(fminf(fmaxf(alpha + kernel, alpha), 1.f));
}
float static_material_effective_detail_roughness(float authored, float geometric_floor,
                                                  float mip_roughness, bool phase_d_enabled)
{ return phase_d_enabled ? fmaxf(authored, fmaxf(geometric_floor, mip_roughness)) : authored; }
float static_material_effective_roughness(float authored, float curvature_floor)
{ return fmaxf(authored, clamp01(curvature_floor)); }
float static_material_visibility(float value, float strength)
{ return 1.f + clamp01(strength) * (clamp01(value) - 1.f); }
