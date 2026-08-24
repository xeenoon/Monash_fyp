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
    return clamp01(sqrtf(variance) * curvature_strength);
}
float static_material_effective_roughness(float authored, float curvature_floor)
{ return fmaxf(authored, clamp01(curvature_floor)); }
float static_material_visibility(float value, float strength)
{ return 1.f + clamp01(strength) * (clamp01(value) - 1.f); }
