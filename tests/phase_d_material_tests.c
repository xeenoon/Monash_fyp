#include "static_material.h"
#include <assert.h>
#include <math.h>

int main(void)
{
    /* Default controls preserve GLB's multiplicative roughness contract. */
    assert(static_material_authored_roughness(.8f, .7f, 0.f) == .56f);
    assert(static_material_authored_roughness(1.f, 1.f, 0.f) == 1.f);
    assert(static_material_curvature_floor(0.f, 1.f) == 0.f);
    float a = .55f, low = static_material_curvature_floor(.01f, .2f);
    float high = static_material_curvature_floor(.04f, .2f);
    assert(isfinite(low) && low >= 0.f && low <= 1.f && high >= low);
    assert(fabsf(static_material_curvature_floor(.04f, 1.f) - .342f) < .01f);
    assert(static_material_effective_roughness(a, low) >= a);
    assert(static_material_effective_roughness(a, 1.f) == 1.f);
    assert(static_material_visibility(.2f, 0.f) == 1.f);
    assert(fabsf(static_material_visibility(.2f, 1.f) - .2f) < 1e-6f);
    /* Normal-map mip filtering: unit vectors add no variance, and shorter
       filtered vectors monotonically broaden perceptual GGX roughness. */
    const float lengths[] = {1.f, .99f, .95f, .8f, .5f};
    float previous = a;
    for (unsigned i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i) {
        float variance = static_material_normal_mip_variance(lengths[i], 1.f);
        float roughness = static_material_mip_roughness(a, variance, .25f, .20f);
        assert(isfinite(variance) && isfinite(roughness));
        assert(roughness >= previous && roughness <= 1.f);
        previous = roughness;
    }
    assert(static_material_normal_mip_variance(1.f, 1.f) == 0.f);
    assert(static_material_normal_mip_variance(.5f, 0.f) == 0.f);
    assert(static_material_mip_roughness(a, 100.f, .25f, .20f) <= sqrtf(a*a + .20f) + 1e-6f);
    assert(static_material_effective_detail_roughness(a, .9f, .6f, false) == a);
    assert(static_material_effective_detail_roughness(a, .9f, .6f, true) == .9f);
    const float invalid[] = {0.f, -1.f, NAN, INFINITY};
    for (unsigned i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
        float v = static_material_normal_mip_variance(invalid[i], 1.f);
        assert(isfinite(v) && v >= 0.f);
    }
    StaticMaterialParameters defaults = {1.f, 1.f, 0.f, 0.f, 0.f};
    assert(defaults.displacement_scale == 0.f);
    return 0;
}
