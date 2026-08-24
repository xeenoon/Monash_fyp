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
    StaticMaterialParameters defaults = {1.f, 1.f, 0.f, 0.f, 0.f};
    assert(defaults.displacement_scale == 0.f);
    return 0;
}
