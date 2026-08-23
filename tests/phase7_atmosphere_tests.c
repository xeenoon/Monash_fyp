#include "atmosphere.h"

#include <assert.h>
#include <math.h>

static void earth_preset_is_physical(void) {
    AtmosphereParameters atmosphere = atmosphere_earth();
    assert(atmosphere_parameters_valid(&atmosphere));
    assert(atmosphere.top_radius_km - atmosphere.bottom_radius_km == 100.0f);
}

static void transmittance_mapping_round_trips(void) {
    AtmosphereParameters atmosphere = atmosphere_earth();
    const float heights[] = {6360.001f, 6385.0f, 6459.0f};
    const float mus[] = {-0.05f, 0.2f, 0.95f};
    for (unsigned i = 0; i < 3; ++i) {
        float uv[2], height, mu;
        atmosphere_transmittance_to_uv(&atmosphere, heights[i], mus[i], uv);
        atmosphere_uv_to_transmittance(&atmosphere, uv, &height, &mu);
        assert(fabsf(height - heights[i]) < 0.01f);
        assert(fabsf(mu - mus[i]) < 0.002f);
    }
}

static void aerial_slices_are_monotonic_and_invertible(void) {
    AtmosphereParameters atmosphere = atmosphere_earth();
    float previous = -1.0f;
    for (unsigned i = 0; i <= 32; ++i) {
        float w = (float)i / 32.0f;
        float depth = atmosphere_aerial_slice_to_depth(
            w, atmosphere.aerial_max_distance_km);
        assert(depth >= previous);
        assert(fabsf(atmosphere_aerial_depth_to_w(
            depth, atmosphere.aerial_max_distance_km) - w) < 1e-6f);
        previous = depth;
    }
}

int main(void) {
    earth_preset_is_physical();
    transmittance_mapping_round_trips();
    aerial_slices_are_monotonic_and_invertible();
    return 0;
}
