#include "shadow_cascade.h"

#include <assert.h>
#include <math.h>

static vec3s transform_point(mat4s matrix, vec3s point) {
    vec4s transformed = glms_mat4_mulv(matrix, glms_vec4(point, 1.0f));
    return (vec3s){{transformed.x / transformed.w,
                    transformed.y / transformed.w,
                    transformed.z / transformed.w}};
}

static void cascades_are_finite_nested_and_cover_the_view_axis(void) {
    ShadowCascadeConfig config = shadow_cascade_default_config(16.0f / 9.0f);
    ShadowCascadeSet cascades;
    vec3s forward = glms_vec3_normalize((vec3s){{0.6f, -0.3f, 0.7f}});
    vec3s sun = glms_vec3_normalize((vec3s){{-0.4f, -1.0f, -0.3f}});
    assert(shadow_cascade_build(&config,
        (WorldPosition){2654321.125, 840.0, -1187654.75}, forward,
        (vec3s){{0.0f, 1.0f, 0.0f}}, sun, &cascades));

    for (uint32_t i = 0; i < SHADOW_CASCADE_COUNT; ++i) {
        assert(isfinite(cascades.radius_m[i]));
        assert(cascades.radius_m[i] > 0.0f);
        if (i) assert(cascades.radius_m[i] > cascades.radius_m[i - 1u]);
        for (uint32_t column = 0; column < 4u; ++column)
            for (uint32_t row = 0; row < 4u; ++row)
                assert(isfinite(cascades.view_projection[i].raw[column][row]));

        float middle = 0.5f * ((i ? config.split_m[i - 1u]
                                  : config.near_plane_m) + config.split_m[i]);
        vec3s clip = transform_point(cascades.view_projection[i],
                                     glms_vec3_scale(forward, middle));
        assert(fabsf(clip.x) <= 1.0f);
        assert(fabsf(clip.y) <= 1.0f);
        assert(clip.z >= 0.0f && clip.z <= 1.0f);
    }
}

static void absolute_texel_snapping_is_stable_at_large_coordinates(void) {
    ShadowCascadeConfig config = shadow_cascade_default_config(16.0f / 9.0f);
    WorldPosition first_camera = {2654321.125, 840.0, -1187654.75};
    vec3s forward = glms_vec3_normalize((vec3s){{0.6f, -0.3f, 0.7f}});
    vec3s sun = glms_vec3_normalize((vec3s){{-0.4f, -1.0f, -0.3f}});
    ShadowCascadeSet first;
    assert(shadow_cascade_build(&config, first_camera, forward,
        (vec3s){{0.0f, 1.0f, 0.0f}}, sun, &first));

    float movement = (2.0f * first.radius_m[0] / config.resolution) * 0.2f;
    WorldPosition second_camera = {first_camera.x + movement,
                                   first_camera.y,
                                   first_camera.z};
    ShadowCascadeSet second;
    assert(shadow_cascade_build(&config, second_camera, forward,
        (vec3s){{0.0f, 1.0f, 0.0f}}, sun, &second));

    WorldPosition fixed_world = {
        first_camera.x + forward.x * 80.0,
        first_camera.y + forward.y * 80.0,
        first_camera.z + forward.z * 80.0,
    };
    vec3s relative_first = {{(float)(fixed_world.x - first_camera.x),
                             (float)(fixed_world.y - first_camera.y),
                             (float)(fixed_world.z - first_camera.z)}};
    vec3s relative_second = {{(float)(fixed_world.x - second_camera.x),
                              (float)(fixed_world.y - second_camera.y),
                              (float)(fixed_world.z - second_camera.z)}};
    vec3s clip_first = transform_point(first.view_projection[0], relative_first);
    vec3s clip_second = transform_point(second.view_projection[0], relative_second);
    float one_shadow_texel_ndc = 2.0f / config.resolution;
    assert(fabsf(clip_first.x - clip_second.x) <= one_shadow_texel_ndc + 1e-5f);
    assert(fabsf(clip_first.y - clip_second.y) <= one_shadow_texel_ndc + 1e-5f);
}

static void invalid_cascade_configuration_is_rejected(void) {
    ShadowCascadeConfig config = shadow_cascade_default_config(1.0f);
    config.split_m[2] = config.split_m[1];
    ShadowCascadeSet cascades;
    assert(!shadow_cascade_build(&config, (WorldPosition){0},
        (vec3s){{0.0f, 0.0f, -1.0f}}, (vec3s){{0.0f, 1.0f, 0.0f}},
        (vec3s){{0.0f, -1.0f, 0.0f}}, &cascades));
}

int main(void) {
    cascades_are_finite_nested_and_cover_the_view_axis();
    absolute_texel_snapping_is_stable_at_large_coordinates();
    invalid_cascade_configuration_is_rejected();
    return 0;
}
