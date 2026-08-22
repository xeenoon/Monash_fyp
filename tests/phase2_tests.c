#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "camera.h"
#include "coordinate.h"

static void check_close(const char *label, double actual, double expected,
                        double tolerance) {
    if (fabs(actual - expected) > tolerance) {
        fprintf(stderr, "%s: got %.12g, expected %.12g (tolerance %.3g)\n",
                label, actual, expected, tolerance);
        exit(EXIT_FAILURE);
    }
}

static void test_infinite_reversed_z(void) {
    Camera camera = {.yaw = -90.0f};
    mat4s projection = camera_projection(&camera, 16.0f / 9.0f);

    vec4s near_view = {{0, 0, -CAMERA_NEAR_PLANE, 1}};
    vec4s near_clip = glms_mat4_mulv(projection, near_view);
    check_close("near depth", near_clip.z / near_clip.w, 1.0, 1e-6);

    vec4s distant_view = {{0, 0, -1000000000.0f, 1}};
    vec4s distant_clip = glms_mat4_mulv(projection, distant_view);
    check_close("distant depth", distant_clip.z / distant_clip.w,
                CAMERA_NEAR_PLANE / 1000000000.0, 1e-15);

    /* CPU equivalent of reconstruct_camera_relative() in common.glsl. */
    vec4s source = {{4.0f, -2.0f, -1234.5f, 1.0f}};
    vec4s clip = glms_mat4_mulv(projection, source);
    vec4s ndc = glms_vec4_scale(clip, 1.0f / clip.w);
    ndc.w = 1.0f;
    vec4s reconstructed = glms_mat4_mulv(glms_mat4_inv(projection), ndc);
    reconstructed = glms_vec4_scale(reconstructed, 1.0f / reconstructed.w);
    check_close("reconstructed x", reconstructed.x, source.x, 1e-3);
    check_close("reconstructed y", reconstructed.y, source.y, 1e-3);
    check_close("reconstructed z", reconstructed.z, source.z, 1e-2);
}

static void test_large_world_cancellation(void) {
    const WorldPosition origins[] = {
        {0.0, 0.0, 0.0},
        {2647500.123456, 1324.654321, -1160500.987654},
        {6378137.123456, -4250000.654321, 3210000.987654},
    };
    for (size_t i = 0; i < sizeof(origins) / sizeof(origins[0]); ++i) {
        const WorldPosition tile_origin = origins[i];
        const WorldPosition camera_world = {
            tile_origin.x + 12.25,
            tile_origin.y - 3.5,
            tile_origin.z + 0.125,
        };
        LocalToWorldTransform transform = coordinate_identity_transform(tile_origin);
        mat4s relative = coordinate_local_to_camera_relative(&transform, camera_world);

        check_close("relative x", relative.raw[3][0], -12.25, 1e-6);
        check_close("relative y", relative.raw[3][1], 3.5, 1e-6);
        check_close("relative z", relative.raw[3][2], -0.125, 1e-6);

        TileLocalPosition local = {1.25f, 2.5f, -4.0f};
        WorldPosition world = coordinate_local_to_world(&transform, local);
        CameraRelativePosition point = coordinate_camera_relative(world, camera_world);
        check_close("local relative x", point.x, -11.0, 1e-6);
        check_close("local relative y", point.y, 6.0, 1e-6);
        check_close("local relative z", point.z, -4.125, 1e-6);
    }
}

int main(void) {
    test_infinite_reversed_z();
    test_large_world_cancellation();
    puts("phase 2 projection and precision tests passed");
    return EXIT_SUCCESS;
}
