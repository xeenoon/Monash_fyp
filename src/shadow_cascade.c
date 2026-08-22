#include "shadow_cascade.h"

#include <math.h>

typedef struct { double x, y, z; } DVec3;

static DVec3 add(DVec3 a, DVec3 b) {
    return (DVec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

static DVec3 subtract(DVec3 a, DVec3 b) {
    return (DVec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static DVec3 scale(DVec3 v, double s) {
    return (DVec3){v.x * s, v.y * s, v.z * s};
}

static double dot(DVec3 a, DVec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static DVec3 cross(DVec3 a, DVec3 b) {
    return (DVec3){a.y * b.z - a.z * b.y,
                   a.z * b.x - a.x * b.z,
                   a.x * b.y - a.y * b.x};
}

static DVec3 normalized(DVec3 v) {
    double length = sqrt(dot(v, v));
    return length > 1e-12 ? scale(v, 1.0 / length) : (DVec3){0};
}

static DVec3 from_vec3(vec3s v) {
    return (DVec3){v.x, v.y, v.z};
}

static bool finite_vec(DVec3 v) {
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

ShadowCascadeConfig shadow_cascade_default_config(float aspect) {
    return (ShadowCascadeConfig){
        .near_plane_m = 0.5f,
        .split_m = {120.0f, 350.0f, 1000.0f, 3000.0f},
        .vertical_fov_radians = glm_rad(60.0f),
        .aspect = aspect,
        .resolution = SHADOW_MAP_RESOLUTION,
    };
}

static mat4s cascade_matrix(DVec3 right, DVec3 up, DVec3 forward,
                            double centre_x, double centre_y, double centre_z,
                            double radius) {
    double z_extent = radius * 4.0;
    mat4s matrix = GLMS_MAT4_ZERO_INIT;
    matrix.raw[0][0] = (float)(right.x / radius);
    matrix.raw[1][0] = (float)(right.y / radius);
    matrix.raw[2][0] = (float)(right.z / radius);
    matrix.raw[3][0] = (float)(-centre_x / radius);
    matrix.raw[0][1] = (float)(up.x / radius);
    matrix.raw[1][1] = (float)(up.y / radius);
    matrix.raw[2][1] = (float)(up.z / radius);
    matrix.raw[3][1] = (float)(-centre_y / radius);
    matrix.raw[0][2] = (float)(forward.x / (2.0 * z_extent));
    matrix.raw[1][2] = (float)(forward.y / (2.0 * z_extent));
    matrix.raw[2][2] = (float)(forward.z / (2.0 * z_extent));
    matrix.raw[3][2] = (float)(-(centre_z - z_extent) / (2.0 * z_extent));
    matrix.raw[3][3] = 1.0f;
    return matrix;
}

/* Adapted from Wicked Engine wiRenderer.cpp:2936-3060
   CreateDirLightShadowCams (Turánszki János, MIT). We retain only its frustum
   bounding sphere, Z extrusion, and texel-grid snap because those are the parts
   that prevent directional cascades from changing scale or swimming. */
bool shadow_cascade_build(const ShadowCascadeConfig *config,
                          WorldPosition camera_world,
                          vec3s camera_forward,
                          vec3s camera_up,
                          vec3s light_direction,
                          ShadowCascadeSet *out) {
    if (!config || !out || config->resolution == 0u ||
        !(config->near_plane_m > 0.0f) || !(config->aspect > 0.0f) ||
        !(config->vertical_fov_radians > 0.0f))
        return false;
    for (uint32_t i = 0; i < SHADOW_CASCADE_COUNT; ++i)
        if (!(config->split_m[i] > (i ? config->split_m[i - 1u]
                                      : config->near_plane_m)))
            return false;

    DVec3 view_forward = normalized(from_vec3(camera_forward));
    DVec3 view_right = normalized(cross(view_forward, from_vec3(camera_up)));
    DVec3 view_up = normalized(cross(view_right, view_forward));
    DVec3 light_forward = normalized(from_vec3(light_direction));
    DVec3 reference = fabs(light_forward.y) > 0.95
        ? (DVec3){0.0, 0.0, 1.0} : (DVec3){0.0, 1.0, 0.0};
    DVec3 light_right = normalized(cross(reference, light_forward));
    DVec3 light_up = normalized(cross(light_forward, light_right));
    if (!finite_vec(view_forward) || !finite_vec(view_right) ||
        !finite_vec(view_up) || !finite_vec(light_forward) ||
        !finite_vec(light_right) || !finite_vec(light_up) ||
        dot(view_right, view_right) < 0.5 || dot(light_right, light_right) < 0.5)
        return false;

    DVec3 camera = {camera_world.x, camera_world.y, camera_world.z};
    double tangent = tan(0.5 * config->vertical_fov_radians);
    double slice_near = config->near_plane_m;
    for (uint32_t cascade = 0; cascade < SHADOW_CASCADE_COUNT; ++cascade) {
        double slice_far = config->split_m[cascade];
        DVec3 corners[8];
        uint32_t cursor = 0;
        for (uint32_t plane = 0; plane < 2u; ++plane) {
            double distance = plane ? slice_far : slice_near;
            double half_y = distance * tangent;
            double half_x = half_y * config->aspect;
            DVec3 centre = scale(view_forward, distance);
            for (int y = -1; y <= 1; y += 2)
                for (int x = -1; x <= 1; x += 2)
                    corners[cursor++] = add(centre,
                        add(scale(view_right, x * half_x), scale(view_up, y * half_y)));
        }

        DVec3 centre = {0};
        for (uint32_t i = 0; i < 8u; ++i) centre = add(centre, corners[i]);
        centre = scale(centre, 1.0 / 8.0);
        double radius = 0.0;
        for (uint32_t i = 0; i < 8u; ++i)
            radius = fmax(radius, sqrt(dot(subtract(corners[i], centre),
                                          subtract(corners[i], centre))));
        /* Quantising the radius prevents tiny floating-point/FOV changes from
           resizing the projection and swimming every shadow texel. */
        radius = ceil(radius * 16.0) / 16.0;
        double texel = (2.0 * radius) / config->resolution;

        /* Wicked snaps the light-space AABB. We perform the snap using the
           absolute double camera position, then subtract it back out before
           narrowing to a camera-relative float matrix. */
        DVec3 absolute_centre = add(camera, centre);
        double camera_x = dot(camera, light_right);
        double camera_y = dot(camera, light_up);
        double snapped_x = floor(dot(absolute_centre, light_right) / texel) * texel;
        double snapped_y = floor(dot(absolute_centre, light_up) / texel) * texel;
        double relative_x = snapped_x - camera_x;
        double relative_y = snapped_y - camera_y;
        double relative_z = dot(centre, light_forward);

        out->view_projection[cascade] = cascade_matrix(
            light_right, light_up, light_forward,
            relative_x, relative_y, relative_z, radius);
        out->radius_m[cascade] = (float)radius;
        slice_near = slice_far;
    }
    return true;
}
