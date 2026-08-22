#include "camera.h"

#include <math.h>

vec3s camera_forward(const Camera *cam) {
    float yaw = glm_rad(cam->yaw);
    float pitch = glm_rad(cam->pitch);
    vec3s direction = {{cosf(yaw)*cosf(pitch), sinf(pitch), sinf(yaw)*cosf(pitch)}};
    return glms_vec3_normalize(direction);
}

mat4s camera_view(const Camera *cam) {
    vec3s forward = camera_forward(cam);
    vec3s center = glms_vec3_add(cam->position, forward);
    return glms_lookat(cam->position, center, (vec3s){{0, 1, 0}});
}

mat4s camera_projection(const Camera *cam, float aspect) {
    (void)cam;
    mat4s projection = glms_perspective(glm_rad(60.0f), aspect, 0.1f, 100.0f);
    projection.raw[1][1] *= -1.0f; /* Vulkan's framebuffer Y axis points down. */
    return projection;
}

void camera_update(Camera *cam, float move_forward, float move_right,
                   float look_dx, float look_dy, bool sprint, float dt) {
    const float mouse_sensitivity = 0.04f;
    cam->yaw += look_dx * mouse_sensitivity;
    cam->pitch -= look_dy * mouse_sensitivity;
    if (cam->pitch > 89.0f) cam->pitch = 89.0f;
    if (cam->pitch < -89.0f) cam->pitch = -89.0f;

    vec3s forward = camera_forward(cam);
    vec3s right = glms_vec3_normalize(glms_vec3_cross(forward, (vec3s){{0, 1, 0}}));
    vec3s movement = glms_vec3_add(glms_vec3_scale(forward, move_forward),
                                   glms_vec3_scale(right, move_right));
    if (glms_vec3_norm2(movement) > 0.0f) {
        float speed = sprint ? 8.0f : 3.5f;
        vec3s step = glms_vec3_scale(glms_vec3_normalize(movement), speed * dt);
        cam->position = glms_vec3_add(cam->position, step);
    }
}
