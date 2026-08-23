#include "camera.h"

#include <math.h>

vec3s camera_forward(const Camera *cam)
{
	float yaw = glm_rad(cam->yaw);
	float pitch = glm_rad(cam->pitch);
	vec3s direction = {{cosf(yaw) * cosf(pitch), sinf(pitch), sinf(yaw) * cosf(pitch)}};
	return glms_vec3_normalize(direction);
}

mat4s camera_view(const Camera *cam)
{
	vec3s forward = camera_forward(cam);
	return glms_lookat((vec3s){{0, 0, 0}}, forward, (vec3s){{0, 1, 0}});
}

mat4s camera_projection(const Camera *cam, float aspect)
{
	(void)cam;
	/* For right-handed view space (visible z < 0), this maps the near plane to
	   depth 1 and approaches 0 at infinity: depth = near / -view_z. */
	const float focal_length = 1.0f / tanf(glm_rad(60.0f) * 0.5f);
	mat4s projection = GLMS_MAT4_ZERO_INIT;
	projection.raw[0][0] = focal_length / aspect;
	projection.raw[1][1] = -focal_length; /* Vulkan framebuffer Y points down. */
	projection.raw[2][3] = -1.0f;
	projection.raw[3][2] = CAMERA_NEAR_PLANE;
	return projection;
}

void camera_update(Camera *cam, float move_forward, float move_right, float look_dx, float look_dy,
				   bool sprint, float dt)
{
	const float mouse_sensitivity = 0.04f;
	cam->yaw += look_dx * mouse_sensitivity;
	cam->pitch -= look_dy * mouse_sensitivity;
	if (cam->pitch > 89.0f)
		cam->pitch = 89.0f;
	if (cam->pitch < -89.0f)
		cam->pitch = -89.0f;

	vec3s forward = camera_forward(cam);
	vec3s right = glms_vec3_normalize(glms_vec3_cross(forward, (vec3s){{0, 1, 0}}));
	vec3s movement =
		glms_vec3_add(glms_vec3_scale(forward, move_forward), glms_vec3_scale(right, move_right));
	if (glms_vec3_norm2(movement) > 0.0f)
	{
		float speed = sprint ? 200.0f : 60.0f;
		vec3s step = glms_vec3_scale(glms_vec3_normalize(movement), speed * dt);
		cam->position.x += (double)step.x;
		cam->position.y += (double)step.y;
		cam->position.z += (double)step.z;
	}
}
