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
	bool moving = glms_vec3_norm2(movement) > 0.0f;

	/* Hold-to-accelerate: the longer shift is held while moving, the faster we
	   go, from a gentle tap up to a hard cap. Releasing shift resets the ramp so
	   the next sprint starts slow again. */
	const float sprint_base_speed = 120.0f;    /* metres/s the instant shift is pressed */
	const float sprint_acceleration = 500.0f;  /* extra metres/s gained per second held */
	const float sprint_max_speed = 4500.0f;    /* enough to cross the 16 km map in a few seconds */
	if (sprint && moving)
		cam->sprint_charge += dt;
	else if (!sprint)
		cam->sprint_charge = 0.0f;

	if (moving)
	{
		float speed = 60.0f;
		if (sprint)
		{
			speed = sprint_base_speed + sprint_acceleration * cam->sprint_charge;
			if (speed > sprint_max_speed)
				speed = sprint_max_speed;
		}
		vec3s step = glms_vec3_scale(glms_vec3_normalize(movement), speed * dt);
		cam->position.x += (double)step.x;
		cam->position.y += (double)step.y;
		cam->position.z += (double)step.z;
	}
}
