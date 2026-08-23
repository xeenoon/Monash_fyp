#include "coordinate.h"

#include <math.h>

LocalToWorldTransform coordinate_identity_transform(WorldPosition translation)
{
	LocalToWorldTransform transform = {.translation = translation};
	transform.rotation[0][0] = 1.0;
	transform.rotation[1][1] = 1.0;
	transform.rotation[2][2] = 1.0;
	return transform;
}

LocalToWorldTransform coordinate_rotation_y(double radians, WorldPosition translation)
{
	LocalToWorldTransform transform = {.translation = translation};
	double c = cos(radians), s = sin(radians);
	transform.rotation[0][0] = c;
	transform.rotation[0][2] = -s;
	transform.rotation[1][1] = 1.0;
	transform.rotation[2][0] = s;
	transform.rotation[2][2] = c;
	return transform;
}

CameraRelativePosition coordinate_camera_relative(WorldPosition world, WorldPosition camera_world)
{
	return (CameraRelativePosition){
		(float)(world.x - camera_world.x),
		(float)(world.y - camera_world.y),
		(float)(world.z - camera_world.z),
	};
}

mat4s coordinate_local_to_camera_relative(const LocalToWorldTransform *local_to_world,
										  WorldPosition camera_world)
{
	CameraRelativePosition relative =
		coordinate_camera_relative(local_to_world->translation, camera_world);
	mat4s result = GLMS_MAT4_IDENTITY_INIT;
	for (int column = 0; column < 3; ++column)
		for (int row = 0; row < 3; ++row)
			result.raw[column][row] = (float)local_to_world->rotation[column][row];
	result.raw[3][0] = relative.x;
	result.raw[3][1] = relative.y;
	result.raw[3][2] = relative.z;
	return result;
}

WorldPosition coordinate_local_to_world(const LocalToWorldTransform *local_to_world,
										TileLocalPosition local)
{
	const double v[3] = {local.x, local.y, local.z};
	double world[3] = {
		local_to_world->translation.x,
		local_to_world->translation.y,
		local_to_world->translation.z,
	};
	for (int column = 0; column < 3; ++column)
		for (int row = 0; row < 3; ++row)
			world[row] += local_to_world->rotation[column][row] * v[column];
	return (WorldPosition){world[0], world[1], world[2]};
}
