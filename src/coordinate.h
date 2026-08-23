#pragma once

#include <cglm/struct.h>

/* Coordinate-space vocabulary used by the renderer.
   - GeodeticPosition is longitude/latitude/height source data.
   - WorldPosition is an absolute, double-precision ECEF or projected position.
   - TileLocalPosition is a small float position stored in a terrain vertex.
   - CameraRelativePosition is a small float position submitted to the GPU.

   LocalToWorldTransform maps tile-local metres to the absolute world frame.
   rotation is column-major, like cglm/GLSL matrices; translation remains double
   precision for the lifetime of the tile. */
typedef struct
{
	double longitude_degrees;
	double latitude_degrees;
	double height_m;
} GeodeticPosition;

typedef struct
{
	double x, y, z;
} WorldPosition;
typedef struct
{
	float x, y, z;
} TileLocalPosition;
typedef struct
{
	float x, y, z;
} CameraRelativePosition;

typedef struct
{
	double rotation[3][3]; /* [column][row] */
	WorldPosition translation;
} LocalToWorldTransform;

LocalToWorldTransform coordinate_identity_transform(WorldPosition translation);

/* The world-space subtraction is deliberately performed before the result is
   narrowed to float. Never replace this with two float casts and a GPU-side
   subtraction: Earth-sized coordinates would lose ground-level precision. */
CameraRelativePosition coordinate_camera_relative(WorldPosition world, WorldPosition camera_world);
mat4s coordinate_local_to_camera_relative(const LocalToWorldTransform *local_to_world,
										  WorldPosition camera_world);
WorldPosition coordinate_local_to_world(const LocalToWorldTransform *local_to_world,
										TileLocalPosition local);
