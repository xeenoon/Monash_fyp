#pragma once

#include "camera.h"
#include "coordinate.h"
#include "input.h"
#include "mesh.h"
#include "renderer.h"

#include <stdbool.h>
#include <stdint.h>

/* The overworld: the 1 km generated Swiss Alps terrain tile set, walked on
 * foot, with the game's three dungeons set into its hillsides as round,
 * hobbit-hole doors built from the dungeons' own stone and wood.
 *
 * The terrain itself is drawn by TerrainRuntime as in the terrain scene; this
 * module owns what sits on it -- a CPU height grid from the finest tile level
 * (for walking and for placing the entrances), the entrance geometry, the
 * player's position on the map, and the follow camera. */

#define OVERWORLD_ENTRANCES 3u

/* What the ground at an entrance is, which decides the dungeon's biome. */
typedef enum
{
	OVERWORLD_GROUND_SNOW,	  /* the highest ground on the map */
	OVERWORLD_GROUND_CLIFF,	  /* steep bare rock */
	OVERWORLD_GROUND_MEADOW,  /* grass-classified alpine pasture */
	OVERWORLD_GROUND_VALLEY,  /* the lowest ground: wet, sheltered */
	OVERWORLD_GROUND_SCREE,	  /* everything else: broken rock slopes */
	OVERWORLD_GROUND_COUNT
} OverworldGround;

typedef struct
{
	WorldPosition position; /* threshold of the door, on the ground */
	float facing;			/* radians; the door looks along (cos, sin) in world XZ */
	OverworldGround ground;
	float elevation_m;
} OverworldEntrance;

enum
{
	OVERWORLD_MESH_MOUND,
	OVERWORLD_MESH_RING,
	OVERWORLD_MESH_DOOR,
	OVERWORLD_MESH_KNOB,
	OVERWORLD_MESH_STEP,
	OVERWORLD_MESH_COUNT
};

typedef struct
{
	/* World-space height grid over the whole dataset (finest level). Sample
	 * (i, j) sits at origin + i * axis_u + j * axis_v. */
	float *heights;
	float *grass; /* 0..1 grass coverage per sample, from the imagery alpha */
	uint32_t samples_u, samples_v;
	double origin[3], axis_u[3], axis_v[3];
	float min_height, max_height;

	OverworldEntrance entrances[OVERWORLD_ENTRANCES];
	Mesh meshes[OVERWORLD_MESH_COUNT];
	bool uploaded[OVERWORLD_MESH_COUNT];

	/* The map camera, Google Earth style: it orbits a focus point on the
	 * ground. Dragging pans the focus, the wheel zooms, right-drag turns. */
	double focus_x, focus_z;
	float distance;		  /* metres from the focus */
	float yaw, pitch;	  /* degrees, Camera convention */
	/* Eased flight toward a selected door. */
	bool flying;
	double fly_x, fly_z;
	float fly_distance;
	float time;
} Overworld;

/* Loads the dataset's finest tiles for heights, picks three well-spaced
 * hillside sites on three different kinds of ground (random per `seed`), and
 * builds the entrance geometry. */
bool overworld_create(Renderer *renderer, Overworld *out, const char *dataset_root, uint32_t seed);
void overworld_destroy(Renderer *renderer, Overworld *world);

/* Ground height at world (x, z); NAN off the map. */
float overworld_height(const Overworld *world, double x, double z);

/* Map navigation for one frame. `drag_dx/dy` are pointer motion in pixels
 * while the left button is held, `rotate_dx` likewise for the right button,
 * `zoom` wheel notches; `pan_forward/right` the keyboard. With `controls`
 * false the camera slowly circles (behind the title screen). */
void overworld_update(Overworld *world, float drag_dx, float drag_dy, float rotate_dx, float zoom,
					  float pan_forward, float pan_right, float viewport_height_px, float dt,
					  bool controls);
/* Ease the camera in to look at entrance `i`. */
void overworld_fly_to(Overworld *world, uint32_t i);

Camera overworld_camera(const Overworld *world);

/* Window-normalised position (0..1, y down) of a world point, or false when
 * it is behind the camera. */
bool overworld_project(const Overworld *world, const Camera *camera, float aspect,
					   WorldPosition point, float *out_u, float *out_v);

/* Appends the entrance draws (also wanted in the shadow pass). */
uint32_t overworld_draws(Overworld *world, WorldPosition camera_position, RendererDraw *out,
						 uint32_t capacity);
