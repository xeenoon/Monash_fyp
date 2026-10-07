#pragma once

#include "camera.h"
#include "coordinate.h"
#include "dungeon_campaign.h"
#include "gltf_scene.h"
#include "input.h"
#include "mesh.h"
#include "renderer.h"

#include <stdbool.h>
#include <stdint.h>

/* The overworld: the complete generated terrain tile set, walked on
 * foot, with the game's dungeons set into its hillsides as round,
 * hobbit-hole doors built from the dungeons' own stone and wood.
 *
 * The terrain itself is drawn by TerrainRuntime as in the terrain scene; this
 * module owns what sits on it -- a CPU height grid from the finest tile level
 * (for walking and for placing the entrances), the entrance geometry, the
 * player's position on the map, and the follow camera. */

#define OVERWORLD_ENTRANCES DUNGEON_CAMPAIGN_LEVELS
/* Five exterior pieces per entrance, plus the active entrance's portal,
 * stairwell, torch primitives and flames. */
#define OVERWORLD_MAX_DRAWS (OVERWORLD_ENTRANCES * 5u + 64u)

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
	OVERWORLD_MESH_PORTAL,
	OVERWORLD_MESH_TUNNEL_WALL,
	OVERWORLD_MESH_TUNNEL_STAIRS,
	OVERWORLD_MESH_COUNT
};

typedef struct
{
	/* World-space height grid over the whole dataset (finest level). Sample
	 * (i, j) sits at origin + i * axis_u + j * axis_v. */
	float *heights;
	float *grass; /* 0..1 green-pasture estimate per sample, from the imagery colour */
	float *snow;  /* 0..1 bright-and-neutral estimate: snow or ice */
	uint32_t samples_u, samples_v;
	double origin[3], axis_u[3], axis_v[3];
	float min_height, max_height;

	OverworldEntrance entrances[OVERWORLD_ENTRANCES];
	Mesh meshes[OVERWORLD_MESH_COUNT];
	bool uploaded[OVERWORLD_MESH_COUNT];
	GltfScene descent_torch;
	bool descent_torch_loaded;
	Mesh descent_flame;
	bool descent_flame_uploaded;

	/* The map camera orbits a focus point on the ground. WASD pans, the arrow
	 * keys rotate/tilt, and minus/equal zoom. */
	double focus_x, focus_z;
	float distance;		  /* metres from the focus */
	float yaw, pitch;	  /* degrees, Camera convention */
	/* Eased flight toward a selected door. */
	bool flying;
	double fly_x, fly_z;
	float fly_distance;
	/* Entrance cinematic. The starting camera is captured so committing to a
	 * door never snaps, regardless of how the map was framed. */
	bool descending;
	uint32_t descent_entrance;
	float descent_progress;
	Camera descent_from;
	float time;
} Overworld;

/* Loads the dataset's finest tiles for heights, distributes entrances over
 * the terrain (deterministically per `seed`), and builds their geometry. */
/* With a positive `half_extent_m`, only that square around (centre_x,
 * centre_z) is loaded and browsable. A non-positive value loads the complete
 * dataset, matching the terrain renderer's full visible map. */
bool overworld_create(Renderer *renderer, Overworld *out, const char *dataset_root, uint32_t seed,
					  double centre_x, double centre_z, double half_extent_m);
void overworld_destroy(Renderer *renderer, Overworld *world);

/* Ground height at world (x, z); NAN off the map. */
float overworld_height(const Overworld *world, double x, double z);

/* Keyboard map navigation for one frame. Pan is WASD, rotate/tilt are the
 * arrow-key axes, and zoom is minus/equal. With `controls` false the camera
 * slowly circles behind the title screen. */
void overworld_update(Overworld *world, float pan_forward, float pan_right, float rotate_yaw,
					  float rotate_pitch, float zoom, float dt, bool controls);
/* Ease the camera in to look at entrance `i`. */
void overworld_fly_to(Overworld *world, uint32_t i);
/* Leave the entrance cinematic without changing the map framing. */
void overworld_resume_map(Overworld *world);

/* Capture the current map view, then move it to and through entrance `i` as
 * progress advances from zero to one. */
void overworld_begin_descent(Overworld *world, uint32_t i);
Camera overworld_descent_camera(Overworld *world, float progress);
/* True once the entrance has filled the view and outdoor terrain can be
 * replaced by the enclosed stairwell. */
bool overworld_descent_is_underground(const Overworld *world);
/* Torch-like point lights along the stairwell, camera-relative like the frame
 * uniform expects. Returns the number written. */
uint32_t overworld_descent_lights(const Overworld *world, WorldPosition camera,
								 vec4s *positions, vec4s *colors, uint32_t capacity);

Camera overworld_camera(const Overworld *world);

/* Window-normalised position (0..1, y down) of a world point, or false when
 * it is behind the camera. */
bool overworld_project(const Overworld *world, const Camera *camera, float aspect,
					   WorldPosition point, float *out_u, float *out_v);

/* Appends the entrance draws (also wanted in the shadow pass). */
uint32_t overworld_draws(Overworld *world, WorldPosition camera_position, RendererDraw *out,
						 uint32_t capacity, uint32_t *out_shadow_count);
