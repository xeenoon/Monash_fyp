#pragma once

#include "coordinate.h"
#include "gltf_scene.h"
#include "gpu_buffer.h"
#include "renderer.h"

#include <stdbool.h>

/* The player character: the baked Indiana Jones runtime asset
 * (assets/characters/indiana_jones/runtime, built by
 * tools/export_indiana_game.py), placed by his feet and turned to face where
 * he is walking.
 *
 * When the asset is skinned and carries "Idle" and "Walk" clips, he is
 * animated: both clips are sampled, blended by walking speed, and the mesh is
 * skinned on the CPU (gltf_scene_skin) into one of two host-visible vertex
 * buffers that alternate frame to frame, so the GPU never reads a buffer the
 * CPU is writing. A static asset falls back to a body bob and sway. */

#define GAME_CHARACTER_MAX_PRIMITIVES 8u

typedef struct
{
	GltfScene scene;
	bool loaded;
	float feet_y;		 /* model-space height of the soles (bounds min Y) */
	float torch_head[3]; /* bind-space centre of the torch's linen head */

	bool animated;
	uint32_t idle_clip, walk_clip;
	uint32_t torch_node, torch_skin; /* the joint the torch is held by */
	float walk_speed;				 /* m/s the Walk clip covers at 1x */
	float idle_time, walk_time, blend;
	GltfTransform *pose, *walk_pose;
	mat4s *world;
	mat4s torch_matrix; /* current skinning matrix of the torch joint */
	Vertex *skinned;	/* scratch, sized to the largest primitive */
	GpuBuffer original[GAME_CHARACTER_MAX_PRIMITIVES];
	GpuBuffer dynamic[2][GAME_CHARACTER_MAX_PRIMITIVES];
	uint32_t frame;
} GameCharacter;

bool game_character_load(Renderer *renderer, GameCharacter *out, const char *runtime_dir);
void game_character_destroy(Renderer *renderer, GameCharacter *character);

/* Advance and apply the animation for one frame at ground speed `speed_mps`.
 * Call once per rendered frame, before the character's draws are recorded. */
void game_character_animate(GameCharacter *character, float speed_mps, float dt);

/* Model to world: feet at `feet`, front (+Z in the model) along `facing`
 * radians in world XZ, with `stride` 0..1 driving the walking bob when the
 * asset has no animation of its own. */
LocalToWorldTransform game_character_transform(const GameCharacter *character, WorldPosition feet,
											   float facing, float stride, float time);
uint32_t game_character_draws(const GameCharacter *character, const LocalToWorldTransform *transform,
							  WorldPosition camera, RendererDraw *out, uint32_t capacity);
WorldPosition game_character_torch_head(const GameCharacter *character,
										const LocalToWorldTransform *transform);
