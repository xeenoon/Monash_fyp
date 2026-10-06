#pragma once

#include "coordinate.h"
#include "gltf_scene.h"
#include "renderer.h"

#include <stdbool.h>

/* The player character: the baked Indiana Jones runtime asset
 * (assets/characters/indiana_jones/runtime, built by
 * tools/export_indiana_game.py), drawn as one static posed mesh -- torch
 * raised in his right hand -- placed by his feet and turned to face where he
 * is walking. The renderer has no skinning, so walking is a body bob and sway
 * rather than a cycle. */

typedef struct
{
	GltfScene scene;
	bool loaded;
	float feet_y;		   /* model-space height of the soles (bounds min Y) */
	float torch_head[3];   /* model-space centre of the torch's linen head */
} GameCharacter;

bool game_character_load(Renderer *renderer, GameCharacter *out, const char *runtime_dir);
void game_character_destroy(Renderer *renderer, GameCharacter *character);

/* Model to world: feet at `feet`, front (+Z in the model) along `facing`
 * radians in world XZ, with `stride` 0..1 driving the walking bob. */
LocalToWorldTransform game_character_transform(const GameCharacter *character, WorldPosition feet,
											   float facing, float stride, float time);
uint32_t game_character_draws(const GameCharacter *character, const LocalToWorldTransform *transform,
							  WorldPosition camera, RendererDraw *out, uint32_t capacity);
WorldPosition game_character_torch_head(const GameCharacter *character,
										const LocalToWorldTransform *transform);
