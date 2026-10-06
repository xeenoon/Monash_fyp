#pragma once
#include "gltf_scene.h"
#include "renderer.h"
#define DUNGEON_MONK_MAX_DRAWS 16u
typedef struct
{
	GltfScene model;
	bool loaded, active;
	WorldPosition position;
	float yaw;
} DungeonMonkArt;
bool dungeon_monk_load(Renderer *r, DungeonMonkArt *art);
void dungeon_monk_destroy(Renderer *r, DungeonMonkArt *art);
uint32_t dungeon_monk_draws(DungeonMonkArt *art, WorldPosition camera, RendererDraw *out,
							uint32_t capacity);
