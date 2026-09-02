#pragma once

#include "dungeon_level.h"
#include "dungeon_mesh.h"
#include "renderer.h"

#include <stdbool.h>

typedef struct
{
	DungeonLevel level;
	DungeonMeshData geometry;
	Mesh meshes[DUNGEON_MESH_BATCH_COUNT];
	bool uploaded[DUNGEON_MESH_BATCH_COUNT];
} DungeonScene;

bool dungeon_scene_create(Renderer *renderer, const char *map_path, DungeonScene *out,
						  DungeonLevelError *error);
uint32_t dungeon_scene_draws(DungeonScene *scene, WorldPosition camera_position,
							 RendererDraw *out, uint32_t capacity);
void dungeon_scene_destroy(Renderer *renderer, DungeonScene *scene);

