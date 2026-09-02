#pragma once

#include "dungeon_level.h"
#include "dungeon_lighting.h"
#include "dungeon_mesh.h"
#include "dungeon_player.h"
#include "renderer.h"

#include <stdbool.h>

typedef struct
{
	DungeonLevel level;
	DungeonMeshData geometry;
	Mesh meshes[DUNGEON_MESH_BATCH_COUNT];
	bool uploaded[DUNGEON_MESH_BATCH_COUNT];
	DungeonPlayer player;
	DungeonLight lights[DUNGEON_MAX_LIGHTS];
	uint32_t light_count;
} DungeonScene;

bool dungeon_scene_create(Renderer *renderer, const char *map_path, DungeonScene *out,
						  DungeonLevelError *error);
uint32_t dungeon_scene_draws(DungeonScene *scene, WorldPosition camera_position,
							 RendererDraw *out, uint32_t capacity);
bool dungeon_scene_update(DungeonScene *scene, float move_forward, float move_right,
						  float camera_yaw_degrees, float dt);
uint32_t dungeon_scene_write_lights(const DungeonScene *scene, WorldPosition camera_position,
								   vec4s *positions, vec4s *colors, uint32_t capacity);
uint32_t dungeon_scene_write_light_blockers(const DungeonScene *scene,
										WorldPosition camera_position, vec4s *blockers,
										uint32_t capacity);
void dungeon_scene_destroy(Renderer *renderer, DungeonScene *scene);
