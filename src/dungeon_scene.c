#include "dungeon_scene.h"

#include "dungeon_grid.h"

#include <stdio.h>

#ifndef DUNGEON_TEXTURE_DIR
#define DUNGEON_TEXTURE_DIR "textures/dungeon/runtime"
#endif

static const char *const ALBEDO_PATHS[DUNGEON_MESH_BATCH_COUNT] = {
	DUNGEON_TEXTURE_DIR "/floor_albedo.jpg",
	DUNGEON_TEXTURE_DIR "/wall_albedo.jpg",
	DUNGEON_TEXTURE_DIR "/exit_albedo.jpg",
};

static const char *const ORM_PATHS[DUNGEON_MESH_BATCH_COUNT] = {
	DUNGEON_TEXTURE_DIR "/floor_orm.png",
	DUNGEON_TEXTURE_DIR "/wall_orm.png",
	DUNGEON_TEXTURE_DIR "/exit_orm.png",
};

static const char *const NORMAL_PATHS[DUNGEON_MESH_BATCH_COUNT] = {
	DUNGEON_TEXTURE_DIR "/floor_normal.png",
	DUNGEON_TEXTURE_DIR "/wall_normal.png",
	DUNGEON_TEXTURE_DIR "/exit_normal.png",
};

static DrawPushConstants dungeon_push(const Mesh *mesh, WorldPosition camera_position)
{
	return (DrawPushConstants){
		.local_to_camera_relative =
			coordinate_local_to_camera_relative(&mesh->local_to_world, camera_position),
		.geometry = {{1.0f, 1.0f, 1.0f, 1.0f}},
		.elevation_uv = {{1.0f, 1.0f, 1.0f, 0.0f}},
		.material = {{1.0f, 1.0f, 1.0f, 1.0f}},
		.debug = {{0.0f, 0.0f, 1.0f, 1.0f}},
	};
}

bool dungeon_scene_create(Renderer *renderer, const char *map_path, DungeonScene *out,
						  DungeonLevelError *error)
{
	if (!renderer || !map_path || !out)
		return false;
	*out = (DungeonScene){0};
	if (!dungeon_grid_compile_file(map_path, 2.0f, &out->level, error) ||
		!dungeon_mesh_build(&out->level, &out->geometry, error))
	{
		dungeon_scene_destroy(renderer, out);
		return false;
	}
	for (uint32_t i = 0; i < DUNGEON_MESH_BATCH_COUNT; ++i)
	{
		DungeonMeshBatch *batch = &out->geometry.batches[i];
		out->meshes[i] = (Mesh){
			.local_to_world = coordinate_identity_transform((WorldPosition){0}),
			.vertices = batch->vertices,
			.vertex_count = batch->vertex_count,
			.indices = batch->indices,
			.index_count = batch->index_count,
			.texture_path = ALBEDO_PATHS[i],
			.orm_path = ORM_PATHS[i],
			.normal_path = NORMAL_PATHS[i],
		};
		mesh_upload(renderer, &out->meshes[i]);
		out->uploaded[i] = true;
	}
	fprintf(stdout, "Dungeon: %s: %u floor runs, %u wall solids, %u draw batches\n", map_path,
			out->level.surface_count, out->level.solid_count, DUNGEON_MESH_BATCH_COUNT);
	return true;
}

uint32_t dungeon_scene_draws(DungeonScene *scene, WorldPosition camera_position,
							 RendererDraw *out, uint32_t capacity)
{
	if (!scene || !out || capacity < DUNGEON_MESH_BATCH_COUNT)
		return 0;
	for (uint32_t i = 0; i < DUNGEON_MESH_BATCH_COUNT; ++i)
		out[i] = (RendererDraw){.mesh = &scene->meshes[i],
								.material_set = scene->meshes[i].material_set,
								.push = dungeon_push(&scene->meshes[i], camera_position),
								.static_mesh = true};
	return DUNGEON_MESH_BATCH_COUNT;
}

void dungeon_scene_destroy(Renderer *renderer, DungeonScene *scene)
{
	if (!scene)
		return;
	if (renderer)
		for (uint32_t i = 0; i < DUNGEON_MESH_BATCH_COUNT; ++i)
			if (scene->uploaded[i])
				mesh_destroy(renderer, &scene->meshes[i]);
	dungeon_mesh_destroy(&scene->geometry);
	dungeon_level_destroy(&scene->level);
	*scene = (DungeonScene){0};
}

