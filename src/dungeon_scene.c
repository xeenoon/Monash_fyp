#include "dungeon_scene.h"

#include "dungeon_grid.h"

#include <float.h>
#include <math.h>
#include <stdio.h>

#ifndef DUNGEON_TEXTURE_DIR
#define DUNGEON_TEXTURE_DIR "textures/dungeon/runtime"
#endif

#ifndef DUNGEON_TORCH_PATH
#define DUNGEON_TORCH_PATH "assets/dungeons/torch/walltorch.gltf"
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

static DrawPushConstants player_push(const Mesh *mesh, WorldPosition camera_position)
{
	DrawPushConstants push = dungeon_push(mesh, camera_position);
	push.geometry = (vec4s){{0.12f, 0.42f, 0.95f, 1.0f}};
	push.material = (vec4s){{0.0f, 1.0f, 1.0f, 1.0f}};
	push.debug.y = 1.0f; /* dynamic object: reject stale temporal history */
	push.debug.w = 0.0f;
	return push;
}

static DrawPushConstants torch_push(const LocalToWorldTransform *transform,
									const GltfMaterial *material, WorldPosition camera_position)
{
	return (DrawPushConstants){
		.local_to_camera_relative =
			coordinate_local_to_camera_relative(transform, camera_position),
		.geometry = {{material->base_color_factor[0], material->base_color_factor[1],
					  material->base_color_factor[2], material->base_color_factor[3]}},
		.elevation_uv = {{material->roughness_factor, material->normal_scale,
						  material->occlusion_strength, 0.0f}},
		.material = {{material->metallic_factor, 1.0f, 1.0f, 1.0f}},
		.debug = {{0.0f, 0.0f, 1.0f, 1.0f}},
	};
}

static bool mount_torch(const DungeonLevel *level, DungeonLight *light,
						LocalToWorldTransform *transform)
{
	float best_distance2 = FLT_MAX;
	DungeonPoint mount = {0}, inward = {0};
	for (uint32_t i = 0; i < level->collider_count; ++i)
	{
		DungeonRect wall = level->colliders[i].bounds;
		DungeonPoint point = {
			fminf(fmaxf(light->position.x, wall.min.x), wall.max.x),
			fminf(fmaxf(light->position.z, wall.min.z), wall.max.z),
		};
		float dx = light->position.x - point.x, dz = light->position.z - point.z;
		float distance2 = dx * dx + dz * dz;
		if (distance2 > 1e-6f && distance2 < best_distance2)
		{
			float inverse_distance = 1.0f / sqrtf(distance2);
			best_distance2 = distance2;
			mount = point;
			inward = (DungeonPoint){dx * inverse_distance, dz * inverse_distance};
		}
	}
	if (best_distance2 == FLT_MAX)
		return false;
	/* The source torch projects along local -Z. Keep its back plate just clear
	   of the wall and put the point light at the head of the mesh. */
	mount.x += inward.x * 0.01f;
	mount.z += inward.z * 0.01f;
	light->position =
		(DungeonPoint){mount.x + inward.x * 0.18f, mount.z + inward.z * 0.18f};
	light->height = level->floor_y + 1.58f;
	double yaw = atan2(-(double)inward.x, -(double)inward.z);
	*transform = coordinate_rotation_y(
		yaw, (WorldPosition){mount.x, level->floor_y + 0.05f, mount.z});
	return true;
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
	dungeon_player_init(&out->player, out->level.spawn);
	out->light_count = dungeon_lighting_build(&out->level, out->lights, DUNGEON_MAX_LIGHTS);
	GltfLoadError torch_error = {0};
	if (gltf_scene_create(renderer, DUNGEON_TORCH_PATH,
					  &(GltfLoadOptions){.placement =
									 coordinate_identity_transform((WorldPosition){0})},
					  &out->torch, &torch_error) != GLTF_LOAD_OK)
	{
		if (error)
			snprintf(error->message, sizeof(error->message), "could not load dungeon torch: %.220s",
					 torch_error.message);
		dungeon_scene_destroy(renderer, out);
		return false;
	}
	for (uint32_t i = 0; i < out->light_count; ++i)
		if (mount_torch(&out->level, &out->lights[i],
						&out->torch_transforms[out->torch_count]))
			++out->torch_count;
	fprintf(stdout, "Dungeon: %s: %u floor runs, %u wall solids, %u draw batches\n", map_path,
			out->level.surface_count, out->level.solid_count,
			DUNGEON_MESH_BATCH_COUNT + out->torch_count * out->torch.primitive_count);
	return true;
}

uint32_t dungeon_scene_draws(DungeonScene *scene, WorldPosition camera_position,
							 RendererDraw *out, uint32_t capacity)
{
	if (!scene || !out || capacity < DUNGEON_MESH_BATCH_COUNT)
		return 0;
	scene->meshes[DUNGEON_MESH_PLAYER].local_to_world.translation =
		(WorldPosition){scene->player.position.x, scene->level.floor_y + 0.03f,
						 scene->player.position.z};
	for (uint32_t i = 0; i < DUNGEON_MESH_BATCH_COUNT; ++i)
		out[i] = (RendererDraw){.mesh = &scene->meshes[i],
								.material_set = scene->meshes[i].material_set,
								.push = i == DUNGEON_MESH_PLAYER
									? player_push(&scene->meshes[i], camera_position)
									: dungeon_push(&scene->meshes[i], camera_position),
								.static_mesh = true};
	uint32_t draw_count = DUNGEON_MESH_BATCH_COUNT;
	for (uint32_t instance = 0; instance < scene->torch_count; ++instance)
		for (uint32_t primitive_index = 0; primitive_index < scene->torch.primitive_count;
			 ++primitive_index)
		{
			if (draw_count >= capacity)
				return draw_count;
			GltfPrimitive *primitive = &scene->torch.primitives[primitive_index];
			GltfMaterial *material = &scene->torch.materials[primitive->material_index];
			out[draw_count++] = (RendererDraw){
				.mesh = &primitive->mesh,
				.material_set = material->descriptor_set,
				.push = torch_push(&scene->torch_transforms[instance], material, camera_position),
				.static_mesh = true,
			};
		}
	return draw_count;
}

bool dungeon_scene_update(DungeonScene *scene, float move_forward, float move_right,
						  float camera_yaw_degrees, float dt)
{
	return dungeon_player_update(&scene->player, &scene->level, move_forward, move_right,
								 camera_yaw_degrees, dt);
}

uint32_t dungeon_scene_write_lights(const DungeonScene *scene, WorldPosition camera_position,
								   vec4s *positions, vec4s *colors, uint32_t capacity)
{
	if (!scene || !positions || !colors)
		return 0;
	uint32_t count = scene->light_count < capacity ? scene->light_count : capacity;
	for (uint32_t i = 0; i < count; ++i)
	{
		WorldPosition world = {scene->lights[i].position.x, scene->lights[i].height,
							   scene->lights[i].position.z};
		CameraRelativePosition relative = coordinate_camera_relative(world, camera_position);
		positions[i] = (vec4s){{relative.x, relative.y, relative.z, scene->lights[i].radius}};
		colors[i] = (vec4s){{scene->lights[i].color[0], scene->lights[i].color[1],
							 scene->lights[i].color[2], scene->lights[i].intensity}};
	}
	return count;
}

uint32_t dungeon_scene_write_light_blockers(const DungeonScene *scene,
										WorldPosition camera_position, vec4s *blockers,
										uint32_t capacity)
{
	if (!scene || !blockers)
		return 0;
	uint32_t count = scene->level.collider_count < capacity ? scene->level.collider_count : capacity;
	for (uint32_t i = 0; i < count; ++i)
	{
		DungeonRect bounds = scene->level.colliders[i].bounds;
		blockers[i] = (vec4s){{bounds.min.x - (float)camera_position.x,
								 bounds.min.z - (float)camera_position.z,
								 bounds.max.x - (float)camera_position.x,
								 bounds.max.z - (float)camera_position.z}};
	}
	return count;
}

void dungeon_scene_destroy(Renderer *renderer, DungeonScene *scene)
{
	if (!scene)
		return;
	if (renderer)
	{
		gltf_scene_destroy(renderer, &scene->torch);
		for (uint32_t i = 0; i < DUNGEON_MESH_BATCH_COUNT; ++i)
			if (scene->uploaded[i])
				mesh_destroy(renderer, &scene->meshes[i]);
	}
	else
		gltf_scene_destroy(NULL, &scene->torch);
	dungeon_mesh_destroy(&scene->geometry);
	dungeon_level_destroy(&scene->level);
	*scene = (DungeonScene){0};
}
