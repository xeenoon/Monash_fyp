#include "dungeon_scene.h"

#include "dungeon_cave.h"
#include "dungeon_grid.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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
	[DUNGEON_MESH_MOSS] = DUNGEON_TEXTURE_DIR "/moss.jpeg",
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

/* Untextured near-black water; real reflections are Phase 5 (fake, walls
 * only) -- for now this just keeps puddles visually distinct from the floor
 * they sit on. */
static DrawPushConstants puddle_push(const Mesh *mesh, WorldPosition camera_position)
{
	DrawPushConstants push = dungeon_push(mesh, camera_position);
	push.geometry = (vec4s){{0.02f, 0.03f, 0.045f, 1.0f}};
	push.material = (vec4s){{0.0f, 1.0f, 0.0f, 0.0f}};
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

/* The source torch's handle starts at local Y=1.14289. Seat that
 * point on the cube top; its head and light share this transform. */
static LocalToWorldTransform player_torch_transform(const DungeonScene *scene)
{
	const double scale = 1.35;
	LocalToWorldTransform transform = coordinate_identity_transform((WorldPosition){
		scene->player.position.x, scene->level.floor_y + 0.73 - 1.14289 * scale,
		scene->player.position.z + 0.06251 * scale});
	for (int i = 0; i < 3; ++i) transform.rotation[i][i] = scale;
	return transform;
}

static DungeonPoint segment_closest_point(DungeonPoint point, DungeonSegment segment)
{
	float dx = segment.b.x - segment.a.x, dz = segment.b.z - segment.a.z;
	float length2 = dx * dx + dz * dz;
	float t = length2 > 1e-12f
				 ? ((point.x - segment.a.x) * dx + (point.z - segment.a.z) * dz) / length2
				 : 0.0f;
	t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
	return (DungeonPoint){segment.a.x + dx * t, segment.a.z + dz * t};
}

static bool mount_torch(const DungeonLevel *level, DungeonLight *light,
						LocalToWorldTransform *transform)
{
	float best_distance2 = FLT_MAX;
	DungeonPoint mount = {0}, inward = {0};
	for (uint32_t i = 0; i < level->collider_count; ++i)
	{
		if (level->colliders[i].type != DUNGEON_COLLIDER_SEGMENT)
			continue;
		DungeonPoint point = segment_closest_point(light->position, level->colliders[i].segment);
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

bool dungeon_scene_create(Renderer *renderer, DungeonScene *out, DungeonLevelError *error)
{
	if (!renderer || !out)
		return false;
	*out = (DungeonScene){0};
	const char *map_override = getenv("DUNGEON_MAP");
	bool compiled;
	if (map_override)
		compiled = dungeon_grid_compile_file(map_override, 2.0f, &out->level, error);
	else
	{
		uint32_t seed = 1u;
		const char *seed_env = getenv("DUNGEON_SEED");
		if (seed_env)
			seed = (uint32_t)strtoul(seed_env, NULL, 10);
		DungeonCaveParams params = dungeon_cave_default_params(seed);
		compiled = dungeon_cave_compile(&params, &out->level, error);
	}
	if (!compiled || !dungeon_mesh_build(&out->level, &out->geometry, error))
	{
		dungeon_scene_destroy(renderer, out);
		return false;
	}
	texture_load(renderer->device, renderer->allocator, renderer->upload, &out->moss_albedo,
		DUNGEON_TEXTURE_DIR "/moss.jpeg", renderer->max_anisotropy);
	for (uint32_t i = 0; i < DUNGEON_MESH_BATCH_COUNT; ++i)
	{
		DungeonMeshBatch *batch = &out->geometry.batches[i];
		if (!batch->vertex_count)
			continue; /* e.g. no puddles were placed; leave uploaded[i] false */
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
		if (i == DUNGEON_MESH_FLOOR || i == DUNGEON_MESH_WALL)
		{
			/* The surface permutation uses binding 3 for the user's moss JPEG. */
			VkDescriptorImageInfo image = {.sampler = out->moss_albedo.sampler,
				.imageView = out->moss_albedo.view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
			VkWriteDescriptorSet write = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = out->meshes[i].material_set, .dstBinding = 3,
				.descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &image};
			vkUpdateDescriptorSets(renderer->device, 1, &write, 0, NULL);
		}
	}
	fprintf(stdout, "Moss: %u tufts, %u triangles\n",
		out->geometry.batches[DUNGEON_MESH_MOSS].vertex_count / 180u,
		out->geometry.batches[DUNGEON_MESH_MOSS].index_count / 3u);
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
	uint32_t uploaded_batches = 0;
	for (uint32_t i = 0; i < DUNGEON_MESH_BATCH_COUNT; ++i)
		uploaded_batches += out->uploaded[i] ? 1u : 0u;
	fprintf(stdout,
		   "Dungeon: %s: %u collider segments, %u puddles, %u draw batches\n",
		   map_override ? map_override : "procedural cave", out->level.collider_count,
		   out->level.puddle_count, uploaded_batches + out->torch_count * out->torch.primitive_count);
	return true;
}

uint32_t dungeon_scene_draws(DungeonScene *scene, WorldPosition camera_position,
							 RendererDraw *out, uint32_t capacity, uint32_t *out_shadow_draw_count)
{
	if (!scene || !out)
		return 0;
	scene->meshes[DUNGEON_MESH_PLAYER].local_to_world.translation =
		(WorldPosition){scene->player.position.x, scene->level.floor_y + 0.03f,
						 scene->player.position.z};
	uint32_t draw_count = 0;
	for (uint32_t i = 0; i < DUNGEON_MESH_BATCH_COUNT && draw_count < capacity; ++i)
	{
		if (i == DUNGEON_MESH_PUDDLE || !scene->uploaded[i])
			continue; /* puddles are appended last, below, and excluded from shadows */
		out[draw_count++] = (RendererDraw){.mesh = &scene->meshes[i],
											.material_set = scene->meshes[i].material_set,
											.push = i == DUNGEON_MESH_PLAYER
												? player_push(&scene->meshes[i], camera_position)
												: dungeon_push(&scene->meshes[i], camera_position),
											.static_mesh = true,
											.pipeline = (i == DUNGEON_MESH_WALL || i == DUNGEON_MESH_FLOOR)
												? RENDERER_PIPELINE_DUNGEON_SURFACE
												: i == DUNGEON_MESH_MOSS ? RENDERER_PIPELINE_DUNGEON_MOSS
												: RENDERER_PIPELINE_AUTO};
		/* Opaque surfaces use alpha to carry floor height in mesh coordinates. */
		out[draw_count - 1].push.geometry.w = scene->level.floor_y;
	}
	/* Reserve the carried torch before optional wall fixtures. */
	LocalToWorldTransform carried = player_torch_transform(scene);
	for (uint32_t i = 0; i < scene->torch.primitive_count && draw_count < capacity; ++i)
	{
		GltfPrimitive *primitive = &scene->torch.primitives[i];
		GltfMaterial *material = &scene->torch.materials[primitive->material_index];
		DrawPushConstants push = torch_push(&carried, material, camera_position);
		push.debug.y = 1.0f; /* carried object: reject stale temporal history */
		out[draw_count++] = (RendererDraw){.mesh = &primitive->mesh,
			.material_set = material->descriptor_set, .push = push, .static_mesh = true};
	}
	for (uint32_t instance = 0; instance < scene->torch_count && draw_count < capacity; ++instance)
		for (uint32_t primitive_index = 0; primitive_index < scene->torch.primitive_count;
			 ++primitive_index)
		{
			if (draw_count >= capacity)
				break;
			GltfPrimitive *primitive = &scene->torch.primitives[primitive_index];
			GltfMaterial *material = &scene->torch.materials[primitive->material_index];
			out[draw_count++] = (RendererDraw){
				.mesh = &primitive->mesh,
				.material_set = material->descriptor_set,
				.push = torch_push(&scene->torch_transforms[instance], material, camera_position),
				.static_mesh = true,
			};
		}
	if (out_shadow_draw_count)
		*out_shadow_draw_count = draw_count; /* a flat coplanar disc casts nothing worth shadowing */
	if (scene->uploaded[DUNGEON_MESH_PUDDLE] && draw_count < capacity)
		out[draw_count++] = (RendererDraw){
			.mesh = &scene->meshes[DUNGEON_MESH_PUDDLE],
			.material_set = scene->meshes[DUNGEON_MESH_PUDDLE].material_set,
			.push = puddle_push(&scene->meshes[DUNGEON_MESH_PUDDLE], camera_position),
			.static_mesh = true,
			.pipeline = RENDERER_PIPELINE_DUNGEON_PUDDLE};
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
	if (!scene || !positions || !colors || capacity == 0)
		return 0;
	/* Slot zero is always the player light, even if the light budget is full. */
	LocalToWorldTransform carried = player_torch_transform(scene);
	WorldPosition head = coordinate_local_to_world(&carried, (TileLocalPosition){0, 1.56f, -0.13f});
	CameraRelativePosition relative = coordinate_camera_relative(head, camera_position);
	positions[0] = (vec4s){{relative.x, relative.y, relative.z, 7.5f}};
	colors[0] = (vec4s){{1.0f, 0.76f, 0.48f, 32.0f}};
	uint32_t count = scene->light_count < capacity - 1u ? scene->light_count : capacity - 1u;
	for (uint32_t i = 0; i < count; ++i)
	{
		WorldPosition world = {scene->lights[i].position.x, scene->lights[i].height,
							   scene->lights[i].position.z};
		CameraRelativePosition relative = coordinate_camera_relative(world, camera_position);
		positions[i + 1u] = (vec4s){{relative.x, relative.y, relative.z, scene->lights[i].radius}};
		colors[i + 1u] = (vec4s){{scene->lights[i].color[0], scene->lights[i].color[1],
							 scene->lights[i].color[2], scene->lights[i].intensity}};
	}
	return count + 1u;
}

static int compare_blocker_distance2(const void *lhs, const void *rhs)
{
	const float *a = lhs, *b = rhs;
	return a[0] < b[0] ? -1 : (a[0] > b[0] ? 1 : 0);
}

uint32_t dungeon_scene_write_light_blockers(const DungeonScene *scene,
										WorldPosition camera_position, vec4s *blockers,
										uint32_t capacity)
{
	if (!scene || !blockers)
		return 0;
	const DungeonLevel *level = &scene->level;
	DungeonPoint camera_xz = {(float)camera_position.x, (float)camera_position.z};
	if (level->occluder_count <= capacity)
	{
		for (uint32_t i = 0; i < level->occluder_count; ++i)
		{
			DungeonSegment segment = level->occluders[i];
			blockers[i] = (vec4s){{segment.a.x - camera_xz.x, segment.a.z - camera_xz.z,
									 segment.b.x - camera_xz.x, segment.b.z - camera_xz.z}};
		}
		return level->occluder_count;
	}
	/* More occluders than the fixed-size shader array holds: keep whichever
	 * are nearest the camera. Runs once per frame over at most a few hundred
	 * segments, so a plain sort is cheap enough not to need anything
	 * cleverer. */
	float(*ranked)[2] = malloc(level->occluder_count * sizeof(*ranked));
	if (!ranked)
		return 0;
	for (uint32_t i = 0; i < level->occluder_count; ++i)
	{
		DungeonPoint closest = segment_closest_point(camera_xz, level->occluders[i]);
		float dx = camera_xz.x - closest.x, dz = camera_xz.z - closest.z;
		ranked[i][0] = dx * dx + dz * dz;
		ranked[i][1] = (float)i;
	}
	qsort(ranked, level->occluder_count, sizeof(*ranked), compare_blocker_distance2);
	for (uint32_t i = 0; i < capacity; ++i)
	{
		DungeonSegment segment = level->occluders[(uint32_t)ranked[i][1]];
		blockers[i] = (vec4s){{segment.a.x - camera_xz.x, segment.a.z - camera_xz.z,
								 segment.b.x - camera_xz.x, segment.b.z - camera_xz.z}};
	}
	free(ranked);
	return capacity;
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
		texture_destroy(renderer->device, renderer->allocator, &scene->moss_albedo);
	}
	else
		gltf_scene_destroy(NULL, &scene->torch);
	dungeon_mesh_destroy(&scene->geometry);
	dungeon_level_destroy(&scene->level);
	*scene = (DungeonScene){0};
}
