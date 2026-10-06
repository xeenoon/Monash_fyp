#include "dungeon_guardian.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef DUNGEON_GUARDIAN_PATH
#define DUNGEON_GUARDIAN_PATH "assets/characters/medusa/medusa.gltf"
#endif

bool dungeon_guardian_load(Renderer *renderer, DungeonGuardianArt *art)
{
	*art = (DungeonGuardianArt){0};
	GltfLoadError error = {0};
	if (gltf_scene_create(renderer, DUNGEON_GUARDIAN_PATH,
		&(GltfLoadOptions){.placement = coordinate_identity_transform((WorldPosition){0})},
		&art->model, &error) != GLTF_LOAD_OK)
	{
		fprintf(stderr, "Guardian: %s\n", error.message);
		return false;
	}
	art->clip = gltf_scene_find_clip(&art->model, "Serpentine");
	art->pose = calloc(art->model.node_count, sizeof(*art->pose));
	art->world = calloc(art->model.node_count, sizeof(*art->world));
	if (!art->pose || !art->world || art->clip == GLTF_NO_NODE ||
		art->model.primitive_count > DUNGEON_GUARDIAN_MAX_DRAWS)
	{
		dungeon_guardian_destroy(renderer, art);
		return false;
	}
	art->loaded = true;
	fprintf(stdout, "Guardian: Medusa, %u curved-surface draws\n", art->model.primitive_count);
	return true;
}

uint32_t dungeon_guardian_draws(DungeonGuardianArt *art, WorldPosition camera,
							   RendererDraw *out, uint32_t capacity)
{
	if (!art->loaded || !art->active)
		return 0;
	gltf_scene_rest_pose(&art->model, art->pose);
	const GltfClip *clip = &art->model.clips[art->clip];
	gltf_clip_sample(&art->model, art->clip, fmodf(art->time, clip->duration), art->pose);
	gltf_scene_world_matrices(&art->model, art->pose, art->world);
	WorldPosition position = art->position;
	position.y += .018 * sinf(art->time * 2.0f);
	LocalToWorldTransform placement = coordinate_rotation_y(art->yaw, position);
	uint32_t count = 0;
	for (uint32_t i = 0; i < art->model.primitive_count && count < capacity; ++i)
	{
		const GltfPrimitive *p = &art->model.primitives[i];
		const GltfMaterial *m = &art->model.materials[p->material_index];
		LocalToWorldTransform transform = p->node == GLTF_NO_NODE ? placement
			: coordinate_compose(&placement, art->world[p->node]);
		DrawPushConstants push = {
			.local_to_camera_relative = coordinate_local_to_camera_relative(&transform, camera),
			.geometry = {{m->base_color_factor[0], m->base_color_factor[1],
				m->base_color_factor[2], 1.0f}},
			.elevation_uv = {{m->roughness_factor, m->normal_scale, m->occlusion_strength, 0}},
			.material = {{m->metallic_factor, 1, 1, 1}},
			.debug = {{0, 1, 1, 1}}, /* Moving sculpture: reject stale TAA history. */
		};
		/* Amber eyes heat to vermilion when the guardian acquires the player. */
		if (p->material_index == 2 && art->hunting)
			push.geometry = (vec4s){{1.0f, .11f, .025f, 1.0f}};
		out[count++] = (RendererDraw){.mesh = &p->mesh,
			.material_set = m->descriptor_set, .push = push, .static_mesh = true};
	}
	return count;
}

void dungeon_guardian_destroy(Renderer *renderer, DungeonGuardianArt *art)
{
	free(art->pose);
	free(art->world);
	gltf_scene_destroy(renderer, &art->model);
	*art = (DungeonGuardianArt){0};
}
