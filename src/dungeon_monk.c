#include "dungeon_monk.h"
#include <stdio.h>
#ifndef DUNGEON_MONK_PATH
#define DUNGEON_MONK_PATH "assets/characters/praying_monk/praying_monk.gltf"
#endif
bool dungeon_monk_load(Renderer *r, DungeonMonkArt *art)
{
	*art = (DungeonMonkArt){0};
	GltfLoadError error = {0};
	if (gltf_scene_create(
			r, DUNGEON_MONK_PATH,
			&(GltfLoadOptions){.placement = coordinate_identity_transform((WorldPosition){0})},
			&art->model, &error) != GLTF_LOAD_OK)
	{
		fprintf(stderr, "Monk: %s\n", error.message);
		return false;
	}
	if (art->model.primitive_count > DUNGEON_MONK_MAX_DRAWS)
	{
		dungeon_monk_destroy(r, art);
		return false;
	}
	art->loaded = true;
	return true;
}
void dungeon_monk_destroy(Renderer *r, DungeonMonkArt *art)
{
	gltf_scene_destroy(r, &art->model);
	*art = (DungeonMonkArt){0};
}
uint32_t dungeon_monk_draws(DungeonMonkArt *art, WorldPosition camera, RendererDraw *out,
							uint32_t capacity)
{
	if (!art->loaded || !art->active)
		return 0;
	LocalToWorldTransform transform = coordinate_rotation_y(art->yaw, art->position);
	uint32_t count = 0;
	for (uint32_t i = 0; i < art->model.primitive_count && count < capacity; ++i)
	{
		const GltfPrimitive *p = &art->model.primitives[i];
		const GltfMaterial *m = &art->model.materials[p->material_index];
		DrawPushConstants push = {
			.local_to_camera_relative = coordinate_local_to_camera_relative(&transform, camera),
			.geometry = {{m->base_color_factor[0], m->base_color_factor[1], m->base_color_factor[2],
						  1}},
			.elevation_uv = {{m->roughness_factor, m->normal_scale, m->occlusion_strength, 0}},
			.material = {{m->metallic_factor, 1, 1, 1}},
			.debug = {{0, 1, 1, 0}}};
		out[count++] = (RendererDraw){
			.mesh = &p->mesh, .material_set = m->descriptor_set, .push = push, .static_mesh = true};
	}
	return count;
}
