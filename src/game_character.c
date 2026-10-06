#include "game_character.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Reads `"key": [a, b, c]` out of the exporter's manifest. */
static bool manifest_vec3(const char *text, const char *key, float out[3])
{
	char pattern[64];
	snprintf(pattern, sizeof(pattern), "\"%s\"", key);
	const char *at = strstr(text, pattern);
	if (!at || !(at = strchr(at, '[')))
		return false;
	return sscanf(at, "[ %f , %f , %f ]", &out[0], &out[1], &out[2]) == 3;
}

bool game_character_load(Renderer *renderer, GameCharacter *out, const char *runtime_dir)
{
	*out = (GameCharacter){0};
	char path[1024];
	snprintf(path, sizeof(path), "%s/manifest.json", runtime_dir);
	FILE *file = fopen(path, "rb");
	if (!file)
	{
		fprintf(stderr, "Character: no runtime asset at %s (run tools/export_indiana_game.py)\n",
				runtime_dir);
		return false;
	}
	char text[4096] = {0};
	fread(text, 1, sizeof(text) - 1, file);
	fclose(file);
	float bounds_min[3] = {0};
	if (!manifest_vec3(text, "bounds_min", bounds_min) ||
		!manifest_vec3(text, "torch_head", out->torch_head))
	{
		fprintf(stderr, "Character: manifest %s is missing bounds or torch_head\n", path);
		return false;
	}
	out->feet_y = bounds_min[1];
	snprintf(path, sizeof(path), "%s/indiana.gltf", runtime_dir);
	GltfLoadError error = {0};
	if (gltf_scene_create(renderer, path,
						  &(GltfLoadOptions){.placement =
												 coordinate_identity_transform((WorldPosition){0})},
						  &out->scene, &error) != GLTF_LOAD_OK)
	{
		fprintf(stderr, "Character: could not load %s: %s\n", path, error.message);
		return false;
	}
	out->loaded = true;
	return true;
}

void game_character_destroy(Renderer *renderer, GameCharacter *character)
{
	if (character->loaded)
		gltf_scene_destroy(renderer, &character->scene);
	*character = (GameCharacter){0};
}

LocalToWorldTransform game_character_transform(const GameCharacter *character, WorldPosition feet,
											   float facing, float stride, float time)
{
	/* rotation_y(a) sends model +Z to (sin a, cos a); the facing is (cos, sin). */
	float sway = stride * 0.05f * sinf(time * 9.0f);
	LocalToWorldTransform t =
		coordinate_rotation_y(1.5707963 - facing + sway,
							  (WorldPosition){feet.x, feet.y - character->feet_y +
														  stride * 0.035 * fabs(sin(time * 9.0)),
											  feet.z});
	return t;
}

uint32_t game_character_draws(const GameCharacter *character, const LocalToWorldTransform *transform,
							  WorldPosition camera, RendererDraw *out, uint32_t capacity)
{
	if (!character->loaded)
		return 0;
	uint32_t count = 0;
	for (uint32_t i = 0; i < character->scene.primitive_count && count < capacity; ++i)
	{
		const GltfPrimitive *primitive = &character->scene.primitives[i];
		const GltfMaterial *material = &character->scene.materials[primitive->material_index];
		out[count++] = (RendererDraw){
			.mesh = &primitive->mesh,
			.material_set = material->descriptor_set,
			.static_mesh = true,
			.push = {
				.local_to_camera_relative =
					coordinate_local_to_camera_relative(transform, camera),
				/* w < 0: double-sided -- the garments are open shells (mesh.frag). */
				.geometry = {{material->base_color_factor[0], material->base_color_factor[1],
							  material->base_color_factor[2], -1.0f}},
				.elevation_uv = {{material->roughness_factor, material->normal_scale,
								  material->occlusion_strength, 0.0f}},
				.material = {{material->metallic_factor, 1.0f, 1.0f, 1.0f}},
				/* y: moving object -- reject stale temporal history. */
				.debug = {{0.0f, 1.0f, 1.0f, 1.0f}},
			}};
	}
	return count;
}

WorldPosition game_character_torch_head(const GameCharacter *character,
										const LocalToWorldTransform *transform)
{
	return coordinate_local_to_world(
		transform, (TileLocalPosition){character->torch_head[0], character->torch_head[1],
									   character->torch_head[2]});
}
