#include "game_character.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Reads `"key": "value"` out of the exporter's manifest. */
static bool manifest_string(const char *text, const char *key, char *out, size_t capacity)
{
	char pattern[64];
	snprintf(pattern, sizeof(pattern), "\"%s\"", key);
	const char *at = strstr(text, pattern);
	if (!at || !(at = strchr(at + strlen(pattern), '"')))
		return false;
	const char *end = strchr(at + 1, '"');
	if (!end || (size_t)(end - at - 1) >= capacity)
		return false;
	memcpy(out, at + 1, (size_t)(end - at - 1));
	out[end - at - 1] = '\0';
	return true;
}

static bool manifest_float(const char *text, const char *key, float *out)
{
	char pattern[64];
	snprintf(pattern, sizeof(pattern), "\"%s\"", key);
	const char *at = strstr(text, pattern);
	return at && (at = strchr(at, ':')) && sscanf(at + 1, " %f", out) == 1;
}

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

	/* Animation, when the asset has a skin and both clips. */
	GltfScene *scene = &out->scene;
	char torch_bone[64] = "hand_r";
	manifest_string(text, "torch_bone", torch_bone, sizeof(torch_bone));
	out->walk_speed = 1.4f;
	out->run_speed = 3.2f;
	manifest_float(text, "walk_speed", &out->walk_speed);
	manifest_float(text, "run_speed", &out->run_speed);
	manifest_float(text, "walk_phase", &out->walk_phase);
	manifest_float(text, "run_phase", &out->run_phase);
	out->idle_clip = gltf_scene_find_clip(scene, "Idle");
	out->walk_clip = gltf_scene_find_clip(scene, "Walk");
	out->run_clip = gltf_scene_find_clip(scene, "Run");
	out->torch_node = gltf_scene_find_node(scene, torch_bone);
	uint32_t largest = 0;
	bool skinned = scene->skin_count > 0 && scene->primitive_count <= GAME_CHARACTER_MAX_PRIMITIVES;
	for (uint32_t i = 0; i < scene->primitive_count; ++i)
		if (scene->primitives[i].mesh.vertex_count > largest)
			largest = scene->primitives[i].mesh.vertex_count;
	if (skinned && out->idle_clip != GLTF_NO_NODE && out->walk_clip != GLTF_NO_NODE)
	{
		out->pose = calloc(scene->node_count, sizeof(*out->pose));
		out->walk_pose = calloc(scene->node_count, sizeof(*out->walk_pose));
		out->run_pose = calloc(scene->node_count, sizeof(*out->run_pose));
		out->world = calloc(scene->node_count, sizeof(*out->world));
		out->skinned = calloc(largest, sizeof(*out->skinned));
		if (!out->pose || !out->walk_pose || !out->run_pose || !out->world || !out->skinned)
		{
			game_character_destroy(renderer, out);
			return false;
		}
		for (uint32_t i = 0; i < scene->primitive_count; ++i)
		{
			Mesh *mesh = &scene->primitives[i].mesh;
			out->original[i] = mesh->vertex_buffer;
			for (int b = 0; b < 2; ++b)
				out->dynamic[b][i] = gpu_buffer_create(
					renderer->device, renderer->allocator, sizeof(Vertex) * mesh->vertex_count,
					VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
					VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		}
		out->torch_skin = scene->primitives[0].skin;
		out->animated = true;
		game_character_animate(out, 0.0f, 0.0f);
	}
	printf("Character: %u primitives, %s%s\n", scene->primitive_count,
		   out->animated ? "skinned, Idle + Walk" : "static pose",
		   out->animated && out->run_clip != GLTF_NO_NODE ? " + Run" : "");
	return true;
}

static float clip_duration(const GltfScene *scene, uint32_t clip)
{
	return clip < scene->clip_count && scene->clips[clip].duration > 0.0f
			   ? scene->clips[clip].duration
			   : 1.0f;
}

/* a = a + (b - a) * w per node: lerped translation and scale, normalised
 * lerp of rotation on the near hemisphere. */
static void blend_pose(GltfTransform *a, const GltfTransform *b, uint32_t count, float w)
{
	if (w <= 0.0f)
		return;
	for (uint32_t n = 0; n < count; ++n)
	{
		for (int k = 0; k < 3; ++k)
		{
			a[n].translation[k] += (b[n].translation[k] - a[n].translation[k]) * w;
			a[n].scale[k] += (b[n].scale[k] - a[n].scale[k]) * w;
		}
		float dot = 0.0f;
		for (int k = 0; k < 4; ++k)
			dot += a[n].rotation[k] * b[n].rotation[k];
		float sign = dot < 0.0f ? -1.0f : 1.0f, length = 0.0f;
		for (int k = 0; k < 4; ++k)
		{
			a[n].rotation[k] += (b[n].rotation[k] * sign - a[n].rotation[k]) * w;
			length += a[n].rotation[k] * a[n].rotation[k];
		}
		length = sqrtf(length);
		for (int k = 0; k < 4 && length > 0.0f; ++k)
			a[n].rotation[k] /= length;
	}
}

void game_character_animate(GameCharacter *c, float speed_mps, float dt)
{
	if (!c->animated)
		return;
	GltfScene *scene = &c->scene;
	bool has_run = c->run_clip != GLTF_NO_NODE && c->run_speed > c->walk_speed;
	/* Idle -> Walk over the first half of walking pace, Walk -> Run between
	 * the two clips' own speeds; both eased so a stop is not a snap. */
	float target = fminf(fmaxf(speed_mps / (c->walk_speed * 0.5f), 0.0f), 1.0f);
	float run_target = has_run ? fminf(fmaxf((speed_mps - c->walk_speed) /
												 (c->run_speed - c->walk_speed), 0.0f), 1.0f)
							   : 0.0f;
	float ease = fminf(1.0f, dt * 8.0f);
	c->blend += (target - c->blend) * ease;
	c->run_blend += (run_target - c->run_blend) * ease;
	/* The gait phase advances at the cadence of whichever clip dominates,
	 * scaled so the feet cover the ground actually travelled: each clip at
	 * its own speed plays at 1x. */
	float walk_duration = clip_duration(scene, c->walk_clip);
	float run_duration = has_run ? clip_duration(scene, c->run_clip) : walk_duration;
	float pace = fmaxf(speed_mps, c->walk_speed * 0.5f);
	float walk_hz = pace / c->walk_speed / walk_duration;
	float run_hz = has_run ? pace / c->run_speed / run_duration : walk_hz;
	c->gait_phase = fmodf(c->gait_phase + dt * (walk_hz + (run_hz - walk_hz) * c->run_blend), 1.0f);
	c->idle_time = fmodf(c->idle_time + dt, clip_duration(scene, c->idle_clip));
	gltf_scene_rest_pose(scene, c->pose);
	gltf_clip_sample(scene, c->idle_clip, c->idle_time, c->pose);
	gltf_scene_rest_pose(scene, c->walk_pose);
	gltf_clip_sample(scene, c->walk_clip, fmodf(c->gait_phase + c->walk_phase, 1.0f) * walk_duration,
					 c->walk_pose);
	if (has_run)
	{
		gltf_scene_rest_pose(scene, c->run_pose);
		gltf_clip_sample(scene, c->run_clip, fmodf(c->gait_phase + c->run_phase, 1.0f) * run_duration,
						 c->run_pose);
		blend_pose(c->walk_pose, c->run_pose, scene->node_count, c->run_blend);
	}
	blend_pose(c->pose, c->walk_pose, scene->node_count, c->blend);
	gltf_scene_world_matrices(scene, c->pose, c->world);
	c->torch_matrix = gltf_scene_joint_matrix(scene, c->torch_skin, c->torch_node, c->world);
	c->frame ^= 1u;
	for (uint32_t i = 0; i < scene->primitive_count; ++i)
	{
		GltfPrimitive *primitive = &scene->primitives[i];
		gltf_scene_skin(scene, i, c->world, c->skinned);
		memcpy(c->dynamic[c->frame][i].allocation.mapped, c->skinned,
			   sizeof(Vertex) * primitive->mesh.vertex_count);
		primitive->mesh.vertex_buffer = c->dynamic[c->frame][i];
	}
}

void game_character_destroy(Renderer *renderer, GameCharacter *character)
{
	for (uint32_t i = 0; i < character->scene.primitive_count && i < GAME_CHARACTER_MAX_PRIMITIVES;
		 ++i)
	{
		if (character->original[i].buffer)
			character->scene.primitives[i].mesh.vertex_buffer = character->original[i];
		for (int b = 0; b < 2; ++b)
			if (character->dynamic[b][i].buffer)
				gpu_buffer_destroy(renderer->device, renderer->allocator, &character->dynamic[b][i]);
	}
	free(character->pose);
	free(character->walk_pose);
	free(character->run_pose);
	free(character->world);
	free(character->skinned);
	if (character->loaded)
		gltf_scene_destroy(renderer, &character->scene);
	*character = (GameCharacter){0};
}

LocalToWorldTransform game_character_transform(const GameCharacter *character, WorldPosition feet,
											   float facing, float stride, float time)
{
	/* rotation_y(a) sends model +Z to (sin a, cos a); the facing is (cos, sin). */
	if (character->animated)
		stride = 0.0f; /* the clips carry the gait */
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
	vec4s head = {{character->torch_head[0], character->torch_head[1], character->torch_head[2], 1.0f}};
	if (character->animated)
		head = glms_mat4_mulv(character->torch_matrix, head);
	return coordinate_local_to_world(transform, (TileLocalPosition){head.x, head.y, head.z});
}
