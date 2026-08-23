#include "quarry.h"

#include "file_utils.h"
#include "renderer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum
{
	QUARRY_VERTICES = 10801,
	QUARRY_INDICES = 60003
};
enum
{
	GLB_HEADER_BYTES = 12,
	GLB_CHUNK_HEADER_BYTES = 8
};
static const float QUARRY_METALLIC_FACTOR = 0.0f;
enum
{
	QUARRY_INDICES_OFFSET = 0,
	QUARRY_UV_OFFSET = 240012,
	QUARRY_POSITION_OFFSET = 326420,
	QUARRY_NORMAL_OFFSET = 456032,
	QUARRY_TANGENT_OFFSET = 585644
};

static uint32_t le_u32(const uint8_t *p)
{
	return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static char *copy_string(const char *source)
{
	size_t bytes = strlen(source) + 1u;
	char *copy = malloc(bytes);
	if (copy)
		memcpy(copy, source, bytes);
	return copy;
}

static bool copy_floats(const uint8_t *src, size_t size, size_t offset, float *out, size_t count)
{
	if (offset > size || count > (size - offset) / sizeof(float))
		return false;
	memcpy(out, src + offset, count * sizeof(float));
	return true;
}

bool quarry_create(struct Renderer *renderer, Quarry *out, const char *directory)
{
	if (!renderer || !out || !directory)
		return false;
	*out = (Quarry){0};
	char model_path[1024], albedo_path[1024], orm_path[1024], normal_path[1024];
	snprintf(model_path, sizeof(model_path), "%s/quarry_cliff_ubhvccfda_high.glb", directory);
	snprintf(albedo_path, sizeof(albedo_path), "%s/quarry_albedo.png", directory);
	snprintf(orm_path, sizeof(orm_path), "%s/quarry_orm.png", directory);
	snprintf(normal_path, sizeof(normal_path), "%s/quarry_normal.png", directory);
	uint8_t *file = NULL;
	size_t size = 0;
	if (file_read_all(model_path, &file, &size) != FILE_READ_OK || size < 32)
	{
		fprintf(stderr, "Could not read quarry GLB: %s\n", model_path);
		free(file);
		return false;
	}
	uint32_t json_bytes = le_u32(file + GLB_HEADER_BYTES);
	size_t bin = GLB_HEADER_BYTES + GLB_CHUNK_HEADER_BYTES + json_bytes + GLB_CHUNK_HEADER_BYTES;
	if (le_u32(file) != UINT32_C(0x46546c67) || le_u32(file + 4) != 2 || bin > size ||
		QUARRY_TANGENT_OFFSET + QUARRY_VERTICES * 16u > size - bin)
	{
		fprintf(stderr, "Unsupported quarry GLB layout\n");
		free(file);
		return false;
	}
	out->owned_vertices = calloc(QUARRY_VERTICES, sizeof(*out->owned_vertices));
	out->owned_indices = malloc(QUARRY_INDICES * sizeof(*out->owned_indices));
	if (!out->owned_vertices || !out->owned_indices)
	{
		free(file);
		quarry_destroy(renderer, out);
		return false;
	}
	const uint8_t *data = file + bin;
	for (uint32_t i = 0; i < QUARRY_VERTICES; ++i)
	{
		Vertex *v = &out->owned_vertices[i];
		if (!copy_floats(data, size - bin, QUARRY_POSITION_OFFSET + i * 12u, v->position, 3) ||
			!copy_floats(data, size - bin, QUARRY_NORMAL_OFFSET + i * 12u, v->normal, 3) ||
			!copy_floats(data, size - bin, QUARRY_UV_OFFSET + i * 8u, v->texcoord, 2) ||
			!copy_floats(data, size - bin, QUARRY_TANGENT_OFFSET + i * 16u, v->tangent, 4))
		{
			free(file);
			quarry_destroy(renderer, out);
			return false;
		}
		for (uint32_t c = 0; c < 3; ++c)
			v->position[c] *= 0.01f;
	}
	if (QUARRY_INDICES_OFFSET + QUARRY_INDICES * sizeof(uint32_t) > size - bin)
	{
		free(file);
		quarry_destroy(renderer, out);
		return false;
	}
	memcpy(out->owned_indices, data + QUARRY_INDICES_OFFSET, QUARRY_INDICES * sizeof(uint32_t));
	free(file);
	out->base = (Mesh){.local_to_world = coordinate_identity_transform((WorldPosition){0}),
					   .vertices = out->owned_vertices,
					   .vertex_count = QUARRY_VERTICES,
					   .indices = out->owned_indices,
					   .index_count = QUARRY_INDICES,
					   .texture_path = copy_string(albedo_path),
					   .orm_path = copy_string(orm_path),
					   .normal_path = copy_string(normal_path)};
	out->metallic_factor = QUARRY_METALLIC_FACTOR;
	if (!out->base.texture_path || !out->base.orm_path || !out->base.normal_path)
	{
		quarry_destroy(renderer, out);
		return false;
	}
	mesh_upload(renderer, &out->base);
	return true;
}

void quarry_destroy(struct Renderer *renderer, Quarry *quarry)
{
	if (!quarry)
		return;
	if (renderer)
		mesh_destroy(renderer, &quarry->base);
	free((void *)quarry->base.texture_path);
	free((void *)quarry->base.orm_path);
	free((void *)quarry->base.normal_path);
	free(quarry->owned_vertices);
	free(quarry->owned_indices);
	*quarry = (Quarry){0};
}
