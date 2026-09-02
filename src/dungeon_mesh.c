#include "dungeon_mesh.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct
{
	DungeonMeshBatch *batch;
	uint32_t vertex_capacity;
	uint32_t index_capacity;
} Builder;

static bool mesh_fail(DungeonLevelError *error, const char *format, ...)
{
	if (error)
	{
		va_list args;
		va_start(args, format);
		vsnprintf(error->message, sizeof(error->message), format, args);
		va_end(args);
	}
	return false;
}

static bool builder_create(Builder *builder, DungeonMeshBatch *batch, uint32_t quads,
						   float material_width_m)
{
	if (quads > UINT32_MAX / 6u)
		return false;
	builder->vertex_capacity = quads * 4u;
	builder->index_capacity = quads * 6u;
	batch->vertices = calloc(builder->vertex_capacity, sizeof(*batch->vertices));
	batch->indices = calloc(builder->index_capacity, sizeof(*batch->indices));
	batch->material_width_m = material_width_m;
	builder->batch = batch;
	return batch->vertices && batch->indices;
}

static bool append_quad(Builder *builder, float positions[4][3], const float normal[3],
						const float tangent[4], float uv[4][2])
{
	DungeonMeshBatch *batch = builder->batch;
	if (batch->vertex_count + 4u > builder->vertex_capacity ||
		batch->index_count + 6u > builder->index_capacity)
		return false;
	uint32_t base = batch->vertex_count;
	for (uint32_t i = 0; i < 4; ++i)
	{
		Vertex *vertex = &batch->vertices[batch->vertex_count++];
		for (uint32_t component = 0; component < 3; ++component)
		{
			vertex->position[component] = positions[i][component];
			vertex->normal[component] = normal[component];
			vertex->tangent[component] = tangent[component];
		}
		vertex->tangent[3] = tangent[3];
		vertex->texcoord[0] = uv[i][0];
		vertex->texcoord[1] = uv[i][1];
	}
	const uint32_t indices[6] = {base, base + 1u, base + 2u, base, base + 2u, base + 3u};
	for (uint32_t i = 0; i < 6; ++i)
		batch->indices[batch->index_count++] = indices[i];
	return true;
}

static bool append_horizontal(Builder *builder, DungeonRect rectangle, float y, bool upward)
{
	float p[4][3] = {{rectangle.min.x, y, rectangle.min.z},
					 {rectangle.max.x, y, rectangle.min.z},
					 {rectangle.max.x, y, rectangle.max.z},
					 {rectangle.min.x, y, rectangle.max.z}};
	float n[3] = {0.0f, upward ? 1.0f : -1.0f, 0.0f};
	float t[4] = {1.0f, 0.0f, 0.0f, upward ? -1.0f : 1.0f};
	float scale = 1.0f / builder->batch->material_width_m;
	float uv[4][2] = {{rectangle.min.x * scale, rectangle.min.z * scale},
					  {rectangle.max.x * scale, rectangle.min.z * scale},
					  {rectangle.max.x * scale, rectangle.max.z * scale},
					  {rectangle.min.x * scale, rectangle.max.z * scale}};
	return append_quad(builder, p, n, t, uv);
}

static bool append_box(Builder *builder, DungeonRect rectangle, float low, float high)
{
	float scale = 1.0f / builder->batch->material_width_m;
	if (!append_horizontal(builder, rectangle, high, true) ||
		!append_horizontal(builder, rectangle, low, false))
		return false;
	float face_positions[4][4][3] = {
		{{rectangle.min.x, low, rectangle.min.z}, {rectangle.max.x, low, rectangle.min.z},
		 {rectangle.max.x, high, rectangle.min.z}, {rectangle.min.x, high, rectangle.min.z}},
		{{rectangle.max.x, low, rectangle.max.z}, {rectangle.min.x, low, rectangle.max.z},
		 {rectangle.min.x, high, rectangle.max.z}, {rectangle.max.x, high, rectangle.max.z}},
		{{rectangle.min.x, low, rectangle.max.z}, {rectangle.min.x, low, rectangle.min.z},
		 {rectangle.min.x, high, rectangle.min.z}, {rectangle.min.x, high, rectangle.max.z}},
		{{rectangle.max.x, low, rectangle.min.z}, {rectangle.max.x, low, rectangle.max.z},
		 {rectangle.max.x, high, rectangle.max.z}, {rectangle.max.x, high, rectangle.min.z}},
	};
	const float normals[4][3] = {{0, 0, -1}, {0, 0, 1}, {-1, 0, 0}, {1, 0, 0}};
	const float tangents[4][4] = {{1, 0, 0, -1}, {-1, 0, 0, -1}, {0, 0, -1, -1},
										{0, 0, 1, -1}};
	float lengths[4] = {rectangle.max.x - rectangle.min.x, rectangle.max.x - rectangle.min.x,
					 rectangle.max.z - rectangle.min.z, rectangle.max.z - rectangle.min.z};
	for (uint32_t face = 0; face < 4; ++face)
	{
		float uv[4][2] = {{0, 0}, {lengths[face] * scale, 0},
						  {lengths[face] * scale, (high - low) * scale},
						  {0, (high - low) * scale}};
		if (!append_quad(builder, face_positions[face], normals[face], tangents[face], uv))
			return false;
	}
	return true;
}

static bool append_beveled_wall(Builder *builder, DungeonRect rectangle, float low, float high,
								float bevel)
{
	float half_width = (rectangle.max.x - rectangle.min.x) * 0.5f;
	float half_depth = (rectangle.max.z - rectangle.min.z) * 0.5f;
	bevel = fminf(bevel, fminf(half_width, half_depth) * 0.25f);
	DungeonPoint ring[8] = {
		{rectangle.min.x + bevel, rectangle.min.z},
		{rectangle.max.x - bevel, rectangle.min.z},
		{rectangle.max.x, rectangle.min.z + bevel},
		{rectangle.max.x, rectangle.max.z - bevel},
		{rectangle.max.x - bevel, rectangle.max.z},
		{rectangle.min.x + bevel, rectangle.max.z},
		{rectangle.min.x, rectangle.max.z - bevel},
		{rectangle.min.x, rectangle.min.z + bevel},
	};
	float scale = 1.0f / builder->batch->material_width_m;
	for (uint32_t edge = 0; edge < 8; ++edge)
	{
		DungeonPoint a = ring[edge], b = ring[(edge + 1u) % 8u];
		float dx = b.x - a.x, dz = b.z - a.z;
		float length = sqrtf(dx * dx + dz * dz);
		float p[4][3] = {{a.x, low, a.z}, {b.x, low, b.z}, {b.x, high, b.z},
						 {a.x, high, a.z}};
		float normal[3] = {dz / length, 0.0f, -dx / length};
		float tangent[4] = {dx / length, 0.0f, dz / length, -1.0f};
		float uv[4][2] = {{0, 0}, {length * scale, 0},
						   {length * scale, (high - low) * scale},
						   {0, (high - low) * scale}};
		if (!append_quad(builder, p, normal, tangent, uv))
			return false;
	}
	/* Eight independent top triangles keep the simple mesh builder generic and
	 * avoid a special polygon index path. The tiny extra vertices are paid once
	 * at level load and give every exposed wall edge a real highlight. */
	DungeonPoint center = {(rectangle.min.x + rectangle.max.x) * 0.5f,
						   (rectangle.min.z + rectangle.max.z) * 0.5f};
	for (uint32_t edge = 0; edge < 8; ++edge)
	{
		DungeonPoint a = ring[edge], b = ring[(edge + 1u) % 8u];
		float p[4][3] = {{center.x, high, center.z}, {a.x, high, a.z}, {b.x, high, b.z},
						 {b.x, high, b.z}};
		float normal[3] = {0, 1, 0};
		float tangent[4] = {1, 0, 0, -1};
		float uv[4][2] = {{center.x * scale, center.z * scale},
						   {a.x * scale, a.z * scale},
						   {b.x * scale, b.z * scale},
						   {b.x * scale, b.z * scale}};
		uint32_t old_vertex_count = builder->batch->vertex_count;
		uint32_t old_index_count = builder->batch->index_count;
		if (!append_quad(builder, p, normal, tangent, uv))
			return false;
		/* Degenerate the quad's second triangle, leaving one indexed triangle
		 * while retaining the builder's fixed-capacity bookkeeping. */
		builder->batch->vertex_count = old_vertex_count + 3u;
		builder->batch->index_count = old_index_count + 3u;
	}
	return true;
}

bool dungeon_mesh_build(const DungeonLevel *level, DungeonMeshData *out,
						DungeonLevelError *error)
{
	if (!level || !out)
		return mesh_fail(error, "level and output mesh are required");
	*out = (DungeonMeshData){0};
	Builder floor = {0}, wall = {0}, exit = {0}, player = {0};
	if (!builder_create(&floor, &out->batches[DUNGEON_MESH_FLOOR], level->surface_count, 2.4f) ||
		!builder_create(&wall, &out->batches[DUNGEON_MESH_WALL], level->solid_count * 16u, 1.8f) ||
		!builder_create(&exit, &out->batches[DUNGEON_MESH_EXIT], 6u, 2.0f) ||
		!builder_create(&player, &out->batches[DUNGEON_MESH_PLAYER], 6u, 1.0f))
	{
		dungeon_mesh_destroy(out);
		return mesh_fail(error, "out of memory building dungeon geometry");
	}
	for (uint32_t i = 0; i < level->surface_count; ++i)
		if (!append_horizontal(&floor, level->surfaces[i].footprint,
							   level->surfaces[i].elevation, true))
			goto capacity_error;
	for (uint32_t i = 0; i < level->solid_count; ++i)
		if (!append_beveled_wall(&wall, level->solids[i].footprint, level->solids[i].base_y,
								  level->solids[i].base_y + level->solids[i].height, 0.04f))
			goto capacity_error;
	DungeonRect exit_rect = {{level->exit.x - 0.65f, level->exit.z - 0.65f},
							 {level->exit.x + 0.65f, level->exit.z + 0.65f}};
	if (!append_box(&exit, exit_rect, level->floor_y + 0.01f, level->floor_y + 0.13f))
		goto capacity_error;
	if (!append_box(&player, (DungeonRect){{-0.35f, -0.35f}, {0.35f, 0.35f}}, 0.0f, 0.7f))
		goto capacity_error;
	return true;

capacity_error:
	dungeon_mesh_destroy(out);
	return mesh_fail(error, "internal dungeon mesh capacity error");
}

void dungeon_mesh_destroy(DungeonMeshData *data)
{
	if (!data)
		return;
	for (uint32_t i = 0; i < DUNGEON_MESH_BATCH_COUNT; ++i)
	{
		free(data->batches[i].vertices);
		free(data->batches[i].indices);
	}
	*data = (DungeonMeshData){0};
}
