#include "dungeon_grid.h"
#include "dungeon_mesh.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static float length3(const float value[3])
{
	return sqrtf(value[0] * value[0] + value[1] * value[1] +
				 value[2] * value[2]);
}

int main(void)
{
	DungeonLevel level = {0};
	DungeonLevelError error = {0};
	assert(dungeon_grid_compile_text("#####\n#S.E#\n#####\n", 2.0f,
									 &level, &error));
	DungeonMeshData mesh = {0};
	assert(dungeon_mesh_build(&level, &mesh, &error));
	assert(mesh.batches[DUNGEON_MESH_FLOOR].vertex_count > 0u);
	assert(mesh.batches[DUNGEON_MESH_FLOOR].index_count > 0u);
	assert(mesh.batches[DUNGEON_MESH_WALL].vertex_count > 0u);
	assert(mesh.batches[DUNGEON_MESH_WALL].index_count > 0u);
	/* Wall end rings share the exact contour with the floor/cap.
	 * Interior rings may wobble, but displacing the bottom ring opens
	 * a visible seam. */
	uint32_t wall_vertex = 0u;
	for (uint32_t loop_index = 0;
		 loop_index < level.contours.loop_count; ++loop_index)
	{
		const DungeonContourLoop *loop =
			&level.contours.loops[loop_index];
		if (loop->point_count < 3u)
			continue;
		for (uint32_t i = 0; i < loop->point_count; ++i)
		{
			const Vertex *bottom = &mesh.batches[DUNGEON_MESH_WALL]
										.vertices[wall_vertex + i];
			const Vertex *top =
				&mesh.batches[DUNGEON_MESH_WALL]
					 .vertices[wall_vertex + 4u * loop->point_count +
							   i];
			assert(fabsf(bottom->position[0] - loop->points[i].x) <
				   1e-6f);
			assert(fabsf(bottom->position[1] - level.floor_y) <
				   1e-6f);
			assert(fabsf(bottom->position[2] - loop->points[i].z) <
				   1e-6f);
			assert(fabsf(top->position[0] - loop->points[i].x) <
				   1e-6f);
			assert(fabsf(top->position[1] -
						 (level.floor_y + level.wall_height)) <
				   1e-6f);
			assert(fabsf(top->position[2] - loop->points[i].z) <
				   1e-6f);
		}
		wall_vertex += 5u * loop->point_count;
	}
	assert(mesh.batches[DUNGEON_MESH_EXIT].vertex_count == 24u);
	assert(mesh.batches[DUNGEON_MESH_PLAYER].vertex_count == 24u);
	/* The ASCII frontend places no puddles; the batch stays
	 * legitimately empty rather than allocating a phantom quad. */
	assert(mesh.batches[DUNGEON_MESH_PUDDLE].vertex_count == 0u);
	assert(mesh.batches[DUNGEON_MESH_PUDDLE].vertices == NULL);
	for (uint32_t batch = 0; batch < DUNGEON_MESH_BATCH_COUNT;
		 ++batch)
		for (uint32_t i = 0; i < mesh.batches[batch].vertex_count;
			 ++i)
		{
			assert(fabsf(length3(
							 mesh.batches[batch].vertices[i].normal) -
						 1.0f) < 1e-4f);
			assert(
				fabsf(
					length3(mesh.batches[batch].vertices[i].tangent) -
					1.0f) < 1e-4f);
		}
	/* Foliage has actual volume, valid topology, and never reaches
	 * the cap. Rebuilding must give identical geometry independent of
	 * render state. */
	const DungeonMeshBatch *moss = &mesh.batches[DUNGEON_MESH_MOSS];
	assert(moss->vertex_count > 0u);
	bool elevated = false;
	for (uint32_t i = 0; i < moss->vertex_count; ++i)
	{
		float y = moss->vertices[i].position[1];
		assert(isfinite(y) && y >= level.floor_y - 0.05f);
		assert(y < level.floor_y + level.wall_height - 0.5f);
		if (y > level.floor_y + 0.05f)
			elevated = true;
	}
	assert(elevated);
	for (uint32_t i = 0; i < moss->index_count; ++i)
		assert(moss->indices[i] < moss->vertex_count);
	DungeonMeshData repeat = {0};
	assert(dungeon_mesh_build(&level, &repeat, &error));
	assert(repeat.batches[DUNGEON_MESH_MOSS].vertex_count ==
		   moss->vertex_count);
	assert(memcmp(repeat.batches[DUNGEON_MESH_MOSS].vertices,
				  moss->vertices,
				  moss->vertex_count * sizeof(Vertex)) == 0);
	dungeon_mesh_destroy(&repeat);

	dungeon_mesh_destroy(&mesh);
	dungeon_level_destroy(&level);
	puts("dungeon mesh tests passed");
	return 0;
}
