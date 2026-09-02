#include "dungeon_grid.h"
#include "dungeon_mesh.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static float length3(const float value[3])
{
	return sqrtf(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
}

int main(void)
{
	DungeonLevel level = {0};
	DungeonLevelError error = {0};
	assert(dungeon_grid_compile_text("#####\n#S.E#\n#####\n", 2.0f, &level, &error));
	DungeonMeshData mesh = {0};
	assert(dungeon_mesh_build(&level, &mesh, &error));
	assert(mesh.batches[DUNGEON_MESH_FLOOR].vertex_count == 4);
	assert(mesh.batches[DUNGEON_MESH_FLOOR].index_count == 6);
	assert(mesh.batches[DUNGEON_MESH_WALL].vertex_count == level.solid_count * 24u);
	assert(mesh.batches[DUNGEON_MESH_EXIT].vertex_count == 24);
	for (uint32_t batch = 0; batch < DUNGEON_MESH_BATCH_COUNT; ++batch)
		for (uint32_t i = 0; i < mesh.batches[batch].vertex_count; ++i)
		{
			assert(fabsf(length3(mesh.batches[batch].vertices[i].normal) - 1.0f) < 1e-6f);
			assert(fabsf(length3(mesh.batches[batch].vertices[i].tangent) - 1.0f) < 1e-6f);
		}
	dungeon_mesh_destroy(&mesh);
	dungeon_level_destroy(&level);
	puts("dungeon mesh tests passed");
	return 0;
}
