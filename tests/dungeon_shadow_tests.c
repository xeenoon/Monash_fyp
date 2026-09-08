#include "dungeon_grid.h"
#include "dungeon_mesh.h"
#include "dungeon_shadow.h"
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(DungeonShadowNode) == 80, "std430 node stride");
_Static_assert(offsetof(DungeonShadowNode, end) == 12,
			   "std430 escape offset");
_Static_assert(offsetof(DungeonShadowNode, maximum) == 16,
			   "std430 bounds offset");
_Static_assert(offsetof(DungeonShadowNode, a) == 32,
			   "std430 triangle offset");

int main(void)
{
	/* More than 64 independent walls; none can disappear due to a
	 * budget. */
	DungeonShadowTriangle triangles[100];
	for (int i = 0; i < 100; i++)
		triangles[i] = (DungeonShadowTriangle){
			{i * 3.f, 0, -1}, {i * 3.f, 2, -1}, {i * 3.f, 0, 1}};
	DungeonShadow s = {0};
	assert(dungeon_shadow_build(triangles, 100, &s));
	assert(s.count == 199 && s.nodes[0].end == s.count);
	for (int i = 0; i < 100; i++)
	{
		float o[3] = {i * 3.f - 1, .5f, 0}, d[3] = {1, 0, 0}, n[3],
			  distance = 2;
		assert(dungeon_shadow_trace(&s, o, d, &distance, n));
		assert(fabsf(distance - 1) < 1e-5f);
		/* A finite-height wall cannot block a ray above its top. */
		o[1] = 3;
		distance = 2;
		assert(!dungeon_shadow_trace(&s, o, d, &distance, n));
		o[1] = .5f;
		distance = .5f;
		assert(!dungeon_shadow_trace(&s, o, d, &distance, n));
		/* Two-sided intersections and parallel-axis slab handling. */
		o[0] = i * 3.f + 1;
		d[0] = -1;
		distance = 2;
		assert(dungeon_shadow_trace(&s, o, d, &distance, n));
	}
	dungeon_shadow_destroy(&s);
	/* Trace the actual bulged mesh: its visible surface must be the
	 * hit. */
	DungeonLevel level = {0};
	DungeonLevelError error = {0};
	DungeonMeshData mesh = {0};
	assert(dungeon_grid_compile_text("#####\n#S.E#\n#####\n", 2,
									 &level, &error));
	assert(dungeon_mesh_build(&level, &mesh, &error));
	DungeonMeshBatch *wall = &mesh.batches[DUNGEON_MESH_WALL];
	uint32_t count = wall->index_count / 3;
	DungeonShadowTriangle *copy = malloc(count * sizeof(*copy));
	assert(copy);
	for (uint32_t i = 0; i < count; i++)
	{
		memcpy(copy[i].a,
			   wall->vertices[wall->indices[3 * i]].position, 12);
		memcpy(copy[i].b,
			   wall->vertices[wall->indices[3 * i + 1]].position, 12);
		memcpy(copy[i].c,
			   wall->vertices[wall->indices[3 * i + 2]].position, 12);
	}
	assert(dungeon_shadow_build(copy, count, &s));
	unsigned checked = 0;
	for (uint32_t i = 0; i < count; i++)
	{
		float e1[3], e2[3], normal[3], o[3], d[3], hit_normal[3];
		for (int k = 0; k < 3; k++)
		{
			e1[k] = copy[i].b[k] - copy[i].a[k];
			e2[k] = copy[i].c[k] - copy[i].a[k];
		}
		normal[0] = e1[1] * e2[2] - e1[2] * e2[1];
		normal[1] = e1[2] * e2[0] - e1[0] * e2[2];
		normal[2] = e1[0] * e2[1] - e1[1] * e2[0];
		float len =
			sqrtf(normal[0] * normal[0] + normal[1] * normal[1] +
				  normal[2] * normal[2]);
		if (len < 1e-6f)
			continue;
		for (int k = 0; k < 3; k++)
		{
			normal[k] /= len;
			o[k] = (copy[i].a[k] + copy[i].b[k] + copy[i].c[k]) / 3 +
				   normal[k] * .01f;
			d[k] = -normal[k];
		}
		float distance = .02f;
		assert(dungeon_shadow_trace(&s, o, d, &distance, hit_normal));
		assert(fabsf(distance - .01f) < .0001f);
		checked++;
	}
	assert(checked > 64);
	free(copy);
	dungeon_shadow_destroy(&s);
	dungeon_mesh_destroy(&mesh);
	dungeon_level_destroy(&level);
	assert(dungeon_shadow_build(NULL, 0, &s) && s.count == 0);
	puts("dungeon shadow geometry tests passed");
}
