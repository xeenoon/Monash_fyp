#pragma once

#include "dungeon_level.h"
#include "mesh.h"

#include <stdbool.h>

typedef enum
{
	DUNGEON_MESH_FLOOR,
	DUNGEON_MESH_WALL,
	DUNGEON_MESH_EXIT,
	DUNGEON_MESH_BATCH_COUNT
} DungeonMeshBatchKind;

typedef struct
{
	Vertex *vertices;
	uint32_t vertex_count;
	uint32_t *indices;
	uint32_t index_count;
	float material_width_m;
} DungeonMeshBatch;

typedef struct
{
	DungeonMeshBatch batches[DUNGEON_MESH_BATCH_COUNT];
} DungeonMeshData;

/* CPU-only material-batched geometry. The scene owns GPU upload because it
 * also owns material descriptors and the renderer lifetime. */
bool dungeon_mesh_build(const DungeonLevel *level, DungeonMeshData *out,
						DungeonLevelError *error);
void dungeon_mesh_destroy(DungeonMeshData *data);

