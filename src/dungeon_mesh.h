#pragma once

#include "dungeon_level.h"
#include "mesh.h"

#include <stdbool.h>

typedef enum
{
	DUNGEON_MESH_FLOOR,
	DUNGEON_MESH_WALL, /* also carries the rock plateau cap -- same material */
	DUNGEON_MESH_EXIT,
	DUNGEON_MESH_PLAYER,
	DUNGEON_MESH_PUDDLE, /* untextured; radial fade packed into texcoord.x */
	DUNGEON_MESH_MOSS, /* static bent fronds, real leaf silhouettes */
	/* The four below are UNIT meshes in local coordinates, built once and
	 * drawn once per instance with their own transform -- the same pattern the
	 * torches use. They are the only dungeon geometry that moves, so they can
	 * never be baked into a level-space batch. */
	DUNGEON_MESH_DOOR,		 /* one leaf, anchored at one end of its blocker */
	/* The lock is mounted on the door face and read head-on, so all of it is
	 * built standing in the XY plane rather than lying on the floor. */
	DUNGEON_MESH_LOCK_BODY,	   /* the lock casing itself, above the mechanism */
	DUNGEON_MESH_LOCK_HOUSING, /* cutaway panel below the casing */
	DUNGEON_MESH_LOCK_CHANNEL, /* one pin's bore through the housing */
	DUNGEON_MESH_LOCK_SPRING,  /* one coil of the spring above a pin */
	DUNGEON_MESH_LOCK_PIN,	   /* one pin; it RISES in its bore */
	DUNGEON_MESH_LOCK_SAFE_PIN, /* one pin on the safe's face; it DRIVES forward */
	DUNGEON_MESH_LOCK_PICK,	   /* the pick, swung to whichever pin is selected */
	DUNGEON_MESH_LOCK_FACE,	   /* the safe's face plate */
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
