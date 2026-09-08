#pragma once

#include "dungeon_level.h"
#include "dungeon_lighting.h"
#include "dungeon_mesh.h"
#include "dungeon_player.h"
#include "dungeon_session.h"
#include "gltf_scene.h"
#include "renderer.h"

#include <stdbool.h>

/* Static batches, one torch's primitives per light, and per doorway either a
 * row of pin bars with their target marks or a dial with its progress pips.
 * Doors and lock hardware are per-instance draws because they move. */
#define DUNGEON_MAX_LOCK_PIECES (3u * DUNGEON_LOCK_MAX_PINS + 2u + DUNGEON_LOCK_MAX_STEPS)
#define DUNGEON_MAX_DRAWS                                                                          \
	(DUNGEON_MESH_BATCH_COUNT + 4u * DUNGEON_MAX_LIGHTS + 1u +                                      \
	 DUNGEON_MAX_DOORS * (1u + DUNGEON_MAX_LOCK_PIECES))

typedef struct
{
	DungeonLevel level;
	DungeonMeshData geometry;
	Mesh meshes[DUNGEON_MESH_BATCH_COUNT];
	bool uploaded[DUNGEON_MESH_BATCH_COUNT];
	Texture moss_albedo;
	DungeonPlayer player;
	DungeonLight lights[DUNGEON_MAX_LIGHTS];
	uint32_t light_count;
	GltfScene torch;
	LocalToWorldTransform torch_transforms[DUNGEON_MAX_LIGHTS];
	uint32_t torch_count;
	/* The dungeon as a game: door state, the lock being picked, and the
	 * collider/occluder arrays that follow from which doors are still shut. */
	DungeonSession session;
} DungeonScene;

/* Picks a frontend from the environment: DUNGEON_MAP=<path> compiles an
 * ASCII map (the legacy/debug frontend, which has no doorways); otherwise
 * DUNGEON_SEED=<uint32> (default 1) generates a rooms-and-hallways level with
 * locked doors. See dungeon_grid.h / dungeon_cave.h. */
bool dungeon_scene_create(Renderer *renderer, DungeonScene *out, DungeonLevelError *error);

/* Fills `out` with up to `capacity` draws and returns how many were written.
 * `out_shadow_draw_count` (may be NULL) receives the length of the PREFIX of
 * `out` that should also be submitted to the shadow pass -- puddles are
 * appended last and excluded from it, since a flat coplanar disc casts no
 * meaningful shadow. */
uint32_t dungeon_scene_draws(DungeonScene *scene, WorldPosition camera_position, RendererDraw *out,
							 uint32_t capacity, uint32_t *out_shadow_draw_count);
/* Moves the player against the session's live colliders (so a shut door stops
 * them) and eases door swings. Returns true on the frame the exit is reached. */
bool dungeon_scene_update(DungeonScene *scene, float move_forward, float move_right,
						  float camera_yaw_degrees, float dt);
uint32_t dungeon_scene_write_lights(const DungeonScene *scene, WorldPosition camera_position,
								   vec4s *positions, vec4s *colors, uint32_t capacity);
uint32_t dungeon_scene_write_light_blockers(const DungeonScene *scene,
										WorldPosition camera_position, vec4s *blockers,
										uint32_t capacity);
void dungeon_scene_destroy(Renderer *renderer, DungeonScene *scene);
