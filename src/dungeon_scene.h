#pragma once

#include "dungeon_lab.h"
#include "dungeon_level.h"
#include "dungeon_lighting.h"
#include "dungeon_mesh.h"
#include "dungeon_shadow.h"
#include "dungeon_player.h"
#include "dungeon_session.h"
#include "gltf_scene.h"
#include "renderer.h"

#include <stdbool.h>

/* Static batches, torch primitives and moving door hardware. Reserve enough
 * per-door draws for the tumbler/safe hardware or the optical board, crystals,
 * glyphs and the branching tracer's bounded beam segment array. */
#define DUNGEON_LOCK_SPRING_COILS 7u
#define DUNGEON_MAX_LOCK_PIECES                                                                    \
	(8u + 2u * DUNGEON_PRISM_MAX + DUNGEON_PRISM_MAX_SEGMENTS +                                    \
	 DUNGEON_LOCK_MAX_PINS * (3u + DUNGEON_LOCK_SPRING_COILS) + DUNGEON_SAFE_PIN_COUNT)
/* Torch fixtures cost their mesh primitives plus one additive flame billboard
 * each, and the carried torch is one more of both. */
#define DUNGEON_MAX_DRAWS                                                                          \
	(DUNGEON_MESH_BATCH_COUNT + 5u * DUNGEON_MAX_LIGHTS + 2u +                                      \
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
	/* Seconds of dungeon time, advanced by dungeon_scene_update. Drives the
	 * torch flicker and the flame shader's animation, so both stop together
	 * with the rest of the world when the scene is not being stepped. */
	float time;
	/* Torch-lab mode: one room, one fixture, free-fly camera. DUNGEON_LAB_NONE
	 * outside the lab; otherwise which of the three demo scenes is running,
	 * which decides what burns and what is drawn. See dungeon_lab.h. */
	DungeonLabScene lab;
	/* The standing torch's pole, scene 2 only: presentation geometry the lab
	 * generates rather than a level batch, like flame_quad below. */
	Mesh lab_pole;
	bool lab_pole_uploaded;
	/* The rag binding at the standing torch's head. Opaque and depth-writing,
	 * so the flame is occluded where it passes behind the cloth. */
	Mesh lab_wrap;
	bool lab_wrap_uploaded;
	/* The flame billboard's own unit quad. It used to borrow the prism lock's
	 * quad, which silently cost every torch its flame in any level without a
	 * prism door -- the torch lab, and any seed that happens to roll none.
	 * Presentation geometry belongs to the scene, not to a puzzle's batch. */
	Mesh flame_quad;
	bool flame_quad_uploaded;
	/* The dungeon as a game: door state, the lock being picked, and the
	 * collider/occluder arrays that follow from which doors are still shut. */
	DungeonSession session;
	/* Eased position of the pick's tip in the focused lock's face plane
	 * (lateral along the door, height above the floor). Presentation only --
	 * the session owns which pin is selected; this is just where the pick has
	 * got to on its way there. */
	float pick_lateral, pick_height;
	bool pick_active;
	DungeonShadow wall_shadow, shadow;
	uint32_t shadow_door_mask;
	bool shadow_uploaded;
} DungeonScene;

/* Picks a frontend from the environment: DUNGEON_LAB=1 builds the one-room
 * torch lab (see dungeon_lab.h); DUNGEON_MAP=<path> compiles an ASCII map (the
 * legacy/debug frontend, which has no doorways); otherwise DUNGEON_SEED=<uint32>
 * (default 1) generates a rooms-and-hallways level with locked doors. See
 * dungeon_lab.h / dungeon_grid.h / dungeon_cave.h. */
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
/* World position of a pin's head, for tests that need to check where a piece
 * of the lock actually ends up on screen rather than trusting the layout. */
bool dungeon_scene_pin_world(const DungeonScene *scene, uint32_t door_index, uint32_t pin,
							 WorldPosition *out);

uint32_t dungeon_scene_write_lights(const DungeonScene *scene, WorldPosition camera_position,
								   vec4s *positions, vec4s *colors, uint32_t capacity);
void dungeon_scene_destroy(Renderer *renderer, DungeonScene *scene);
bool dungeon_scene_prepare_shadows(DungeonScene *scene, Renderer *renderer);
