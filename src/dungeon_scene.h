#pragma once

#include "dungeon_lab.h"
#include "dungeon_level.h"
#include "dungeon_lighting.h"
#include "dungeon_mesh.h"
#include "dungeon_padlock.h"
#include "dungeon_padlock_pose.h"
#include "dungeon_shadow.h"
#include "dungeon_player.h"
#include "dungeon_session.h"
#include "game_character.h"
#include "gltf_scene.h"
#include "renderer.h"

#include <stdbool.h>

/* Static batches, torch primitives and moving door hardware. Reserve enough
 * per-door draws for the tumbler/safe hardware or the optical board, crystals,
 * glyphs and the branching tracer's bounded beam segment array. */
#define DUNGEON_LOCK_SPRING_COILS 7u
/* One draw per node group in the padlock model -- the exporter joins every part
 * that shares a bone, so this is the rig's bone count with headroom, not the
 * source's 275 objects. Checked against the asset at load. */
#define DUNGEON_PADLOCK_MAX_PRIMITIVES 32u
#define DUNGEON_MAX_LOCK_PIECES                                                                    \
	(8u + 2u * DUNGEON_PRISM_MAX + DUNGEON_PRISM_MAX_SEGMENTS +                                    \
	 DUNGEON_LOCK_MAX_PINS * (3u + DUNGEON_LOCK_SPRING_COILS) + DUNGEON_SAFE_PIN_COUNT +           \
	 DUNGEON_PADLOCK_MAX_PRIMITIVES)
/* Torch fixtures cost their mesh primitives plus one additive flame billboard
 * each, and the carried torch is one more of both. */
/* Game props (treasure chest, hidden gem, the guardian) are scaled, tinted
 * instances of the player's cube -- placeholder art until real models land. */
#define DUNGEON_MAX_PROPS 24u
#define DUNGEON_MAX_DRAWS                                                                          \
	(DUNGEON_MESH_BATCH_COUNT + 5u * DUNGEON_MAX_LIGHTS + 2u +                                      \
	 DUNGEON_MAX_DOORS * (1u + DUNGEON_MAX_LOCK_PIECES) + DUNGEON_MAX_PROPS)

typedef struct
{
	WorldPosition position; /* centre of the box's base */
	float size[3];			/* metres along local X, Y, Z */
	float yaw, pitch;		/* radians: about +Y, then about local +X */
	float color[3];
	float metallic;
} DungeonProp;

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
	/* The padlock: one imported, animated model, posed once per door that is
	 * wearing it. Absent when the runtime asset has not been exported (see
	 * tools/export_padlock_runtime.py), in which case pin-tumbler doors fall
	 * back to the generated cutaway mechanism -- the game stays playable from
	 * a fresh checkout without a 20 MB Blender file having been baked. */
	GltfScene padlock;
	bool padlock_loaded;
	/* The clips, nodes and measurements the game looks up by name, bound once
	 * at load -- see dungeon_padlock_pose.h. */
	DungeonPadlockModel padlock_model;
	/* Scratch for one pose, sized to the model. Owned here rather than built
	 * per draw: posing runs once per door per frame. */
	GltfTransform *padlock_pose;
	mat4s *padlock_world;
	DungeonPadlockAnim padlock_anim[DUNGEON_MAX_DOORS];
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
	/* Filled by the game layer (dungeon_game.c) each frame; drawn after the
	 * level and included in the shadow pass. */
	DungeonProp props[DUNGEON_MAX_PROPS];
	/* The player character, when the game supplies one (NULL draws the
	 * placeholder cube and the separate carried torch, as the harness and the
	 * torch lab expect). Borrowed, not owned. */
	const GameCharacter *character;
	float player_facing; /* radians in world XZ, eased toward travel */
	float player_stride; /* 0 standing .. 1 full walking pace */
	uint32_t prop_count;
	/* Multiplies the wall and floor albedo: how a biome re-themes the same
	 * stone. Zero (an unset scene) reads as white. */
	float surface_tint[3];
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
/* As above, but the procedural frontend uses `seed` instead of DUNGEON_SEED.
 * DUNGEON_MAP / DUNGEON_LAB still take precedence. */
bool dungeon_scene_create_seeded(Renderer *renderer, DungeonScene *out, uint32_t seed,
								 DungeonLevelError *error);

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

/* Whether this door wears the imported padlock rather than the generated
 * cutaway mechanism. The two are read by different assertions: the generated
 * pick is POSED from the puzzle and dungeon_lock_layout is the authority on
 * whether it cuts through the lock, while the padlock's pick is AUTHORED and
 * the only part of its motion this engine chooses is which pin it is on. */
bool dungeon_scene_door_wears_padlock(const DungeonScene *scene, uint32_t door);

/* Whether the padlock's pick has been driven to the pin the keys are on.
 * The clips move the pick; the engine only offsets it along the keyway by one
 * pin's spacing per selection step, and this is that offset read back and
 * resolved to a pin index. An inverted axis, a wrong scale or a step taken in
 * the wrong frame all land here -- the same class of bug that put the pins on
 * screen in the wrong order once already. */
bool dungeon_scene_padlock_pick_on_selection(DungeonScene *scene, uint32_t door);

/* How far pin `pin` has travelled from its rest position, in world metres,
 * with the lock posed as it would be drawn this frame. Zero for a pin that has
 * not moved. This is the end of the whole chain -- clip found by name, layered
 * over the others, sampled, chained down the hierarchy and composed with the
 * door's placement -- reported as a number a script can assert on instead of a
 * picture somebody has to look at. */
bool dungeon_scene_padlock_pin_travel(DungeonScene *scene, uint32_t door, uint32_t pin,
									  float *out_metres);

/* Where the pick's node sits in the world, with the lock posed as it would be
 * drawn this frame -- so a script can assert that changing the selection
 * actually moved it, rather than somebody checking by eye. */
bool dungeon_scene_padlock_pick_world(DungeonScene *scene, uint32_t door, WorldPosition *out);

/* Centre and half-extents of the carried torch's flame billboard, in world
 * metres. The carried torch is moved into the framing while a lock is being
 * picked, and "moved into the framing" is a claim about where it lands
 * relative to the near plane and to the pins -- both of which this lets a
 * script assert instead of somebody looking at the picture. */
void dungeon_scene_player_flame(const DungeonScene *scene, WorldPosition *out_centre,
								float *out_half_width, float *out_half_height);

uint32_t dungeon_scene_write_lights(const DungeonScene *scene, WorldPosition camera_position,
								   vec4s *positions, vec4s *colors, uint32_t capacity);
void dungeon_scene_destroy(Renderer *renderer, DungeonScene *scene);
bool dungeon_scene_prepare_shadows(DungeonScene *scene, Renderer *renderer);
