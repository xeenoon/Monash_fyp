#pragma once

#include "coordinate.h"
#include "dungeon_geometry.h"
#include "dungeon_lock.h"
#include "dungeon_padlock.h"
#include "dungeon_session.h"
#include "gltf_scene.h"

/* Where the padlock model's parts go: binding the clips and nodes the game
 * needs by name, placing the lock on a door, and posing it for a given puzzle
 * state.
 *
 * Split out of dungeon_scene so it can be tested against the REAL exported
 * asset without a Vulkan device -- `gltf_scene_parse` is CPU-only, and every
 * claim worth making here is about metres and node transforms rather than
 * pixels. "Moving the pick shows you which pin you are on" is a question about
 * how far the pick travels, and it should be answerable from a test rather than
 * by somebody looking at the screen. */

/* How much bigger than life the lock is drawn.
 *
 * The model is true size -- a 72 mm industrial padlock, 120 mm over the
 * shackle -- because that is what the asset is and an asset that lies about its
 * scale is a trap for everything downstream. At true size on a 2.4 m door it is
 * a thumbnail: the generated mechanism it replaces was half a metre across, and
 * the focus camera, the pick stance and the lock's one reserved light are all
 * framed for something of that order. 3.2 puts the lock at 0.38 m over the
 * shackle, which reads at a metre and keeps the mechanism's pin stacks -- 4 mm
 * apart in life -- far enough apart to count. */
#define DUNGEON_PADLOCK_SCALE 4.5f

/* How far the lock is turned from its side toward its broad front face, in
 * degrees.
 *
 * These two want opposite things and the model does not let you have both. The
 * four pin stacks are spaced along the KEYWAY, front to back, the way a real
 * pin tumbler is built -- so they only line up across the screen when the lock
 * is viewed from its side. Mounted that way the padlock is edge-on: a tall thin
 * slab that does not read as a padlock at all. Mounted the other way it is
 * unmistakably a padlock with its mechanism hidden behind itself.
 *
 * The exporter's inspection window is cut for whatever angle this names, and
 * verifies by ray-cast that the pins are still in view from it -- so changing
 * this constant means re-exporting the asset, not just rebuilding.
 *
 * The gameplay view deliberately chooses 90: the broad lock face is square to
 * the camera, exactly as it is presented to the player. */
#define DUNGEON_PADLOCK_FACE_TURN_DEGREES 90.0f

/* How far an unset pin travels across its four slots, as a multiple of the set
 * travel the pop clip authored.
 *
 * Under 1.0 this was physically honest and unplayable: a real pin tumbler moves
 * its pins about 3 mm, which at the drawn scale put a whole key press inside
 * two pixels, and the puzzle cannot be solved from a readout nobody can see.
 * Over 1.0 the slots are spread wider than the lock really moves them, so a pin
 * near the top of its range stands about where a set one does -- which is the
 * cost, and it is a small one: a pin SETTING is announced by its own pop clip,
 * with the driver dropping and the spring compressing above it, not by height
 * alone.
 *
 * The alternative was to keep this under 1.0 and grow DUNGEON_PADLOCK_SCALE
 * until the millimetres survived, which reaches a lock taller than the doorway
 * long before it reaches a legible one. */
#define DUNGEON_PADLOCK_PIN_HEIGHT_SPAN 1.30f

typedef struct
{
	const GltfScene *model;
	uint32_t clip[DUNGEON_PADLOCK_CLIP_COUNT];
	float duration[DUNGEON_PADLOCK_CLIP_COUNT];
	uint32_t pin_node[DUNGEON_PADLOCK_PINS];
	uint32_t pick_node;
	/* Read off the asset rather than written down: `pin_travel` is the local
	 * translation a pin gains over its own pop clip, so an unset pin can be
	 * shown as a FRACTION of the travel the animator authored, and `pick_step`
	 * is one pin's worth of keyway depth in the pick's own parent frame.
	 * Re-timing or re-spacing the lock in Blender changes both without anything
	 * here being edited. */
	vec3s pin_travel[DUNGEON_PADLOCK_PINS];
	vec3s pick_step;
	/* Rest-pose bounds in model metres: how far the lock reaches BEHIND its
	 * origin along the door normal once turned (so its back can sit on the
	 * standoff plane instead of inside the leaf), and where its middle is (so
	 * the focus camera's aim point lands on the mechanism). */
	float reach, centre_height;
} DungeonPadlockModel;

/* Binds an imported scene. Fails, with a message, when the asset is missing
 * anything the game looks up by name -- silently drawing a lock whose pins
 * never move would be worse. */
bool dungeon_padlock_model_bind(DungeonPadlockModel *out, const GltfScene *model, char *error,
								size_t error_size);

/* The lock on a door face. `origin` is the lock origin, `across` runs along the
 * door (the camera's right) and `normal` points out of it toward the player --
 * the frame dungeon_scene's lock_frame builds. */
LocalToWorldTransform dungeon_padlock_model_placement(const DungeonPadlockModel *model,
													  DungeonPoint origin, DungeonPoint across,
													  DungeonPoint normal);

/* Rest pose, then every playing clip in turn, then the two things the clips
 * cannot know -- how far an unset pin has been raised, and which pin the pick
 * is on. `pose` and `world` are both the model's node count long.
 *
 * Pass `apply_state` false to get the clips alone, which is how a caller
 * measures what the ENGINE contributed rather than what the animator did. */
void dungeon_padlock_model_pose(const DungeonPadlockModel *model, const DungeonPadlockAnim *anim,
								const DungeonPinTumbler *pins, bool apply_state,
								GltfTransform *pose, mat4s *world);

/* Whether `node` carries the pick's own geometry, which hangs off the PICK
 * bone. The pick is only drawn while a clip is driving it: the model's rest
 * pose has it sitting in the keyway -- it is a lock and a pick modelled
 * together -- so a door nobody is working would otherwise wear a lockpick for
 * ever. */
bool dungeon_padlock_model_node_is_pick(const DungeonPadlockModel *model, uint32_t node);

/* The pick's offset along the keyway, in pins, as the engine applied it --
 * measured by posing twice, since the clips drive the pick along that same axis
 * and comparing against the rest pose would read the animator's insertion
 * travel as well. Returns false when there is no pick to measure. */
bool dungeon_padlock_model_pick_offset(const DungeonPadlockModel *model,
									   const DungeonPadlockAnim *anim,
									   const DungeonPinTumbler *pins, GltfTransform *pose,
									   mat4s *world, float *out_pins);
