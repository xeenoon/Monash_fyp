#pragma once

#include "dungeon_lock.h"

#include <stdbool.h>
#include <stdint.h>

/* The lock's geometry in its own face-local frame, shared by the renderer that
 * draws it and by the tests that check it does not intersect itself:
 *
 *   x  lateral along the door, 0 at the centre of the lock, +x screen-right
 *   y  height above the dungeon floor
 *   z  proud of the door face, +z toward the player
 *
 * Every piece is placed by the CENTRE of its thin axis, matching how
 * dungeon_scene positions instances, so a piece at proud p with half-thickness
 * t occupies [p - t, p + t].
 *
 * This lives in the headless library on purpose. The pick sweeps across the
 * whole mechanism as the player changes pins, and "does the pick pass through
 * the lock" is a question about boxes, not about pixels -- it should be
 * answerable without a Vulkan device. */

/* Heights above the floor. */
#define DUNGEON_LOCK_BODY_Y 1.00f
#define DUNGEON_LOCK_BODY_TOP_Y 1.36f
#define DUNGEON_LOCK_HOUSING_Y 0.52f
#define DUNGEON_LOCK_HOUSING_TOP_Y 0.94f
#define DUNGEON_LOCK_CHANNEL_Y 0.56f
#define DUNGEON_LOCK_CHANNEL_TOP_Y 0.90f
#define DUNGEON_LOCK_PIN_BASE_Y 0.575f
#define DUNGEON_LOCK_PIN_RISE_M 0.068f
#define DUNGEON_LOCK_PIN_LENGTH_M 0.11f

/* Lateral extents and spacing. */
#define DUNGEON_LOCK_BORE_PITCH_M 0.135f
#define DUNGEON_LOCK_BODY_HALF_X 0.115f
#define DUNGEON_LOCK_HOUSING_HALF_X 0.262f
#define DUNGEON_LOCK_CHANNEL_HALF_X 0.034f
#define DUNGEON_LOCK_PIN_RADIUS_M 0.024f
#define DUNGEON_LOCK_SPRING_RADIUS_M 0.026f

/* The safe's face: a round plate standing on the door with a stepped bezel,
 * and four pins in a row across it. A pin's local origin is its BACK, buried in
 * the plate, so `proud + length` is where its head sits -- flush-and-a-bit when
 * the pin is at rest, a whole DRIVE further out when it has been driven.
 *
 * The row has to fit inside the bezel: the outermost pin centre is 1.5 pitches
 * from the middle, and that plus a pin radius is what must stay under
 * DUNGEON_LOCK_FACE_BEZEL_RADIUS_M. */
#define DUNGEON_LOCK_FACE_Y 0.72f
#define DUNGEON_LOCK_FACE_RADIUS_M 0.175f
#define DUNGEON_LOCK_FACE_BEZEL_RADIUS_M 0.150f
#define DUNGEON_LOCK_FACE_PLATE_M 0.052f
#define DUNGEON_LOCK_FACE_BEZEL_M 0.066f
#define DUNGEON_SAFE_PIN_RADIUS_M 0.026f
#define DUNGEON_SAFE_PIN_LENGTH_M 0.075f
#define DUNGEON_SAFE_PIN_PITCH_M 0.076f
/* Off the plate's back face, not zero: a pin whose end cap were coplanar with
 * the plate's would z-fight it wherever the two are seen edge-on. */
#define DUNGEON_SAFE_PIN_BASE_PROUD_M 0.006f
/* How far a driven pin travels toward the player. Large next to the 9 mm of pin
 * showing at rest, because that contrast IS the puzzle's readout. */
#define DUNGEON_SAFE_PIN_DRIVE_M 0.055f
/* The row sits this far ABOVE the plate's centre. The focus camera looks down
 * on the lock at roughly 24 degrees, and anything standing out of the face
 * projects that much lower on screen than the plate behind it: laid out on the
 * centreline the row hung off the bottom of the plate, and a driven pin cleared
 * its edge entirely. Tuned against the head of a pin at rest, so the driven
 * ones drop back toward the middle as they come out -- which reads as travel
 * toward the player rather than as a row that has slipped. */
#define DUNGEON_SAFE_PIN_LIFT_M 0.030f

/* Proud offsets: how far out of the door face each layer sits. */
#define DUNGEON_LOCK_PROUD_BODY_M 0.070f
#define DUNGEON_LOCK_PROUD_HOUSING_M 0.032f
#define DUNGEON_LOCK_PROUD_BACK_M 0.030f
#define DUNGEON_LOCK_PROUD_PIN_M 0.062f
/* The pick rides in front of the whole mechanism. That is what makes "the pick
 * never passes through the lock" true by construction rather than by careful
 * posing, and it is the invariant dungeon_lock_layout_pick_clear checks. */
#define DUNGEON_LOCK_PROUD_PICK_M 0.104f

/* How far a pin sinks back into its bore once it is at its target -- the only
 * signal that it has set, now that nothing is highlighted. */
#define DUNGEON_LOCK_PIN_SET_SINK_M 0.026f

#define DUNGEON_LOCK_SPRING_COIL_COUNT 7u
#define DUNGEON_LOCK_MAX_PARTS 64u
#define DUNGEON_LOCK_PICK_PARTS 3u

/* The pick's tip stub reaches this far up from the shaft, so the tip lands on
 * the underside of the pin it is working. */
#define DUNGEON_LOCK_PICK_STUB_M 0.050f

typedef struct
{
	float min[3], max[3];
} DungeonLockBox;

bool dungeon_lock_box_overlap(const DungeonLockBox *a, const DungeonLockBox *b);

/* Every solid part of the mechanism except the pick, for the given pin state.
 * Returns how many boxes were written. */
uint32_t dungeon_lock_layout_parts(const DungeonPinTumbler *pins, DungeonLockBox *out,
								   uint32_t capacity);

/* The pick's own boxes with its tip at (lateral, height). Writes
 * DUNGEON_LOCK_PICK_PARTS entries. */
uint32_t dungeon_lock_layout_pick(float lateral, float height, DungeonLockBox *out,
								  uint32_t capacity);

/* Where the pick's tip belongs when working `pin`: centred on its bore, with
 * the stub reaching up to the underside of the pin. */
void dungeon_lock_layout_pick_target(const DungeonPinTumbler *pins, uint32_t pin,
									 float *out_lateral, float *out_height);

/* Lateral centre of a pin's bore. */
float dungeon_lock_layout_bore_lateral(const DungeonPinTumbler *pins, uint32_t pin);

/* True when a pick at (lateral, height) clears every part of the mechanism. */
bool dungeon_lock_layout_pick_clear(const DungeonPinTumbler *pins, float lateral, float height);
