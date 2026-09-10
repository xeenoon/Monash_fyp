#pragma once

#include "dungeon_level.h"
#include "dungeon_lock.h"
#include "dungeon_prism.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* The dungeon as a game rather than as geometry: which doors are open, which
 * lock the player is currently picking, and the collider and light-occluder
 * arrays that follow from that. Ported from Silent Labyrinth's GameplaySession
 * (repos/fit3162_game, src/GameplaySession.cs) -- its GameplayPhase machine,
 * its NearestTarget interaction search, and its UnlockFirstDoor side effect.
 *
 * The level itself stays immutable. A door opening changes this session's
 * blocker arrays, never the compiled level, which is what lets a door open
 * without recompiling any contour, mesh, or triangulation. */

#define DUNGEON_MAX_DOORS 8u
#define DUNGEON_INTERACTION_RANGE_M 1.9f

/* The lock is mounted on the door face, centred, on the side the player
 * approaches from. STANDOFF clears the 0.18 m leaf; CENTRE_Y is the middle of
 * the casing-and-cutaway strip, which is what the focus camera aims at. */
#define DUNGEON_LOCK_STANDOFF_M 0.14f
#define DUNGEON_LOCK_CENTRE_Y_M 0.92f
/* Where the player stands to work it: off to one side, so their own body is
 * not what the close-up frames. */
#define DUNGEON_PICK_STANDOFF_M 0.95f
#define DUNGEON_PICK_LATERAL_M 0.58f

typedef enum
{
	DUNGEON_PHASE_EXPLORING,
	DUNGEON_PHASE_PIN_TUMBLER,
	DUNGEON_PHASE_SAFE_PINS,
	DUNGEON_PHASE_PRISM
} DungeonPhase;

typedef enum
{
	DUNGEON_STATUS_EXPLORING,
	DUNGEON_STATUS_NOTHING_IN_REACH,
	DUNGEON_STATUS_MATCH_THE_PINS,
	DUNGEON_STATUS_ENTER_THE_SEQUENCE,
	DUNGEON_STATUS_PINS_DO_NOT_ALIGN,
	DUNGEON_STATUS_SEQUENCE_PROGRESS,
	DUNGEON_STATUS_SAFE_RESET,
	DUNGEON_STATUS_LOCK_OPEN
} DungeonStatus;

typedef struct
{
	bool open;
	float swing; /* 0 shut .. 1 fully swung; eased, so opening reads as motion */
	DungeonPinTumbler pins;
	DungeonSafePins safe;
	DungeonPrismPuzzle prism;
	float prism_solved_time; /* hold the lit key briefly before the door opens */
	/* The safe's face, as an animation rather than as state. `safe_push[i]` is
	 * how far pin i has travelled toward the player -- 0 flush with the face,
	 * 1 fully driven -- eased here so the renderer gets the motion without
	 * owning any part of the puzzle. It is derived every frame from
	 * dungeon_safe_pins_driven and the selection, so a reset springs every pin
	 * back out with no extra bookkeeping.
	 *
	 * `safe_shake` is a decaying kick given both to a driven pin and to a
	 * reset, so a press always lands with a jolt whichever way it went. */
	float safe_push[DUNGEON_SAFE_PIN_COUNT];
	float safe_shake;
	/* Which side of the door plane the lock hardware stands on: the side the
	 * player starts from, so the first door met is never locked from behind. */
	float hardware_side;
} DungeonDoorState;

typedef struct
{
	const DungeonLevel *level;
	DungeonDoorState doors[DUNGEON_MAX_DOORS];
	uint32_t door_count;

	DungeonPhase phase;
	DungeonStatus status;
	uint32_t focused_door; /* only meaningful while phase != EXPLORING */

	/* Level colliders/occluders plus one entry per still-shut door. Rebuilt
	 * when a door opens, which is rare enough not to need anything cleverer. */
	DungeonCollider *colliders;
	uint32_t collider_count;
	DungeonSegment *occluders;
	uint32_t occluder_count;
} DungeonSession;

/* Borrows `level` for the session's lifetime; does not take ownership. */
bool dungeon_session_create(DungeonSession *session, const DungeonLevel *level);
void dungeon_session_destroy(DungeonSession *session);

/* Nearest still-shut door within DUNGEON_INTERACTION_RANGE_M, or UINT32_MAX.
 * The port of NearestTarget: closest wins, already-open doors are skipped. */
uint32_t dungeon_session_nearest_door(const DungeonSession *session, DungeonPoint player);

/* From EXPLORING, opens the lock the player is standing at and switches to its
 * phase. Returns whether a lock was entered. */
bool dungeon_session_interact(DungeonSession *session, DungeonPoint player);
void dungeon_session_cancel(DungeonSession *session);
/* Direction keys browse, or rotate on left/right after Enter. */
void dungeon_session_prism_direction(DungeonSession *session, DungeonPrismDirection direction);
void dungeon_session_prism_confirm(DungeonSession *session);

/* Pin tumbler only: pick a pin, then raise or lower it. */
void dungeon_session_move_selection(DungeonSession *session, int direction);
void dungeon_session_adjust(DungeonSession *session, int direction);
/* Safe only: walk the selection along the row of face pins. Edge-triggered like
 * the pin tumbler's keys -- four pins is far too short a row to hold a key
 * down. Wraps; moving banks nothing and loses nothing. */
void dungeon_session_move_safe_pin(DungeonSession *session, int direction);

/* Safe only: the confirm key. Drives the selected pin if it is the one the
 * order wants next, otherwise springs the whole face back out and starts the
 * order again. Returns whether a pin was driven. */
bool dungeon_session_press_safe_pin(DungeonSession *session);

/* How far the pin under the selection creeps forward, as a fraction of a fully
 * driven pin's travel. It is the only marker of which pin the keys are on, and
 * it is kept well short of the whole travel so that a hovered pin can never be
 * mistaken for a driven one. */
#define DUNGEON_SAFE_PIN_HOVER 0.22f
/* Pin tumbler only: try the lock. Returns whether it opened. */
bool dungeon_session_confirm(DungeonSession *session);

/* Eases door swings. Call once per frame; independent of puzzle input. */
void dungeon_session_update(DungeonSession *session, float dt);

/* Window-title status line, mirroring the source's GameplayWindowTitle -- this
 * renderer has no text drawing, so the title bar is where status goes. */
void dungeon_session_status_text(const DungeonSession *session, char *out, size_t capacity);

/* The lock hardware's world position for a given door. The camera frames this
 * rather than the door centre -- centring the door puts a 2.4 m leaf across the
 * view and leaves the hardware a few pixels tall at the bottom. */
DungeonPoint dungeon_session_lock_origin(const DungeonSession *session, uint32_t door_index);

/* dungeon_session_lock_origin for the door currently being picked. */
DungeonPoint dungeon_session_focus_point(const DungeonSession *session);

/* Where the player should stand while picking the focused lock: beside the
 * hardware rather than over it. Without this the player cube sits squarely
 * between the close-up camera and the puzzle it is framing. */
DungeonPoint dungeon_session_pick_stance(const DungeonSession *session);

/* The yaw, in degrees, a camera must look along to face the focused lock from
 * the side its hardware stands on -- i.e. the side the player approached from.
 * Without this the lock is framed from whatever angle exploring left behind,
 * which for half the doors means reading the puzzle through the door. */
float dungeon_session_focus_facing_degrees(const DungeonSession *session);
