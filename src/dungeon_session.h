#pragma once

#include "dungeon_level.h"
#include "dungeon_lock.h"

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
	DUNGEON_PHASE_VAULT_DIAL
} DungeonPhase;

typedef enum
{
	DUNGEON_STATUS_EXPLORING,
	DUNGEON_STATUS_NOTHING_IN_REACH,
	DUNGEON_STATUS_MATCH_THE_PINS,
	DUNGEON_STATUS_ENTER_THE_SEQUENCE,
	DUNGEON_STATUS_PINS_DO_NOT_ALIGN,
	DUNGEON_STATUS_SEQUENCE_PROGRESS,
	DUNGEON_STATUS_LOCK_OPEN
} DungeonStatus;

typedef struct
{
	bool open;
	float swing; /* 0 shut .. 1 fully swung; eased, so opening reads as motion */
	DungeonPinTumbler pins;
	DungeonVaultDial dial;
	DungeonDialDirection last_input;
	bool has_last_input;
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

/* Pin tumbler only: pick a pin, then raise or lower it. */
void dungeon_session_move_selection(DungeonSession *session, int direction);
void dungeon_session_adjust(DungeonSession *session, int direction);
/* Vault dial only: enter one direction of the combination. */
void dungeon_session_turn(DungeonSession *session, DungeonDialDirection direction);
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
