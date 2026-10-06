#pragma once

#include <stdbool.h>
#include <stdint.h>

/* What the padlock model is DOING, as a state machine with no renderer and no
 * glTF dependency -- so tests/dungeon_padlock_tests.c runs it headlessly, the
 * same way dungeon_lock runs the puzzle itself.
 *
 * The lock's presentation is authored animation, not a simulation: eight clips
 * exported from assets/dungeons/padlock/padlock_animated.blend. This decides
 * which of them are playing and how far into each. It never decides anything
 * about the puzzle -- dungeon_lock owns pin heights and whether the lock opens,
 * and this only watches.
 *
 * Clips LAYER. The exporter strips every channel that never leaves the node's
 * rest pose, so Pin_02_Pop touches the second pin stack and nothing else, and
 * it can run over Pick_Jiggle_Loop without snapping the pick back. That is the
 * whole reason a pin popping while the pick works is two clips rather than a
 * clip authored per combination. */

#define DUNGEON_PADLOCK_PINS 4u

typedef enum
{
	DUNGEON_PADLOCK_CLIP_INSERT,
	DUNGEON_PADLOCK_CLIP_WITHDRAW,
	DUNGEON_PADLOCK_CLIP_JIGGLE,
	DUNGEON_PADLOCK_CLIP_PIN_01_POP,
	DUNGEON_PADLOCK_CLIP_PIN_02_POP,
	DUNGEON_PADLOCK_CLIP_PIN_03_POP,
	DUNGEON_PADLOCK_CLIP_PIN_04_POP,
	DUNGEON_PADLOCK_CLIP_UNLOCK,
	DUNGEON_PADLOCK_CLIP_COUNT
} DungeonPadlockClip;

/* The clip names in the exported glTF, indexed by DungeonPadlockClip. The
 * engine looks clips up by name because a name survives a re-export and an
 * index does not. */
extern const char *const DUNGEON_PADLOCK_CLIP_NAMES[DUNGEON_PADLOCK_CLIP_COUNT];

typedef struct
{
	bool playing;
	bool loop;
	/* Seconds into the clip. A one-shot that has finished parks here at its
	 * duration and stays playing: sampling clamps, so the clip goes on holding
	 * its last pose -- which is how a set pin stays set without a second
	 * "finished" pose to author or store. */
	float time;
} DungeonPadlockLayer;

typedef struct
{
	DungeonPadlockLayer layers[DUNGEON_PADLOCK_CLIP_COUNT];
	/* Whether the pick is in the keyway. Distinct from "the player is picking
	 * this lock": the pick keeps working through the insert and the withdraw,
	 * which are the clips either side of that. */
	bool engaged;
	bool popped[DUNGEON_PADLOCK_PINS];
	bool unlocking;
	/* Where each pin is DRAWN, in puzzle slots, eased toward where the puzzle
	 * has actually put it. The puzzle moves a pin a whole slot on a key press;
	 * easing is what makes that read as a pin rising rather than a pin
	 * teleporting, and it is presentation only -- dungeon_lock still owns the
	 * integer the lock is solved against. */
	float shown[DUNGEON_PADLOCK_PINS];
} DungeonPadlockAnim;

/* Nothing playing, no pin popped: a lock nobody has touched. */
void dungeon_padlock_reset(DungeonPadlockAnim *anim);

/* The player started or stopped working this lock. Starting runs the insert
 * and then loops the jiggle; stopping runs the withdraw and drops the jiggle.
 * Idempotent, so the scene can call it every frame from the session's phase
 * rather than having to detect the edge itself. */
void dungeon_padlock_engage(DungeonPadlockAnim *anim, bool engaged);

/* Pin `pin` has just set. Plays its pop once and then holds it set. Calling it
 * again for a pin already popped does nothing -- a set pin cannot be knocked
 * back out (see dungeon_pin_tumbler_adjust), so neither can its animation. */
void dungeon_padlock_pop_pin(DungeonPadlockAnim *anim, uint32_t pin);

/* The lock opened. The unlock clip starts from all four pins set and turns the
 * core, retracts the balls and lifts the shackle; it holds open at the end.
 * The pick stops working -- there is nothing left to pick. */
void dungeon_padlock_unlock(DungeonPadlockAnim *anim);

/* Eases the drawn pin positions toward `heights` (one slot per pin, from
 * dungeon_lock). Call once a frame alongside dungeon_padlock_update. A pin that
 * has reached its target never moves again, so this comes to rest on its own
 * rather than needing to be told the pin has set. */
void dungeon_padlock_show_heights(DungeonPadlockAnim *anim, const uint8_t *heights,
								  uint32_t count, float dt);

/* Advances every playing layer. `durations` is one clip length in seconds per
 * DungeonPadlockClip, read from the asset -- the state machine never hardcodes
 * how long a clip is, because the clips are authored and can be re-timed
 * without touching this file. */
void dungeon_padlock_update(DungeonPadlockAnim *anim, const float *durations, float dt);

/* Whether the pick should be DRAWN at all.
 *
 * The model's rest pose has the pick sitting in the keyway -- it is a lock and
 * a pick modelled together -- so a door nobody is working would otherwise show
 * a lockpick permanently stuck in it. The pick belongs to the player: it
 * appears when the insert starts, stays for the jiggle and the opening, and is
 * gone once the withdraw has run its course. */
bool dungeon_padlock_pick_visible(const DungeonPadlockAnim *anim);

/* Whether pin `pin` is fully set as far as the ANIMATION is concerned: its pop
 * has played out. The renderer holds an unset pin at a fraction of the authored
 * travel and leaves a popped one to the clip. */
bool dungeon_padlock_pin_popped(const DungeonPadlockAnim *anim, uint32_t pin);
