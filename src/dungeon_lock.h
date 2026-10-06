#pragma once

#include <stdbool.h>
#include <stdint.h>

/* The two lock minigames, ported from Silent Labyrinth's PinTumblerPuzzle and
 * VaultDialPuzzle (repos/fit3162_game, src/GameplaySession.cs). Pure state
 * machines with no renderer, level, or input dependency, so they are testable
 * headlessly -- the presentation layer in dungeon_scene reads these fields and
 * builds geometry from them, it never owns puzzle state itself.
 *
 * The one deliberate departure from the source is seeding: the C# targets are
 * hardcoded ([1,2,1] and LEFT/RIGHT/UP/DOWN), here they follow DUNGEON_SEED
 * per door. */

#define DUNGEON_LOCK_MAX_PINS 5u
#define DUNGEON_LOCK_PIN_STATES 4u

/* Pins standing on the face of the safe. Four, always -- the order is a
 * permutation of them, so the count is both how many pins there are and how
 * many presses a solved safe took. */
#define DUNGEON_SAFE_PIN_COUNT 4u

typedef struct
{
	uint8_t target[DUNGEON_LOCK_MAX_PINS];
	uint8_t heights[DUNGEON_LOCK_MAX_PINS];
	uint32_t pin_count;
	uint32_t selected;
	bool solved;
} DungeonPinTumbler;

/* The safe. Four pins stand in a row on the front of its face, and there is a
 * hidden order to press them in. Move the selection along the row -- that is
 * free, it banks nothing and loses nothing -- and press confirm on a pin. If it
 * is the one the order wants next it is DRIVEN: it slides forward out of the
 * face toward the player and stays there. Press any other pin, including one
 * already driven, and the whole face resets: every pin springs back flush and
 * the order starts again from the first.
 *
 * `order` is a permutation, so every pin is driven exactly once and a solved
 * safe is one with all four standing proud. A free sequence with repeats could
 * not be read off the face at all -- pressing an already-driven pin would have
 * nothing left to show.
 *
 * There is no tell for which pin is next. A pin driving forward is the ONLY
 * feedback the lock gives, which is what makes the search a search; the earlier
 * version of this puzzle lit and flashed an index pin to say "press now", which
 * made it a reaction test rather than something to work out. */
typedef struct
{
	uint8_t order[DUNGEON_SAFE_PIN_COUNT];
	uint32_t pin_count;
	uint32_t progress;
	uint32_t selected;
	bool solved;
} DungeonSafePins;

/* Every pin starts at height 0 and every target is at least 1, so every pin has
 * to be moved before it can lock. */
void dungeon_pin_tumbler_init(DungeonPinTumbler *lock, uint32_t seed, uint32_t pin_count);

/* Clamped, matching the source's Math.Clamp -- moving off either end is a
 * no-op, it does not wrap. */
void dungeon_pin_tumbler_move(DungeonPinTumbler *lock, int direction);

/* True once a pin is sitting on its target. A set pin has dropped into place
 * and will not move again -- see dungeon_pin_tumbler_adjust. */
bool dungeon_pin_tumbler_pin_set(const DungeonPinTumbler *lock, uint32_t pin);

/* A pin that has reached its target is LOCKED and ignores input: it has
 * dropped into place and cannot be knocked back out. That keeps the puzzle a
 * short, monotonic search -- work one pin until it sets, move to the next --
 * rather than a state you can undo by leaning on a key. Generation guarantees
 * no pin starts on its target, so every bore is one the player has to work. */
/* Raises the selected pin by one level when direction is positive. Downward
 * input is ignored; a pin freezes permanently as soon as it reaches target. */
void dungeon_pin_tumbler_adjust(DungeonPinTumbler *lock, int direction);

/* All-or-nothing against the target. Sets and returns `solved`. */
bool dungeon_pin_tumbler_submit(DungeonPinTumbler *lock);

/* Seeds the press order. Nothing is driven and the selection starts on pin 0. */
void dungeon_safe_pins_init(DungeonSafePins *lock, uint32_t seed);

/* Whether this pin has already been driven forward -- that is, whether it
 * appears in the part of the order banked so far. This is what the renderer
 * animates and what the player reads the puzzle's state off. */
bool dungeon_safe_pins_driven(const DungeonSafePins *lock, uint32_t pin);

/* Whether the selected pin is the one the order wants next. Deliberately NOT
 * shown to the player -- the session uses it, and the tests and the harness
 * drive the lock with it, but no light on the face reports it. */
bool dungeon_safe_pins_selected_is_next(const DungeonSafePins *lock);

/* Moves the selection along the row. Wraps: four pins is short enough that
 * clamping at the ends would only ever be an annoyance. Free -- moving banks
 * nothing and loses nothing, so the press is the whole risk in the puzzle. */
void dungeon_safe_pins_move(DungeonSafePins *lock, int direction);

/* The confirm key. Drives the selected pin and advances if it was the one the
 * order wanted next, opening the safe on the last one; on any other pin it
 * springs the whole face back out and starts the order again. Returns whether a
 * pin was driven. */
bool dungeon_safe_pins_press(DungeonSafePins *lock);
