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
#define DUNGEON_LOCK_MAX_STEPS 6u

/* Ticks around the safe dial's rim. One full turn is this many steps, which is
 * also how many dots are engraved on the face. */
#define DUNGEON_DIAL_POSITIONS 24u

typedef struct
{
	uint8_t target[DUNGEON_LOCK_MAX_PINS];
	uint8_t heights[DUNGEON_LOCK_MAX_PINS];
	uint32_t pin_count;
	uint32_t selected;
	bool solved;
} DungeonPinTumbler;

/* A safe dial. Turn it a tick at a time and watch the red index pin: it flashes
 * whenever the tick under it is the next number of the combination. Press
 * confirm while it is flashing to bank that number and move on to the next.
 * Press confirm while it is not, and the whole lock resets to the first number.
 *
 * Direction does not matter -- the flash is the signal, so there is nothing to
 * feel for and no reason to constrain which way the dial turns. */
typedef struct
{
	uint8_t combination[DUNGEON_LOCK_MAX_STEPS];
	uint32_t step_count;
	uint32_t progress;
	uint32_t position;
	bool solved;
} DungeonVaultDial;

/* Every pin starts at 0, so a target of all zeroes would be solved on sight;
 * generation re-rolls until at least one pin is off zero. */
void dungeon_pin_tumbler_init(DungeonPinTumbler *lock, uint32_t seed, uint32_t pin_count);

/* Clamped, matching the source's Math.Clamp -- moving off either end is a
 * no-op, it does not wrap. */
void dungeon_pin_tumbler_move(DungeonPinTumbler *lock, int direction);

/* True once a pin is sitting on its target. A set pin has dropped into place
 * and will not move again -- see dungeon_pin_tumbler_adjust. */
bool dungeon_pin_tumbler_pin_set(const DungeonPinTumbler *lock, uint32_t pin);

/* Wraps mod DUNGEON_LOCK_PIN_STATES, matching the source's (v + d + 4) % 4.
 * Height wraps even though selection clamps; that asymmetry is the source's.
 *
 * A pin that has reached its target is LOCKED and ignores this: it has dropped
 * into place and cannot be knocked back out. That is what keeps the puzzle a
 * short, monotonic search -- work one pin until it sets, move to the next --
 * rather than a state you can undo by leaning on a key. It also means a pin
 * whose target is its resting position is set before the player touches
 * anything, which reads correctly: it is already seated in its bore. */
void dungeon_pin_tumbler_adjust(DungeonPinTumbler *lock, int direction);

/* All-or-nothing against the target. Sets and returns `solved`. */
bool dungeon_pin_tumbler_submit(DungeonPinTumbler *lock);

void dungeon_vault_dial_init(DungeonVaultDial *lock, uint32_t seed, uint32_t step_count);

/* Whether the tick currently under the index pin is the next number of the
 * combination -- that is, whether the pin should be flashing. */
bool dungeon_vault_dial_on_number(const DungeonVaultDial *lock);

/* Turns one tick, either way. Returns dungeon_vault_dial_on_number for the tick
 * it arrived at, so a caller can react to the pin starting to flash. Turning
 * alone never banks anything and never loses anything. */
bool dungeon_vault_dial_step(DungeonVaultDial *lock, int direction);

/* The confirm key. On a flashing pin this banks the number and advances, and
 * opens the lock on the last one. Anywhere else it resets the combination to
 * the beginning -- which is the whole risk in the puzzle, since turning is
 * free. Returns whether a number was banked. */
bool dungeon_vault_dial_commit(DungeonVaultDial *lock);
