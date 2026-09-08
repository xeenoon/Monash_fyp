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

typedef enum
{
	DUNGEON_DIAL_LEFT,
	DUNGEON_DIAL_RIGHT,
	DUNGEON_DIAL_UP,
	DUNGEON_DIAL_DOWN
} DungeonDialDirection;

typedef struct
{
	uint8_t target[DUNGEON_LOCK_MAX_PINS];
	uint8_t heights[DUNGEON_LOCK_MAX_PINS];
	uint32_t pin_count;
	uint32_t selected;
	bool solved;
} DungeonPinTumbler;

typedef struct
{
	DungeonDialDirection combination[DUNGEON_LOCK_MAX_STEPS];
	uint32_t step_count;
	uint32_t progress;
	bool solved;
} DungeonVaultDial;

/* Every pin starts at 0, so a target of all zeroes would be solved on sight;
 * generation re-rolls until at least one pin is off zero. */
void dungeon_pin_tumbler_init(DungeonPinTumbler *lock, uint32_t seed, uint32_t pin_count);

/* Clamped, matching the source's Math.Clamp -- moving off either end is a
 * no-op, it does not wrap. */
void dungeon_pin_tumbler_move(DungeonPinTumbler *lock, int direction);

/* Wraps mod DUNGEON_LOCK_PIN_STATES, matching the source's (v + d + 4) % 4.
 * Height wraps even though selection clamps; that asymmetry is the source's. */
void dungeon_pin_tumbler_adjust(DungeonPinTumbler *lock, int direction);

/* All-or-nothing against the target. Sets and returns `solved`. */
bool dungeon_pin_tumbler_submit(DungeonPinTumbler *lock);

void dungeon_vault_dial_init(DungeonVaultDial *lock, uint32_t seed, uint32_t step_count);

/* Advances on a match, and resets progress to ZERO on any miss -- including a
 * miss that happens to equal step 0, which the source does not credit either.
 * Returns whether the lock is now solved. */
bool dungeon_vault_dial_turn(DungeonVaultDial *lock, DungeonDialDirection direction);
