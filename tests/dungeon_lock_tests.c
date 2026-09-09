/* Fidelity tests for the two lock minigames ported from Silent Labyrinth
 * (repos/fit3162_game, src/GameplaySession.cs). Most of these pin behaviour
 * that is easy to "improve" by accident and would change the difficulty of
 * every lock in the game. */
#include "dungeon_lock.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

/* The source clamps selection (Math.Clamp) but wraps height ((v + d + 4) % 4).
 * That asymmetry is deliberate and both halves are load-bearing: wrapping
 * selection would let a player scroll past the last pin, clamping height would
 * make the lowest and highest pin settings harder to reach than the middle. */
static void selection_clamps_and_height_wraps(void)
{
	DungeonPinTumbler lock = {0};
	dungeon_pin_tumbler_init(&lock, 1234u, 3u);
	assert(lock.pin_count == 3u);
	assert(lock.selected == 0u);

	dungeon_pin_tumbler_move(&lock, -1);
	assert(lock.selected == 0u); /* clamped, not wrapped to the last pin */
	for (int i = 0; i < 10; ++i)
		dungeon_pin_tumbler_move(&lock, 1);
	assert(lock.selected == 2u);

	dungeon_pin_tumbler_move(&lock, -2);
	assert(lock.selected == 0u);
	/* Force pin 0 off its target first: a pin sitting on its target is locked
	 * in place, so the wrap below would otherwise be testing nothing. */
	lock.target[0] = 2u;
	assert(!dungeon_pin_tumbler_pin_set(&lock, 0u));
	dungeon_pin_tumbler_adjust(&lock, -1);
	assert(lock.heights[0] == DUNGEON_LOCK_PIN_STATES - 1u); /* wrapped down from 0 */
	dungeon_pin_tumbler_adjust(&lock, 1);
	assert(lock.heights[0] == 0u);
	/* Both wrap directions are covered above. A full cycle is no longer a
	 * meaningful thing to assert: every cycle passes through the pin's target,
	 * where it sets and locks -- which a_set_pin_is_locked_in_place covers. */

	/* Adjusting one pin must not disturb any other. */
	dungeon_pin_tumbler_move(&lock, 1);
	lock.target[1] = 3u;
	dungeon_pin_tumbler_adjust(&lock, 2);
	assert(lock.heights[0] == 0u && lock.heights[2] == 0u);
}

/* A pin that reaches its target drops into place and stops responding. This is
 * the whole shape of the puzzle: raise a pin until it sets, then move on, with
 * no way to undo progress by leaning on a key. */
static void a_set_pin_is_locked_in_place(void)
{
	DungeonPinTumbler lock = {0};
	dungeon_pin_tumbler_init(&lock, 99u, 4u);
	lock.target[0] = 2u;
	lock.heights[0] = 0u;
	lock.selected = 0u;
	assert(!dungeon_pin_tumbler_pin_set(&lock, 0u));

	dungeon_pin_tumbler_adjust(&lock, 1);
	assert(lock.heights[0] == 1u);
	dungeon_pin_tumbler_adjust(&lock, 1);
	assert(lock.heights[0] == 2u);
	assert(dungeon_pin_tumbler_pin_set(&lock, 0u));

	/* Locked: neither direction moves it, and it does not wrap away either. */
	for (int i = 0; i < 8; ++i)
	{
		dungeon_pin_tumbler_adjust(&lock, 1);
		dungeon_pin_tumbler_adjust(&lock, -1);
		assert(lock.heights[0] == 2u);
	}

	/* The pick can still be moved over a locked pin -- only its height is
	 * frozen, not the selection. */
	dungeon_pin_tumbler_move(&lock, 1);
	assert(lock.selected == 1u);

	/* Generation never hands out a pin that is already on its target, across
	 * every pin count and a wide sweep of seeds -- so no bore is ever locked
	 * before the player has touched it. */
	for (uint32_t seed = 1u; seed <= 400u; ++seed)
		for (uint32_t count = 2u; count <= DUNGEON_LOCK_MAX_PINS; ++count)
		{
			DungeonPinTumbler fresh = {0};
			dungeon_pin_tumbler_init(&fresh, seed, count);
			for (uint32_t pin = 0; pin < fresh.pin_count; ++pin)
			{
				assert(fresh.heights[pin] == 0u);
				assert(!dungeon_pin_tumbler_pin_set(&fresh, pin));
			}
		}

	/* And with every pin set the lock opens -- locking cannot strand the
	 * player in a state that no longer submits. */
	DungeonPinTumbler solved = {0};
	dungeon_pin_tumbler_init(&solved, 4242u, 4u);
	for (uint32_t pin = 0; pin < solved.pin_count; ++pin)
	{
		solved.selected = pin;
		uint32_t guard = DUNGEON_LOCK_PIN_STATES + 1u;
		while (!dungeon_pin_tumbler_pin_set(&solved, pin) && guard--)
			dungeon_pin_tumbler_adjust(&solved, 1);
		assert(dungeon_pin_tumbler_pin_set(&solved, pin));
	}
	assert(dungeon_pin_tumbler_submit(&solved));
}

static void pin_targets_are_seeded_never_trivial_and_all_or_nothing(void)
{
	for (uint32_t seed = 1u; seed <= 200u; ++seed)
	{
		DungeonPinTumbler lock = {0};
		dungeon_pin_tumbler_init(&lock, seed, 3u);
		/* EVERY pin has to be moved. Pins start at 0 and a set pin is locked in
		 * place, so a zero target would be a bore the player can neither move
		 * nor needs to. */
		for (uint32_t pin = 0; pin < lock.pin_count; ++pin)
		{
			assert(lock.target[pin] >= 1u);
			assert(lock.target[pin] < DUNGEON_LOCK_PIN_STATES);
			assert(!dungeon_pin_tumbler_pin_set(&lock, pin));
		}
		assert(!dungeon_pin_tumbler_submit(&lock));
	}

	DungeonPinTumbler a = {0}, b = {0};
	dungeon_pin_tumbler_init(&a, 77u, 3u);
	dungeon_pin_tumbler_init(&b, 77u, 3u);
	assert(memcmp(a.target, b.target, sizeof(a.target)) == 0); /* deterministic per seed */

	/* Every pin but one correct still fails: submission is all-or-nothing. */
	for (uint32_t pin = 0; pin < a.pin_count; ++pin)
		a.heights[pin] = a.target[pin];
	a.heights[a.pin_count - 1u] = (uint8_t)((a.target[a.pin_count - 1u] + 1u) %
											DUNGEON_LOCK_PIN_STATES);
	assert(!dungeon_pin_tumbler_submit(&a));
	assert(!a.solved);
	a.heights[a.pin_count - 1u] = a.target[a.pin_count - 1u];
	assert(dungeon_pin_tumbler_submit(&a));
	assert(a.solved);
}

/* The safe dial. This departs from the source game deliberately: Silent
 * Labyrinth's VaultDialPuzzle was a four-key direction sequence, and this is a
 * combination dial you turn while watching a red index pin. What carried over
 * is the shape -- an ordered sequence of four things to find. */
static void the_dial_banks_a_number_only_when_the_pin_is_lit(void)
{
	DungeonVaultDial lock = {0};
	dungeon_vault_dial_init(&lock, 4321u, 4u);
	assert(lock.step_count == 4u);
	assert(!lock.solved && lock.progress == 0u && lock.position == 0u);

	/* Numbers are spread around the rim and never sit on the starting tick, or
	 * the pin would be lit before the player turned anything. */
	assert(!dungeon_vault_dial_on_number(&lock));
	for (uint32_t step = 0; step < lock.step_count; ++step)
	{
		assert(lock.combination[step] != 0u);
		assert(lock.combination[step] < DUNGEON_DIAL_POSITIONS);
	}

	/* Turning is free: it banks nothing and, crucially, loses nothing. A whole
	 * revolution leaves progress exactly where it started. */
	for (uint32_t i = 0; i < DUNGEON_DIAL_POSITIONS; ++i)
		dungeon_vault_dial_step(&lock, 1);
	assert(lock.progress == 0u);
	assert(lock.position == 0u);

	/* Direction does not matter -- the pin lights on the number either way. */
	uint32_t lit_going_right = 0, lit_going_left = 0;
	for (uint32_t i = 0; i < DUNGEON_DIAL_POSITIONS; ++i)
		lit_going_right += dungeon_vault_dial_step(&lock, 1) ? 1u : 0u;
	for (uint32_t i = 0; i < DUNGEON_DIAL_POSITIONS; ++i)
		lit_going_left += dungeon_vault_dial_step(&lock, -1) ? 1u : 0u;
	assert(lit_going_right == 1u && lit_going_left == 1u);

	/* Confirm on a dark pin throws the combination away. */
	while (dungeon_vault_dial_on_number(&lock))
		dungeon_vault_dial_step(&lock, 1);
	assert(!dungeon_vault_dial_commit(&lock));
	assert(lock.progress == 0u && !lock.solved);

	/* Confirm on a lit pin banks it. */
	uint32_t guard = DUNGEON_DIAL_POSITIONS + 1u;
	while (!dungeon_vault_dial_on_number(&lock) && guard--)
		dungeon_vault_dial_step(&lock, 1);
	assert(dungeon_vault_dial_on_number(&lock));
	assert(dungeon_vault_dial_commit(&lock));
	assert(lock.progress == 1u);
	/* And the pin goes dark again straight away: it is now pointing at a tick
	 * that is no longer the number wanted. */
	assert(!dungeon_vault_dial_on_number(&lock));

	/* A wrong confirm partway through costs every number banked so far. */
	guard = DUNGEON_DIAL_POSITIONS + 1u;
	while (!dungeon_vault_dial_on_number(&lock) && guard--)
		dungeon_vault_dial_step(&lock, 1);
	assert(dungeon_vault_dial_commit(&lock));
	assert(lock.progress == 2u);
	while (dungeon_vault_dial_on_number(&lock))
		dungeon_vault_dial_step(&lock, 1);
	assert(!dungeon_vault_dial_commit(&lock));
	assert(lock.progress == 0u);
}

/* Every dial has to be openable, and a solved one has to stay solved. */
static void every_dial_can_actually_be_opened(void)
{
	for (uint32_t seed = 1u; seed <= 300u; ++seed)
		for (uint32_t steps = 2u; steps <= DUNGEON_LOCK_MAX_STEPS; ++steps)
		{
			DungeonVaultDial lock = {0};
			dungeon_vault_dial_init(&lock, seed, steps);
			uint32_t guard = DUNGEON_DIAL_POSITIONS * DUNGEON_LOCK_MAX_STEPS * 4u;
			while (!lock.solved && guard--)
			{
				if (dungeon_vault_dial_on_number(&lock))
					dungeon_vault_dial_commit(&lock);
				else
					dungeon_vault_dial_step(&lock, 1);
			}
			assert(lock.solved);
			assert(lock.progress == lock.step_count);

			/* Nothing moves a solved dial, and confirm on one cannot reset it
			 * back to zero after the door has already opened. */
			uint32_t settled = lock.position;
			assert(!dungeon_vault_dial_step(&lock, 1));
			assert(!dungeon_vault_dial_commit(&lock));
			assert(lock.position == settled);
			assert(lock.solved && lock.progress == lock.step_count);
		}
}

int main(void)
{
	selection_clamps_and_height_wraps();
	pin_targets_are_seeded_never_trivial_and_all_or_nothing();
	a_set_pin_is_locked_in_place();
	the_dial_banks_a_number_only_when_the_pin_is_lit();
	every_dial_can_actually_be_opened();
	puts("dungeon lock tests passed");
	return 0;
}
