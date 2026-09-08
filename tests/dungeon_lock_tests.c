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
	dungeon_pin_tumbler_adjust(&lock, -1);
	assert(lock.heights[0] == DUNGEON_LOCK_PIN_STATES - 1u); /* wrapped down from 0 */
	dungeon_pin_tumbler_adjust(&lock, 1);
	assert(lock.heights[0] == 0u);
	for (uint32_t i = 0; i < DUNGEON_LOCK_PIN_STATES; ++i)
		dungeon_pin_tumbler_adjust(&lock, 1);
	assert(lock.heights[0] == 0u); /* a full cycle returns to where it started */

	/* Adjusting one pin must not disturb any other. */
	dungeon_pin_tumbler_move(&lock, 1);
	dungeon_pin_tumbler_adjust(&lock, 2);
	assert(lock.heights[0] == 0u && lock.heights[2] == 0u);
}

static void pin_targets_are_seeded_never_trivial_and_all_or_nothing(void)
{
	for (uint32_t seed = 1u; seed <= 200u; ++seed)
	{
		DungeonPinTumbler lock = {0};
		dungeon_pin_tumbler_init(&lock, seed, 3u);
		/* Every pin starts at 0, so an all-zero target would be solved before
		 * the player touched anything. */
		bool any_nonzero = false;
		for (uint32_t pin = 0; pin < lock.pin_count; ++pin)
		{
			assert(lock.target[pin] < DUNGEON_LOCK_PIN_STATES);
			any_nonzero = any_nonzero || lock.target[pin] != 0u;
		}
		assert(any_nonzero);
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

/* The source resets progress to zero outright on a miss rather than re-testing
 * the wrong input against step 0. Re-testing would be a strictly easier lock,
 * so the reset is behaviour worth pinning rather than an oversight to fix. */
static void a_wrong_turn_resets_progress_to_zero(void)
{
	DungeonVaultDial lock = {0};
	dungeon_vault_dial_init(&lock, 4321u, 4u);
	assert(lock.step_count == 4u);
	assert(!lock.solved && lock.progress == 0u);

	assert(!dungeon_vault_dial_turn(&lock, lock.combination[0]));
	assert(lock.progress == 1u);

	DungeonDialDirection wrong =
		(DungeonDialDirection)((lock.combination[1] + 1u) % 4u);
	assert(!dungeon_vault_dial_turn(&lock, wrong));
	assert(lock.progress == 0u);

	/* Even when the wrong input happens to equal step 0, it earns no credit:
	 * the reset lands on zero, it does not restart at one. */
	DungeonVaultDial replay = {0};
	dungeon_vault_dial_init(&replay, 4321u, 4u);
	dungeon_vault_dial_turn(&replay, replay.combination[0]);
	if (replay.combination[1] != replay.combination[0])
	{
		dungeon_vault_dial_turn(&replay, replay.combination[0]);
		assert(replay.progress == 0u);
	}

	/* The full combination, entered cleanly, opens it -- and further turns on
	 * a solved lock are harmless. */
	DungeonVaultDial solved = {0};
	dungeon_vault_dial_init(&solved, 4321u, 4u);
	for (uint32_t step = 0; step < solved.step_count; ++step)
	{
		bool done = dungeon_vault_dial_turn(&solved, solved.combination[step]);
		assert(done == (step + 1u == solved.step_count));
	}
	assert(solved.solved && solved.progress == solved.step_count);
	assert(dungeon_vault_dial_turn(&solved, DUNGEON_DIAL_UP));
	assert(solved.solved && solved.progress == solved.step_count);
}

int main(void)
{
	selection_clamps_and_height_wraps();
	pin_targets_are_seeded_never_trivial_and_all_or_nothing();
	a_wrong_turn_resets_progress_to_zero();
	puts("dungeon lock tests passed");
	return 0;
}
