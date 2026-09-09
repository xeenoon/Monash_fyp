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

/* The safe. This departs from the source game deliberately: Silent Labyrinth's
 * VaultDialPuzzle was a four-key direction sequence, and this is four pins on
 * the safe's face pressed in a hidden order. What carried over is the shape --
 * an ordered sequence of four things to find, with a wrong guess costing every
 * one found so far. */
static void the_safe_drives_a_pin_only_when_it_is_next_in_the_order(void)
{
	DungeonSafePins lock = {0};
	dungeon_safe_pins_init(&lock, 4321u);
	assert(lock.pin_count == DUNGEON_SAFE_PIN_COUNT);
	assert(!lock.solved && lock.progress == 0u && lock.selected == 0u);

	/* The order is a permutation: every pin appears exactly once, so a solved
	 * safe has all four standing proud and no press is ever wasted on a pin
	 * that is already out. */
	uint32_t seen[DUNGEON_SAFE_PIN_COUNT] = {0};
	for (uint32_t step = 0; step < lock.pin_count; ++step)
	{
		assert(lock.order[step] < DUNGEON_SAFE_PIN_COUNT);
		++seen[lock.order[step]];
	}
	for (uint32_t pin = 0; pin < DUNGEON_SAFE_PIN_COUNT; ++pin)
		assert(seen[pin] == 1u);

	/* Nothing is driven before anything is pressed. */
	for (uint32_t pin = 0; pin < DUNGEON_SAFE_PIN_COUNT; ++pin)
		assert(!dungeon_safe_pins_driven(&lock, pin));

	/* Moving is free: it drives nothing and, crucially, loses nothing. A whole
	 * lap of the row leaves progress exactly where it started, and the
	 * selection wraps rather than clamping. */
	for (uint32_t i = 0; i < DUNGEON_SAFE_PIN_COUNT; ++i)
		dungeon_safe_pins_move(&lock, 1);
	assert(lock.selected == 0u && lock.progress == 0u);
	dungeon_safe_pins_move(&lock, -1);
	assert(lock.selected == DUNGEON_SAFE_PIN_COUNT - 1u);
	dungeon_safe_pins_move(&lock, 1);
	assert(lock.selected == 0u);

	/* Exactly one pin in the row is the next one, whichever way you arrive at
	 * it -- there is no direction to feel for, only the order. */
	uint32_t next_count = 0;
	for (uint32_t i = 0; i < DUNGEON_SAFE_PIN_COUNT; ++i)
	{
		next_count += dungeon_safe_pins_selected_is_next(&lock) ? 1u : 0u;
		dungeon_safe_pins_move(&lock, 1);
	}
	assert(next_count == 1u);

	/* Pressing the wrong pin drives nothing. */
	uint32_t guard = DUNGEON_SAFE_PIN_COUNT + 1u;
	while (dungeon_safe_pins_selected_is_next(&lock) && guard--)
		dungeon_safe_pins_move(&lock, 1);
	assert(!dungeon_safe_pins_press(&lock));
	assert(lock.progress == 0u && !lock.solved);

	/* Pressing the right one drives it, and it stays driven. */
	guard = DUNGEON_SAFE_PIN_COUNT + 1u;
	while (!dungeon_safe_pins_selected_is_next(&lock) && guard--)
		dungeon_safe_pins_move(&lock, 1);
	assert(dungeon_safe_pins_selected_is_next(&lock));
	uint32_t first = lock.selected;
	assert(dungeon_safe_pins_press(&lock));
	assert(lock.progress == 1u);
	assert(dungeon_safe_pins_driven(&lock, first));
	/* And that pin is no longer the one wanted: pressing it again is now a
	 * wrong press like any other, which is exactly what makes a repeat in the
	 * order impossible to express. */
	assert(!dungeon_safe_pins_selected_is_next(&lock));

	/* A wrong press partway through costs every pin driven so far -- including
	 * one thrown away by pressing a pin that is already out. */
	guard = DUNGEON_SAFE_PIN_COUNT + 1u;
	while (!dungeon_safe_pins_selected_is_next(&lock) && guard--)
		dungeon_safe_pins_move(&lock, 1);
	assert(dungeon_safe_pins_press(&lock));
	assert(lock.progress == 2u);
	lock.selected = first; /* already driven, and no longer the one wanted */
	assert(!dungeon_safe_pins_press(&lock));
	assert(lock.progress == 0u);
	for (uint32_t pin = 0; pin < DUNGEON_SAFE_PIN_COUNT; ++pin)
		assert(!dungeon_safe_pins_driven(&lock, pin));
}

/* Every safe has to be openable, and a solved one has to stay solved. */
static void every_safe_can_actually_be_opened(void)
{
	for (uint32_t seed = 1u; seed <= 300u; ++seed)
	{
		DungeonSafePins lock = {0};
		dungeon_safe_pins_init(&lock, seed);
		uint32_t guard = DUNGEON_SAFE_PIN_COUNT * DUNGEON_SAFE_PIN_COUNT * 4u;
		while (!lock.solved && guard--)
		{
			if (dungeon_safe_pins_selected_is_next(&lock))
				dungeon_safe_pins_press(&lock);
			else
				dungeon_safe_pins_move(&lock, 1);
		}
		assert(lock.solved);
		assert(lock.progress == lock.pin_count);
		for (uint32_t pin = 0; pin < DUNGEON_SAFE_PIN_COUNT; ++pin)
			assert(dungeon_safe_pins_driven(&lock, pin));

		/* Nothing moves a solved safe, and a press on one cannot spring the
		 * face back out after the door has already opened. */
		uint32_t settled = lock.selected;
		dungeon_safe_pins_move(&lock, 1);
		assert(!dungeon_safe_pins_press(&lock));
		assert(lock.selected == settled);
		assert(lock.solved && lock.progress == lock.pin_count);
	}
}

int main(void)
{
	selection_clamps_and_height_wraps();
	pin_targets_are_seeded_never_trivial_and_all_or_nothing();
	a_set_pin_is_locked_in_place();
	the_safe_drives_a_pin_only_when_it_is_next_in_the_order();
	every_safe_can_actually_be_opened();
	puts("dungeon lock tests passed");
	return 0;
}
