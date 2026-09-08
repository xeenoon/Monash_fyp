/* The pick sweeps across the whole mechanism as the player changes pins, and
 * whether it passes through the lock is a question about boxes, not pixels --
 * so it is answered here, exhaustively and without a Vulkan device, rather than
 * by looking at a capture and hoping.
 *
 * The first version of the pick failed both halves of this: its tip stub hung
 * below the shaft instead of rising to the pin, and it was posed by rolling the
 * whole thing about the door normal, which swung the shaft diagonally through
 * the housing on its way between bores. */
#include "dungeon_lock_layout.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* Walks every reachable pin state and asserts the pick clears the mechanism
 * while working each pin in turn. */
static void the_pick_clears_the_lock_in_every_pin_state(void)
{
	uint32_t checked = 0;
	for (uint32_t pin_count = 3u; pin_count <= DUNGEON_LOCK_MAX_PINS; ++pin_count)
	{
		/* Every combination of pin heights, not just the ones a solution
		 * passes through: the player can leave the pins in any arrangement. */
		uint32_t states = 1u;
		for (uint32_t i = 0; i < pin_count; ++i)
			states *= DUNGEON_LOCK_PIN_STATES;
		for (uint32_t state = 0; state < states; ++state)
		{
			DungeonPinTumbler pins = {.pin_count = pin_count};
			uint32_t remaining = state;
			for (uint32_t i = 0; i < pin_count; ++i)
			{
				pins.heights[i] = (uint8_t)(remaining % DUNGEON_LOCK_PIN_STATES);
				remaining /= DUNGEON_LOCK_PIN_STATES;
				/* Seated pins sink back into their bores, which moves them --
				 * so the clearance has to hold for both, not just one. */
				pins.target[i] = (uint8_t)((i + state) % DUNGEON_LOCK_PIN_STATES);
			}
			for (uint32_t pin = 0; pin < pin_count; ++pin)
			{
				pins.selected = pin;
				float lateral = 0.0f, height = 0.0f;
				dungeon_lock_layout_pick_target(&pins, pin, &lateral, &height);
				assert(dungeon_lock_layout_pick_clear(&pins, lateral, height));
				++checked;
			}
		}
	}
	assert(checked > 1000u);
}

/* The pick eases between pins rather than teleporting, so the states above are
 * only the endpoints. Every intermediate frame has to clear as well -- this is
 * the half a per-state check would miss, and the half that a diagonally swung
 * pick failed. */
static void the_pick_clears_the_lock_all_the_way_between_pins(void)
{
	for (uint32_t pin_count = 3u; pin_count <= DUNGEON_LOCK_MAX_PINS; ++pin_count)
		for (uint32_t from_height = 0; from_height < DUNGEON_LOCK_PIN_STATES; ++from_height)
			for (uint32_t to_height = 0; to_height < DUNGEON_LOCK_PIN_STATES; ++to_height)
				for (uint32_t from = 0; from < pin_count; ++from)
					for (uint32_t to = 0; to < pin_count; ++to)
					{
						DungeonPinTumbler pins = {.pin_count = pin_count};
						for (uint32_t i = 0; i < pin_count; ++i)
						{
							/* Worst case for the sweep: every pin the shaft
							 * passes is raised as far as it goes. */
							pins.heights[i] = (uint8_t)(DUNGEON_LOCK_PIN_STATES - 1u);
							pins.target[i] = (uint8_t)(DUNGEON_LOCK_PIN_STATES - 1u);
						}
						pins.heights[from] = (uint8_t)from_height;
						pins.heights[to] = (uint8_t)to_height;
						float start_x = 0.0f, start_y = 0.0f, end_x = 0.0f, end_y = 0.0f;
						dungeon_lock_layout_pick_target(&pins, from, &start_x, &start_y);
						dungeon_lock_layout_pick_target(&pins, to, &end_x, &end_y);
						const uint32_t steps = 24u;
						for (uint32_t step = 0; step <= steps; ++step)
						{
							float t = (float)step / (float)steps;
							float x = start_x + (end_x - start_x) * t;
							float y = start_y + (end_y - start_y) * t;
							assert(dungeon_lock_layout_pick_clear(&pins, x, y));
						}
					}
}

/* Overlap has to be a real intersection test, not "boxes are near each other",
 * or the sweep above passes for the wrong reason. */
static void box_overlap_is_exact(void)
{
	DungeonLockBox a = {{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}};
	DungeonLockBox inside = {{0.4f, 0.4f, 0.4f}, {0.6f, 0.6f, 0.6f}};
	DungeonLockBox touching = {{1.0f, 0.0f, 0.0f}, {2.0f, 1.0f, 1.0f}};
	DungeonLockBox apart = {{1.001f, 0.0f, 0.0f}, {2.0f, 1.0f, 1.0f}};
	DungeonLockBox behind = {{0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 2.0f}};
	assert(dungeon_lock_box_overlap(&a, &inside));
	assert(!dungeon_lock_box_overlap(&a, &touching)); /* face contact is not overlap */
	assert(!dungeon_lock_box_overlap(&a, &apart));
	assert(!dungeon_lock_box_overlap(&a, &behind)); /* separated on z alone still separates */

	/* And a pick deliberately shoved back into the mechanism must be caught,
	 * otherwise the sweeps above prove nothing. */
	DungeonPinTumbler pins = {.pin_count = 4u};
	DungeonLockBox parts[DUNGEON_LOCK_MAX_PARTS];
	uint32_t part_count = dungeon_lock_layout_parts(&pins, parts, DUNGEON_LOCK_MAX_PARTS);
	assert(part_count > 4u);
	DungeonLockBox intruder = {{-0.05f, 0.60f, DUNGEON_LOCK_PROUD_BACK_M - 0.005f},
							   {0.05f, 0.70f, DUNGEON_LOCK_PROUD_BACK_M + 0.005f}};
	bool hit = false;
	for (uint32_t i = 0; i < part_count; ++i)
		hit = hit || dungeon_lock_box_overlap(&intruder, &parts[i]);
	assert(hit);
}

int main(void)
{
	box_overlap_is_exact();
	the_pick_clears_the_lock_in_every_pin_state();
	the_pick_clears_the_lock_all_the_way_between_pins();
	puts("dungeon lock layout tests passed");
	return 0;
}
