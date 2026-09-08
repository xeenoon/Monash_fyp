/* The lock session on top of a real generated level: interaction range, the
 * phase machine, and the side effect that matters most -- a solved lock has to
 * actually drop the door's collider, or the player picks it and still cannot
 * walk through. */
#include "dungeon_cave.h"
#include "dungeon_session.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Seeds are generated levels, so find one with both lock kinds present rather
 * than hand-building a level the compile path would never produce. */
static bool compile_level_with_lock(DungeonLockKind wanted, DungeonLevel *out, uint32_t *out_door)
{
	for (uint32_t seed = 1u; seed <= 40u; ++seed)
	{
		DungeonCaveParams params = dungeon_cave_default_params(seed);
		DungeonLevelError error = {0};
		DungeonLevel level = {0};
		if (!dungeon_cave_compile(&params, &level, &error))
			continue;
		for (uint32_t i = 0; i < level.door_count; ++i)
			if (level.doors[i].lock == wanted)
			{
				*out = level;
				*out_door = i;
				return true;
			}
		dungeon_level_destroy(&level);
	}
	return false;
}

static bool collider_present(const DungeonSession *session, DungeonSegment segment)
{
	for (uint32_t i = 0; i < session->collider_count; ++i)
	{
		const DungeonSegment *other = &session->colliders[i].segment;
		if (fabsf(other->a.x - segment.a.x) < 1e-4f && fabsf(other->a.z - segment.a.z) < 1e-4f &&
			fabsf(other->b.x - segment.b.x) < 1e-4f && fabsf(other->b.z - segment.b.z) < 1e-4f)
			return true;
	}
	return false;
}

static void a_shut_door_blocks_and_a_picked_one_does_not(void)
{
	DungeonLevel level = {0};
	uint32_t door_index = 0;
	assert(compile_level_with_lock(DUNGEON_LOCK_PIN_TUMBLER, &level, &door_index));
	DungeonSession session = {0};
	assert(dungeon_session_create(&session, &level));

	DungeonSegment blocker = level.doors[door_index].blocker;
	/* Shut: the door contributes one collider AND one light occluder on top of
	 * the level's own, so it stops the player and casts an occlusion shadow. */
	assert(session.collider_count == level.collider_count + session.door_count);
	assert(session.occluder_count == level.occluder_count + session.door_count);
	assert(collider_present(&session, blocker));

	/* Out of range, interacting does nothing but say so. */
	DungeonPoint far_away = {level.doors[door_index].center.x + 40.0f,
							 level.doors[door_index].center.z + 40.0f};
	assert(!dungeon_session_interact(&session, far_away));
	assert(session.phase == DUNGEON_PHASE_EXPLORING);
	assert(session.status == DUNGEON_STATUS_NOTHING_IN_REACH);

	DungeonPoint at_door = level.doors[door_index].center;
	assert(dungeon_session_interact(&session, at_door));
	assert(session.phase == DUNGEON_PHASE_PIN_TUMBLER);
	assert(session.focused_door == door_index);

	/* A wrong submission leaves the player standing at a shut door, still in
	 * the puzzle -- it must not quietly bail back to exploring. */
	DungeonPinTumbler *pins = &session.doors[door_index].pins;
	pins->heights[0] = (uint8_t)((pins->target[0] + 1u) % DUNGEON_LOCK_PIN_STATES);
	assert(!dungeon_session_confirm(&session));
	assert(session.phase == DUNGEON_PHASE_PIN_TUMBLER);
	assert(session.status == DUNGEON_STATUS_PINS_DO_NOT_ALIGN);
	assert(!session.doors[door_index].open);
	assert(collider_present(&session, blocker));

	for (uint32_t pin = 0; pin < pins->pin_count; ++pin)
		pins->heights[pin] = pins->target[pin];
	assert(dungeon_session_confirm(&session));
	assert(session.phase == DUNGEON_PHASE_EXPLORING);
	assert(session.doors[door_index].open);
	assert(!collider_present(&session, blocker));
	assert(session.collider_count == level.collider_count + session.door_count - 1u);
	assert(session.occluder_count == level.occluder_count + session.door_count - 1u);

	/* An opened door is no longer an interaction target. */
	assert(dungeon_session_nearest_door(&session, at_door) != door_index);

	/* The swing eases open rather than snapping, but the collider went the
	 * instant it unlocked -- the player never waits on an animation. */
	assert(session.doors[door_index].swing < 1.0f);
	for (int frame = 0; frame < 240; ++frame)
		dungeon_session_update(&session, 1.0f / 60.0f);
	assert(session.doors[door_index].swing > 0.99f);

	dungeon_session_destroy(&session);
	dungeon_level_destroy(&level);
}

/* The safe dial through the session: held turning at a fixed tick rate however
 * fast the frame loop runs, a red index pin that lights on the number, and a
 * confirm that either banks it or throws the whole combination away. */
static void the_dial_banks_on_confirm_and_resets_on_a_dark_pin(void)
{
	DungeonLevel level = {0};
	uint32_t door_index = 0;
	assert(compile_level_with_lock(DUNGEON_LOCK_VAULT_DIAL, &level, &door_index));
	DungeonSession session = {0};
	assert(dungeon_session_create(&session, &level));

	DungeonPoint at_door = level.doors[door_index].center;
	assert(dungeon_session_interact(&session, at_door));
	assert(session.phase == DUNGEON_PHASE_VAULT_DIAL);

	DungeonDoorState *state = &session.doors[door_index];
	assert(state->dial.position == 0u && state->dial_shake == 0.0f);

	/* A frame shorter than a tick moves nothing; the banked time carries over,
	 * so the dial turns at the same rate at any frame rate. */
	uint32_t before = state->dial.position;
	dungeon_session_spin_dial(&session, 1.0f, DUNGEON_DIAL_TICK_SECONDS * 0.4f);
	assert(state->dial.position == before);
	dungeon_session_spin_dial(&session, 1.0f, DUNGEON_DIAL_TICK_SECONDS * 0.7f);
	assert(state->dial.position == before + 1u);

	/* Letting go drops the part-banked tick, so tapping cannot creep the dial
	 * round between presses. */
	dungeon_session_spin_dial(&session, 1.0f, DUNGEON_DIAL_TICK_SECONDS * 0.9f);
	dungeon_session_spin_dial(&session, 0.0f, DUNGEON_DIAL_TICK_SECONDS);
	uint32_t held = state->dial.position;
	dungeon_session_spin_dial(&session, 1.0f, DUNGEON_DIAL_TICK_SECONDS * 0.5f);
	assert(state->dial.position == held);

	/* Bank two numbers, then confirm on a dark pin and lose both. */
	uint32_t guard = 4000u;
	while (state->dial.progress < 2u && guard--)
		if (dungeon_session_spin_dial(&session, 1.0f, DUNGEON_DIAL_TICK_SECONDS))
			assert(dungeon_session_commit_dial(&session));
	assert(state->dial.progress == 2u);
	assert(session.phase == DUNGEON_PHASE_VAULT_DIAL);

	guard = DUNGEON_DIAL_POSITIONS + 1u;
	while (dungeon_session_spin_dial(&session, 1.0f, DUNGEON_DIAL_TICK_SECONDS) && guard--)
		;
	assert(!dungeon_session_commit_dial(&session));
	assert(state->dial.progress == 0u);
	assert(session.status == DUNGEON_STATUS_DIAL_RESET);
	assert(!session.doors[door_index].open);
	/* A reset still jolts: confirm always lands with feedback, whichever way it
	 * went, or a player cannot tell it registered at all. */
	assert(state->dial_shake > 0.5f);
	for (int frame = 0; frame < 120; ++frame)
		dungeon_session_update(&session, 1.0f / 60.0f);
	assert(state->dial_shake == 0.0f);

	/* Work it the whole way and it opens on the last number. */
	guard = 4000u;
	while (session.phase == DUNGEON_PHASE_VAULT_DIAL && guard--)
		if (dungeon_session_spin_dial(&session, 1.0f, DUNGEON_DIAL_TICK_SECONDS))
			dungeon_session_commit_dial(&session);
	assert(session.doors[door_index].open);
	assert(session.phase == DUNGEON_PHASE_EXPLORING);

	dungeon_session_destroy(&session);
	dungeon_level_destroy(&level);
}

/* Every generated door must be reachable from spawn while every other door is
 * still shut -- otherwise the level asks the player to pick a lock they cannot
 * get to. The generator guarantees each door is a chokepoint on the route, so
 * the doors form a chain: at any moment exactly one is the next one. */
static void locks_are_encountered_one_at_a_time(void)
{
	for (uint32_t seed = 1u; seed <= 12u; ++seed)
	{
		DungeonCaveParams params = dungeon_cave_default_params(seed);
		DungeonLevel level = {0};
		DungeonLevelError error = {0};
		assert(dungeon_cave_compile(&params, &level, &error));
		DungeonSession session = {0};
		assert(dungeon_session_create(&session, &level));
		assert(session.door_count == level.door_count);
		for (uint32_t i = 0; i < session.door_count; ++i)
		{
			assert(level.doors[i].lock != DUNGEON_LOCK_NONE);
			/* Both puzzle kinds are seeded and non-trivial at construction. */
			assert(session.doors[i].pins.pin_count >= 3u);
			assert(session.doors[i].dial.step_count >= 4u);
			assert(session.doors[i].dial.position == 0u);
			assert(!session.doors[i].open);
			assert(fabsf(session.doors[i].hardware_side) == 1.0f);
		}
		dungeon_session_destroy(&session);
		dungeon_level_destroy(&level);
	}
}


/* Two seeding bugs that made every lock in the game feel like the same lock,
 * both found by playing rather than by any test here -- hence this one.
 *
 * The first: splitmix advances its state by the same golden constant a naive
 * `seed * K + C` seeding multiplies by, so "which seed" and "how far into the
 * stream" collapse onto one axis. Seed s after n draws IS seed s+1 after n-1
 * draws, and since adjacent DUNGEON_SEEDs differ by only a draw or two before
 * the doors are placed, whole levels shared their combinations.
 *
 * The second: a door seeded both its lock kinds from the same value, so its
 * dial combination came out as the same digits as its pin target. */
static void every_lock_gets_its_own_answer(void)
{
	uint32_t seeds[64] = {0};
	uint32_t seed_count = 0;

	for (uint32_t level_seed = 1u; level_seed <= 20u; ++level_seed)
	{
		DungeonCaveParams params = dungeon_cave_default_params(level_seed);
		DungeonLevel level = {0};
		DungeonLevelError error = {0};
		assert(dungeon_cave_compile(&params, &level, &error));
		DungeonSession session = {0};
		assert(dungeon_session_create(&session, &level));

		for (uint32_t door = 0; door < session.door_count; ++door)
		{
			/* No door anywhere, in any level, repeats another's seed. */
			for (uint32_t i = 0; i < seed_count; ++i)
				assert(seeds[i] != level.doors[door].seed);
			if (seed_count < 64u)
				seeds[seed_count++] = level.doors[door].seed;

		}
		dungeon_session_destroy(&session);
		dungeon_level_destroy(&level);
	}
	assert(seed_count > 20u); /* the sweep actually produced doors to check */

	/* And the two locks on a door draw from separate streams. Tested on the
	 * primitives rather than on generated doors, because a handful of doors
	 * colliding by chance is expected -- what is not expected, and what the
	 * shared-seed bug produced, is EVERY seed colliding. Chance here is about
	 * 1 in 64; the assertion leaves a wide margin above that and still fails
	 * outright on the old behaviour. */
	uint32_t collisions = 0, trials = 0;
	for (uint32_t seed = 1u; seed <= 400u; ++seed)
	{
		DungeonPinTumbler pins = {0};
		DungeonVaultDial dial = {0};
		dungeon_pin_tumbler_init(&pins, seed, 4u);
		dungeon_vault_dial_init(&dial, seed, 4u);
		bool identical = true;
		for (uint32_t i = 0; i < 4u; ++i)
			if ((uint32_t)pins.target[i] !=
				(uint32_t)dial.combination[i] % DUNGEON_LOCK_PIN_STATES)
				identical = false;
		collisions += identical ? 1u : 0u;
		++trials;
	}
	assert(collisions * 8u < trials);
}

int main(void)
{
	a_shut_door_blocks_and_a_picked_one_does_not();
	the_dial_banks_on_confirm_and_resets_on_a_dark_pin();
	locks_are_encountered_one_at_a_time();
	every_lock_gets_its_own_answer();
	puts("dungeon session tests passed");
	return 0;
}
