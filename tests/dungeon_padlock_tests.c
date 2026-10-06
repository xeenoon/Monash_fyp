/* The padlock's clip state machine, headlessly: no renderer, no glTF, no
 * asset. What is pinned here is the layering contract the exporter's channel
 * stripping exists to serve -- a pin can pop WHILE the pick is working, and a
 * finished one-shot holds its last pose instead of springing back. */
#include "dungeon_padlock.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* Durations standing in for the asset's, so the machine is exercised without
 * one. dungeon_padlock never hardcodes a clip length; it is handed them. */
static const float DURATIONS[DUNGEON_PADLOCK_CLIP_COUNT] = {
	[DUNGEON_PADLOCK_CLIP_INSERT] = 1.75f,   [DUNGEON_PADLOCK_CLIP_WITHDRAW] = 1.75f,
	[DUNGEON_PADLOCK_CLIP_JIGGLE] = 2.0f,    [DUNGEON_PADLOCK_CLIP_PIN_01_POP] = 1.75f,
	[DUNGEON_PADLOCK_CLIP_PIN_02_POP] = 1.75f, [DUNGEON_PADLOCK_CLIP_PIN_03_POP] = 1.75f,
	[DUNGEON_PADLOCK_CLIP_PIN_04_POP] = 1.75f, [DUNGEON_PADLOCK_CLIP_UNLOCK] = 3.75f,
};

static void step(DungeonPadlockAnim *anim, float seconds)
{
	for (float elapsed = 0.0f; elapsed < seconds; elapsed += 1.0f / 60.0f)
		dungeon_padlock_update(anim, DURATIONS, 1.0f / 60.0f);
}

static bool playing(const DungeonPadlockAnim *anim, DungeonPadlockClip clip)
{
	return anim->layers[clip].playing;
}

static void every_clip_is_named(void)
{
	for (uint32_t i = 0; i < DUNGEON_PADLOCK_CLIP_COUNT; ++i)
	{
		assert(DUNGEON_PADLOCK_CLIP_NAMES[i]);
		assert(DUNGEON_PADLOCK_CLIP_NAMES[i][0]);
		/* Names, not indices, are what survives a re-export of the asset. */
		for (uint32_t j = 0; j < i; ++j)
			assert(strcmp(DUNGEON_PADLOCK_CLIP_NAMES[i], DUNGEON_PADLOCK_CLIP_NAMES[j]) != 0);
	}
}

static void engaging_inserts_then_loops(void)
{
	DungeonPadlockAnim anim;
	dungeon_padlock_reset(&anim);
	assert(!playing(&anim, DUNGEON_PADLOCK_CLIP_INSERT));
	/* A lock nobody has touched carries no pick. */
	assert(!dungeon_padlock_pick_visible(&anim));
	dungeon_padlock_engage(&anim, true);
	assert(dungeon_padlock_pick_visible(&anim));
	assert(playing(&anim, DUNGEON_PADLOCK_CLIP_INSERT));
	assert(!playing(&anim, DUNGEON_PADLOCK_CLIP_JIGGLE));
	/* Called again every frame from the session's phase: it must not restart. */
	step(&anim, 0.5f);
	float before = anim.layers[DUNGEON_PADLOCK_CLIP_INSERT].time;
	dungeon_padlock_engage(&anim, true);
	assert(anim.layers[DUNGEON_PADLOCK_CLIP_INSERT].time == before);

	step(&anim, DURATIONS[DUNGEON_PADLOCK_CLIP_INSERT] + 0.1f);
	assert(!playing(&anim, DUNGEON_PADLOCK_CLIP_INSERT));
	assert(playing(&anim, DUNGEON_PADLOCK_CLIP_JIGGLE));
	assert(anim.layers[DUNGEON_PADLOCK_CLIP_JIGGLE].loop);
	/* The loop wraps rather than parking at the end. */
	step(&anim, DURATIONS[DUNGEON_PADLOCK_CLIP_JIGGLE] * 2.5f);
	assert(playing(&anim, DUNGEON_PADLOCK_CLIP_JIGGLE));
	assert(anim.layers[DUNGEON_PADLOCK_CLIP_JIGGLE].time < DURATIONS[DUNGEON_PADLOCK_CLIP_JIGGLE]);
}

static void disengaging_withdraws_and_drops_the_jiggle(void)
{
	DungeonPadlockAnim anim;
	dungeon_padlock_reset(&anim);
	dungeon_padlock_engage(&anim, true);
	step(&anim, DURATIONS[DUNGEON_PADLOCK_CLIP_INSERT] + 0.5f);
	assert(playing(&anim, DUNGEON_PADLOCK_CLIP_JIGGLE));
	dungeon_padlock_engage(&anim, false);
	/* Both drive the pick: a jiggle left running under the withdraw would
	 * fight it for the same node, and the later layer would simply win. */
	assert(!playing(&anim, DUNGEON_PADLOCK_CLIP_JIGGLE));
	assert(playing(&anim, DUNGEON_PADLOCK_CLIP_WITHDRAW));
	assert(dungeon_padlock_pick_visible(&anim));
	step(&anim, DURATIONS[DUNGEON_PADLOCK_CLIP_WITHDRAW] + 0.1f);
	/* A finished withdraw leaves nothing playing: the pick is out. */
	for (uint32_t i = 0; i < DUNGEON_PADLOCK_CLIP_COUNT; ++i)
		assert(!playing(&anim, (DungeonPadlockClip)i));
	/* And with nothing playing the pick is not drawn at all: the model's rest
	 * pose has it sitting in the keyway, so a door nobody is working would
	 * otherwise wear a lockpick for ever. */
	assert(!dungeon_padlock_pick_visible(&anim));
}

static void a_pin_pops_while_the_pick_works(void)
{
	DungeonPadlockAnim anim;
	dungeon_padlock_reset(&anim);
	dungeon_padlock_engage(&anim, true);
	step(&anim, DURATIONS[DUNGEON_PADLOCK_CLIP_INSERT] + 0.2f);
	dungeon_padlock_pop_pin(&anim, 1u);
	/* A pin popping on its own is the lock's business, not the pick's. */
	assert(dungeon_padlock_pick_visible(&anim));
	/* The whole point of the exporter stripping rest channels: these two run
	 * at once, over node sets that do not overlap. */
	assert(playing(&anim, DUNGEON_PADLOCK_CLIP_JIGGLE));
	assert(playing(&anim, DUNGEON_PADLOCK_CLIP_PIN_02_POP));
	assert(!playing(&anim, DUNGEON_PADLOCK_CLIP_PIN_01_POP));
	assert(dungeon_padlock_pin_popped(&anim, 1u));
	assert(!dungeon_padlock_pin_popped(&anim, 0u));

	/* A set pin cannot be knocked back out, so neither can its clip restart. */
	step(&anim, 0.3f);
	float elapsed = anim.layers[DUNGEON_PADLOCK_CLIP_PIN_02_POP].time;
	dungeon_padlock_pop_pin(&anim, 1u);
	assert(anim.layers[DUNGEON_PADLOCK_CLIP_PIN_02_POP].time == elapsed);

	/* Once over, the pop HOLDS: it stays playing with its time parked at the
	 * duration, and sampling clamps, which is what keeps the pin set with no
	 * second pose to author. */
	step(&anim, DURATIONS[DUNGEON_PADLOCK_CLIP_PIN_02_POP] + 1.0f);
	assert(playing(&anim, DUNGEON_PADLOCK_CLIP_PIN_02_POP));
	assert(anim.layers[DUNGEON_PADLOCK_CLIP_PIN_02_POP].time ==
		   DURATIONS[DUNGEON_PADLOCK_CLIP_PIN_02_POP]);
	assert(playing(&anim, DUNGEON_PADLOCK_CLIP_JIGGLE));
}

static void opening_holds_every_pin_set(void)
{
	DungeonPadlockAnim anim;
	dungeon_padlock_reset(&anim);
	dungeon_padlock_engage(&anim, true);
	for (uint32_t pin = 0; pin < DUNGEON_PADLOCK_PINS; ++pin)
		dungeon_padlock_pop_pin(&anim, pin);
	step(&anim, 2.0f);
	dungeon_padlock_unlock(&anim);
	assert(playing(&anim, DUNGEON_PADLOCK_CLIP_UNLOCK));
	/* The pick stops working -- there is nothing left to pick -- but the pops
	 * stay held, or every pin would spring out for the frame between the pops
	 * ending and the unlock clip's own pin channels taking over. */
	assert(!playing(&anim, DUNGEON_PADLOCK_CLIP_JIGGLE));
	assert(!playing(&anim, DUNGEON_PADLOCK_CLIP_INSERT));
	assert(!playing(&anim, DUNGEON_PADLOCK_CLIP_WITHDRAW));
	for (uint32_t pin = 0; pin < DUNGEON_PADLOCK_PINS; ++pin)
		assert(playing(&anim, (DungeonPadlockClip)(DUNGEON_PADLOCK_CLIP_PIN_01_POP + pin)));

	/* An open lock stays open: engaging it again does nothing, and the opening
	 * clip holds its last pose rather than shutting the shackle. */
	dungeon_padlock_engage(&anim, true);
	step(&anim, DURATIONS[DUNGEON_PADLOCK_CLIP_UNLOCK] + 2.0f);
	assert(!playing(&anim, DUNGEON_PADLOCK_CLIP_INSERT));
	assert(playing(&anim, DUNGEON_PADLOCK_CLIP_UNLOCK));
	assert(anim.layers[DUNGEON_PADLOCK_CLIP_UNLOCK].time == DURATIONS[DUNGEON_PADLOCK_CLIP_UNLOCK]);
}

/* A key press has to read as a pin rising, so where a pin is DRAWN eases toward
 * where the puzzle put it rather than teleporting there -- and comes to rest on
 * its own once the puzzle stops moving it. */
static void a_pin_eases_toward_its_slot(void)
{
	DungeonPadlockAnim anim;
	dungeon_padlock_reset(&anim);
	const uint8_t raised[DUNGEON_PADLOCK_PINS] = {2u, 0u, 0u, 0u};
	assert(anim.shown[0] == 0.0f);
	dungeon_padlock_show_heights(&anim, raised, DUNGEON_PADLOCK_PINS, 1.0f / 60.0f);
	/* One frame in, it has started but nowhere near arrived. */
	assert(anim.shown[0] > 0.0f && anim.shown[0] < 1.0f);
	for (int i = 0; i < 240; ++i)
		dungeon_padlock_show_heights(&anim, raised, DUNGEON_PADLOCK_PINS, 1.0f / 60.0f);
	assert(fabsf(anim.shown[0] - 2.0f) < 1e-3f);
	/* Pins the puzzle has not touched stay put. */
	for (uint32_t pin = 1; pin < DUNGEON_PADLOCK_PINS; ++pin)
		assert(anim.shown[pin] == 0.0f);
	/* Settled is settled: no drift once it has arrived. */
	float arrived = anim.shown[0];
	for (int i = 0; i < 600; ++i)
		dungeon_padlock_show_heights(&anim, raised, DUNGEON_PADLOCK_PINS, 1.0f / 60.0f);
	assert(fabsf(anim.shown[0] - arrived) < 1e-4f);
}

static void out_of_range_pins_are_ignored(void)
{
	DungeonPadlockAnim anim;
	dungeon_padlock_reset(&anim);
	dungeon_padlock_pop_pin(&anim, DUNGEON_PADLOCK_PINS);
	dungeon_padlock_pop_pin(&anim, 4000u);
	for (uint32_t i = 0; i < DUNGEON_PADLOCK_CLIP_COUNT; ++i)
		assert(!playing(&anim, (DungeonPadlockClip)i));
	assert(!dungeon_padlock_pin_popped(&anim, DUNGEON_PADLOCK_PINS));
	/* Every entry point tolerates NULL: the scene calls them from a loop over
	 * doors that may have no padlock loaded at all. */
	dungeon_padlock_reset(NULL);
	dungeon_padlock_engage(NULL, true);
	dungeon_padlock_pop_pin(NULL, 0u);
	dungeon_padlock_unlock(NULL);
	dungeon_padlock_update(NULL, DURATIONS, 0.016f);
	dungeon_padlock_update(&anim, NULL, 0.016f);
	dungeon_padlock_show_heights(NULL, NULL, 0u, 0.016f);
	dungeon_padlock_show_heights(&anim, NULL, DUNGEON_PADLOCK_PINS, 0.016f);
}

int main(void)
{
	every_clip_is_named();
	engaging_inserts_then_loops();
	disengaging_withdraws_and_drops_the_jiggle();
	a_pin_pops_while_the_pick_works();
	opening_holds_every_pin_set();
	a_pin_eases_toward_its_slot();
	out_of_range_pins_are_ignored();
	printf("dungeon padlock animation tests passed\n");
	return 0;
}
