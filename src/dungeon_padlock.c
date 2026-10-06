#include "dungeon_padlock.h"

#include <math.h>
#include <string.h>

const char *const DUNGEON_PADLOCK_CLIP_NAMES[DUNGEON_PADLOCK_CLIP_COUNT] = {
	[DUNGEON_PADLOCK_CLIP_INSERT] = "Pick_Insert",
	[DUNGEON_PADLOCK_CLIP_WITHDRAW] = "Pick_Withdraw",
	[DUNGEON_PADLOCK_CLIP_JIGGLE] = "Pick_Jiggle_Loop",
	[DUNGEON_PADLOCK_CLIP_PIN_01_POP] = "Pin_01_Pop",
	[DUNGEON_PADLOCK_CLIP_PIN_02_POP] = "Pin_02_Pop",
	[DUNGEON_PADLOCK_CLIP_PIN_03_POP] = "Pin_03_Pop",
	[DUNGEON_PADLOCK_CLIP_PIN_04_POP] = "Pin_04_Pop",
	[DUNGEON_PADLOCK_CLIP_UNLOCK] = "Unlock_All_Four_Set",
};

static void play(DungeonPadlockAnim *anim, DungeonPadlockClip clip, bool loop)
{
	anim->layers[clip] = (DungeonPadlockLayer){.playing = true, .loop = loop, .time = 0.0f};
}

static void stop(DungeonPadlockAnim *anim, DungeonPadlockClip clip)
{
	anim->layers[clip] = (DungeonPadlockLayer){0};
}

static bool finished(const DungeonPadlockAnim *anim, DungeonPadlockClip clip,
					 const float *durations)
{
	return anim->layers[clip].playing && !anim->layers[clip].loop &&
		   anim->layers[clip].time >= durations[clip];
}

void dungeon_padlock_reset(DungeonPadlockAnim *anim)
{
	if (anim)
		*anim = (DungeonPadlockAnim){0};
}

void dungeon_padlock_engage(DungeonPadlockAnim *anim, bool engaged)
{
	if (!anim || anim->engaged == engaged || anim->unlocking)
		return;
	anim->engaged = engaged;
	if (engaged)
	{
		stop(anim, DUNGEON_PADLOCK_CLIP_WITHDRAW);
		play(anim, DUNGEON_PADLOCK_CLIP_INSERT, false);
	}
	else
	{
		/* The jiggle goes at once rather than being left to finish its loop:
		 * the withdraw starts from the pick seated in the keyway, and a jiggle
		 * still running underneath it would fight it for the same node. */
		stop(anim, DUNGEON_PADLOCK_CLIP_JIGGLE);
		stop(anim, DUNGEON_PADLOCK_CLIP_INSERT);
		play(anim, DUNGEON_PADLOCK_CLIP_WITHDRAW, false);
	}
}

void dungeon_padlock_pop_pin(DungeonPadlockAnim *anim, uint32_t pin)
{
	if (!anim || pin >= DUNGEON_PADLOCK_PINS || anim->popped[pin])
		return;
	anim->popped[pin] = true;
	play(anim, (DungeonPadlockClip)(DUNGEON_PADLOCK_CLIP_PIN_01_POP + pin), false);
}

void dungeon_padlock_unlock(DungeonPadlockAnim *anim)
{
	if (!anim || anim->unlocking)
		return;
	anim->unlocking = true;
	anim->engaged = false;
	stop(anim, DUNGEON_PADLOCK_CLIP_JIGGLE);
	stop(anim, DUNGEON_PADLOCK_CLIP_INSERT);
	stop(anim, DUNGEON_PADLOCK_CLIP_WITHDRAW);
	/* The pin pops stay held: the unlock clip's own pin channels take the four
	 * stacks from set to turned, and dropping the pops here would spring every
	 * pin back out for the frame between the two. */
	play(anim, DUNGEON_PADLOCK_CLIP_UNLOCK, false);
}

/* Fast enough that a key press lands as a movement rather than a drift, slow
 * enough to be a movement at all: a slot is about six millimetres at the drawn
 * scale, and the eye needs a few frames of it to register as the pin rising. */
#define DUNGEON_PADLOCK_PIN_EASE 18.0f

void dungeon_padlock_show_heights(DungeonPadlockAnim *anim, const uint8_t *heights,
								  uint32_t count, float dt)
{
	if (!anim || !heights || !(dt > 0.0f))
		return;
	float blend = 1.0f - expf(-DUNGEON_PADLOCK_PIN_EASE * dt);
	for (uint32_t pin = 0; pin < DUNGEON_PADLOCK_PINS && pin < count; ++pin)
		anim->shown[pin] += ((float)heights[pin] - anim->shown[pin]) * blend;
}

void dungeon_padlock_update(DungeonPadlockAnim *anim, const float *durations, float dt)
{
	if (!anim || !durations || !(dt > 0.0f))
		return;
	for (uint32_t i = 0; i < DUNGEON_PADLOCK_CLIP_COUNT; ++i)
	{
		DungeonPadlockLayer *layer = &anim->layers[i];
		if (!layer->playing)
			continue;
		layer->time += dt;
		if (layer->loop && durations[i] > 0.0f)
			while (layer->time >= durations[i])
				layer->time -= durations[i];
		else if (layer->time > durations[i])
			layer->time = durations[i]; /* hold the last pose; sampling clamps */
	}
	/* The insert hands over to the loop rather than the two overlapping: both
	 * drive the pick, and the jiggle's first frame is the insert's last. */
	if (finished(anim, DUNGEON_PADLOCK_CLIP_INSERT, durations) && anim->engaged)
	{
		stop(anim, DUNGEON_PADLOCK_CLIP_INSERT);
		play(anim, DUNGEON_PADLOCK_CLIP_JIGGLE, true);
	}
	/* A finished withdraw leaves the pick where the clip parks it -- out of the
	 * keyway and at rest -- so nothing has to hold it there. */
	if (finished(anim, DUNGEON_PADLOCK_CLIP_WITHDRAW, durations) && !anim->engaged)
		stop(anim, DUNGEON_PADLOCK_CLIP_WITHDRAW);
}

bool dungeon_padlock_pick_visible(const DungeonPadlockAnim *anim)
{
	if (!anim)
		return false;
	/* Exactly the clips that drive the pick. Anything else playing -- a pin
	 * popping on its own -- is the lock's business, not the pick's. */
	return anim->layers[DUNGEON_PADLOCK_CLIP_INSERT].playing ||
		   anim->layers[DUNGEON_PADLOCK_CLIP_JIGGLE].playing ||
		   anim->layers[DUNGEON_PADLOCK_CLIP_WITHDRAW].playing ||
		   anim->layers[DUNGEON_PADLOCK_CLIP_UNLOCK].playing;
}

bool dungeon_padlock_pin_popped(const DungeonPadlockAnim *anim, uint32_t pin)
{
	return anim && pin < DUNGEON_PADLOCK_PINS && anim->popped[pin];
}
