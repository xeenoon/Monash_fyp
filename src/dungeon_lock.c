#include "dungeon_lock.h"

/* The same splitmix64 the cave generator uses, kept local for the same reason:
 * explicit state, so a given DUNGEON_SEED reproduces a given combination no
 * matter what else in the process has drawn random numbers first.
 *
 * `domain` separates the two locks' streams. Seeding both from the door's seed
 * alone gave a door's safe order the same digits as its pin target, because
 * both drew the same numbers from the same starting state.
 *
 * The finalizer on the seed matters for the same reason it does in
 * dungeon_cave.c: lock_rng_next advances by the constant a plain seeding would
 * multiply by, which would make neighbouring seeds share streams. */
typedef struct
{
	uint64_t state;
} LockRng;

#define DUNGEON_LOCK_DOMAIN_PINS 0x5FA1B7C39D2E4801ULL
#define DUNGEON_LOCK_DOMAIN_SAFE 0xC13E9A47B62D5F0BULL

static LockRng lock_rng_create(uint32_t seed, uint64_t domain)
{
	uint64_t state = (uint64_t)seed * 0x9E3779B97F4A7C15ULL + domain;
	state = (state ^ (state >> 30)) * 0xBF58476D1CE4E5B9ULL;
	state = (state ^ (state >> 27)) * 0x94D049BB133111EBULL;
	LockRng rng = {.state = state ^ (state >> 31)};
	return rng;
}

static uint64_t lock_rng_next(LockRng *rng)
{
	uint64_t z = (rng->state += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

static uint32_t lock_rng_index(LockRng *rng, uint32_t count)
{
	return count ? (uint32_t)(lock_rng_next(rng) % count) : 0u;
}

static uint32_t clamp_count(uint32_t count, uint32_t low, uint32_t high)
{
	return count < low ? low : (count > high ? high : count);
}

void dungeon_pin_tumbler_init(DungeonPinTumbler *lock, uint32_t seed, uint32_t pin_count)
{
	if (!lock)
		return;
	*lock = (DungeonPinTumbler){0};
	lock->pin_count = clamp_count(pin_count, 2u, DUNGEON_LOCK_MAX_PINS);
	LockRng rng = lock_rng_create(seed, DUNGEON_LOCK_DOMAIN_PINS);
	/* Zero remains the unset starting position; targets use the original three
	 * set levels. */
	for (uint32_t pin = 0; pin < lock->pin_count; ++pin)
		lock->target[pin] = (uint8_t)(1u + lock_rng_index(&rng, DUNGEON_LOCK_PIN_STATES - 1u));
	/* Avoid the one degenerate result where the whole solved row is level. */
	bool all_same = true;
	for (uint32_t pin = 1; pin < lock->pin_count; ++pin)
		all_same = all_same && lock->target[pin] == lock->target[0];
	if (all_same && lock->pin_count > 1u)
		lock->target[lock->pin_count - 1u] =
			(uint8_t)(1u + lock->target[0] % (DUNGEON_LOCK_PIN_STATES - 1u));
}

void dungeon_pin_tumbler_move(DungeonPinTumbler *lock, int direction)
{
	if (!lock || !lock->pin_count)
		return;
	long selected = (long)lock->selected + direction;
	if (selected < 0)
		selected = 0;
	if (selected > (long)lock->pin_count - 1)
		selected = (long)lock->pin_count - 1;
	lock->selected = (uint32_t)selected;
}

bool dungeon_pin_tumbler_pin_set(const DungeonPinTumbler *lock, uint32_t pin)
{
	if (!lock || pin >= lock->pin_count)
		return false;
	return lock->heights[pin] == lock->target[pin];
}

void dungeon_pin_tumbler_adjust(DungeonPinTumbler *lock, int direction)
{
	if (!lock || lock->selected >= lock->pin_count || direction <= 0)
		return;
	/* Set pins are locked in place. Checked before the wrap below, so a pin can
	 * never be walked off its target and back round. */
	if (dungeon_pin_tumbler_pin_set(lock, lock->selected))
		return;
	/* Up is the only tumbler action: one press raises one visible level. There
	 * is no downward wrap from level zero to the top, which made every pin look
	 * as though it had to be rammed fully upward. Since every target is above
	 * zero, stepping upward must encounter it; the set check above freezes the
	 * pin there permanently. */
	lock->heights[lock->selected]++;
}

bool dungeon_pin_tumbler_submit(DungeonPinTumbler *lock)
{
	if (!lock)
		return false;
	bool matched = lock->pin_count > 0u;
	for (uint32_t pin = 0; pin < lock->pin_count; ++pin)
		if (lock->heights[pin] != lock->target[pin])
			matched = false;
	lock->solved = matched;
	return matched;
}

void dungeon_safe_pins_init(DungeonSafePins *lock, uint32_t seed)
{
	if (!lock)
		return;
	*lock = (DungeonSafePins){0};
	lock->pin_count = DUNGEON_SAFE_PIN_COUNT;
	for (uint32_t pin = 0; pin < lock->pin_count; ++pin)
		lock->order[pin] = (uint8_t)pin;

	/* Fisher-Yates over the pins themselves, so the order is a PERMUTATION by
	 * construction rather than four independent draws. Independent draws can
	 * name the same pin twice, and the second press of a pin that is already
	 * standing proud has nothing to show for it -- the face would stop being a
	 * readable account of how far the player has got. */
	LockRng rng = lock_rng_create(seed, DUNGEON_LOCK_DOMAIN_SAFE);
	for (uint32_t i = lock->pin_count; i > 1u; --i)
	{
		uint32_t j = lock_rng_index(&rng, i);
		uint8_t swap = lock->order[i - 1u];
		lock->order[i - 1u] = lock->order[j];
		lock->order[j] = swap;
	}
}

bool dungeon_safe_pins_driven(const DungeonSafePins *lock, uint32_t pin)
{
	if (!lock || pin >= lock->pin_count)
		return false;
	uint32_t banked = lock->progress < lock->pin_count ? lock->progress : lock->pin_count;
	for (uint32_t step = 0; step < banked; ++step)
		if (lock->order[step] == pin)
			return true;
	return false;
}

bool dungeon_safe_pins_selected_is_next(const DungeonSafePins *lock)
{
	if (!lock || lock->solved || lock->progress >= lock->pin_count)
		return false;
	return lock->selected == lock->order[lock->progress];
}

void dungeon_safe_pins_move(DungeonSafePins *lock, int direction)
{
	if (!lock || !lock->pin_count || lock->solved || !direction)
		return;
	int step = direction > 0 ? 1 : -1;
	int count = (int)lock->pin_count;
	lock->selected = (uint32_t)(((int)lock->selected + step + count) % count);
}

bool dungeon_safe_pins_press(DungeonSafePins *lock)
{
	if (!lock || !lock->pin_count || lock->solved)
		return false;
	if (!dungeon_safe_pins_selected_is_next(lock))
	{
		/* Pressing the wrong pin -- an untouched one out of turn, or one
		 * already driven -- throws the whole order away. Moving the selection
		 * costs nothing, so this is the only way to lose progress, and the only
		 * reason to think before pressing. */
		lock->progress = 0;
		return false;
	}
	++lock->progress;
	lock->solved = lock->progress == lock->pin_count;
	return true;
}
