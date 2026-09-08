#include "dungeon_lock.h"

/* The same splitmix64 the cave generator uses, kept local for the same reason:
 * explicit state, so a given DUNGEON_SEED reproduces a given combination no
 * matter what else in the process has drawn random numbers first.
 *
 * `domain` separates the two locks' streams. Seeding both from the door's seed
 * alone gave a door's dial combination the same digits as its pin target,
 * because both drew the same numbers from the same starting state.
 *
 * The finalizer on the seed matters for the same reason it does in
 * dungeon_cave.c: lock_rng_next advances by the constant a plain seeding would
 * multiply by, which would make neighbouring seeds share streams. */
typedef struct
{
	uint64_t state;
} LockRng;

#define DUNGEON_LOCK_DOMAIN_PINS 0x5FA1B7C39D2E4801ULL
#define DUNGEON_LOCK_DOMAIN_DIAL 0xC13E9A47B62D5F0BULL

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
	bool trivial = true;
	while (trivial)
	{
		for (uint32_t pin = 0; pin < lock->pin_count; ++pin)
		{
			lock->target[pin] = (uint8_t)lock_rng_index(&rng, DUNGEON_LOCK_PIN_STATES);
			if (lock->target[pin] != 0u)
				trivial = false;
		}
	}
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

void dungeon_pin_tumbler_adjust(DungeonPinTumbler *lock, int direction)
{
	if (!lock || lock->selected >= lock->pin_count)
		return;
	int height = (int)lock->heights[lock->selected] + direction;
	int states = (int)DUNGEON_LOCK_PIN_STATES;
	height = ((height % states) + states) % states;
	lock->heights[lock->selected] = (uint8_t)height;
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

void dungeon_vault_dial_init(DungeonVaultDial *lock, uint32_t seed, uint32_t step_count)
{
	if (!lock)
		return;
	*lock = (DungeonVaultDial){0};
	lock->step_count = clamp_count(step_count, 2u, DUNGEON_LOCK_MAX_STEPS);
	LockRng rng = lock_rng_create(seed, DUNGEON_LOCK_DOMAIN_DIAL);
	for (uint32_t step = 0; step < lock->step_count; ++step)
		lock->combination[step] = (DungeonDialDirection)lock_rng_index(&rng, 4u);
}

bool dungeon_vault_dial_turn(DungeonVaultDial *lock, DungeonDialDirection direction)
{
	if (!lock || !lock->step_count)
		return false;
	if (lock->solved)
		return true;
	/* A miss resets to zero outright rather than re-testing the input against
	 * step 0, which is exactly what the source does. Do not "fix" this into a
	 * partial-match restart: it changes the difficulty of every combination. */
	lock->progress = direction == lock->combination[lock->progress] ? lock->progress + 1u : 0u;
	lock->solved = lock->progress == lock->step_count;
	return lock->solved;
}
