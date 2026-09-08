#include "dungeon_lock_layout.h"

static DungeonLockBox box_from(float centre_x, float half_x, float low_y, float high_y,
							   float centre_z, float half_z)
{
	return (DungeonLockBox){{centre_x - half_x, low_y, centre_z - half_z},
							{centre_x + half_x, high_y, centre_z + half_z}};
}

bool dungeon_lock_box_overlap(const DungeonLockBox *a, const DungeonLockBox *b)
{
	if (!a || !b)
		return false;
	for (uint32_t axis = 0; axis < 3u; ++axis)
		if (a->max[axis] <= b->min[axis] || b->max[axis] <= a->min[axis])
			return false;
	return true;
}

float dungeon_lock_layout_bore_lateral(const DungeonPinTumbler *pins, uint32_t pin)
{
	if (!pins || !pins->pin_count)
		return 0.0f;
	float span = (float)(pins->pin_count - 1u) * DUNGEON_LOCK_BORE_PITCH_M * 0.5f;
	return (float)pin * DUNGEON_LOCK_BORE_PITCH_M - span;
}

static float pin_base_height(const DungeonPinTumbler *pins, uint32_t pin)
{
	return DUNGEON_LOCK_PIN_BASE_Y + (float)pins->heights[pin] * DUNGEON_LOCK_PIN_RISE_M;
}

uint32_t dungeon_lock_layout_parts(const DungeonPinTumbler *pins, DungeonLockBox *out,
								   uint32_t capacity)
{
	if (!pins || !out)
		return 0;
	uint32_t count = 0;
#define EMIT(box)                                                                                  \
	do                                                                                             \
	{                                                                                              \
		if (count < capacity)                                                                      \
			out[count++] = (box);                                                                  \
	} while (0)

	/* The casing, taken as one envelope: bezel, keyhole boss and shackle all
	 * sit inside it, and nothing needs to distinguish them for a clearance
	 * test. */
	EMIT(box_from(0.0f, DUNGEON_LOCK_BODY_HALF_X, DUNGEON_LOCK_BODY_Y, DUNGEON_LOCK_BODY_TOP_Y,
				  0.0f, DUNGEON_LOCK_PROUD_BODY_M));
	EMIT(box_from(0.0f, DUNGEON_LOCK_HOUSING_HALF_X, DUNGEON_LOCK_HOUSING_Y,
				  DUNGEON_LOCK_HOUSING_TOP_Y, 0.0f, DUNGEON_LOCK_PROUD_HOUSING_M));

	for (uint32_t pin = 0; pin < pins->pin_count; ++pin)
	{
		float lateral = dungeon_lock_layout_bore_lateral(pins, pin);
		EMIT(box_from(lateral, DUNGEON_LOCK_CHANNEL_HALF_X, DUNGEON_LOCK_CHANNEL_Y,
					  DUNGEON_LOCK_CHANNEL_TOP_Y, DUNGEON_LOCK_PROUD_BACK_M, 0.020f));

		bool seated = pins->heights[pin] == pins->target[pin];
		float proud = DUNGEON_LOCK_PROUD_PIN_M - (seated ? DUNGEON_LOCK_PIN_SET_SINK_M : 0.0f);
		float base = pin_base_height(pins, pin);
		EMIT(box_from(lateral, DUNGEON_LOCK_PIN_RADIUS_M, base, base + DUNGEON_LOCK_PIN_LENGTH_M,
					  proud, DUNGEON_LOCK_PIN_RADIUS_M));

		/* The spring coils occupy the headroom left above the pin; one box over
		 * the whole stack is enough for clearance purposes. */
		float coil_low = base + DUNGEON_LOCK_PIN_LENGTH_M + 0.010f;
		float coil_high = DUNGEON_LOCK_HOUSING_TOP_Y - 0.030f;
		if (coil_high > coil_low)
			EMIT(box_from(lateral, DUNGEON_LOCK_SPRING_RADIUS_M, coil_low, coil_high,
						  DUNGEON_LOCK_PROUD_PIN_M, DUNGEON_LOCK_SPRING_RADIUS_M));
	}
#undef EMIT
	return count;
}

uint32_t dungeon_lock_layout_pick(float lateral, float height, DungeonLockBox *out,
								  uint32_t capacity)
{
	if (!out || capacity < DUNGEON_LOCK_PICK_PARTS)
		return 0;
	float proud = DUNGEON_LOCK_PROUD_PICK_M;
	/* Tip stub, rising to the underside of the pin being worked. */
	out[0] = box_from(lateral, 0.0085f, height, height + DUNGEON_LOCK_PICK_STUB_M, proud, 0.0085f);
	/* The bend, and then the shaft running off the left of the frame. Both sit
	 * below the tip, which is what makes the pick look like it is lifting the
	 * pin rather than hanging off it. */
	out[1] = (DungeonLockBox){{lateral - 0.100f, height - 0.012f, proud - 0.0075f},
							  {lateral - 0.005f, height + 0.006f, proud + 0.0075f}};
	out[2] = (DungeonLockBox){{lateral - 0.950f, height - 0.030f, proud - 0.0080f},
							  {lateral - 0.095f, height - 0.014f, proud + 0.0080f}};
	return DUNGEON_LOCK_PICK_PARTS;
}

void dungeon_lock_layout_pick_target(const DungeonPinTumbler *pins, uint32_t pin,
									 float *out_lateral, float *out_height)
{
	if (!pins || pin >= pins->pin_count)
		return;
	if (out_lateral)
		*out_lateral = dungeon_lock_layout_bore_lateral(pins, pin);
	if (out_height)
		*out_height = pin_base_height(pins, pin) - DUNGEON_LOCK_PICK_STUB_M;
}

bool dungeon_lock_layout_pick_clear(const DungeonPinTumbler *pins, float lateral, float height)
{
	DungeonLockBox parts[DUNGEON_LOCK_MAX_PARTS];
	DungeonLockBox pick[DUNGEON_LOCK_PICK_PARTS];
	uint32_t part_count = dungeon_lock_layout_parts(pins, parts, DUNGEON_LOCK_MAX_PARTS);
	uint32_t pick_count = dungeon_lock_layout_pick(lateral, height, pick, DUNGEON_LOCK_PICK_PARTS);
	for (uint32_t p = 0; p < pick_count; ++p)
		for (uint32_t q = 0; q < part_count; ++q)
			if (dungeon_lock_box_overlap(&pick[p], &parts[q]))
				return false;
	return true;
}
