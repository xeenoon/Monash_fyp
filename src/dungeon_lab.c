#include "dungeon_lab.h"

#include <stdio.h>

/* Corner spacing of the lab's field. Fine enough that marching squares puts
 * the wall exactly where the rect says, coarse enough that the room is a few
 * hundred corners rather than a few thousand. */
#define LAB_CELL_M 0.25f
/* Solid rock kept outside the room on every side, so the walls have thickness
 * to be seen from and the contour never runs off the edge of the field. */
#define LAB_MARGIN_M 1.5f

static float lab_extent_m(void)
{
	return DUNGEON_LAB_ROOM_M + 2.0f * LAB_MARGIN_M;
}

DungeonPoint dungeon_lab_torch_position(DungeonLabScene scene)
{
	if (scene == DUNGEON_LAB_STANDING)
		return (DungeonPoint){DUNGEON_LAB_ROOM_M * 0.5f, DUNGEON_LAB_ROOM_M * 0.5f};
	/* A third of a metre off the north wall's midpoint. mount_torch only uses
	 * this to pick which wall segment is nearest and which way is inward; it
	 * then traces the rendered wall and reseats the fixture against it. */
	return (DungeonPoint){DUNGEON_LAB_ROOM_M * 0.5f, 0.34f};
}

bool dungeon_lab_compile(DungeonLevel *out, DungeonLevelError *error)
{
	if (!out)
		return false;
	float extent = lab_extent_m();
	uint32_t corners = (uint32_t)(extent / LAB_CELL_M) + 1u;
	DungeonField field;
	if (!dungeon_field_create(corners, corners, LAB_CELL_M,
							  (DungeonPoint){-LAB_MARGIN_M, -LAB_MARGIN_M}, &field))
	{
		if (error)
			snprintf(error->message, sizeof(error->message), "out of memory building torch lab");
		return false;
	}
	/* A hard step, not a feathered one: the 0.5 isoline then lands midway
	 * between the last solid and first open corner and marching squares
	 * reconstructs four straight walls and four right angles. Feathering here
	 * would round the room into a cave pocket, which is the one thing this
	 * scene must not do -- a curved wall makes it impossible to tell whether
	 * the torch is seated against the wall or floating off it. */
	dungeon_field_stamp_rect(
		&field,
		(DungeonRect){{0.0f, 0.0f}, {DUNGEON_LAB_ROOM_M, DUNGEON_LAB_ROOM_M}}, 0.0f);

	/* One puddle, off-centre and clear of the walls. It sits between the north
	 * wall and the middle of the room, so it is lit in every scene: the puddle
	 * shader's fake reflection reads the real point lights, and a puddle in
	 * the dark proves nothing. */
	DungeonPuddle puddle = {.center = {DUNGEON_LAB_ROOM_M * 0.5f, DUNGEON_LAB_ROOM_M * 0.30f},
							.radius = 1.1f};

	/* The carried-torch scene puts the player at the spawn, so keep it clear of
	 * both the puddle and the pole in the middle of the room. The exit is
	 * tucked in the far corner, where its marker stays out of the way of
	 * whatever is being looked at. */
	DungeonPoint spawn = {DUNGEON_LAB_ROOM_M * 0.5f, DUNGEON_LAB_ROOM_M * 0.72f};
	DungeonPoint exit_point = {DUNGEON_LAB_ROOM_M - 1.0f, DUNGEON_LAB_ROOM_M - 1.0f};
	/* Takes ownership of `field` either way, and copies the puddle. No doors:
	 * a lock has nothing to say about a flame. */
	return dungeon_level_compile_field(&field, spawn, exit_point, &puddle, 1u, NULL, 0u, 0.0f,
									   2.4f, out, error);
}
