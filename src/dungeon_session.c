#include "dungeon_session.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Level blockers plus one entry per still-shut door. Both arrays are rebuilt
 * wholesale rather than patched, because the player's swept-circle solver and
 * the shader's blocker array both want one flat contiguous run and neither
 * cares that a few entries came and went. */
static bool rebuild_blockers(DungeonSession *session)
{
	const DungeonLevel *level = session->level;
	uint32_t shut = 0;
	for (uint32_t i = 0; i < session->door_count; ++i)
		shut += session->doors[i].open ? 0u : 1u;

	uint32_t collider_total = level->collider_count + shut;
	uint32_t occluder_total = level->occluder_count + shut;
	DungeonCollider *colliders = malloc((collider_total ? collider_total : 1u) * sizeof(*colliders));
	DungeonSegment *occluders = malloc((occluder_total ? occluder_total : 1u) * sizeof(*occluders));
	if (!colliders || !occluders)
	{
		free(colliders);
		free(occluders);
		return false;
	}
	memcpy(colliders, level->colliders, level->collider_count * sizeof(*colliders));
	memcpy(occluders, level->occluders, level->occluder_count * sizeof(*occluders));
	uint32_t write = level->collider_count, occlude_write = level->occluder_count;
	for (uint32_t i = 0; i < session->door_count; ++i)
	{
		if (session->doors[i].open)
			continue;
		/* A shut door is a SEGMENT, never a rectangle: dungeon_light_segment_blocked
		 * and segment_crosses_blocker in mesh.frag both read blockers that way,
		 * and a mismatch there has already been one real bug in this scene. */
		colliders[write++] = (DungeonCollider){.type = DUNGEON_COLLIDER_SEGMENT,
											   .segment = level->doors[i].blocker};
		occluders[occlude_write++] = level->doors[i].blocker;
	}
	free(session->colliders);
	free(session->occluders);
	session->colliders = colliders;
	session->collider_count = write;
	session->occluders = occluders;
	session->occluder_count = occlude_write;
	return true;
}

bool dungeon_session_create(DungeonSession *session, const DungeonLevel *level)
{
	if (!session || !level)
		return false;
	*session = (DungeonSession){0};
	session->level = level;
	session->door_count =
		level->door_count < DUNGEON_MAX_DOORS ? level->door_count : DUNGEON_MAX_DOORS;
	session->phase = DUNGEON_PHASE_EXPLORING;
	session->status = DUNGEON_STATUS_EXPLORING;
	for (uint32_t i = 0; i < session->door_count; ++i)
	{
		const DungeonDoorway *door = &level->doors[i];
		DungeonDoorState *state = &session->doors[i];
		/* Pin counts and combination lengths vary a little per door so the
		 * second lock is not simply the first one again. */
		dungeon_pin_tumbler_init(&state->pins, door->seed, 3u + (door->seed & 1u));
		dungeon_vault_dial_init(&state->dial, door->seed, 4u + ((door->seed >> 3) & 1u));
		/* Put the lock hardware on the side the player approaches from, so the
		 * first door met is never locked from behind. */
		float normal_x = cosf(door->yaw), normal_z = sinf(door->yaw);
		float to_spawn = (level->spawn.x - door->center.x) * normal_x +
						 (level->spawn.z - door->center.z) * normal_z;
		state->hardware_side = to_spawn < 0.0f ? -1.0f : 1.0f;
	}
	if (!rebuild_blockers(session))
	{
		dungeon_session_destroy(session);
		return false;
	}
	return true;
}

void dungeon_session_destroy(DungeonSession *session)
{
	if (!session)
		return;
	free(session->colliders);
	free(session->occluders);
	*session = (DungeonSession){0};
}

uint32_t dungeon_session_nearest_door(const DungeonSession *session, DungeonPoint player)
{
	if (!session)
		return UINT32_MAX;
	uint32_t nearest = UINT32_MAX;
	float best = DUNGEON_INTERACTION_RANGE_M * DUNGEON_INTERACTION_RANGE_M;
	for (uint32_t i = 0; i < session->door_count; ++i)
	{
		if (session->doors[i].open)
			continue;
		float dx = player.x - session->level->doors[i].center.x;
		float dz = player.z - session->level->doors[i].center.z;
		float distance2 = dx * dx + dz * dz;
		if (distance2 <= best)
		{
			best = distance2;
			nearest = i;
		}
	}
	return nearest;
}

bool dungeon_session_interact(DungeonSession *session, DungeonPoint player)
{
	if (!session || session->phase != DUNGEON_PHASE_EXPLORING)
		return false;
	uint32_t door = dungeon_session_nearest_door(session, player);
	if (door == UINT32_MAX)
	{
		session->status = DUNGEON_STATUS_NOTHING_IN_REACH;
		return false;
	}
	session->focused_door = door;
	if (session->level->doors[door].lock == DUNGEON_LOCK_VAULT_DIAL)
	{
		session->phase = DUNGEON_PHASE_VAULT_DIAL;
		session->status = DUNGEON_STATUS_ENTER_THE_SEQUENCE;
	}
	else
	{
		session->phase = DUNGEON_PHASE_PIN_TUMBLER;
		session->status = DUNGEON_STATUS_MATCH_THE_PINS;
	}
	return true;
}

void dungeon_session_cancel(DungeonSession *session)
{
	if (!session)
		return;
	session->phase = DUNGEON_PHASE_EXPLORING;
	session->status = DUNGEON_STATUS_EXPLORING;
}

/* The port of UnlockFirstDoor: the door opens, its blocker leaves both arrays,
 * and the session drops straight back to exploring. */
static void unlock_focused_door(DungeonSession *session)
{
	session->doors[session->focused_door].open = true;
	rebuild_blockers(session);
	session->phase = DUNGEON_PHASE_EXPLORING;
	session->status = DUNGEON_STATUS_LOCK_OPEN;
}

void dungeon_session_move_selection(DungeonSession *session, int direction)
{
	if (!session || session->phase != DUNGEON_PHASE_PIN_TUMBLER)
		return;
	dungeon_pin_tumbler_move(&session->doors[session->focused_door].pins, direction);
}

void dungeon_session_adjust(DungeonSession *session, int direction)
{
	if (!session || session->phase != DUNGEON_PHASE_PIN_TUMBLER)
		return;
	dungeon_pin_tumbler_adjust(&session->doors[session->focused_door].pins, direction);
}

bool dungeon_session_spin_dial(DungeonSession *session, float direction, float dt)
{
	if (!session || session->phase != DUNGEON_PHASE_VAULT_DIAL || dt <= 0.0f)
		return false;
	DungeonDoorState *state = &session->doors[session->focused_door];
	if (direction == 0.0f)
	{
		/* Let go and the next tick starts fresh, so tapping the key does not
		 * bank fractions of a tick between presses. */
		state->dial_spin_accumulator = 0.0f;
		return dungeon_vault_dial_on_number(&state->dial);
	}
	int step = direction > 0.0f ? 1 : -1;
	state->dial_spin_accumulator += dt;
	while (state->dial_spin_accumulator >= DUNGEON_DIAL_TICK_SECONDS)
	{
		state->dial_spin_accumulator -= DUNGEON_DIAL_TICK_SECONDS;
		dungeon_vault_dial_step(&state->dial, step);
	}
	return dungeon_vault_dial_on_number(&state->dial);
}

bool dungeon_session_commit_dial(DungeonSession *session)
{
	if (!session || session->phase != DUNGEON_PHASE_VAULT_DIAL)
		return false;
	DungeonDoorState *state = &session->doors[session->focused_door];
	bool banked = dungeon_vault_dial_commit(&state->dial);
	state->dial_shake = 1.0f;
	if (!banked)
	{
		session->status = DUNGEON_STATUS_DIAL_RESET;
		return false;
	}
	session->status = DUNGEON_STATUS_SEQUENCE_PROGRESS;
	if (state->dial.solved)
		unlock_focused_door(session);
	return true;
}

bool dungeon_session_confirm(DungeonSession *session)
{
	if (!session || session->phase != DUNGEON_PHASE_PIN_TUMBLER)
		return false;
	if (!dungeon_pin_tumbler_submit(&session->doors[session->focused_door].pins))
	{
		session->status = DUNGEON_STATUS_PINS_DO_NOT_ALIGN;
		return false;
	}
	unlock_focused_door(session);
	return true;
}

void dungeon_session_update(DungeonSession *session, float dt)
{
	if (!session || dt <= 0.0f)
		return;
	/* Exponential approach, frame-rate independent, matching how the dungeon
	 * camera damps its follow. */
	float blend = 1.0f - expf(-6.0f * dt);
	float shake_decay = 1.0f - expf(-7.0f * dt);
	for (uint32_t i = 0; i < session->door_count; ++i)
	{
		float target = session->doors[i].open ? 1.0f : 0.0f;
		session->doors[i].swing += (target - session->doors[i].swing) * blend;
		session->doors[i].dial_shake -= session->doors[i].dial_shake * shake_decay;
		if (session->doors[i].dial_shake < 1e-3f)
			session->doors[i].dial_shake = 0.0f;
		/* Free-running, so the index pin's flash has a phase without the
		 * renderer having to keep one. */
		session->doors[i].dial_flash_time += dt;
	}
}

void dungeon_session_status_text(const DungeonSession *session, char *out, size_t capacity)
{
	if (!out || !capacity)
		return;
	if (!session)
	{
		snprintf(out, capacity, "Dungeon Explorer");
		return;
	}
	const DungeonDoorState *state =
		session->focused_door < session->door_count ? &session->doors[session->focused_door] : NULL;
	switch (session->phase)
	{
	case DUNGEON_PHASE_PIN_TUMBLER:
	{
		char heights[DUNGEON_LOCK_MAX_PINS + 1u] = {0};
		for (uint32_t pin = 0; state && pin < state->pins.pin_count; ++pin)
			heights[pin] = (char)('0' + state->pins.heights[pin]);
		snprintf(out, capacity, "Dungeon | Lockpick | PIN TUMBLER | Pin:%u | Values:%s | %s",
				 state ? state->pins.selected + 1u : 0u, heights,
				 session->status == DUNGEON_STATUS_PINS_DO_NOT_ALIGN
					 ? "THE PINS DO NOT ALIGN"
					 : "RAISE EACH PIN UNTIL IT SETS AND LOCKS");
		return;
	}
	case DUNGEON_PHASE_VAULT_DIAL:
	{
		bool flashing = state && dungeon_vault_dial_on_number(&state->dial);
		snprintf(out, capacity, "Dungeon | Lockpick | SAFE DIAL | Number:%u/%u | Tick:%u | %s",
				 state ? state->dial.progress : 0u, state ? state->dial.step_count : 0u,
				 state ? state->dial.position : 0u,
				 session->status == DUNGEON_STATUS_DIAL_RESET
					 ? "WRONG NUMBER  COMBINATION RESET"
					 : (flashing ? "PIN LIT  PRESS ENTER" : "TURN AND WATCH THE PIN"));
		return;
	}
	case DUNGEON_PHASE_EXPLORING:
	default:
		break;
	}
	uint32_t shut = 0;
	for (uint32_t i = 0; i < session->door_count; ++i)
		shut += session->doors[i].open ? 0u : 1u;
	const char *note = session->status == DUNGEON_STATUS_LOCK_OPEN	  ? "LOCK OPEN"
					   : session->status == DUNGEON_STATUS_NOTHING_IN_REACH ? "NOTHING IN REACH"
																	 : "FIND THE LOCKED DOORS";
	snprintf(out, capacity, "Dungeon | Explore | Locked:%u/%u | %s", shut, session->door_count,
			 note);
}

float dungeon_session_focus_facing_degrees(const DungeonSession *session)
{
	if (!session || session->focused_door >= session->door_count)
		return 90.0f;
	const DungeonDoorway *door = &session->level->doors[session->focused_door];
	float side = session->doors[session->focused_door].hardware_side;
	/* dungeon_camera places itself at target - (cos yaw, ., sin yaw) * trail,
	 * so it looks ALONG (cos yaw, sin yaw). To stand on the hardware side and
	 * look at the door, that direction is the inward normal. */
	float look_x = -cosf(door->yaw) * side, look_z = -sinf(door->yaw) * side;
	float degrees = atan2f(look_z, look_x) * (180.0f / 3.14159265358979f);
	return degrees < 0.0f ? degrees + 360.0f : degrees;
}

DungeonPoint dungeon_session_lock_origin(const DungeonSession *session, uint32_t door_index)
{
	if (!session || door_index >= session->door_count)
		return (DungeonPoint){0.0f, 0.0f};
	const DungeonDoorway *door = &session->level->doors[door_index];
	float side = session->doors[door_index].hardware_side;
	DungeonPoint normal = {cosf(door->yaw) * side, sinf(door->yaw) * side};
	return (DungeonPoint){door->center.x + normal.x * DUNGEON_LOCK_STANDOFF_M,
						  door->center.z + normal.z * DUNGEON_LOCK_STANDOFF_M};
}

DungeonPoint dungeon_session_pick_stance(const DungeonSession *session)
{
	if (!session || session->focused_door >= session->door_count)
		return (DungeonPoint){0.0f, 0.0f};
	const DungeonDoorway *door = &session->level->doors[session->focused_door];
	float side = session->doors[session->focused_door].hardware_side;
	DungeonPoint normal = {cosf(door->yaw) * side, sinf(door->yaw) * side};
	DungeonPoint across = {-normal.z, normal.x};
	/* Off the corridor centreline, so the player is beside the lock rather
	 * than between it and the camera. */
	return (DungeonPoint){door->center.x + normal.x * DUNGEON_PICK_STANDOFF_M -
							  across.x * DUNGEON_PICK_LATERAL_M,
						  door->center.z + normal.z * DUNGEON_PICK_STANDOFF_M -
							  across.z * DUNGEON_PICK_LATERAL_M};
}

DungeonPoint dungeon_session_focus_point(const DungeonSession *session)
{
	if (!session || session->focused_door >= session->door_count)
		return (DungeonPoint){0.0f, 0.0f};
	return dungeon_session_lock_origin(session, session->focused_door);
}
