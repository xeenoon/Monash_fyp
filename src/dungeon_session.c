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
		/* Pin counts vary a little per door so the second lock is not simply
		 * the first one again; the safe always has four pins, and takes its
		 * variety from the order they have to be pressed in. */
		dungeon_pin_tumbler_init(&state->pins, door->seed, 3u + (door->seed & 1u));
		dungeon_safe_pins_init(&state->safe, door->seed);
		dungeon_prism_init(&state->prism, door->seed);
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
	if (session->level->doors[door].lock == DUNGEON_LOCK_PRISM)
	{
		session->phase = DUNGEON_PHASE_PRISM;
		session->doors[door].prism.rotating = false;
	}
	else if (session->level->doors[door].lock == DUNGEON_LOCK_SAFE_PINS)
	{
		session->phase = DUNGEON_PHASE_SAFE_PINS;
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

void dungeon_session_prism_direction(DungeonSession *session, DungeonPrismDirection direction)
{
	if (!session || session->phase != DUNGEON_PHASE_PRISM)
		return;
	DungeonPrismPuzzle *p = &session->doors[session->focused_door].prism;
	if (!p->rotating)
		dungeon_prism_move(p, direction);
	else if (direction == DUNGEON_PRISM_EAST)
		dungeon_prism_turn(p, 1);
	else if (direction == DUNGEON_PRISM_WEST)
		dungeon_prism_turn(p, -1);
}

void dungeon_session_prism_confirm(DungeonSession *session)
{
	if (session && session->phase == DUNGEON_PHASE_PRISM)
		dungeon_prism_confirm(&session->doors[session->focused_door].prism);
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

void dungeon_session_move_safe_pin(DungeonSession *session, int direction)
{
	if (!session || session->phase != DUNGEON_PHASE_SAFE_PINS)
		return;
	dungeon_safe_pins_move(&session->doors[session->focused_door].safe, direction);
}

bool dungeon_session_press_safe_pin(DungeonSession *session)
{
	if (!session || session->phase != DUNGEON_PHASE_SAFE_PINS)
		return false;
	DungeonDoorState *state = &session->doors[session->focused_door];
	bool driven = dungeon_safe_pins_press(&state->safe);
	state->safe_shake = 1.0f;
	if (!driven)
	{
		session->status = DUNGEON_STATUS_SAFE_RESET;
		return false;
	}
	session->status = DUNGEON_STATUS_SEQUENCE_PROGRESS;
	if (state->safe.solved)
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
	/* Much faster than the door swing: a safe pin driving forward is a
	 * mechanism snapping into place, not a leaf sinking into the floor. */
	float push_blend = 1.0f - expf(-18.0f * dt);
	for (uint32_t i = 0; i < session->door_count; ++i)
	{
		DungeonDoorState *state = &session->doors[i];
		dungeon_prism_update(&state->prism, dt);
		if (state->prism.solved && !state->open)
			state->prism_solved_time += dt;
		if (state->prism.solved && !state->open && state->prism_solved_time >= .45f)
		{
			uint32_t previous_focus = session->focused_door;
			DungeonPhase previous_phase = session->phase;
			DungeonStatus previous_status = session->status;
			session->focused_door = i;
			unlock_focused_door(session);
			if (previous_focus != i)
			{
				session->focused_door = previous_focus;
				session->phase = previous_phase;
				session->status = previous_status;
			}
		}
		float target = state->open ? 1.0f : 0.0f;
		state->swing += (target - state->swing) * blend;
		state->safe_shake -= state->safe_shake * shake_decay;
		if (state->safe_shake < 1e-3f)
			state->safe_shake = 0.0f;
		/* The whole animation, derived rather than remembered: a driven pin
		 * travels all the way out, the one under the selection creeps out a
		 * fraction of that, and everything else falls back flush. A reset
		 * therefore springs the face back out with no extra state to clear. */
		bool picking = session->phase == DUNGEON_PHASE_SAFE_PINS && session->focused_door == i;
		for (uint32_t pin = 0; pin < state->safe.pin_count; ++pin)
		{
			float rest = picking && state->safe.selected == pin ? DUNGEON_SAFE_PIN_HOVER : 0.0f;
			float want = dungeon_safe_pins_driven(&state->safe, pin) ? 1.0f : rest;
			state->safe_push[pin] += (want - state->safe_push[pin]) * push_blend;
		}
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
	case DUNGEON_PHASE_PRISM:
	{
		const DungeonPrism *optic = state ? &state->prism.prisms[state->prism.selected] : NULL;
		const char *kind = !optic || optic->kind == DUNGEON_OPTIC_PRISM ? "PRISM"
						   : optic->kind == DUNGEON_OPTIC_CONVEX		? "CONVEX LENS (FOCUS)"
																		: "CONCAVE LENS (SPREAD)";
		snprintf(out, capacity,
				 "Dungeon | OPTICS | %s %u/%u | %u deg | LIGHT %.0f%% | %s | Q/E: leave", kind,
				 state ? state->prism.selected + 1 : 0, state ? state->prism.count : 0,
				 optic ? optic->orientation : 0,
				 state ? 100.0 * fminf(state->prism.key_power / DUNGEON_OPTIC_KEY_POWER, 1.f) : 0.0,
				 state && state->prism.rotating ? "LEFT/RIGHT: turn (hold to rotate) ENTER: release"
												: "ARROWS: choose ENTER: select LIGHT THE KEY");
		return;
	}
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
	case DUNGEON_PHASE_SAFE_PINS:
	{
		snprintf(out, capacity, "Dungeon | Lockpick | SAFE | Driven:%u/%u | Pin:%u | %s",
				 state ? state->safe.progress : 0u, state ? state->safe.pin_count : 0u,
				 state ? state->safe.selected + 1u : 0u,
				 session->status == DUNGEON_STATUS_SAFE_RESET
					 ? "WRONG PIN  THE FACE RESET"
					 : "PRESS THE PINS IN THE RIGHT ORDER");
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
