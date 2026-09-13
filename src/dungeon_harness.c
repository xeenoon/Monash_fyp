#include "dungeon_harness.h"

#include "dungeon_lock_layout.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HARNESS_TIMESTEP (1.0f / 60.0f)
#define HARNESS_MAX_LINES 512u
#define HARNESS_LINE_CAPACITY 192u

struct DungeonHarness
{
	char lines[HARNESS_MAX_LINES][HARNESS_LINE_CAPACITY];
	uint32_t line_count;
	uint32_t cursor;

	uint32_t frames_remaining; /* frames the current command still consumes */
	float walk_forward, walk_right;
	/* walk_to steers at a world point instead of a fixed heading, because the
	   camera yaw is still easing back from the lock while the player leaves --
	   a constant "forward" would walk them somewhere else entirely. */
	bool walk_seeking;
	DungeonPoint walk_target;
	char pending_capture[256];
	uint32_t capture_delay; /* frames to settle before the capture is taken */

	uint32_t checks_run, checks_failed;
	/* sweep_pins state: driving the selection back and forth while watching the
	 * pick's clearance frame by frame. */
	bool driving_prisms;
	uint32_t prism_frames;
	bool sweeping;
	uint32_t sweep_frames, sweep_violations;
	int sweep_direction;
	bool finished;
};

static void trim(char *text)
{
	char *hash = strchr(text, '#');
	if (hash)
		*hash = '\0';
	size_t length = strlen(text);
	while (length && (text[length - 1] == '\n' || text[length - 1] == '\r' ||
					  text[length - 1] == ' ' || text[length - 1] == '\t'))
		text[--length] = '\0';
}

DungeonHarness *dungeon_harness_create(bool *out_script_error)
{
	if (out_script_error)
		*out_script_error = false;
	const char *path = getenv("DUNGEON_SCRIPT");
	if (!path || !*path)
		return NULL;
	FILE *file = fopen(path, "r");
	if (!file)
	{
		fprintf(stderr, "DUNGEON_SCRIPT: could not open %s\n", path);
		if (out_script_error)
			*out_script_error = true;
		return NULL;
	}
	DungeonHarness *harness = calloc(1, sizeof(*harness));
	if (!harness)
	{
		fclose(file);
		if (out_script_error)
			*out_script_error = true;
		return NULL;
	}
	char line[HARNESS_LINE_CAPACITY];
	while (harness->line_count < HARNESS_MAX_LINES && fgets(line, sizeof(line), file))
	{
		trim(line);
		const char *start = line;
		while (*start == ' ' || *start == '\t')
			++start;
		if (!*start)
			continue;
		snprintf(harness->lines[harness->line_count++], HARNESS_LINE_CAPACITY, "%s", start);
	}
	fclose(file);
	fprintf(stdout, "Harness: %s, %u commands\n", path, harness->line_count);
	return harness;
}

void dungeon_harness_destroy(DungeonHarness *harness) { free(harness); }

float dungeon_harness_timestep(const DungeonHarness *harness)
{
	(void)harness;
	return HARNESS_TIMESTEP;
}

int dungeon_harness_exit_code(const DungeonHarness *harness)
{
	if (!harness)
		return 0;
	return harness->checks_failed ? 1 : 0;
}

static void check(DungeonHarness *harness, bool passed, const char *format, ...)
{
	char detail[192];
	va_list args;
	va_start(args, format);
	vsnprintf(detail, sizeof(detail), format, args);
	va_end(args);
	++harness->checks_run;
	if (!passed)
		++harness->checks_failed;
	fprintf(stdout, "%s %s\n", passed ? "  PASS" : "  FAIL", detail);
}

/* Where the player should stand to be in reach of a door, on the side its lock
 * hardware faces (or deliberately beyond it, to check they got through). */
static DungeonPoint door_approach(const DungeonScene *scene, uint32_t door_index, bool far_side)
{
	const DungeonDoorway *door = &scene->level.doors[door_index];
	float side = scene->session.doors[door_index].hardware_side * (far_side ? -1.0f : 1.0f);
	float normal_x = cosf(door->yaw) * side, normal_z = sinf(door->yaw) * side;
	return (DungeonPoint){door->center.x + normal_x * 1.10f, door->center.z + normal_z * 1.10f};
}

/* Signed side of the door plane the player is on, positive on the approach
 * side. Crossing to negative is what "walked through the gate" means. */
static float side_of_door(const DungeonScene *scene, uint32_t door_index)
{
	const DungeonDoorway *door = &scene->level.doors[door_index];
	DungeonPoint player = scene->player.position;
	float along = (player.x - door->center.x) * cosf(door->yaw) +
				  (player.z - door->center.z) * sinf(door->yaw);
	return along * scene->session.doors[door_index].hardware_side;
}

static const char *phase_name(DungeonPhase phase)
{
	switch (phase)
	{
	case DUNGEON_PHASE_PIN_TUMBLER: return "pin";
	case DUNGEON_PHASE_SAFE_PINS: return "safe";
	case DUNGEON_PHASE_PRISM:
		return "prism";
	default: return "exploring";
	}
}

/* Fills the focused lock with its own solution. The input path is covered by
 * `press` commands and by the headless lock tests; this is for the frames that
 * are about the door opening, not about entering a combination. */
static void solve_focused(DungeonHarness *harness, DungeonScene *scene)
{
	DungeonSession *session = &scene->session;
	if (session->phase == DUNGEON_PHASE_PIN_TUMBLER)
	{
		DungeonPinTumbler *pins = &session->doors[session->focused_door].pins;
		for (uint32_t pin = 0; pin < pins->pin_count; ++pin)
			pins->heights[pin] = pins->target[pin];
		check(harness, dungeon_session_confirm(session), "solve: pin tumbler accepted");
	}
	else if (session->phase == DUNGEON_PHASE_SAFE_PINS)
	{
		/* Walk the selection along the row and press whenever it is standing on
		 * the pin the order wants next -- through the same session calls the
		 * keys drive. Writing the progress counter would skip the very path
		 * this is meant to prove. A full lap of the row per pin is the worst
		 * case, so the guard is generous. */
		uint32_t guard = DUNGEON_SAFE_PIN_COUNT * DUNGEON_SAFE_PIN_COUNT * 4u;
		while (session->phase == DUNGEON_PHASE_SAFE_PINS && guard--)
		{
			if (dungeon_safe_pins_selected_is_next(&session->doors[session->focused_door].safe))
				dungeon_session_press_safe_pin(session);
			else
				dungeon_session_move_safe_pin(session, 1);
		}
		check(harness, session->phase == DUNGEON_PHASE_EXPLORING, "solve: safe accepted");
	}
	else
		check(harness, false, "solve: no lock is being picked");
}

static void run_expect(DungeonHarness *harness, DungeonScene *scene, const DungeonCamera *camera,
					   const char *arguments)
{
	char what[32] = {0};
	char rest[96] = {0};
	int parsed = sscanf(arguments, "%31s %95[^\n]", what, rest);
	if (parsed < 1)
	{
		check(harness, false, "expect: missing predicate");
		return;
	}
	const DungeonSession *session = &scene->session;
	if (!strcmp(what, "prism_selected") || !strcmp(what, "prism_rotating") ||
		!strcmp(what, "prism_solved"))
	{
		bool active = session->phase == DUNGEON_PHASE_PRISM;
		const DungeonPrismPuzzle *p = &session->doors[session->focused_door].prism;
		bool ok = !strcmp(what, "prism_selected")	? p->selected == (uint32_t)atoi(rest)
				  : !strcmp(what, "prism_rotating") ? p->rotating == (atoi(rest) != 0)
													: p->solved;
		check(harness, active && ok, "%s %s", what, rest);
		return;
	}
	if (!strcmp(what, "phase"))
	{
		const char *actual = phase_name(session->phase);
		check(harness, !strcmp(actual, rest), "expect phase %s (actual %s)", rest, actual);
	}
	else if (!strcmp(what, "facing"))
	{
		float tolerance = (float)atof(rest);
		float wanted = dungeon_session_focus_facing_degrees(session);
		float delta = fabsf(fmodf(camera->camera.yaw - wanted + 540.0f, 360.0f) - 180.0f);
		check(harness, delta <= tolerance, "expect facing +/-%.1f deg (yaw %.1f, want %.1f, off by %.2f)",
			  (double)tolerance, (double)camera->camera.yaw, (double)wanted, (double)delta);
	}
	else if (!strcmp(what, "aimed"))
	{
		/* Angle between where the camera looks and where the lock actually is.
		 * Independent of window size, unlike judging centring from a capture --
		 * the window manager here hands out whatever aspect it likes. */
		float tolerance = (float)atof(rest);
		DungeonPoint lock = dungeon_session_focus_point(session);
		float yaw = camera->camera.yaw * (3.14159265358979f / 180.0f);
		float pitch = camera->camera.pitch * (3.14159265358979f / 180.0f);
		float forward[3] = {cosf(yaw) * cosf(pitch), sinf(pitch), sinf(yaw) * cosf(pitch)};
		float to_lock[3] = {(float)(lock.x - camera->camera.position.x),
							(float)(DUNGEON_LOCK_CENTRE_Y_M - camera->camera.position.y),
							(float)(lock.z - camera->camera.position.z)};
		float length = sqrtf(to_lock[0] * to_lock[0] + to_lock[1] * to_lock[1] +
							 to_lock[2] * to_lock[2]);
		float degrees = 180.0f;
		if (length > 1e-4f)
		{
			float dot = (forward[0] * to_lock[0] + forward[1] * to_lock[1] +
						 forward[2] * to_lock[2]) /
						length;
			dot = dot > 1.0f ? 1.0f : (dot < -1.0f ? -1.0f : dot);
			degrees = acosf(dot) * (180.0f / 3.14159265358979f);
		}
		check(harness, degrees <= tolerance, "expect aimed +/-%.1f deg (off by %.2f, range %.2f m)",
			  (double)tolerance, (double)degrees, (double)length);
	}
	else if (!strcmp(what, "all_pins_free"))
	{
		bool picking = session->phase == DUNGEON_PHASE_PIN_TUMBLER;
		const DungeonPinTumbler *pins =
			picking ? &session->doors[session->focused_door].pins : NULL;
		uint32_t set_count = 0;
		for (uint32_t pin = 0; pins && pin < pins->pin_count; ++pin)
			set_count += dungeon_pin_tumbler_pin_set(pins, pin) ? 1u : 0u;
		check(harness, picking && set_count == 0u, "expect all_pins_free (%u of %u already set)",
			  set_count, pins ? pins->pin_count : 0u);
	}
	else if (!strcmp(what, "pin_locked") || !strcmp(what, "pin_free"))
	{
		uint32_t index = (uint32_t)atoi(rest);
		bool want_locked = !strcmp(what, "pin_locked");
		bool picking = session->phase == DUNGEON_PHASE_PIN_TUMBLER;
		const DungeonPinTumbler *pins =
			picking ? &session->doors[session->focused_door].pins : NULL;
		bool valid = pins && index < pins->pin_count;
		bool locked = valid && dungeon_pin_tumbler_pin_set(pins, index);
		check(harness, valid && locked == want_locked, "expect %s %u (actual %s)", what, index,
			  !valid ? "no such pin" : (locked ? "locked" : "free"));
	}
	else if (!strcmp(what, "pick_clear"))
	{
		bool picking = session->phase == DUNGEON_PHASE_PIN_TUMBLER;
		const DungeonPinTumbler *pins =
			picking ? &session->doors[session->focused_door].pins : NULL;
		bool clear = picking && dungeon_lock_layout_pick_clear(pins, scene->pick_lateral,
															   scene->pick_height);
		check(harness, clear, "expect pick_clear (tip at lateral %.3f, height %.3f)",
			  (double)scene->pick_lateral, (double)scene->pick_height);
	}
	else if (!strcmp(what, "pins_left_to_right"))
	{
		/* Pin 0 must land left of the last pin on screen, or the arrow keys
		 * walk the selection the wrong way. Projects the pin positions onto the
		 * camera's right vector rather than trusting the layout constants --
		 * this is exactly the axis that was inverted. */
		float yaw = camera->camera.yaw * (3.14159265358979f / 180.0f);
		DungeonPoint forward = {cosf(yaw), sinf(yaw)};
		DungeonPoint right = {-forward.z, forward.x};
		uint32_t door = session->focused_door;
		bool picking = session->phase == DUNGEON_PHASE_PIN_TUMBLER;
		uint32_t last = picking ? session->doors[door].pins.pin_count - 1u : 0u;
		WorldPosition first_pin = {0}, last_pin = {0};
		bool ok = picking && dungeon_scene_pin_world(scene, door, 0u, &first_pin) &&
				  dungeon_scene_pin_world(scene, door, last, &last_pin);
		float first_on_right =
			ok ? (float)first_pin.x * right.x + (float)first_pin.z * right.z : 0.0f;
		float last_on_right =
			ok ? (float)last_pin.x * right.x + (float)last_pin.z * right.z : 0.0f;
		check(harness, ok && first_on_right < last_on_right,
			  "expect pins_left_to_right (pin 0 at %.3f, pin %u at %.3f along camera right)",
			  (double)first_on_right, last, (double)last_on_right);
	}
	else if (!strcmp(what, "focus"))
	{
		float minimum = (float)atof(rest);
		float actual = dungeon_camera_focus_blend(camera);
		check(harness, actual >= minimum, "expect focus >= %.2f (actual %.3f)", (double)minimum,
			  (double)actual);
	}
	else if (!strcmp(what, "door_open") || !strcmp(what, "door_shut"))
	{
		uint32_t index = (uint32_t)atoi(rest);
		bool want_open = !strcmp(what, "door_open");
		bool valid = index < session->door_count;
		bool open = valid && session->doors[index].open;
		check(harness, valid && open == want_open, "expect %s %u (actual %s)", what, index,
			  !valid ? "no such door" : (open ? "open" : "shut"));
	}
	else if (!strcmp(what, "past_door"))
	{
		uint32_t index = (uint32_t)atoi(rest);
		bool valid = index < session->door_count;
		float side = valid ? side_of_door(scene, index) : 1.0f;
		check(harness, valid && side < 0.0f, "expect past_door %u (signed side %.2f)", index,
			  (double)side);
	}
	else if (!strcmp(what, "pin_selected"))
	{
		uint32_t wanted = (uint32_t)atoi(rest);
		bool picking = session->phase == DUNGEON_PHASE_PIN_TUMBLER;
		uint32_t actual = picking ? session->doors[session->focused_door].pins.selected : UINT32_MAX;
		check(harness, picking && actual == wanted, "expect pin_selected %u (actual %d)", wanted,
			  picking ? (int)actual : -1);
	}
	else if (!strcmp(what, "pin_height"))
	{
		uint32_t index = 0, wanted = 0;
		bool parsed_pair = sscanf(rest, "%u %u", &index, &wanted) == 2;
		bool picking = session->phase == DUNGEON_PHASE_PIN_TUMBLER;
		const DungeonPinTumbler *pins =
			picking ? &session->doors[session->focused_door].pins : NULL;
		bool valid = parsed_pair && pins && index < pins->pin_count;
		check(harness, valid && pins->heights[index] == wanted, "expect pin_height %u = %u (actual %d)",
			  index, wanted, valid ? (int)pins->heights[index] : -1);
	}
	else if (!strcmp(what, "safe_pin_next") || !strcmp(what, "safe_pin_wrong"))
	{
		bool want_next = !strcmp(what, "safe_pin_next");
		bool picking = session->phase == DUNGEON_PHASE_SAFE_PINS;
		bool next = picking && dungeon_safe_pins_selected_is_next(
									&session->doors[session->focused_door].safe);
		check(harness, picking && next == want_next, "expect %s (actual %s)", what,
			  !picking ? "no safe" : (next ? "next" : "wrong"));
	}
	else if (!strcmp(what, "safe_progress") || !strcmp(what, "safe_selected"))
	{
		uint32_t wanted = (uint32_t)atoi(rest);
		bool picking = session->phase == DUNGEON_PHASE_SAFE_PINS;
		const DungeonSafePins *safe =
			picking ? &session->doors[session->focused_door].safe : NULL;
		uint32_t actual =
			!safe ? UINT32_MAX
				  : (!strcmp(what, "safe_progress") ? safe->progress : safe->selected);
		check(harness, picking && actual == wanted, "expect %s %u (actual %d)", what, wanted,
			  picking ? (int)actual : -1);
	}
	else if (!strcmp(what, "safe_pins_out") || !strcmp(what, "safe_pins_flush"))
	{
		/* The ANIMATION, not the puzzle state: how far the renderer will
		 * actually push each pin toward the player on the next frame. A driven
		 * pin that never travelled would pass every other check here, and the
		 * travel is the only thing this lock tells the player. The order is
		 * seeded, so a script counts the pins that are out rather than naming
		 * them. */
		bool picking = session->phase == DUNGEON_PHASE_SAFE_PINS;
		const DungeonDoorState *state =
			picking ? &session->doors[session->focused_door] : NULL;
		uint32_t out_count = 0, proud = 0;
		for (uint32_t pin = 0; state && pin < state->safe.pin_count; ++pin)
		{
			out_count += state->safe_push[pin] > 0.75f ? 1u : 0u;
			proud += state->safe_push[pin] > DUNGEON_SAFE_PIN_HOVER + 0.05f ? 1u : 0u;
		}
		if (!strcmp(what, "safe_pins_flush"))
			check(harness, picking && proud == 0u, "expect safe_pins_flush (actual %d proud)",
				  picking ? (int)proud : -1);
		else
		{
			uint32_t wanted = (uint32_t)atoi(rest);
			check(harness, picking && out_count == wanted, "expect safe_pins_out %u (actual %d)",
				  wanted, picking ? (int)out_count : -1);
		}
	}
	else
		check(harness, false, "expect: unknown predicate '%s'", what);
}

/* Returns how many frames the command consumes; 0 for instantaneous ones. */
static uint32_t run_command(DungeonHarness *harness, DungeonScene *scene,
							DungeonCamera *camera, Input *input, const char *line)
{
	char verb[32] = {0};
	char rest[160] = {0};
	sscanf(line, "%31s %159[^\n]", verb, rest);

	if (!strcmp(verb, "teleport"))
	{
		float x = 0.0f, z = 0.0f;
		if (sscanf(rest, "%f %f", &x, &z) == 2)
			scene->player.position = (DungeonPoint){x, z};
		else
			check(harness, false, "teleport: expected <x> <z>");
		return 0;
	}
	if (!strcmp(verb, "teleport_door"))
	{
		uint32_t index = 0;
		char where[16] = {0};
		sscanf(rest, "%u %15s", &index, where);
		if (index >= scene->session.door_count)
		{
			check(harness, false, "teleport_door: no door %u (level has %u)", index,
				  scene->session.door_count);
			return 0;
		}
		scene->player.position = door_approach(scene, index, !strcmp(where, "far"));
		fprintf(stdout, "  teleported to door %u (%s side) at (%.2f, %.2f)\n", index,
				!strcmp(where, "far") ? "far" : "approach", (double)scene->player.position.x,
				(double)scene->player.position.z);
		return 0;
	}
	if (!strcmp(verb, "dump_layout"))
	{
		dungeon_level_print(&scene->level);
		return 0;
	}
	if (!strcmp(verb, "set_yaw"))
	{
		/* Point the camera somewhere deliberately wrong, so that a later
		 * `expect facing` proves the focus actually TURNED rather than
		 * happening to agree with where exploring already looked. */
		camera->camera.yaw = fmodf((float)atof(rest) + 360.0f, 360.0f);
		camera->explore_yaw = camera->camera.yaw;
		return 0;
	}
	if (!strcmp(verb, "set_pitch"))
	{
		/* The other half of aiming. dungeon_camera_orbit derives height and
		 * trailing distance from pitch at a fixed orbit radius, so go through
		 * it rather than writing the pitch field directly -- setting pitch on
		 * its own leaves the follow code framing a different place than the
		 * view is pointing. Orbiting is clamped, so a request outside the
		 * playable range lands at the limit; `report` prints where it got to. */
		float target = (float)atof(rest);
		for (int step = 0; step < 4000; ++step)
		{
			float error_degrees = target - camera->camera.pitch;
			if (fabsf(error_degrees) < 0.05f)
				break;
			dungeon_camera_orbit(camera, 0.0f, error_degrees > 0.0f ? 1.0f : -1.0f, 0.002f);
		}
		camera->explore_height = camera->height;
		camera->explore_trailing = camera->trailing_distance;
		return 0;
	}
	if (!strcmp(verb, "facing_of"))
	{
		uint32_t index = (uint32_t)atoi(rest);
		if (index >= scene->session.door_count)
		{
			check(harness, false, "facing_of: no door %u", index);
			return 0;
		}
		/* Focus geometry is per-door, so borrow the focus index to read it. */
		uint32_t saved = scene->session.focused_door;
		scene->session.focused_door = index;
		fprintf(stdout, "  door %u faces %.1f deg\n", index,
				(double)dungeon_session_focus_facing_degrees(&scene->session));
		scene->session.focused_door = saved;
		return 0;
	}
	if (!strcmp(verb, "wait"))
		return (uint32_t)(fmaxf((float)atof(rest), 0.0f) / HARNESS_TIMESTEP) + 1u;
	if (!strcmp(verb, "walk_to") || !strcmp(verb, "walk_through"))
	{
		float seconds = 0.0f;
		if (!strcmp(verb, "walk_through"))
		{
			uint32_t index = 0;
			if (sscanf(rest, "%u %f", &index, &seconds) != 2 ||
				index >= scene->session.door_count)
			{
				check(harness, false, "walk_through: expected <door> <seconds>");
				return 0;
			}
			harness->walk_target = door_approach(scene, index, true);
		}
		else
		{
			float x = 0.0f, z = 0.0f;
			if (sscanf(rest, "%f %f %f", &x, &z, &seconds) != 3)
			{
				check(harness, false, "walk_to: expected <x> <z> <seconds>");
				return 0;
			}
			harness->walk_target = (DungeonPoint){x, z};
		}
		harness->walk_seeking = true;
		return (uint32_t)(fmaxf(seconds, 0.0f) / HARNESS_TIMESTEP) + 1u;
	}
	if (!strcmp(verb, "walk"))
	{
		float forward = 0.0f, right = 0.0f, seconds = 0.0f;
		if (sscanf(rest, "%f %f %f", &forward, &right, &seconds) != 3)
		{
			check(harness, false, "walk: expected <forward> <right> <seconds>");
			return 0;
		}
		harness->walk_forward = forward;
		harness->walk_right = right;
		return (uint32_t)(fmaxf(seconds, 0.0f) / HARNESS_TIMESTEP) + 1u;
	}
	if (!strcmp(verb, "press"))
	{
		if (!strcmp(rest, "interact")) input->interact = true;
		else if (!strcmp(rest, "left")) input->puzzle_left = true;
		else if (!strcmp(rest, "right")) input->puzzle_right = true;
		else if (!strcmp(rest, "up")) input->puzzle_up = true;
		else if (!strcmp(rest, "down")) input->puzzle_down = true;
		else if (!strcmp(rest, "confirm")) input->puzzle_confirm = true;
		else if (!strcmp(rest, "cancel")) input->puzzle_cancel = true;
		else
		{
			check(harness, false, "press: unknown key '%s'", rest);
			return 0;
		}
		return 1u; /* exactly one frame, matching an edge-triggered key */
	}
	if (!strcmp(verb, "sweep_pins"))
	{
		/* Walk the selection across every pin and back, checking the pick's
		 * clearance on EVERY frame in between rather than only where it comes
		 * to rest. The eased travel is the part that a per-state check misses,
		 * and it is where a badly posed pick cut through the housing. */
		if (scene->session.phase != DUNGEON_PHASE_PIN_TUMBLER)
		{
			check(harness, false, "sweep_pins: no pin tumbler is being picked");
			return 0;
		}
		harness->sweeping = true;
		harness->sweep_violations = 0;
		harness->sweep_frames = 0;
		harness->sweep_direction = 1;
		return 1u;
	}
	if (!strcmp(verb, "drive_safe_pins"))
	{
		/* Walks the selection to the pin the order wants next and presses it,
		 * until N more pins stand driven -- real selection moves and real
		 * presses, so the wrap, the press and the animation are all exercised
		 * rather than bypassed. The order is seeded, so a script cannot name
		 * the pins itself. */
		DungeonSession *session = &scene->session;
		if (session->phase != DUNGEON_PHASE_SAFE_PINS)
		{
			check(harness, false, "drive_safe_pins: no safe is being picked");
			return 0;
		}
		uint32_t wanted = (uint32_t)atoi(rest);
		DungeonSafePins *safe = &session->doors[session->focused_door].safe;
		uint32_t start = safe->progress;
		uint32_t guard = 4000u;
		while (session->phase == DUNGEON_PHASE_SAFE_PINS && safe->progress < start + wanted &&
			   guard--)
		{
			if (dungeon_safe_pins_selected_is_next(safe))
				dungeon_session_press_safe_pin(session);
			else
				dungeon_session_move_safe_pin(session, 1);
		}
		return 1u;
	}
	if (!strcmp(verb, "select_wrong_pin"))
	{
		/* Moves the selection onto a pin the order does NOT want next, so the
		 * script can then press confirm through the real input path and show
		 * that doing so throws the whole order away. */
		DungeonSession *session = &scene->session;
		if (session->phase != DUNGEON_PHASE_SAFE_PINS)
		{
			check(harness, false, "select_wrong_pin: no safe is being picked");
			return 0;
		}
		uint32_t guard = DUNGEON_SAFE_PIN_COUNT + 1u;
		while (dungeon_safe_pins_selected_is_next(&session->doors[session->focused_door].safe) &&
			   guard--)
			dungeon_session_move_safe_pin(session, 1);
		return 1u;
	}
	if (!strcmp(verb, "solve"))
	{
		solve_focused(harness, scene);
		return 0;
	}
	if (!strcmp(verb, "capture"))
	{
		snprintf(harness->pending_capture, sizeof(harness->pending_capture), "%s", rest);
		/* Temporal resolve blends over several frames; capturing immediately
		 * after a state change photographs the transition, not the state. The
		 * frame loop then holds until post_frame clears the pending path --
		 * without that hold the following commands run first and the capture
		 * ends up photographing a state several commands later. */
		harness->capture_delay = 8u;
		return 1u;
	}
	if (!strcmp(verb, "report"))
	{
		const DungeonSession *session = &scene->session;
		fprintf(stdout,
				"  state: phase=%s focus=%.3f yaw=%.1f pitch=%.1f height=%.2f player=(%.2f,%.2f)",
				phase_name(session->phase), (double)dungeon_camera_focus_blend(camera),
				(double)camera->camera.yaw, (double)camera->camera.pitch, (double)camera->height,
				(double)scene->player.position.x, (double)scene->player.position.z);
		for (uint32_t i = 0; i < session->door_count; ++i)
			fprintf(stdout, " door%u=%s", i, session->doors[i].open ? "open" : "shut");
		if (session->phase == DUNGEON_PHASE_PIN_TUMBLER)
		{
			const DungeonPinTumbler *pins = &session->doors[session->focused_door].pins;
			fprintf(stdout, " selected=%u heights=", pins->selected);
			for (uint32_t pin = 0; pin < pins->pin_count; ++pin)
				fprintf(stdout, "%u", pins->heights[pin]);
			fprintf(stdout, " target=");
			for (uint32_t pin = 0; pin < pins->pin_count; ++pin)
				fprintf(stdout, "%u", pins->target[pin]);
		}
		else if (session->phase == DUNGEON_PHASE_SAFE_PINS)
		{
			const DungeonDoorState *safe_state = &session->doors[session->focused_door];
			const DungeonSafePins *safe = &safe_state->safe;
			fprintf(stdout, " driven=%u/%u selected=%u next=%s shake=%.2f push=", safe->progress,
					safe->pin_count, safe->selected,
					dungeon_safe_pins_selected_is_next(safe) ? "YES" : "no",
					(double)safe_state->safe_shake);
			for (uint32_t pin = 0; pin < safe->pin_count; ++pin)
				fprintf(stdout, "%s%.2f", pin ? "," : "", (double)safe_state->safe_push[pin]);
		}
		fprintf(stdout, "\n");
		return 0;
	}
	if (!strcmp(verb, "expect"))
	{
		run_expect(harness, scene, camera, rest);
		return 0;
	}
	if (!strcmp(verb, "drive_prisms"))
	{
		check(harness, scene->session.phase == DUNGEON_PHASE_PRISM,
			  "drive_prisms: prism lock focused");
		harness->driving_prisms = scene->session.phase == DUNGEON_PHASE_PRISM;
		harness->prism_frames = 0;
		return 1;
	}
	if (!strcmp(verb, "quit"))
	{
		harness->finished = true;
		return 0;
	}
	check(harness, false, "unknown command '%s'", verb);
	return 0;
}

/* Writes this frame's movement. A seeking walk is resolved against the CURRENT
 * camera yaw every frame, because dungeon_player_update projects WASD onto the
 * camera basis and that basis is still rotating back from the lock. */
static void harness_apply_movement(DungeonHarness *harness, const DungeonScene *scene,
								   const DungeonCamera *camera, Input *input)
{
	if (!harness->walk_seeking)
	{
		input->move_forward = harness->walk_forward;
		input->move_right = harness->walk_right;
		return;
	}
	DungeonPoint player = scene->player.position;
	float dx = harness->walk_target.x - player.x, dz = harness->walk_target.z - player.z;
	float length = sqrtf(dx * dx + dz * dz);
	if (length < 1e-3f)
	{
		input->move_forward = input->move_right = 0.0f;
		return;
	}
	dx /= length;
	dz /= length;
	float yaw = camera->camera.yaw * (3.14159265358979f / 180.0f);
	DungeonPoint forward = {cosf(yaw), sinf(yaw)};
	DungeonPoint right = {-forward.z, forward.x};
	input->move_forward = dx * forward.x + dz * forward.z;
	input->move_right = dx * right.x + dz * right.z;
}

/* Solve through actual per-frame Input flags; no state or session mutations.
 * BFS follows the same navigation graph available to the player. */
static bool drive_prism_input(DungeonHarness *h, const DungeonSession *s, Input *input)
{
	if (!h->driving_prisms)
		return false;
	if (s->phase != DUNGEON_PHASE_PRISM || ++h->prism_frames > 4000)
	{
		check(h, false, "drive_prisms: phase changed or input budget exceeded");
		h->driving_prisms = false;
		return false;
	}
	const DungeonPrismPuzzle *p = &s->doors[s->focused_door].prism;
	if (p->solved)
	{
		check(h, true, "drive_prisms: key reached through keyboard input in %u frames",
			  h->prism_frames);
		h->driving_prisms = false;
		return false;
	}
	if (p->turn_remaining > 0)
		return true;
	unsigned target = 0;
	while (target < p->count && p->prisms[target].orientation == p->solution[target])
		++target;
	if (target == p->count)
	{
		check(h, false, "drive_prisms: witness failed to solve");
		h->driving_prisms = false;
		return false;
	}
	if (p->selected == target)
	{
		if (!p->rotating)
			input->puzzle_confirm = true;
		else
		{
			unsigned clockwise = (p->solution[target] + 360 - p->prisms[target].orientation) % 360;
			if (clockwise <= 180)
				input->puzzle_right = true;
			else
				input->puzzle_left = true;
		}
	}
	else if (p->rotating)
		input->puzzle_confirm = true;
	else
	{
		unsigned queue[DUNGEON_PRISM_MAX], first[DUNGEON_PRISM_MAX] = {0};
		bool seen[DUNGEON_PRISM_MAX] = {false};
		unsigned head = 0, tail = 0;
		queue[tail++] = p->selected;
		seen[p->selected] = true;
		while (head < tail && !seen[target])
		{
			unsigned from = queue[head++];
			for (unsigned d = 0; d < 4; ++d)
			{
				unsigned to = dungeon_prism_neighbor(p, from, d);
				if (seen[to])
					continue;
				seen[to] = true;
				queue[tail++] = to;
				first[to] = from == p->selected ? d : first[from];
			}
		}
		if (!seen[target])
		{
			check(h, false, "drive_prisms: unreachable prism");
			h->driving_prisms = false;
			return false;
		}
		switch (first[target])
		{
		case DUNGEON_PRISM_NORTH:
			input->puzzle_up = true;
			break;
		case DUNGEON_PRISM_EAST:
			input->puzzle_right = true;
			break;
		case DUNGEON_PRISM_SOUTH:
			input->puzzle_down = true;
			break;
		default:
			input->puzzle_left = true;
			break;
		}
	}
	return true;
}

bool dungeon_harness_pre_frame(DungeonHarness *harness, DungeonScene *scene,
							   DungeonCamera *camera, Input *input)
{
	if (!harness || !scene || !input)
		return true;
	// A script owns its input; live keyboard events must not corrupt checks.
	input->puzzle_left = input->puzzle_right = input->puzzle_up = input->puzzle_down = false;
	input->puzzle_confirm = input->puzzle_cancel = input->interact = false;
	input->move_forward = 0.0f;
	input->move_right = 0.0f;
	input->orbit_yaw = 0.0f;
	input->orbit_pitch = 0.0f;
	if (harness->pending_capture[0])
		return true; /* hold the world still until the capture is actually taken */
	if (drive_prism_input(harness, &scene->session, input))
		return true;
	if (harness->sweeping)
	{
		DungeonSession *session = &scene->session;
		if (session->phase != DUNGEON_PHASE_PIN_TUMBLER)
		{
			/* The sweep sets pins as it goes, so it can open the lock before
			 * running its course. Report what was checked rather than dropping
			 * the assertion on the floor. */
			harness->sweeping = false;
			check(harness, harness->sweep_violations == 0u,
				  "sweep_pins: pick cleared the lock on all %u animation frames (%u hits, lock "
				  "opened mid-sweep)",
				  harness->sweep_frames, harness->sweep_violations);
		}
		else
		{
			const DungeonPinTumbler *pins = &session->doors[session->focused_door].pins;
			if (!dungeon_lock_layout_pick_clear(pins, scene->pick_lateral, scene->pick_height))
				++harness->sweep_violations;
			++harness->sweep_frames;
			/* One key press every few frames, so the pick is caught mid-travel
			 * and not only once it has settled. */
			if (harness->sweep_frames % 5u == 0u)
			{
				if (session->doors[session->focused_door].pins.selected == 0u)
					harness->sweep_direction = 1;
				else if (session->doors[session->focused_door].pins.selected ==
						 pins->pin_count - 1u)
					harness->sweep_direction = -1;
				dungeon_session_move_selection(session, harness->sweep_direction);
				/* Move the pin too, so the tip's height travels as well as its
				 * lateral position. */
				dungeon_session_adjust(session, harness->sweep_direction);
			}
			if (harness->sweep_frames >= 240u)
			{
				harness->sweeping = false;
				check(harness, harness->sweep_violations == 0u,
					  "sweep_pins: pick cleared the lock on all %u animation frames (%u hits)",
					  harness->sweep_frames, harness->sweep_violations);
			}
		}
		return true;
	}
	if (harness->frames_remaining)
	{
		--harness->frames_remaining;
		harness_apply_movement(harness, scene, camera, input);
		return true;
	}
	harness->walk_forward = harness->walk_right = 0.0f;
	harness->walk_seeking = false;
	/* Run instantaneous commands until one consumes a frame, so a run of
	 * expects and a teleport all land in the same frame's state. */
	while (!harness->finished && harness->cursor < harness->line_count)
	{
		const char *line = harness->lines[harness->cursor++];
		fprintf(stdout, "[%u] %s\n", harness->cursor, line);
		uint32_t frames = run_command(harness, scene, camera, input, line);
		if (frames)
		{
			harness->frames_remaining = frames - 1u;
			harness_apply_movement(harness, scene, camera, input);
			return true;
		}
	}
	if (harness->cursor >= harness->line_count || harness->finished)
	{
		fprintf(stdout, "Harness: %u checks, %u failed\n", harness->checks_run,
				harness->checks_failed);
		return false;
	}
	return true;
}

void dungeon_harness_post_frame(DungeonHarness *harness, Renderer *renderer)
{
	if (!harness || !renderer || !harness->pending_capture[0])
		return;
	if (harness->capture_delay)
	{
		--harness->capture_delay;
		return;
	}
	check(harness, renderer_capture_swapchain(renderer, harness->pending_capture),
		  "capture %s", harness->pending_capture);
	harness->pending_capture[0] = '\0';
}
