#include "dungeon_harness.h"

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
	case DUNGEON_PHASE_VAULT_DIAL: return "dial";
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
	else if (session->phase == DUNGEON_PHASE_VAULT_DIAL)
	{
		const DungeonVaultDial *dial = &session->doors[session->focused_door].dial;
		/* Continue from wherever the player left off rather than replaying from
		 * step zero: entering step 0 while progress is at 2 is a miss, which
		 * resets the dial and makes solve fail on a half-entered combination. */
		for (uint32_t guard = 0;
			 guard < DUNGEON_LOCK_MAX_STEPS * 2u && session->phase == DUNGEON_PHASE_VAULT_DIAL;
			 ++guard)
			dungeon_session_turn(session, dial->combination[dial->progress]);
		check(harness, session->phase == DUNGEON_PHASE_EXPLORING, "solve: vault dial accepted");
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
	else if (!strcmp(what, "dial_progress"))
	{
		uint32_t wanted = (uint32_t)atoi(rest);
		bool picking = session->phase == DUNGEON_PHASE_VAULT_DIAL;
		uint32_t actual = picking ? session->doors[session->focused_door].dial.progress : UINT32_MAX;
		check(harness, picking && actual == wanted, "expect dial_progress %u (actual %d)", wanted,
			  picking ? (int)actual : -1);
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
	if (!strcmp(verb, "set_yaw"))
	{
		/* Point the camera somewhere deliberately wrong, so that a later
		 * `expect facing` proves the focus actually TURNED rather than
		 * happening to agree with where exploring already looked. */
		camera->camera.yaw = fmodf((float)atof(rest) + 360.0f, 360.0f);
		camera->explore_yaw = camera->camera.yaw;
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
	if (!strcmp(verb, "press_combination"))
	{
		/* Enters the next N steps of the focused dial's combination through the
		 * real input path, one key per frame -- unlike `solve`, which reaches
		 * past the input entirely. Used to show banked progress mid-puzzle. */
		DungeonSession *session = &scene->session;
		if (session->phase != DUNGEON_PHASE_VAULT_DIAL)
		{
			check(harness, false, "press_combination: no dial is being picked");
			return 0;
		}
		const DungeonVaultDial *dial = &session->doors[session->focused_door].dial;
		uint32_t steps = (uint32_t)atoi(rest);
		for (uint32_t i = 0; i < steps && dial->progress < dial->step_count; ++i)
		{
			switch (dial->combination[dial->progress])
			{
			case DUNGEON_DIAL_LEFT: input->puzzle_left = true; break;
			case DUNGEON_DIAL_RIGHT: input->puzzle_right = true; break;
			case DUNGEON_DIAL_UP: input->puzzle_up = true; break;
			case DUNGEON_DIAL_DOWN: input->puzzle_down = true; break;
			}
			/* The main loop consumes one direction per frame, so each step has
			 * to be handed over on its own frame. */
			dungeon_session_turn(session, dial->combination[dial->progress]);
			input->puzzle_left = input->puzzle_right = false;
			input->puzzle_up = input->puzzle_down = false;
		}
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
		else if (session->phase == DUNGEON_PHASE_VAULT_DIAL)
		{
			const DungeonVaultDial *dial = &session->doors[session->focused_door].dial;
			fprintf(stdout, " progress=%u/%u", dial->progress, dial->step_count);
		}
		fprintf(stdout, "\n");
		return 0;
	}
	if (!strcmp(verb, "expect"))
	{
		run_expect(harness, scene, camera, rest);
		return 0;
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

bool dungeon_harness_pre_frame(DungeonHarness *harness, DungeonScene *scene,
							   DungeonCamera *camera, Input *input)
{
	if (!harness || !scene || !input)
		return true;
	input->move_forward = 0.0f;
	input->move_right = 0.0f;
	input->orbit_yaw = 0.0f;
	input->orbit_pitch = 0.0f;
	if (harness->pending_capture[0])
		return true; /* hold the world still until the capture is actually taken */
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
