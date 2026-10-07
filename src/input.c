#include "input.h"

#include <stdio.h>
#include <string.h>

static void set_mouse_capture(Input *in, SDL_Window *window, bool captured)
{
	if (SDL_SetWindowRelativeMouseMode(window, captured))
	{
		in->mouse_captured = captured;
		if (captured)
			SDL_HideCursor();
		else
			SDL_ShowCursor();
	}
	else
		fprintf(stderr, "Could not change relative mouse mode: %s\n", SDL_GetError());
}

void input_set_mode(Input *in, SDL_Window *window, InputMode mode)
{
	if (in->mode == mode)
		return;
	if (in->mode == INPUT_TEXT)
		SDL_StopTextInput(window);
	if (in->mode == INPUT_GAME && mode != INPUT_GAME)
	{
		in->restore_capture = in->mouse_captured;
		set_mouse_capture(in, window, false);
	}
	if (mode == INPUT_GAME)
		set_mouse_capture(in, window, in->restore_capture);
	if (mode == INPUT_TEXT)
		SDL_StartTextInput(window);
	in->mode = mode;
}

static void append_edit(Input *in, TextEditKind kind, const char *text)
{
	if (in->edit_count >= INPUT_MAX_EDITS)
		return;
	TextEdit *edit = &in->edits[in->edit_count++];
	*edit = (TextEdit){.kind = kind};
	if (text)
		snprintf(edit->text, sizeof(edit->text), "%s", text);
}

void input_poll(Input *in, SDL_Window *window)
{
	in->edit_count = 0;
	in->quit = false;
	in->resized = false;
	in->toggle_quarry_shading = false;
	in->cycle_temporal_debug = false;
	in->reload_shaders = false;
	in->toggle_depth_debug = false;
	in->toggle_lod_debug = false;
	in->cycle_relighting = false;
	in->cycle_surface_debug = false;
	in->cycle_shadow_debug = false;
	in->cycle_atmosphere_debug = false;
	in->rotate_sun = false;
	in->previous_atmosphere_slice = false;
	in->next_atmosphere_slice = false;
	in->dump_shader_data = false;
	in->clear_shader_dump = false;
	in->toggle_auto_exposure = false;
	in->interact = false;
	in->puzzle_left = in->puzzle_right = in->puzzle_up = in->puzzle_down = false;
	in->puzzle_confirm = false;
	in->puzzle_cancel = false;
	in->menu_up = in->menu_down = in->menu_left = in->menu_right = false;
	in->escape = false;
	in->restart = false;
	in->mouse_dx = in->mouse_dy = 0.0f;
	in->mouse_left_pressed = in->mouse_left_released = false;
	in->wheel = 0.0f;
	in->map_zoom = 0.0f;
	in->tab = false;

	SDL_Event event;
	while (SDL_PollEvent(&event))
	{
		if (event.type == SDL_EVENT_QUIT)
			in->quit = true;
		if (in->mode == INPUT_TEXT && event.type == SDL_EVENT_TEXT_INPUT)
		{
			append_edit(in, TEXT_INSERT, event.text.text);
			continue;
		}
		if (in->mode != INPUT_GAME && event.type == SDL_EVENT_KEY_DOWN)
		{
			SDL_Keycode key = event.key.key;
			if (in->mode == INPUT_TEXT)
			{
				switch (key)
				{
				case SDLK_BACKSPACE:
					append_edit(in, TEXT_BACKSPACE, NULL);
					break;
				case SDLK_DELETE:
					append_edit(in, TEXT_DELETE, NULL);
					break;
				case SDLK_LEFT:
					append_edit(in, TEXT_LEFT, NULL);
					break;
				case SDLK_RIGHT:
					append_edit(in, TEXT_RIGHT, NULL);
					break;
				case SDLK_HOME:
					append_edit(in, TEXT_HOME, NULL);
					break;
				case SDLK_END:
					append_edit(in, TEXT_END, NULL);
					break;
				case SDLK_RETURN:
					if (!event.key.repeat)
						append_edit(in, TEXT_SUBMIT, NULL);
					break;
				case SDLK_ESCAPE:
					if (!event.key.repeat)
						in->escape = true;
					break;
				default:
					break;
				}
			}
			else if (!event.key.repeat)
			{
				in->menu_up |= key == SDLK_UP || key == SDLK_W;
				in->menu_down |= key == SDLK_DOWN || key == SDLK_S;
				in->puzzle_confirm |= key == SDLK_RETURN || key == SDLK_SPACE;
				in->escape |= key == SDLK_ESCAPE;
			}
			continue;
		}
		if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.key == SDLK_ESCAPE)
			in->escape = true;
		if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.key == SDLK_F3)
			in->toggle_quarry_shading = true;
		if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.key == SDLK_F4)
			in->cycle_temporal_debug = true;
		/* Edge-triggered so one press reloads once, not every frame F5 is held. */
		if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.key == SDLK_F5)
			in->reload_shaders = true;
		if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.key == SDLK_F6)
			in->toggle_depth_debug = true;
		if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.key == SDLK_F7)
			in->toggle_lod_debug = true;
		if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.key == SDLK_F8)
			in->cycle_relighting = true;
		if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.key == SDLK_F9)
			in->cycle_surface_debug = true;
		if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.key == SDLK_F10)
			in->cycle_shadow_debug = true;
		if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.key == SDLK_F11)
			in->cycle_atmosphere_debug = true;
		if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat &&
			event.key.key == SDLK_LEFTBRACKET)
			in->previous_atmosphere_slice = true;
		if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat &&
			event.key.key == SDLK_RIGHTBRACKET)
			in->next_atmosphere_slice = true;
		/* Debug shader dump: X appends the current frame's records, C clears. */
		if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.key == SDLK_X)
			in->dump_shader_data = true;
		if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.key == SDLK_C)
			in->clear_shader_dump = true;
		/* E toggles auto-exposure (eye adaptation); off holds a fixed exposure.
		 * In the dungeon it is instead the interact key -- that scene runs with
		 * auto-exposure disabled, so the two never both apply. */
		if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.key == SDLK_E)
		{
			in->toggle_auto_exposure = true;
			in->interact = true;
		}
		/* Lock picking. Edge-triggered on key-down with repeats rejected, so a
		 * held key cannot walk a pin through every height in one frame.
		 *
		 * The arrow keys alone, deliberately: WASD used to be bound here as
		 * well, which left a player at a lock with no way to look around it
		 * except the mouse. WASD keeps its held movement meaning instead, and
		 * the dungeon spends it on nudging the inspection view while a lock is
		 * up (see dungeon_camera_focus_pan). */
		if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat)
			switch (event.key.key)
			{
			case SDLK_LEFT:
				in->puzzle_left = true;
				break;
			case SDLK_RIGHT:
				in->puzzle_right = true;
				break;
			case SDLK_UP:
				in->puzzle_up = true;
				break;
			case SDLK_DOWN:
				in->puzzle_down = true;
				break;
			case SDLK_RETURN:
			case SDLK_SPACE:
				in->puzzle_confirm = true;
				break;
			case SDLK_Q:
				in->puzzle_cancel = true;
				break;
			case SDLK_R:
				in->restart = true;
				break;
			default:
				break;
			}
		if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat)
			switch (event.key.key)
			{
			case SDLK_UP:
			case SDLK_W:
				in->menu_up = true;
				break;
			case SDLK_DOWN:
			case SDLK_S:
				in->menu_down = true;
				break;
			case SDLK_LEFT:
			case SDLK_A:
				in->menu_left = true;
				break;
			case SDLK_RIGHT:
			case SDLK_D:
				in->menu_right = true;
				break;
			default:
				break;
			}
		if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat &&
			(event.key.key == SDLK_LCTRL || event.key.key == SDLK_RCTRL))
			set_mouse_capture(in, window, !in->mouse_captured);
		if (event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED)
			in->resized = true;
		if (event.type == SDL_EVENT_MOUSE_MOTION)
		{
			in->mouse_x = event.motion.x;
			in->mouse_y = event.motion.y;
			in->mouse_dx += event.motion.xrel;
			in->mouse_dy += event.motion.yrel;
		}
		if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN || event.type == SDL_EVENT_MOUSE_BUTTON_UP)
		{
			bool down = event.type == SDL_EVENT_MOUSE_BUTTON_DOWN;
			in->mouse_x = event.button.x;
			in->mouse_y = event.button.y;
			if (event.button.button == SDL_BUTTON_LEFT)
			{
				in->mouse_left = down;
				in->mouse_left_pressed |= down;
				in->mouse_left_released |= !down;
			}
			if (event.button.button == SDL_BUTTON_RIGHT)
				in->mouse_right = down;
		}
		if (event.type == SDL_EVENT_MOUSE_WHEEL)
			in->wheel += event.wheel.y;
		if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.key == SDLK_TAB)
			in->tab = true;
	}

	float mouse_x = 0.0f, mouse_y = 0.0f;
	SDL_GetRelativeMouseState(&mouse_x, &mouse_y);
	in->look_dx = in->mouse_captured ? mouse_x : 0.0f;
	in->look_dy = in->mouse_captured ? mouse_y : 0.0f;

	const bool *keys = SDL_GetKeyboardState(NULL);
	if (in->mode != INPUT_GAME)
	{
		in->rotate_sun = in->sprint = false;
		in->move_forward = in->move_right = in->orbit_yaw = in->orbit_pitch = 0;
		in->map_zoom = 0.0f;
		in->look_dx = in->look_dy = 0;
		return;
	}
	in->rotate_sun = keys[SDL_SCANCODE_F12];
	in->move_forward =
		(keys[SDL_SCANCODE_W] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_S] ? 1.0f : 0.0f);
	in->move_right =
		(keys[SDL_SCANCODE_D] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_A] ? 1.0f : 0.0f);
	in->orbit_yaw = (keys[SDL_SCANCODE_RIGHT] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_LEFT] ? 1.0f : 0.0f);
	in->orbit_pitch = (keys[SDL_SCANCODE_DOWN] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_UP] ? 1.0f : 0.0f);
	in->map_zoom = (keys[SDL_SCANCODE_EQUALS] ? 1.0f : 0.0f) -
				   (keys[SDL_SCANCODE_MINUS] ? 1.0f : 0.0f);
	in->sprint = keys[SDL_SCANCODE_LSHIFT];
}
