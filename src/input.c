#include "input.h"

#include <stdio.h>

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

void input_poll(Input *in, SDL_Window *window)
{
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

	SDL_Event event;
	while (SDL_PollEvent(&event))
	{
		if (event.type == SDL_EVENT_QUIT)
			in->quit = true;
		if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE)
			in->quit = true;
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
		/* E toggles auto-exposure (eye adaptation); off holds a fixed exposure. */
		if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.key == SDLK_E)
			in->toggle_auto_exposure = true;
		if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat &&
			(event.key.key == SDLK_LCTRL || event.key.key == SDLK_RCTRL))
			set_mouse_capture(in, window, !in->mouse_captured);
		if (event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED)
			in->resized = true;
	}

	float mouse_x = 0.0f, mouse_y = 0.0f;
	SDL_GetRelativeMouseState(&mouse_x, &mouse_y);
	in->look_dx = in->mouse_captured ? mouse_x : 0.0f;
	in->look_dy = in->mouse_captured ? mouse_y : 0.0f;

	const bool *keys = SDL_GetKeyboardState(NULL);
	in->rotate_sun = keys[SDL_SCANCODE_F12];
	if (in->mouse_captured)
	{
		in->move_forward =
			(keys[SDL_SCANCODE_W] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_S] ? 1.0f : 0.0f);
		in->move_right = (keys[SDL_SCANCODE_D] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_A] ? 1.0f : 0.0f);
		in->sprint = keys[SDL_SCANCODE_LSHIFT];
	}
	else
	{
		in->move_forward = 0.0f;
		in->move_right = 0.0f;
		in->sprint = false;
	}
}
