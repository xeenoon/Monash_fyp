#include "input.h"

void input_poll(Input *in, SDL_Window *window)
{
	(void)window;
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
	in->cycle_sun = false;
	in->previous_atmosphere_slice = false;
	in->next_atmosphere_slice = false;
	in->dump_shader_data = false;
	in->clear_shader_dump = false;

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
		if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.key == SDLK_F12)
			in->cycle_sun = true;
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
		if (event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED)
			in->resized = true;
	}

	float mouse_x = 0.0f, mouse_y = 0.0f;
	SDL_GetRelativeMouseState(&mouse_x, &mouse_y);
	in->look_dx = mouse_x;
	in->look_dy = mouse_y;

	const bool *keys = SDL_GetKeyboardState(NULL);
	in->move_forward = (keys[SDL_SCANCODE_W] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_S] ? 1.0f : 0.0f);
	in->move_right = (keys[SDL_SCANCODE_D] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_A] ? 1.0f : 0.0f);
	in->sprint = keys[SDL_SCANCODE_LSHIFT];
}
