#include "input.h"

void input_poll(Input *in, SDL_Window *window) {
    (void)window;
    in->quit = false;
    in->resized = false;
    in->reload_shaders = false;

    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_EVENT_QUIT) in->quit = true;
        if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_ESCAPE) in->quit = true;
        /* Edge-triggered so one press reloads once, not every frame F5 is held. */
        if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.key == SDLK_F5)
            in->reload_shaders = true;
        if (event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) in->resized = true;
    }

    float mouse_x = 0.0f, mouse_y = 0.0f;
    SDL_GetRelativeMouseState(&mouse_x, &mouse_y);
    in->look_dx = mouse_x;
    in->look_dy = mouse_y;

    const bool *keys = SDL_GetKeyboardState(NULL);
    in->move_forward = (keys[SDL_SCANCODE_W] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_S] ? 1.0f : 0.0f);
    in->move_right   = (keys[SDL_SCANCODE_D] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_A] ? 1.0f : 0.0f);
    in->sprint       = keys[SDL_SCANCODE_LSHIFT];
}
