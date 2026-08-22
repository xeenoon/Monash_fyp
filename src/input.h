#pragma once

#include <stdbool.h>
#include <SDL3/SDL.h>

typedef struct {
    float look_dx, look_dy;         /* mouse delta this frame  */
    float move_forward, move_right; /* -1..1 from WASD         */
    bool  sprint;                   /* shift held              */
    bool  quit;                     /* window close / escape   */
    bool  resized;                  /* framebuffer size change */
    bool  reload_shaders;           /* F5 pressed this frame   */
    bool  toggle_depth_debug;       /* F6 pressed this frame   */
} Input;

/* Pumps SDL events + keyboard/mouse state into `in`. */
void input_poll(Input *in, SDL_Window *window);
