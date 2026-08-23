#pragma once

#include <stdbool.h>
#include <SDL3/SDL.h>

typedef struct {
    float look_dx, look_dy;         /* mouse delta this frame  */
    float move_forward, move_right; /* -1..1 from WASD         */
    bool  sprint;                   /* shift held              */
    bool  quit;                     /* window close / escape   */
    bool  resized;                  /* framebuffer size change */
    bool  toggle_quarry_shading;    /* F3 pressed this frame   */
    bool  cycle_temporal_debug;     /* F4 pressed this frame   */
    bool  reload_shaders;           /* F5 pressed this frame   */
    bool  toggle_depth_debug;       /* F6 pressed this frame   */
    bool  toggle_lod_debug;         /* F7 pressed this frame   */
    bool  cycle_relighting;         /* F8 pressed this frame   */
    bool  cycle_surface_debug;      /* F9 pressed this frame   */
    bool  cycle_shadow_debug;       /* F10 pressed this frame  */
    bool  cycle_atmosphere_debug;   /* F11 pressed this frame  */
    bool  cycle_sun;                /* F12 pressed this frame  */
    bool  previous_atmosphere_slice;
    bool  next_atmosphere_slice;
    bool  dump_shader_data;         /* X pressed (debug builds) */
    bool  clear_shader_dump;        /* C pressed (debug builds) */
} Input;

/* Pumps SDL events + keyboard/mouse state into `in`. */
void input_poll(Input *in, SDL_Window *window);
