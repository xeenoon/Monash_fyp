#pragma once

#include <SDL3/SDL.h>
#include <stdbool.h>

typedef struct
{
	float look_dx, look_dy;			/* mouse delta this frame  */
	float orbit_yaw, orbit_pitch; /* arrow keys */
	float move_forward, move_right; /* -1..1 from WASD         */
	bool sprint;					/* shift held              */
	bool quit;						/* window close / escape   */
	bool resized;					/* framebuffer size change */
	bool mouse_captured;			/* camera mode vs free cursor (Ctrl toggles) */
	bool toggle_quarry_shading;		/* F3 pressed this frame   */
	bool cycle_temporal_debug;		/* F4 pressed this frame   */
	bool reload_shaders;			/* F5 pressed this frame   */
	bool toggle_depth_debug;		/* F6 pressed this frame   */
	bool toggle_lod_debug;			/* F7 pressed this frame   */
	bool cycle_relighting;			/* F8 pressed this frame   */
	bool cycle_surface_debug;		/* F9 pressed this frame   */
	bool cycle_shadow_debug;		/* F10 pressed this frame  */
	bool cycle_atmosphere_debug;	/* F11 pressed this frame  */
	bool rotate_sun;				/* F12 held                */
	bool previous_atmosphere_slice;
	bool next_atmosphere_slice;
	bool dump_shader_data;	/* X pressed (dump-enabled builds) */
	bool clear_shader_dump; /* C pressed (dump-enabled builds) */
	bool toggle_auto_exposure; /* E pressed */
	/* Dungeon lock picking. All edge-triggered: one press is one pin step, one
	 * dial entry, or one attempt -- holding a key must not spin a tumbler.
	 * The directions fire on the arrows AND on WASD, as the source game does;
	 * outside a lock they are simply ignored, so they cost the free-fly and
	 * exploring paths nothing. `interact` shares E with the auto-exposure
	 * toggle, which the dungeon scene does not use. */
	bool interact;
	bool puzzle_left, puzzle_right, puzzle_up, puzzle_down;
	bool puzzle_confirm; /* Enter or Space: try the lock */
	bool puzzle_cancel;	 /* Q: step back from a lock without solving it */
} Input;

/* Pumps SDL events + keyboard/mouse state into `in`. */
void input_poll(Input *in, SDL_Window *window);
