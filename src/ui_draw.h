#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* A tiny software painter for the renderer's UI canvas (renderer_ui_pixels).
 * Straight-alpha RGBA8, display-encoded, row 0 at the top. Everything the menus
 * and HUD need -- filled panels, baked-font text, stars -- and nothing else. */

typedef enum
{
	UI_FONT_SMALL,
	UI_FONT_BODY,
	UI_FONT_HEADING,
	UI_FONT_TITLE
} UiFont;

typedef enum
{
	UI_ALIGN_LEFT,
	UI_ALIGN_CENTRE,
	UI_ALIGN_RIGHT
} UiAlign;

typedef struct
{
	uint8_t r, g, b, a;
} UiColor;

typedef struct
{
	uint8_t *pixels;
	int width, height;
} UiCanvas;

void ui_clear(UiCanvas *c);
/* Source-over blend of a solid colour into a rectangle (clipped). */
void ui_fill_rect(UiCanvas *c, int x, int y, int w, int h, UiColor color);
/* Rectangle outline `thickness` pixels wide, drawn inside the rectangle. */
void ui_frame_rect(UiCanvas *c, int x, int y, int w, int h, int thickness, UiColor color);
/* Vertical gradient from `top` to `bottom`. */
void ui_gradient_rect(UiCanvas *c, int x, int y, int w, int h, UiColor top, UiColor bottom);
int ui_text_width(UiFont font, const char *text);
int ui_line_height(UiFont font);
/* `y` is the top of the line box. Draws a 2 px drop shadow first so text reads
 * over a lit scene. Returns the width drawn. */
int ui_text(UiCanvas *c, UiFont font, int x, int y, UiAlign align, UiColor color, const char *text);
/* Word-wraps `text` into `max_width`; returns the y below the last line. */
int ui_text_wrapped(UiCanvas *c, UiFont font, int x, int y, int max_width, UiColor color,
					const char *text);
/* Five-pointed star centred at (cx, cy); hollow draws only a dim fill. */
void ui_star(UiCanvas *c, float cx, float cy, float radius, UiColor color, bool filled);
/* Wrap the complete text once per draw, reveal without changing its layout.
 * Returns consumed characters; stops at the panel's line limit. */
size_t ui_text_reveal(UiCanvas *c, UiFont font, int x, int y, int width, unsigned max_lines,
					  UiColor color, const char *text, size_t visible);
