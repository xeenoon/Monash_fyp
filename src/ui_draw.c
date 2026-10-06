#include "ui_draw.h"

#include "ui_font_data.h"

#include <math.h>
#include <string.h>

static void blend_pixel(UiCanvas *c, int x, int y, UiColor color, uint32_t coverage)
{
	if (x < 0 || y < 0 || x >= c->width || y >= c->height)
		return;
	uint32_t a = (uint32_t)color.a * coverage / 255u;
	if (!a)
		return;
	uint8_t *p = c->pixels + ((size_t)y * (size_t)c->width + (size_t)x) * 4u;
	/* Straight-alpha source-over onto a straight-alpha destination. */
	uint32_t da = p[3];
	uint32_t out_a = a + da * (255u - a) / 255u;
	if (!out_a)
		return;
	const uint8_t src[3] = {color.r, color.g, color.b};
	for (int i = 0; i < 3; ++i)
		p[i] = (uint8_t)((src[i] * a + p[i] * da * (255u - a) / 255u) / out_a);
	p[3] = (uint8_t)out_a;
}

void ui_clear(UiCanvas *c)
{
	memset(c->pixels, 0, (size_t)c->width * (size_t)c->height * 4u);
}

void ui_fill_rect(UiCanvas *c, int x, int y, int w, int h, UiColor color)
{
	for (int j = y < 0 ? 0 : y; j < y + h && j < c->height; ++j)
		for (int i = x < 0 ? 0 : x; i < x + w && i < c->width; ++i)
			blend_pixel(c, i, j, color, 255u);
}

void ui_frame_rect(UiCanvas *c, int x, int y, int w, int h, int t, UiColor color)
{
	ui_fill_rect(c, x, y, w, t, color);
	ui_fill_rect(c, x, y + h - t, w, t, color);
	ui_fill_rect(c, x, y + t, t, h - 2 * t, color);
	ui_fill_rect(c, x + w - t, y + t, t, h - 2 * t, color);
}

void ui_gradient_rect(UiCanvas *c, int x, int y, int w, int h, UiColor top, UiColor bottom)
{
	for (int j = 0; j < h; ++j)
	{
		float t = h > 1 ? (float)j / (float)(h - 1) : 0.0f;
		UiColor row = {(uint8_t)(top.r + (bottom.r - top.r) * t),
					   (uint8_t)(top.g + (bottom.g - top.g) * t),
					   (uint8_t)(top.b + (bottom.b - top.b) * t),
					   (uint8_t)(top.a + (bottom.a - top.a) * t)};
		ui_fill_rect(c, x, y + j, w, 1, row);
	}
}

static const UiGlyph *glyph(const UiFontData *font, char ch)
{
	int code = (unsigned char)ch;
	if (code < UI_FONT_FIRST_CHAR || code > UI_FONT_LAST_CHAR)
		code = '?';
	return &font->glyphs[code - UI_FONT_FIRST_CHAR];
}

int ui_text_width(UiFont font, const char *text)
{
	int width = 0;
	for (const char *p = text; *p; ++p)
		width += glyph(&ui_fonts[font], *p)->advance;
	return width;
}

int ui_line_height(UiFont font)
{
	return ui_fonts[font].line_height;
}

static void draw_run(UiCanvas *c, const UiFontData *font, int x, int y, UiColor color,
					 const char *text)
{
	for (const char *p = text; *p; ++p)
	{
		const UiGlyph *g = glyph(font, *p);
		const uint8_t *bitmap = font->pixels + g->offset;
		for (int j = 0; j < font->line_height; ++j)
			for (int i = 0; i < g->width; ++i)
			{
				uint8_t coverage = bitmap[j * g->width + i];
				if (coverage)
					blend_pixel(c, x + g->left + i, y + j, color, coverage);
			}
		x += g->advance;
	}
}

int ui_text(UiCanvas *c, UiFont font, int x, int y, UiAlign align, UiColor color, const char *text)
{
	int width = ui_text_width(font, text);
	if (align == UI_ALIGN_CENTRE)
		x -= width / 2;
	else if (align == UI_ALIGN_RIGHT)
		x -= width;
	int shadow = font >= UI_FONT_HEADING ? 3 : 2;
	draw_run(c, &ui_fonts[font], x + shadow, y + shadow, (UiColor){0, 0, 0, (uint8_t)(color.a * 3 / 4)},
			 text);
	draw_run(c, &ui_fonts[font], x, y, color, text);
	return width;
}

int ui_text_wrapped(UiCanvas *c, UiFont font, int x, int y, int max_width, UiColor color,
					const char *text)
{
	char line[512];
	size_t length = 0;
	const char *p = text;
	while (*p)
	{
		/* Next word, including its trailing spaces. */
		const char *end = p;
		while (*end && *end != ' ' && *end != '\n')
			++end;
		char candidate[512];
		size_t word = (size_t)(end - p);
		if (length + word + 1 >= sizeof(line))
			break;
		memcpy(candidate, line, length);
		memcpy(candidate + length, p, word);
		candidate[length + word] = '\0';
		if (length && ui_text_width(font, candidate) > max_width)
		{
			line[length] = '\0';
			ui_text(c, font, x, y, UI_ALIGN_LEFT, color, line);
			y += ui_line_height(font);
			length = 0;
			continue; /* retry this word on the fresh line */
		}
		memcpy(line, candidate, length + word);
		length += word;
		p = end;
		if (*p == '\n')
		{
			line[length] = '\0';
			ui_text(c, font, x, y, UI_ALIGN_LEFT, color, line);
			y += ui_line_height(font);
			length = 0;
			++p;
		}
		else if (*p == ' ')
		{
			line[length++] = ' ';
			++p;
		}
	}
	if (length)
	{
		line[length] = '\0';
		ui_text(c, font, x, y, UI_ALIGN_LEFT, color, line);
		y += ui_line_height(font);
	}
	return y;
}

size_t ui_text_reveal(UiCanvas *c, UiFont font, int x, int y, int width, unsigned max_lines,
					  UiColor color, const char *text, size_t visible)
{
	size_t start = 0, length = strlen(text);
	for (unsigned row = 0; row < max_lines && start < length; ++row)
	{
		size_t end = start, space = start;
		int pixels = 0;
		while (end < length && text[end] != '\n')
		{
			int advance = glyph(&ui_fonts[font], text[end])->advance;
			if (pixels + advance > width && end > start)
				break;
			pixels += advance;
			if (text[end] == ' ')
				space = end;
			++end;
		}
		if (end < length && text[end] != '\n' && space > start)
			end = space;
		if (visible > start)
		{
			char line[512];
			size_t n = (visible < end ? visible : end) - start;
			if (n >= sizeof(line))
				n = sizeof(line) - 1;
			memcpy(line, text + start, n);
			line[n] = 0;
			ui_text(c, font, x, y + (int)row * ui_line_height(font), UI_ALIGN_LEFT, color, line);
		}
		start = end;
		while (text[start] == ' ')
			++start;
		if (text[start] == '\n')
			++start;
	}
	return start;
}

/* Point-in-star by the even-odd rule over its ten-vertex outline, 4x4
 * supersampled so the edges are smooth at menu sizes. */
static bool inside_star(const float *vx, const float *vy, float px, float py)
{
	bool inside = false;
	for (int i = 0, j = 9; i < 10; j = i++)
		if ((vy[i] > py) != (vy[j] > py) &&
			px < (vx[j] - vx[i]) * (py - vy[i]) / (vy[j] - vy[i]) + vx[i])
			inside = !inside;
	return inside;
}

void ui_star(UiCanvas *c, float cx, float cy, float radius, UiColor color, bool filled)
{
	float vx[10], vy[10];
	for (int i = 0; i < 10; ++i)
	{
		float r = (i & 1) ? radius * 0.45f : radius;
		float angle = -1.5707963f + (float)i * 0.6283185f;
		vx[i] = cx + cosf(angle) * r;
		vy[i] = cy + sinf(angle) * r;
	}
	UiColor fill = filled ? color : (UiColor){40, 36, 30, 200};
	for (int y = (int)(cy - radius) - 1; y <= (int)(cy + radius) + 1; ++y)
		for (int x = (int)(cx - radius) - 1; x <= (int)(cx + radius) + 1; ++x)
		{
			uint32_t hits = 0;
			for (int sy = 0; sy < 4; ++sy)
				for (int sx = 0; sx < 4; ++sx)
					hits += inside_star(vx, vy, (float)x + (sx + 0.5f) / 4.0f,
										(float)y + (sy + 0.5f) / 4.0f);
			if (hits)
				blend_pixel(c, x, y, fill, hits * 255u / 16u);
		}
}
