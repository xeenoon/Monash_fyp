#include "dialogue.h"
#include "input.h"
#include "ui_draw.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void content_and_progression(void)
{
	Dialogue d;
	monk_dialogue_init(&d);
	uint8_t *pixels = calloc(960 * 540, 4);
	assert(pixels);
	UiCanvas canvas = {pixels, 960, 540};
	assert(d.entry_count == 14);
	for (unsigned i = 0; i < d.entry_count; ++i)
	{
		assert(d.entries[i].page_count > 0 && d.entries[i].page_count <= DIALOGUE_MAX_PAGES);
		for (unsigned p = 0; p < d.entries[i].page_count; ++p)
		{
			const char *text = d.entries[i].pages[p];
			for (const unsigned char *c = (const unsigned char *)text; *c; ++c)
				assert(*c >= 32 && *c <= 126);
			assert(ui_text_reveal(&canvas, UI_FONT_SMALL, 62, 300, 836, 4,
								  (UiColor){255, 255, 255, 255}, text, SIZE_MAX) == strlen(text));
		}
	}
	for (unsigned i = 0; i < 4; ++i)
	{
		dialogue_open(&d);
		assert(d.entry == i);
		dialogue_confirm(&d);
		assert(d.phase == DIALOGUE_CHOICES);
		dialogue_close(&d);
	}
	dialogue_open(&d);
	assert(!strcmp(d.entries[d.entry].id, "confession"));
	dialogue_confirm(&d);
	dialogue_confirm(&d);
	assert(d.page == 1 && d.entry == 4);
	dialogue_update(&d, .1f);
	float reveal = d.reveal;
	dialogue_close(&d);
	dialogue_update(&d, 10);
	dialogue_open(&d);
	assert(d.page == 1 && d.reveal == reveal);
	for (unsigned story = 4; story < 14; ++story)
	{
		assert(d.entry == story);
		while (d.phase != DIALOGUE_CHOICES)
			dialogue_confirm(&d);
		d.cursor = 1;
		dialogue_confirm(&d);
	}
	assert(d.entry == 4); /* ten complete stories before repeating */
	free(pixels);
}
static void interruption_and_editing(void)
{
	Dialogue d;
	monk_dialogue_init(&d);
	dialogue_open(&d);
	dialogue_confirm(&d);
	dialogue_confirm(&d);
	assert(d.phase == DIALOGUE_EDIT);
	TextEdit insert = {.kind = TEXT_INSERT, .text = "wow im sorry"};
	dialogue_edit(&d, &insert);
	dialogue_edit(&d, &(TextEdit){.kind = TEXT_SUBMIT});
	assert(d.phase == DIALOGUE_INTERRUPT && !strcmp(d.fragment, "wow im so--"));
	dialogue_confirm(&d);
	assert(d.entry == 0);
	dialogue_update(&d, .31f);
	assert(d.entry == 0);
	dialogue_update(&d, .02f);
	assert(d.entry == 1 && d.phase == DIALOGUE_REVEAL);
	assert(!strcmp(d.fragment, "wow im so--"));
	dialogue_confirm(&d);
	dialogue_confirm(&d);
	dialogue_edit(&d, &(TextEdit){.kind = TEXT_INSERT, .text = "   "});
	dialogue_edit(&d, &(TextEdit){.kind = TEXT_SUBMIT});
	assert(d.phase == DIALOGUE_EDIT);
	dialogue_edit(&d, &(TextEdit){.kind = TEXT_HOME});
	dialogue_edit(&d, &(TextEdit){.kind = TEXT_INSERT, .text = "ab"});
	dialogue_edit(&d, &(TextEdit){.kind = TEXT_LEFT});
	dialogue_edit(&d, &(TextEdit){.kind = TEXT_DELETE});
	assert(!strcmp(d.reply, "a   "));
	dialogue_edit(&d, &(TextEdit){.kind = TEXT_BACKSPACE});
	assert(!strcmp(d.reply, "   "));
	dialogue_edit(&d, &(TextEdit){.kind = TEXT_INSERT, .text = "\xc3\xa9"});
	assert(!strcmp(d.reply, "   "));
	TextEdit large = {.kind = TEXT_INSERT};
	memset(large.text, 'x', 120);
	large.text[120] = 0;
	dialogue_edit(&d, &large);
	dialogue_edit(&d, &large);
	assert(strlen(d.reply) == 120);
}
static void input_isolation(void)
{
	assert(SDL_Init(SDL_INIT_VIDEO));
	SDL_Window *window = SDL_CreateWindow("dialogue input test", 100, 100, SDL_WINDOW_HIDDEN);
	assert(window);
	Input input = {0};
	input_set_mode(&input, window, INPUT_TEXT);
	const SDL_Keycode keys[] = {SDLK_R, SDLK_E, SDLK_Q, SDLK_W, SDLK_SPACE, SDLK_C, SDLK_X};
	for (unsigned i = 0; i < sizeof(keys) / sizeof(keys[0]); ++i)
	{
		SDL_Event e = {.type = SDL_EVENT_KEY_DOWN};
		e.key.key = keys[i];
		assert(SDL_PushEvent(&e));
	}
	SDL_Event text = {.type = SDL_EVENT_TEXT_INPUT};
	text.text.text = "really wow im sorry";
	assert(SDL_PushEvent(&text));
	SDL_Event enter = {.type = SDL_EVENT_KEY_DOWN};
	enter.key.key = SDLK_RETURN;
	assert(SDL_PushEvent(&enter));
	input_poll(&input, window);
	assert(!input.restart && !input.interact && !input.puzzle_confirm && !input.menu_up);
	assert(!input.dump_shader_data && !input.clear_shader_dump && !input.toggle_auto_exposure);
	assert(input.move_forward == 0 && input.edit_count == 2);
	assert(input.edits[0].kind == TEXT_INSERT && input.edits[1].kind == TEXT_SUBMIT);
	input_set_mode(&input, window, INPUT_GAME);
	SDL_DestroyWindow(window);
	SDL_Quit();
}
int main(void)
{
	content_and_progression();
	interruption_and_editing();
	input_isolation();
	puts("Dialogue progression, page layout, interruption, editing and SDL input isolation passed");
	return 0;
}
