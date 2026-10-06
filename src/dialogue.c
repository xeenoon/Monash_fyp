#include "dialogue.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

void dialogue_init(Dialogue *d, const DialogueEntry *entries, unsigned count, unsigned repeat_from)
{
	*d = (Dialogue){.entries = entries,
					.entry_count = count,
					.repeat_from = repeat_from < count ? repeat_from : 0};
}
const char *dialogue_text(const Dialogue *d)
{
	return d->entry_count ? d->entries[d->entry].pages[d->page] : "";
}
static void begin_page(Dialogue *d)
{
	d->phase = DIALOGUE_REVEAL;
	d->reveal = 0;
	d->elapsed = 0;
	d->complete = false;
}
static void next_entry(Dialogue *d)
{
	d->entry = (d->entry + 1 < d->entry_count) ? d->entry + 1 : d->repeat_from;
	d->page = 0;
	d->cursor = 0;
	d->reply[0] = 0;
	d->caret = 0;
	begin_page(d);
}
void dialogue_open(Dialogue *d)
{
	if (!d->entry_count || d->active)
		return;
	if (!d->started)
	{
		d->started = true;
		begin_page(d);
	}
	else if (d->complete)
		next_entry(d);
	d->active = true;
}
void dialogue_close(Dialogue *d) { d->active = false; }
static void finish_reveal(Dialogue *d)
{
	d->reveal = (float)strlen(dialogue_text(d));
	d->complete = d->page + 1 == d->entries[d->entry].page_count;
	d->phase = d->complete ? DIALOGUE_CHOICES : DIALOGUE_PAGE;
}
void dialogue_update(Dialogue *d, float dt)
{
	if (!d->active || !isfinite(dt) || dt <= 0)
		return;
	d->elapsed += dt;
	if (d->phase == DIALOGUE_REVEAL)
	{
		d->reveal += dt * 42;
		if (d->reveal >= strlen(dialogue_text(d)))
			finish_reveal(d);
	}
	else if (d->phase == DIALOGUE_INTERRUPT && d->elapsed >= .32f)
		next_entry(d);
}
void dialogue_confirm(Dialogue *d)
{
	if (!d->active)
		return;
	switch (d->phase)
	{
	case DIALOGUE_REVEAL:
		finish_reveal(d);
		break;
	case DIALOGUE_PAGE:
		++d->page;
		begin_page(d);
		break;
	case DIALOGUE_CHOICES:
		if (d->cursor == 0)
		{
			d->phase = DIALOGUE_EDIT;
			d->reply[0] = 0;
			d->caret = 0;
		}
		else if (d->cursor == 1)
		{
			d->fragment[0] = 0;
			next_entry(d);
		}
		else
			dialogue_close(d);
		break;
	default:
		break;
	}
}
void dialogue_edit(Dialogue *d, const TextEdit *edit)
{
	if (!d->active || d->phase != DIALOGUE_EDIT)
		return;
	size_t length = strlen(d->reply);
	switch (edit->kind)
	{
	case TEXT_INSERT:
		/* Font atlas is ASCII. Reject unsupported UTF-8 as whole sequences,
		 * rather than displaying one question mark per byte. */
		for (const unsigned char *p = (const unsigned char *)edit->text; *p; ++p)
		{
			if (*p < 32 || *p > 126 || length >= DIALOGUE_REPLY_CAPACITY - 1)
				continue;
			memmove(d->reply + d->caret + 1, d->reply + d->caret, length - d->caret + 1);
			d->reply[d->caret++] = (char)*p;
			++length;
		}
		break;
	case TEXT_BACKSPACE:
		if (d->caret)
		{
			memmove(d->reply + d->caret - 1, d->reply + d->caret, length - d->caret + 1);
			--d->caret;
		}
		break;
	case TEXT_DELETE:
		if (d->caret < length)
			memmove(d->reply + d->caret, d->reply + d->caret + 1, length - d->caret);
		break;
	case TEXT_LEFT:
		if (d->caret)
			--d->caret;
		break;
	case TEXT_RIGHT:
		if (d->caret < length)
			++d->caret;
		break;
	case TEXT_HOME:
		d->caret = 0;
		break;
	case TEXT_END:
		d->caret = length;
		break;
	case TEXT_CANCEL:
		d->phase = DIALOGUE_CHOICES;
		break;
	case TEXT_SUBMIT:
	{
		if (strspn(d->reply, " ") == length)
			break;
		size_t n = length > 1 ? length - 1 : 1;
		if (n > 9)
			n = 9;
		snprintf(d->fragment, sizeof(d->fragment), "%.*s--", (int)n, d->reply);
		d->phase = DIALOGUE_INTERRUPT;
		d->elapsed = 0;
		d->complete = false;
		break;
	}
	}
}
