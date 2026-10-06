#pragma once
#include <stdbool.h>
#include <stddef.h>

#define DIALOGUE_REPLY_CAPACITY 121
#define DIALOGUE_MAX_PAGES 6
typedef enum
{
	TEXT_INSERT,
	TEXT_BACKSPACE,
	TEXT_DELETE,
	TEXT_LEFT,
	TEXT_RIGHT,
	TEXT_HOME,
	TEXT_END,
	TEXT_SUBMIT,
	TEXT_CANCEL
} TextEditKind;
typedef struct
{
	TextEditKind kind;
	char text[DIALOGUE_REPLY_CAPACITY];
} TextEdit;
typedef struct
{
	const char *id;
	const char *pages[DIALOGUE_MAX_PAGES];
	unsigned page_count;
} DialogueEntry;
typedef enum
{
	DIALOGUE_REVEAL,
	DIALOGUE_PAGE,
	DIALOGUE_CHOICES,
	DIALOGUE_EDIT,
	DIALOGUE_INTERRUPT
} DialoguePhase;
typedef struct
{
	const DialogueEntry *entries;
	unsigned entry_count, entry, page, cursor, repeat_from;
	DialoguePhase phase;
	bool active, started, complete;
	float elapsed, reveal;
	char reply[DIALOGUE_REPLY_CAPACITY], fragment[32];
	size_t caret;
} Dialogue;
void dialogue_init(Dialogue *d, const DialogueEntry *entries, unsigned count, unsigned repeat_from);
void dialogue_open(Dialogue *d);
void dialogue_close(Dialogue *d);
void dialogue_update(Dialogue *d, float dt);
void dialogue_confirm(Dialogue *d);
void dialogue_edit(Dialogue *d, const TextEdit *edit);
const char *dialogue_text(const Dialogue *d);
void monk_dialogue_init(Dialogue *d);
