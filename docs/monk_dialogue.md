# Praying monk and dialogue

Every generated dungeon reserves a corner of its starting room for the monk.
He faces the player spawn, stays seated, blocks walking through his body, and
offers `[E] Talk to the monk` within two metres with a clear line of sight.
His position is cached with the dungeon. Older caches are regenerated because
the format and generator versions have changed. Debug ASCII maps and torch
labs do not add the monk.

Conversations use the existing software UI canvas and menu appearance. Enter,
Space, or clicking the panel reveals/advances a page. At the end of an entry,
use arrows/W/S or click to choose Say something, Keep listening, or Leave.
Escape leaves immediately, including while editing. Enter submits a reply.
The editor supports insertion, Backspace/Delete, arrows, Home/End, and at most
120 printable ASCII characters, matching the current font atlas. Unsupported
characters are ignored. It does not provide clipboard paste or IME composition.

The player gets only a short prefix of their reply on screen, followed by `--`.
After 0.32 seconds the monk moves to his next entry, regardless of the reply.
The fragment remains visible above his next speech. Empty/space-only replies
do nothing. Keep listening moves directly to the next entry.

The first three entries are gibberish, the fourth warns about the snake lady,
and the fifth begins a six-page confession. Nine additional two-page stories
follow. All ten stories are heard before the sequence cycles back to the
confession. Page advancement does not advance the story counter. Closing a
completed story advances on the next conversation; closing an unfinished story
resumes that page. Progress survives deaths, restarts, and dungeon visits while
the game is running; it is not saved across application launches.

Gameplay and the run timer freeze during dialogue; ambient torch animation and
UI timing continue. The ordinary follow camera eases to keep the monk above
the panel, keeping the player's orbit/zoom settings and returning on exit.

`dialogue.c` is a renderer-independent controller. `monk_dialogue.c` contains
the authored entries and page links. `DungeonGame` owns conversation memory,
`input.c` routes ordered edit events before gameplay bindings, and `ui_draw.c`
reveals text on fixed wrapped lines. No dialogue service or runtime AI is used.

Build and validate:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target terrain_renderer dialogue_tests monk_placement_tests dungeon_cave_tests -j6
ctest --test-dir build -R 'monk_|dungeon_cave' --output-on-failure
python3 tests/monk_dialogue_replay.py
```

The last check runs the actual game and requires its usual display, Vulkan,
and Alps data. It exercises progression, typing, timed interruption, frozen
gameplay, leaving/resuming, and reloading a cached dungeon. The game script
supports `teleport monk`, `text <remainder of command>`, `submit`,
`expect_dialogue <entry-id>`, `expect_phase <phase>`, `expect_reply <text>`,
`expect_fragment <text>`, `remember_world`, and `expect_frozen`. Script commands
are separated by semicolons; semicolons cannot be embedded in scripted replies.
Game-script runs use a fixed 1/60-second timestep for repeatable timing.
