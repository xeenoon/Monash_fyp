#pragma once

#include "dungeon_level.h"

/* The grid is deliberately an input adapter. Successful compilation leaves no
 * row/column representation in DungeonLevel, so render and collision code can
 * also consume future polygonal or hand-authored level frontends. */
bool dungeon_grid_compile_text(const char *text, float cell_size, DungeonLevel *out,
							   DungeonLevelError *error);
bool dungeon_grid_compile_file(const char *path, float cell_size, DungeonLevel *out,
							   DungeonLevelError *error);

