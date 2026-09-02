#include "dungeon_level.h"

#include <stdlib.h>

void dungeon_level_destroy(DungeonLevel *level)
{
	if (!level)
		return;
	free(level->surfaces);
	free(level->solids);
	free(level->colliders);
	*level = (DungeonLevel){0};
}

