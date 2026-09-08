/* Dev tool: generates one cave with dungeon_cave_generate and writes its
 * field + spawn/exit/puddles as a small binary blob for tools/preview_cave.py
 * to render. Exists so cave layouts are inspected from a PNG file, never by
 * screenshotting the running game -- see the "no desktop screenshots" and
 * "validate with dump tools" project conventions. */
#include "dungeon_cave.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
	uint32_t seed = 1u;
	const char *out_path = "cave_dump.bin";
	for (int i = 1; i < argc; ++i)
	{
		if (!strcmp(argv[i], "--seed") && i + 1 < argc)
			seed = (uint32_t)strtoul(argv[++i], NULL, 10);
		else if (!strcmp(argv[i], "--out") && i + 1 < argc)
			out_path = argv[++i];
		else
		{
			fprintf(stderr, "usage: %s [--seed N] [--out path.bin]\n", argv[0]);
			return 1;
		}
	}
	DungeonCaveParams params = dungeon_cave_default_params(seed);
	DungeonCaveResult cave = {0};
	if (!dungeon_cave_generate(&params, &cave))
	{
		fprintf(stderr, "dungeon_cave_generate failed for seed %u\n", seed);
		return 1;
	}
	FILE *file = fopen(out_path, "wb");
	if (!file)
	{
		fprintf(stderr, "could not open %s for writing\n", out_path);
		dungeon_cave_destroy(&cave);
		return 1;
	}
	fwrite(&cave.field.width, sizeof(uint32_t), 1, file);
	fwrite(&cave.field.height, sizeof(uint32_t), 1, file);
	fwrite(&cave.field.cell_size, sizeof(float), 1, file);
	fwrite(&cave.field.origin, sizeof(DungeonPoint), 1, file);
	fwrite(cave.field.values, sizeof(float), (size_t)cave.field.width * cave.field.height, file);
	fwrite(&cave.spawn, sizeof(DungeonPoint), 1, file);
	fwrite(&cave.exit, sizeof(DungeonPoint), 1, file);
	fwrite(&cave.puddle_count, sizeof(uint32_t), 1, file);
	for (uint32_t i = 0; i < cave.puddle_count; ++i)
	{
		fwrite(&cave.puddles[i].center, sizeof(DungeonPoint), 1, file);
		fwrite(&cave.puddles[i].radius, sizeof(float), 1, file);
	}
	fwrite(&cave.door_count, sizeof(uint32_t), 1, file);
	for (uint32_t i = 0; i < cave.door_count; ++i)
	{
		fwrite(&cave.doors[i].blocker.a, sizeof(DungeonPoint), 1, file);
		fwrite(&cave.doors[i].blocker.b, sizeof(DungeonPoint), 1, file);
		uint32_t lock = (uint32_t)cave.doors[i].lock;
		fwrite(&lock, sizeof(uint32_t), 1, file);
	}
	fclose(file);
	printf("wrote %s: %ux%u corners, spawn=(%.2f,%.2f) exit=(%.2f,%.2f) puddles=%u doors=%u\n",
		  out_path, cave.field.width, cave.field.height, (double)cave.spawn.x, (double)cave.spawn.z,
		  (double)cave.exit.x, (double)cave.exit.z, cave.puddle_count, cave.door_count);
	dungeon_cave_destroy(&cave);
	return 0;
}
