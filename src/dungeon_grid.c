#include "dungeon_grid.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
	char *cells;
	size_t width, height;
} Grid;

static bool fail(DungeonLevelError *error, size_t line, size_t column, const char *format, ...)
{
	if (error)
	{
		error->line = line;
		error->column = column;
		va_list args;
		va_start(args, format);
		vsnprintf(error->message, sizeof(error->message), format, args);
		va_end(args);
	}
	return false;
}

static DungeonRect cell_rect(size_t x, size_t z, size_t width, size_t height, float cell_size)
{
	float origin_x = -(float)width * cell_size * 0.5f;
	float origin_z = -(float)height * cell_size * 0.5f;
	return (DungeonRect){{origin_x + (float)x * cell_size, origin_z + (float)z * cell_size},
						 {origin_x + (float)(x + 1) * cell_size,
						  origin_z + (float)(z + 1) * cell_size}};
}

static bool walkable(char cell)
{
	return cell == '.' || cell == 'S' || cell == 'E';
}

static void grid_destroy(Grid *grid)
{
	free(grid->cells);
	*grid = (Grid){0};
}

static bool parse_grid(const char *text, Grid *grid, DungeonLevelError *error)
{
	if (!text || !*text)
		return fail(error, 0, 0, "map is empty");
	size_t width = 0, height = 0;
	const char *line = text;
	while (*line)
	{
		const char *end = strchr(line, '\n');
		size_t length = end ? (size_t)(end - line) : strlen(line);
		if (length && line[length - 1] == '\r')
			--length;
		if (!length)
			return fail(error, height + 1, 1, "empty rows are not allowed");
		if (!width)
			width = length;
		else if (length != width)
			return fail(error, height + 1, length + 1, "row has length %zu; expected %zu", length,
						width);
		++height;
		if (!end)
			break;
		line = end + 1;
		if (!*line)
			break;
	}
	if (width < 3 || height < 3)
		return fail(error, 1, 1, "map must be at least 3 by 3");
	if (width > UINT32_MAX || height > UINT32_MAX || width > SIZE_MAX / height)
		return fail(error, 0, 0, "map is too large");
	grid->cells = malloc(width * height);
	if (!grid->cells)
		return fail(error, 0, 0, "out of memory");
	grid->width = width;
	grid->height = height;
	line = text;
	for (size_t z = 0; z < height; ++z)
	{
		for (size_t x = 0; x < width; ++x)
		{
			char cell = line[x];
			if (cell != '#' && !walkable(cell))
			{
				grid_destroy(grid);
				return fail(error, z + 1, x + 1, "unknown map symbol '%c'", cell);
			}
			grid->cells[z * width + x] = cell;
		}
		const char *end = strchr(line, '\n');
		if (end)
			line = end + 1;
	}
	return true;
}

static bool validate_markers_and_reachability(const Grid *grid, size_t *spawn_index,
											  size_t *exit_index, DungeonLevelError *error)
{
	uint32_t starts = 0, exits = 0;
	for (size_t i = 0; i < grid->width * grid->height; ++i)
	{
		if (grid->cells[i] == 'S')
		{
			*spawn_index = i;
			++starts;
		}
		else if (grid->cells[i] == 'E')
		{
			*exit_index = i;
			++exits;
		}
	}
	if (starts != 1 || exits != 1)
		return fail(error, 0, 0, "map requires exactly one S and one E (found %u and %u)", starts,
					exits);
	size_t count = grid->width * grid->height;
	bool *visited = calloc(count, sizeof(*visited));
	size_t *queue = malloc(count * sizeof(*queue));
	if (!visited || !queue)
	{
		free(visited);
		free(queue);
		return fail(error, 0, 0, "out of memory");
	}
	size_t read = 0, write = 0;
	queue[write++] = *spawn_index;
	visited[*spawn_index] = true;
	while (read < write)
	{
		size_t index = queue[read++];
		size_t x = index % grid->width, z = index / grid->width;
		const int dx[4] = {1, -1, 0, 0};
		const int dz[4] = {0, 0, 1, -1};
		for (int direction = 0; direction < 4; ++direction)
		{
			long nx = (long)x + dx[direction], nz = (long)z + dz[direction];
			if (nx < 0 || nz < 0 || (size_t)nx >= grid->width || (size_t)nz >= grid->height)
				continue;
			size_t neighbour = (size_t)nz * grid->width + (size_t)nx;
			if (!visited[neighbour] && walkable(grid->cells[neighbour]))
			{
				visited[neighbour] = true;
				queue[write++] = neighbour;
			}
		}
	}
	bool reachable = visited[*exit_index];
	free(visited);
	free(queue);
	return reachable ? true : fail(error, *exit_index / grid->width + 1,
								 *exit_index % grid->width + 1, "exit is unreachable from spawn");
}

static bool compile_surfaces(const Grid *grid, float cell_size, DungeonLevel *level,
							 DungeonLevelError *error)
{
	/* Merge horizontal runs. This keeps the output generic and compact while
	 * retaining uncomplicated metre-scaled UV generation downstream. */
	level->surfaces = calloc(grid->width * grid->height, sizeof(*level->surfaces));
	if (!level->surfaces)
		return fail(error, 0, 0, "out of memory");
	for (size_t z = 0; z < grid->height; ++z)
	{
		for (size_t x = 0; x < grid->width;)
		{
			if (!walkable(grid->cells[z * grid->width + x]))
			{
				++x;
				continue;
			}
			size_t end = x + 1;
			while (end < grid->width && walkable(grid->cells[z * grid->width + end]))
				++end;
			DungeonRect first = cell_rect(x, z, grid->width, grid->height, cell_size);
			DungeonRect last = cell_rect(end - 1, z, grid->width, grid->height, cell_size);
			level->surfaces[level->surface_count++] = (DungeonSurface){
				.footprint = {first.min, last.max}, .elevation = 0.0f, .material = 0};
			x = end;
		}
	}
	return true;
}

static bool compile_solids(const Grid *grid, float cell_size, DungeonLevel *level,
						  DungeonLevelError *error)
{
	size_t count = grid->width * grid->height;
	bool *used = calloc(count, sizeof(*used));
	level->solids = calloc(count, sizeof(*level->solids));
	level->colliders = calloc(count, sizeof(*level->colliders));
	if (!used || !level->solids || !level->colliders)
	{
		free(used);
		return fail(error, 0, 0, "out of memory");
	}
	for (size_t z = 0; z < grid->height; ++z)
	{
		for (size_t x = 0; x < grid->width; ++x)
		{
			size_t index = z * grid->width + x;
			if (grid->cells[index] != '#' || used[index])
				continue;
			size_t run_width = 1;
			while (x + run_width < grid->width &&
				   grid->cells[z * grid->width + x + run_width] == '#' &&
				   !used[z * grid->width + x + run_width])
				++run_width;
			size_t run_height = 1;
			for (;;)
			{
				size_t nz = z + run_height;
				if (nz >= grid->height)
					break;
				bool full = true;
				for (size_t rx = 0; rx < run_width; ++rx)
					if (grid->cells[nz * grid->width + x + rx] != '#' ||
						used[nz * grid->width + x + rx])
						full = false;
				if (!full)
					break;
				++run_height;
			}
			for (size_t rz = 0; rz < run_height; ++rz)
				for (size_t rx = 0; rx < run_width; ++rx)
					used[(z + rz) * grid->width + x + rx] = true;
			DungeonRect first = cell_rect(x, z, grid->width, grid->height, cell_size);
			DungeonRect last =
				cell_rect(x + run_width - 1, z + run_height - 1, grid->width, grid->height,
						  cell_size);
			DungeonRect bounds = {first.min, last.max};
			level->solids[level->solid_count++] =
				(DungeonSolid){.footprint = bounds, .base_y = 0.0f, .height = 2.4f, .material = 1};
			level->colliders[level->collider_count++] =
				(DungeonCollider){.type = DUNGEON_COLLIDER_AABB, .bounds = bounds};
		}
	}
	free(used);
	return true;
}

bool dungeon_grid_compile_text(const char *text, float cell_size, DungeonLevel *out,
							   DungeonLevelError *error)
{
	if (!out)
		return fail(error, 0, 0, "output level is required");
	*out = (DungeonLevel){0};
	if (error)
		*error = (DungeonLevelError){0};
	if (!(cell_size > 0.0f))
		return fail(error, 0, 0, "cell size must be positive");
	Grid grid = {0};
	if (!parse_grid(text, &grid, error))
		return false;
	size_t spawn = 0, exit = 0;
	if (!validate_markers_and_reachability(&grid, &spawn, &exit, error))
	{
		grid_destroy(&grid);
		return false;
	}
	out->cell_size = cell_size;
	out->floor_y = 0.0f;
	DungeonRect spawn_rect = cell_rect(spawn % grid.width, spawn / grid.width, grid.width,
									grid.height, cell_size);
	DungeonRect exit_rect = cell_rect(exit % grid.width, exit / grid.width, grid.width, grid.height,
								 cell_size);
	out->spawn = (DungeonPoint){(spawn_rect.min.x + spawn_rect.max.x) * 0.5f,
							  (spawn_rect.min.z + spawn_rect.max.z) * 0.5f};
	out->exit = (DungeonPoint){(exit_rect.min.x + exit_rect.max.x) * 0.5f,
						   (exit_rect.min.z + exit_rect.max.z) * 0.5f};
	bool ok = compile_surfaces(&grid, cell_size, out, error) &&
			  compile_solids(&grid, cell_size, out, error);
	grid_destroy(&grid);
	if (!ok)
		dungeon_level_destroy(out);
	return ok;
}

bool dungeon_grid_compile_file(const char *path, float cell_size, DungeonLevel *out,
							   DungeonLevelError *error)
{
	FILE *file = fopen(path, "rb");
	if (!file)
		return fail(error, 0, 0, "could not open map: %s", path ? path : "(null)");
	if (fseek(file, 0, SEEK_END) != 0)
	{
		fclose(file);
		return fail(error, 0, 0, "could not seek map");
	}
	long size = ftell(file);
	if (size < 0 || fseek(file, 0, SEEK_SET) != 0)
	{
		fclose(file);
		return fail(error, 0, 0, "could not determine map size");
	}
	char *text = malloc((size_t)size + 1);
	if (!text)
	{
		fclose(file);
		return fail(error, 0, 0, "out of memory");
	}
	size_t read = fread(text, 1, (size_t)size, file);
	fclose(file);
	if (read != (size_t)size)
	{
		free(text);
		return fail(error, 0, 0, "could not read complete map");
	}
	text[read] = '\0';
	bool ok = dungeon_grid_compile_text(text, cell_size, out, error);
	free(text);
	return ok;
}

