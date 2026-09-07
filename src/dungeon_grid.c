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

static DungeonPoint cell_center(const Grid *grid, float cell_size, size_t x, size_t z)
{
	float origin_x = -(float)grid->width * cell_size * 0.5f;
	float origin_z = -(float)grid->height * cell_size * 0.5f;
	return (DungeonPoint){origin_x + ((float)x + 0.5f) * cell_size,
						  origin_z + ((float)z + 0.5f) * cell_size};
}

/* Rasterizes the char grid into a field whose corners sit at cell CENTERS,
 * with one extra ring of always-solid corners around the outside. Placing
 * corners at centers (rather than at cell boundaries) means the crossing
 * marching squares finds between two hard 0/1 corners falls exactly halfway
 * between them -- i.e. exactly on the true shared cell edge -- so the ASCII
 * frontend still reconstructs crisp rectilinear rooms despite going through
 * the same continuous-field pipeline as the cave generator. The solid outer
 * ring guarantees a closed border, same invariant dungeon_cave keeps by
 * staying inside its margin. */
static bool rasterize(const Grid *grid, float cell_size, DungeonField *out)
{
	uint32_t width = (uint32_t)grid->width + 2u;
	uint32_t height = (uint32_t)grid->height + 2u;
	float origin_x = -(float)grid->width * cell_size * 0.5f - 0.5f * cell_size;
	float origin_z = -(float)grid->height * cell_size * 0.5f - 0.5f * cell_size;
	if (!dungeon_field_create(width, height, cell_size, (DungeonPoint){origin_x, origin_z}, out))
		return false;
	for (uint32_t fz = 0; fz < height; ++fz)
		for (uint32_t fx = 0; fx < width; ++fx)
		{
			long cx = (long)fx - 1, cz = (long)fz - 1;
			bool open = cx >= 0 && cz >= 0 && (size_t)cx < grid->width && (size_t)cz < grid->height &&
					   walkable(grid->cells[(size_t)cz * grid->width + (size_t)cx]);
			dungeon_field_set(out, fx, fz, open ? 1.0f : 0.0f);
		}
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
	DungeonPoint spawn_point = cell_center(&grid, cell_size, spawn % grid.width, spawn / grid.width);
	DungeonPoint exit_point = cell_center(&grid, cell_size, exit % grid.width, exit / grid.width);
	DungeonField field = {0};
	if (!rasterize(&grid, cell_size, &field))
	{
		grid_destroy(&grid);
		return fail(error, 0, 0, "out of memory rasterizing map");
	}
	grid_destroy(&grid);
	return dungeon_level_compile_field(&field, spawn_point, exit_point, NULL, 0u, 0.0f, 2.4f, out,
									   error);
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
