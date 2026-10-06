#include "dungeon_level.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define DUNGEON_COLLIDER_EPSILON_M 0.4f /* player-scale collision fidelity */
#define DUNGEON_OCCLUDER_EPSILON_M 1.0f /* coarser: only feeds the fixed-size blocker array */

#define DUNGEON_CACHE_FORMAT_VERSION 2u
#define DUNGEON_CACHE_HEADER_SIZE 32u
#define DUNGEON_CACHE_MAX_FIELD_DIMENSION 4096u
#define DUNGEON_CACHE_MAX_FIELD_SAMPLES (16u * 1024u * 1024u)
#define DUNGEON_CACHE_MAX_GEOMETRY_ITEMS (32u * 1024u * 1024u)
#define DUNGEON_CACHE_MAX_LEVEL_ITEMS (1024u * 1024u)

static const uint8_t DUNGEON_CACHE_MAGIC[8] = {'D', 'N', 'G', 'N', 'C', 'A', 'C', 'H'};

typedef struct
{
	FILE *file;
	uint32_t hash;
	bool ok;
} CacheWriter;

typedef struct
{
	FILE *file;
	uint32_t hash;
	bool ok;
} CacheReader;

static uint32_t cache_u32(const uint8_t *p)
{
	return (uint32_t)p[0] | (uint32_t)p[1] << 8u | (uint32_t)p[2] << 16u |
		   (uint32_t)p[3] << 24u;
}

static void cache_store_u32(uint8_t *p, uint32_t value)
{
	p[0] = (uint8_t)value;
	p[1] = (uint8_t)(value >> 8u);
	p[2] = (uint8_t)(value >> 16u);
	p[3] = (uint8_t)(value >> 24u);
}

static uint32_t cache_hash(uint32_t hash, const uint8_t *bytes, size_t size)
{
	for (size_t i = 0; i < size; ++i)
	{
		hash ^= bytes[i];
		hash *= 16777619u;
	}
	return hash;
}

static bool cache_write_bytes(CacheWriter *writer, const void *data, size_t size)
{
	if (!writer->ok || fwrite(data, 1u, size, writer->file) != size)
	{
		writer->ok = false;
		return false;
	}
	writer->hash = cache_hash(writer->hash, data, size);
	return true;
}

static bool cache_write_u32(CacheWriter *writer, uint32_t value)
{
	uint8_t bytes[4];
	cache_store_u32(bytes, value);
	return cache_write_bytes(writer, bytes, sizeof(bytes));
}

static bool cache_write_float(CacheWriter *writer, float value)
{
	uint32_t bits = 0;
	memcpy(&bits, &value, sizeof(bits));
	return cache_write_u32(writer, bits);
}

static bool cache_write_point(CacheWriter *writer, DungeonPoint point)
{
	return cache_write_float(writer, point.x) && cache_write_float(writer, point.z);
}

static bool cache_write_segment(CacheWriter *writer, DungeonSegment segment)
{
	return cache_write_point(writer, segment.a) && cache_write_point(writer, segment.b);
}

static bool cache_write_triangle_mesh(CacheWriter *writer, const DungeonTriangleMesh *mesh)
{
	if (!cache_write_u32(writer, mesh->vertex_count) || !cache_write_u32(writer, mesh->index_count))
		return false;
	for (uint32_t i = 0; i < mesh->vertex_count; ++i)
		if (!cache_write_point(writer, mesh->positions[i]))
			return false;
	for (uint32_t i = 0; i < mesh->index_count; ++i)
		if (!cache_write_u32(writer, mesh->indices[i]))
			return false;
	return true;
}

static bool cache_write_level(CacheWriter *writer, const DungeonLevel *level)
{
	if (!cache_write_u32(writer, level->field.width) ||
		!cache_write_u32(writer, level->field.height) ||
		!cache_write_float(writer, level->field.cell_size) ||
		!cache_write_point(writer, level->field.origin) ||
		!cache_write_point(writer, level->spawn) || !cache_write_point(writer, level->exit) ||
		!cache_write_float(writer, level->floor_y) ||
		!cache_write_float(writer, level->wall_height) ||
		!cache_write_u32(writer, level->has_monk_spawn) ||
		!cache_write_point(writer, level->monk_spawn))
		return false;
	size_t field_count = (size_t)level->field.width * level->field.height;
	for (size_t i = 0; i < field_count; ++i)
		if (!cache_write_float(writer, level->field.values[i]))
			return false;

	if (!cache_write_u32(writer, level->contours.loop_count))
		return false;
	for (uint32_t i = 0; i < level->contours.loop_count; ++i)
	{
		const DungeonContourLoop *loop = &level->contours.loops[i];
		if (!cache_write_u32(writer, loop->point_count))
			return false;
		for (uint32_t p = 0; p < loop->point_count; ++p)
			if (!cache_write_point(writer, loop->points[p]))
				return false;
	}
	if (!cache_write_triangle_mesh(writer, &level->floor_triangles) ||
		!cache_write_triangle_mesh(writer, &level->plateau_triangles) ||
		!cache_write_u32(writer, level->collider_count))
		return false;
	for (uint32_t i = 0; i < level->collider_count; ++i)
		if (!cache_write_u32(writer, (uint32_t)level->colliders[i].type) ||
			!cache_write_segment(writer, level->colliders[i].segment))
			return false;
	if (!cache_write_u32(writer, level->occluder_count))
		return false;
	for (uint32_t i = 0; i < level->occluder_count; ++i)
		if (!cache_write_segment(writer, level->occluders[i]))
			return false;
	if (!cache_write_u32(writer, level->puddle_count))
		return false;
	for (uint32_t i = 0; i < level->puddle_count; ++i)
		if (!cache_write_point(writer, level->puddles[i].center) ||
			!cache_write_float(writer, level->puddles[i].radius))
			return false;
	if (!cache_write_u32(writer, level->door_count))
		return false;
	for (uint32_t i = 0; i < level->door_count; ++i)
	{
		const DungeonDoorway *door = &level->doors[i];
		if (!cache_write_point(writer, door->center) || !cache_write_float(writer, door->yaw) ||
			!cache_write_float(writer, door->half_width) ||
			!cache_write_segment(writer, door->blocker) ||
			!cache_write_u32(writer, (uint32_t)door->lock) ||
			!cache_write_u32(writer, door->seed))
			return false;
	}
	return writer->ok;
}

static bool cache_read_bytes(CacheReader *reader, void *data, size_t size)
{
	if (!reader->ok || fread(data, 1u, size, reader->file) != size)
	{
		reader->ok = false;
		return false;
	}
	reader->hash = cache_hash(reader->hash, data, size);
	return true;
}

static bool cache_read_u32(CacheReader *reader, uint32_t *value)
{
	uint8_t bytes[4];
	if (!cache_read_bytes(reader, bytes, sizeof(bytes)))
		return false;
	*value = cache_u32(bytes);
	return true;
}

static bool cache_read_float(CacheReader *reader, float *value)
{
	uint32_t bits = 0;
	if (!cache_read_u32(reader, &bits))
		return false;
	memcpy(value, &bits, sizeof(bits));
	return isfinite(*value);
}

static bool cache_read_point(CacheReader *reader, DungeonPoint *point)
{
	return cache_read_float(reader, &point->x) && cache_read_float(reader, &point->z);
}

static bool cache_read_segment(CacheReader *reader, DungeonSegment *segment)
{
	return cache_read_point(reader, &segment->a) && cache_read_point(reader, &segment->b);
}

static void *cache_allocate(CacheReader *reader, uint32_t count, size_t item_size,
							uint32_t maximum)
{
	if (!count)
		return NULL;
	if (count > maximum || item_size > SIZE_MAX / count)
	{
		reader->ok = false;
		return NULL;
	}
	void *allocation = calloc(count, item_size);
	if (!allocation)
		reader->ok = false;
	return allocation;
}

static bool cache_read_triangle_mesh(CacheReader *reader, DungeonTriangleMesh *mesh)
{
	if (!cache_read_u32(reader, &mesh->vertex_count) ||
		!cache_read_u32(reader, &mesh->index_count))
		return false;
	mesh->positions = cache_allocate(reader, mesh->vertex_count, sizeof(*mesh->positions),
									   DUNGEON_CACHE_MAX_GEOMETRY_ITEMS);
	mesh->indices = cache_allocate(reader, mesh->index_count, sizeof(*mesh->indices),
								 DUNGEON_CACHE_MAX_GEOMETRY_ITEMS);
	if ((mesh->vertex_count && !mesh->positions) || (mesh->index_count && !mesh->indices))
		return false;
	for (uint32_t i = 0; i < mesh->vertex_count; ++i)
		if (!cache_read_point(reader, &mesh->positions[i]))
			return false;
	for (uint32_t i = 0; i < mesh->index_count; ++i)
		if (!cache_read_u32(reader, &mesh->indices[i]) || mesh->indices[i] >= mesh->vertex_count)
		{
			reader->ok = false;
			return false;
		}
	return true;
}

static bool cache_read_level(CacheReader *reader, DungeonLevel *level)
{
	uint32_t has_monk = 0;
	if (!cache_read_u32(reader, &level->field.width) ||
		!cache_read_u32(reader, &level->field.height) ||
		!cache_read_float(reader, &level->field.cell_size) ||
		!cache_read_point(reader, &level->field.origin) ||
		!cache_read_point(reader, &level->spawn) || !cache_read_point(reader, &level->exit) ||
		!cache_read_float(reader, &level->floor_y) ||
		!cache_read_float(reader, &level->wall_height) || !cache_read_u32(reader, &has_monk) ||
		has_monk > 1 || !cache_read_point(reader, &level->monk_spawn))
		return false;
	level->has_monk_spawn = has_monk != 0;
	if (level->field.width < 2u || level->field.height < 2u ||
		level->field.width > DUNGEON_CACHE_MAX_FIELD_DIMENSION ||
		level->field.height > DUNGEON_CACHE_MAX_FIELD_DIMENSION ||
		!(level->field.cell_size > 0.0f) || !(level->wall_height > 0.0f))
	{
		reader->ok = false;
		return false;
	}
	size_t field_count = (size_t)level->field.width * level->field.height;
	if (field_count > DUNGEON_CACHE_MAX_FIELD_SAMPLES)
	{
		reader->ok = false;
		return false;
	}
	level->field.values = cache_allocate(reader, (uint32_t)field_count,
									   sizeof(*level->field.values),
									   DUNGEON_CACHE_MAX_FIELD_SAMPLES);
	if (!level->field.values)
		return false;
	for (size_t i = 0; i < field_count; ++i)
		if (!cache_read_float(reader, &level->field.values[i]))
			return false;

	if (!cache_read_u32(reader, &level->contours.loop_count) ||
		level->contours.loop_count > DUNGEON_CACHE_MAX_LEVEL_ITEMS)
	{
		reader->ok = false;
		return false;
	}
	level->contours.loops = cache_allocate(reader, level->contours.loop_count,
										 sizeof(*level->contours.loops),
										 DUNGEON_CACHE_MAX_LEVEL_ITEMS);
	uint64_t contour_points = 0;
	for (uint32_t i = 0; reader->ok && i < level->contours.loop_count; ++i)
	{
		DungeonContourLoop *loop = &level->contours.loops[i];
		if (!cache_read_u32(reader, &loop->point_count))
			return false;
		contour_points += loop->point_count;
		if (loop->point_count < 3u || contour_points > DUNGEON_CACHE_MAX_GEOMETRY_ITEMS)
		{
			reader->ok = false;
			return false;
		}
		loop->points = cache_allocate(reader, loop->point_count, sizeof(*loop->points),
									  DUNGEON_CACHE_MAX_GEOMETRY_ITEMS);
		if (!loop->points)
			return false;
		for (uint32_t p = 0; p < loop->point_count; ++p)
			if (!cache_read_point(reader, &loop->points[p]))
				return false;
	}
	if (!cache_read_triangle_mesh(reader, &level->floor_triangles) ||
		!cache_read_triangle_mesh(reader, &level->plateau_triangles))
		return false;

	if (!cache_read_u32(reader, &level->collider_count) ||
		level->collider_count > DUNGEON_CACHE_MAX_LEVEL_ITEMS)
		return reader->ok = false;
	level->colliders = cache_allocate(reader, level->collider_count, sizeof(*level->colliders),
									DUNGEON_CACHE_MAX_LEVEL_ITEMS);
	for (uint32_t i = 0; reader->ok && i < level->collider_count; ++i)
	{
		uint32_t type = 0;
		if (!cache_read_u32(reader, &type) || type != DUNGEON_COLLIDER_SEGMENT ||
			!cache_read_segment(reader, &level->colliders[i].segment))
			return reader->ok = false;
		level->colliders[i].type = (DungeonColliderType)type;
	}
	if (!cache_read_u32(reader, &level->occluder_count) ||
		level->occluder_count > DUNGEON_CACHE_MAX_LEVEL_ITEMS)
		return reader->ok = false;
	level->occluders = cache_allocate(reader, level->occluder_count, sizeof(*level->occluders),
									DUNGEON_CACHE_MAX_LEVEL_ITEMS);
	for (uint32_t i = 0; reader->ok && i < level->occluder_count; ++i)
		if (!cache_read_segment(reader, &level->occluders[i]))
			return false;
	if (!cache_read_u32(reader, &level->puddle_count) ||
		level->puddle_count > DUNGEON_CACHE_MAX_LEVEL_ITEMS)
		return reader->ok = false;
	level->puddles = cache_allocate(reader, level->puddle_count, sizeof(*level->puddles),
								  DUNGEON_CACHE_MAX_LEVEL_ITEMS);
	for (uint32_t i = 0; reader->ok && i < level->puddle_count; ++i)
		if (!cache_read_point(reader, &level->puddles[i].center) ||
			!cache_read_float(reader, &level->puddles[i].radius) ||
			!(level->puddles[i].radius > 0.0f))
			return reader->ok = false;
	if (!cache_read_u32(reader, &level->door_count) ||
		level->door_count > DUNGEON_CACHE_MAX_LEVEL_ITEMS)
		return reader->ok = false;
	level->doors = cache_allocate(reader, level->door_count, sizeof(*level->doors),
								DUNGEON_CACHE_MAX_LEVEL_ITEMS);
	for (uint32_t i = 0; reader->ok && i < level->door_count; ++i)
	{
		DungeonDoorway *door = &level->doors[i];
		uint32_t lock = 0;
		if (!cache_read_point(reader, &door->center) || !cache_read_float(reader, &door->yaw) ||
			!cache_read_float(reader, &door->half_width) || !(door->half_width > 0.0f) ||
			!cache_read_segment(reader, &door->blocker) || !cache_read_u32(reader, &lock) ||
			lock > DUNGEON_LOCK_PRISM || !cache_read_u32(reader, &door->seed))
			return reader->ok = false;
		door->lock = (DungeonLockKind)lock;
	}
	return reader->ok;
}

static DungeonLevelCacheResult cache_read_header(FILE *file, uint32_t dungeon_id,
										 uint32_t content_version, uint32_t *seed,
										 uint32_t *checksum)
{
	uint8_t header[DUNGEON_CACHE_HEADER_SIZE];
	if (fread(header, 1u, sizeof(header), file) != sizeof(header) ||
		memcmp(header, DUNGEON_CACHE_MAGIC, sizeof(DUNGEON_CACHE_MAGIC)) != 0)
		return DUNGEON_LEVEL_CACHE_INVALID;
	if (cache_u32(header + 8u) != DUNGEON_CACHE_FORMAT_VERSION ||
		cache_u32(header + 12u) != dungeon_id || cache_u32(header + 20u) != content_version)
		return DUNGEON_LEVEL_CACHE_STALE;
	*seed = cache_u32(header + 16u);
	if (checksum)
		*checksum = cache_u32(header + 24u);
	return DUNGEON_LEVEL_CACHE_LOADED;
}

DungeonLevelCacheResult dungeon_level_cache_probe(const char *path, uint32_t dungeon_id,
										   uint32_t content_version, uint32_t *out_seed)
{
	if (!path || !out_seed)
		return DUNGEON_LEVEL_CACHE_IO_ERROR;
	FILE *file = fopen(path, "rb");
	if (!file)
		return errno == ENOENT ? DUNGEON_LEVEL_CACHE_MISSING : DUNGEON_LEVEL_CACHE_IO_ERROR;
	uint32_t seed = 0;
	DungeonLevelCacheResult result =
		cache_read_header(file, dungeon_id, content_version, &seed, NULL);
	if (fclose(file) != 0 && result == DUNGEON_LEVEL_CACHE_LOADED)
		result = DUNGEON_LEVEL_CACHE_IO_ERROR;
	if (result == DUNGEON_LEVEL_CACHE_LOADED)
		*out_seed = seed;
	return result;
}

DungeonLevelCacheResult dungeon_level_cache_load(const char *path, uint32_t dungeon_id,
										  uint32_t seed, uint32_t content_version,
										  DungeonLevel *out)
{
	if (!path || !out)
		return DUNGEON_LEVEL_CACHE_IO_ERROR;
	*out = (DungeonLevel){0};
	FILE *file = fopen(path, "rb");
	if (!file)
		return errno == ENOENT ? DUNGEON_LEVEL_CACHE_MISSING : DUNGEON_LEVEL_CACHE_IO_ERROR;
	uint32_t stored_seed = 0, checksum = 0;
	DungeonLevelCacheResult result =
		cache_read_header(file, dungeon_id, content_version, &stored_seed, &checksum);
	if (result != DUNGEON_LEVEL_CACHE_LOADED || stored_seed != seed)
	{
		fclose(file);
		return result == DUNGEON_LEVEL_CACHE_LOADED ? DUNGEON_LEVEL_CACHE_STALE : result;
	}
	CacheReader reader = {.file = file, .hash = 2166136261u, .ok = true};
	bool loaded = cache_read_level(&reader, out);
	int tail = loaded ? fgetc(file) : 0;
	bool valid = loaded && reader.ok && tail == EOF && !ferror(file) && reader.hash == checksum;
	if (fclose(file) != 0)
		valid = false;
	if (!valid)
	{
		dungeon_level_destroy(out);
		return DUNGEON_LEVEL_CACHE_INVALID;
	}
	return DUNGEON_LEVEL_CACHE_LOADED;
}

bool dungeon_level_cache_save(const char *path, uint32_t dungeon_id, uint32_t seed,
							  uint32_t content_version, const DungeonLevel *level)
{
	if (!path || !level || !level->field.values)
		return false;
	size_t path_length = strlen(path);
	char *temporary = malloc(path_length + 40u);
	if (!temporary)
		return false;
	snprintf(temporary, path_length + 40u, "%s.tmp.%ld", path, (long)getpid());
	FILE *file = fopen(temporary, "wb");
	if (!file)
	{
		free(temporary);
		return false;
	}
	uint8_t header[DUNGEON_CACHE_HEADER_SIZE] = {0};
	memcpy(header, DUNGEON_CACHE_MAGIC, sizeof(DUNGEON_CACHE_MAGIC));
	cache_store_u32(header + 8u, DUNGEON_CACHE_FORMAT_VERSION);
	cache_store_u32(header + 12u, dungeon_id);
	cache_store_u32(header + 16u, seed);
	cache_store_u32(header + 20u, content_version);
	CacheWriter writer = {.file = file, .hash = 2166136261u, .ok = true};
	bool ok = fwrite(header, 1u, sizeof(header), file) == sizeof(header) &&
			  cache_write_level(&writer, level) && fflush(file) == 0;
	if (ok)
	{
		uint8_t checksum[4];
		cache_store_u32(checksum, writer.hash);
		ok = fseek(file, 24L, SEEK_SET) == 0 &&
			 fwrite(checksum, 1u, sizeof(checksum), file) == sizeof(checksum) && fflush(file) == 0;
	}
	if (fclose(file) != 0)
		ok = false;
	if (ok)
		ok = rename(temporary, path) == 0;
	if (!ok)
		remove(temporary);
	free(temporary);
	return ok;
}

static bool fail(DungeonLevelError *error, const char *format, ...)
{
	if (error)
	{
		error->line = error->column = 0;
		va_list args;
		va_start(args, format);
		vsnprintf(error->message, sizeof(error->message), format, args);
		va_end(args);
	}
	return false;
}

/* Simplifies every loop to `epsilon` and flattens the results into one flat
 * segment array (each loop of N points becomes N wrap-around segments). */
static bool build_segments(const DungeonContourSet *contours, float epsilon,
						   DungeonSegment **out_segments, uint32_t *out_count)
{
	*out_segments = NULL;
	*out_count = 0;
	if (!contours->loop_count)
		return true;
	DungeonContourLoop *simplified = calloc(contours->loop_count, sizeof(*simplified));
	if (!simplified)
		return false;
	uint32_t total_points = 0;
	for (uint32_t i = 0; i < contours->loop_count; ++i)
	{
		if (!dungeon_contour_simplify(&contours->loops[i], epsilon, &simplified[i]))
		{
			for (uint32_t k = 0; k < i; ++k)
				dungeon_contour_loop_destroy(&simplified[k]);
			free(simplified);
			return false;
		}
		total_points += simplified[i].point_count;
	}
	DungeonSegment *segments = total_points ? malloc(total_points * sizeof(*segments)) : NULL;
	if (total_points && !segments)
	{
		for (uint32_t i = 0; i < contours->loop_count; ++i)
			dungeon_contour_loop_destroy(&simplified[i]);
		free(simplified);
		return false;
	}
	uint32_t write = 0;
	for (uint32_t i = 0; i < contours->loop_count; ++i)
	{
		DungeonContourLoop *loop = &simplified[i];
		for (uint32_t p = 0; p < loop->point_count; ++p)
			segments[write++] =
				(DungeonSegment){loop->points[p], loop->points[(p + 1u) % loop->point_count]};
		dungeon_contour_loop_destroy(loop);
	}
	free(simplified);
	*out_segments = segments;
	*out_count = total_points;
	return true;
}

bool dungeon_level_compile_field(DungeonField *field, DungeonPoint spawn, DungeonPoint exit,
								 const DungeonPuddle *puddles, uint32_t puddle_count,
								 const DungeonDoorway *doors, uint32_t door_count, float floor_y,
								 float wall_height, DungeonLevel *out, DungeonLevelError *error)
{
	if (!field || !out)
		return fail(error, "level and field are required");
	*out = (DungeonLevel){0};
	if (error)
		*error = (DungeonLevelError){0};

	if (!dungeon_contour_extract(field, 0.5f, &out->contours))
	{
		dungeon_field_destroy(field);
		return fail(error, "contour extraction failed");
	}
	if (!dungeon_contour_triangulate_region(field, 0.5f, true, &out->floor_triangles) ||
		!dungeon_contour_triangulate_region(field, 0.5f, false, &out->plateau_triangles))
	{
		dungeon_field_destroy(field);
		dungeon_level_destroy(out);
		return fail(error, "region triangulation failed");
	}

	DungeonSegment *collider_segments = NULL;
	uint32_t collider_seg_count = 0;
	if (!build_segments(&out->contours, DUNGEON_COLLIDER_EPSILON_M, &collider_segments,
					   &collider_seg_count))
	{
		dungeon_field_destroy(field);
		dungeon_level_destroy(out);
		return fail(error, "out of memory building colliders");
	}
	if (collider_seg_count)
	{
		out->colliders = malloc(collider_seg_count * sizeof(*out->colliders));
		if (!out->colliders)
		{
			free(collider_segments);
			dungeon_field_destroy(field);
			dungeon_level_destroy(out);
			return fail(error, "out of memory building colliders");
		}
		for (uint32_t i = 0; i < collider_seg_count; ++i)
			out->colliders[i] =
				(DungeonCollider){.type = DUNGEON_COLLIDER_SEGMENT, .segment = collider_segments[i]};
		out->collider_count = collider_seg_count;
	}
	free(collider_segments);

	if (!build_segments(&out->contours, DUNGEON_OCCLUDER_EPSILON_M, &out->occluders,
					   &out->occluder_count))
	{
		dungeon_field_destroy(field);
		dungeon_level_destroy(out);
		return fail(error, "out of memory building light occluders");
	}

	if (puddle_count)
	{
		out->puddles = malloc(puddle_count * sizeof(*out->puddles));
		if (!out->puddles)
		{
			dungeon_field_destroy(field);
			dungeon_level_destroy(out);
			return fail(error, "out of memory copying puddles");
		}
		memcpy(out->puddles, puddles, puddle_count * sizeof(*out->puddles));
		out->puddle_count = puddle_count;
	}

	if (door_count)
	{
		out->doors = malloc(door_count * sizeof(*out->doors));
		if (!out->doors)
		{
			dungeon_field_destroy(field);
			dungeon_level_destroy(out);
			return fail(error, "out of memory copying doorways");
		}
		memcpy(out->doors, doors, door_count * sizeof(*out->doors));
		out->door_count = door_count;
	}

	out->field = *field;
	*field = (DungeonField){0}; /* ownership moved; caller's destroy becomes a no-op */
	out->spawn = spawn;
	out->exit = exit;
	out->floor_y = floor_y;
	out->wall_height = wall_height;
	return true;
}

void dungeon_level_destroy(DungeonLevel *level)
{
	if (!level)
		return;
	dungeon_field_destroy(&level->field);
	dungeon_contour_destroy(&level->contours);
	dungeon_triangle_mesh_destroy(&level->floor_triangles);
	dungeon_triangle_mesh_destroy(&level->plateau_triangles);
	free(level->colliders);
	free(level->occluders);
	free(level->puddles);
	free(level->doors);
	*level = (DungeonLevel){0};
}

void dungeon_level_print(const DungeonLevel *level)
{
	if (!level)
		return;
	printf("=== DUNGEON LAYOUT ===\n");
	printf("Floor Y: %.2f, Wall Height: %.2f\n", level->floor_y, level->wall_height);
	printf("Spawn: (%.2f, %.2f)\n", level->spawn.x, level->spawn.z);
	printf("Exit: (%.2f, %.2f)\n", level->exit.x, level->exit.z);
	printf("\nPuddles: %u\n", level->puddle_count);
	for (uint32_t i = 0; i < level->puddle_count; ++i)
	{
		printf("  %u. (%.2f, %.2f) radius=%.2f\n", i, level->puddles[i].center.x,
			   level->puddles[i].center.z, level->puddles[i].radius);
	}
	printf("\nDoors: %u\n", level->door_count);
	const char *lock_names[] = {"NONE", "PIN_TUMBLER", "SAFE_PINS", "PRISM"};
	for (uint32_t i = 0; i < level->door_count; ++i)
	{
		const DungeonDoorway *door = &level->doors[i];
		float yaw_deg = door->yaw * 180.0f / 3.14159265f;
		const char *lock_name =
			door->lock < 4 ? lock_names[door->lock] : "UNKNOWN";
		printf("  %u. (%.2f, %.2f) yaw=%.1f° half_width=%.2f lock=%s seed=%u\n", i,
			   door->center.x, door->center.z, yaw_deg, door->half_width, lock_name, door->seed);
	}
	printf("======================\n");
}
