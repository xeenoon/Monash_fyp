#include "dungeon_level.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DUNGEON_COLLIDER_EPSILON_M 0.4f /* player-scale collision fidelity */
#define DUNGEON_OCCLUDER_EPSILON_M 1.0f /* coarser: only feeds the fixed-size blocker array */

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
