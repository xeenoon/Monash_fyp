#pragma once

#include "dungeon_field.h"
#include "dungeon_geometry.h"

#include <stdbool.h>
#include <stdint.h>

/* A single closed polyline on the field's 0.5 isoline. Winding is chosen so
 * that walking the loop in order keeps open floor (field value >= iso) on the
 * left -- i.e. counter-clockwise around an island of open floor, clockwise
 * around an island of rock inside open floor. */
typedef struct
{
	DungeonPoint *points;
	uint32_t point_count;
} DungeonContourLoop;

typedef struct
{
	DungeonContourLoop *loops;
	uint32_t loop_count;
} DungeonContourSet;

/* Flat XZ triangle soup (Y is applied by the caller). Shares the exact same
 * per-cell crossing computation as dungeon_contour_extract, so a floor
 * triangulation's boundary always coincides with the matching contour loop --
 * no seams between collision/visual boundaries and the fill. */
typedef struct
{
	DungeonPoint *positions;
	uint32_t vertex_count;
	uint32_t *indices;
	uint32_t index_count;
} DungeonTriangleMesh;

bool dungeon_contour_extract(const DungeonField *field, float iso, DungeonContourSet *out);
void dungeon_contour_destroy(DungeonContourSet *set);

/* want_inside true triangulates the field>=iso region (floor); false
 * triangulates the field<iso region (rock plateau). */
bool dungeon_contour_triangulate_region(const DungeonField *field, float iso, bool want_inside,
										DungeonTriangleMesh *out);
void dungeon_triangle_mesh_destroy(DungeonTriangleMesh *mesh);

/* Douglas-Peucker simplification of a closed loop, writing a new loop with
 * point_count <= loop->point_count (never zero, retains at least a triangle).
 * Used to shrink the ~2000-segment raw contour down to the handful of
 * light-blocker/collider segments the renderer's fixed-size arrays can hold. */
bool dungeon_contour_simplify(const DungeonContourLoop *loop, float epsilon,
							  DungeonContourLoop *out);
void dungeon_contour_loop_destroy(DungeonContourLoop *loop);
