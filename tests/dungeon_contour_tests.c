#include "dungeon_contour.h"
#include "dungeon_field.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static void bilinear_sample_matches_hand_computation(void)
{
	DungeonField field = {0};
	assert(dungeon_field_create(3, 3, 1.0f, (DungeonPoint){0.0f, 0.0f}, &field));
	for (uint32_t z = 0; z < 3; ++z)
	{
		dungeon_field_set(&field, 0, z, 0.0f);
		dungeon_field_set(&field, 1, z, 1.0f);
		dungeon_field_set(&field, 2, z, 1.0f);
	}
	assert(fabsf(dungeon_field_sample(&field, (DungeonPoint){0.5f, 0.5f}) - 0.5f) < 1e-6f);
	assert(fabsf(dungeon_field_sample(&field, (DungeonPoint){1.5f, 1.0f}) - 1.0f) < 1e-6f);
	assert(dungeon_field_sample(&field, (DungeonPoint){-5.0f, -5.0f}) == 0.0f);
	dungeon_field_destroy(&field);
}

/* A single open corner surrounded by solid corners crosses exactly the four
 * edges incident to it, which must stitch into one 4-point diamond loop --
 * the smallest case that exercises stitching without the ambiguous saddle. */
static void single_open_corner_forms_one_diamond_loop(void)
{
	DungeonField field = {0};
	assert(dungeon_field_create(5, 5, 1.0f, (DungeonPoint){0.0f, 0.0f}, &field));
	dungeon_field_set(&field, 2, 2, 1.0f);
	DungeonContourSet contours = {0};
	assert(dungeon_contour_extract(&field, 0.5f, &contours));
	assert(contours.loop_count == 1u);
	assert(contours.loops[0].point_count == 4u);
	for (uint32_t i = 0; i < 4u; ++i)
	{
		DungeonPoint p = contours.loops[0].points[i];
		assert(p.x > 1.0f && p.x < 3.0f && p.z > 1.0f && p.z < 3.0f);
	}
	dungeon_contour_destroy(&contours);
	dungeon_field_destroy(&field);
}

static float loop_signed_area(const DungeonContourLoop *loop)
{
	double area2 = 0.0;
	for (uint32_t i = 0; i < loop->point_count; ++i)
	{
		DungeonPoint a = loop->points[i], b = loop->points[(i + 1u) % loop->point_count];
		area2 += (double)a.x * b.z - (double)b.x * a.z;
	}
	return (float)(area2 * 0.5);
}

static float loop_perimeter(const DungeonContourLoop *loop)
{
	float perimeter = 0.0f;
	for (uint32_t i = 0; i < loop->point_count; ++i)
	{
		DungeonPoint a = loop->points[i], b = loop->points[(i + 1u) % loop->point_count];
		float dx = b.x - a.x, dz = b.z - a.z;
		perimeter += sqrtf(dx * dx + dz * dz);
	}
	return perimeter;
}

static float mesh_area(const DungeonTriangleMesh *mesh)
{
	float area = 0.0f;
	for (uint32_t i = 0; i + 2u < mesh->index_count; i += 3u)
	{
		DungeonPoint a = mesh->positions[mesh->indices[i]];
		DungeonPoint b = mesh->positions[mesh->indices[i + 1u]];
		DungeonPoint c = mesh->positions[mesh->indices[i + 2u]];
		float cross = (b.x - a.x) * (c.z - a.z) - (c.x - a.x) * (b.z - a.z);
		area += 0.5f * fabsf(cross);
	}
	return area;
}

/* An analytic disc stamped into an otherwise-solid field: the contour and its
 * matching triangulation must both recover circle perimeter/area to within
 * grid resolution, and the loop must wind with open floor on the left
 * (positive signed area, since it's an island of floor). */
static void disc_field_recovers_circle_measurements(void)
{
	const float cell_size = 0.1f;
	const float radius = 3.0f;
	DungeonField field = {0};
	assert(dungeon_field_create(121, 121, cell_size, (DungeonPoint){-6.0f, -6.0f}, &field));
	dungeon_field_stamp_disc(&field, (DungeonPoint){0.0f, 0.0f}, radius, cell_size * 1.5f);

	DungeonContourSet contours = {0};
	assert(dungeon_contour_extract(&field, 0.5f, &contours));
	assert(contours.loop_count == 1u);
	float signed_area = loop_signed_area(&contours.loops[0]);
	float perimeter = loop_perimeter(&contours.loops[0]);
	float expected_area = (float)M_PI * radius * radius;
	float expected_perimeter = 2.0f * (float)M_PI * radius;
	assert(signed_area > 0.0f); /* CCW: open floor is inside, on the left */
	assert(fabsf(signed_area - expected_area) / expected_area < 0.08f);
	assert(fabsf(perimeter - expected_perimeter) / expected_perimeter < 0.08f);

	DungeonTriangleMesh floor = {0};
	assert(dungeon_contour_triangulate_region(&field, 0.5f, true, &floor));
	float floor_area = mesh_area(&floor);
	assert(fabsf(floor_area - signed_area) / signed_area < 0.02f);

	DungeonTriangleMesh rock = {0};
	assert(dungeon_contour_triangulate_region(&field, 0.5f, false, &rock));
	float total_area = (121.0f - 1.0f) * cell_size * (121.0f - 1.0f) * cell_size;
	assert(fabsf((floor_area + mesh_area(&rock)) - total_area) / total_area < 1e-3f);

	DungeonContourLoop simplified = {0};
	assert(dungeon_contour_simplify(&contours.loops[0], cell_size * 0.5f, &simplified));
	assert(simplified.point_count >= 3u && simplified.point_count < contours.loops[0].point_count);
	for (uint32_t i = 0; i < simplified.point_count; ++i)
	{
		bool found = false;
		for (uint32_t j = 0; j < contours.loops[0].point_count && !found; ++j)
		{
			DungeonPoint a = simplified.points[i], b = contours.loops[0].points[j];
			found = fabsf(a.x - b.x) < 1e-6f && fabsf(a.z - b.z) < 1e-6f;
		}
		assert(found); /* Douglas-Peucker only ever keeps existing points */
	}

	dungeon_contour_loop_destroy(&simplified);
	dungeon_triangle_mesh_destroy(&rock);
	dungeon_triangle_mesh_destroy(&floor);
	dungeon_contour_destroy(&contours);
	dungeon_field_destroy(&field);
}

int main(void)
{
	bilinear_sample_matches_hand_computation();
	single_open_corner_forms_one_diamond_loop();
	disc_field_recovers_circle_measurements();
	puts("dungeon contour tests passed");
	return 0;
}
