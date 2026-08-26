#include "terrain_grid.h"
#include "terrain_quadtree.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* terrain_grid's GPU hooks are deliberately not part of this CPU unit test. */
void mesh_upload(struct Renderer *renderer, Mesh *mesh)
{
	(void)renderer;
	(void)mesh;
}
void mesh_destroy(struct Renderer *renderer, Mesh *mesh)
{
	(void)renderer;
	(void)mesh;
}

static void require(bool condition, const char *message)
{
	if (!condition)
	{
		fprintf(stderr, "phase4 test failed: %s\n", message);
		exit(EXIT_FAILURE);
	}
}

static TerrainTileBounds bounds(double cx, double cy, double cz, double half, float error)
{
	TerrainTileBounds result = {
		.center = {cx, cy, cz},
		.radius_m = sqrt(3.0 * half * half),
		.geometric_error_m = error,
	};
	for (uint32_t i = 0; i < 8; ++i)
	{
		result.corners[i][0] = cx + ((i & 1u) ? half : -half);
		result.corners[i][1] = cy + ((i & 2u) ? half : -half);
		result.corners[i][2] = cz + ((i & 4u) ? half : -half);
	}
	return result;
}

static TerrainQuadtreeView view_at(double z, double forward_z)
{
	return (TerrainQuadtreeView){
		.camera_world = {0.0, 0.0, z},
		.forward = {0.0, 0.0, forward_z},
		.up = {0.0, 1.0, 0.0},
		.vertical_fov_radians = 60.0 * 3.14159265358979323846 / 180.0,
		.aspect = 16.0 / 9.0,
		.near_plane_m = 0.5,
		.viewport_height_px = 600.0,
	};
}

static void make_resident(TerrainQuadtree *tree, uint32_t index, TerrainTileBounds tile_bounds)
{
	uint32_t generation = tree->nodes[index].generation;
	require(terrain_quadtree_cpu_ready(tree, index, generation, &tile_bounds, 100,
									   (void *)(uintptr_t)(index + 1u)),
			"requested -> CPU ready transition failed");
	require(terrain_quadtree_upload_begin(tree, index, generation),
			"CPU ready -> upload pending transition failed");
	require(terrain_quadtree_resident(tree, index, generation, 200),
			"upload pending -> resident transition failed");
}

static void test_grid(void)
{
	TerrainGrid grid;
	require(terrain_grid_create(&grid, 5), "could not create pooled grid");
	require(grid.mesh.vertex_count == 57, "pooled grid vertex count is wrong");
	require(grid.mesh.index_count == 192, "pooled grid index count is wrong");
	for (uint32_t i = 0; i < grid.mesh.index_count; ++i)
		require(grid.indices[i] < grid.mesh.vertex_count, "pooled grid index is out of range");
	require(grid.vertices[0].position[0] == -0.5f && grid.vertices[24].position[2] == 0.5f,
			"surface grid is not normalized");
	require(grid.vertices[26].untextured == 2.0f, "skirt extrusion marker is missing");
	terrain_grid_destroy(NULL, &grid);
}

static void test_quadtree(void)
{
	TerrainQuadtreeSettings settings = terrain_quadtree_default_settings();
	settings.split_threshold_px = 4.0f;
	settings.merge_threshold_px = 2.0f;
	settings.max_nodes = 64;
	settings.max_new_requests_per_frame = 4;
	settings.max_resident_tiles = 16;
	settings.eviction_frames = 2;

	TerrainQuadtree tree;
	require(terrain_quadtree_init(&tree, &settings), "quadtree init failed");
	require(terrain_quadtree_request_root(&tree), "root request failed");
	require(terrain_quadtree_next_request(&tree) == 0, "root priority is wrong");
	require(!terrain_quadtree_cpu_ready(&tree, 0, tree.nodes[0].generation + 1u,
										&(TerrainTileBounds){0}, 0, NULL),
			"stale generation was accepted");
	make_resident(&tree, 0, bounds(0, 0, 0, 500, 10.0f));

	TerrainQuadtreeView near = view_at(-900.0, 1.0);
	terrain_quadtree_select(&tree, &near, 1);
	require(tree.draw_count == 1 && tree.draw_nodes[0] == 0,
			"parent was not retained while children load");
	require(tree.node_count == 5, "child quad was not created atomically");
	for (uint32_t q = 0; q < 4; ++q)
	{
		uint32_t child = tree.nodes[0].children[q];
		require(tree.nodes[child].state == TERRAIN_TILE_REQUESTED, "child was not requested");
		require(terrain_tile_key_quadrant(tree.nodes[child].key) == q,
				"Rocky quadrant numbering changed");
		make_resident(
			&tree, child,
			bounds((q & 1u) ? 250.0 : -250.0, 0.0, (q & 2u) ? 250.0 : -250.0, 250.0, 1.0f));
	}
	float scale_bias[4];
	terrain_tile_parent_uv_scale_bias((TerrainTileKey){1, 1, 1}, scale_bias);
	require(scale_bias[0] == 0.5f && scale_bias[1] == 0.5f && scale_bias[2] == 0.5f &&
				scale_bias[3] == 0.5f,
			"parent fallback UV mapping is wrong");

	terrain_quadtree_select(&tree, &near, 2);
	require(tree.draw_count == 4, "complete child quad did not replace parent");

	TerrainQuadtreeView hysteresis = view_at(-1800.0, 1.0);
	terrain_quadtree_select(&tree, &hysteresis, 3);
	require(tree.draw_count == 4, "LOD hysteresis did not retain the split between thresholds");

	TerrainQuadtreeView far = view_at(-4000.0, 1.0);
	terrain_quadtree_select(&tree, &far, 8);
	require(tree.draw_count == 1 && tree.draw_nodes[0] == 0,
			"quadtree did not collapse below merge threshold");

	TerrainQuadtreeView away = view_at(-4000.0, -1.0);
	terrain_quadtree_select(&tree, &away, 9);
	require(tree.draw_count == 0, "eight-corner frustum culling failed");

	terrain_quadtree_select(&tree, &far, 10);
	terrain_quadtree_schedule_evictions(&tree);
	uint32_t evicted = 0;
	for (;;)
	{
		uint32_t index = terrain_quadtree_next_eviction(&tree);
		if (index == TERRAIN_QUADTREE_INVALID_NODE)
			break;
		require(terrain_quadtree_evicted(&tree, index), "eviction transition failed");
		evicted++;
	}
	require(evicted == 4, "sibling quad was not evicted together");
	require(tree.resident_tiles == 1 && tree.cpu_bytes == 100 && tree.gpu_bytes == 200,
			"resident memory accounting is wrong");

	terrain_quadtree_select(&tree, &near, 11);
	uint32_t request = terrain_quadtree_next_request(&tree);
	require(request != TERRAIN_QUADTREE_INVALID_NODE, "evicted child was not re-requested");
	uint32_t stale_generation = tree.nodes[request].generation;
	require(terrain_quadtree_cancel(&tree, request), "request cancellation failed");
	require(!terrain_quadtree_cpu_ready(&tree, request, stale_generation, &(TerrainTileBounds){0},
										0, NULL),
			"cancelled completion was accepted");

	terrain_quadtree_destroy(&tree);
}

static void test_unavailable_child_fallback(void)
{
	TerrainQuadtreeSettings settings = terrain_quadtree_default_settings();
	settings.split_threshold_px = 4.0f;
	settings.merge_threshold_px = 2.0f;

	TerrainQuadtree tree;
	require(terrain_quadtree_init(&tree, &settings), "quadtree init failed");
	require(terrain_quadtree_request_root(&tree), "root request failed");
	make_resident(&tree, 0, bounds(0, 0, 0, 500, 1.0f));

	TerrainQuadtreeView near = view_at(-900.0, 1.0);
	terrain_quadtree_select(&tree, &near, 1);
	require(tree.node_count == 5, "child quad was not created for fallback test");
	for (uint32_t quadrant = 0; quadrant < 4; ++quadrant)
	{
		uint32_t child = tree.nodes[0].children[quadrant];
		require(tree.nodes[child].state == TERRAIN_TILE_REQUESTED,
				"fallback child was not requested");
		uint32_t generation = tree.nodes[child].generation;
		require(terrain_quadtree_unavailable(&tree, child, generation),
				"missing child was not marked unavailable");
	}

	terrain_quadtree_select(&tree, &near, 2);
	require(tree.draw_count == 1 && tree.draw_nodes[0] == 0,
			"resident parent was not retained when children are missing");

	terrain_quadtree_select(&tree, &near, 3);
	for (uint32_t quadrant = 0; quadrant < 4; ++quadrant)
		require(tree.nodes[tree.nodes[0].children[quadrant]].state ==
					TERRAIN_TILE_UNAVAILABLE,
				"missing child was requested repeatedly");

	terrain_quadtree_destroy(&tree);
}

int main(void)
{
	test_grid();
	test_quadtree();
	test_unavailable_child_fallback();
	puts("phase 4 quadtree tests passed");
	return EXIT_SUCCESS;
}
