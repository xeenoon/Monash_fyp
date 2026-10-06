#pragma once

#include "dungeon_geometry.h"

#include <stdbool.h>
#include <stdint.h>

/* A scalar occupancy field sampled at grid *corners* (not cells): width by
 * height corners bound (width-1) by (height-1) cells. Values are occupancy in
 * roughly [0,1], 1 meaning open floor and 0 meaning solid rock; dungeon_contour
 * extracts geometry from the 0.5 isoline. Blurring this field is what makes
 * cave walls read as rounded rock instead of low-poly facets -- see
 * dungeon_field_blur. */
typedef struct
{
	float *values; /* row-major, width * height */
	uint32_t width, height;
	float cell_size;   /* metres between adjacent corners */
	DungeonPoint origin; /* world XZ of corner (0,0) */
} DungeonField;

bool dungeon_field_create(uint32_t width, uint32_t height, float cell_size, DungeonPoint origin,
						  DungeonField *out);
void dungeon_field_destroy(DungeonField *field);

float dungeon_field_get(const DungeonField *field, uint32_t x, uint32_t z);
void dungeon_field_set(DungeonField *field, uint32_t x, uint32_t z, float value);
DungeonPoint dungeon_field_corner_world(const DungeonField *field, uint32_t x, uint32_t z);

/* Bilinear sample in world space. Positions outside the field read as fully
 * solid (0.0) so contours never need to consider an unbounded plane. */
float dungeon_field_sample(const DungeonField *field, DungeonPoint world);

/* Unions a soft disc of occupancy into the field (max-blend with the existing
 * value), full strength inside (radius - feather) and easing to zero at
 * (radius + feather). Discs are the only primitive the cave generator needs;
 * corridors are dense chains of them. */
void dungeon_field_stamp_disc(DungeonField *field, DungeonPoint center, float radius,
							  float feather);

/* Separable 3-tap box blur, repeated `iterations` times (a cheap stand-in for
 * a Gaussian). This is the step that turns carved discs into a continuous
 * rounded cave; contours are extracted after this runs, never before. */
void dungeon_field_blur(DungeonField *field, uint32_t iterations);

/* Flood-fills open corners (value >= iso) connected to the corner nearest
 * `seed_world`, then zeroes every open corner NOT in that component. Called
 * once after blurring so reachability holds by construction: nothing
 * downstream needs its own connectivity check. */
void dungeon_field_keep_largest_component(DungeonField *field, float iso, DungeonPoint seed_world);

/* Breadth-first search over open corners (4-connected) starting at the corner
 * nearest `start_world`. Returns false if no open corner exists at all.
 * `out_farthest` receives the world position of the most steps-distant open
 * corner reached -- used to place the exit at the end of a real path rather
 * than a lucky short one. */
bool dungeon_field_bfs_farthest(const DungeonField *field, float iso, DungeonPoint start_world,
								DungeonPoint *out_farthest);

/* How many 4-connected steps of open field separate `start_world` from
   `goal_world`: walking distance rather than straight-line distance, so a goal
   on the other side of a wall is correctly far away. Returns false when the
   goal cannot be reached at all. Used to order a level's doors by the order a
   player actually meets them. */
bool dungeon_field_bfs_steps(const DungeonField *field, float iso, DungeonPoint start_world,
							 DungeonPoint goal_world, uint32_t *out_steps);

/* Breadth-first search from start_world to goal_world over open corners.
   Returns true if goal is reachable, false otherwise. On success, out_initial_direction
   receives the normalized direction of the first step away from start_world toward goal. */
bool dungeon_field_bfs_direction(const DungeonField *field, float iso, DungeonPoint start_world,
								  DungeonPoint goal_world, DungeonPoint *out_initial_direction);

/* Two-pass chamfer distance transform (metres) from every corner to the
 * nearest solid (value < iso) corner; open corners with no solid anywhere in
 * the field read as a large sentinel distance. `out_distance` must have
 * field->width * field->height slots. Used to keep puddles clear of walls. */
void dungeon_field_distance_to_solid(const DungeonField *field, float iso, float *out_distance);

/* Unions a rectangle of occupancy into the field (max-blend, like the disc).
 * `feather` is the half-width of the smoothstep band straddling the rect
 * boundary; pass 0 for a hard step. A hard step is the point: the 0.5 isoline
 * then lands midway between the last solid and first open corner, so marching
 * squares reconstructs straight edges and exact right angles. Discs cannot
 * produce a square corner, which is why rooms and hallways need this. */
void dungeon_field_stamp_rect(DungeonField *field, DungeonRect rect, float feather);

/* dungeon_field_blur, applied only where `mask` (row-major, width * height,
 * non-zero to blur) allows; corners outside the mask keep their exact value.
 * A NULL mask is identical to dungeon_field_blur. This is what lets one field
 * carry both looks at once: organic pockets round off into rock while room
 * and hallway corners stay square. */
void dungeon_field_blur_masked(DungeonField *field, uint32_t iterations, const uint8_t *mask);

/* 4-connected reachability between the corners nearest `from` and `to`.
 * `blocked` (optional, row-major, non-zero meaning "treat as solid") overlays
 * extra obstacles without mutating or copying the field -- this is how a door
 * footprint is tested for being a real chokepoint. Returns false if either
 * endpoint is itself solid or blocked. */
bool dungeon_field_reachable(const DungeonField *field, float iso, DungeonPoint from,
							 DungeonPoint to, const uint8_t *blocked);
