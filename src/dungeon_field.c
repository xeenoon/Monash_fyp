#include "dungeon_field.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

bool dungeon_field_create(uint32_t width, uint32_t height, float cell_size, DungeonPoint origin,
						  DungeonField *out)
{
	if (!out || width < 2u || height < 2u || !(cell_size > 0.0f))
		return false;
	*out = (DungeonField){0};
	if ((uint64_t)width * (uint64_t)height > (uint64_t)UINT32_MAX)
		return false;
	out->values = calloc((size_t)width * height, sizeof(*out->values));
	if (!out->values)
		return false;
	out->width = width;
	out->height = height;
	out->cell_size = cell_size;
	out->origin = origin;
	return true;
}

void dungeon_field_destroy(DungeonField *field)
{
	if (!field)
		return;
	free(field->values);
	*field = (DungeonField){0};
}

float dungeon_field_get(const DungeonField *field, uint32_t x, uint32_t z)
{
	if (x >= field->width || z >= field->height)
		return 0.0f;
	return field->values[(size_t)z * field->width + x];
}

void dungeon_field_set(DungeonField *field, uint32_t x, uint32_t z, float value)
{
	if (x >= field->width || z >= field->height)
		return;
	field->values[(size_t)z * field->width + x] = value;
}

DungeonPoint dungeon_field_corner_world(const DungeonField *field, uint32_t x, uint32_t z)
{
	return (DungeonPoint){field->origin.x + (float)x * field->cell_size,
						  field->origin.z + (float)z * field->cell_size};
}

static void world_to_grid(const DungeonField *field, DungeonPoint world, float *gx, float *gz)
{
	*gx = (world.x - field->origin.x) / field->cell_size;
	*gz = (world.z - field->origin.z) / field->cell_size;
}

float dungeon_field_sample(const DungeonField *field, DungeonPoint world)
{
	float gx, gz;
	world_to_grid(field, world, &gx, &gz);
	if (gx < 0.0f || gz < 0.0f || gx > (float)(field->width - 1u) ||
		gz > (float)(field->height - 1u))
		return 0.0f;
	uint32_t x0 = (uint32_t)gx, z0 = (uint32_t)gz;
	uint32_t x1 = x0 + 1u < field->width ? x0 + 1u : x0;
	uint32_t z1 = z0 + 1u < field->height ? z0 + 1u : z0;
	float tx = gx - (float)x0, tz = gz - (float)z0;
	float v00 = dungeon_field_get(field, x0, z0);
	float v10 = dungeon_field_get(field, x1, z0);
	float v01 = dungeon_field_get(field, x0, z1);
	float v11 = dungeon_field_get(field, x1, z1);
	float top = v00 + (v10 - v00) * tx;
	float bottom = v01 + (v11 - v01) * tx;
	return top + (bottom - top) * tz;
}

void dungeon_field_stamp_disc(DungeonField *field, DungeonPoint center, float radius,
							  float feather)
{
	if (!field || !(radius > 0.0f))
		return;
	float outer = radius + feather;
	float gx, gz;
	world_to_grid(field, center, &gx, &gz);
	float cell_radius = outer / field->cell_size + 1.0f;
	long x0 = (long)floorf(gx - cell_radius), x1 = (long)ceilf(gx + cell_radius);
	long z0 = (long)floorf(gz - cell_radius), z1 = (long)ceilf(gz + cell_radius);
	if (x0 < 0)
		x0 = 0;
	if (z0 < 0)
		z0 = 0;
	if (x1 >= (long)field->width)
		x1 = (long)field->width - 1;
	if (z1 >= (long)field->height)
		z1 = (long)field->height - 1;
	for (long z = z0; z <= z1; ++z)
		for (long x = x0; x <= x1; ++x)
		{
			DungeonPoint world = dungeon_field_corner_world(field, (uint32_t)x, (uint32_t)z);
			float dx = world.x - center.x, dz = world.z - center.z;
			float distance = sqrtf(dx * dx + dz * dz);
			/* 1 inside (radius - feather), easing to 0 at (radius + feather). */
			float value;
			if (distance <= radius - feather)
				value = 1.0f;
			else if (distance >= outer)
				value = 0.0f;
			else
			{
				float t = (outer - distance) / fmaxf(2.0f * feather, 1e-6f);
				value = t * t * (3.0f - 2.0f * t); /* smoothstep */
			}
			size_t index = (size_t)z * field->width + (size_t)x;
			if (value > field->values[index])
				field->values[index] = value;
		}
}

/* Separable 3-tap box blur pass with edge-clamped borders: horizontal blurs
 * along x holding z fixed, otherwise along z holding x fixed. */
static void blur_pass(const float *src, float *dst, uint32_t width, uint32_t height,
					  bool horizontal)
{
	for (uint32_t z = 0; z < height; ++z)
		for (uint32_t x = 0; x < width; ++x)
		{
			float a, b, c;
			b = src[(size_t)z * width + x];
			if (horizontal)
			{
				uint32_t lo = x > 0u ? x - 1u : x;
				uint32_t hi = x + 1u < width ? x + 1u : x;
				a = src[(size_t)z * width + lo];
				c = src[(size_t)z * width + hi];
			}
			else
			{
				uint32_t lo = z > 0u ? z - 1u : z;
				uint32_t hi = z + 1u < height ? z + 1u : z;
				a = src[(size_t)lo * width + x];
				c = src[(size_t)hi * width + x];
			}
			dst[(size_t)z * width + x] = (a + 2.0f * b + c) * 0.25f;
		}
}

void dungeon_field_blur(DungeonField *field, uint32_t iterations)
{
	if (!field || !iterations)
		return;
	size_t count = (size_t)field->width * field->height;
	float *temp = malloc(count * sizeof(*temp));
	if (!temp)
		return;
	for (uint32_t i = 0; i < iterations; ++i)
	{
		blur_pass(field->values, temp, field->width, field->height, true);
		blur_pass(temp, field->values, field->width, field->height, false);
	}
	free(temp);
}

static void nearest_corner(const DungeonField *field, DungeonPoint world, uint32_t *out_x,
						   uint32_t *out_z)
{
	float gx, gz;
	world_to_grid(field, world, &gx, &gz);
	long x = lroundf(gx), z = lroundf(gz);
	if (x < 0)
		x = 0;
	if (z < 0)
		z = 0;
	if (x >= (long)field->width)
		x = (long)field->width - 1;
	if (z >= (long)field->height)
		z = (long)field->height - 1;
	*out_x = (uint32_t)x;
	*out_z = (uint32_t)z;
}

/* Expanding ring search for the nearest corner satisfying `want_open` at or
 * above/below `iso`; returns false if none exists in the field at all. */
static bool find_nearest_matching(const DungeonField *field, float iso, uint32_t start_x,
								  uint32_t start_z, bool want_open, uint32_t *out_x,
								  uint32_t *out_z)
{
	uint32_t max_radius = field->width > field->height ? field->width : field->height;
	for (uint32_t radius = 0; radius <= max_radius; ++radius)
	{
		long x0 = (long)start_x - (long)radius, x1 = (long)start_x + (long)radius;
		long z0 = (long)start_z - (long)radius, z1 = (long)start_z + (long)radius;
		for (long z = z0; z <= z1; ++z)
		{
			if (z < 0 || z >= (long)field->height)
				continue;
			bool edge_row = (z == z0 || z == z1);
			for (long x = x0; x <= x1; ++x)
			{
				if (x < 0 || x >= (long)field->width)
					continue;
				if (!edge_row && x != x0 && x != x1)
					continue; /* only the ring perimeter, interior already checked */
				float value = dungeon_field_get(field, (uint32_t)x, (uint32_t)z);
				bool open = value >= iso;
				if (open == want_open)
				{
					*out_x = (uint32_t)x;
					*out_z = (uint32_t)z;
					return true;
				}
			}
		}
	}
	return false;
}

void dungeon_field_keep_largest_component(DungeonField *field, float iso, DungeonPoint seed_world)
{
	if (!field)
		return;
	size_t count = (size_t)field->width * field->height;
	uint32_t seed_x, seed_z;
	nearest_corner(field, seed_world, &seed_x, &seed_z);
	if (dungeon_field_get(field, seed_x, seed_z) < iso &&
		!find_nearest_matching(field, iso, seed_x, seed_z, true, &seed_x, &seed_z))
	{
		/* Nothing open anywhere; leave the field as-is. */
		return;
	}
	bool *visited = calloc(count, sizeof(*visited));
	uint32_t *stack = malloc(count * sizeof(*stack));
	if (!visited || !stack)
	{
		free(visited);
		free(stack);
		return;
	}
	uint32_t top = 0;
	size_t seed_index = (size_t)seed_z * field->width + seed_x;
	stack[top++] = (uint32_t)seed_index;
	visited[seed_index] = true;
	while (top > 0)
	{
		uint32_t index = stack[--top];
		uint32_t x = index % field->width, z = index / field->width;
		const int dx[4] = {1, -1, 0, 0};
		const int dz[4] = {0, 0, 1, -1};
		for (int d = 0; d < 4; ++d)
		{
			long nx = (long)x + dx[d], nz = (long)z + dz[d];
			if (nx < 0 || nz < 0 || nx >= (long)field->width || nz >= (long)field->height)
				continue;
			size_t neighbour = (size_t)nz * field->width + (size_t)nx;
			if (visited[neighbour] || field->values[neighbour] < iso)
				continue;
			visited[neighbour] = true;
			stack[top++] = (uint32_t)neighbour;
		}
	}
	for (size_t i = 0; i < count; ++i)
		if (!visited[i] && field->values[i] >= iso)
			field->values[i] = 0.0f;
	free(visited);
	free(stack);
}

bool dungeon_field_bfs_farthest(const DungeonField *field, float iso, DungeonPoint start_world,
								DungeonPoint *out_farthest)
{
	if (!field || !out_farthest)
		return false;
	uint32_t start_x, start_z;
	nearest_corner(field, start_world, &start_x, &start_z);
	if (dungeon_field_get(field, start_x, start_z) < iso &&
		!find_nearest_matching(field, iso, start_x, start_z, true, &start_x, &start_z))
		return false;
	size_t count = (size_t)field->width * field->height;
	int32_t *distance = malloc(count * sizeof(*distance));
	uint32_t *queue = malloc(count * sizeof(*queue));
	if (!distance || !queue)
	{
		free(distance);
		free(queue);
		return false;
	}
	for (size_t i = 0; i < count; ++i)
		distance[i] = -1;
	uint32_t read = 0, write = 0;
	size_t start_index = (size_t)start_z * field->width + start_x;
	distance[start_index] = 0;
	queue[write++] = (uint32_t)start_index;
	uint32_t farthest_index = (uint32_t)start_index;
	while (read < write)
	{
		uint32_t index = queue[read++];
		if (distance[index] > distance[farthest_index])
			farthest_index = index;
		uint32_t x = index % field->width, z = index / field->width;
		const int dx[4] = {1, -1, 0, 0};
		const int dz[4] = {0, 0, 1, -1};
		for (int d = 0; d < 4; ++d)
		{
			long nx = (long)x + dx[d], nz = (long)z + dz[d];
			if (nx < 0 || nz < 0 || nx >= (long)field->width || nz >= (long)field->height)
				continue;
			size_t neighbour = (size_t)nz * field->width + (size_t)nx;
			if (distance[neighbour] >= 0 || field->values[neighbour] < iso)
				continue;
			distance[neighbour] = distance[index] + 1;
			queue[write++] = (uint32_t)neighbour;
		}
	}
	*out_farthest =
		dungeon_field_corner_world(field, farthest_index % field->width, farthest_index / field->width);
	free(distance);
	free(queue);
	return true;
}

bool dungeon_field_bfs_direction(const DungeonField *field, float iso, DungeonPoint start_world,
								  DungeonPoint goal_world, DungeonPoint *out_initial_direction)
{
	if (!field || !out_initial_direction)
		return false;
	uint32_t start_x, start_z, goal_x, goal_z;
	nearest_corner(field, start_world, &start_x, &start_z);
	nearest_corner(field, goal_world, &goal_x, &goal_z);
	if ((dungeon_field_get(field, start_x, start_z) < iso &&
		 !find_nearest_matching(field, iso, start_x, start_z, true, &start_x, &start_z)) ||
		(dungeon_field_get(field, goal_x, goal_z) < iso &&
		 !find_nearest_matching(field, iso, goal_x, goal_z, true, &goal_x, &goal_z)))
		return false;
	size_t count = (size_t)field->width * field->height;
	int32_t *distance = malloc(count * sizeof(*distance));
	uint32_t *parent = malloc(count * sizeof(*parent));
	uint32_t *queue = malloc(count * sizeof(*queue));
	if (!distance || !parent || !queue)
	{
		free(distance);
		free(parent);
		free(queue);
		return false;
	}
	for (size_t i = 0; i < count; ++i)
	{
		distance[i] = -1;
		parent[i] = (uint32_t)-1;
	}
	uint32_t read = 0, write = 0;
	size_t start_index = (size_t)start_z * field->width + start_x;
	size_t goal_index = (size_t)goal_z * field->width + goal_x;
	distance[start_index] = 0;
	queue[write++] = (uint32_t)start_index;
	bool found = false;
	while (read < write && !found)
	{
		uint32_t index = queue[read++];
		if (index == goal_index)
		{
			found = true;
			break;
		}
		uint32_t x = index % field->width, z = index / field->width;
		const int dx[4] = {1, -1, 0, 0};
		const int dz[4] = {0, 0, 1, -1};
		for (int d = 0; d < 4; ++d)
		{
			long nx = (long)x + dx[d], nz = (long)z + dz[d];
			if (nx < 0 || nz < 0 || nx >= (long)field->width || nz >= (long)field->height)
				continue;
			size_t neighbour = (size_t)nz * field->width + (size_t)nx;
			if (distance[neighbour] >= 0 || field->values[neighbour] < iso)
				continue;
			distance[neighbour] = distance[index] + 1;
			parent[neighbour] = index;
			queue[write++] = (uint32_t)neighbour;
		}
	}
	if (!found)
	{
		free(distance);
		free(parent);
		free(queue);
		return false;
	}
	uint32_t current = (uint32_t)goal_index;
	while (parent[current] != (uint32_t)-1 && distance[current] > 1)
		current = parent[current];
	uint32_t first_step_x = current % field->width;
	uint32_t first_step_z = current / field->width;
	DungeonPoint first_pos = dungeon_field_corner_world(field, first_step_x, first_step_z);
	DungeonPoint direction = {first_pos.x - start_world.x, first_pos.z - start_world.z};
	float length = sqrtf(direction.x * direction.x + direction.z * direction.z);
	if (length > 1e-6f)
	{
		direction.x /= length;
		direction.z /= length;
	}
	*out_initial_direction = direction;
	free(distance);
	free(parent);
	free(queue);
	return true;
}

void dungeon_field_distance_to_solid(const DungeonField *field, float iso, float *out_distance)
{
	if (!field || !out_distance)
		return;
	uint32_t width = field->width, height = field->height;
	float axis = field->cell_size, diagonal = field->cell_size * 1.41421356f;
	const float large = 1e8f;
	for (uint32_t z = 0; z < height; ++z)
		for (uint32_t x = 0; x < width; ++x)
			out_distance[(size_t)z * width + x] =
				dungeon_field_get(field, x, z) < iso ? 0.0f : large;
	/* Forward pass: top-left to bottom-right. */
	for (uint32_t z = 0; z < height; ++z)
		for (uint32_t x = 0; x < width; ++x)
		{
			float *here = &out_distance[(size_t)z * width + x];
			if (x > 0u)
				*here = fminf(*here, out_distance[(size_t)z * width + x - 1u] + axis);
			if (z > 0u)
			{
				*here = fminf(*here, out_distance[(size_t)(z - 1u) * width + x] + axis);
				if (x > 0u)
					*here = fminf(*here, out_distance[(size_t)(z - 1u) * width + x - 1u] + diagonal);
				if (x + 1u < width)
					*here = fminf(*here, out_distance[(size_t)(z - 1u) * width + x + 1u] + diagonal);
			}
		}
	/* Backward pass: bottom-right to top-left. */
	for (long z = (long)height - 1; z >= 0; --z)
		for (long x = (long)width - 1; x >= 0; --x)
		{
			float *here = &out_distance[(size_t)z * width + (size_t)x];
			if (x + 1 < (long)width)
				*here = fminf(*here, out_distance[(size_t)z * width + (size_t)x + 1u] + axis);
			if (z + 1 < (long)height)
			{
				*here = fminf(*here, out_distance[(size_t)(z + 1) * width + (size_t)x] + axis);
				if (x + 1 < (long)width)
					*here = fminf(*here,
								 out_distance[(size_t)(z + 1) * width + (size_t)x + 1u] + diagonal);
				if (x > 0)
					*here = fminf(*here,
								 out_distance[(size_t)(z + 1) * width + (size_t)x - 1u] + diagonal);
			}
		}
}

void dungeon_field_stamp_rect(DungeonField *field, DungeonRect rect, float feather)
{
	if (!field || rect.max.x < rect.min.x || rect.max.z < rect.min.z)
		return;
	float gx0, gz0, gx1, gz1;
	world_to_grid(field, rect.min, &gx0, &gz0);
	world_to_grid(field, rect.max, &gx1, &gz1);
	float cell_feather = feather / field->cell_size + 1.0f;
	long x0 = (long)floorf(gx0 - cell_feather), x1 = (long)ceilf(gx1 + cell_feather);
	long z0 = (long)floorf(gz0 - cell_feather), z1 = (long)ceilf(gz1 + cell_feather);
	if (x0 < 0)
		x0 = 0;
	if (z0 < 0)
		z0 = 0;
	if (x1 >= (long)field->width)
		x1 = (long)field->width - 1;
	if (z1 >= (long)field->height)
		z1 = (long)field->height - 1;
	for (long z = z0; z <= z1; ++z)
		for (long x = x0; x <= x1; ++x)
		{
			DungeonPoint world = dungeon_field_corner_world(field, (uint32_t)x, (uint32_t)z);
			/* Signed distance to the rectangle: negative inside, positive out.
			 * Zero feather therefore writes a hard 1/0 step, which is what
			 * keeps room and hallway corners square through marching squares. */
			float ox = fmaxf(rect.min.x - world.x, world.x - rect.max.x);
			float oz = fmaxf(rect.min.z - world.z, world.z - rect.max.z);
			float outside_x = fmaxf(ox, 0.0f), outside_z = fmaxf(oz, 0.0f);
			float distance = sqrtf(outside_x * outside_x + outside_z * outside_z) +
							 fminf(fmaxf(ox, oz), 0.0f);
			float value;
			if (distance <= -feather)
				value = 1.0f;
			else if (distance >= feather)
				value = 0.0f;
			else
			{
				float t = (feather - distance) / fmaxf(2.0f * feather, 1e-6f);
				value = t * t * (3.0f - 2.0f * t); /* smoothstep, matching the disc */
			}
			size_t index = (size_t)z * field->width + (size_t)x;
			if (value > field->values[index])
				field->values[index] = value;
		}
}

void dungeon_field_blur_masked(DungeonField *field, uint32_t iterations, const uint8_t *mask)
{
	if (!field || !iterations)
		return;
	if (!mask)
	{
		dungeon_field_blur(field, iterations);
		return;
	}
	size_t count = (size_t)field->width * field->height;
	float *blurred = malloc(count * sizeof(*blurred));
	float *temp = malloc(count * sizeof(*temp));
	if (!blurred || !temp)
	{
		free(blurred);
		free(temp);
		return;
	}
	/* Blur a full copy, then take the result only where the mask allows. The
	 * blur is separable and reads neighbours, so it cannot be restricted in
	 * place without the masked-out corners still leaking a partial pass. */
	memcpy(blurred, field->values, count * sizeof(*blurred));
	for (uint32_t i = 0; i < iterations; ++i)
	{
		blur_pass(blurred, temp, field->width, field->height, true);
		blur_pass(temp, blurred, field->width, field->height, false);
	}
	for (size_t i = 0; i < count; ++i)
		if (mask[i])
			field->values[i] = blurred[i];
	free(blurred);
	free(temp);
}

/* A corner is traversable when it is open in the field AND not covered by the
 * caller's `blocked` overlay -- the overlay is how a door footprint is tested
 * without mutating (or copying) the field it belongs to. */
static bool corner_traversable(const DungeonField *field, float iso, const uint8_t *blocked,
							   size_t index)
{
	if (blocked && blocked[index])
		return false;
	return field->values[index] >= iso;
}

bool dungeon_field_reachable(const DungeonField *field, float iso, DungeonPoint from,
							 DungeonPoint to, const uint8_t *blocked)
{
	if (!field)
		return false;
	uint32_t from_x, from_z, to_x, to_z;
	nearest_corner(field, from, &from_x, &from_z);
	nearest_corner(field, to, &to_x, &to_z);
	size_t start_index = (size_t)from_z * field->width + from_x;
	size_t goal_index = (size_t)to_z * field->width + to_x;
	if (!corner_traversable(field, iso, blocked, start_index) ||
		!corner_traversable(field, iso, blocked, goal_index))
		return false;
	if (start_index == goal_index)
		return true;
	size_t count = (size_t)field->width * field->height;
	bool *visited = calloc(count, sizeof(*visited));
	uint32_t *queue = malloc(count * sizeof(*queue));
	if (!visited || !queue)
	{
		free(visited);
		free(queue);
		return false;
	}
	uint32_t read = 0, write = 0;
	visited[start_index] = true;
	queue[write++] = (uint32_t)start_index;
	bool found = false;
	while (read < write && !found)
	{
		uint32_t index = queue[read++];
		uint32_t x = index % field->width, z = index / field->width;
		const int dx[4] = {1, -1, 0, 0};
		const int dz[4] = {0, 0, 1, -1};
		for (int d = 0; d < 4; ++d)
		{
			long nx = (long)x + dx[d], nz = (long)z + dz[d];
			if (nx < 0 || nz < 0 || nx >= (long)field->width || nz >= (long)field->height)
				continue;
			size_t neighbour = (size_t)nz * field->width + (size_t)nx;
			if (visited[neighbour] || !corner_traversable(field, iso, blocked, neighbour))
				continue;
			if (neighbour == goal_index)
			{
				found = true;
				break;
			}
			visited[neighbour] = true;
			queue[write++] = (uint32_t)neighbour;
		}
	}
	free(visited);
	free(queue);
	return found;
}
