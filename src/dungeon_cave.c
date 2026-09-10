#include "dungeon_cave.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DUNGEON_CAVE_TAU 6.28318530718f
#define CAVE_MAX_ORGANIC_DISCS 4096u
#define CAVE_MAX_LOOP_CYCLE 3u /* tree hops a loop hallway may short-circuit */

/* Small local splitmix64, seeded from the public uint32_t seed. Deliberately
 * not libc rand(): its state is explicit, so identical params always produce
 * an identical level regardless of what else in the process has called
 * rand() first. */
typedef struct
{
	uint64_t state;
} CaveRng;

/* The seed is run through splitmix's finalizer before it becomes the state.
 * That is not decoration. cave_rng_next advances the state by the SAME golden
 * constant this seeding used to multiply by, so a plain `seed * K + C` makes
 * "which seed" and "how far into the stream" the same axis: seed s after n
 * draws lands on exactly the state of seed s+1 after n-1 draws. Since adjacent
 * seeds differ by only a draw or two before the doors are placed, DUNGEON_SEED
 * 4 and 5 were handing their locks identical combinations. */
static CaveRng cave_rng_create(uint32_t seed)
{
	uint64_t state = (uint64_t)seed * 0x9E3779B97F4A7C15ULL + 0xA24BAED4963EE407ULL;
	state = (state ^ (state >> 30)) * 0xBF58476D1CE4E5B9ULL;
	state = (state ^ (state >> 27)) * 0x94D049BB133111EBULL;
	CaveRng rng = {.state = state ^ (state >> 31)};
	return rng;
}

static uint64_t cave_rng_next(CaveRng *rng)
{
	uint64_t z = (rng->state += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

static float cave_rng_01(CaveRng *rng) { return (float)((cave_rng_next(rng) >> 40) * (1.0 / 16777216.0)); }

static float cave_rng_range(CaveRng *rng, float low, float high)
{
	return low + (high - low) * cave_rng_01(rng);
}

static uint32_t cave_rng_index(CaveRng *rng, uint32_t count)
{
	if (!count)
		return 0u;
	uint32_t index = (uint32_t)(cave_rng_01(rng) * (float)count);
	return index < count ? index : count - 1u;
}

DungeonCaveParams dungeon_cave_default_params(uint32_t seed)
{
	return (DungeonCaveParams){
		.seed = seed,
		.extent_m = 44.0f,
		.cell_size = 0.25f,
		.room_count = 9u,
		.room_min_m = 5.0f,
		.room_max_m = 8.5f,
		.hall_width_m = 2.0f,
		.extra_loops = 2u,
		.worm_count = 10u,
		.worm_steps = 9u,
		.step_length = 0.9f,
		.turn_rate = 0.55f,
		.min_radius = 1.0f,
		.max_radius = 1.9f,
		.chamber_count = 7u,
		.chamber_radius = 2.6f,
		.blur_iterations = 3u,
		.door_max = 3u,
		.puddle_max = 14u,
		.puddle_min_radius = 0.6f,
		.puddle_max_radius = 1.8f,
	};
}

static float clampf(float value, float low, float high)
{
	return value < low ? low : (value > high ? high : value);
}

/* Signed distance from a point to a rectangle: negative inside, positive out.
 * Every "is this organic pocket far enough from the rectilinear half" test
 * goes through this one function. */
static float rect_distance(DungeonRect rect, DungeonPoint point)
{
	float ox = fmaxf(rect.min.x - point.x, point.x - rect.max.x);
	float oz = fmaxf(rect.min.z - point.z, point.z - rect.max.z);
	float outside_x = fmaxf(ox, 0.0f), outside_z = fmaxf(oz, 0.0f);
	return sqrtf(outside_x * outside_x + outside_z * outside_z) + fminf(fmaxf(ox, oz), 0.0f);
}

typedef struct
{
	DungeonRect rect;
	DungeonPoint center;
} CaveRoom;

typedef struct
{
	DungeonPoint center;
	float radius;
	uint32_t room; /* which room this pocket hangs off; merges across rooms are rejected */
} CaveOrganicDisc;

/* Everything the carve passes need to reason about each other. Rooms and halls
 * are the rectilinear half; organic discs are the cave half. */
typedef struct
{
	CaveRoom *rooms;
	uint32_t room_count;
	DungeonRect *halls;
	uint32_t hall_count;
	CaveOrganicDisc *organic;
	uint32_t organic_count;
	uint8_t *organic_mask; /* per field corner; drives dungeon_field_blur_masked */
} CaveLayout;

/* Randomized Kruskal over the room-grid neighbour edges gives a spanning tree
 * (so the level is connected by construction, before any BFS), and the edges
 * it rejects are the pool the extra loop hallways come from. */
typedef struct
{
	uint32_t a, b;
} CaveEdge;

static uint32_t find_root(uint32_t *parent, uint32_t node)
{
	while (parent[node] != node)
		node = parent[node] = parent[parent[node]];
	return node;
}

/* Scratch for the spanning-tree walks below: one entry per room, plus one
 * edge-index mask per hallway. Bundled because every walk needs all of it. */
typedef struct
{
	uint32_t *queue;
	int32_t *depth;
	int32_t *parent_edge; /* index into the tree edge array, -1 at the root */
	uint8_t *edge_mask;
} CaveTreeScratch;

/* Walks the spanning tree from `from`, then retraces to `to` marking the edges
 * on that path in `out_mask` (which the caller clears). Returns the path length
 * in hops, or UINT32_MAX if `to` was not reached. */
static uint32_t tree_path(const CaveEdge *tree, uint32_t tree_count, uint32_t room_count,
						  uint32_t from, uint32_t to, const CaveTreeScratch *scratch,
						  uint8_t *out_mask)
{
	for (uint32_t i = 0; i < room_count; ++i)
	{
		scratch->depth[i] = -1;
		scratch->parent_edge[i] = -1;
	}
	uint32_t read = 0, write = 0;
	scratch->depth[from] = 0;
	scratch->queue[write++] = from;
	while (read < write)
	{
		uint32_t node = scratch->queue[read++];
		for (uint32_t i = 0; i < tree_count; ++i)
		{
			uint32_t other = tree[i].a == node ? tree[i].b : (tree[i].b == node ? tree[i].a : node);
			if (other == node || scratch->depth[other] >= 0)
				continue;
			scratch->depth[other] = scratch->depth[node] + 1;
			scratch->parent_edge[other] = (int32_t)i;
			scratch->queue[write++] = other;
		}
	}
	if (scratch->depth[to] < 0)
		return UINT32_MAX;
	uint32_t length = (uint32_t)scratch->depth[to];
	for (uint32_t node = to; scratch->parent_edge[node] >= 0;)
	{
		uint32_t edge = (uint32_t)scratch->parent_edge[node];
		if (out_mask)
			out_mask[edge] = 1u;
		node = tree[edge].a == node ? tree[edge].b : tree[edge].a;
	}
	return length;
}

/* The room farthest from `from` in tree hops -- a stand-in for where the exit
 * will land, computed before the field-space BFS that actually places it. */
static uint32_t tree_farthest_room(const CaveEdge *tree, uint32_t tree_count, uint32_t room_count,
								   uint32_t from, const CaveTreeScratch *scratch)
{
	tree_path(tree, tree_count, room_count, from, from, scratch, NULL);
	uint32_t best = from;
	for (uint32_t i = 0; i < room_count; ++i)
		if (scratch->depth[i] > scratch->depth[best])
			best = i;
	return best;
}

/* Marks every field corner a disc touches, so the masked blur rounds off the
 * organic half and leaves room and hallway corners exactly as stamped. */
static void mark_organic(const DungeonField *field, uint8_t *mask, DungeonPoint center,
						 float radius)
{
	float reach = radius + field->cell_size * 3.0f;
	for (uint32_t z = 0; z < field->height; ++z)
		for (uint32_t x = 0; x < field->width; ++x)
		{
			DungeonPoint world = dungeon_field_corner_world(field, x, z);
			float dx = world.x - center.x, dz = world.z - center.z;
			if (dx * dx + dz * dz <= reach * reach)
				mask[(size_t)z * field->width + x] = 1u;
		}
}

/* A pocket disc is admissible only if it clears every OTHER room, every
 * pocket belonging to a different room, and -- once it has left its own room
 * -- every hallway. Hallway rects run from room centre to room centre, so they
 * overlap the source room's interior; testing them only outside that room is
 * what stops the rule rejecting every pocket outright.
 *
 * This is a cheap geometric filter, not a proof. The authority on whether a
 * pocket opened a second route is door_is_chokepoint below, which re-asks the
 * question on the finished field. */
static bool organic_disc_admissible(const CaveLayout *layout, uint32_t room, DungeonPoint center,
									float radius, float clearance)
{
	for (uint32_t i = 0; i < layout->room_count; ++i)
	{
		if (i == room)
			continue;
		if (rect_distance(layout->rooms[i].rect, center) < radius + clearance)
			return false;
	}
	if (rect_distance(layout->rooms[room].rect, center) > 0.0f)
		for (uint32_t i = 0; i < layout->hall_count; ++i)
			if (rect_distance(layout->halls[i], center) < radius + clearance)
				return false;
	for (uint32_t i = 0; i < layout->organic_count; ++i)
	{
		if (layout->organic[i].room == room)
			continue;
		float dx = center.x - layout->organic[i].center.x;
		float dz = center.z - layout->organic[i].center.z;
		float gap = radius + layout->organic[i].radius + clearance;
		if (dx * dx + dz * dz < gap * gap)
			return false;
	}
	return true;
}

static void carve_organic_disc(DungeonField *field, CaveLayout *layout, uint32_t room,
							   DungeonPoint center, float radius)
{
	dungeon_field_stamp_disc(field, center, radius, field->cell_size * 1.5f);
	mark_organic(field, layout->organic_mask, center, radius);
	if (layout->organic_count < CAVE_MAX_ORGANIC_DISCS)
		layout->organic[layout->organic_count++] =
			(CaveOrganicDisc){.center = center, .radius = radius, .room = room};
}

/* Picks a point on a room's wall and the outward normal there. Grottos and
 * alcoves both grow from one of these, which is what keeps every pocket
 * attached to exactly one room. */
static void room_wall_point(CaveRng *rng, const CaveRoom *room, DungeonPoint *out_point,
							DungeonPoint *out_normal)
{
	uint32_t side = cave_rng_index(rng, 4u);
	float t = cave_rng_range(rng, 0.25f, 0.75f);
	switch (side)
	{
	case 0:
		*out_point = (DungeonPoint){room->rect.min.x + (room->rect.max.x - room->rect.min.x) * t,
									room->rect.min.z};
		*out_normal = (DungeonPoint){0.0f, -1.0f};
		break;
	case 1:
		*out_point = (DungeonPoint){room->rect.min.x + (room->rect.max.x - room->rect.min.x) * t,
									room->rect.max.z};
		*out_normal = (DungeonPoint){0.0f, 1.0f};
		break;
	case 2:
		*out_point = (DungeonPoint){room->rect.min.x,
									room->rect.min.z + (room->rect.max.z - room->rect.min.z) * t};
		*out_normal = (DungeonPoint){-1.0f, 0.0f};
		break;
	default:
		*out_point = (DungeonPoint){room->rect.max.x,
									room->rect.min.z + (room->rect.max.z - room->rect.min.z) * t};
		*out_normal = (DungeonPoint){1.0f, 0.0f};
		break;
	}
}

/* One hallway: an L of two axis-aligned rects, horizontal leg first or
 * vertical leg first. Kept around after carving because door placement needs
 * to see every hallway at once -- a point that is a chokepoint on this L can
 * still be sitting inside another L that crosses it. */
typedef struct
{
	DungeonPoint a, b;
	bool horizontal_first;
	bool spanning; /* only spanning-tree hallways get a door candidate */
	uint32_t rect_index; /* into layout->halls; this hallway owns two entries */
} CaveHall;

static void hall_rects(const CaveHall *hall, float half, DungeonRect *out_first,
					   DungeonRect *out_second)
{
	DungeonPoint a = hall->a, b = hall->b;
	float x_lo = fminf(a.x, b.x), x_hi = fmaxf(a.x, b.x);
	float z_lo = fminf(a.z, b.z), z_hi = fmaxf(a.z, b.z);
	if (hall->horizontal_first)
	{
		*out_first = (DungeonRect){{x_lo - half, a.z - half}, {x_hi + half, a.z + half}};
		*out_second = (DungeonRect){{b.x - half, z_lo - half}, {b.x + half, z_hi + half}};
	}
	else
	{
		*out_first = (DungeonRect){{a.x - half, z_lo - half}, {a.x + half, z_hi + half}};
		*out_second = (DungeonRect){{x_lo - half, b.z - half}, {x_hi + half, b.z + half}};
	}
}

/* The L's corner, where the corridor is wider than hall_width and a door
 * would not seal it. Door samples keep clear of it. */
static DungeonPoint hall_corner(const CaveHall *hall)
{
	return hall->horizontal_first ? (DungeonPoint){hall->b.x, hall->a.z}
								  : (DungeonPoint){hall->a.x, hall->b.z};
}

/* Finds a point along a hallway where a door would seal the corridor: clear of
 * every room, clear of the L's own corner, and clear of every OTHER hallway
 * (a crossing hallway is a hole a door cannot cover). Returns the point plus
 * the corridor direction there; false if the hallway has no such point. */
static bool hall_door_point(const CaveLayout *layout, const CaveHall *halls, uint32_t hall_count,
							uint32_t hall_index, float half, DungeonPoint *out_point,
							DungeonPoint *out_direction)
{
	const CaveHall *hall = &halls[hall_index];
	DungeonPoint corner = hall_corner(hall);
	DungeonPoint legs[2][2]; /* [leg][start, end] */
	if (hall->horizontal_first)
	{
		legs[0][0] = hall->a;
		legs[0][1] = corner;
		legs[1][0] = corner;
		legs[1][1] = hall->b;
	}
	else
	{
		legs[0][0] = hall->a;
		legs[0][1] = corner;
		legs[1][0] = corner;
		legs[1][1] = hall->b;
	}
	float clearance = half + 0.45f;
	bool found = false;
	float best_score = -1.0f;
	for (uint32_t leg = 0; leg < 2u; ++leg)
	{
		DungeonPoint start = legs[leg][0], end = legs[leg][1];
		float dx = end.x - start.x, dz = end.z - start.z;
		float length = sqrtf(dx * dx + dz * dz);
		if (length < clearance * 2.0f + 0.5f)
			continue;
		DungeonPoint direction = {dx / length, dz / length};
		const uint32_t samples = 32u;
		for (uint32_t i = 1; i < samples; ++i)
		{
			float t = (float)i / (float)samples;
			DungeonPoint point = {start.x + dx * t, start.z + dz * t};
			float from_corner = sqrtf((point.x - corner.x) * (point.x - corner.x) +
									  (point.z - corner.z) * (point.z - corner.z));
			if (from_corner < clearance)
				continue;
			bool clear = true;
			for (uint32_t r = 0; r < layout->room_count && clear; ++r)
				if (rect_distance(layout->rooms[r].rect, point) < clearance)
					clear = false;
			for (uint32_t h = 0; h < hall_count && clear; ++h)
			{
				if (h == hall_index)
					continue;
				DungeonRect first, second;
				hall_rects(&halls[h], half, &first, &second);
				if (rect_distance(first, point) < clearance ||
					rect_distance(second, point) < clearance)
					clear = false;
			}
			if (!clear)
				continue;
			/* Prefer the middle of the admissible stretch: the further a door
			 * sits from anything else, the more room its swing and its lock
			 * geometry have. */
			float score = fminf(t, 1.0f - t) * length;
			if (score > best_score)
			{
				best_score = score;
				*out_point = point;
				*out_direction = direction;
				found = true;
			}
		}
	}
	return found;
}

/* Solidifies a door's footprint in an overlay and asks whether spawn can still
 * reach exit. This is the invariant the whole hybrid layout rests on: a lock
 * is only placed where the answer is no. */
static bool door_is_chokepoint(const DungeonField *field, uint8_t *overlay, DungeonPoint spawn,
							   DungeonPoint exit, const DungeonDoorway *door)
{
	size_t count = (size_t)field->width * field->height;
	memset(overlay, 0, count);
	DungeonPoint normal = {cosf(door->yaw), sinf(door->yaw)};
	DungeonPoint across = {-normal.z, normal.x};
	float half_thickness = field->cell_size * 2.0f;
	float half_span = door->half_width + field->cell_size * 2.0f;
	for (uint32_t z = 0; z < field->height; ++z)
		for (uint32_t x = 0; x < field->width; ++x)
		{
			DungeonPoint world = dungeon_field_corner_world(field, x, z);
			float dx = world.x - door->center.x, dz = world.z - door->center.z;
			float along = dx * normal.x + dz * normal.z;
			float lateral = dx * across.x + dz * across.z;
			if (fabsf(along) <= half_thickness && fabsf(lateral) <= half_span)
				overlay[(size_t)z * field->width + x] = 1u;
		}
	return !dungeon_field_reachable(field, 0.5f, spawn, exit, overlay);
}

bool dungeon_cave_generate(const DungeonCaveParams *params, DungeonCaveResult *out)
{
	if (!params || !out || !(params->cell_size > 0.0f) || !params->room_count ||
		!(params->hall_width_m > 0.0f))
		return false;
	*out = (DungeonCaveResult){0};
	float half_extent = params->extent_m * 0.5f;
	/* Carving and blurring never reach the field border, so every contour
	 * loop dungeon_contour_extract finds is guaranteed closed. */
	float margin = fmaxf(params->max_radius, params->chamber_radius) +
				  (float)params->blur_iterations * params->cell_size * 2.0f + 1.0f;
	float safe_half = half_extent - margin;
	if (!(safe_half > 4.0f))
		return false;

	/* The room grid is squared up so every cell has the same extent and the
	 * neighbour graph is a plain 4-connected lattice. */
	uint32_t cols = (uint32_t)ceilf(sqrtf((float)params->room_count));
	if (cols < 1u)
		cols = 1u;
	/* Shrink the grid until a room plus the rock gap around it fits in a cell.
	 * A tight extent_m therefore degrades to fewer, larger rooms instead of
	 * failing outright -- callers asking for a compact level still get one,
	 * with correspondingly fewer hallways to put doors in. */
	float cell_extent = 0.0f, gap = 0.0f, max_room = 0.0f, min_room = 0.0f;
	for (;;)
	{
		cell_extent = (2.0f * safe_half) / (float)cols;
		gap = fminf(params->hall_width_m * 2.2f, cell_extent * 0.45f);
		max_room = fminf(params->room_max_m, cell_extent - gap);
		min_room = fminf(params->room_min_m, max_room);
		if (max_room >= 2.0f || cols == 1u)
			break;
		--cols;
	}
	uint32_t rows = cols;
	uint32_t room_count = rows * cols;
	if (!(max_room > 1.5f))
		return false;

	uint32_t corners = (uint32_t)ceilf(params->extent_m / params->cell_size) + 1u;
	DungeonPoint origin = {-half_extent, -half_extent};
	if (!dungeon_field_create(corners, corners, params->cell_size, origin, &out->field))
		return false;
	size_t corner_count = (size_t)out->field.width * out->field.height;

	uint32_t max_edges = rows * (cols - 1u) + cols * (rows - 1u);
	uint32_t edge_capacity = max_edges ? max_edges : 1u;
	CaveLayout layout = {0};
	CaveHall *halls = NULL;
	CaveEdge *edges = NULL;
	CaveEdge *leftover = NULL;
	CaveEdge *tree_edges = NULL;
	uint32_t *parent = NULL;
	uint32_t *tree_queue = NULL;
	int32_t *tree_depth = NULL;
	int32_t *tree_parent_edge = NULL;
	uint8_t *critical_mask = NULL;
	uint8_t *loop_mask = NULL;
	uint8_t *overlay = NULL;
	layout.rooms = malloc(room_count * sizeof(*layout.rooms));
	layout.halls = malloc(2u * edge_capacity * sizeof(*layout.halls));
	layout.organic = malloc(CAVE_MAX_ORGANIC_DISCS * sizeof(*layout.organic));
	layout.organic_mask = calloc(corner_count, sizeof(*layout.organic_mask));
	halls = malloc(edge_capacity * sizeof(*halls));
	edges = malloc(edge_capacity * sizeof(*edges));
	leftover = malloc(edge_capacity * sizeof(*leftover));
	tree_edges = malloc(edge_capacity * sizeof(*tree_edges));
	parent = malloc(room_count * sizeof(*parent));
	tree_queue = malloc(room_count * sizeof(*tree_queue));
	tree_depth = malloc(room_count * sizeof(*tree_depth));
	tree_parent_edge = malloc(room_count * sizeof(*tree_parent_edge));
	critical_mask = malloc(edge_capacity * sizeof(*critical_mask));
	loop_mask = malloc(edge_capacity * sizeof(*loop_mask));
	overlay = malloc(corner_count * sizeof(*overlay));
	out->doors = malloc((params->door_max ? params->door_max : 1u) * sizeof(*out->doors));
	if (!layout.rooms || !layout.halls || !layout.organic || !layout.organic_mask || !halls ||
		!edges || !leftover || !tree_edges || !parent || !tree_queue || !tree_depth ||
		!tree_parent_edge || !critical_mask || !loop_mask || !overlay || !out->doors)
		goto fail;
	layout.room_count = room_count;

	CaveRng rng = cave_rng_create(params->seed);

	for (uint32_t row = 0; row < rows; ++row)
		for (uint32_t col = 0; col < cols; ++col)
		{
			float cell_x = -safe_half + (float)col * cell_extent;
			float cell_z = -safe_half + (float)row * cell_extent;
			float width = cave_rng_range(&rng, min_room, max_room);
			float depth = cave_rng_range(&rng, min_room, max_room);
			float x = cell_x + gap * 0.5f + cave_rng_range(&rng, 0.0f, cell_extent - gap - width);
			float z = cell_z + gap * 0.5f + cave_rng_range(&rng, 0.0f, cell_extent - gap - depth);
			CaveRoom *room = &layout.rooms[row * cols + col];
			room->rect = (DungeonRect){{x, z}, {x + width, z + depth}};
			room->center = (DungeonPoint){x + width * 0.5f, z + depth * 0.5f};
		}

	uint32_t edge_count = 0;
	for (uint32_t row = 0; row < rows; ++row)
		for (uint32_t col = 0; col < cols; ++col)
		{
			uint32_t index = row * cols + col;
			if (col + 1u < cols)
				edges[edge_count++] = (CaveEdge){index, index + 1u};
			if (row + 1u < rows)
				edges[edge_count++] = (CaveEdge){index, index + cols};
		}
	for (uint32_t i = edge_count; i > 1u; --i)
	{
		uint32_t j = cave_rng_index(&rng, i);
		CaveEdge swap = edges[i - 1u];
		edges[i - 1u] = edges[j];
		edges[j] = swap;
	}

	/* Randomized Kruskal: the accepted edges are a spanning tree, so the level
	 * is connected before any BFS runs, and the rejected ones are the pool the
	 * extra loop hallways come from. */
	for (uint32_t i = 0; i < room_count; ++i)
		parent[i] = i;
	uint32_t hall_count = 0, leftover_count = 0;
	for (uint32_t i = 0; i < edge_count; ++i)
	{
		uint32_t root_a = find_root(parent, edges[i].a), root_b = find_root(parent, edges[i].b);
		if (root_a == root_b)
		{
			leftover[leftover_count++] = edges[i];
			continue;
		}
		parent[root_a] = root_b;
		tree_edges[hall_count] = edges[i];
		halls[hall_count++] = (CaveHall){.a = layout.rooms[edges[i].a].center,
										 .b = layout.rooms[edges[i].b].center,
										 .horizontal_first = (cave_rng_next(&rng) & 1u) != 0u,
										 .spanning = true};
	}
	/* Loop hallways must close SHORT cycles. A chord between two rooms far
	 * apart in the spanning tree turns every tree edge along the path between
	 * them into a bypass, and on a grid this small that is enough to leave the
	 * level with no chokepoint at all -- which is how the first version of
	 * this generator produced zero placeable doors. Bounding the cycle length
	 * keeps most of the tree bridging while still breaking up the corridors. */
	CaveTreeScratch scratch = {.queue = tree_queue,
							   .depth = tree_depth,
							   .parent_edge = tree_parent_edge,
							   .edge_mask = loop_mask};
	uint32_t tree_count = hall_count;
	memset(critical_mask, 0, edge_capacity);
	uint32_t exit_room = tree_farthest_room(tree_edges, tree_count, room_count, 0u, &scratch);
	tree_path(tree_edges, tree_count, room_count, 0u, exit_room, &scratch, critical_mask);

	uint32_t added_loops = 0;
	for (uint32_t i = 0; i < leftover_count && added_loops < params->extra_loops; ++i)
	{
		memset(loop_mask, 0, edge_capacity);
		uint32_t cycle = tree_path(tree_edges, tree_count, room_count, leftover[i].a,
								   leftover[i].b, &scratch, loop_mask);
		if (cycle > CAVE_MAX_LOOP_CYCLE)
			continue;
		/* A chord whose cycle covers part of the spawn-to-exit route turns
		 * those hallways into bypasses, and they are exactly the hallways
		 * worth putting a lock in. Loops go elsewhere. */
		bool touches_route = false;
		for (uint32_t e = 0; e < tree_count && !touches_route; ++e)
			touches_route = loop_mask[e] && critical_mask[e];
		if (touches_route)
			continue;
		halls[hall_count++] = (CaveHall){.a = layout.rooms[leftover[i].a].center,
										 .b = layout.rooms[leftover[i].b].center,
										 .horizontal_first = (cave_rng_next(&rng) & 1u) != 0u,
										 .spanning = false};
		++added_loops;
	}

	for (uint32_t i = 0; i < room_count; ++i)
		dungeon_field_stamp_rect(&out->field, layout.rooms[i].rect, 0.0f);

	float half_hall = params->hall_width_m * 0.5f;
	for (uint32_t i = 0; i < hall_count; ++i)
	{
		DungeonRect first, second;
		hall_rects(&halls[i], half_hall, &first, &second);
		halls[i].rect_index = layout.hall_count;
		layout.halls[layout.hall_count++] = first;
		layout.halls[layout.hall_count++] = second;
		dungeon_field_stamp_rect(&out->field, first, 0.0f);
		dungeon_field_stamp_rect(&out->field, second, 0.0f);
	}

	float pocket_clearance = params->hall_width_m * 0.45f;
	for (uint32_t grotto = 0; grotto < params->worm_count; ++grotto)
	{
		uint32_t room = cave_rng_index(&rng, room_count);
		DungeonPoint wall, normal;
		room_wall_point(&rng, &layout.rooms[room], &wall, &normal);
		float radius = cave_rng_range(&rng, params->min_radius, params->max_radius);
		float heading = atan2f(normal.z, normal.x) + cave_rng_range(&rng, -0.35f, 0.35f);
		DungeonPoint position = {wall.x + normal.x * radius * 0.4f,
								 wall.z + normal.z * radius * 0.4f};
		for (uint32_t step = 0; step < params->worm_steps; ++step)
		{
			if (fabsf(position.x) > safe_half || fabsf(position.z) > safe_half)
				break;
			if (!organic_disc_admissible(&layout, room, position, radius, pocket_clearance))
				break;
			carve_organic_disc(&out->field, &layout, room, position, radius);
			heading += cave_rng_range(&rng, -params->turn_rate, params->turn_rate);
			radius = clampf(radius + cave_rng_range(&rng, -0.12f, 0.12f), params->min_radius,
							params->max_radius);
			position = (DungeonPoint){position.x + cosf(heading) * params->step_length,
									  position.z + sinf(heading) * params->step_length};
		}
	}
	for (uint32_t alcove = 0; alcove < params->chamber_count; ++alcove)
	{
		uint32_t room = cave_rng_index(&rng, room_count);
		DungeonPoint wall, normal;
		room_wall_point(&rng, &layout.rooms[room], &wall, &normal);
		float radius = params->chamber_radius * cave_rng_range(&rng, 0.6f, 1.0f);
		DungeonPoint center = {wall.x + normal.x * radius * 0.35f,
							   wall.z + normal.z * radius * 0.35f};
		if (fabsf(center.x) > safe_half || fabsf(center.z) > safe_half)
			continue;
		if (organic_disc_admissible(&layout, room, center, radius, pocket_clearance))
			carve_organic_disc(&out->field, &layout, room, center, radius);
	}

	/* Rounding is what makes rock read as rock, but it would also round off
	 * every room corner -- so it runs only over the pockets. */
	dungeon_field_blur_masked(&out->field, params->blur_iterations, layout.organic_mask);

	out->spawn = layout.rooms[0].center;
	dungeon_field_keep_largest_component(&out->field, 0.5f, out->spawn);
	if (!dungeon_field_bfs_farthest(&out->field, 0.5f, out->spawn, &out->exit))
		goto fail;

	for (uint32_t i = 0; i < hall_count && out->door_count < params->door_max; ++i)
	{
		if (!halls[i].spanning)
			continue;
		DungeonPoint point, direction;
		if (!hall_door_point(&layout, halls, hall_count, i, half_hall, &point, &direction))
			continue;
		float yaw = atan2f(direction.z, direction.x);
		DungeonPoint across = {-direction.z, direction.x};
		DungeonDoorway door = {
			.center = point,
			.yaw = yaw,
			.half_width = half_hall,
			.blocker = {{point.x - across.x * half_hall, point.z - across.z * half_hall},
						{point.x + across.x * half_hall, point.z + across.z * half_hall}},
			.seed = (uint32_t)(cave_rng_next(&rng) >> 32),
		};
		if (!door_is_chokepoint(&out->field, overlay, out->spawn, out->exit, &door))
			continue; /* a pocket or a loop hallway already routes around it */
		const DungeonLockKind lock_kinds[] = {DUNGEON_LOCK_PIN_TUMBLER, DUNGEON_LOCK_SAFE_PINS,
											  DUNGEON_LOCK_PRISM};
		door.lock = lock_kinds[out->door_count % 3u];
		out->doors[out->door_count++] = door;
	}

	free(layout.rooms);
	free(layout.halls);
	free(layout.organic);
	free(layout.organic_mask);
	free(halls);
	free(edges);
	free(leftover);
	free(tree_edges);
	free(parent);
	free(tree_queue);
	free(tree_depth);
	free(tree_parent_edge);
	free(critical_mask);
	free(loop_mask);
	free(overlay);
	layout = (CaveLayout){0};

	if (!params->puddle_max)
		return true;
	float *distance = malloc(corner_count * sizeof(*distance));
	out->puddles = malloc(params->puddle_max * sizeof(*out->puddles));
	if (!distance || !out->puddles)
	{
		free(distance);
		free(out->puddles);
		out->puddles = NULL;
		free(out->doors);
		out->doors = NULL;
		out->door_count = 0;
		dungeon_field_destroy(&out->field);
		return false;
	}
	dungeon_field_distance_to_solid(&out->field, 0.5f, distance);

	uint32_t attempts = params->puddle_max * 20u + 40u;
	for (uint32_t attempt = 0; attempt < attempts && out->puddle_count < params->puddle_max;
		++attempt)
	{
		uint32_t x = cave_rng_index(&rng, out->field.width);
		uint32_t z = cave_rng_index(&rng, out->field.height);
		if (dungeon_field_get(&out->field, x, z) < 0.5f)
			continue;
		float clearance = distance[(size_t)z * out->field.width + x];
		float radius = cave_rng_range(&rng, params->puddle_min_radius, params->puddle_max_radius);
		if (clearance < radius + 0.3f)
		{
			radius = clearance - 0.3f;
			if (radius < params->puddle_min_radius * 0.5f)
				continue;
		}
		DungeonPoint center = dungeon_field_corner_world(&out->field, x, z);
		bool too_close = false;
		for (uint32_t i = 0; i < out->puddle_count; ++i)
		{
			float dx = center.x - out->puddles[i].center.x, dz = center.z - out->puddles[i].center.z;
			float min_gap = radius + out->puddles[i].radius + 0.5f;
			if (dx * dx + dz * dz < min_gap * min_gap)
			{
				too_close = true;
				break;
			}
		}
		/* A puddle under a door would be sliced by the swinging leaf. */
		for (uint32_t i = 0; i < out->door_count && !too_close; ++i)
		{
			float dx = center.x - out->doors[i].center.x, dz = center.z - out->doors[i].center.z;
			float min_gap = radius + out->doors[i].half_width + 0.6f;
			if (dx * dx + dz * dz < min_gap * min_gap)
				too_close = true;
		}
		if (too_close)
			continue;
		out->puddles[out->puddle_count++] = (DungeonPuddle){.center = center, .radius = radius};
	}
	free(distance);
	return true;

fail:
	free(layout.rooms);
	free(layout.halls);
	free(layout.organic);
	free(layout.organic_mask);
	free(halls);
	free(edges);
	free(leftover);
	free(tree_edges);
	free(parent);
	free(tree_queue);
	free(tree_depth);
	free(tree_parent_edge);
	free(critical_mask);
	free(loop_mask);
	free(overlay);
	free(out->doors);
	out->doors = NULL;
	out->door_count = 0;
	dungeon_field_destroy(&out->field);
	return false;
}

bool dungeon_cave_compile(const DungeonCaveParams *params, DungeonLevel *out,
						  DungeonLevelError *error)
{
	DungeonCaveResult cave = {0};
	if (!dungeon_cave_generate(params, &cave))
	{
		if (error)
			snprintf(error->message, sizeof(error->message), "level generation failed for seed %u",
					params ? params->seed : 0u);
		return false;
	}
	/* dungeon_level_compile_field takes ownership of cave.field (moves it on
	 * success, destroys it on failure) and copies the puddles and doors, so
	 * only those two arrays are still ours to free here. */
	bool ok = dungeon_level_compile_field(&cave.field, cave.spawn, cave.exit, cave.puddles,
										  cave.puddle_count, cave.doors, cave.door_count, 0.0f,
										  2.4f, out, error);
	free(cave.puddles);
	free(cave.doors);
	return ok;
}

void dungeon_cave_destroy(DungeonCaveResult *result)
{
	if (!result)
		return;
	dungeon_field_destroy(&result->field);
	free(result->puddles);
	free(result->doors);
	*result = (DungeonCaveResult){0};
}
