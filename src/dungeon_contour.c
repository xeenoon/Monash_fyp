#include "dungeon_contour.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* One crossing point on a grid edge (a corner-to-corner segment), plus the
 * up-to-two cell-contributed connections used to stitch crossings into closed
 * loops. Horizontal edges (between (i,j) and (i+1,j)) are indexed first,
 * followed by vertical edges (between (i,j) and (i,j+1)); both extract() and
 * triangulate_region() build the identical array so a loop and a fill can
 * never disagree about where the boundary sits. */
typedef struct
{
	bool exists;
	DungeonPoint point;
	uint32_t neighbor[2];
} EdgeNode;

static float edge_t(float a, float b, float iso)
{
	float d = b - a;
	if (fabsf(d) < 1e-8f)
		return 0.5f;
	float t = (iso - a) / d;
	return t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
}

static uint32_t horiz_id(uint32_t width, uint32_t i, uint32_t j) { return j * (width - 1u) + i; }
static uint32_t vert_id(uint32_t width, uint32_t height, uint32_t i, uint32_t j)
{
	return (width - 1u) * height + j * width + i;
}

static EdgeNode *compute_edges(const DungeonField *field, float iso, uint32_t *out_count)
{
	uint32_t width = field->width, height = field->height;
	uint32_t total = (width - 1u) * height + width * (height - 1u);
	EdgeNode *edges = calloc(total, sizeof(*edges));
	if (!edges)
		return NULL;
	for (uint32_t i = 0; i < total; ++i)
		edges[i].neighbor[0] = edges[i].neighbor[1] = UINT32_MAX;
	for (uint32_t j = 0; j < height; ++j)
		for (uint32_t i = 0; i + 1u < width; ++i)
		{
			float a = dungeon_field_get(field, i, j), b = dungeon_field_get(field, i + 1u, j);
			EdgeNode *node = &edges[horiz_id(width, i, j)];
			node->exists = (a >= iso) != (b >= iso);
			if (node->exists)
			{
				float t = edge_t(a, b, iso);
				DungeonPoint pa = dungeon_field_corner_world(field, i, j);
				DungeonPoint pb = dungeon_field_corner_world(field, i + 1u, j);
				node->point = (DungeonPoint){pa.x + (pb.x - pa.x) * t, pa.z + (pb.z - pa.z) * t};
			}
		}
	for (uint32_t j = 0; j + 1u < height; ++j)
		for (uint32_t i = 0; i < width; ++i)
		{
			float a = dungeon_field_get(field, i, j), b = dungeon_field_get(field, i, j + 1u);
			EdgeNode *node = &edges[vert_id(width, height, i, j)];
			node->exists = (a >= iso) != (b >= iso);
			if (node->exists)
			{
				float t = edge_t(a, b, iso);
				DungeonPoint pa = dungeon_field_corner_world(field, i, j);
				DungeonPoint pb = dungeon_field_corner_world(field, i, j + 1u);
				node->point = (DungeonPoint){pa.x + (pb.x - pa.x) * t, pa.z + (pb.z - pa.z) * t};
			}
		}
	*out_count = total;
	return edges;
}

static void attach(EdgeNode *edges, uint32_t a, uint32_t b)
{
	if (edges[a].neighbor[0] == UINT32_MAX)
		edges[a].neighbor[0] = b;
	else
		edges[a].neighbor[1] = b;
	if (edges[b].neighbor[0] == UINT32_MAX)
		edges[b].neighbor[0] = a;
	else
		edges[b].neighbor[1] = a;
}

/* Orients a freshly stitched loop so open floor (field >= iso) is on the left
 * while walking points in order -- CCW around an island of floor, CW around a
 * rock island surrounded by floor. Undirected stitching can hand back either
 * winding, so this is resolved once per loop with a single field probe rather
 * than trying to track direction through the (order-independent) stitching. */
static void fix_winding(const DungeonField *field, DungeonContourLoop *loop)
{
	if (loop->point_count < 3)
		return;
	DungeonPoint p0 = loop->points[0], p1 = loop->points[1];
	float dx = p1.x - p0.x, dz = p1.z - p0.z;
	float length = sqrtf(dx * dx + dz * dz);
	if (length < 1e-6f)
		return;
	float nx = -dz / length, nz = dx / length; /* left of travel direction */
	DungeonPoint mid = {(p0.x + p1.x) * 0.5f, (p0.z + p1.z) * 0.5f};
	float eps = field->cell_size * 0.25f;
	float left = dungeon_field_sample(field, (DungeonPoint){mid.x + nx * eps, mid.z + nz * eps});
	float right = dungeon_field_sample(field, (DungeonPoint){mid.x - nx * eps, mid.z - nz * eps});
	if (left >= right)
		return;
	for (uint32_t i = 0; i < loop->point_count / 2u; ++i)
	{
		DungeonPoint temp = loop->points[i];
		loop->points[i] = loop->points[loop->point_count - 1u - i];
		loop->points[loop->point_count - 1u - i] = temp;
	}
}

bool dungeon_contour_extract(const DungeonField *field, float iso, DungeonContourSet *out)
{
	if (!field || !out)
		return false;
	*out = (DungeonContourSet){0};
	uint32_t width = field->width, height = field->height;
	if (width < 2u || height < 2u)
		return true; /* nothing to extract, not an error */
	uint32_t edge_count = 0;
	EdgeNode *edges = compute_edges(field, iso, &edge_count);
	if (!edges)
		return false;
	for (uint32_t j = 0; j + 1u < height; ++j)
		for (uint32_t i = 0; i + 1u < width; ++i)
		{
			uint32_t s_id = horiz_id(width, i, j), n_id = horiz_id(width, i, j + 1u);
			uint32_t w_id = vert_id(width, height, i, j), e_id = vert_id(width, height, i + 1u, j);
			bool cs = edges[s_id].exists, cn = edges[n_id].exists;
			bool cw = edges[w_id].exists, ce = edges[e_id].exists;
			int count = (int)cs + (int)cn + (int)cw + (int)ce;
			if (count == 0)
				continue;
			if (count == 4)
			{
				bool bl = dungeon_field_get(field, i, j) >= iso;
				if (bl)
				{
					attach(edges, w_id, s_id);
					attach(edges, e_id, n_id);
				}
				else
				{
					attach(edges, s_id, e_id);
					attach(edges, w_id, n_id);
				}
				continue;
			}
			/* count == 2: pair the only two crossing edges present. */
			uint32_t ids[2];
			uint32_t found = 0;
			if (cw)
				ids[found++] = w_id;
			if (cs)
				ids[found++] = s_id;
			if (ce)
				ids[found++] = e_id;
			if (cn)
				ids[found++] = n_id;
			attach(edges, ids[0], ids[1]);
		}
	bool *visited = calloc(edge_count, sizeof(*visited));
	if (!visited)
	{
		free(edges);
		return false;
	}
	uint32_t loop_capacity = 0;
	for (uint32_t start = 0; start < edge_count; ++start)
	{
		if (!edges[start].exists || visited[start])
			continue;
		uint32_t point_capacity = 8, point_count = 0;
		DungeonPoint *points = malloc(point_capacity * sizeof(*points));
		if (!points)
		{
			free(visited);
			free(edges);
			dungeon_contour_destroy(out);
			return false;
		}
		uint32_t previous = UINT32_MAX, current = start;
		bool closed = false;
		for (;;)
		{
			if (visited[current] && current == start)
			{
				closed = true;
				break;
			}
			if (visited[current])
				break; /* malformed chain; keep what we have, drop the rest */
			visited[current] = true;
			if (point_count == point_capacity)
			{
				point_capacity *= 2u;
				DungeonPoint *grown = realloc(points, point_capacity * sizeof(*points));
				if (!grown)
				{
					free(points);
					free(visited);
					free(edges);
					dungeon_contour_destroy(out);
					return false;
				}
				points = grown;
			}
			points[point_count++] = edges[current].point;
			uint32_t n0 = edges[current].neighbor[0], n1 = edges[current].neighbor[1];
			uint32_t next = (n0 != previous && n0 != UINT32_MAX) ? n0 : n1;
			previous = current;
			current = next;
			if (current == UINT32_MAX)
				break;
			if (current == start)
			{
				closed = true;
				break;
			}
		}
		if (!closed || point_count < 3u)
		{
			/* An open field border (rock all the way around) never produces
			 * this; treat it defensively as noise and drop the fragment. */
			free(points);
			continue;
		}
		DungeonContourLoop loop = {.points = points, .point_count = point_count};
		fix_winding(field, &loop);
		if (out->loop_count == loop_capacity)
		{
			loop_capacity = loop_capacity ? loop_capacity * 2u : 4u;
			DungeonContourLoop *grown = realloc(out->loops, loop_capacity * sizeof(*out->loops));
			if (!grown)
			{
				free(points);
				free(visited);
				free(edges);
				dungeon_contour_destroy(out);
				return false;
			}
			out->loops = grown;
		}
		out->loops[out->loop_count++] = loop;
	}
	free(visited);
	free(edges);
	return true;
}

void dungeon_contour_destroy(DungeonContourSet *set)
{
	if (!set)
		return;
	for (uint32_t i = 0; i < set->loop_count; ++i)
		free(set->loops[i].points);
	free(set->loops);
	*set = (DungeonContourSet){0};
}

void dungeon_contour_loop_destroy(DungeonContourLoop *loop)
{
	if (!loop)
		return;
	free(loop->points);
	*loop = (DungeonContourLoop){0};
}

typedef struct
{
	DungeonPoint *positions;
	uint32_t vertex_capacity, vertex_count;
	uint32_t *indices;
	uint32_t index_capacity, index_count;
} MeshBuilder;

static bool mesh_reserve(MeshBuilder *builder, uint32_t extra_vertices, uint32_t extra_indices)
{
	if (builder->vertex_count + extra_vertices > builder->vertex_capacity)
	{
		uint32_t capacity = builder->vertex_capacity ? builder->vertex_capacity * 2u : 64u;
		while (capacity < builder->vertex_count + extra_vertices)
			capacity *= 2u;
		DungeonPoint *grown = realloc(builder->positions, capacity * sizeof(*builder->positions));
		if (!grown)
			return false;
		builder->positions = grown;
		builder->vertex_capacity = capacity;
	}
	if (builder->index_count + extra_indices > builder->index_capacity)
	{
		uint32_t capacity = builder->index_capacity ? builder->index_capacity * 2u : 64u;
		while (capacity < builder->index_count + extra_indices)
			capacity *= 2u;
		uint32_t *grown = realloc(builder->indices, capacity * sizeof(*builder->indices));
		if (!grown)
			return false;
		builder->indices = grown;
		builder->index_capacity = capacity;
	}
	return true;
}

/* Fans `verts[0..count)` (already CCW, count in [3,5]) into the mesh. */
static bool emit_fan(MeshBuilder *builder, const DungeonPoint *verts, uint32_t count)
{
	if (count < 3u)
		return true;
	if (!mesh_reserve(builder, count, (count - 2u) * 3u))
		return false;
	uint32_t base = builder->vertex_count;
	for (uint32_t i = 0; i < count; ++i)
		builder->positions[builder->vertex_count++] = verts[i];
	for (uint32_t i = 1; i + 1u < count; ++i)
	{
		builder->indices[builder->index_count++] = base;
		builder->indices[builder->index_count++] = base + i;
		builder->indices[builder->index_count++] = base + i + 1u;
	}
	return true;
}

bool dungeon_contour_triangulate_region(const DungeonField *field, float iso, bool want_inside,
										DungeonTriangleMesh *out)
{
	if (!field || !out)
		return false;
	*out = (DungeonTriangleMesh){0};
	uint32_t width = field->width, height = field->height;
	if (width < 2u || height < 2u)
		return true;
	uint32_t edge_count = 0;
	EdgeNode *edges = compute_edges(field, iso, &edge_count);
	if (!edges)
		return false;
	MeshBuilder builder = {0};
	for (uint32_t j = 0; j + 1u < height; ++j)
		for (uint32_t i = 0; i + 1u < width; ++i)
		{
			DungeonPoint corner_bl = dungeon_field_corner_world(field, i, j);
			DungeonPoint corner_br = dungeon_field_corner_world(field, i + 1u, j);
			DungeonPoint corner_tr = dungeon_field_corner_world(field, i + 1u, j + 1u);
			DungeonPoint corner_tl = dungeon_field_corner_world(field, i, j + 1u);
			bool in_bl = (dungeon_field_get(field, i, j) >= iso) == want_inside;
			bool in_br = (dungeon_field_get(field, i + 1u, j) >= iso) == want_inside;
			bool in_tr = (dungeon_field_get(field, i + 1u, j + 1u) >= iso) == want_inside;
			bool in_tl = (dungeon_field_get(field, i, j + 1u) >= iso) == want_inside;
			uint32_t s_id = horiz_id(width, i, j), n_id = horiz_id(width, i, j + 1u);
			uint32_t w_id = vert_id(width, height, i, j), e_id = vert_id(width, height, i + 1u, j);
			bool cs = edges[s_id].exists, cn = edges[n_id].exists;
			bool cw = edges[w_id].exists, ce = edges[e_id].exists;
			bool checkerboard = cs && cn && cw && ce;
			bool ok = true;
			if (checkerboard)
			{
				/* Fixed saddle resolution: cut each "in" corner off on its
				 * own, matching the single-corner case below. Rare after
				 * blurring, and either resolution is a one-cell cosmetic
				 * choice, not a correctness issue. */
				if (in_bl)
				{
					DungeonPoint t1[3] = {corner_bl, edges[s_id].point, edges[w_id].point};
					DungeonPoint t2[3] = {corner_tr, edges[n_id].point, edges[e_id].point};
					ok = emit_fan(&builder, t1, 3u) && emit_fan(&builder, t2, 3u);
				}
				else
				{
					DungeonPoint t1[3] = {corner_br, edges[e_id].point, edges[s_id].point};
					DungeonPoint t2[3] = {corner_tl, edges[w_id].point, edges[n_id].point};
					ok = emit_fan(&builder, t1, 3u) && emit_fan(&builder, t2, 3u);
				}
			}
			else
			{
				DungeonPoint verts[8];
				uint32_t n = 0;
				if (in_bl)
					verts[n++] = corner_bl;
				if (cs)
					verts[n++] = edges[s_id].point;
				if (in_br)
					verts[n++] = corner_br;
				if (ce)
					verts[n++] = edges[e_id].point;
				if (in_tr)
					verts[n++] = corner_tr;
				if (cn)
					verts[n++] = edges[n_id].point;
				if (in_tl)
					verts[n++] = corner_tl;
				if (cw)
					verts[n++] = edges[w_id].point;
				ok = emit_fan(&builder, verts, n);
			}
			if (!ok)
			{
				free(edges);
				free(builder.positions);
				free(builder.indices);
				*out = (DungeonTriangleMesh){0};
				return false;
			}
		}
	free(edges);
	out->positions = builder.positions;
	out->vertex_count = builder.vertex_count;
	out->indices = builder.indices;
	out->index_count = builder.index_count;
	return true;
}

void dungeon_triangle_mesh_destroy(DungeonTriangleMesh *mesh)
{
	if (!mesh)
		return;
	free(mesh->positions);
	free(mesh->indices);
	*mesh = (DungeonTriangleMesh){0};
}

static void dp_recurse(const DungeonPoint *points, uint32_t lo, uint32_t hi, float epsilon,
					   bool *keep)
{
	if (hi <= lo + 1u)
		return;
	DungeonPoint a = points[lo], b = points[hi];
	float dx = b.x - a.x, dz = b.z - a.z;
	float length2 = dx * dx + dz * dz;
	float max_distance2 = -1.0f;
	uint32_t max_index = lo;
	for (uint32_t i = lo + 1u; i < hi; ++i)
	{
		DungeonPoint p = points[i];
		float t = length2 > 1e-12f ? ((p.x - a.x) * dx + (p.z - a.z) * dz) / length2 : 0.0f;
		t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
		float px = a.x + dx * t, pz = a.z + dz * t;
		float ddx = p.x - px, ddz = p.z - pz;
		float distance2 = ddx * ddx + ddz * ddz;
		if (distance2 > max_distance2)
		{
			max_distance2 = distance2;
			max_index = i;
		}
	}
	if (max_distance2 > epsilon * epsilon)
	{
		keep[max_index] = true;
		dp_recurse(points, lo, max_index, epsilon, keep);
		dp_recurse(points, max_index, hi, epsilon, keep);
	}
}

bool dungeon_contour_simplify(const DungeonContourLoop *loop, float epsilon,
							  DungeonContourLoop *out)
{
	if (!loop || !out || loop->point_count < 3u)
		return false;
	*out = (DungeonContourLoop){0};
	uint32_t n = loop->point_count;
	bool *keep = calloc(n, sizeof(*keep));
	if (!keep)
		return false;
	uint32_t mid = n / 2u;
	keep[0] = true;
	keep[mid] = true;
	dp_recurse(loop->points, 0, mid, epsilon, keep);
	/* Second chain wraps mid -> n-1 -> point[0]; simplify it as its own
	 * contiguous array so dp_recurse never needs to know about wraparound. */
	uint32_t tail_length = n - mid + 1u;
	DungeonPoint *tail = malloc(tail_length * sizeof(*tail));
	bool *tail_keep = calloc(tail_length, sizeof(*tail_keep));
	if (!tail || !tail_keep)
	{
		free(keep);
		free(tail);
		free(tail_keep);
		return false;
	}
	for (uint32_t k = 0; k < n - mid; ++k)
		tail[k] = loop->points[mid + k];
	tail[tail_length - 1u] = loop->points[0];
	tail_keep[0] = true;
	tail_keep[tail_length - 1u] = true;
	dp_recurse(tail, 0, tail_length - 1u, epsilon, tail_keep);
	for (uint32_t k = 1; k + 1u < tail_length; ++k)
		if (tail_keep[k])
			keep[mid + k] = true;
	free(tail);
	free(tail_keep);
	uint32_t kept = 0;
	for (uint32_t i = 0; i < n; ++i)
		kept += keep[i] ? 1u : 0u;
	if (kept < 3u)
	{
		/* Degenerated below a usable polygon; hand back the input unchanged
		 * rather than produce something collision/lighting can't use. */
		free(keep);
		DungeonPoint *copy = malloc(n * sizeof(*copy));
		if (!copy)
			return false;
		memcpy(copy, loop->points, n * sizeof(*copy));
		out->points = copy;
		out->point_count = n;
		return true;
	}
	DungeonPoint *result = malloc(kept * sizeof(*result));
	if (!result)
	{
		free(keep);
		return false;
	}
	uint32_t w = 0;
	for (uint32_t i = 0; i < n; ++i)
		if (keep[i])
			result[w++] = loop->points[i];
	free(keep);
	out->points = result;
	out->point_count = kept;
	return true;
}
