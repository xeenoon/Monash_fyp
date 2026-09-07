#include "dungeon_mesh.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct
{
	DungeonMeshBatch *batch;
	uint32_t vertex_capacity;
	uint32_t index_capacity;
} Builder;

static bool mesh_fail(DungeonLevelError *error, const char *format,
					  ...)
{
	if (error)
	{
		va_list args;
		va_start(args, format);
		vsnprintf(error->message, sizeof(error->message), format,
				  args);
		va_end(args);
	}
	return false;
}

static bool builder_create(Builder *builder, DungeonMeshBatch *batch,
						   uint32_t vertex_capacity,
						   uint32_t index_capacity,
						   float material_width_m)
{
	batch->vertices =
		vertex_capacity
			? calloc(vertex_capacity, sizeof(*batch->vertices))
			: NULL;
	batch->indices = index_capacity ? calloc(index_capacity,
											 sizeof(*batch->indices))
									: NULL;
	batch->material_width_m = material_width_m;
	builder->batch = batch;
	builder->vertex_capacity = vertex_capacity;
	builder->index_capacity = index_capacity;
	return (!vertex_capacity || batch->vertices) &&
		   (!index_capacity || batch->indices);
}

static bool append_quad(Builder *builder, float positions[4][3],
						const float normal[3], const float tangent[4],
						float uv[4][2])
{
	DungeonMeshBatch *batch = builder->batch;
	if (batch->vertex_count + 4u > builder->vertex_capacity ||
		batch->index_count + 6u > builder->index_capacity)
		return false;
	uint32_t base = batch->vertex_count;
	for (uint32_t i = 0; i < 4; ++i)
	{
		Vertex *vertex = &batch->vertices[batch->vertex_count++];
		for (uint32_t component = 0; component < 3; ++component)
		{
			vertex->position[component] = positions[i][component];
			vertex->normal[component] = normal[component];
			vertex->tangent[component] = tangent[component];
		}
		vertex->tangent[3] = tangent[3];
		vertex->texcoord[0] = uv[i][0];
		vertex->texcoord[1] = uv[i][1];
	}
	const uint32_t indices[6] = {base, base + 1u, base + 2u,
								 base, base + 2u, base + 3u};
	for (uint32_t i = 0; i < 6; ++i)
		batch->indices[batch->index_count++] = indices[i];
	return true;
}

static bool append_horizontal(Builder *builder, DungeonRect rectangle,
							  float y, bool upward)
{
	float p[4][3] = {{rectangle.min.x, y, rectangle.min.z},
					 {rectangle.max.x, y, rectangle.min.z},
					 {rectangle.max.x, y, rectangle.max.z},
					 {rectangle.min.x, y, rectangle.max.z}};
	float n[3] = {0.0f, upward ? 1.0f : -1.0f, 0.0f};
	float t[4] = {1.0f, 0.0f, 0.0f, upward ? -1.0f : 1.0f};
	float scale = 1.0f / builder->batch->material_width_m;
	float uv[4][2] = {
		{rectangle.min.x * scale, rectangle.min.z * scale},
		{rectangle.max.x * scale, rectangle.min.z * scale},
		{rectangle.max.x * scale, rectangle.max.z * scale},
		{rectangle.min.x * scale, rectangle.max.z * scale}};
	return append_quad(builder, p, n, t, uv);
}

static bool append_box(Builder *builder, DungeonRect rectangle,
					   float low, float high)
{
	float scale = 1.0f / builder->batch->material_width_m;
	if (!append_horizontal(builder, rectangle, high, true) ||
		!append_horizontal(builder, rectangle, low, false))
		return false;
	float face_positions[4][4][3] = {
		{{rectangle.min.x, low, rectangle.min.z},
		 {rectangle.max.x, low, rectangle.min.z},
		 {rectangle.max.x, high, rectangle.min.z},
		 {rectangle.min.x, high, rectangle.min.z}},
		{{rectangle.max.x, low, rectangle.max.z},
		 {rectangle.min.x, low, rectangle.max.z},
		 {rectangle.min.x, high, rectangle.max.z},
		 {rectangle.max.x, high, rectangle.max.z}},
		{{rectangle.min.x, low, rectangle.max.z},
		 {rectangle.min.x, low, rectangle.min.z},
		 {rectangle.min.x, high, rectangle.min.z},
		 {rectangle.min.x, high, rectangle.max.z}},
		{{rectangle.max.x, low, rectangle.min.z},
		 {rectangle.max.x, low, rectangle.max.z},
		 {rectangle.max.x, high, rectangle.max.z},
		 {rectangle.max.x, high, rectangle.min.z}},
	};
	const float normals[4][3] = {
		{0, 0, -1}, {0, 0, 1}, {-1, 0, 0}, {1, 0, 0}};
	const float tangents[4][4] = {
		{1, 0, 0, -1}, {-1, 0, 0, -1}, {0, 0, -1, -1}, {0, 0, 1, -1}};
	float lengths[4] = {rectangle.max.x - rectangle.min.x,
						rectangle.max.x - rectangle.min.x,
						rectangle.max.z - rectangle.min.z,
						rectangle.max.z - rectangle.min.z};
	for (uint32_t face = 0; face < 4; ++face)
	{
		float uv[4][2] = {
			{0, 0},
			{lengths[face] * scale, 0},
			{lengths[face] * scale, (high - low) * scale},
			{0, (high - low) * scale}};
		if (!append_quad(builder, face_positions[face], normals[face],
						 tangents[face], uv))
			return false;
	}
	return true;
}

/* Lifts a flat XZ triangulation (floor fill or rock plateau cap) to a
 * fixed Y with a uniform upward or downward normal -- both are
 * constant-Y caps, never a sloped surface, so a per-vertex normal
 * isn't needed. */
static bool append_region_mesh(Builder *builder,
							   const DungeonTriangleMesh *region,
							   float y, bool upward)
{
	DungeonMeshBatch *batch = builder->batch;
	if (!region->vertex_count)
		return true;
	if (batch->vertex_count + region->vertex_count >
			builder->vertex_capacity ||
		batch->index_count + region->index_count >
			builder->index_capacity)
		return false;
	float scale = 1.0f / batch->material_width_m;
	float normal_y = upward ? 1.0f : -1.0f;
	float tangent_w = upward ? -1.0f : 1.0f;
	uint32_t base = batch->vertex_count;
	for (uint32_t i = 0; i < region->vertex_count; ++i)
	{
		Vertex *vertex = &batch->vertices[batch->vertex_count++];
		vertex->position[0] = region->positions[i].x;
		vertex->position[1] = y;
		vertex->position[2] = region->positions[i].z;
		vertex->normal[0] = 0.0f;
		vertex->normal[1] = normal_y;
		vertex->normal[2] = 0.0f;
		vertex->tangent[0] = 1.0f;
		vertex->tangent[1] = 0.0f;
		vertex->tangent[2] = 0.0f;
		vertex->tangent[3] = tangent_w;
		vertex->texcoord[0] = region->positions[i].x * scale;
		vertex->texcoord[1] = region->positions[i].z * scale;
	}
	for (uint32_t i = 0; i < region->index_count; ++i)
		batch->indices[batch->index_count++] =
			base + region->indices[i];
	return true;
}

/* A cheap, deterministic per-loop phase so neighbouring walls don't
 * all bulge in lockstep; not a hash worth its own header, just enough
 * spread. */
static float loop_wobble_phase(uint32_t loop_index)
{
	uint32_t mixed = loop_index * 2654435761u;
	return (float)(mixed % 1000u) / 1000.0f * 6.2831853f;
}

/* Extrudes every contour loop into a vertical ring strip from floor_y
 * to floor_y + wall_height. Per-vertex normals/tangents are averaged
 * across the loop's two adjacent edges (not per-face), and a small
 * outward-only, arc-length-driven wobble breaks up the extrusion --
 * together these are what keep the wall reading as rounded rock
 * instead of a faceted prism. UV.u is arc length, UV.v is height,
 * both in metres; there is one seam where the loop wraps back to its
 * start. */
static bool append_wall_loops(Builder *builder,
							  const DungeonContourSet *contours,
							  float floor_y, float wall_height,
							  uint32_t rings)
{
	DungeonMeshBatch *batch = builder->batch;
	float scale = 1.0f / batch->material_width_m;
	const float max_bulge = 0.16f;
	for (uint32_t loop_index = 0; loop_index < contours->loop_count;
		 ++loop_index)
	{
		const DungeonContourLoop *loop = &contours->loops[loop_index];
		uint32_t m = loop->point_count;
		if (m < 3u)
			continue;
		float phase = loop_wobble_phase(loop_index);

		float *arc = malloc(m * sizeof(*arc));
		DungeonPoint *normal = malloc(m * sizeof(*normal));
		DungeonPoint *tangent_dir = malloc(m * sizeof(*tangent_dir));
		float *handedness = malloc(m * sizeof(*handedness));
		if (!arc || !normal || !tangent_dir || !handedness)
		{
			free(arc);
			free(normal);
			free(tangent_dir);
			free(handedness);
			return false;
		}
		arc[0] = 0.0f;
		for (uint32_t i = 1; i < m; ++i)
		{
			float dx = loop->points[i].x - loop->points[i - 1u].x;
			float dz = loop->points[i].z - loop->points[i - 1u].z;
			arc[i] = arc[i - 1u] + sqrtf(dx * dx + dz * dz);
		}
		for (uint32_t i = 0; i < m; ++i)
		{
			DungeonPoint prev = loop->points[(i + m - 1u) % m];
			DungeonPoint curr = loop->points[i];
			DungeonPoint next = loop->points[(i + 1u) % m];
			/* "Right of travel" is outward, since loops wind with
			 * open floor on the left (see dungeon_contour.c's
			 * fix_winding). */
			float bx = curr.x - prev.x, bz = curr.z - prev.z;
			float blen = sqrtf(bx * bx + bz * bz);
			DungeonPoint normal_before =
				blen > 1e-6f ? (DungeonPoint){bz / blen, -bx / blen}
							 : (DungeonPoint){1.0f, 0.0f};
			float ax = next.x - curr.x, az = next.z - curr.z;
			float alen = sqrtf(ax * ax + az * az);
			DungeonPoint normal_after =
				alen > 1e-6f ? (DungeonPoint){az / alen, -ax / alen}
							 : normal_before;
			float nx = normal_before.x + normal_after.x,
				  nz = normal_before.z + normal_after.z;
			float nlen = sqrtf(nx * nx + nz * nz);
			normal[i] = nlen > 1e-6f
							? (DungeonPoint){nx / nlen, nz / nlen}
							: normal_before;

			float tx = next.x - prev.x, tz = next.z - prev.z;
			float tlen = sqrtf(tx * tx + tz * tz);
			tangent_dir[i] =
				tlen > 1e-6f
					? (DungeonPoint){tx / tlen, tz / tlen}
					: (DungeonPoint){-normal[i].z, normal[i].x};
			float k = normal[i].z * tangent_dir[i].x -
					  normal[i].x * tangent_dir[i].z;
			handedness[i] = k >= 0.0f ? 1.0f : -1.0f;
		}

		uint32_t base_vertex = batch->vertex_count;
		if (batch->vertex_count + m * rings >
				builder->vertex_capacity ||
			batch->index_count + (rings - 1u) * m * 6u >
				builder->index_capacity)
		{
			free(arc);
			free(normal);
			free(tangent_dir);
			free(handedness);
			return false;
		}
		for (uint32_t ring = 0; ring < rings; ++ring)
		{
			float h_frac = (float)ring / (float)(rings - 1u);
			float y = floor_y + wall_height * h_frac;
			/* The floor and plateau cap both terminate on the
			 * undisplaced contour. Pin the end rings to that shared
			 * boundary so the wall cannot open a crack at either
			 * join; fade the rocky wobble in only across the interior
			 * rings. */
			float seam_fade = (ring == 0u || ring + 1u == rings)
								  ? 0.0f
								  : sinf(h_frac * 3.14159265f);
			for (uint32_t i = 0; i < m; ++i)
			{
				float s = arc[i];
				float wobble =
					sinf(s * 0.6f + phase) * 0.5f +
					sinf(s * 1.7f - phase * 1.3f + h_frac * 2.0f) *
						0.25f;
				float bulge =
					max_bulge * (wobble * 0.5f + 0.5f) * seam_fade;
				float px = loop->points[i].x + normal[i].x * bulge;
				float pz = loop->points[i].z + normal[i].z * bulge;
				Vertex *vertex =
					&batch->vertices[batch->vertex_count++];
				vertex->position[0] = px;
				vertex->position[1] = y;
				vertex->position[2] = pz;
				vertex->normal[0] = normal[i].x;
				vertex->normal[1] = 0.0f;
				vertex->normal[2] = normal[i].z;
				vertex->tangent[0] = tangent_dir[i].x;
				vertex->tangent[1] = 0.0f;
				vertex->tangent[2] = tangent_dir[i].z;
				vertex->tangent[3] = handedness[i];
				vertex->texcoord[0] = s * scale;
				vertex->texcoord[1] = (y - floor_y) * scale;
			}
		}
		for (uint32_t ring = 0; ring + 1u < rings; ++ring)
			for (uint32_t i = 0; i < m; ++i)
			{
				uint32_t i1 = (i + 1u) % m;
				uint32_t a = base_vertex + ring * m + i;
				uint32_t b = base_vertex + ring * m + i1;
				uint32_t c = base_vertex + (ring + 1u) * m + i1;
				uint32_t d = base_vertex + (ring + 1u) * m + i;
				batch->indices[batch->index_count++] = a;
				batch->indices[batch->index_count++] = b;
				batch->indices[batch->index_count++] = c;
				batch->indices[batch->index_count++] = a;
				batch->indices[batch->index_count++] = c;
				batch->indices[batch->index_count++] = d;
			}
		free(arc);
		free(normal);
		free(tangent_dir);
		free(handedness);
	}
	return true;
}

/* A single triangle fan per puddle. The radial coordinate (0 at the
 * centre, 1 at the rim) is packed into texcoord.x for the puddle
 * shader's rim fade; the batch carries no texture, so texcoord.y is
 * unused. Per-angle noise on the rim radius keeps puddles reading as
 * irregular pools, not perfect discs. */
static bool append_puddle_fan(Builder *builder, DungeonPoint center,
							  float radius, float y,
							  uint32_t segments)
{
	DungeonMeshBatch *batch = builder->batch;
	if (batch->vertex_count + segments + 1u >
			builder->vertex_capacity ||
		batch->index_count + segments * 3u > builder->index_capacity)
		return false;
	uint32_t base = batch->vertex_count;
	Vertex *center_vertex = &batch->vertices[batch->vertex_count++];
	*center_vertex = (Vertex){.position = {center.x, y, center.z},
							  .normal = {0.0f, 1.0f, 0.0f},
							  .texcoord = {0.0f, 0.0f},
							  .tangent = {1.0f, 0.0f, 0.0f, -1.0f}};
	for (uint32_t i = 0; i < segments; ++i)
	{
		float angle = (float)i / (float)segments * 6.2831853f;
		float wobble = 1.0f +
					   0.12f * sinf(angle * 3.0f + center.x * 1.7f +
									center.z * 2.3f) +
					   0.06f * sinf(angle * 7.0f - center.z);
		float r = radius * wobble;
		Vertex *vertex = &batch->vertices[batch->vertex_count++];
		*vertex = (Vertex){.position = {center.x + cosf(angle) * r, y,
										center.z + sinf(angle) * r},
						   .normal = {0.0f, 1.0f, 0.0f},
						   .texcoord = {1.0f, 0.0f},
						   .tangent = {1.0f, 0.0f, 0.0f, -1.0f}};
	}
	for (uint32_t i = 0; i < segments; ++i)
	{
		uint32_t i1 = (i + 1u) % segments;
		batch->indices[batch->index_count++] = base;
		batch->indices[batch->index_count++] = base + 1u + i;
		batch->indices[batch->index_count++] = base + 1u + i1;
	}
	return true;
}

/* Deterministic foliage: sample the finished triangles so roots
 * conform to displaced walls as well as floors. No camera or frame
 * state participates. */
static float moss_random(uint32_t *state)
{
	*state = *state * 1664525u + 1013904223u;
	return (float)(*state >> 8) * (1.0f / 16777216.0f);
}

static float moss_patch(float x, float z)
{
	return sinf(x * 0.83f + sinf(z * 0.57f)) *
		   sinf(z * 0.91f + sinf(x * 0.43f));
}

static void moss_cross(const float a[3], const float b[3],
					   float out[3])
{
	out[0] = a[1] * b[2] - a[2] * b[1];
	out[1] = a[2] * b[0] - a[0] * b[2];
	out[2] = a[0] * b[1] - a[1] * b[0];
}

static void moss_normalize(float v[3])
{
	float length = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	if (length > 1e-8f)
		for (int k = 0; k < 3; ++k)
			v[k] /= length;
}

static bool append_moss_tuft(Builder *builder, const float root[3],
							 const float up[3], float size,
							 uint32_t *rng)
{
	float axis[3] = {0, 1, 0}, side[3], forward[3];
	if (fabsf(up[1]) > 0.9f)
	{
		axis[0] = 1;
		axis[1] = 0;
	}
	moss_cross(up, axis, side);
	moss_normalize(side);
	moss_cross(side, up, forward);
	float angle = moss_random(rng) * 6.2831853f;
	for (int frond = 0; frond < 3; ++frond)
	{
		float a = angle + frond * 2.0943951f;
		float right[3], bend[3];
		for (int k = 0; k < 3; ++k)
		{
			right[k] = side[k] * cosf(a) + forward[k] * sinf(a);
			bend[k] = forward[k] * cosf(a) - side[k] * sinf(a);
		}
		float h = size * (0.7f + moss_random(rng) * 0.6f);
		float u = moss_random(rng) * 0.8f,
			  v = moss_random(rng) * 0.8f;
		/* A segmented stem bends with the leaves and joins their
		 * roots. */
		for (int row = 0; row < 5; ++row)
		{
			float p[4][3];
			for (int j = 0; j < 4; ++j)
			{
				float t = (row + (j >= 2 ? 1.0f : 0.0f)) * 0.19f;
				float w =
					(j == 0 || j == 3 ? -1.0f : 1.0f) * h * 0.012f;
				for (int k = 0; k < 3; ++k)
					p[j][k] = root[k] + up[k] * (0.002f + h * t) +
							  bend[k] * h * t * t * 0.65f +
							  right[k] * w;
			}
			float e1[3], e2[3], n[3], tangent[4];
			for (int k = 0; k < 3; ++k)
			{
				e1[k] = p[1][k] - p[0][k];
				e2[k] = p[2][k] - p[0][k];
			}
			moss_cross(e1, e2, n);
			moss_normalize(n);
			moss_normalize(e1);
			for (int k = 0; k < 3; ++k)
				tangent[k] = e1[k];
			tangent[3] = 1;
			float uv[4][2] = {{u, v},
							  {u + 0.01f, v},
							  {u + 0.01f, v + 0.02f},
							  {u, v + 0.02f}};
			if (!append_quad(builder, p, n, tangent, uv))
				return false;
		}
		/* Five opposing leaf pairs along a bowed stem. Each diamond
		 * is geometry, so both the visible outline and shadow have
		 * leaf gaps. */
		for (int row = 0; row < 5; ++row)
			for (int sign = -1; sign <= 1; sign += 2)
			{
				float t = 0.12f + row * 0.17f;
				float width = h * (0.36f - t * 0.23f);
				float p[4][3];
				const float ts[4] = {t, t + 0.07f, t + 0.17f,
									 t + 0.10f};
				const float ws[4] = {0, 0.78f, 1, 0.16f};
				for (int j = 0; j < 4; ++j)
					for (int k = 0; k < 3; ++k)
						p[j][k] = root[k] +
								  up[k] * (0.004f + h * ts[j]) +
								  bend[k] * h *
									  (ts[j] * ts[j] * 0.65f +
									   ws[j] * 0.08f) +
								  right[k] * sign * width * ws[j];
				float e1[3], e2[3], n[3], tangent[4];
				for (int k = 0; k < 3; ++k)
				{
					e1[k] = p[1][k] - p[0][k];
					e2[k] = p[2][k] - p[0][k];
				}
				moss_cross(e1, e2, n);
				moss_normalize(n);
				moss_normalize(e1);
				for (int k = 0; k < 3; ++k)
					tangent[k] = e1[k];
				tangent[3] = 1;
				float uv[4][2] = {{u, v + t * 0.15f},
								  {u + 0.1f, v + t * 0.15f},
								  {u + 0.1f, v + (t + 0.17f) * 0.15f},
								  {u, v + (t + 0.17f) * 0.15f}};
				if (!append_quad(builder, p, n, tangent, uv))
					return false;
			}
	}
	return true;
}

static bool build_moss(const DungeonLevel *level,
					   DungeonMeshData *out)
{
	const uint32_t max_tufts = 4096u;
	Builder builder = {0};
	if (!builder_create(&builder, &out->batches[DUNGEON_MESH_MOSS],
						max_tufts * 180u, max_tufts * 270u, 2.0f))
		return false;
	uint32_t rng = 0x6d6f7373u, tufts = 0;
	for (int surface = DUNGEON_MESH_FLOOR;
		 surface <= DUNGEON_MESH_WALL; ++surface)
	{
		const DungeonMeshBatch *batch = &out->batches[surface];
		uint32_t surface_start = tufts, seen = 0;
		for (uint32_t tri = 0; tri < batch->index_count; tri += 3)
		{
			const Vertex *a = &batch->vertices[batch->indices[tri]];
			const Vertex *b =
				&batch->vertices[batch->indices[tri + 1]];
			const Vertex *c =
				&batch->vertices[batch->indices[tri + 2]];
			if (a->position[1] >= level->floor_y + 1.3f &&
				b->position[1] >= level->floor_y + 1.3f &&
				c->position[1] >= level->floor_y + 1.3f)
				continue;
			float ab[3], ac[3], cross[3];
			for (int k = 0; k < 3; ++k)
			{
				ab[k] = b->position[k] - a->position[k];
				ac[k] = c->position[k] - a->position[k];
			}
			moss_cross(ab, ac, cross);
			float area = 0.5f * sqrtf(cross[0] * cross[0] +
									  cross[1] * cross[1] +
									  cross[2] * cross[2]);
			float expected = area * 48.0f;
			uint32_t candidates = (uint32_t)expected;
			if (moss_random(&rng) < expected - candidates)
				++candidates;
			for (uint32_t sample = 0; sample < candidates; ++sample)
			{
				float u = moss_random(&rng), v = moss_random(&rng);
				if (u + v > 1)
				{
					u = 1 - u;
					v = 1 - v;
				}
				float root[3], n[3];
				for (int k = 0; k < 3; ++k)
				{
					root[k] = a->position[k] + ab[k] * u + ac[k] * v;
					n[k] = a->normal[k] * (1 - u - v) +
						   b->normal[k] * u + c->normal[k] * v;
				}
				float height = root[1] - level->floor_y;
				/* Sideways fronds need clearance above the floor. */
				if (surface == DUNGEON_MESH_WALL && height < 0.22f) continue;
				float patch = moss_patch(root[0], root[2]);
				if (patch < 0.30f || height > 1.25f || n[1] < -0.1f)
					continue;
				if (fabsf(root[0] - level->exit.x) < 0.8f &&
					fabsf(root[2] - level->exit.z) < 0.8f)
					continue;
				bool wet = false;
				for (uint32_t i = 0; i < level->puddle_count; ++i)
				{
					float dx = root[0] - level->puddles[i].center.x,
						  dz = root[2] - level->puddles[i].center.z;
					if (height < 0.03f &&
						dx * dx + dz * dz <
							level->puddles[i].radius *
								level->puddles[i].radius)
						wet = true;
				}
				if (wet || moss_random(&rng) >
							   fminf(1.0f, (patch - 0.30f) * 3.0f) *
								   (1 - height / 1.4f))
					continue;
				moss_normalize(n);
				float size = (0.09f + 0.10f * moss_random(&rng)) *
							 (0.5f + 0.5f * patch);
				/* Reservoir sampling keeps the budget spatially unbiased and
				 * reserves half for walls instead of exhausting it on floors. */
				uint32_t slot = seen++;
				if (slot >= max_tufts / 2u)
				{
					slot = (uint32_t)(moss_random(&rng) * seen);
					if (slot >= max_tufts / 2u) continue;
				}
				else ++tufts;
				builder.batch->vertex_count = (surface_start + slot) * 180u;
				builder.batch->index_count = (surface_start + slot) * 270u;
				if (!append_moss_tuft(&builder, root, n, size, &rng)) return false;
				builder.batch->vertex_count = tufts * 180u;
				builder.batch->index_count = tufts * 270u;
			}
		}
	}
	return true;
}

bool dungeon_mesh_build(const DungeonLevel *level,
						DungeonMeshData *out,
						DungeonLevelError *error)
{
	if (!level || !out)
		return mesh_fail(error, "level and output mesh are required");
	*out = (DungeonMeshData){0};

	const uint32_t wall_rings = 5u;
	uint32_t wall_vertex_total =
		level->plateau_triangles.vertex_count;
	uint32_t wall_index_total = level->plateau_triangles.index_count;
	for (uint32_t i = 0; i < level->contours.loop_count; ++i)
	{
		uint32_t m = level->contours.loops[i].point_count;
		if (m < 3u)
			continue;
		wall_vertex_total += m * wall_rings;
		wall_index_total += (wall_rings - 1u) * m * 6u;
	}
	const uint32_t puddle_segments = 24u;
	uint32_t puddle_vertex_total =
		level->puddle_count * (puddle_segments + 1u);
	uint32_t puddle_index_total =
		level->puddle_count * puddle_segments * 3u;

	Builder floor = {0}, wall = {0}, exit = {0}, player = {0},
			puddle = {0};
	if (!builder_create(&floor, &out->batches[DUNGEON_MESH_FLOOR],
						level->floor_triangles.vertex_count,
						level->floor_triangles.index_count, 2.4f) ||
		!builder_create(&wall, &out->batches[DUNGEON_MESH_WALL],
						wall_vertex_total, wall_index_total, 1.8f) ||
		!builder_create(&exit, &out->batches[DUNGEON_MESH_EXIT], 24u,
						36u, 2.0f) ||
		!builder_create(&player, &out->batches[DUNGEON_MESH_PLAYER],
						24u, 36u, 1.0f) ||
		!builder_create(&puddle, &out->batches[DUNGEON_MESH_PUDDLE],
						puddle_vertex_total, puddle_index_total,
						1.0f))
	{
		dungeon_mesh_destroy(out);
		return mesh_fail(error,
						 "out of memory building dungeon geometry");
	}

	if (!append_region_mesh(&floor, &level->floor_triangles,
							level->floor_y, true))
		goto capacity_error;
	if (!append_wall_loops(&wall, &level->contours, level->floor_y,
						   level->wall_height, wall_rings))
		goto capacity_error;
	if (!append_region_mesh(&wall, &level->plateau_triangles,
							level->floor_y + level->wall_height,
							true))
		goto capacity_error;
	DungeonRect exit_rect = {
		{level->exit.x - 0.65f, level->exit.z - 0.65f},
		{level->exit.x + 0.65f, level->exit.z + 0.65f}};
	if (!append_box(&exit, exit_rect, level->floor_y + 0.01f,
					level->floor_y + 0.13f))
		goto capacity_error;
	if (!append_box(&player,
					(DungeonRect){{-0.35f, -0.35f}, {0.35f, 0.35f}},
					0.0f, 0.7f))
		goto capacity_error;
	for (uint32_t i = 0; i < level->puddle_count; ++i)
		if (!append_puddle_fan(&puddle, level->puddles[i].center,
							   level->puddles[i].radius,
							   level->floor_y + 0.015f,
							   puddle_segments))
			goto capacity_error;

	if (!build_moss(level, out))
		goto capacity_error;
	return true;

capacity_error:
	dungeon_mesh_destroy(out);
	return mesh_fail(error, "internal dungeon mesh capacity error");
}

void dungeon_mesh_destroy(DungeonMeshData *data)
{
	if (!data)
		return;
	for (uint32_t i = 0; i < DUNGEON_MESH_BATCH_COUNT; ++i)
	{
		free(data->batches[i].vertices);
		free(data->batches[i].indices);
	}
	*data = (DungeonMeshData){0};
}
