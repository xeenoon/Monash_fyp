#include "dungeon_shadow.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static float centre(const DungeonShadowTriangle *t, int axis)
{
	return (t->a[axis] + t->b[axis] + t->c[axis]) / 3.f;
}
static uint32_t build(DungeonShadow *s, DungeonShadowTriangle *t,
					  uint32_t n)
{
	uint32_t index = s->count++;
	DungeonShadowNode *node = &s->nodes[index];
	for (int k = 0; k < 3; k++)
	{
		node->minimum[k] = FLT_MAX;
		node->maximum[k] = -FLT_MAX;
		for (uint32_t i = 0; i < n; i++)
		{
			node->minimum[k] =
				fminf(node->minimum[k],
					  fminf(t[i].a[k], fminf(t[i].b[k], t[i].c[k])));
			node->maximum[k] =
				fmaxf(node->maximum[k],
					  fmaxf(t[i].a[k], fmaxf(t[i].b[k], t[i].c[k])));
		}
		node->minimum[k] -= 0.0001f;
		node->maximum[k] += 0.0001f;
	}
	if (n == 1)
	{
		node->leaf = 1;
		memcpy(node->a, t->a, 12);
		memcpy(node->b, t->b, 12);
		memcpy(node->c, t->c, 12);
	}
	else
	{
		int axis = 0;
		for (int k = 1; k < 3; k++)
			if (node->maximum[k] - node->minimum[k] >
				node->maximum[axis] - node->minimum[axis])
				axis = k;
		float split =
			(node->minimum[axis] + node->maximum[axis]) * 0.5f;
		uint32_t mid = 0;
		for (uint32_t i = 0; i < n; i++)
			if (centre(&t[i], axis) < split)
			{
				DungeonShadowTriangle tmp = t[mid];
				t[mid++] = t[i];
				t[i] = tmp;
			}
		/* Balanced fallback also bounds recursion for pathological
		 * meshes. */
		if (mid < n / 4 || mid > n - n / 4 || !mid || mid == n)
			mid = n / 2;
		build(s, t, mid);
		build(s, t + mid, n - mid);
	}
	node->end = s->count;
	return index;
}
bool dungeon_shadow_build(const DungeonShadowTriangle *triangles,
						  uint32_t count, DungeonShadow *out)
{
	*out = (DungeonShadow){0};
	if (!count)
		return true;
	if (!triangles || count > UINT32_MAX / 2)
		return false;
	DungeonShadowTriangle *copy =
		malloc((size_t)count * sizeof(*copy));
	out->nodes = calloc((size_t)count * 2, sizeof(*out->nodes));
	if (!copy || !out->nodes)
	{
		free(copy);
		dungeon_shadow_destroy(out);
		return false;
	}
	memcpy(copy, triangles, (size_t)count * sizeof(*copy));
	build(out, copy, count);
	free(copy);
	return true;
}
void dungeon_shadow_destroy(DungeonShadow *s)
{
	free(s->nodes);
	*s = (DungeonShadow){0};
}
static float dot(const float a[3], const float b[3])
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
static void cross(const float a[3], const float b[3], float r[3])
{
	r[0] = a[1] * b[2] - a[2] * b[1];
	r[1] = a[2] * b[0] - a[0] * b[2];
	r[2] = a[0] * b[1] - a[1] * b[0];
}
bool dungeon_shadow_trace(const DungeonShadow *s, const float o[3],
						  const float d[3], float *limit,
						  float normal[3])
{
	bool hit = false;
	for (uint32_t i = 0; i < s->count;)
	{
		const DungeonShadowNode *n = &s->nodes[i];
		float lo = 0, hi = *limit;
		for (int k = 0; k < 3; k++)
		{
			if (fabsf(d[k]) < 1e-8f)
			{
				if (o[k] < n->minimum[k] || o[k] > n->maximum[k])
					hi = -1;
			}
			else
			{
				float a = (n->minimum[k] - o[k]) / d[k],
					  b = (n->maximum[k] - o[k]) / d[k];
				lo = fmaxf(lo, fminf(a, b));
				hi = fminf(hi, fmaxf(a, b));
			}
		}
		if (hi < lo)
		{
			i = n->end;
			continue;
		}
		++i;
		if (!n->leaf)
			continue;
		float e1[3], e2[3], p[3], q[3], v[3];
		for (int k = 0; k < 3; k++)
		{
			e1[k] = n->b[k] - n->a[k];
			e2[k] = n->c[k] - n->a[k];
			v[k] = o[k] - n->a[k];
		}
		cross(d, e2, p);
		float det = dot(e1, p);
		if (fabsf(det) < 1e-8f)
			continue;
		float u = dot(v, p) / det;
		if (u < 0 || u > 1)
			continue;
		cross(v, e1, q);
		float w = dot(d, q) / det;
		if (w < 0 || u + w > 1)
			continue;
		float t = dot(e2, q) / det;
		if (t <= 0.0001f || t >= *limit)
			continue;
		*limit = t;
		hit = true;
		cross(e1, e2, normal);
		float length = sqrtf(dot(normal, normal));
		for (int k = 0; k < 3; k++)
			normal[k] /= length;
	}
	return hit;
}
