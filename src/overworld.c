#include "overworld.h"

#include "terrain_tile.h"

#include "stb_image.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef DUNGEON_TEXTURE_DIR
#define DUNGEON_TEXTURE_DIR "textures/dungeon"
#endif

#define WALK_SPEED 4.5f
#define RUN_SPEED 9.0f
#define ENTRANCE_SPACING_M 220.0f
#define PI_F 3.14159265f

/* --- Height grid ------------------------------------------------------------ */

static void sample_world(const TerrainTile *tile, uint32_t x, uint32_t y, double out[3])
{
	uint32_t cells_x = tile->header.sample_width - 2u * tile->header.gutter - 1u;
	uint32_t cells_y = tile->header.sample_height - 2u * tile->header.gutter - 1u;
	double span_x = tile->header.extent[2] - tile->header.extent[0];
	double span_z = tile->header.extent[3] - tile->header.extent[1];
	float h = terrain_tile_height(tile, x + tile->header.gutter, y + tile->header.gutter);
	TileLocalPosition local = {(float)(x * span_x / cells_x - span_x * 0.5),
							   isfinite(h) ? h - tile->header.min_height_m : 0.0f,
							   (float)(y * span_z / cells_y - span_z * 0.5)};
	WorldPosition w = coordinate_local_to_world(&tile->header.local_to_world, local);
	out[0] = w.x;
	out[1] = isfinite(h) ? w.y : NAN;
	out[2] = w.z;
}

static bool load_heights(Overworld *world, const char *root)
{
	/* The finest level present: try 4, 3, 2 ... */
	int level = -1;
	char path[1024];
	for (int l = 6; l >= 0 && level < 0; --l)
	{
		snprintf(path, sizeof(path), "%s/tiles/%d/0/0.trn", root, l);
		FILE *f = fopen(path, "rb");
		if (f)
		{
			fclose(f);
			level = l;
		}
	}
	if (level < 0)
		return false;
	uint32_t tiles = 1u << (uint32_t)level;
	uint32_t cells = 0;
	for (uint32_t ty = 0; ty < tiles; ++ty)
		for (uint32_t tx = 0; tx < tiles; ++tx)
		{
			snprintf(path, sizeof(path), "%s/tiles/%d/%u/%u.trn", root, level, tx, ty);
			TerrainTile tile = {0};
			if (terrain_tile_load(path, &tile) != TERRAIN_TILE_OK)
			{
				fprintf(stderr, "Overworld: could not load %s\n", path);
				return false;
			}
			uint32_t tile_cells = tile.header.sample_width - 2u * tile.header.gutter - 1u;
			if (!world->heights)
			{
				cells = tile_cells;
				world->samples_u = world->samples_v = tiles * cells + 1u;
				size_t n = (size_t)world->samples_u * world->samples_v;
				world->heights = malloc(sizeof(float) * n);
				world->grass = calloc(n, sizeof(float));
				if (!world->heights || !world->grass)
				{
					terrain_tile_unload(&tile);
					return false;
				}
				for (size_t i = 0; i < n; ++i)
					world->heights[i] = NAN;
			}
			for (uint32_t y = 0; y <= cells; ++y)
				for (uint32_t x = 0; x <= cells; ++x)
				{
					double p[3];
					sample_world(&tile, x, y, p);
					uint32_t gu = tx * cells + x, gv = ty * cells + y;
					world->heights[(size_t)gv * world->samples_u + gu] = (float)p[1];
					if (tx == 0 && ty == 0 && y == 0 && x <= 1)
					{
						if (x == 0)
							memcpy(world->origin, p, sizeof(p));
						else
							for (int k = 0; k < 3; ++k)
								world->axis_u[k] = p[k] - world->origin[k];
					}
					if (tx == 0 && ty == 0 && x == 0 && y == 1)
						for (int k = 0; k < 3; ++k)
							world->axis_v[k] = p[k] - world->origin[k];
				}
			terrain_tile_unload(&tile);
		}
	world->axis_u[1] = world->axis_v[1] = 0.0;
	world->min_height = 1e9f;
	world->max_height = -1e9f;
	for (size_t i = 0; i < (size_t)world->samples_u * world->samples_v; ++i)
		if (isfinite(world->heights[i]))
		{
			world->min_height = fminf(world->min_height, world->heights[i]);
			world->max_height = fmaxf(world->max_height, world->heights[i]);
		}
	/* Grass coverage from the root imagery's alpha (the rock/grass classifier). */
	snprintf(path, sizeof(path), "%s/imagery/0/0/0.png", root);
	int w = 0, h = 0, channels = 0;
	unsigned char *rgba = stbi_load(path, &w, &h, &channels, 4);
	if (rgba && w > 2 && h > 2)
	{
		for (uint32_t v = 0; v < world->samples_v; ++v)
			for (uint32_t u = 0; u < world->samples_u; ++u)
			{
				/* One-texel gutter around the imagery. */
				int px = 1 + (int)((float)u / (float)(world->samples_u - 1u) * (float)(w - 3));
				int py = 1 + (int)((float)v / (float)(world->samples_v - 1u) * (float)(h - 3));
				world->grass[(size_t)v * world->samples_u + u] =
					rgba[((size_t)py * (size_t)w + (size_t)px) * 4u + 3u] / 255.0f;
			}
	}
	stbi_image_free(rgba);
	printf("Overworld: %ux%u height samples (level %d), %.0f..%.0f m, axes u=(%.2f,%.2f) "
		   "v=(%.2f,%.2f)\n",
		   world->samples_u, world->samples_v, level, world->min_height, world->max_height,
		   world->axis_u[0], world->axis_u[2], world->axis_v[0], world->axis_v[2]);
	return true;
}

/* World XZ to fractional grid coordinates. */
static bool to_grid(const Overworld *world, double x, double z, double *u, double *v)
{
	double dx = x - world->origin[0], dz = z - world->origin[2];
	double uu = world->axis_u[0] * world->axis_u[0] + world->axis_u[2] * world->axis_u[2];
	double vv = world->axis_v[0] * world->axis_v[0] + world->axis_v[2] * world->axis_v[2];
	*u = (dx * world->axis_u[0] + dz * world->axis_u[2]) / uu;
	*v = (dx * world->axis_v[0] + dz * world->axis_v[2]) / vv;
	return *u >= 0.0 && *v >= 0.0 && *u <= world->samples_u - 1.0 && *v <= world->samples_v - 1.0;
}

static void from_grid(const Overworld *world, double u, double v, double *x, double *z)
{
	*x = world->origin[0] + u * world->axis_u[0] + v * world->axis_v[0];
	*z = world->origin[2] + u * world->axis_u[2] + v * world->axis_v[2];
}

static float grid_sample(const Overworld *world, const float *field, double u, double v)
{
	uint32_t iu = (uint32_t)u, iv = (uint32_t)v;
	if (iu >= world->samples_u - 1u)
		iu = world->samples_u - 2u;
	if (iv >= world->samples_v - 1u)
		iv = world->samples_v - 2u;
	float fu = (float)(u - iu), fv = (float)(v - iv);
	const float *row0 = field + (size_t)iv * world->samples_u;
	const float *row1 = row0 + world->samples_u;
	/* Match the rendered triangle split (v00-v11 diagonal) rather than a
	 * bilinear patch, so the feet sit on the surface that is drawn. */
	if (fu >= fv)
		return row0[iu] + (row0[iu + 1] - row0[iu]) * fu + (row1[iu + 1] - row0[iu + 1]) * fv;
	return row0[iu] + (row1[iu] - row0[iu]) * fv + (row1[iu + 1] - row1[iu]) * fu;
}

float overworld_height(const Overworld *world, double x, double z)
{
	double u, v;
	if (!world->heights || !to_grid(world, x, z, &u, &v))
		return NAN;
	return grid_sample(world, world->heights, u, v);
}

/* Downhill direction and slope (rise over run) at a world point. */
static float slope_at(const Overworld *world, double x, double z, float *downhill)
{
	const double e = 3.0;
	float hx0 = overworld_height(world, x - e, z), hx1 = overworld_height(world, x + e, z);
	float hz0 = overworld_height(world, x, z - e), hz1 = overworld_height(world, x, z + e);
	if (!isfinite(hx0) || !isfinite(hx1) || !isfinite(hz0) || !isfinite(hz1))
		return INFINITY;
	float gx = (hx1 - hx0) / (float)(2.0 * e), gz = (hz1 - hz0) / (float)(2.0 * e);
	if (downhill)
		*downhill = atan2f(-gz, -gx);
	return sqrtf(gx * gx + gz * gz);
}

/* --- Entrance placement ------------------------------------------------------- */

static uint32_t hash_u32(uint32_t x)
{
	x ^= x >> 16;
	x *= 0x7feb352du;
	x ^= x >> 15;
	x *= 0x846ca68bu;
	x ^= x >> 16;
	return x;
}

static float random01(uint32_t *state)
{
	*state = hash_u32(*state + 0x9e3779b9u);
	return (float)(*state & 0xFFFFFFu) / (float)0x1000000u;
}

static OverworldGround classify(const Overworld *world, double x, double z)
{
	double u, v;
	to_grid(world, x, z, &u, &v);
	float h = overworld_height(world, x, z);
	float relief = fmaxf(world->max_height - world->min_height, 1.0f);
	float elevation = (h - world->min_height) / relief;
	float slope = slope_at(world, x, z, NULL);
	float grass = grid_sample(world, world->grass, u, v);
	if (elevation > 0.78f)
		return OVERWORLD_GROUND_SNOW;
	if (slope > 0.75f)
		return OVERWORLD_GROUND_CLIFF;
	if (grass > 0.30f)
		return OVERWORLD_GROUND_MEADOW;
	if (elevation < 0.25f)
		return OVERWORLD_GROUND_VALLEY;
	return OVERWORLD_GROUND_SCREE;
}

static void place_entrances(Overworld *world, uint32_t seed)
{
	uint32_t state = seed * 2654435761u + 17u;
	double centre_u = (world->samples_u - 1) * 0.5, centre_v = (world->samples_v - 1) * 0.5;
	double cx, cz;
	from_grid(world, centre_u, centre_v, &cx, &cz);
	bool used[OVERWORLD_GROUND_COUNT] = {0};
	uint32_t placed = 0;
	/* Rejection sampling, relaxing the rules if a map refuses: first demand a
	 * hillside (a hole needs a hill), different ground for every door and
	 * proper spacing; then let the ground repeat; then take any slope. */
	for (int pass = 0; pass < 3 && placed < OVERWORLD_ENTRANCES; ++pass)
		for (int attempt = 0; attempt < 6000 && placed < OVERWORLD_ENTRANCES; ++attempt)
		{
			/* Within the middle of the map, where the player starts. */
			double u = (0.12 + 0.76 * random01(&state)) * (world->samples_u - 1);
			double v = (0.12 + 0.76 * random01(&state)) * (world->samples_v - 1);
			double x, z;
			from_grid(world, u, v, &x, &z);
			float downhill = 0.0f;
			float slope = slope_at(world, x, z, &downhill);
			float min_slope = pass < 2 ? 0.22f : 0.08f, max_slope = pass < 2 ? 0.7f : 1.2f;
			if (!isfinite(slope) || slope < min_slope || slope > max_slope)
				continue;
			/* The doorstep itself must be walkable: flat enough just in front. */
			double fx = x + cos(downhill) * 3.0, fz = z + sin(downhill) * 3.0;
			float front_slope = slope_at(world, fx, fz, NULL);
			if (!isfinite(front_slope) || front_slope > 0.9f)
				continue;
			double distance_from_centre = hypot(x - cx, z - cz);
			if (distance_from_centre < 70.0 || distance_from_centre > 380.0)
				continue;
			bool crowded = false;
			for (uint32_t i = 0; i < placed; ++i)
				crowded |= hypot(x - world->entrances[i].position.x,
								 z - world->entrances[i].position.z) < ENTRANCE_SPACING_M;
			if (crowded)
				continue;
			OverworldGround ground = classify(world, x, z);
			if (pass == 0 && used[ground])
				continue;
			used[ground] = true;
			world->entrances[placed++] = (OverworldEntrance){
				.position = {x, overworld_height(world, x, z), z},
				.facing = downhill,
				.ground = ground,
				.elevation_m = overworld_height(world, x, z),
			};
		}
	/* Open on the whole map: centred, high, looking north-ish. */
	world->focus_x = cx;
	world->focus_z = cz;
	static const char *ground_names[] = {"snow", "cliff", "meadow", "valley", "scree"};
	for (uint32_t i = 0; i < placed; ++i)
		printf("Overworld: entrance %u at (%.1f, %.1f, %.1f) on %s ground, %.0f m from centre\n", i,
			   world->entrances[i].position.x, world->entrances[i].position.y,
			   world->entrances[i].position.z, ground_names[world->entrances[i].ground],
			   hypot(world->entrances[i].position.x - cx, world->entrances[i].position.z - cz));
}

/* --- Entrance geometry --------------------------------------------------------- */
/* Built in a local frame where the door looks down +Z, the threshold is the
 * origin and +Y is up. All four pieces are a single hobbit-hole: a turf-and-
 * stone mound bulging out of the hillside, a stone ring, a round plank door
 * with a knob in its middle, and a doorstep. */

typedef struct
{
	Vertex *v;
	uint32_t *i;
	uint32_t nv, ni, cap_v, cap_i;
} Builder;

static bool reserve(Builder *b, uint32_t nv, uint32_t ni)
{
	if (b->nv + nv > b->cap_v)
	{
		uint32_t cap = (b->cap_v + nv) * 2u;
		Vertex *v = realloc(b->v, sizeof(Vertex) * cap);
		if (!v)
			return false;
		b->v = v;
		b->cap_v = cap;
	}
	if (b->ni + ni > b->cap_i)
	{
		uint32_t cap = (b->cap_i + ni) * 2u;
		uint32_t *i = realloc(b->i, sizeof(uint32_t) * cap);
		if (!i)
			return false;
		b->i = i;
		b->cap_i = cap;
	}
	return true;
}

static uint32_t vert(Builder *b, float px, float py, float pz, float nx, float ny, float nz, float u,
					 float v)
{
	float len = sqrtf(nx * nx + ny * ny + nz * nz);
	if (len > 0.0f)
	{
		nx /= len;
		ny /= len;
		nz /= len;
	}
	b->v[b->nv] = (Vertex){.position = {px, py, pz}, .normal = {nx, ny, nz}, .texcoord = {u, v}};
	return b->nv++;
}

static void quad(Builder *b, uint32_t a, uint32_t c, uint32_t d, uint32_t e)
{
	/* a-c-d-e counter-clockwise seen from the front. */
	b->i[b->ni++] = a;
	b->i[b->ni++] = c;
	b->i[b->ni++] = d;
	b->i[b->ni++] = a;
	b->i[b->ni++] = d;
	b->i[b->ni++] = e;
}

/* Per-vertex tangents from UV gradients (Lengyel). */
static void build_tangents(Builder *b)
{
	float(*tan)[3] = calloc(b->nv, sizeof(*tan));
	float(*bit)[3] = calloc(b->nv, sizeof(*bit));
	if (!tan || !bit)
	{
		free(tan);
		free(bit);
		return;
	}
	for (uint32_t t = 0; t + 2 < b->ni; t += 3)
	{
		Vertex *p[3] = {&b->v[b->i[t]], &b->v[b->i[t + 1]], &b->v[b->i[t + 2]]};
		float e1[3], e2[3];
		for (int k = 0; k < 3; ++k)
		{
			e1[k] = p[1]->position[k] - p[0]->position[k];
			e2[k] = p[2]->position[k] - p[0]->position[k];
		}
		float du1 = p[1]->texcoord[0] - p[0]->texcoord[0], dv1 = p[1]->texcoord[1] - p[0]->texcoord[1];
		float du2 = p[2]->texcoord[0] - p[0]->texcoord[0], dv2 = p[2]->texcoord[1] - p[0]->texcoord[1];
		float det = du1 * dv2 - du2 * dv1;
		if (fabsf(det) < 1e-12f)
			continue;
		float r = 1.0f / det;
		for (int c = 0; c < 3; ++c)
			for (int k = 0; k < 3; ++k)
			{
				tan[b->i[t + c]][k] += (e1[k] * dv2 - e2[k] * dv1) * r;
				bit[b->i[t + c]][k] += (e2[k] * du1 - e1[k] * du2) * r;
			}
	}
	for (uint32_t i = 0; i < b->nv; ++i)
	{
		float *n = b->v[i].normal, *t = tan[i];
		float d = n[0] * t[0] + n[1] * t[1] + n[2] * t[2];
		float o[3] = {t[0] - n[0] * d, t[1] - n[1] * d, t[2] - n[2] * d};
		float len = sqrtf(o[0] * o[0] + o[1] * o[1] + o[2] * o[2]);
		if (len < 1e-8f)
		{
			/* Any perpendicular will do where the UVs are degenerate. */
			o[0] = fabsf(n[0]) < 0.9f ? 1.0f : 0.0f;
			o[1] = fabsf(n[0]) < 0.9f ? 0.0f : 1.0f;
			o[2] = 0.0f;
			len = 1.0f;
		}
		float c[3] = {n[1] * o[2] - n[2] * o[1], n[2] * o[0] - n[0] * o[2], n[0] * o[1] - n[1] * o[0]};
		float w = c[0] * bit[i][0] + c[1] * bit[i][1] + c[2] * bit[i][2] < 0.0f ? -1.0f : 1.0f;
		b->v[i].tangent[0] = o[0] / len;
		b->v[i].tangent[1] = o[1] / len;
		b->v[i].tangent[2] = o[2] / len;
		b->v[i].tangent[3] = w;
	}
	free(tan);
	free(bit);
}

/* Half-ellipsoid mound bulging from the hillside behind the door, with the
 * front flattened into a vertical face for the ring to sit in. */
static bool build_mound(Builder *b)
{
	const int rings = 18, segments = 40;
	const float rx = 3.6f, ry = 3.0f, rz = 3.4f, centre_z = -1.7f, front = -0.3f;
	if (!reserve(b, (uint32_t)((rings + 1) * (segments + 1)), (uint32_t)(rings * segments * 6)))
		return false;
	uint32_t base = b->nv;
	for (int r = 0; r <= rings; ++r)
	{
		float phi = (float)r / rings * (PI_F * 0.5f); /* 0 at the ground, pi/2 at the top */
		for (int s = 0; s <= segments; ++s)
		{
			float theta = (float)s / segments * 2.0f * PI_F;
			float x = rx * cosf(phi) * cosf(theta);
			float y = ry * sinf(phi) - 0.6f; /* sink the base into the slope */
			float z = centre_z + rz * cosf(phi) * sinf(theta);
			float nz = cosf(phi) * sinf(theta) / rz;
			/* Flatten the front: everything forward of the facade plane is
			 * pushed back onto it, with a vertical normal facing out. */
			if (z > front)
			{
				z = front;
				nz = 4.0f;
			}
			vert(b, x, y, z, cosf(phi) * cosf(theta) / rx, sinf(phi) / ry, nz,
				 theta * rx / 1.6f, phi * ry / 1.6f);
		}
	}
	for (int r = 0; r < rings; ++r)
		for (int s = 0; s < segments; ++s)
		{
			uint32_t a = base + (uint32_t)(r * (segments + 1) + s);
			uint32_t n = a + (uint32_t)(segments + 1);
			quad(b, a, a + 1, n + 1, n);
		}
	return true;
}

/* Annulus extruded along Z: stone ring (or the door, with inner radius 0). */
static bool build_annulus(Builder *b, float inner, float outer, float z0, float z1, float cy,
						  float uv_scale)
{
	const int segments = 48;
	if (!reserve(b, (uint32_t)(segments + 1) * 8u, (uint32_t)segments * 24u))
		return false;
	for (int s = 0; s < segments; ++s)
	{
		float a0 = (float)s / segments * 2.0f * PI_F, a1 = (float)(s + 1) / segments * 2.0f * PI_F;
		float c0 = cosf(a0), s0 = sinf(a0), c1 = cosf(a1), s1 = sinf(a1);
		/* Front and back faces use planar UVs so plank/stone texture runs
		 * straight across; the rims use arc length. */
		for (int face = 0; face < 2; ++face)
		{
			float z = face ? z0 : z1, nz = face ? -1.0f : 1.0f;
			uint32_t p0 = vert(b, inner * c0, cy + inner * s0, z, 0, 0, nz, inner * c0 * uv_scale,
							   (cy + inner * s0) * uv_scale);
			uint32_t p1 = vert(b, outer * c0, cy + outer * s0, z, 0, 0, nz, outer * c0 * uv_scale,
							   (cy + outer * s0) * uv_scale);
			uint32_t p2 = vert(b, outer * c1, cy + outer * s1, z, 0, 0, nz, outer * c1 * uv_scale,
							   (cy + outer * s1) * uv_scale);
			uint32_t p3 = vert(b, inner * c1, cy + inner * s1, z, 0, 0, nz, inner * c1 * uv_scale,
							   (cy + inner * s1) * uv_scale);
			if (face)
				quad(b, p0, p3, p2, p1);
			else
				quad(b, p0, p1, p2, p3);
		}
		float arc0 = a0 * outer * uv_scale, arc1 = a1 * outer * uv_scale;
		uint32_t o0 = vert(b, outer * c0, cy + outer * s0, z1, c0, s0, 0, arc0, z1 * uv_scale);
		uint32_t o1 = vert(b, outer * c1, cy + outer * s1, z1, c1, s1, 0, arc1, z1 * uv_scale);
		uint32_t o2 = vert(b, outer * c1, cy + outer * s1, z0, c1, s1, 0, arc1, z0 * uv_scale);
		uint32_t o3 = vert(b, outer * c0, cy + outer * s0, z0, c0, s0, 0, arc0, z0 * uv_scale);
		quad(b, o0, o3, o2, o1);
		if (inner > 0.0f)
		{
			uint32_t i0 = vert(b, inner * c0, cy + inner * s0, z1, -c0, -s0, 0, arc0, z1 * uv_scale);
			uint32_t i1 = vert(b, inner * c1, cy + inner * s1, z1, -c1, -s1, 0, arc1, z1 * uv_scale);
			uint32_t i2 = vert(b, inner * c1, cy + inner * s1, z0, -c1, -s1, 0, arc1, z0 * uv_scale);
			uint32_t i3 = vert(b, inner * c0, cy + inner * s0, z0, -c0, -s0, 0, arc0, z0 * uv_scale);
			quad(b, i0, i1, i2, i3);
		}
	}
	return true;
}

static bool build_box(Builder *b, float x0, float y0, float z0, float x1, float y1, float z1,
					  float uv_scale)
{
	if (!reserve(b, 24, 36))
		return false;
	const float n[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
	for (int f = 0; f < 6; ++f)
	{
		float c[4][3];
		int axis = f / 2;
		float sign = (f & 1) ? -1.0f : 1.0f;
		float fixed = sign > 0 ? (axis == 0 ? x1 : axis == 1 ? y1 : z1) : (axis == 0 ? x0 : axis == 1 ? y0 : z0);
		int ua = (axis + 1) % 3, va = (axis + 2) % 3;
		float lo[3] = {x0, y0, z0}, hi[3] = {x1, y1, z1};
		float corners[4][2] = {{lo[ua], lo[va]}, {hi[ua], lo[va]}, {hi[ua], hi[va]}, {lo[ua], hi[va]}};
		uint32_t idx[4];
		for (int k = 0; k < 4; ++k)
		{
			c[k][axis] = fixed;
			c[k][ua] = corners[k][0];
			c[k][va] = corners[k][1];
			idx[k] = vert(b, c[k][0], c[k][1], c[k][2], n[f][0], n[f][1], n[f][2],
						  corners[k][0] * uv_scale, corners[k][1] * uv_scale);
		}
		if (sign > 0)
			quad(b, idx[0], idx[1], idx[2], idx[3]);
		else
			quad(b, idx[0], idx[3], idx[2], idx[1]);
	}
	return true;
}

static bool upload(Renderer *renderer, Overworld *world, int which, Builder *b, const char *albedo,
				   const char *orm, const char *normal)
{
	build_tangents(b);
	world->meshes[which] = (Mesh){
		.local_to_world = coordinate_identity_transform((WorldPosition){0}),
		.vertices = b->v,
		.vertex_count = b->nv,
		.indices = b->i,
		.index_count = b->ni,
		.texture_path = albedo,
		.orm_path = orm,
		.normal_path = normal,
	};
	mesh_upload(renderer, &world->meshes[which]);
	world->uploaded[which] = true;
	return true;
}

static bool build_entrance_meshes(Renderer *renderer, Overworld *world)
{
	Builder mound = {0}, ring = {0}, door = {0}, knob = {0}, step = {0};
	/* Door: a round plank disc, its centre 1.05 m up so it meets the ground. */
	const float door_radius = 1.05f, door_cy = 1.0f;
	bool ok = build_mound(&mound) &&
			  build_annulus(&ring, door_radius - 0.02f, door_radius + 0.42f, -0.45f, 0.32f,
							door_cy, 0.55f) &&
			  build_annulus(&door, 0.0f, door_radius, -0.08f, 0.0f, door_cy, 0.6f) &&
			  build_box(&knob, -0.07f, door_cy - 0.07f, 0.0f, 0.07f, door_cy + 0.07f, 0.16f, 2.0f) &&
			  build_box(&step, -1.3f, -0.6f, 0.3f, 1.3f, 0.06f, 1.5f, 0.6f);
	if (!ok)
	{
		free(mound.v), free(mound.i), free(ring.v), free(ring.i), free(door.v), free(door.i);
		free(knob.v), free(knob.i), free(step.v), free(step.i);
		return false;
	}
	upload(renderer, world, OVERWORLD_MESH_MOUND, &mound, DUNGEON_TEXTURE_DIR "/wall_albedo.jpg",
		   DUNGEON_TEXTURE_DIR "/wall_orm.png", DUNGEON_TEXTURE_DIR "/wall_normal.png");
	upload(renderer, world, OVERWORLD_MESH_RING, &ring, DUNGEON_TEXTURE_DIR "/floor_albedo.jpg",
		   DUNGEON_TEXTURE_DIR "/floor_orm.png", DUNGEON_TEXTURE_DIR "/floor_normal.png");
	upload(renderer, world, OVERWORLD_MESH_DOOR, &door, DUNGEON_TEXTURE_DIR "/exit_albedo.jpg",
		   DUNGEON_TEXTURE_DIR "/exit_orm.png", DUNGEON_TEXTURE_DIR "/exit_normal.png");
	upload(renderer, world, OVERWORLD_MESH_KNOB, &knob, DUNGEON_TEXTURE_DIR "/lock_albedo.jpg",
		   DUNGEON_TEXTURE_DIR "/lock_orm.png", DUNGEON_TEXTURE_DIR "/lock_normal.png");
	upload(renderer, world, OVERWORLD_MESH_STEP, &step, DUNGEON_TEXTURE_DIR "/floor_albedo.jpg",
		   DUNGEON_TEXTURE_DIR "/floor_orm.png", DUNGEON_TEXTURE_DIR "/floor_normal.png");
	return true;
}

/* --- Lifecycle ------------------------------------------------------------------ */

bool overworld_create(Renderer *renderer, Overworld *out, const char *dataset_root, uint32_t seed)
{
	*out = (Overworld){0};
	if (!load_heights(out, dataset_root))
	{
		fprintf(stderr, "Overworld: no terrain tiles under %s\n", dataset_root);
		overworld_destroy(NULL, out);
		return false;
	}
	place_entrances(out, seed);
	out->pitch = -58.0f;
	out->yaw = -90.0f;
	out->distance = 950.0f;
	if (!build_entrance_meshes(renderer, out))
	{
		overworld_destroy(renderer, out);
		return false;
	}
	return true;
}

void overworld_destroy(Renderer *renderer, Overworld *world)
{
	for (int i = 0; i < OVERWORLD_MESH_COUNT; ++i)
		if (world->uploaded[i])
		{
			if (renderer)
				mesh_destroy(renderer, &world->meshes[i]);
			free((void *)world->meshes[i].vertices);
			free((void *)world->meshes[i].indices);
		}
	free(world->heights);
	free(world->grass);
	*world = (Overworld){0};
}

/* --- Map camera ----------------------------------------------------------------- */

#define MIN_DISTANCE 18.0f
#define MAX_DISTANCE 1400.0f

static void clamp_focus(Overworld *world)
{
	double u, v;
	to_grid(world, world->focus_x, world->focus_z, &u, &v);
	double max_u = world->samples_u - 1.0, max_v = world->samples_v - 1.0;
	u = u < 0.0 ? 0.0 : (u > max_u ? max_u : u);
	v = v < 0.0 ? 0.0 : (v > max_v ? max_v : v);
	from_grid(world, u, v, &world->focus_x, &world->focus_z);
}

void overworld_update(Overworld *world, float drag_dx, float drag_dy, float rotate_dx, float zoom,
					  float pan_forward, float pan_right, float viewport_height_px, float dt,
					  bool controls)
{
	world->time += dt;
	if (!controls)
	{
		world->yaw += dt * 3.0f; /* a slow turn behind the menus */
		world->flying = false;
		return;
	}
	float yaw = world->yaw * PI_F / 180.0f;
	float fx = cosf(yaw), fz = sinf(yaw); /* ground-plane forward */
	float rx = -fz, rz = fx;			  /* ground-plane right */
	bool user_moved = drag_dx != 0.0f || drag_dy != 0.0f || zoom != 0.0f || rotate_dx != 0.0f ||
					  pan_forward != 0.0f || pan_right != 0.0f;
	if (user_moved)
		world->flying = false;
	/* Grab-the-ground drag: one pixel moves the focus by the ground distance
	 * one pixel covers at the focus, so the terrain tracks the pointer. */
	float metres_per_pixel = 2.0f * world->distance * tanf(30.0f * PI_F / 180.0f) /
							 fmaxf(viewport_height_px, 1.0f);
	float pitch_stretch = 1.0f / fmaxf(sinf(-world->pitch * PI_F / 180.0f), 0.3f);
	world->focus_x -= (double)(rx * drag_dx - fx * drag_dy * pitch_stretch) * metres_per_pixel;
	world->focus_z -= (double)(rz * drag_dx - fz * drag_dy * pitch_stretch) * metres_per_pixel;
	float pan_speed = world->distance * 0.9f;
	world->focus_x += (double)(fx * pan_forward + rx * pan_right) * pan_speed * dt;
	world->focus_z += (double)(fz * pan_forward + rz * pan_right) * pan_speed * dt;
	world->yaw += rotate_dx * 0.25f;
	/* Each notch is a fixed ratio, like every map: zoom feels the same at any
	 * height. Lower in means a flatter, more scenic view of the doors. */
	world->distance = fminf(MAX_DISTANCE, fmaxf(MIN_DISTANCE, world->distance * powf(0.85f, zoom)));
	if (world->flying)
	{
		float k = 1.0f - expf(-3.5f * dt);
		world->focus_x += (world->fly_x - world->focus_x) * k;
		world->focus_z += (world->fly_z - world->focus_z) * k;
		world->distance += (world->fly_distance - world->distance) * k;
		if (fabs(world->fly_x - world->focus_x) < 0.05 && fabsf(world->fly_distance - world->distance) < 0.1f)
			world->flying = false;
	}
	float t = (world->distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE);
	world->pitch = -(28.0f + 35.0f * sqrtf(fmaxf(t, 0.0f)));
	clamp_focus(world);
}

void overworld_fly_to(Overworld *world, uint32_t i)
{
	const OverworldEntrance *e = &world->entrances[i % OVERWORLD_ENTRANCES];
	world->flying = true;
	/* Aim a little in front of the door so the mound sits mid-frame. */
	world->fly_x = e->position.x + cos(e->facing) * 2.0;
	world->fly_z = e->position.z + sin(e->facing) * 2.0;
	world->fly_distance = 30.0f;
	/* Turn to face the door head-on: the camera looks back along its facing. */
	float want = e->facing * 180.0f / PI_F + 180.0f;
	world->yaw += remainderf(want - world->yaw, 360.0f);
}

Camera overworld_camera(const Overworld *world)
{
	float ground = overworld_height(world, world->focus_x, world->focus_z);
	if (!isfinite(ground))
		ground = world->min_height;
	float yaw = world->yaw * PI_F / 180.0f, pitch = world->pitch * PI_F / 180.0f;
	double back = world->distance * cosf(pitch);
	Camera camera = {.yaw = world->yaw, .pitch = world->pitch};
	camera.position = (WorldPosition){world->focus_x - cos(yaw) * back,
									  ground - sin(pitch) * world->distance,
									  world->focus_z - sin(yaw) * back};
	float below = overworld_height(world, camera.position.x, camera.position.z);
	if (isfinite(below) && camera.position.y < below + 4.0)
		camera.position.y = below + 4.0;
	return camera;
}

bool overworld_project(const Overworld *world, const Camera *camera, float aspect,
					   WorldPosition point, float *out_u, float *out_v)
{
	(void)world;
	mat4s view = camera_view(camera);
	mat4s projection = camera_projection(camera, aspect);
	vec4s relative = {{(float)(point.x - camera->position.x), (float)(point.y - camera->position.y),
					   (float)(point.z - camera->position.z), 1.0f}};
	vec4s clip = glms_mat4_mulv(glms_mat4_mul(projection, view), relative);
	if (clip.w <= 0.01f)
		return false;
	*out_u = 0.5f + 0.5f * clip.x / clip.w;
	*out_v = 0.5f + 0.5f * clip.y / clip.w; /* Vulkan clip y already points down */
	return true;
}

static DrawPushConstants entrance_push(const LocalToWorldTransform *transform,
									   WorldPosition camera)
{
	return (DrawPushConstants){
		.local_to_camera_relative = coordinate_local_to_camera_relative(transform, camera),
		.geometry = {{1.0f, 1.0f, 1.0f, 1.0f}},
		.elevation_uv = {{1.0f, 1.0f, 1.0f, 0.0f}},
		.material = {{1.0f, 1.0f, 1.0f, 1.0f}},
		.debug = {{0.0f, 0.0f, 1.0f, 1.0f}},
	};
}

uint32_t overworld_draws(Overworld *world, WorldPosition camera, RendererDraw *out, uint32_t capacity)
{
	uint32_t count = 0;
	for (uint32_t e = 0; e < OVERWORLD_ENTRANCES; ++e)
	{
		const OverworldEntrance *entrance = &world->entrances[e];
		/* Local +Z (the door's outward normal) onto the entrance facing:
		 * rotation_y(a) sends +Z to (sin a, cos a), so a = pi/2 - facing. */
		LocalToWorldTransform transform =
			coordinate_rotation_y(PI_F * 0.5 - entrance->facing,
								  (WorldPosition){entrance->position.x, entrance->position.y - 0.12,
												  entrance->position.z});
		for (int m = 0; m < OVERWORLD_MESH_COUNT && count < capacity; ++m)
		{
			if (!world->uploaded[m])
				continue;
			out[count++] = (RendererDraw){.mesh = &world->meshes[m],
										  .material_set = world->meshes[m].material_set,
										  .push = entrance_push(&transform, camera),
										  .static_mesh = true};
		}
	}
	return count;
}
