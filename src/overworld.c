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
#ifndef DUNGEON_TORCH_PATH
#define DUNGEON_TORCH_PATH "assets/dungeons/torch/walltorch.gltf"
#endif

#define WALK_SPEED 4.5f
#define RUN_SPEED 9.0f
#define MIN_DISTANCE 18.0f
#define MAX_DISTANCE 4200.0f
#define PI_F 3.14159265f

static const Vertex DESCENT_FLAME_VERTICES[4] = {
	{{-0.5f, -0.5f, 0}, {0, 0, 1}, {0, 0}, 0, {1, 0, 0, 1}},
	{{0.5f, -0.5f, 0}, {0, 0, 1}, {1, 0}, 0, {1, 0, 0, 1}},
	{{0.5f, 0.5f, 0}, {0, 0, 1}, {1, 1}, 0, {1, 0, 0, 1}},
	{{-0.5f, 0.5f, 0}, {0, 0, 1}, {0, 1}, 0, {1, 0, 0, 1}},
};
static const uint32_t DESCENT_FLAME_INDICES[6] = {0, 1, 2, 0, 2, 3};

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

/* Reads "extent": [west, south, east, north] from the dataset manifest. */
static bool manifest_extent(const char *root, double extent[4])
{
	char path[1024];
	snprintf(path, sizeof(path), "%s/manifest.json", root);
	FILE *file = fopen(path, "rb");
	if (!file)
		return false;
	char text[16384] = {0};
	size_t n = fread(text, 1, sizeof(text) - 1, file);
	fclose(file);
	text[n] = '\0';
	const char *at = strstr(text, "\"extent\"");
	if (!at || !(at = strchr(at, '[')))
		return false;
	return sscanf(at, "[ %lf , %lf , %lf , %lf ]", &extent[0], &extent[1], &extent[2],
				  &extent[3]) == 4;
}

/* Loads the finest tile level under the square region centred on
 * (centre_x, centre_z) into one height grid, plus a grass and a snow estimate
 * per sample from the same tiles' imagery. World X is easting and world Z is
 * minus northing; tile y = 0 is the dataset's north edge. */
static bool load_heights(Overworld *world, const char *root, double centre_x, double centre_z,
						 double half_extent)
{
	int level = -1;
	char path[1024];
	for (int l = 8; l >= 0 && level < 0; --l)
	{
		snprintf(path, sizeof(path), "%s/tiles/%d", root, l);
		FILE *probe = fopen(path, "rb");
		if (probe)
		{
			fclose(probe);
			level = l;
		}
	}
	double extent[4];
	if (level < 0 || !manifest_extent(root, extent))
		return false;
	uint32_t tiles = 1u << (uint32_t)level;
	double tile_w = (extent[2] - extent[0]) / tiles, tile_h = (extent[3] - extent[1]) / tiles;
	double northing = -centre_z;
	int tx0 = half_extent > 0.0 ? (int)floor((centre_x - half_extent - extent[0]) / tile_w) : 0;
	int tx1 = half_extent > 0.0 ? (int)floor((centre_x + half_extent - extent[0]) / tile_w)
								 : (int)tiles - 1;
	int ty0 = half_extent > 0.0 ? (int)floor((extent[3] - (northing + half_extent)) / tile_h) : 0;
	int ty1 = half_extent > 0.0 ? (int)floor((extent[3] - (northing - half_extent)) / tile_h)
								 : (int)tiles - 1;
	tx0 = tx0 < 0 ? 0 : tx0;
	ty0 = ty0 < 0 ? 0 : ty0;
	tx1 = tx1 >= (int)tiles ? (int)tiles - 1 : tx1;
	ty1 = ty1 >= (int)tiles ? (int)tiles - 1 : ty1;
	uint32_t cells = 0;
	for (int ty = ty0; ty <= ty1; ++ty)
		for (int tx = tx0; tx <= tx1; ++tx)
		{
			snprintf(path, sizeof(path), "%s/tiles/%d/%d/%d.trn", root, level, tx, ty);
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
				world->samples_u = (uint32_t)(tx1 - tx0 + 1) * cells + 1u;
				world->samples_v = (uint32_t)(ty1 - ty0 + 1) * cells + 1u;
				size_t count = (size_t)world->samples_u * world->samples_v;
				world->heights = malloc(sizeof(float) * count);
				world->grass = calloc(count, sizeof(float));
				world->snow = calloc(count, sizeof(float));
				if (!world->heights || !world->grass || !world->snow)
				{
					terrain_tile_unload(&tile);
					return false;
				}
				for (size_t i = 0; i < count; ++i)
					world->heights[i] = NAN;
			}
			snprintf(path, sizeof(path), "%s/imagery/%d/%d/%d.png", root, level, tx, ty);
			int iw = 0, ih = 0, channels = 0;
			unsigned char *rgb = stbi_load(path, &iw, &ih, &channels, 3);
			for (uint32_t y = 0; y <= cells; ++y)
				for (uint32_t x = 0; x <= cells; ++x)
				{
					double p[3];
					sample_world(&tile, x, y, p);
					uint32_t gu = (uint32_t)(tx - tx0) * cells + x;
					uint32_t gv = (uint32_t)(ty - ty0) * cells + y;
					size_t index = (size_t)gv * world->samples_u + gu;
					world->heights[index] = (float)p[1];
					if (tx == tx0 && ty == ty0 && y == 0 && x <= 1)
					{
						if (x == 0)
							memcpy(world->origin, p, sizeof(p));
						else
							for (int k = 0; k < 3; ++k)
								world->axis_u[k] = p[k] - world->origin[k];
					}
					if (tx == tx0 && ty == ty0 && x == 0 && y == 1)
						for (int k = 0; k < 3; ++k)
							world->axis_v[k] = p[k] - world->origin[k];
					if (rgb && iw > 2 && ih > 2)
					{
						/* One-texel gutter; average a 5x5 patch so a single
						 * rock or tree does not decide the ground. */
						int cx = 1 + (int)((float)x / (float)cells * (float)(iw - 3));
						int cy = 1 + (int)((float)y / (float)cells * (float)(ih - 3));
						float r = 0, g = 0, b = 0;
						int n = 0;
						for (int dy = -2; dy <= 2; ++dy)
							for (int dx = -2; dx <= 2; ++dx)
							{
								int sx = cx + dx, sy = cy + dy;
								if (sx < 0 || sy < 0 || sx >= iw || sy >= ih)
									continue;
								const unsigned char *t = rgb + ((size_t)sy * (size_t)iw + (size_t)sx) * 3u;
								r += t[0];
								g += t[1];
								b += t[2];
								++n;
							}
						r /= 255.0f * n;
						g /= 255.0f * n;
						b /= 255.0f * n;
						float hi = fmaxf(r, fmaxf(g, b)), lo = fminf(r, fminf(g, b));
						world->grass[index] = fminf(1.0f, fmaxf(0.0f, (g - fmaxf(r, b)) * 12.0f));
						world->snow[index] =
							fminf(1.0f, fmaxf(0.0f, (hi - 0.62f) * 4.0f)) * (hi - lo < 0.12f ? 1.0f : 0.0f);
					}
				}
			stbi_image_free(rgb);
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
	printf("Overworld: level %d tiles x %d..%d y %d..%d, %ux%u samples, %.0f..%.0f m\n", level,
		   tx0, tx1, ty0, ty1, world->samples_u, world->samples_v, world->min_height,
		   world->max_height);
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
	float snow = grid_sample(world, world->snow, u, v);
	if (snow > 0.5f || elevation > 0.85f)
		return OVERWORLD_GROUND_SNOW;
	if (slope > 0.75f)
		return OVERWORLD_GROUND_CLIFF;
	if (grass > 0.35f)
		return OVERWORLD_GROUND_MEADOW;
	if (elevation < 0.25f)
		return OVERWORLD_GROUND_VALLEY;
	return OVERWORLD_GROUND_SCREE;
}

static bool place_entrances(Overworld *world, uint32_t seed)
{
	uint32_t state = seed * 2654435761u + 17u;
	double centre_u = (world->samples_u - 1) * 0.5, centre_v = (world->samples_v - 1) * 0.5;
	double cx, cz;
	from_grid(world, centre_u, centre_v, &cx, &cz);
	uint32_t placed = 0;
	/* Mitchell-style best-candidate blue noise. Terrain height is the only
	 * validity rule: every iteration samples the full map and retains the point
	 * farthest from its nearest existing entrance. This produces Voronoi-like,
	 * nearly equidistant sites without pushing mountains or glaciers aside. */
	while (placed < OVERWORLD_ENTRANCES)
	{
		OverworldEntrance best = {0};
		double best_score = -1.0;
		for (int attempt = 0; attempt < 30000; ++attempt)
		{
			/* A two-percent rim keeps interpolation inside the height grid while
			 * still using effectively the entire visible dataset. */
			double u = (0.02 + 0.96 * random01(&state)) * (world->samples_u - 1);
			double v = (0.02 + 0.96 * random01(&state)) * (world->samples_v - 1);
			double x, z;
			from_grid(world, u, v, &x, &z);
			float elevation = overworld_height(world, x, z);
			if (!isfinite(elevation))
				continue;
			float downhill = 0.0f;
			if (!isfinite(slope_at(world, x, z, &downhill)))
				downhill = 2.0f * PI_F * random01(&state);

			double nearest2 = 0.0;
			if (placed)
			{
				nearest2 = 1e30;
				for (uint32_t i = 0; i < placed; ++i)
				{
					double dx = x - world->entrances[i].position.x;
					double dz = z - world->entrances[i].position.z;
					nearest2 = fmin(nearest2, dx * dx + dz * dz);
				}
			}
			if (nearest2 > best_score)
			{
				best = (OverworldEntrance){
					.position = {x, elevation, z},
					.facing = downhill,
					.ground = classify(world, x, z),
					.elevation_m = elevation,
				};
				best_score = nearest2;
			}
		}
		if (best_score < 0.0)
		{
			fprintf(stderr, "Overworld: placed only %u of %u dungeon entrances\n", placed,
					OVERWORLD_ENTRANCES);
			return false;
		}
		world->entrances[placed++] = best;
	}
	/* Open on the whole map: centred, high, looking north-ish. */
	world->focus_x = cx;
	world->focus_z = cz;
	static const char *ground_names[] = {"snow", "cliff", "meadow", "valley", "scree"};
	double min_x = world->entrances[0].position.x, max_x = min_x;
	double min_z = world->entrances[0].position.z, max_z = min_z;
	double nearest = 1e30;
	for (uint32_t i = 0; i < placed; ++i)
	{
		printf("Overworld: entrance %u at (%.1f, %.1f, %.1f) on %s ground, %.0f m from centre\n", i,
			   world->entrances[i].position.x, world->entrances[i].position.y,
			   world->entrances[i].position.z, ground_names[world->entrances[i].ground],
			   hypot(world->entrances[i].position.x - cx, world->entrances[i].position.z - cz));
		min_x = fmin(min_x, world->entrances[i].position.x);
		max_x = fmax(max_x, world->entrances[i].position.x);
		min_z = fmin(min_z, world->entrances[i].position.z);
		max_z = fmax(max_z, world->entrances[i].position.z);
		for (uint32_t j = 0; j < i; ++j)
			nearest = fmin(nearest, hypot(world->entrances[i].position.x - world->entrances[j].position.x,
									  world->entrances[i].position.z - world->entrances[j].position.z));
	}
	printf("Overworld: entrance spread %.0f x %.0f m, nearest pair %.0f m\n", max_x - min_x,
		   max_z - min_z, nearest);
	return true;
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

static bool append_surface(Builder *b, const float p[4][3], const float n[3], int u_axis,
						   int v_axis, float uv_scale)
{
	if (!reserve(b, 4, 6))
		return false;
	uint32_t index[4];
	for (int i = 0; i < 4; ++i)
		index[i] = vert(b, p[i][0], p[i][1], p[i][2], n[0], n[1], n[2],
						p[i][u_axis] * uv_scale, p[i][v_axis] * uv_scale);
	quad(b, index[0], index[1], index[2], index[3]);
	return true;
}

/* A real, textured dungeon throat is built with the overworld because it must
 * already exist before the procedural dungeon is generated. It uses exactly
 * the dungeon wall/floor PBR sets and the normal mesh pipeline. */
static bool build_descent(Builder *walls, Builder *stairs)
{
	const int steps = 32, arch_segments = 14;
	const float run = 1.25f, rise = 0.43f, half = 2.25f, spring = 2.25f;
	for (int i = 0; i < steps; ++i)
	{
		float z0 = -(float)i * run, z1 = -(float)(i + 1) * run;
		float y0 = -(float)i * rise, y1 = -(float)(i + 1) * rise;
		const float tread[4][3] = {{-half, y0, z0}, {half, y0, z0},
									 {half, y0, z1}, {-half, y0, z1}};
		const float riser[4][3] = {{-half, y1, z1}, {half, y1, z1},
									 {half, y0, z1}, {-half, y0, z1}};
		const float left[4][3] = {{-half, y0, z0}, {-half, y1, z1},
								  {-half, y1 + spring, z1}, {-half, y0 + spring, z0}};
		const float right[4][3] = {{half, y0, z0}, {half, y0 + spring, z0},
								   {half, y1 + spring, z1}, {half, y1, z1}};
		if (!append_surface(stairs, tread, (const float[3]){0, 1, 0}, 0, 2, 0.55f) ||
			!append_surface(stairs, riser, (const float[3]){0, 0, 1}, 0, 1, 0.55f) ||
			!append_surface(walls, left, (const float[3]){1, 0, 0}, 2, 1, 0.55f) ||
			!append_surface(walls, right, (const float[3]){-1, 0, 0}, 2, 1, 0.55f))
			return false;
		for (int segment = 0; segment < arch_segments; ++segment)
		{
			float a0 = (float)segment / arch_segments * PI_F;
			float a1 = (float)(segment + 1) / arch_segments * PI_F;
			float x0 = cosf(a0) * half, x1 = cosf(a1) * half;
			float h0 = spring + sinf(a0) * half, h1 = spring + sinf(a1) * half;
			const float ceiling[4][3] = {{x0, y0 + h0, z0}, {x1, y0 + h1, z0},
									   {x1, y1 + h1, z1}, {x0, y1 + h0, z1}};
			float middle = (a0 + a1) * 0.5f;
			const float normal[3] = {-cosf(middle), -sinf(middle), 0};
			if (!append_surface(walls, ceiling, normal, 0, 2, 0.55f))
				return false;
		}
	}
	/* Close the far end so first-time generation freezes on masonry rather than
	 * a view through the back of the transition set. */
	float bottom = -(float)steps * rise;
	float end_z = -(float)steps * run;
	const float end[4][3] = {{-half, bottom, end_z}, {half, bottom, end_z},
								{half, bottom + spring * 2.0f, end_z},
								{-half, bottom + spring * 2.0f, end_z}};
	return append_surface(walls, end, (const float[3]){0, 0, 1}, 0, 1, 0.55f);
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
	Builder mound = {0}, ring = {0}, door = {0}, knob = {0}, step = {0}, portal = {0},
			walls = {0}, stairs = {0};
	/* Door: a round plank disc, its centre 1.05 m up so it meets the ground. */
	const float door_radius = 1.05f, door_cy = 1.0f;
	bool ok = build_mound(&mound) &&
			  build_annulus(&ring, door_radius - 0.02f, door_radius + 0.42f, -0.45f, 0.32f,
							door_cy, 0.55f) &&
			  build_annulus(&door, 0.0f, door_radius, -0.08f, 0.0f, door_cy, 0.6f) &&
			  build_box(&knob, -0.07f, door_cy - 0.07f, 0.0f, 0.07f, door_cy + 0.07f, 0.16f, 2.0f) &&
			  build_box(&step, -1.3f, -0.6f, 0.3f, 1.3f, 0.06f, 1.5f, 0.6f) &&
			  build_annulus(&portal, 0.0f, door_radius - 0.01f, -0.18f, -0.16f, door_cy,
						0.6f) &&
			  build_descent(&walls, &stairs);
	if (!ok)
	{
		free(mound.v), free(mound.i), free(ring.v), free(ring.i), free(door.v), free(door.i);
		free(knob.v), free(knob.i), free(step.v), free(step.i);
		free(portal.v), free(portal.i);
		free(walls.v), free(walls.i), free(stairs.v), free(stairs.i);
		return false;
	}
	/* Turf over the top (the dungeons' moss), dressed stone round the door. */
	upload(renderer, world, OVERWORLD_MESH_MOUND, &mound, DUNGEON_TEXTURE_DIR "/moss.jpeg", NULL,
		   NULL);
	upload(renderer, world, OVERWORLD_MESH_RING, &ring, DUNGEON_TEXTURE_DIR "/wall_albedo.jpg",
		   DUNGEON_TEXTURE_DIR "/wall_orm.png", DUNGEON_TEXTURE_DIR "/wall_normal.png");
	upload(renderer, world, OVERWORLD_MESH_DOOR, &door, DUNGEON_TEXTURE_DIR "/exit_albedo.jpg",
		   DUNGEON_TEXTURE_DIR "/exit_orm.png", DUNGEON_TEXTURE_DIR "/exit_normal.png");
	upload(renderer, world, OVERWORLD_MESH_KNOB, &knob, DUNGEON_TEXTURE_DIR "/lock_albedo.jpg",
		   DUNGEON_TEXTURE_DIR "/lock_orm.png", DUNGEON_TEXTURE_DIR "/lock_normal.png");
	upload(renderer, world, OVERWORLD_MESH_STEP, &step, DUNGEON_TEXTURE_DIR "/floor_albedo.jpg",
		   DUNGEON_TEXTURE_DIR "/floor_orm.png", DUNGEON_TEXTURE_DIR "/floor_normal.png");
	/* A near-black disc immediately behind the leaf makes the doorway a true
	 * portal while outdoor terrain is still being rendered. Without it, the
	 * hillside occupying the tunnel volume is visible through the open door. */
	upload(renderer, world, OVERWORLD_MESH_PORTAL, &portal, NULL, NULL, NULL);
	upload(renderer, world, OVERWORLD_MESH_TUNNEL_WALL, &walls,
		   DUNGEON_TEXTURE_DIR "/wall_albedo.jpg", DUNGEON_TEXTURE_DIR "/wall_orm.png",
		   DUNGEON_TEXTURE_DIR "/wall_normal.png");
	upload(renderer, world, OVERWORLD_MESH_TUNNEL_STAIRS, &stairs,
		   DUNGEON_TEXTURE_DIR "/floor_albedo.jpg", DUNGEON_TEXTURE_DIR "/floor_orm.png",
		   DUNGEON_TEXTURE_DIR "/floor_normal.png");
	/* The dungeon surface shader's fourth sampler is its moss detail. Reuse the
	 * already-cached moss texture from the mound so the passage follows the
	 * exact same descriptor and shader contract as generated walls and floors. */
	VkDescriptorImageInfo moss = {
		.sampler = world->meshes[OVERWORLD_MESH_MOUND].texture.sampler,
		.imageView = world->meshes[OVERWORLD_MESH_MOUND].texture.view,
		.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	};
	VkWriteDescriptorSet writes[2] = {0};
	for (uint32_t i = 0; i < 2u; ++i)
		writes[i] = (VkWriteDescriptorSet){
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = world->meshes[OVERWORLD_MESH_TUNNEL_WALL + i].material_set,
			.dstBinding = 3,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo = &moss,
		};
	vkUpdateDescriptorSets(renderer->device, 2, writes, 0, NULL);
	return true;
}

/* --- Lifecycle ------------------------------------------------------------------ */

bool overworld_create(Renderer *renderer, Overworld *out, const char *dataset_root, uint32_t seed,
					  double centre_x, double centre_z, double half_extent_m)
{
	*out = (Overworld){0};
	if (!load_heights(out, dataset_root, centre_x, centre_z, half_extent_m))
	{
		fprintf(stderr, "Overworld: no terrain tiles under %s\n", dataset_root);
		overworld_destroy(NULL, out);
		return false;
	}
	if (!place_entrances(out, seed))
	{
		overworld_destroy(NULL, out);
		return false;
	}
	out->pitch = -58.0f;
	out->yaw = -90.0f;
	/* Begin with the expanded campaign in view; players can still zoom or fly
	 * down to an individual entrance. */
	out->distance = half_extent_m > 0.0 ? fminf(MAX_DISTANCE, (float)half_extent_m * 2.5f)
									 : MAX_DISTANCE;
	if (!build_entrance_meshes(renderer, out))
	{
		overworld_destroy(renderer, out);
		return false;
	}
	GltfLoadError torch_error = {0};
	if (gltf_scene_create(renderer, DUNGEON_TORCH_PATH,
						  &(GltfLoadOptions){.placement =
											 coordinate_identity_transform((WorldPosition){0})},
						  &out->descent_torch, &torch_error) != GLTF_LOAD_OK)
	{
		fprintf(stderr, "Overworld: could not load descent torch: %s\n", torch_error.message);
		overworld_destroy(renderer, out);
		return false;
	}
	out->descent_torch_loaded = true;
	out->descent_flame = (Mesh){
		.local_to_world = coordinate_identity_transform((WorldPosition){0}),
		.vertices = DESCENT_FLAME_VERTICES,
		.vertex_count = 4,
		.indices = DESCENT_FLAME_INDICES,
		.index_count = 6,
	};
	mesh_upload(renderer, &out->descent_flame);
	out->descent_flame_uploaded = true;
	return true;
}

void overworld_destroy(Renderer *renderer, Overworld *world)
{
	if (world->descent_flame_uploaded && renderer)
		mesh_destroy(renderer, &world->descent_flame);
	if (world->descent_torch_loaded)
		gltf_scene_destroy(renderer, &world->descent_torch);
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
	free(world->snow);
	*world = (Overworld){0};
}

/* --- Map camera ----------------------------------------------------------------- */

static void clamp_focus(Overworld *world)
{
	double u, v;
	to_grid(world, world->focus_x, world->focus_z, &u, &v);
	double max_u = world->samples_u - 1.0, max_v = world->samples_v - 1.0;
	u = u < 0.0 ? 0.0 : (u > max_u ? max_u : u);
	v = v < 0.0 ? 0.0 : (v > max_v ? max_v : v);
	from_grid(world, u, v, &world->focus_x, &world->focus_z);
}

void overworld_update(Overworld *world, float pan_forward, float pan_right, float rotate_yaw,
					  float rotate_pitch, float zoom, float dt, bool controls)
{
	world->time += dt;
	if (world->descending)
		return;
	if (!controls)
	{
		world->yaw += dt * 3.0f; /* a slow turn behind the menus */
		world->flying = false;
		return;
	}
	float yaw = world->yaw * PI_F / 180.0f;
	float fx = cosf(yaw), fz = sinf(yaw); /* ground-plane forward */
	float rx = -fz, rz = fx;			  /* ground-plane right */
	bool user_moved = zoom != 0.0f || rotate_yaw != 0.0f || rotate_pitch != 0.0f ||
					  pan_forward != 0.0f || pan_right != 0.0f;
	if (user_moved)
		world->flying = false;
	float pan_speed = world->distance * 0.9f;
	world->focus_x += (double)(fx * pan_forward + rx * pan_right) * pan_speed * dt;
	world->focus_z += (double)(fz * pan_forward + rz * pan_right) * pan_speed * dt;
	world->yaw += rotate_yaw * 75.0f * dt;
	world->pitch = fminf(-25.0f, fmaxf(-82.0f, world->pitch + rotate_pitch * 55.0f * dt));
	/* Keyboard zoom is a continuous axis, so exponentiate by frame time to keep
	 * it consistent at every frame rate and every map height. */
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
	clamp_focus(world);
}

void overworld_fly_to(Overworld *world, uint32_t i)
{
	const OverworldEntrance *e = &world->entrances[i % OVERWORLD_ENTRANCES];
	world->descending = false;
	world->flying = true;
	/* Aim a little in front of the door so the mound sits mid-frame. */
	world->fly_x = e->position.x + cos(e->facing) * 2.0;
	world->fly_z = e->position.z + sin(e->facing) * 2.0;
	world->fly_distance = 30.0f;
	/* Turn to face the door head-on: the camera looks back along its facing. */
	float want = e->facing * 180.0f / PI_F + 180.0f;
	world->yaw += remainderf(want - world->yaw, 360.0f);
}

void overworld_resume_map(Overworld *world)
{
	world->descending = false;
	world->flying = false;
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

static float ease(float t)
{
	t = fminf(1.0f, fmaxf(0.0f, t));
	return t * t * (3.0f - 2.0f * t);
}

static Camera camera_mix(Camera a, Camera b, float t)
{
	t = ease(t);
	Camera result = a;
	result.position.x += (b.position.x - a.position.x) * t;
	result.position.y += (b.position.y - a.position.y) * t;
	result.position.z += (b.position.z - a.position.z) * t;
	result.yaw += remainderf(b.yaw - a.yaw, 360.0f) * t;
	result.pitch += (b.pitch - a.pitch) * t;
	return result;
}

void overworld_begin_descent(Overworld *world, uint32_t i)
{
	world->descent_from = overworld_camera(world);
	world->descent_entrance = i % OVERWORLD_ENTRANCES;
	world->descent_progress = 0.0f;
	world->descending = true;
	world->flying = false;
}

Camera overworld_descent_camera(Overworld *world, float progress)
{
	if (!world->descending)
		overworld_begin_descent(world, 0);
	progress = fminf(1.0f, fmaxf(0.0f, progress));
	world->descent_progress = progress;
	const OverworldEntrance *e = &world->entrances[world->descent_entrance];
	float fx = cosf(e->facing), fz = sinf(e->facing);
	float inward_yaw = e->facing * 180.0f / PI_F + 180.0f;
	Camera threshold = {
		.position = {e->position.x + fx * 5.5, e->position.y + 1.70,
					 e->position.z + fz * 5.5},
		.yaw = inward_yaw,
		.pitch = -4.0f,
	};
	if (progress < 0.42f)
		return camera_mix(world->descent_from, threshold, progress / 0.42f);
	float travel = 0.0f;
	float local_z;
	if (progress < 0.48f)
	{
		/* Move up to the black portal, but never enter the hillside while the
		 * terrain is present. */
		float approach = ease((progress - 0.42f) / 0.06f);
		local_z = 5.5f + (1.1f - 5.5f) * approach;
	}
	else
	{
		/* Cut from just outside the opaque portal to safely inside the stone
		 * throat. Both sides are black at the cut, so no terrain intersection is
		 * ever shown. Stop with another lit flight still ahead for the load hold. */
		travel = ease((progress - 0.48f) / 0.52f);
		local_z = -1.6f - travel * 22.4f;
	}
	float floor = local_z < 0.0f ? local_z * (0.43f / 1.25f) : 0.0f;
	Camera inside = {
		.position = {e->position.x + fx * local_z,
					 e->position.y - 0.15 + floor + 1.62 + sinf(travel * 16.0f) * 0.025f,
					 e->position.z + fz * local_z},
		.yaw = inward_yaw,
		.pitch = progress < 0.48f ? -8.0f : -14.0f + sinf(travel * 8.0f) * 0.7f,
	};
	return inside;
}

bool overworld_descent_is_underground(const Overworld *world)
{
	return world->descending && world->descent_progress >= 0.48f;
}

uint32_t overworld_descent_lights(const Overworld *world, WorldPosition camera,
								 vec4s *positions, vec4s *colors, uint32_t capacity)
{
	if (!world->descending || !positions || !colors)
		return 0;
	const OverworldEntrance *e = &world->entrances[world->descent_entrance];
	LocalToWorldTransform placement =
		coordinate_rotation_y(PI_F * 0.5 - e->facing,
						  (WorldPosition){e->position.x, e->position.y - 0.15, e->position.z});
	const uint32_t count = capacity < 6u ? capacity : 6u;
	for (uint32_t i = 0; i < count; ++i)
	{
		float z = -3.5f - (float)i * 6.4f;
		float floor = z * (0.43f / 1.25f);
		float side = (i & 1u) ? 2.20f : -2.20f;
		WorldPosition light = coordinate_local_to_world(
			&placement, (TileLocalPosition){side, floor + 1.67f, z});
		CameraRelativePosition relative = coordinate_camera_relative(light, camera);
		float flicker = 0.88f + 0.12f * sinf(world->time * 11.0f + (float)i * 2.31f);
		positions[i] = (vec4s){{relative.x, relative.y, relative.z, 7.0f}};
		colors[i] = (vec4s){{1.0f, 0.48f, 0.18f, 14.0f * flicker}};
	}
	return count;
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

static mat4s transform_matrix(const LocalToWorldTransform *transform)
{
	mat4s matrix = GLMS_MAT4_IDENTITY_INIT;
	for (int column = 0; column < 3; ++column)
		for (int row = 0; row < 3; ++row)
			matrix.raw[column][row] = (float)transform->rotation[column][row];
	matrix.raw[3][0] = (float)transform->translation.x;
	matrix.raw[3][1] = (float)transform->translation.y;
	matrix.raw[3][2] = (float)transform->translation.z;
	return matrix;
}

static LocalToWorldTransform descent_torch_transform(const LocalToWorldTransform *placement,
											 uint32_t index)
{
	float z = -3.5f - (float)index * 6.4f;
	float floor = z * (0.43f / 1.25f);
	bool right = (index & 1u) != 0;
	float inward = right ? -1.0f : 1.0f;
	LocalToWorldTransform local = coordinate_rotation_y(
		atan2(-(double)inward, 0.0),
		(WorldPosition){right ? 2.24 : -2.24, floor + 0.05f, z});
	return coordinate_compose(placement, transform_matrix(&local));
}

static DrawPushConstants gltf_push(const LocalToWorldTransform *transform,
								   const GltfMaterial *material, WorldPosition camera)
{
	return (DrawPushConstants){
		.local_to_camera_relative = coordinate_local_to_camera_relative(transform, camera),
		.geometry = {{material->base_color_factor[0], material->base_color_factor[1],
					  material->base_color_factor[2], material->base_color_factor[3]}},
		.elevation_uv = {{material->roughness_factor, material->normal_scale,
						  material->occlusion_strength, 0.0f}},
		.material = {{material->metallic_factor, 1.0f, 1.0f, 1.0f}},
		.debug = {{0.0f, 0.0f, 1.0f, 1.0f}},
	};
}

static LocalToWorldTransform descent_flame_transform(WorldPosition anchor, WorldPosition camera)
{
	double dx = camera.x - anchor.x, dz = camera.z - anchor.z;
	double length = hypot(dx, dz);
	if (length < 1e-6)
	{
		dx = 0.0;
		dz = 1.0;
		length = 1.0;
	}
	dx /= length;
	dz /= length;
	const double width = 0.22, height = 0.44, base_drop = 0.114;
	LocalToWorldTransform transform = {0};
	transform.rotation[0][0] = dz * width;
	transform.rotation[0][2] = -dx * width;
	transform.rotation[1][1] = height;
	transform.rotation[2][0] = dx;
	transform.rotation[2][2] = dz;
	transform.translation =
		(WorldPosition){anchor.x, anchor.y + height * 0.5 - base_drop, anchor.z};
	return transform;
}

uint32_t overworld_draws(Overworld *world, WorldPosition camera, RendererDraw *out,
						 uint32_t capacity, uint32_t *out_shadow_count)
{
	uint32_t count = 0;
	for (uint32_t e = 0; e < OVERWORLD_ENTRANCES; ++e)
	{
		const OverworldEntrance *entrance = &world->entrances[e];
		/* Local +Z (the door's outward normal) onto the entrance facing:
		 * rotation_y(a) sends +Z to (sin a, cos a), so a = pi/2 - facing. */
		LocalToWorldTransform placement =
			coordinate_rotation_y(PI_F * 0.5 - entrance->facing,
								  (WorldPosition){entrance->position.x, entrance->position.y - 0.15,
												  entrance->position.z});
		LocalToWorldTransform transform = placement;
		const double scale = 1.4; /* big enough to read from the map */
		for (int c = 0; c < 3; ++c)
			for (int r = 0; r < 3; ++r)
				transform.rotation[c][r] *= scale;
		/* Albedo tints: a painted green door, a polished brass knob, and turf
		 * a shade greener than the cave moss it borrows. */
		static const float tints[OVERWORLD_MESH_COUNT][3] = {
			[OVERWORLD_MESH_MOUND] = {0.85f, 1.05f, 0.7f},
			[OVERWORLD_MESH_RING] = {1.1f, 1.05f, 1.0f},
			[OVERWORLD_MESH_DOOR] = {0.55f, 1.25f, 0.6f},
			[OVERWORLD_MESH_KNOB] = {1.8f, 1.4f, 0.6f},
			[OVERWORLD_MESH_STEP] = {1.1f, 1.05f, 1.0f},
			[OVERWORLD_MESH_PORTAL] = {0.008f, 0.006f, 0.004f},
			[OVERWORLD_MESH_TUNNEL_WALL] = {1.0f, 1.0f, 1.0f},
			[OVERWORLD_MESH_TUNNEL_STAIRS] = {1.0f, 1.0f, 1.0f},
		};
		for (int m = 0; m < OVERWORLD_MESH_COUNT && count < capacity; ++m)
		{
			if (!world->uploaded[m])
				continue;
			bool tunnel = m == OVERWORLD_MESH_TUNNEL_WALL || m == OVERWORLD_MESH_TUNNEL_STAIRS;
			bool portal = m == OVERWORLD_MESH_PORTAL;
			if (tunnel && (!world->descending || e != world->descent_entrance))
				continue;
			if (portal && (!world->descending || e != world->descent_entrance ||
						   overworld_descent_is_underground(world)))
				continue;
			/* Once through the threshold, neither terrain nor the decorative
			 * exterior mound may occupy the camera. Only the real stairwell stays. */
			if (!tunnel && !portal && overworld_descent_is_underground(world))
				continue;
			/* The outdoor mound is enlarged for map readability; the stairwell is
			 * authored in real metres and meets its threshold without that scale. */
			LocalToWorldTransform piece_transform = tunnel ? placement : transform;
			if (world->descending && e == world->descent_entrance &&
				(m == OVERWORLD_MESH_DOOR || m == OVERWORLD_MESH_KNOB))
			{
				/* Swing the round door into the mound before the camera crosses
				 * the threshold. Door and knob use the same hinge transform. */
				float open = ease((world->descent_progress - 0.25f) / 0.23f);
				float angle = -open * 1.42f;
				float cs = cosf(angle), sn = sinf(angle), pivot = -1.05f;
				mat4s hinge = GLMS_MAT4_IDENTITY_INIT;
				hinge.raw[0][0] = cs;
				hinge.raw[0][2] = -sn;
				hinge.raw[2][0] = sn;
				hinge.raw[2][2] = cs;
				hinge.raw[3][0] = pivot * (1.0f - cs);
				hinge.raw[3][2] = pivot * sn;
				piece_transform = coordinate_compose(&transform, hinge);
			}
			DrawPushConstants push = entrance_push(&piece_transform, camera);
			push.geometry = (vec4s){{tints[m][0], tints[m][1], tints[m][2], 1.0f}};
			if (tunnel)
				push.geometry.w = -13.76f; /* floor at the bottom step, for moss height */
			if (m == OVERWORLD_MESH_MOUND || m == OVERWORLD_MESH_DOOR || portal)
				push.material.x = 0.0f; /* turf and paint are never metal */
			out[count++] = (RendererDraw){.mesh = &world->meshes[m],
										  .material_set = world->meshes[m].material_set,
										  .push = push,
										  .static_mesh = true,
										  .pipeline = tunnel ? RENDERER_PIPELINE_DUNGEON_SURFACE
														 : RENDERER_PIPELINE_AUTO};
		}
	}
	if (world->descending && world->descent_torch_loaded)
	{
		const OverworldEntrance *entrance = &world->entrances[world->descent_entrance];
		LocalToWorldTransform placement = coordinate_rotation_y(
			PI_F * 0.5 - entrance->facing,
			(WorldPosition){entrance->position.x, entrance->position.y - 0.15,
							entrance->position.z});
		for (uint32_t torch = 0; torch < 6u; ++torch)
		{
			LocalToWorldTransform transform = descent_torch_transform(&placement, torch);
			for (uint32_t primitive_index = 0;
				 primitive_index < world->descent_torch.primitive_count && count < capacity;
				 ++primitive_index)
			{
				GltfPrimitive *primitive = &world->descent_torch.primitives[primitive_index];
				GltfMaterial *material =
					&world->descent_torch.materials[primitive->material_index];
				out[count++] = (RendererDraw){
					.mesh = &primitive->mesh,
					.material_set = material->descriptor_set,
					.push = gltf_push(&transform, material, camera),
					.static_mesh = true,
				};
			}
		}
	}
	if (out_shadow_count)
		*out_shadow_count = count;
	if (world->descending && world->descent_flame_uploaded)
	{
		const OverworldEntrance *entrance = &world->entrances[world->descent_entrance];
		LocalToWorldTransform placement = coordinate_rotation_y(
			PI_F * 0.5 - entrance->facing,
			(WorldPosition){entrance->position.x, entrance->position.y - 0.15,
							entrance->position.z});
		for (uint32_t torch = 0; torch < 6u && count < capacity; ++torch)
		{
			LocalToWorldTransform fixture = descent_torch_transform(&placement, torch);
			WorldPosition anchor = coordinate_local_to_world(
				&fixture, (TileLocalPosition){0.0f, 1.5223f, -0.03492f});
			float phase = (float)torch * 2.31f;
			float flicker = 0.88f + 0.12f * sinf(world->time * 11.0f + phase);
			anchor.y += sinf(world->time * 8.3f + phase) * 0.012f;
			LocalToWorldTransform flame = descent_flame_transform(anchor, camera);
			out[count++] = (RendererDraw){
				.mesh = &world->descent_flame,
				.material_set = world->descent_flame.material_set,
				.push = {
					.local_to_camera_relative = coordinate_local_to_camera_relative(&flame, camera),
					.geometry = {{1.0f, 1.0f, 1.0f, flicker}},
					.elevation_uv = {{world->time, phase, 0.22f, 0.44f}},
					.material = {{0.025f * sinf(world->time * 9.7f + phase), 0.0f, 90.0f,
								  0.0f}},
					.debug = {{0.0f, 1.0f, 0.0f, 0.0f}},
				},
				.static_mesh = true,
				.pipeline = RENDERER_PIPELINE_DUNGEON_FLAME,
			};
		}
	}
	return count;
}
