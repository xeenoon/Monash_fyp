#include "dungeon_prism.h"
#include <float.h>
#include <math.h>
#include <stddef.h>

#define PI 3.14159265358979323846f
#define EPS .00002f
#define BOARD .355f
#define RAY_SAMPLES 17u
#define MAX_PENDING 256u
#define MAX_BOUNCES 18u
static DungeonPrismPoint add(DungeonPrismPoint a, DungeonPrismPoint b)
{
	return (DungeonPrismPoint){a.x + b.x, a.y + b.y};
}
static DungeonPrismPoint sub(DungeonPrismPoint a, DungeonPrismPoint b)
{
	return (DungeonPrismPoint){a.x - b.x, a.y - b.y};
}
static DungeonPrismPoint mul(DungeonPrismPoint p, float s)
{
	return (DungeonPrismPoint){p.x * s, p.y * s};
}
static float dot(DungeonPrismPoint a, DungeonPrismPoint b) { return a.x * b.x + a.y * b.y; }
static float cross(DungeonPrismPoint a, DungeonPrismPoint b) { return a.x * b.y - a.y * b.x; }
static float length(DungeonPrismPoint a) { return sqrtf(dot(a, a)); }
static DungeonPrismPoint unit(DungeonPrismPoint a)
{
	return mul(a, 1.f / fmaxf(length(a), 1e-10f));
}
static DungeonPrismPoint direction(float angle)
{
	return (DungeonPrismPoint){cosf(angle), sinf(angle)};
}
static DungeonPrismPoint rotate(DungeonPrismPoint p, float a)
{
	return (DungeonPrismPoint){p.x * cosf(a) - p.y * sinf(a), p.x * sinf(a) + p.y * cosf(a)};
}
static float optic_angle(const DungeonPrism *p) { return -(float)p->orientation * PI / 180.f; }

float dungeon_lens_half_width(DungeonOpticKind kind, float y)
{
	float radius = DUNGEON_OPTIC_CURVATURE, aperture = DUNGEON_OPTIC_APERTURE;
	float root = sqrtf(fmaxf(radius * radius - y * y, 0));
	return DUNGEON_OPTIC_EDGE + (kind == DUNGEON_OPTIC_CONVEX
									 ? root - sqrtf(radius * radius - aperture * aperture)
									 : radius - root);
}
bool dungeon_optics_refract(DungeonPrismPoint incident, DungeonPrismPoint normal, float n1,
							float n2, DungeonPrismPoint *transmitted, float *reflectance)
{
	float cos_i = fmaxf(0, fminf(1, -dot(incident, normal))), eta = n1 / n2;
	float k = 1 - eta * eta * (1 - cos_i * cos_i);
	if (k <= 0)
	{
		*reflectance = 1;
		*transmitted = (DungeonPrismPoint){0};
		return false;
	}
	float cos_t = sqrtf(k);
	*transmitted = unit(add(mul(incident, eta), mul(normal, eta * cos_i - cos_t)));
	float rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t);
	float rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t);
	*reflectance = .5f * (rs * rs + rp * rp);
	return true;
}

/* Exact circular surfaces for the lenses, straight sides for the prism.
 * Normals point out of glass; no grid intersection or angle snapping. */
static bool optic_hit(const DungeonPrism *optic, DungeonPrismPoint origin, DungeonPrismPoint ray,
					  float *distance, DungeonPrismPoint *normal)
{
	float angle = optic_angle(optic);
	DungeonPrismPoint o = rotate(sub(origin, optic->position), -angle), d = rotate(ray, -angle),
					  n = {0};
	float best = *distance;
	if (optic->kind == DUNGEON_OPTIC_PRISM)
	{
		const DungeonPrismPoint corners[3] = {{-.038f, .038f}, {.038f, .038f}, {.038f, -.038f}};
		for (unsigned i = 0; i < 3; ++i)
		{
			DungeonPrismPoint a = corners[i], edge = sub(corners[(i + 1) % 3], a),
							  delta = sub(a, o);
			float den = cross(d, edge);
			if (fabsf(den) < 1e-7f)
				continue;
			float t = cross(delta, edge) / den, u = cross(delta, d) / den;
			if (t > EPS && t < best && u >= 0 && u <= 1)
			{
				best = t;
				n = unit((DungeonPrismPoint){-edge.y, edge.x});
			}
		}
	}
	else
	{
		float radius = DUNGEON_OPTIC_CURVATURE, aperture = DUNGEON_OPTIC_APERTURE;
		float center = optic->kind == DUNGEON_OPTIC_CONVEX
						   ? sqrtf(radius * radius - aperture * aperture) - DUNGEON_OPTIC_EDGE
						   : radius + DUNGEON_OPTIC_EDGE;
		for (int side = -1; side <= 1; side += 2)
		{
			float cx = side * (optic->kind == DUNGEON_OPTIC_CONVEX ? -center : center);
			DungeonPrismPoint rel = sub(o, (DungeonPrismPoint){cx, 0});
			float b = dot(rel, d), disc = b * b - dot(rel, rel) + radius * radius;
			if (disc < 0)
				continue;
			for (int root = -1; root <= 1; root += 2)
			{
				float t = -b + root * sqrtf(disc);
				DungeonPrismPoint hit = add(o, mul(d, t));
				if (t <= EPS || t >= best || fabsf(hit.y) > aperture || hit.x * side < 0 ||
					fabsf(fabsf(hit.x) - dungeon_lens_half_width(optic->kind, hit.y)) > .0001f)
					continue;
				best = t;
				n = mul(sub(hit, (DungeonPrismPoint){cx, 0}), 1.f / radius);
				if (optic->kind == DUNGEON_OPTIC_CONCAVE)
					n = mul(n, -1);
			}
			if (fabsf(d.y) > 1e-7f)
			{
				float t = (side * aperture - o.y) / d.y;
				float x = o.x + d.x * t;
				if (t > EPS && t < best &&
					fabsf(x) <= dungeon_lens_half_width(optic->kind, aperture))
				{
					best = t;
					n = (DungeonPrismPoint){0, (float)side};
				}
			}
		}
	}
	if (best >= *distance)
		return false;
	*distance = best;
	*normal = rotate(n, angle);
	return true;
}
static float board_exit(DungeonPrismPoint p, DungeonPrismPoint d)
{
	float best = 2;
	if (fabsf(d.x) > 1e-7f)
		best = fminf(best, ((d.x > 0 ? BOARD : -BOARD) - p.x) / d.x);
	if (fabsf(d.y) > 1e-7f)
		best = fminf(best, ((d.y > 0 ? BOARD : -BOARD) - p.y) / d.y);
	return fmaxf(best, 0);
}
typedef struct
{
	DungeonPrismPoint p, d;
	float power;
	int medium;
	unsigned bounce, channel;
} Ray;
static void enqueue(Ray *queue, unsigned *count, Ray ray, DungeonPrismTrace *trace)
{
	if (ray.power < .0008f)
		return;
	if (ray.bounce >= MAX_BOUNCES || *count >= MAX_PENDING)
	{
		trace->limited = true;
		return;
	}
	queue[(*count)++] = ray;
}
/* A physical receiver plane with a 12mm entrance. It absorbs intersecting
 * rays once; scattered light elsewhere on the medallion contributes nothing. */
static float receiver_distance(DungeonPrismPoint source, DungeonPrismPoint ray,
							   DungeonPrismPoint key, float angle)
{
	DungeonPrismPoint normal = direction(angle);
	float facing = dot(ray, normal);
	if (facing <= .98f)
		return FLT_MAX;
	float t = dot(sub(key, source), normal) / facing;
	if (t < 0)
		return FLT_MAX;
	DungeonPrismPoint hit = add(source, mul(ray, t));
	return length(sub(hit, key)) <= DUNGEON_OPTIC_KEY_RADIUS ? t : FLT_MAX;
}
DungeonPrismTrace dungeon_prism_trace(const DungeonPrismPuzzle *p)
{
	DungeonPrismTrace t = {0};
	if (!p || p->count > DUNGEON_PRISM_MAX)
		return t;
	Ray queue[MAX_PENDING];
	unsigned count = 0;
	/* Ideal collimated emitter: spatial samples share one direction.
	 * Only optical surfaces change their angles; no artificial fan/reset. */
	DungeonPrismPoint perpendicular = direction(p->source_angle + PI / 2);
	for (unsigned c = 0; c < 3; ++c)
		for (unsigned sample = 0; sample < RAY_SAMPLES; ++sample)
		{
			float x = 2.f * sample / (RAY_SAMPLES - 1) - 1;
			queue[count++] = (Ray){add(p->source, mul(perpendicular, x * .006f)),
								   direction(p->source_angle),
								   1.f / RAY_SAMPLES,
								   -1,
								   0,
								   c};
		}
	while (count && t.count < DUNGEON_PRISM_MAX_SEGMENTS)
	{
		unsigned brightest = 0;
		for (unsigned i = 1; i < count; ++i)
			if (queue[i].power > queue[brightest].power)
				brightest = i;
		Ray ray = queue[brightest];
		queue[brightest] = queue[--count];
		float distance = board_exit(ray.p, ray.d);
		DungeonPrismPoint normal = {0};
		int target = -1;
		for (unsigned i = 0; i < p->count; ++i)
		{
			if (ray.medium >= 0 && ray.medium != (int)i)
				continue;
			if (optic_hit(&p->prisms[i], ray.p, ray.d, &distance, &normal))
				target = (int)i;
		}
		float receiver =
			ray.medium < 0 ? receiver_distance(ray.p, ray.d, p->key, p->key_angle) : FLT_MAX;
		bool collected = receiver <= distance;
		if (collected)
			distance = receiver;
		DungeonPrismPoint end = add(ray.p, mul(ray.d, distance));
		t.segments[t.count++] = (DungeonPrismSegment){
			ray.p, end, ray.power, .0048f, .0048f, ray.medium >= 0, (uint8_t)ray.channel};
		if (collected)
		{
			t.key_power += ray.power / 3.f;
			continue;
		}
		if (target < 0)
			continue;
		t.lit_mask |= 1u << target;
		bool entering = ray.medium < 0;
		DungeonPrismPoint opposing = entering ? normal : mul(normal, -1);
		const float ior[3] = {1.507f, 1.513f, 1.522f};
		float reflectance;
		DungeonPrismPoint refracted;
		bool transmits =
			dungeon_optics_refract(ray.d, opposing, entering ? 1 : ior[ray.channel],
								   entering ? ior[ray.channel] : 1, &refracted, &reflectance);
		float power = ray.power * expf(-(entering ? 0 : .3f) * distance);
		DungeonPrismPoint reflected = sub(ray.d, mul(opposing, 2 * dot(ray.d, opposing)));
		if (transmits)
			enqueue(queue, &count,
					(Ray){add(end, mul(refracted, EPS * 2)), refracted, power * (1 - reflectance),
						  entering ? target : -1, ray.bounce + 1, ray.channel},
					&t);
		enqueue(queue, &count,
				(Ray){add(end, mul(reflected, EPS * 2)), reflected, power * reflectance, ray.medium,
					  ray.bounce + 1, ray.channel},
				&t);
		if (transmits && power * reflectance >= .0008f)
			t.split_mask |= 1u << target;
	}
	if (count)
		t.limited = true;
	t.hit_key = t.key_power >= DUNGEON_OPTIC_KEY_POWER;
	return t;
}

static uint32_t random_index(uint64_t *state, uint32_t n)
{
	uint64_t z = (*state += UINT64_C(0x9E3779B97F4A7C15));
	z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
	z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
	return (uint32_t)((z ^ (z >> 31)) % n);
}
static float random_range(uint64_t *rng, float low, float high)
{
	return low + (high - low) * random_index(rng, 10001) / 10000.f;
}
/* Nominal green ray through a single element: used to BUILD the witness.
 * The final layout is always checked with the full wavelength/ray bundle. */
static bool principal_exit(const DungeonPrism *optic, DungeonPrismPoint origin,
						   DungeonPrismPoint ray, DungeonPrismPoint *out,
						   DungeonPrismPoint *out_ray)
{
	bool inside = false;
	for (unsigned step = 0; step < 10; ++step)
	{
		float distance = 2;
		DungeonPrismPoint normal;
		if (!optic_hit(optic, origin, ray, &distance, &normal))
			return false;
		DungeonPrismPoint hit = add(origin, mul(ray, distance));
		DungeonPrismPoint opposing = inside ? mul(normal, -1) : normal, next;
		float reflection;
		if (dungeon_optics_refract(ray, opposing, inside ? 1.513f : 1, inside ? 1 : 1.513f, &next,
								   &reflection))
		{
			if (inside)
			{
				*out = add(hit, mul(next, EPS * 2));
				*out_ray = next;
				return true;
			}
			inside = true;
		}
		else
			next = sub(ray, mul(opposing, 2 * dot(ray, opposing)));
		origin = add(hit, mul(next, EPS * 2));
		ray = next;
	}
	return false;
}
static bool generate(DungeonPrismPuzzle *p, uint64_t *rng)
{
	*p = (DungeonPrismPuzzle){.count = DUNGEON_PRISM_MAX};
	p->source = (DungeonPrismPoint){-.345f, random_range(rng, -.23f, .23f)};
	p->source_angle = random_range(rng, -.45f, .45f);
	DungeonPrismPoint origin = p->source, ray = direction(p->source_angle);
	const DungeonOpticKind kinds[5] = {DUNGEON_OPTIC_PRISM, DUNGEON_OPTIC_CONCAVE,
									   DUNGEON_OPTIC_PRISM, DUNGEON_OPTIC_PRISM,
									   DUNGEON_OPTIC_CONVEX};
	for (unsigned i = 0; i < p->count; ++i)
	{
		bool placed = false;
		for (unsigned trial = 0; trial < 24 && !placed; ++trial)
		{
			unsigned angle = kinds[i] == DUNGEON_OPTIC_PRISM
								 ? random_index(rng, 360)
								 : (unsigned)((int)lroundf(-atan2f(ray.y, ray.x) * 180 / PI) + 720 +
											  (int)random_index(rng, 17) - 8) %
									   360;
			DungeonPrism candidate = {.orientation = (uint16_t)angle, .kind = kinds[i]};
			DungeonPrismPoint aim = add(origin, mul(ray, random_range(rng, .115f, .185f)));
			DungeonPrismPoint offset = candidate.kind == DUNGEON_OPTIC_PRISM
										   ? (DungeonPrismPoint){.012f, .012f}
										   : (DungeonPrismPoint){0, 0};
			candidate.position = sub(aim, rotate(offset, optic_angle(&candidate)));
			if (fabsf(candidate.position.x) > .282f || fabsf(candidate.position.y) > .282f)
				continue;
			bool clear = true;
			for (unsigned j = 0; j < i; ++j)
				if (length(sub(candidate.position, p->prisms[j].position)) < .115f)
					clear = false;
			DungeonPrismPoint exit, outgoing;
			if (!clear || !principal_exit(&candidate, origin, ray, &exit, &outgoing))
				continue;
			// Reject a direct return into hardware already placed.
			for (unsigned j = 0; j < i; ++j)
			{
				float dist = .11f;
				DungeonPrismPoint n;
				if (optic_hit(&p->prisms[j], exit, outgoing, &dist, &n))
					clear = false;
			}
			if (!clear)
				continue;
			p->prisms[i] = candidate;
			p->solution[i] = (uint16_t)angle;
			origin = exit;
			ray = outgoing;
			placed = true;
		}
		if (!placed)
			return false;
	}
	float distance = board_exit(origin, ray);
	p->key = (DungeonPrismPoint){2, 2}; // trace without a receiver before selecting its focus
	p->key_angle = atan2f(ray.y, ray.x);
	DungeonPrismTrace bundle = dungeon_prism_trace(p);
	float best_power = 0;
	for (unsigned sample = 0; sample < 48; ++sample)
	{
		float along = .06f + (distance - .08f) * sample / 47.f;
		if (along < .07f || along > distance - .018f)
			continue;
		DungeonPrismPoint key = add(origin, mul(ray, along));
		bool clear = fabsf(key.x) < .325f && fabsf(key.y) < .325f;
		for (unsigned j = 0; j < p->count; ++j)
			if (length(sub(key, p->prisms[j].position)) < .080f)
				clear = false;
		if (!clear || fabsf(cross(sub(key, p->source), direction(p->source_angle))) < .075f)
			continue;
		float power = 0;
		for (unsigned j = 0; j < bundle.count; ++j)
		{
			DungeonPrismSegment segment = bundle.segments[j];
			if (segment.in_glass)
				continue;
			DungeonPrismPoint travel = sub(segment.b, segment.a);
			float hit = receiver_distance(segment.a, unit(travel), key, p->key_angle);
			if (hit <= length(travel))
				power += segment.intensity / 3.f;
		}
		if (power > best_power)
		{
			best_power = power;
			p->key = key;
		}
	}
	if (best_power < DUNGEON_OPTIC_KEY_POWER + .03f)
		return false;
	for (unsigned start = 0; start < p->count; ++start)
	{
		bool seen[DUNGEON_PRISM_MAX] = {false};
		seen[start] = true;
		for (unsigned pass = 0; pass < p->count; ++pass)
			for (unsigned j = 0; j < p->count; ++j)
				if (seen[j])
					for (unsigned d = 0; d < 4; ++d)
						seen[dungeon_prism_neighbor(p, j, d)] = true;
		for (unsigned j = 0; j < p->count; ++j)
			if (!seen[j])
				return false;
	}
	DungeonPrismTrace trace = dungeon_prism_trace(p);
	return trace.hit_key && trace.lit_mask == (1u << p->count) - 1u;
}
void dungeon_prism_init(DungeonPrismPuzzle *p, uint32_t seed)
{
	if (!p)
		return;
	uint64_t rng = (uint64_t)seed * UINT64_C(0xD6E8FEB86659FD93) ^ UINT64_C(0x843ED1AC792B506F);
	bool valid = false;
	for (unsigned attempt = 0; attempt < 256 && !valid; ++attempt)
		valid = generate(p, &rng);
	if (!valid)
	{
		// Verified focused solution with two distinct lens kinds. Rigid
		// rotation/translation preserve its optics while varying placement.
		*p = (DungeonPrismPuzzle){
			.count = 5,
			.source = {-.344999999f, -.0448500067f},
			.source_angle = -.0440099835f,
			.key = {.00827110559f, .119878098f},
			.key_angle = 2.44076943f,
			.prisms = {{{-.213951796f, -.0412507392f}, 81, DUNGEON_OPTIC_PRISM},
					   {{-.0292890817f, -.139883608f}, 20, DUNGEON_OPTIC_CONCAVE},
					   {{.0806415677f, -.213524252f}, 305, DUNGEON_OPTIC_PRISM},
					   {{.227919832f, -.0819103047f}, 221, DUNGEON_OPTIC_PRISM},
					   {{.101412773f, .0409898274f}, 222, DUNGEON_OPTIC_CONVEX}},
			.solution = {81, 20, 305, 221, 222}};
		unsigned spin = random_index(&rng, 360);
		float angle = spin * PI / 180;
		DungeonPrismPoint shift = {random_range(&rng, -.004f, .004f),
								   random_range(&rng, -.004f, .004f)};
		p->source = add(rotate(p->source, angle), shift);
		p->key = add(rotate(p->key, angle), shift);
		p->source_angle += angle;
		p->key_angle += angle;
		for (unsigned i = 0; i < p->count; ++i)
		{
			p->prisms[i].position = add(rotate(p->prisms[i].position, angle), shift);
			p->prisms[i].orientation = p->solution[i] = (p->solution[i] + 360 - spin) % 360;
		}
	}
	for (unsigned i = 0; i < p->count; ++i)
		p->prisms[i].orientation = (p->solution[i] + 30 + random_index(&rng, 300)) % 360;
	for (unsigned attempt = 0; attempt < 360 && dungeon_prism_trace(p).hit_key; ++attempt)
		p->prisms[0].orientation = (p->prisms[0].orientation + 1) % 360;
	p->key_power = dungeon_prism_trace(p).key_power;
}
uint32_t dungeon_prism_neighbor(const DungeonPrismPuzzle *p, uint32_t from, DungeonPrismDirection d)
{
	if (!p || from >= p->count || (unsigned)d > 3)
		return from;
	const DungeonPrismPoint axes[4] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
	float best = FLT_MAX;
	unsigned selected = from;
	for (unsigned i = 0; i < p->count; ++i)
	{
		DungeonPrismPoint delta = sub(p->prisms[i].position, p->prisms[from].position);
		float forward = dot(delta, axes[d]);
		if (forward <= .0001f)
			continue;
		float sideways = fabsf(cross(delta, axes[d]));
		float score = length(delta) + sideways * 1.4f;
		if (score < best)
		{
			best = score;
			selected = i;
		}
	}
	return selected;
}
void dungeon_prism_move(DungeonPrismPuzzle *p, DungeonPrismDirection d)
{
	if (p && !p->solved && !p->rotating && p->turn_remaining <= 0)
		p->selected = dungeon_prism_neighbor(p, p->selected, d);
}
void dungeon_prism_confirm(DungeonPrismPuzzle *p)
{
	if (p && !p->solved && p->turn_remaining <= 0)
		p->rotating = !p->rotating;
}
void dungeon_prism_turn(DungeonPrismPuzzle *p, int direction)
{
	if (!p || p->solved || !p->rotating || p->selected >= p->count || p->turn_remaining > 0 ||
		!direction)
		return;
	p->turn_direction = direction > 0 ? 1 : -1;
	p->turn_remaining = DUNGEON_PRISM_TURN_SECONDS;
}
bool dungeon_prism_update(DungeonPrismPuzzle *p, float dt)
{
	if (!p || p->solved || p->turn_remaining <= 0 || !(dt > 0) || !isfinite(dt))
		return false;
	p->turn_remaining -= dt;
	if (p->turn_remaining > 0)
		return false;
	p->turn_remaining = 0;
	p->prisms[p->selected].orientation =
		(p->prisms[p->selected].orientation + p->turn_direction + 360) % 360;
	DungeonPrismTrace trace = dungeon_prism_trace(p);
	p->key_power = trace.key_power;
	p->solved = trace.hit_key;
	return p->solved;
}
