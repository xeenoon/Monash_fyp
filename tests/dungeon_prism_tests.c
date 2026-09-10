#include "dungeon_prism.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static void snell_and_total_internal_reflection(void)
{
	DungeonPrismPoint transmitted;
	float reflection;
	// Air into glass at 30 degrees: sin(theta_t) = sin(30)/1.5 = 1/3.
	assert(dungeon_optics_refract((DungeonPrismPoint){.8660254f, .5f}, (DungeonPrismPoint){-1, 0},
								  1, 1.5f, &transmitted, &reflection));
	assert(fabsf(transmitted.y - 1.f / 3) < 1e-5f);
	assert(reflection > .04f && reflection < .05f);
	assert(dungeon_optics_refract((DungeonPrismPoint){1, 0}, (DungeonPrismPoint){-1, 0}, 1, 1.5f,
								  &transmitted, &reflection));
	assert(fabsf(reflection - .04f) < 1e-5f);
	assert(!dungeon_optics_refract((DungeonPrismPoint){.5f, .8660254f}, (DungeonPrismPoint){-1, 0},
								   1.5f, 1, &transmitted, &reflection));
	assert(reflection == 1);
	// Parallel slab: the second interface restores the incoming direction.
	DungeonPrismPoint inside, outside;
	assert(dungeon_optics_refract((DungeonPrismPoint){.8660254f, .5f}, (DungeonPrismPoint){-1, 0},
								  1, 1.5f, &inside, &reflection));
	assert(
		dungeon_optics_refract(inside, (DungeonPrismPoint){-1, 0}, 1.5f, 1, &outside, &reflection));
	assert(fabsf(outside.y - .5f) < 1e-5f);
	assert(fabsf(outside.x - .8660254f) < 1e-5f);
}
static void lenses_focus_and_diverge(void)
{
	for (unsigned kind = DUNGEON_OPTIC_CONVEX; kind <= DUNGEON_OPTIC_CONCAVE; ++kind)
	{
		DungeonPrismPuzzle p = {.count = 1,
								.source = {-.30f, 0},
								.source_angle = 0,
								.key = {.345f, .2f},
								.prisms = {{{0, 0}, 0, (DungeonOpticKind)kind}}};
		DungeonPrismTrace t = dungeon_prism_trace(&p);
		unsigned checked = 0;
		for (unsigned i = 0; i < t.count; ++i)
		{
			DungeonPrismSegment s = t.segments[i];
			if (s.channel != 1 || s.intensity < .04f || s.a.x < 0 || s.a.x > .025f || s.b.x < .3f ||
				fabsf(s.a.y) < .002f)
				continue;
			float slope = (s.b.y - s.a.y) / (s.b.x - s.a.x);
			float at_focus = s.a.y + slope * (.14f - s.a.x);
			if (kind == DUNGEON_OPTIC_CONVEX)
			{
				assert(slope * s.a.y < 0); // bends toward optical axis
				assert(fabsf(at_focus) < fabsf(s.a.y) * .6f);
			}
			else
			{
				assert(slope * s.a.y > 0); // bends away from axis
				assert(fabsf(at_focus) > fabsf(s.a.y) * 1.5f);
			}
			++checked;
		}
		assert(checked >= 4);
	}
}
static void receiver_requires_concentration(void)
{
	DungeonPrismPuzzle p = {.source = {-.30f, 0}, .key = {.25f, 0}, .key_angle = 0};
	DungeonPrismTrace direct = dungeon_prism_trace(&p);
	assert(direct.hit_key && direct.key_power > .9f && direct.key_power < 1.001f);
	for (unsigned i = 0; i < direct.count; ++i)
	{
		assert(fabsf(direct.segments[i].a.y - direct.segments[i].b.y) < 1e-6f);
		assert(fabsf(direct.segments[i].b.x - p.key.x) < 1e-5f);
	}
	p.count = 1;
	p.prisms[0] = (DungeonPrism){{0, 0}, 0, DUNGEON_OPTIC_CONVEX};
	p.key.x = .12f;
	DungeonPrismTrace focused = dungeon_prism_trace(&p);
	assert(focused.hit_key && focused.key_power > .8f);
	p.prisms[0].kind = DUNGEON_OPTIC_CONCAVE;
	p.key.x = .30f;
	DungeonPrismTrace spread = dungeon_prism_trace(&p);
	assert(!spread.hit_key && spread.key_power > 0);
	p.key.y = .20f;
	assert(dungeon_prism_trace(&p).key_power == 0);
}
static void generation(void)
{
	for (unsigned seed = 0; seed < 10000; ++seed)
	{
		DungeonPrismPuzzle p, again;
		dungeon_prism_init(&p, seed);
		dungeon_prism_init(&again, seed);
		assert(memcmp(&p, &again, sizeof(p)) == 0);
		assert(p.count == 5 && !dungeon_prism_trace(&p).hit_key);
		unsigned lenses = 0, off_axis = 0;
		for (unsigned i = 0; i < p.count; ++i)
		{
			lenses |= 1u << p.prisms[i].kind;
			off_axis += p.solution[i] % 90 != 0;
			for (unsigned j = 0; j < i; ++j)
				assert(hypotf(p.prisms[i].position.x - p.prisms[j].position.x,
							  p.prisms[i].position.y - p.prisms[j].position.y) > .11f);
			p.prisms[i].orientation = p.solution[i];
		}
		assert(lenses == 7 && off_axis >= 1);
		DungeonPrismTrace t = dungeon_prism_trace(&p);
		assert(t.hit_key && t.key_power >= DUNGEON_OPTIC_KEY_POWER &&
			   t.count <= DUNGEON_PRISM_MAX_SEGMENTS);
		assert(t.lit_mask == 31);
		for (unsigned start = 0; start < p.count; ++start)
		{
			bool seen[DUNGEON_PRISM_MAX] = {false};
			seen[start] = true;
			for (unsigned pass = 0; pass < p.count; ++pass)
				for (unsigned i = 0; i < p.count; ++i)
					if (seen[i])
						for (unsigned d = 0; d < 4; ++d)
							seen[dungeon_prism_neighbor(&p, i, d)] = true;
			for (unsigned i = 0; i < p.count; ++i)
				assert(seen[i]);
		}
	}
}
static void controls(void)
{
	DungeonPrismPuzzle p;
	dungeon_prism_init(&p, 42);
	unsigned original = p.prisms[0].orientation;
	dungeon_prism_turn(&p, 1);
	assert(p.turn_remaining == 0);
	dungeon_prism_confirm(&p);
	dungeon_prism_move(&p, DUNGEON_PRISM_EAST);
	assert(p.selected == 0);
	dungeon_prism_turn(&p, -1);
	dungeon_prism_confirm(&p);
	assert(p.rotating);
	dungeon_prism_update(&p, DUNGEON_PRISM_TURN_SECONDS / 2);
	assert(p.prisms[0].orientation == original);
	dungeon_prism_update(&p, DUNGEON_PRISM_TURN_SECONDS);
	assert(p.prisms[0].orientation == (original + 359) % 360);
}
int main(void)
{
	snell_and_total_internal_reflection();
	lenses_focus_and_diverge();
	controls();
	receiver_requires_concentration();
	generation();
	puts("Optical physics, focusing/defocusing, controls and 10,000 solvable off-grid layouts "
		 "passed.");
	return 0;
}
