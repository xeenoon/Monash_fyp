#pragma once

#include <stdbool.h>
#include <stdint.h>

#define DUNGEON_PRISM_MAX 5u
#define DUNGEON_PRISM_MAX_SEGMENTS 1024u
#define DUNGEON_PRISM_TURN_SECONDS .025f
#define DUNGEON_OPTIC_APERTURE .042f
#define DUNGEON_OPTIC_CURVATURE .12f
#define DUNGEON_OPTIC_EDGE .004f
#define DUNGEON_OPTIC_ANGLES 360u
#define DUNGEON_OPTIC_KEY_RADIUS .006f
#define DUNGEON_OPTIC_KEY_POWER .28f

typedef enum
{
	DUNGEON_PRISM_NORTH,
	DUNGEON_PRISM_EAST,
	DUNGEON_PRISM_SOUTH,
	DUNGEON_PRISM_WEST
} DungeonPrismDirection;
typedef enum
{
	DUNGEON_OPTIC_PRISM,
	DUNGEON_OPTIC_CONVEX,
	DUNGEON_OPTIC_CONCAVE
} DungeonOpticKind;
typedef struct
{
	float x, y;
} DungeonPrismPoint;
typedef struct
{
	DungeonPrismPoint position; /* board-local metres; no grid */
	uint16_t orientation;		/* clockwise degrees, including arbitrary angles */
	DungeonOpticKind kind;
} DungeonPrism;
typedef struct
{
	DungeonPrismPoint a, b;
	float intensity, width_a, width_b; /* metres; individual sampled ray footprint */
	bool in_glass;
	uint8_t channel; /* R/G/B: wavelength-dependent dispersion */
} DungeonPrismSegment;
typedef struct
{
	DungeonPrismSegment segments[DUNGEON_PRISM_MAX_SEGMENTS];
	uint32_t count, lit_mask, split_mask;
	float key_power;
	bool hit_key, limited;
} DungeonPrismTrace;
typedef struct
{
	DungeonPrism prisms[DUNGEON_PRISM_MAX];
	uint16_t solution[DUNGEON_PRISM_MAX];
	uint32_t count, selected;
	DungeonPrismPoint source, key;
	float source_angle, key_angle; /* radians, counterclockwise from board right */
	float key_power;			   /* current received power, not a time accumulator */
	bool rotating, solved;
	float turn_remaining;
	int turn_direction;
} DungeonPrismPuzzle;

void dungeon_prism_init(DungeonPrismPuzzle *puzzle, uint32_t seed);
DungeonPrismTrace dungeon_prism_trace(const DungeonPrismPuzzle *puzzle);
void dungeon_prism_move(DungeonPrismPuzzle *puzzle, DungeonPrismDirection direction);
void dungeon_prism_confirm(DungeonPrismPuzzle *puzzle);
void dungeon_prism_turn(DungeonPrismPuzzle *puzzle, int direction);
bool dungeon_prism_update(DungeonPrismPuzzle *puzzle, float dt);
uint32_t dungeon_prism_neighbor(const DungeonPrismPuzzle *puzzle, uint32_t from,
								DungeonPrismDirection direction);
/* Analytic optical primitives, also shared with geometry tests. */
float dungeon_lens_half_width(DungeonOpticKind kind, float y);
bool dungeon_optics_refract(DungeonPrismPoint incident, DungeonPrismPoint opposing_normal, float n1,
							float n2, DungeonPrismPoint *transmitted, float *reflectance);
