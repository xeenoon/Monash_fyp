#pragma once

#include "dungeon_level.h"

/* A third dungeon frontend, alongside dungeon_grid (ASCII maps) and
 * dungeon_cave (procedural caves): one small square room holding exactly one
 * of everything worth looking at -- one torch, one puddle, and whatever moss
 * the standard scatter puts on eight metres of floor and wall.
 *
 * It exists because the game's camera cannot answer questions about a torch.
 * That camera is a fixed 14-metre orbit, so a flame is a dozen pixels across
 * and can only be seen from above; judging a flame's shape, its seating on the
 * fixture, or how far its bloom reaches from there is guessing. The lab builds
 * this room and hands the free-fly camera over, so the same geometry the real
 * dungeon draws can be walked around and put a nose against.
 *
 * Nothing downstream of dungeon_level_compile_field knows this frontend
 * exists -- it is the same DungeonLevel the cave produces, which is the point:
 * what is inspected here is the shipping material and shader path, not a
 * mock-up of it. */

/* Which fixture the room is built around. The room itself is identical in all
 * three; only what is burning in it changes, because the three cases put the
 * flame in genuinely different situations -- backed by a wall a hand's width
 * behind it, standing in open air with nothing behind it at all, and moving
 * with the player. A flame that reads correctly in one can be wrong in
 * another, which is exactly what the wall-mounted case turned out to hide. */
typedef enum
{
	DUNGEON_LAB_NONE = 0,
	DUNGEON_LAB_WALL = 1,	  /* bracket on the north wall, flame against stone */
	DUNGEON_LAB_STANDING = 2, /* free-standing torch on a pole, open air behind it */
	DUNGEON_LAB_CARRIED = 3,  /* the player cube carrying one, as the game draws it */
} DungeonLabScene;

/* Room interior size in metres. */
#define DUNGEON_LAB_ROOM_M 8.0f

bool dungeon_lab_compile(DungeonLevel *out, DungeonLevelError *error);

/* Where the lab wants its single torch, per scene. dungeon_scene_create places
 * one light at the relevant point instead of calling dungeon_lighting_build,
 * so the room holds one fire and no exit beacon competing with it.
 *
 * WALL: against the middle of the north wall -- mount_torch traces the
 *       rendered wall from here and reseats the fixture against it.
 * STANDING: the middle of the room, where the pole stands.
 * CARRIED: unused; the light rides the player. */
DungeonPoint dungeon_lab_torch_position(DungeonLabScene scene);

/* Height of the standing torch's flame above the floor, and the pole that
 * holds it up. Shared by the pole mesh and the light placement so the two
 * cannot drift apart. */
#define DUNGEON_LAB_POLE_HEIGHT_M 1.45f
#define DUNGEON_LAB_POLE_RADIUS_M 0.035f
/* The rag binding at the head: how far down the shaft the cloth reaches, and
 * how many bands are wound on. A torch head is fuel bound to a stick, so the
 * wrap is the thing that is burning -- the flame's collar comes down over it
 * and the cloth occludes whatever is behind. */
#define DUNGEON_LAB_WRAP_HEIGHT_M 0.15f
#define DUNGEON_LAB_WRAP_BANDS 6u
