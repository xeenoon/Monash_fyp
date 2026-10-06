#pragma once

#include "dungeon_scene.h"
#include "input.h"
#include "overworld.h"
#include "ui_draw.h"

#include <stdbool.h>
#include <stdint.h>

/* The dungeon as a whole game: title screen, how-to-play, a level select of
 * procedurally themed dungeons, and the run itself -- a treasure chest at the
 * exit that completes the level, a hidden gem, a guardian that patrols and
 * chases, a game-over screen, restart, and a one-to-three star rating.
 *
 * It sits on top of DungeonScene/DungeonSession rather than inside them: the
 * lock puzzles, collision and rendering are unchanged, and the scripted test
 * harness still drives the bare scene with none of this running. */

#define DUNGEON_GAME_LEVELS 3u

typedef enum
{
	DUNGEON_GAME_TITLE,
	DUNGEON_GAME_HOW_TO_PLAY,
	DUNGEON_GAME_OVERWORLD, /* walking the Alps between the three entrances */
	DUNGEON_GAME_LOADING,
	DUNGEON_GAME_PLAYING,
	DUNGEON_GAME_PAUSED,
	DUNGEON_GAME_OVER,
	DUNGEON_GAME_COMPLETE
} DungeonGameScreen;

typedef struct
{
	const char *name;
	const char *blurb;
	float surface[3]; /* wall/floor albedo multiplier */
	float light[3];	  /* torch light colour multiplier */
	float flame[3];	  /* torch flame tint */
} DungeonBiome;

typedef struct
{
	uint32_t biome;
	uint32_t seed;
	uint32_t best_stars;
	bool completed;
	char name[64];
} DungeonGameLevel;

typedef enum
{
	GUARD_PATROL,
	GUARD_CHASE,
	GUARD_SEARCH
} DungeonGuardMode;

typedef struct
{
	bool active;
	DungeonPoint position, home, target, last_seen;
	float facing; /* radians, atan2(z, x) */
	DungeonGuardMode mode;
	float wait, repath, lost_timer, alert_flash;
	float bob;
} DungeonGuardian;

typedef enum
{
	DUNGEON_GAME_ACTION_NONE,
	DUNGEON_GAME_ACTION_LOAD, /* load levels[load_level]; then dungeon_game_begin_level */
	DUNGEON_GAME_ACTION_QUIT,
	DUNGEON_GAME_ACTION_RETURN /* back to the overworld, at levels[current]'s door */
} DungeonGameAction;

typedef struct
{
	DungeonGameScreen screen, return_screen;
	/* Which world is on screen: the overworld (title, walking, its pause
	 * menu) or the loaded dungeon. */
	bool in_overworld;
	const Overworld *overworld; /* for the HUD; set by the caller */
	/* The map: where each door's pin lands on the UI canvas this frame (set
	 * by the caller from the camera), and what the pointer is doing. */
	struct
	{
		float x, y;
		bool visible;
	} markers[DUNGEON_GAME_LEVELS];
	float pointer_x, pointer_y;
	int hovered, selected; /* -1 none */
	int fly_request;	   /* door the camera should fly to, or -1 */
	int cursor;
	float ui_time, screen_time;
	uint32_t session_seed;
	DungeonGameLevel levels[DUNGEON_GAME_LEVELS];
	uint32_t current;
	uint32_t load_level;
	int loading_frames;
	DungeonGameScreen after_load;

	/* The run. */
	float run_time;
	bool spotted, gem_taken, chest_open;
	float chest_lid; /* 0 shut .. 1 open, eased */
	float complete_timer, over_timer;
	uint32_t stars;
	DungeonPoint chest, gem;
	bool has_gem;
	DungeonGuardian guard;
	char banner[96];
	float banner_timer;
	char prompt[128];

	/* Navigation grid over the level field's corners (guardian pathing). */
	uint8_t *clear;		/* corner is far enough from rock for the guardian */
	int32_t *distance;	/* BFS scratch, steps to the current target */
	uint32_t *queue;
	uint32_t grid_w, grid_h;
} DungeonGame;

void dungeon_game_init(DungeonGame *game, uint32_t session_seed);
void dungeon_game_destroy(DungeonGame *game);

uint32_t dungeon_game_level_seed(const DungeonGame *game, uint32_t level);
/* Theme each level from the ground its entrance stands on. */
void dungeon_game_assign_grounds(DungeonGame *game, const OverworldGround grounds[DUNGEON_GAME_LEVELS]);
/* E at entrance `level` in the overworld. */
void dungeon_game_enter(DungeonGame *game, uint32_t level);
/* Map interaction for one frame, in UI canvas pixels: hover, click to select
 * a pin (and fly to it), click the card's Descend button, Tab to cycle,
 * Enter/E to descend into the selection. */
void dungeon_game_overworld_pointer(DungeonGame *game, float canvas_x, float canvas_y, bool click,
									bool tab, bool confirm);
const char *dungeon_game_biome_name(const DungeonGame *game, uint32_t level);

/* After the scene for levels[current] is (re)created: theme it, place the
 * chest, gem and guardian, and reset the run. */
void dungeon_game_begin_level(DungeonGame *game, DungeonScene *scene, uint32_t level);

/* Whether the scene's own explore/lock input should run this frame. */
bool dungeon_game_playing(const DungeonGame *game);

/* Menus and the run's own rules. Call once per frame before the scene update. */
DungeonGameAction dungeon_game_update(DungeonGame *game, const Input *input, DungeonScene *scene,
									  float dt);
/* Handles E near the chest. Returns true if it consumed the interact press. */
bool dungeon_game_interact(DungeonGame *game, DungeonScene *scene);
/* After the scene update: guardian, pickups, chest collision, props. */
void dungeon_game_post_update(DungeonGame *game, DungeonScene *scene, float dt);

/* Test support: a walkable point in front of the guardian, same side of every
 * shut door. */
DungeonPoint dungeon_game_debug_point_near_guard(DungeonGame *game, const DungeonScene *scene);

void dungeon_game_draw_ui(DungeonGame *game, const DungeonScene *scene, UiCanvas *canvas);
