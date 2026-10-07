#include "dungeon_game.h"
#include "dungeon_collision.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GUARD_RADIUS_M 0.5f
#define GUARD_PATROL_SPEED 1.5f
#define GUARD_CHASE_SPEED 3.1f /* the player walks at 3.5: outrunnable, barely */
#define GUARD_SIGHT_M 8.0f
#define GUARD_CLOSE_SIGHT_M 2.5f /* heard, not seen: no view cone this close */
#define GUARD_FOV_COS 0.34f		 /* ~140 degree cone */
#define GUARD_CATCH_M 0.8f
#define GUARD_LOSE_S 3.0f
#define CHEST_REACH_M 1.8f
#define CHEST_BLOCK_M 0.85f
#define GEM_PICKUP_M 0.9f

static const DungeonBiome biomes[] = {
	{"Ember Vault", "Volcanic halls where the stone still holds the heat of the forge.",
	 {1.05f, 0.76f, 0.60f}, {1.10f, 0.80f, 0.62f}, {1.0f, 0.8f, 0.65f}},
	{"Frost Catacombs", "Burial vaults sealed in ice; the torches burn a cold blue.",
	 {0.72f, 0.86f, 1.10f}, {0.55f, 0.75f, 1.25f}, {0.55f, 0.8f, 1.6f}},
	{"Verdant Ruins", "A temple swallowed by the jungle. Moss eats the carvings.",
	 {0.78f, 0.98f, 0.68f}, {0.75f, 1.05f, 0.6f}, {0.7f, 1.3f, 0.6f}},
	{"Sandstone Tomb", "A desert king's tomb, dry as bone and gilded in torchlight.",
	 {1.22f, 1.02f, 0.70f}, {1.10f, 0.95f, 0.72f}, {1.0f, 1.0f, 1.0f}},
	{"Obsidian Deep", "Black glass caverns lit by violet witch-fire.",
	 {0.52f, 0.50f, 0.62f}, {0.85f, 0.55f, 1.25f}, {0.9f, 0.55f, 1.6f}},
	{"Sunken Grotto", "Flooded sea caves where the light runs green and cold.",
	 {0.70f, 0.92f, 0.95f}, {0.6f, 1.0f, 1.0f}, {0.6f, 1.2f, 1.3f}},
};
#define BIOME_COUNT (sizeof(biomes) / sizeof(biomes[0]))

static const char *const campaign_names[DUNGEON_GAME_LEVELS] = {
	"The Ashen Crown",       "The Frozen Reliquary", "The Mossbound Temple",
	"The Sunken Archive",    "The Obsidian Chapel",  "The Gilded Sepulchre",
	"The Hollow Bastion",    "The Whispering Vault", "The Drowned Labyrinth",
	"The Cinder Warrens",    "The Ivory Catacomb",   "The Shattered Sanctum",
	"The Moonlit Crypt",     "The Nameless Halls",   "The Thorned Depths",
	"The Silent Ossuary",    "The Iron Passages",    "The Buried Observatory",
	"The Serpent Chambers",  "The Last Necropolis",
};

static const UiColor WHITE = {240, 232, 214, 255};
static const UiColor GOLD = {255, 200, 80, 255};
static const UiColor DIM = {170, 160, 140, 255};
static const UiColor RED = {235, 70, 55, 255};

static uint32_t hash_u32(uint32_t x)
{
	x ^= x >> 16;
	x *= 0x7feb352du;
	x ^= x >> 15;
	x *= 0x846ca68bu;
	x ^= x >> 16;
	return x;
}

static uint32_t game_random(DungeonGame *game)
{
	game->session_seed = hash_u32(game->session_seed + 0x9e3779b9u);
	return game->session_seed;
}

static float randf(DungeonGame *game)
{
	return (float)(game_random(game) & 0xFFFFFFu) / (float)0x1000000u;
}

void dungeon_game_init(DungeonGame *game, uint32_t session_seed)
{
	*game = (DungeonGame){0};
	monk_dialogue_init(&game->monk_dialogue);
	game->session_seed = session_seed ? session_seed : 1u;
	/* A fresh, balanced rotation of biomes every session. There are more
	 * dungeons than biome families, but every dungeon has its own layout. */
	uint32_t order[BIOME_COUNT];
	for (uint32_t i = 0; i < BIOME_COUNT; ++i)
		order[i] = i;
	for (uint32_t i = BIOME_COUNT - 1u; i > 0; --i)
	{
		uint32_t j = game_random(game) % (i + 1u);
		uint32_t t = order[i];
		order[i] = order[j];
		order[j] = t;
	}
	for (uint32_t i = 0; i < DUNGEON_GAME_LEVELS; ++i)
	{
		DungeonGameLevel *level = &game->levels[i];
		level->biome = order[i % BIOME_COUNT];
		level->seed = 1u + game_random(game) % 99991u;
		snprintf(level->name, sizeof(level->name), "%s", campaign_names[i]);
	}
	game->screen = DUNGEON_GAME_TITLE;
	game->in_overworld = true;
	game->hovered = game->selected = game->fly_request = -1;
}

void dungeon_game_assign_grounds(DungeonGame *game, const OverworldGround grounds[DUNGEON_GAME_LEVELS])
{
	/* Ground to biome. With twenty entrances, repeating a terrain-appropriate
	 * family is preferable to forcing an unrelated theme merely for uniqueness.
	 * Scree is generic broken rock, so it alternates Ember and Sandstone. */
	for (uint32_t i = 0; i < DUNGEON_GAME_LEVELS; ++i)
	{
		uint32_t want;
		switch (grounds[i])
		{
		case OVERWORLD_GROUND_SNOW: want = 1; break;	/* Frost Catacombs */
		case OVERWORLD_GROUND_CLIFF: want = 4; break;	/* Obsidian Deep */
		case OVERWORLD_GROUND_MEADOW: want = 2; break;	/* Verdant Ruins */
		case OVERWORLD_GROUND_VALLEY: want = 5; break;	/* Sunken Grotto */
		default: want = i & 1u ? 0u : 3u; break; /* Ember / Sandstone */
		}
		game->levels[i].biome = want;
	}
}

void dungeon_game_enter(DungeonGame *game, uint32_t level)
{
	if (game->screen != DUNGEON_GAME_OVERWORLD)
		return;
	game->load_level = level % DUNGEON_GAME_LEVELS;
	game->screen = DUNGEON_GAME_LOADING;
	game->loading_frames = 2;
}

const char *dungeon_game_biome_name(const DungeonGame *game, uint32_t level)
{
	return biomes[game->levels[level % DUNGEON_GAME_LEVELS].biome].name;
}

void dungeon_game_destroy(DungeonGame *game)
{
	free(game->clear);
	free(game->distance);
	free(game->queue);
	game->clear = NULL;
	game->distance = NULL;
	game->queue = NULL;
}

static bool save_progress(const DungeonGame *game)
{
	if (!game->progress_path[0])
		return false;
	char temporary[sizeof(game->progress_path) + 8u];
	int written = snprintf(temporary, sizeof(temporary), "%s.tmp", game->progress_path);
	if (written < 0 || (size_t)written >= sizeof(temporary))
		return false;
	FILE *file = fopen(temporary, "wb");
	if (!file)
		return false;
	bool ok = fprintf(file, "DUNGEON_PROGRESS 1 %u\n", DUNGEON_GAME_LEVELS) > 0;
	for (uint32_t i = 0; ok && i < DUNGEON_GAME_LEVELS; ++i)
		ok = fprintf(file, "%u %u %u\n", i, game->levels[i].completed ? 1u : 0u,
					 game->levels[i].best_stars) > 0;
	ok = fclose(file) == 0 && ok;
	if (ok)
		ok = rename(temporary, game->progress_path) == 0;
	if (!ok)
		remove(temporary);
	return ok;
}

void dungeon_game_set_progress_path(DungeonGame *game, const char *path)
{
	if (!game || !path || !*path)
		return;
	snprintf(game->progress_path, sizeof(game->progress_path), "%s", path);
	FILE *file = fopen(game->progress_path, "rb");
	if (!file)
		return;
	unsigned version = 0, count = 0;
	bool ok = fscanf(file, "DUNGEON_PROGRESS %u %u", &version, &count) == 2 && version == 1u &&
			  count == DUNGEON_GAME_LEVELS;
	for (uint32_t expected = 0; ok && expected < DUNGEON_GAME_LEVELS; ++expected)
	{
		unsigned id = 0, completed = 0, stars = 0;
		ok = fscanf(file, "%u %u %u", &id, &completed, &stars) == 3 && id == expected &&
			 completed <= 1u && stars <= 3u && (!completed || stars > 0u);
		if (ok)
		{
			game->levels[expected].completed = completed != 0u;
			game->levels[expected].best_stars = stars;
		}
	}
	fclose(file);
	if (!ok)
	{
		for (uint32_t i = 0; i < DUNGEON_GAME_LEVELS; ++i)
		{
			game->levels[i].completed = false;
			game->levels[i].best_stars = 0;
		}
		fprintf(stderr, "Dungeon progress: ignored invalid file %s\n", game->progress_path);
	}
	else
	{
		uint32_t completed = 0;
		for (uint32_t i = 0; i < DUNGEON_GAME_LEVELS; ++i)
			completed += game->levels[i].completed ? 1u : 0u;
		printf("Dungeon progress: loaded %u / %u completed from %s\n", completed,
			   DUNGEON_GAME_LEVELS, game->progress_path);
	}
}

uint32_t dungeon_game_level_seed(const DungeonGame *game, uint32_t level)
{
	return game->levels[level % DUNGEON_GAME_LEVELS].seed;
}

void dungeon_game_set_level_seed(DungeonGame *game, uint32_t level, uint32_t seed)
{
	uint32_t index = level % DUNGEON_GAME_LEVELS;
	DungeonGameLevel *entry = &game->levels[index];
	entry->seed = seed;
	snprintf(entry->name, sizeof(entry->name), "%s", campaign_names[index]);
}

bool dungeon_game_playing(const DungeonGame *game)
{
	return game->screen == DUNGEON_GAME_PLAYING && game->over_timer <= 0.0f &&
		   !game->chest_open;
}

/* --- Navigation grid ----------------------------------------------------- */

static float point_distance(DungeonPoint a, DungeonPoint b)
{
	float dx = a.x - b.x, dz = a.z - b.z;
	return sqrtf(dx * dx + dz * dz);
}

static float segment_distance(DungeonPoint p, DungeonSegment s)
{
	float vx = s.b.x - s.a.x, vz = s.b.z - s.a.z;
	float len2 = vx * vx + vz * vz;
	float t = len2 > 0.0f ? ((p.x - s.a.x) * vx + (p.z - s.a.z) * vz) / len2 : 0.0f;
	t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
	DungeonPoint q = {s.a.x + vx * t, s.a.z + vz * t};
	return point_distance(p, q);
}

static uint32_t corner_index(const DungeonGame *game, const DungeonField *field, DungeonPoint p)
{
	float fx = (p.x - field->origin.x) / field->cell_size;
	float fz = (p.z - field->origin.z) / field->cell_size;
	int x = (int)lroundf(fx), z = (int)lroundf(fz);
	x = x < 0 ? 0 : (x >= (int)game->grid_w ? (int)game->grid_w - 1 : x);
	z = z < 0 ? 0 : (z >= (int)game->grid_h ? (int)game->grid_h - 1 : z);
	return (uint32_t)z * game->grid_w + (uint32_t)x;
}

static DungeonPoint corner_world(const DungeonGame *game, const DungeonField *field, uint32_t i)
{
	return dungeon_field_corner_world(field, i % game->grid_w, i / game->grid_w);
}

static bool door_blocks(const DungeonScene *scene, DungeonPoint p, bool ignore_doors)
{
	if (ignore_doors)
		return false;
	for (uint32_t d = 0; d < scene->level.door_count && d < scene->session.door_count; ++d)
		if (!scene->session.doors[d].open &&
			segment_distance(p, scene->level.doors[d].blocker) < GUARD_RADIUS_M + 0.25f)
			return true;
	return false;
}

/* Breadth-first step counts from `start` over clear corners (8-connected, no
 * corner cutting) into game->distance; -1 is unreachable. */
static void bfs(DungeonGame *game, const DungeonScene *scene, DungeonPoint start, bool ignore_doors)
{
	const DungeonField *field = &scene->level.field;
	uint32_t n = game->grid_w * game->grid_h;
	for (uint32_t i = 0; i < n; ++i)
		game->distance[i] = -1;
	uint32_t s = corner_index(game, field, start);
	if (!game->clear[s])
	{
		/* Start just off the grid's clear set (a wall-hugging player): seed
		 * from the nearest clear corner in a small window instead. */
		float best = 1e9f;
		uint32_t best_i = s;
		int sx = (int)(s % game->grid_w), sz = (int)(s / game->grid_w);
		for (int dz = -4; dz <= 4; ++dz)
			for (int dx = -4; dx <= 4; ++dx)
			{
				int x = sx + dx, z = sz + dz;
				if (x < 0 || z < 0 || x >= (int)game->grid_w || z >= (int)game->grid_h)
					continue;
				uint32_t i = (uint32_t)z * game->grid_w + (uint32_t)x;
				float d = (float)(dx * dx + dz * dz);
				if (game->clear[i] && d < best)
				{
					best = d;
					best_i = i;
				}
			}
		s = best_i;
		if (!game->clear[s])
			return;
	}
	uint32_t head = 0, tail = 0;
	game->distance[s] = 0;
	game->queue[tail++] = s;
	while (head < tail)
	{
		uint32_t i = game->queue[head++];
		int x = (int)(i % game->grid_w), z = (int)(i / game->grid_w);
		for (int dz = -1; dz <= 1; ++dz)
			for (int dx = -1; dx <= 1; ++dx)
			{
				if (!dx && !dz)
					continue;
				int nx = x + dx, nz = z + dz;
				if (nx < 0 || nz < 0 || nx >= (int)game->grid_w || nz >= (int)game->grid_h)
					continue;
				uint32_t j = (uint32_t)nz * game->grid_w + (uint32_t)nx;
				if (game->distance[j] >= 0 || !game->clear[j])
					continue;
				if (dx && dz &&
					(!game->clear[(uint32_t)z * game->grid_w + (uint32_t)nx] ||
					 !game->clear[(uint32_t)nz * game->grid_w + (uint32_t)x]))
					continue;
				if (door_blocks(scene, corner_world(game, field, j), ignore_doors))
					continue;
				game->distance[j] = game->distance[i] + 1;
				game->queue[tail++] = j;
			}
	}
}

static bool line_of_sight(const DungeonScene *scene, DungeonPoint a, DungeonPoint b)
{
	float length = point_distance(a, b);
	int steps = (int)(length / 0.2f) + 1;
	for (int i = 1; i < steps; ++i)
	{
		float t = (float)i / (float)steps;
		DungeonPoint p = {a.x + (b.x - a.x) * t, a.z + (b.z - a.z) * t};
		if (dungeon_field_sample(&scene->level.field, p) < 0.5f)
			return false;
	}
	/* Nobody sees through a shut door. */
	for (uint32_t d = 0; d < scene->level.door_count && d < scene->session.door_count; ++d)
	{
		if (scene->session.doors[d].open)
			continue;
		DungeonSegment door = scene->level.doors[d].blocker;
		float rx = b.x - a.x, rz = b.z - a.z;
		float sx = door.b.x - door.a.x, sz = door.b.z - door.a.z;
		float denominator = rx * sz - rz * sx;
		if (fabsf(denominator) < 1e-6f)
			continue;
		float qx = door.a.x - a.x, qz = door.a.z - a.z;
		float t = (qx * sz - qz * sx) / denominator;
		float u = (qx * rz - qz * rx) / denominator;
		if (t >= 0.0f && t <= 1.0f && u >= -0.1f && u <= 1.1f)
			return false;
	}
	return true;
}

/* --- Level setup ---------------------------------------------------------- */

static void theme_scene(const DungeonGame *game, DungeonScene *scene)
{
	const DungeonBiome *biome = &biomes[game->levels[game->current].biome];
	memcpy(scene->surface_tint, biome->surface, sizeof(biome->surface));
	for (uint32_t i = 0; i < scene->light_count; ++i)
	{
		DungeonLight *light = &scene->lights[i];
		if (point_distance(light->position, scene->level.exit) < 0.5f)
			continue; /* the exit beacon keeps its own colour: it marks the goal */
		for (int c = 0; c < 3; ++c)
		{
			light->color[c] *= biome->light[c];
			light->flame_tint[c] *= biome->flame[c];
		}
	}
}

void dungeon_game_begin_level(DungeonGame *game, DungeonScene *scene, uint32_t level)
{
	game->current = level % DUNGEON_GAME_LEVELS;
	dialogue_close(&game->monk_dialogue);
	game->screen = DUNGEON_GAME_PLAYING;
	game->in_overworld = false;
	game->run_time = 0.0f;
	game->spotted = game->gem_taken = game->chest_open = false;
	game->chest_lid = 0.0f;
	game->complete_timer = game->over_timer = 0.0f;
	game->stars = 0;
	game->banner[0] = '\0';
	game->banner_timer = 0.0f;
	game->guard = (DungeonGuardian){0};
	game->has_gem = false;
	theme_scene(game, scene);

	const DungeonField *field = &scene->level.field;
	free(game->clear);
	free(game->distance);
	free(game->queue);
	game->grid_w = field->width;
	game->grid_h = field->height;
	uint32_t n = game->grid_w * game->grid_h;
	game->clear = calloc(n, 1);
	game->distance = malloc(sizeof(int32_t) * n);
	game->queue = malloc(sizeof(uint32_t) * n);
	float *to_solid = malloc(sizeof(float) * n);
	int32_t *from_spawn_shut = malloc(sizeof(int32_t) * n);
	int32_t *from_spawn = malloc(sizeof(int32_t) * n);
	if (!game->clear || !game->distance || !game->queue || !to_solid || !from_spawn_shut ||
		!from_spawn)
	{
		free(to_solid);
		free(from_spawn_shut);
		free(from_spawn);
		return;
	}
	dungeon_field_distance_to_solid(field, 0.5f, to_solid);
	for (uint32_t i = 0; i < n; ++i)
		game->clear[i] = to_solid[i] >= GUARD_RADIUS_M;

	game->chest = scene->level.exit;
	bfs(game, scene, scene->level.spawn, false);
	memcpy(from_spawn_shut, game->distance, sizeof(int32_t) * n);
	bfs(game, scene, scene->level.spawn, true);
	memcpy(from_spawn, game->distance, sizeof(int32_t) * n);
	bfs(game, scene, scene->level.exit, true);
	/* The exit itself often sits too close to rock to be a clear corner; the
	 * BFS seeded from the nearest clear one, which is where distance is 0. */
	int32_t exit_steps = -1;
	for (uint32_t i = 0; i < n && exit_steps < 0; ++i)
		if (game->distance[i] == 0)
			exit_steps = from_spawn[i];
	float cell = field->cell_size;

	/* The gem: as far as possible from both the route's ends, somewhere the
	 * player has to go looking. */
	int32_t best_gem = -1;
	for (uint32_t i = 0; i < n; ++i)
	{
		if (from_spawn[i] < 0 || game->distance[i] < 0 || to_solid[i] < 0.7f)
			continue;
		int32_t score = from_spawn[i] < game->distance[i] ? from_spawn[i] : game->distance[i];
		if (score > best_gem)
		{
			best_gem = score;
			game->gem = corner_world(game, field, i);
		}
	}
	game->has_gem = best_gem * cell > 6.0f;

	/* The guardian: behind the first locked door if there is one, a third to
	 * two-thirds of the way along the route, in a room big enough to patrol. */
	float best_guard = -1.0f;
	for (int pass = 0; pass < 2 && best_guard < 0.0f; ++pass)
		for (uint32_t i = 0; i < n; ++i)
		{
			if (from_spawn[i] < 0 || to_solid[i] < 1.1f)
				continue;
			if (pass == 0 && from_spawn_shut[i] >= 0)
				continue; /* first pass: only past a door */
			float along = exit_steps > 0 ? (float)from_spawn[i] / (float)exit_steps : 0.0f;
			if (along < 0.3f || along > 0.75f || from_spawn[i] * cell < 12.0f)
				continue;
			float score = to_solid[i] + randf(game) * 0.6f;
			if (score > best_guard)
			{
				best_guard = score;
				game->guard.home = corner_world(game, field, i);
			}
		}
	if (best_guard > 0.0f && scene->lab == DUNGEON_LAB_NONE)
	{
		game->guard.active = true;
		game->guard.position = game->guard.home;
		game->guard.target = game->guard.home;
		game->guard.mode = GUARD_PATROL;
		game->guard.wait = 1.5f;
	}
	free(to_solid);
	free(from_spawn_shut);
	free(from_spawn);
	snprintf(game->banner, sizeof(game->banner), "%s", game->levels[game->current].name);
	game->banner_timer = 3.5f;
	printf("Dungeon game: level %u '%s' (%s), chest (%.1f, %.1f), gem %s (%.1f, %.1f), "
		   "guardian %s (%.1f, %.1f)\n",
		   game->current, game->levels[game->current].name,
		   biomes[game->levels[game->current].biome].name, game->chest.x, game->chest.z,
		   game->has_gem ? "yes" : "no", game->gem.x, game->gem.z,
		   game->guard.active ? "yes" : "no", game->guard.home.x, game->guard.home.z);
}

/* --- Menus ---------------------------------------------------------------- */

static DungeonGameAction request_load(DungeonGame *game, uint32_t level)
{
	game->load_level = level % DUNGEON_GAME_LEVELS;
	game->screen = DUNGEON_GAME_LOADING;
	game->loading_frames = 2; /* paint the loading card before blocking on the load */
	return DUNGEON_GAME_ACTION_NONE;
}

static DungeonGameAction return_to_overworld(DungeonGame *game)
{
	game->screen = DUNGEON_GAME_OVERWORLD;
	game->cursor = 0;
	game->screen_time = 0.0f;
	game->in_overworld = true;
	return DUNGEON_GAME_ACTION_RETURN;
}

static int menu_step(DungeonGame *game, const Input *in, int count)
{
	if (in->menu_up)
		game->cursor = (game->cursor + count - 1) % count;
	if (in->menu_down)
		game->cursor = (game->cursor + 1) % count;
	return in->puzzle_confirm || in->interact ? game->cursor : -1;
}

static void go(DungeonGame *game, DungeonGameScreen screen)
{
	game->return_screen = game->screen;
	game->screen = screen;
	game->cursor = 0;
	game->screen_time = 0.0f;
}

DungeonGameAction dungeon_game_update(DungeonGame *game, const Input *in, DungeonScene *scene,
									  float dt)
{
	game->ui_time += dt;
	game->screen_time += dt;
	if (game->banner_timer > 0.0f)
		game->banner_timer -= dt;
	switch (game->screen)
	{
	case DUNGEON_GAME_TITLE:
	{
		int choice = menu_step(game, in, 3);
		if (choice == 0)
			go(game, DUNGEON_GAME_OVERWORLD);
		else if (choice == 1)
			go(game, DUNGEON_GAME_HOW_TO_PLAY);
		else if (choice == 2 || in->escape)
			return DUNGEON_GAME_ACTION_QUIT;
		break;
	}
	case DUNGEON_GAME_HOW_TO_PLAY:
		if (in->escape || in->puzzle_confirm || in->interact)
		{
			DungeonGameScreen back = game->return_screen;
			go(game, back == DUNGEON_GAME_PAUSED ? DUNGEON_GAME_PAUSED : DUNGEON_GAME_TITLE);
			game->cursor = back == DUNGEON_GAME_PAUSED ? 2 - (game->in_overworld ? 1 : 0) : 1;
		}
		break;
	case DUNGEON_GAME_OVERWORLD:
		/* Walking and the entrances are driven from main (overworld.c);
		 * entering one comes back through dungeon_game_enter. */
		if (in->escape)
			go(game, DUNGEON_GAME_PAUSED);
		break;
	case DUNGEON_GAME_LOADING:
		if (--game->loading_frames <= 0)
			return DUNGEON_GAME_ACTION_LOAD;
		break;
	case DUNGEON_GAME_PLAYING:
		game->run_time += dt;
		if (game->over_timer > 0.0f)
		{
			game->over_timer -= dt;
			if (game->over_timer <= 0.0f)
				go(game, DUNGEON_GAME_OVER);
			break;
		}
		if (game->chest_open)
		{
			game->complete_timer += dt;
			if (game->complete_timer > 2.2f)
				go(game, DUNGEON_GAME_COMPLETE);
			break;
		}
		if (in->escape)
		{
			if (scene->session.phase != DUNGEON_PHASE_EXPLORING)
				dungeon_session_cancel(&scene->session);
			go(game, DUNGEON_GAME_PAUSED);
		}
		else if (in->restart && scene->session.phase == DUNGEON_PHASE_EXPLORING)
			return request_load(game, game->current);
		break;
	case DUNGEON_GAME_DIALOGUE:
	{
		Dialogue *d = &game->monk_dialogue;
		dialogue_update(d, dt);
		if (in->escape)
			dialogue_close(d);
		else
		{
			if (d->phase == DIALOGUE_CHOICES)
			{
				if (in->menu_up)
					d->cursor = (d->cursor + 2) % 3;
				if (in->menu_down)
					d->cursor = (d->cursor + 1) % 3;
			}
			for (unsigned i = 0; i < in->edit_count; ++i)
				dialogue_edit(d, &in->edits[i]);
			if (in->puzzle_confirm)
				dialogue_confirm(d);
		}
		if (!d->active)
			game->screen = DUNGEON_GAME_PLAYING;
		break;
	}
	case DUNGEON_GAME_PAUSED:
		if (game->in_overworld)
		{
			/* Resume, How to Play, Title Screen, Quit Game. */
			int choice = menu_step(game, in, 4);
			if (in->escape || choice == 0)
			{
				game->screen = DUNGEON_GAME_OVERWORLD;
				game->cursor = 0;
			}
			else if (choice == 1)
				go(game, DUNGEON_GAME_HOW_TO_PLAY);
			else if (choice == 2)
				go(game, DUNGEON_GAME_TITLE);
			else if (choice == 3)
				return DUNGEON_GAME_ACTION_QUIT;
			break;
		}
		else
		{
			/* Resume, Restart Level, How to Play, Leave Dungeon, Quit Game. */
			int choice = menu_step(game, in, 5);
			if (in->escape || choice == 0)
			{
				game->screen = DUNGEON_GAME_PLAYING;
				game->cursor = 0;
			}
			else if (choice == 1 || in->restart)
				return request_load(game, game->current);
			else if (choice == 2)
				go(game, DUNGEON_GAME_HOW_TO_PLAY);
			else if (choice == 3)
				return return_to_overworld(game);
			else if (choice == 4)
				return DUNGEON_GAME_ACTION_QUIT;
		}
		break;
	case DUNGEON_GAME_OVER:
	{
		int choice = menu_step(game, in, 2);
		if (choice == 0 || in->restart)
			return request_load(game, game->current);
		if (choice == 1 || in->escape)
			return return_to_overworld(game);
		break;
	}
	case DUNGEON_GAME_COMPLETE:
	{
		int choice = menu_step(game, in, 2);
		if (choice == 0 || in->escape)
			return return_to_overworld(game);
		if (choice == 1 || in->restart)
			return request_load(game, game->current);
		break;
	}
	}
	return DUNGEON_GAME_ACTION_NONE;
}

bool dungeon_game_monk_in_reach(const DungeonGame *game, const DungeonScene *scene)
{
	(void)game;
	if (!scene->monk.active || scene->session.phase != DUNGEON_PHASE_EXPLORING ||
		point_distance(scene->player.position, scene->level.monk_spawn) > 2.0f)
		return false;
	/* Reject interactions through a corner wall. */
	for (int i = 1; i < 16; ++i)
	{
		float t = i / 16.0f;
		DungeonPoint p = {scene->player.position.x * (1 - t) + scene->level.monk_spawn.x * t,
						  scene->player.position.z * (1 - t) + scene->level.monk_spawn.z * t};
		if (dungeon_field_sample(&scene->level.field, p) < .5f)
			return false;
	}
	return true;
}

bool dungeon_game_interact(DungeonGame *game, DungeonScene *scene)
{
	if (dungeon_game_playing(game) && dungeon_game_monk_in_reach(game, scene))
	{
		dialogue_open(&game->monk_dialogue);
		game->screen = DUNGEON_GAME_DIALOGUE;
		return true;
	}
	if (game->chest_open || point_distance(scene->player.position, game->chest) > CHEST_REACH_M)
		return false;
	game->chest_open = true;
	game->complete_timer = 0.0f;
	game->stars = 1u + (game->spotted ? 0u : 1u) + (game->gem_taken ? 1u : 0u);
	DungeonGameLevel *level = &game->levels[game->current];
	level->completed = true;
	if (game->stars > level->best_stars)
		level->best_stars = game->stars;
	if (game->progress_path[0] && !save_progress(game))
		fprintf(stderr, "Dungeon progress: could not save %s\n", game->progress_path);
	return true;
}

/* --- Guardian ------------------------------------------------------------- */

static void guard_pick_patrol_target(DungeonGame *game, const DungeonScene *scene)
{
	/* Somewhere it can actually walk to from here, with the doors as they are
	 * now, and not too close to home so the patrol covers the room. */
	const DungeonField *field = &scene->level.field;
	bfs(game, scene, game->guard.position, false);
	float cell = field->cell_size;
	uint32_t n = game->grid_w * game->grid_h;
	for (int attempt = 0; attempt < 400; ++attempt)
	{
		uint32_t i = game_random(game) % n;
		float steps = (float)game->distance[i] * cell;
		if (game->distance[i] < 0 || steps < 4.0f || steps > 16.0f)
			continue;
		DungeonPoint p = corner_world(game, field, i);
		if (point_distance(p, game->guard.home) > 14.0f)
			continue;
		game->guard.target = p;
		game->guard.repath = 0.0f;
		return;
	}
	game->guard.target = game->guard.home;
}

/* One step along the BFS gradient toward the target set by the last bfs(). */
static bool guard_step(DungeonGame *game, const DungeonScene *scene, float speed, float dt)
{
	DungeonGuardian *g = &game->guard;
	const DungeonField *field = &scene->level.field;
	if (g->repath <= 0.0f)
	{
		bfs(game, scene, g->target, false);
		g->repath = g->mode == GUARD_CHASE ? 0.25f : 1.0f;
	}
	if (point_distance(g->position, g->target) < 0.3f)
		return true;
	uint32_t here = corner_index(game, field, g->position);
	int x = (int)(here % game->grid_w), z = (int)(here / game->grid_w);
	int32_t best = game->distance[here] >= 0 ? game->distance[here] : INT32_MAX;
	DungeonPoint goal = g->target;
	bool found = false;
	for (int dz = -2; dz <= 2; ++dz)
		for (int dx = -2; dx <= 2; ++dx)
		{
			int nx = x + dx, nz = z + dz;
			if (nx < 0 || nz < 0 || nx >= (int)game->grid_w || nz >= (int)game->grid_h)
				continue;
			int32_t d = game->distance[(uint32_t)nz * game->grid_w + (uint32_t)nx];
			if (d >= 0 && d < best)
			{
				best = d;
				goal = dungeon_field_corner_world(field, (uint32_t)nx, (uint32_t)nz);
				found = true;
			}
		}
	if (!found && game->distance[here] < 0)
		return true; /* stranded (target unreachable): give up on it */
	if (!found)
		goal = g->target;
	float dx = goal.x - g->position.x, dz = goal.z - g->position.z;
	float length = sqrtf(dx * dx + dz * dz);
	if (length > 1e-4f)
	{
		float step = fminf(speed * dt, length);
		g->position.x += dx / length * step;
		g->position.z += dz / length * step;
		/* Turn toward travel smoothly rather than snapping. */
		float want = atan2f(dz, dx);
		float delta = remainderf(want - g->facing, 6.2831853f);
		g->facing += delta * fminf(1.0f, dt * 8.0f);
	}
	return false;
}

static void update_guard(DungeonGame *game, DungeonScene *scene, float dt)
{
	DungeonGuardian *g = &game->guard;
	if (!g->active)
		return;
	if (getenv("DUNGEON_GAME_DEBUG") && (int)(g->bob * 2.0f) != (int)((g->bob + dt) * 2.0f))
		printf("guard mode=%d pos=(%.2f,%.2f) target=(%.2f,%.2f) facing=%.2f player=(%.2f,%.2f) "
			   "dist=%.2f los=%d run=%.1f\n",
			   g->mode, g->position.x, g->position.z, g->target.x, g->target.z, g->facing,
			   scene->player.position.x, scene->player.position.z,
			   point_distance(g->position, scene->player.position),
			   line_of_sight(scene, g->position, scene->player.position), game->run_time);
	g->bob += dt;
	if (g->alert_flash > 0.0f)
		g->alert_flash -= dt;
	g->repath -= dt;
	DungeonPoint player = scene->player.position;
	float distance = point_distance(player, g->position);
	bool sees = false;
	if (distance < GUARD_SIGHT_M && game->run_time > 2.0f)
	{
		float fx = cosf(g->facing), fz = sinf(g->facing);
		float dot = distance > 1e-3f
						? ((player.x - g->position.x) * fx + (player.z - g->position.z) * fz) / distance
						: 1.0f;
		sees = (distance < GUARD_CLOSE_SIGHT_M || dot > GUARD_FOV_COS) &&
			   line_of_sight(scene, g->position, player);
	}
	if (sees)
	{
		if (g->mode != GUARD_CHASE)
		{
			g->alert_flash = 1.2f;
			g->repath = 0.0f;
			if (!game->spotted)
			{
				snprintf(game->banner, sizeof(game->banner), "You've been spotted!");
				game->banner_timer = 2.0f;
			}
			game->spotted = true;
		}
		g->mode = GUARD_CHASE;
		g->last_seen = player;
		g->lost_timer = 0.0f;
	}
	switch (g->mode)
	{
	case GUARD_PATROL:
		if (g->wait > 0.0f)
		{
			g->wait -= dt;
			g->facing += dt * 0.9f; /* looks around while it waits */
			if (g->wait <= 0.0f)
				guard_pick_patrol_target(game, scene);
			break;
		}
		if (guard_step(game, scene, GUARD_PATROL_SPEED, dt))
			g->wait = 1.0f + randf(game) * 1.5f;
		break;
	case GUARD_CHASE:
		if (!sees)
		{
			g->lost_timer += dt;
			if (g->lost_timer > GUARD_LOSE_S)
			{
				g->mode = GUARD_SEARCH;
				g->target = g->last_seen;
				g->repath = 0.0f;
				break;
			}
		}
		if (point_distance(g->target, g->last_seen) > 0.6f)
		{
			g->target = g->last_seen;
			if (g->repath > 0.25f)
				g->repath = 0.0f;
		}
		guard_step(game, scene, GUARD_CHASE_SPEED, dt);
		break;
	case GUARD_SEARCH:
		if (guard_step(game, scene, GUARD_PATROL_SPEED * 1.3f, dt))
		{
			g->mode = GUARD_PATROL;
			g->wait = 2.5f;
		}
		break;
	}
	if (distance < GUARD_CATCH_M && game->over_timer <= 0.0f)
	{
		game->over_timer = 1.2f;
		snprintf(game->banner, sizeof(game->banner), "Caught by the guardian!");
		game->banner_timer = 1.5f;
	}
}

DungeonPoint dungeon_game_debug_point_near_guard(DungeonGame *game, const DungeonScene *scene)
{
	/* A walkable spot about four metres in front of the guardian, on its side
	 * of every shut door -- for scripted tests of sighting and capture. */
	const DungeonField *field = &scene->level.field;
	bfs(game, scene, game->guard.position, false);
	DungeonPoint ahead = {game->guard.position.x + cosf(game->guard.facing) * 4.0f,
						  game->guard.position.z + sinf(game->guard.facing) * 4.0f};
	DungeonPoint best = game->guard.position;
	float best_score = 1e9f;
	for (uint32_t i = 0; i < game->grid_w * game->grid_h; ++i)
	{
		if (game->distance[i] < 0)
			continue;
		DungeonPoint p = corner_world(game, field, i);
		float score = point_distance(p, ahead);
		if (score < best_score && point_distance(p, game->guard.position) > 3.0f)
		{
			best_score = score;
			best = p;
		}
	}
	game->guard.repath = 0.0f;
	return best;
}

/* --- Props ---------------------------------------------------------------- */

static void add_prop(DungeonScene *scene, DungeonProp prop)
{
	if (scene->prop_count < DUNGEON_MAX_PROPS)
		scene->props[scene->prop_count++] = prop;
}

/* Local (x, y, z) offset rotated by yaw about +Y and added to a base. */
static WorldPosition offset(DungeonPoint base, float floor_y, float yaw, float x, float y, float z)
{
	float c = cosf(yaw), s = sinf(yaw);
	return (WorldPosition){base.x + c * x + s * z, floor_y + y, base.z - s * x + c * z};
}

static void build_props(DungeonGame *game, DungeonScene *scene)
{
	scene->prop_count = 0;
	float floor = scene->level.floor_y;
	/* Chest: wooden body, gold bands, a lid hinged along its back edge. */
	float yaw = 0.0f;
	DungeonPoint c = game->chest;
	const float wood[3] = {0.36f, 0.19f, 0.07f}, gold[3] = {1.0f, 0.72f, 0.22f};
	add_prop(scene, (DungeonProp){offset(c, floor, yaw, 0, 0.13f, 0), {0.95f, 0.46f, 0.62f}, yaw, 0,
								  {wood[0], wood[1], wood[2]}, 0.0f});
	for (int band = -1; band <= 1; band += 2)
		add_prop(scene, (DungeonProp){offset(c, floor, yaw, band * 0.3f, 0.12f, 0),
									  {0.08f, 0.48f, 0.64f}, yaw, 0,
									  {gold[0], gold[1], gold[2]}, 1.0f});
	float lid = game->chest_lid * 1.85f;
	float hinge_z = -0.31f, hinge_y = 0.13f + 0.46f;
	/* The lid's base centre sits half its depth in front of the hinge, swung
	 * up and back by `lid` radians about local X. */
	float ly = hinge_y + sinf(lid) * 0.31f, lz = hinge_z + cosf(lid) * 0.31f;
	add_prop(scene, (DungeonProp){offset(c, floor, yaw, 0, ly, lz), {0.97f, 0.16f, 0.64f}, yaw,
								  -lid, {wood[0] * 1.1f, wood[1] * 1.1f, wood[2]}, 0.0f});
	add_prop(scene, (DungeonProp){offset(c, floor, yaw, 0, ly + 0.01f, lz),
								  {0.99f, 0.17f, 0.12f}, yaw, -lid, {gold[0], gold[1], gold[2]},
								  1.0f});
	if (game->chest_open)
	{
		float rise = fminf(game->complete_timer * 0.6f, 0.25f);
		add_prop(scene, (DungeonProp){offset(c, floor, yaw, 0, 0.45f + rise, 0),
									  {0.7f, 0.12f, 0.45f}, yaw + game->ui_time, 0,
									  {1.6f, 1.15f, 0.35f}, 1.0f});
	}
	/* The hidden gem: a slowly spinning, hovering crystal. */
	if (game->has_gem && !game->gem_taken)
	{
		float hover = 0.55f + 0.08f * sinf(game->ui_time * 2.2f);
		float spin = game->ui_time * 1.6f;
		add_prop(scene, (DungeonProp){offset(game->gem, floor, 0, 0, hover, 0), {0.26f, 0.26f, 0.26f},
									  spin, 0.62f, {0.35f, 1.6f, 1.9f}, 0.0f});
		add_prop(scene, (DungeonProp){offset(game->gem, floor, 0, 0, hover + 0.04f, 0),
									  {0.2f, 0.2f, 0.2f}, -spin * 1.3f, -0.62f,
									  {0.6f, 1.9f, 2.0f}, 0.0f});
	}
	/* The guardian: a hulking dark figure with eyes that burn when it hunts. */
	DungeonGuardian *g = &game->guard;
	if (g->active)
	{
		float face = 1.5707963f - g->facing; /* local +Z along facing */
		float step = sinf(g->bob * (g->mode == GUARD_CHASE ? 12.0f : 6.0f)) * 0.04f;
		const float body[3] = {0.55f, 0.12f, 0.08f};
		add_prop(scene, (DungeonProp){offset(g->position, floor, face, 0, 0, 0), {0.7f, 1.25f, 0.5f},
									  face, 0, {body[0], body[1], body[2]}, 0.3f});
		add_prop(scene, (DungeonProp){offset(g->position, floor, face, 0, 1.25f + step, 0.02f),
									  {0.46f, 0.42f, 0.44f}, face, 0,
									  {body[0] * 1.3f, body[1], body[2]}, 0.3f});
		for (int side = -1; side <= 1; side += 2)
			add_prop(scene, (DungeonProp){offset(g->position, floor, face, side * 0.36f, 0.55f - step,
												 0.06f),
										  {0.18f, 0.62f, 0.2f}, face, 0.0f,
										  {body[0], body[1], body[2]}, 0.3f});
		bool hunting = g->mode == GUARD_CHASE;
		float eye[3] = {hunting ? 4.0f : 2.4f, hunting ? 0.3f : 1.6f, hunting ? 0.15f : 0.3f};
		for (int side = -1; side <= 1; side += 2)
			add_prop(scene, (DungeonProp){offset(g->position, floor, face, side * 0.1f, 1.48f + step,
												 0.23f),
										  {0.09f, 0.05f, 0.03f}, face, 0,
										  {eye[0], eye[1], eye[2]}, 0.0f});
	}
}

void dungeon_game_post_update(DungeonGame *game, DungeonScene *scene, float dt)
{
	if (game->screen == DUNGEON_GAME_PLAYING && !scene->conversation_paused &&
		game->over_timer <= 0.0f)
	{
		if (scene->monk.active)
		{
			DungeonPoint *player = &scene->player.position;
			DungeonPoint home = scene->level.monk_spawn;
			float distance = point_distance(*player, home);
			float radius = .72f + scene->player.radius;
			if (distance < radius)
			{
				float dx = distance > .0001f ? (player->x - home.x) / distance : .70710678f;
				float dz = distance > .0001f ? (player->z - home.z) / distance : .70710678f;
				DungeonPoint delta = {dx * (radius - distance), dz * (radius - distance)};
				*player =
					dungeon_collision_move(scene->session.colliders, scene->session.collider_count,
										   *player, delta, scene->player.radius);
			}
		}
		if (!game->chest_open && scene->session.phase == DUNGEON_PHASE_EXPLORING)
			update_guard(game, scene, dt);
		/* The chest is solid: push the player back out of it. */
		DungeonPoint *p = &scene->player.position;
		float d = point_distance(*p, game->chest);
		if (d < CHEST_BLOCK_M && d > 1e-4f)
		{
			p->x = game->chest.x + (p->x - game->chest.x) / d * CHEST_BLOCK_M;
			p->z = game->chest.z + (p->z - game->chest.z) / d * CHEST_BLOCK_M;
		}
		if (game->has_gem && !game->gem_taken && point_distance(*p, game->gem) < GEM_PICKUP_M)
		{
			game->gem_taken = true;
			snprintf(game->banner, sizeof(game->banner), "Hidden gem found!");
			game->banner_timer = 2.5f;
		}
	}
	if (game->chest_open)
		game->chest_lid += (1.0f - game->chest_lid) * fminf(1.0f, dt * 3.0f);
	build_props(game, scene);
}

/* --- UI ------------------------------------------------------------------- */

#define W ((int)RENDERER_UI_WIDTH)
#define H ((int)RENDERER_UI_HEIGHT)

static const char *const TITLE_ITEMS[] = {"Play", "How to Play", "Quit"};
static const char *const DIALOGUE_ITEMS[] = {"Say something", "Keep listening", "Leave"};
static const char *const DUNGEON_PAUSE_ITEMS[] = {"Resume", "Restart Level", "How to Play",
											  "Leave Dungeon", "Quit Game"};
static const char *const OVERWORLD_PAUSE_ITEMS[] = {"Resume", "How to Play", "Title Screen",
												"Quit Game"};
static const char *const GAME_OVER_ITEMS[] = {"Restart Level", "Return to the Surface"};
static const char *const COMPLETE_ITEMS[] = {"Return to the Surface", "Replay"};

static bool menu_spec(const DungeonGame *game, int *y, const char *const **items, int *count)
{
	switch (game->screen)
	{
	case DUNGEON_GAME_DIALOGUE:
		if (game->monk_dialogue.phase != DIALOGUE_CHOICES)
			return false;
		*y = 392;
		*items = DIALOGUE_ITEMS;
		*count = 3;
		return true;
	case DUNGEON_GAME_TITLE:
		*y = 290;
		*items = TITLE_ITEMS;
		*count = 3;
		return true;
	case DUNGEON_GAME_PAUSED:
		*y = 190;
		*items = game->in_overworld ? OVERWORLD_PAUSE_ITEMS : DUNGEON_PAUSE_ITEMS;
		*count = game->in_overworld ? 4 : 5;
		return true;
	case DUNGEON_GAME_OVER:
		*y = 290;
		*items = GAME_OVER_ITEMS;
		*count = 2;
		return true;
	case DUNGEON_GAME_COMPLETE:
		*y = 350;
		*items = COMPLETE_ITEMS;
		*count = 2;
		return true;
	default: return false;
	}
}

bool dungeon_game_menu_pointer(DungeonGame *game, float x, float y, bool click)
{
	if (game->screen == DUNGEON_GAME_DIALOGUE && (game->monk_dialogue.phase == DIALOGUE_REVEAL ||
												  game->monk_dialogue.phase == DIALOGUE_PAGE))
		return click && x >= 35 && x <= W - 35 && y >= 243 && y <= 527;
	int top = 0, count = 0;
	const char *const *items = NULL;
	if (!menu_spec(game, &top, &items, &count))
		return false;
	for (int i = 0; i < count; ++i)
	{
		int line = top + i * 44;
		int width = ui_text_width(UI_FONT_BODY, items[i]) + 70;
		if (x >= (float)(W / 2 - width / 2) && x <= (float)(W / 2 + width / 2) &&
			y >= (float)(line - 4) && y <= (float)(line + 34))
		{
			game->cursor = i;
			if (game->screen == DUNGEON_GAME_DIALOGUE)
				game->monk_dialogue.cursor = (unsigned)i;
			return click;
		}
	}
	return false;
}

static void dim_screen(UiCanvas *c, uint8_t alpha)
{
	ui_fill_rect(c, 0, 0, W, H, (UiColor){6, 5, 8, alpha});
}

static void panel(UiCanvas *c, int x, int y, int w, int h)
{
	ui_fill_rect(c, x, y, w, h, (UiColor){14, 11, 9, 215});
	ui_frame_rect(c, x, y, w, h, 2, (UiColor){150, 112, 52, 230});
	ui_frame_rect(c, x + 5, y + 5, w - 10, h - 10, 1, (UiColor){90, 68, 34, 200});
}

static void menu(UiCanvas *c, const DungeonGame *game, int y, const char *const *items, int count)
{
	for (int i = 0; i < count; ++i)
	{
		bool selected = i == game->cursor;
		int line = y + i * 44;
		if (selected)
		{
			int w = ui_text_width(UI_FONT_BODY, items[i]) + 70;
			ui_fill_rect(c, W / 2 - w / 2, line - 4, w, 38, (UiColor){120, 80, 25, 120});
			float pulse = 0.5f + 0.5f * sinf(game->ui_time * 5.0f);
			ui_text(c, UI_FONT_BODY, W / 2 - w / 2 + 12 + (int)(pulse * 4), line, UI_ALIGN_LEFT, GOLD,
					">");
		}
		ui_text(c, UI_FONT_BODY, W / 2, line, UI_ALIGN_CENTRE, selected ? GOLD : WHITE, items[i]);
	}
}

static void stars_row(UiCanvas *c, int cx, int cy, float radius, uint32_t filled, float reveal)
{
	for (int i = 0; i < 3; ++i)
	{
		float x = (float)cx + (float)(i - 1) * radius * 2.4f;
		bool on = (uint32_t)i < filled && reveal > (float)i;
		float r = radius;
		if (on && reveal < (float)i + 1.0f)
			r *= 1.0f + 0.5f * (1.0f - (reveal - (float)i)); /* pop in */
		ui_star(c, x, (float)cy, r, GOLD, on);
	}
}

static void footer(UiCanvas *c, const char *text)
{
	ui_text(c, UI_FONT_SMALL, W / 2, H - 34, UI_ALIGN_CENTRE, DIM, text);
}

static void format_time(float seconds, char *out, size_t n)
{
	int s = (int)seconds;
	snprintf(out, n, "%d:%02d", s / 60, s % 60);
}

static void draw_title(DungeonGame *game, UiCanvas *c)
{
	ui_gradient_rect(c, 0, 0, W, H, (UiColor){5, 4, 6, 150}, (UiColor){5, 4, 6, 235});
	ui_text(c, UI_FONT_TITLE, W / 2, 92, UI_ALIGN_CENTRE, GOLD, "SILENT LABYRINTH");
	ui_fill_rect(c, W / 2 - 200, 182, 400, 2, (UiColor){150, 112, 52, 220});
	ui_text(c, UI_FONT_BODY, W / 2, 196, UI_ALIGN_CENTRE, DIM,
			"Pick the locks. Evade the guardian. Claim the treasure.");
	menu(c, game, 290, TITLE_ITEMS, 3);
	footer(c, "W/S or arrows to choose   -   Enter to select");
}

static void draw_how_to_play(DungeonGame *game, UiCanvas *c)
{
	(void)game;
	dim_screen(c, 200);
	panel(c, 70, 30, W - 140, H - 60);
	ui_text(c, UI_FONT_HEADING, W / 2, 46, UI_ALIGN_CENTRE, GOLD, "How to Play");
	int y = 108, x = 110, col = 300;
	static const char *rows[][2] = {
		{"Map", "Drag to pan, scroll to zoom, click a pin, then Descend"},
		{"Move", "W A S D        Turn camera: Left / Right arrows"},
		{"Interact", "E  -  pick a lock, open the treasure chest"},
		{"Pin tumbler", "Left/Right choose a pin, Up/Down raise it until it sets. Enter tries."},
		{"Safe", "Left/Right choose a pin, Enter presses. Find the hidden order."},
		{"Light lock", "Arrows pick an optic, Enter grabs it, turn to focus light on the keyhole."},
		{"Leave a lock", "Q or E"},
		{"Pause / Restart", "Esc  /  R"},
	};
	for (size_t i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i)
	{
		ui_text(c, UI_FONT_SMALL, x, y, UI_ALIGN_LEFT, GOLD, rows[i][0]);
		ui_text(c, UI_FONT_SMALL, x + col - 140, y, UI_ALIGN_LEFT, WHITE, rows[i][1]);
		y += 28;
	}
	y += 10;
	ui_fill_rect(c, x, y, W - 220, 1, (UiColor){150, 112, 52, 160});
	y += 14;
	ui_text_wrapped(c, UI_FONT_SMALL, x, y, W - 220, WHITE,
					"Find the treasure chest at the end of each dungeon to complete it. Locked doors "
					"stand in the way - every one must be picked. A guardian patrols the halls: if "
					"it catches you, the run is over. Earn a star for finishing, a star for never "
					"being spotted, and a star for finding the hidden gem.");
	footer(c, "Enter or Esc to go back");
}

/* The selection card's Descend button, in canvas pixels. */
#define CARD_W 520
#define CARD_H 170
#define CARD_X (W / 2 - CARD_W / 2)
#define CARD_Y (H - CARD_H - 20)
#define BUTTON_W 170
#define BUTTON_H 44
#define BUTTON_X (CARD_X + CARD_W - BUTTON_W - 20)
#define BUTTON_Y (CARD_Y + CARD_H - BUTTON_H - 18)

void dungeon_game_overworld_pointer(DungeonGame *game, float x, float y, bool click, bool tab,
									bool confirm)
{
	game->pointer_x = x;
	game->pointer_y = y;
	if (game->screen != DUNGEON_GAME_OVERWORLD)
		return;
	game->hovered = -1;
	float best = 30.0f * 30.0f;
	for (uint32_t i = 0; i < DUNGEON_GAME_LEVELS; ++i)
	{
		if (!game->markers[i].visible)
			continue;
		/* The pin's head sits 46 px above the door; either end selects. */
		float dx = x - game->markers[i].x;
		float dy_foot = y - game->markers[i].y, dy_head = y - (game->markers[i].y - 46.0f);
		float d = fminf(dx * dx + dy_foot * dy_foot, dx * dx + dy_head * dy_head);
		if (d < best)
		{
			best = d;
			game->hovered = (int)i;
		}
	}
	bool on_button = game->selected >= 0 && x >= BUTTON_X && x <= BUTTON_X + BUTTON_W &&
					 y >= BUTTON_Y && y <= BUTTON_Y + BUTTON_H;
	if (click)
	{
		if (on_button)
		{
			dungeon_game_enter(game, (uint32_t)game->selected);
			return;
		}
		if (game->hovered >= 0)
		{
			game->selected = game->hovered;
			game->fly_request = game->hovered;
		}
		else if (!(x >= CARD_X && x <= CARD_X + CARD_W && y >= CARD_Y && y <= CARD_Y + CARD_H))
			game->selected = -1; /* a click on open ground puts the card away */
	}
	if (tab)
	{
		game->selected = (game->selected + 1) % (int)DUNGEON_GAME_LEVELS;
		game->fly_request = game->selected;
	}
	if (confirm && game->selected >= 0)
		dungeon_game_enter(game, (uint32_t)game->selected);
}

static void draw_pin(DungeonGame *game, UiCanvas *c, uint32_t i)
{
	if (!game->markers[i].visible)
		return;
	const DungeonGameLevel *level = &game->levels[i];
	bool hot = game->hovered == (int)i || game->selected == (int)i;
	int x = (int)game->markers[i].x, y = (int)game->markers[i].y;
	UiColor head = hot ? GOLD : (UiColor){200, 70, 50, 255};
	/* Stalk, shadowed so it reads on snow as well as rock. */
	ui_fill_rect(c, x - 1, y - 40, 4, 40, (UiColor){0, 0, 0, 120});
	ui_fill_rect(c, x - 2, y - 42, 3, 42, (UiColor){240, 232, 214, 230});
	ui_fill_rect(c, x - 4, y - 2, 8, 4, (UiColor){0, 0, 0, 160});
	/* Head: a disc with the dungeon number. */
	float pulse = hot ? 1.0f + 0.08f * sinf(game->ui_time * 6.0f) : 1.0f;
	int r = (int)(15.0f * pulse);
	for (int dy = -r - 2; dy <= r + 2; ++dy)
		for (int dx = -r - 2; dx <= r + 2; ++dx)
		{
			int d2 = dx * dx + dy * dy;
			if (d2 <= (r + 2) * (r + 2))
				ui_fill_rect(c, x + dx, y - 46 + dy, 1, 1,
							 d2 <= r * r ? head : (UiColor){20, 14, 10, 230});
		}
	char number[12];
	snprintf(number, sizeof(number), "%u", i + 1u);
	ui_text(c, UI_FONT_BODY, x, y - 46 - 15, UI_ALIGN_CENTRE, (UiColor){20, 14, 10, 255}, number);
	/* Earned stars stay visible on the map even when the full label is folded. */
	if (level->best_stars)
		for (uint32_t s = 0; s < 3; ++s)
			ui_star(c, (float)(x - 12 + (int)s * 12), (float)(y - 18), 5.0f, GOLD,
					s < level->best_stars);
	/* Twenty full labels would bury the map. Keep the numbered pins compact and
	 * expand only the one the player is inspecting. */
	if (!hot)
		return;
	/* Label to the right of the head. */
	const char *name = level->name;
	int w = ui_text_width(UI_FONT_SMALL, name) + 16;
	ui_fill_rect(c, x + 20, y - 58, w, 46, (UiColor){0, 0, 0, hot ? 170 : 120});
	ui_text(c, UI_FONT_SMALL, x + 28, y - 56, UI_ALIGN_LEFT, hot ? GOLD : WHITE, name);
}

static void draw_overworld_hud(DungeonGame *game, UiCanvas *c)
{
	ui_fill_rect(c, 0, 0, W, 44, (UiColor){0, 0, 0, 110});
	uint32_t completed = 0;
	for (uint32_t i = 0; i < DUNGEON_GAME_LEVELS; ++i)
		completed += game->levels[i].completed ? 1u : 0u;
	char progress[96];
	snprintf(progress, sizeof(progress), "%u dungeons lie under these hills.  %u / %u complete.",
			 DUNGEON_GAME_LEVELS, completed, DUNGEON_GAME_LEVELS);
	ui_text(c, UI_FONT_SMALL, W - 16, 12, UI_ALIGN_RIGHT, WHITE, progress);
	for (uint32_t i = 0; i < DUNGEON_GAME_LEVELS; ++i)
		if ((int)i != game->selected && (int)i != game->hovered)
			draw_pin(game, c, i);
	if (game->hovered >= 0 && game->hovered != game->selected)
		draw_pin(game, c, (uint32_t)game->hovered);
	if (game->selected >= 0)
	{
		draw_pin(game, c, (uint32_t)game->selected);
		const DungeonGameLevel *level = &game->levels[game->selected];
		const DungeonBiome *biome = &biomes[level->biome];
		panel(c, CARD_X, CARD_Y, CARD_W, CARD_H);
		char title[96];
		snprintf(title, sizeof(title), "%u.  %s", game->selected + 1, biome->name);
		ui_text(c, UI_FONT_BODY, CARD_X + 22, CARD_Y + 14, UI_ALIGN_LEFT, GOLD, title);
		ui_text(c, UI_FONT_SMALL, CARD_X + 22, CARD_Y + 48, UI_ALIGN_LEFT, DIM, level->name);
		ui_text_wrapped(c, UI_FONT_SMALL, CARD_X + 22, CARD_Y + 74, CARD_W - 44, WHITE, biome->blurb);
		stars_row(c, CARD_X + 70, BUTTON_Y + BUTTON_H / 2, 13.0f, level->best_stars, 3.0f);
		bool on_button = game->pointer_x >= BUTTON_X && game->pointer_x <= BUTTON_X + BUTTON_W &&
						 game->pointer_y >= BUTTON_Y && game->pointer_y <= BUTTON_Y + BUTTON_H;
		ui_fill_rect(c, BUTTON_X, BUTTON_Y, BUTTON_W, BUTTON_H,
					 on_button ? (UiColor){190, 130, 40, 240} : (UiColor){120, 80, 25, 220});
		ui_frame_rect(c, BUTTON_X, BUTTON_Y, BUTTON_W, BUTTON_H, 2, GOLD);
		ui_text(c, UI_FONT_BODY, BUTTON_X + BUTTON_W / 2, BUTTON_Y + 7, UI_ALIGN_CENTRE, WHITE,
				"Descend");
	}
	else
		footer(c, "Drag to pan   Scroll to zoom   Right-drag to rotate   Click a pin to choose a dungeon");
}

static void draw_loading(DungeonGame *game, UiCanvas *c)
{
	ui_fill_rect(c, 0, 0, W, H, (UiColor){4, 3, 5, 255});
	const DungeonGameLevel *level = &game->levels[game->load_level];
	ui_text(c, UI_FONT_SMALL, W / 2, H / 2 - 70, UI_ALIGN_CENTRE, DIM, "Descending into");
	ui_text(c, UI_FONT_HEADING, W / 2, H / 2 - 42, UI_ALIGN_CENTRE, GOLD,
			biomes[level->biome].name);
	ui_text(c, UI_FONT_BODY, W / 2, H / 2 + 10, UI_ALIGN_CENTRE, WHITE, level->name);
	ui_text(c, UI_FONT_SMALL, W / 2, H / 2 + 70, UI_ALIGN_CENTRE, DIM, "Loading...");
}

static void draw_hud(DungeonGame *game, const DungeonScene *scene, UiCanvas *c)
{
	const DungeonGameLevel *level = &game->levels[game->current];
	const DungeonSession *session = &scene->session;
	bool picking = session->phase != DUNGEON_PHASE_EXPLORING;
	/* Top left: where you are, and how long you've been here. */
	ui_fill_rect(c, 0, 0, 330, 62, (UiColor){0, 0, 0, 110});
	ui_text(c, UI_FONT_SMALL, 14, 8, UI_ALIGN_LEFT, GOLD, biomes[level->biome].name);
	char line[160];
	char clock[16];
	format_time(game->run_time, clock, sizeof(clock));
	uint32_t shut = 0;
	for (uint32_t i = 0; i < session->door_count; ++i)
		shut += session->doors[i].open ? 0u : 1u;
	snprintf(line, sizeof(line), "%s   %s   Locked doors: %u", level->name, clock, shut);
	ui_text(c, UI_FONT_SMALL, 14, 32, UI_ALIGN_LEFT, WHITE, line);
	/* Top right: the three stars' objectives. */
	int ox = W - 270;
	ui_fill_rect(c, ox - 10, 0, 280, 96, (UiColor){0, 0, 0, 110});
	struct
	{
		const char *text;
		bool done, failed;
	} goals[3] = {{"Open the treasure chest", game->chest_open, false},
				  {"Stay unseen by the guardian", !game->spotted && game->chest_open, game->spotted},
				  {game->has_gem ? "Find the hidden gem" : "No gem in this dungeon", game->gem_taken,
				   !game->has_gem}};
	for (int i = 0; i < 3; ++i)
	{
		UiColor col = goals[i].failed ? (UiColor){130, 70, 60, 255} : goals[i].done ? GOLD : WHITE;
		ui_star(c, (float)ox + 8.0f, 18.0f + (float)i * 28.0f, 9.0f, GOLD,
				goals[i].done || (i == 1 && !goals[i].failed));
		ui_text(c, UI_FONT_SMALL, ox + 24, 7 + i * 28, UI_ALIGN_LEFT, col, goals[i].text);
	}
	/* Bottom: context prompt or the lock's controls and status. */
	line[0] = '\0';
	char status[160] = {0};
	if (picking)
	{
		if (session->phase == DUNGEON_PHASE_PIN_TUMBLER)
			snprintf(line, sizeof(line),
					 "Left/Right: choose pin    Up/Down: raise/lower    Enter: try    Q: leave");
		else if (session->phase == DUNGEON_PHASE_SAFE_PINS)
			snprintf(line, sizeof(line), "Left/Right: choose pin    Enter: press    Q: leave");
		else
			snprintf(line, sizeof(line),
					 "Arrows: choose optic    Enter: grab / release    Left/Right: rotate    Q: leave");
		char full[160] = {0};
		dungeon_session_status_text(session, full, sizeof(full));
		/* Drop the "Dungeon | Lockpick | " window-title prefix. */
		const char *text = full;
		for (int skip = 0; skip < 2 && strstr(text, " | "); ++skip)
			if (!strncmp(text, "Dungeon", 7) || !strncmp(text, "Lockpick", 8))
				text = strstr(text, " | ") + 3;
		snprintf(status, sizeof(status), "%s", text);
	}
	else if (dungeon_game_monk_in_reach(game, scene))
		snprintf(line, sizeof(line), "[E]  Talk to the monk");
	else if (!game->chest_open && point_distance(scene->player.position, game->chest) < CHEST_REACH_M)
		snprintf(line, sizeof(line), "[E]  Open the treasure chest");
	else
	{
		uint32_t door = dungeon_session_nearest_door(session, scene->player.position);
		if (door != UINT32_MAX && door < scene->level.door_count)
		{
			DungeonLockKind kind = scene->level.doors[door].lock;
			snprintf(line, sizeof(line), "[E]  Pick the %s",
					 kind == DUNGEON_LOCK_SAFE_PINS	 ? "safe lock"
					 : kind == DUNGEON_LOCK_PRISM	 ? "light lock"
					 : kind == DUNGEON_LOCK_PIN_TUMBLER ? "pin-tumbler padlock"
													 : "door");
		}
	}
	if (status[0])
	{
		/* The session's status line leads with the phase; keep it short. */
		ui_fill_rect(c, W / 2 - 330, H - 104, 660, 34, (UiColor){0, 0, 0, 130});
		ui_text(c, UI_FONT_SMALL, W / 2, H - 98, UI_ALIGN_CENTRE, GOLD, status);
	}
	if (line[0])
	{
		int w = ui_text_width(UI_FONT_BODY, line) + 40;
		ui_fill_rect(c, W / 2 - w / 2, H - 62, w, 42, (UiColor){0, 0, 0, 150});
		ui_text(c, UI_FONT_BODY, W / 2, H - 56, UI_ALIGN_CENTRE, WHITE, line);
	}
	/* Danger: red edges while the guardian hunts. */
	if (game->guard.active && game->guard.mode == GUARD_CHASE)
	{
		float pulse = 0.55f + 0.45f * sinf(game->ui_time * 9.0f);
		for (int i = 0; i < 24; ++i)
		{
			uint8_t a = (uint8_t)((24 - i) * 5 * pulse);
			UiColor edge = {200, 20, 15, a};
			ui_fill_rect(c, 0, i, W, 1, edge);
			ui_fill_rect(c, 0, H - 1 - i, W, 1, edge);
			ui_fill_rect(c, i, 0, 1, H, edge);
			ui_fill_rect(c, W - 1 - i, 0, 1, H, edge);
		}
	}
	if (game->banner_timer > 0.0f && game->banner[0])
	{
		float a = fminf(1.0f, game->banner_timer * 2.0f);
		UiColor col = game->over_timer > 0.0f || game->guard.alert_flash > 0.0f ? RED : GOLD;
		col.a = (uint8_t)(255 * a);
		ui_text(c, UI_FONT_HEADING, W / 2, 120, UI_ALIGN_CENTRE, col, game->banner);
	}
	if (game->over_timer > 0.0f)
		ui_fill_rect(c, 0, 0, W, H, (UiColor){120, 0, 0, (uint8_t)(160 * (1.2f - game->over_timer))});
}

static void draw_pause(DungeonGame *game, UiCanvas *c)
{
	dim_screen(c, 170);
	panel(c, W / 2 - 200, 90, 400, 360);
	ui_text(c, UI_FONT_HEADING, W / 2, 110, UI_ALIGN_CENTRE, GOLD, "Paused");
	if (game->in_overworld)
		menu(c, game, 190, OVERWORLD_PAUSE_ITEMS, 4);
	else
		menu(c, game, 190, DUNGEON_PAUSE_ITEMS, 5);
}

static void draw_game_over(DungeonGame *game, UiCanvas *c)
{
	ui_gradient_rect(c, 0, 0, W, H, (UiColor){70, 0, 0, 170}, (UiColor){10, 0, 0, 235});
	ui_text(c, UI_FONT_TITLE, W / 2, 110, UI_ALIGN_CENTRE, RED, "CAUGHT");
	ui_text(c, UI_FONT_BODY, W / 2, 205, UI_ALIGN_CENTRE, WHITE,
			"The guardian dragged you back into the dark.");
	menu(c, game, 290, GAME_OVER_ITEMS, 2);
	footer(c, "R to restart instantly");
}

static void draw_complete(DungeonGame *game, UiCanvas *c)
{
	dim_screen(c, 175);
	panel(c, W / 2 - 280, 50, 560, 440);
	ui_text(c, UI_FONT_HEADING, W / 2, 70, UI_ALIGN_CENTRE, GOLD, "Treasure Claimed!");
	ui_text(c, UI_FONT_SMALL, W / 2, 120, UI_ALIGN_CENTRE, DIM,
			game->levels[game->current].name);
	stars_row(c, W / 2, 182, 32.0f, game->stars, game->screen_time * 2.5f);
	char clock[16];
	format_time(game->run_time, clock, sizeof(clock));
	char line[96];
	int y = 240;
	snprintf(line, sizeof(line), "Treasure found   -   %s", clock);
	ui_text(c, UI_FONT_SMALL, W / 2, y, UI_ALIGN_CENTRE, GOLD, line);
	ui_text(c, UI_FONT_SMALL, W / 2, y + 26, UI_ALIGN_CENTRE, game->spotted ? DIM : GOLD,
			game->spotted ? "Spotted by the guardian" : "Never spotted");
	ui_text(c, UI_FONT_SMALL, W / 2, y + 52, UI_ALIGN_CENTRE, game->gem_taken ? GOLD : DIM,
			game->gem_taken ? "Hidden gem found" : "Hidden gem missed");
	menu(c, game, 350, COMPLETE_ITEMS, 2);
}

static void draw_dialogue(DungeonGame *game, UiCanvas *c)
{
	Dialogue *d = &game->monk_dialogue;
	panel(c, 35, 243, W - 70, 284);
	ui_text(c, UI_FONT_HEADING, 61, 250, UI_ALIGN_LEFT, GOLD, "The Praying Monk");
	char page[48];
	snprintf(page, sizeof(page), "%u / %u", d->page + 1, d->entries[d->entry].page_count);
	ui_text(c, UI_FONT_SMALL, W - 62, 265, UI_ALIGN_RIGHT, DIM, page);
	if (d->fragment[0])
	{
		ui_fill_rect(c, 45, 206, W - 90, 32, (UiColor){14, 11, 9, 220});
		char said[64];
		snprintf(said, sizeof(said), "Player: %s", d->fragment);
		ui_text(c, UI_FONT_SMALL, 61, 210, UI_ALIGN_LEFT, DIM, said);
	}
	ui_text_reveal(c, UI_FONT_SMALL, 62, 300, W - 124, 4, WHITE, dialogue_text(d),
				   (size_t)d->reveal);
	if (d->phase == DIALOGUE_CHOICES)
	{
		game->cursor = (int)d->cursor;
		menu(c, game, 392, DIALOGUE_ITEMS, 3);
	}
	else if (d->phase == DIALOGUE_EDIT)
	{
		ui_text(c, UI_FONT_SMALL, 62, 393, UI_ALIGN_LEFT, GOLD,
				"Your reply (up to 120 characters)");
		ui_fill_rect(c, 59, 424, W - 118, 38, (UiColor){3, 3, 4, 235});
		ui_frame_rect(c, 59, 424, W - 118, 38, 1, GOLD);
		char before[DIALOGUE_REPLY_CAPACITY];
		memcpy(before, d->reply, d->caret);
		before[d->caret] = 0;
		size_t start = 0;
		while (start < d->caret && ui_text_width(UI_FONT_BODY, before + start) > W - 150)
			++start;
		ui_text_reveal(c, UI_FONT_BODY, 68, 427, W - 145, 1, WHITE, d->reply + start, SIZE_MAX);
		if (fmodf(game->ui_time, 1.0f) < .6f)
			ui_fill_rect(c, 68 + ui_text_width(UI_FONT_BODY, before + start), 430, 2, 25, GOLD);
		ui_text(c, UI_FONT_SMALL, 62, 481, UI_ALIGN_LEFT, DIM, "Enter: speak    Esc: leave");
	}
	else if (d->phase == DIALOGUE_INTERRUPT)
	{
		ui_text(c, UI_FONT_SMALL, 62, 468, UI_ALIGN_LEFT, GOLD, "You try to get a word in...");
	}
	else
	{
		ui_text(c, UI_FONT_SMALL, 62, 481, UI_ALIGN_LEFT, DIM,
				"Enter / Space / Click: continue    Esc: leave");
	}
}

void dungeon_game_draw_ui(DungeonGame *game, const DungeonScene *scene, UiCanvas *c)
{
	ui_clear(c);
	switch (game->screen)
	{
	case DUNGEON_GAME_TITLE: draw_title(game, c); break;
	case DUNGEON_GAME_HOW_TO_PLAY: draw_how_to_play(game, c); break;
	case DUNGEON_GAME_OVERWORLD: draw_overworld_hud(game, c); break;
	case DUNGEON_GAME_LOADING: draw_loading(game, c); break;
	case DUNGEON_GAME_PLAYING: draw_hud(game, scene, c); break;
	case DUNGEON_GAME_DIALOGUE:
		draw_dialogue(game, c);
		break;
	case DUNGEON_GAME_PAUSED: draw_pause(game, c); break;
	case DUNGEON_GAME_OVER: draw_game_over(game, c); break;
	case DUNGEON_GAME_COMPLETE: draw_complete(game, c); break;
	}
}
