#pragma once

#include "gltf_scene.h"
#include "renderer.h"

/* The imported spline sculpture is shared by all poses. Crown pieces rotate
 * about their buried roots, retaining their continuous swept surfaces. */
#define DUNGEON_GUARDIAN_MAX_DRAWS 112u

typedef struct
{
	GltfScene model;
	GltfTransform *pose;
	mat4s *world;
	uint32_t clip;
	bool loaded, active, hunting;
	WorldPosition position;
	float yaw, time;
} DungeonGuardianArt;

bool dungeon_guardian_load(Renderer *renderer, DungeonGuardianArt *art);
uint32_t dungeon_guardian_draws(DungeonGuardianArt *art, WorldPosition camera,
							   RendererDraw *out, uint32_t capacity);
void dungeon_guardian_destroy(Renderer *renderer, DungeonGuardianArt *art);
