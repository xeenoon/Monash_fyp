#include "dungeon_scene.h"

#include "dungeon_cave.h"
#include "dungeon_collision.h"
#include "dungeon_grid.h"
#include "dungeon_lock_layout.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef DUNGEON_TEXTURE_DIR
#define DUNGEON_TEXTURE_DIR "textures/dungeon/runtime"
#endif

#ifndef DUNGEON_TORCH_PATH
#define DUNGEON_TORCH_PATH "assets/dungeons/torch/walltorch.gltf"
#endif

static const char *const ALBEDO_PATHS[DUNGEON_MESH_BATCH_COUNT] = {
	DUNGEON_TEXTURE_DIR "/floor_albedo.jpg",
	DUNGEON_TEXTURE_DIR "/wall_albedo.jpg",
	DUNGEON_TEXTURE_DIR "/exit_albedo.jpg",
	[DUNGEON_MESH_MOSS] = DUNGEON_TEXTURE_DIR "/moss.jpeg",
	/* The door leaf borrows the exit's medieval-wood set: it is the only wood
	 * material checked in, and it is exactly what a dungeon door wants. */
	[DUNGEON_MESH_DOOR] = DUNGEON_TEXTURE_DIR "/exit_albedo.jpg",
	/* Aged metal on the STRUCTURE only -- casing, housing, bores, dial. The
	 * READOUTS (pins, springs, shear marks) stay untextured on purpose:
	 * mesh.frag multiplies the albedo sample by draw.geometry.rgb, so a tint
	 * can only ever darken what the texture already has, and on a saturated
	 * rust sample a blue "selected" pin came out muddy orange because there is
	 * no blue in it to keep. Their colour is the information; the structure is
	 * what carries the material. */
	[DUNGEON_MESH_LOCK_BODY] = DUNGEON_TEXTURE_DIR "/lock_albedo.jpg",
	[DUNGEON_MESH_LOCK_HOUSING] = DUNGEON_TEXTURE_DIR "/lock_albedo.jpg",
	[DUNGEON_MESH_LOCK_CHANNEL] = DUNGEON_TEXTURE_DIR "/lock_albedo.jpg",
	[DUNGEON_MESH_LOCK_DIAL] = DUNGEON_TEXTURE_DIR "/lock_albedo.jpg",
	/* The pins, their springs and the pick all carry the same worn metal now
	 * that none of them encode state through colour. */
	[DUNGEON_MESH_LOCK_PIN] = DUNGEON_TEXTURE_DIR "/lock_albedo.jpg",
	[DUNGEON_MESH_LOCK_SPRING] = DUNGEON_TEXTURE_DIR "/lock_albedo.jpg",
	[DUNGEON_MESH_LOCK_PICK] = DUNGEON_TEXTURE_DIR "/lock_albedo.jpg",
};

static const char *const ORM_PATHS[DUNGEON_MESH_BATCH_COUNT] = {
	DUNGEON_TEXTURE_DIR "/floor_orm.png",
	DUNGEON_TEXTURE_DIR "/wall_orm.png",
	DUNGEON_TEXTURE_DIR "/exit_orm.png",
	[DUNGEON_MESH_DOOR] = DUNGEON_TEXTURE_DIR "/exit_orm.png",
	[DUNGEON_MESH_LOCK_BODY] = DUNGEON_TEXTURE_DIR "/lock_orm.png",
	[DUNGEON_MESH_LOCK_HOUSING] = DUNGEON_TEXTURE_DIR "/lock_orm.png",
	[DUNGEON_MESH_LOCK_CHANNEL] = DUNGEON_TEXTURE_DIR "/lock_orm.png",
	[DUNGEON_MESH_LOCK_DIAL] = DUNGEON_TEXTURE_DIR "/lock_orm.png",
	[DUNGEON_MESH_LOCK_PIN] = DUNGEON_TEXTURE_DIR "/lock_orm.png",
	[DUNGEON_MESH_LOCK_SPRING] = DUNGEON_TEXTURE_DIR "/lock_orm.png",
	[DUNGEON_MESH_LOCK_PICK] = DUNGEON_TEXTURE_DIR "/lock_orm.png",
};

static const char *const NORMAL_PATHS[DUNGEON_MESH_BATCH_COUNT] = {
	DUNGEON_TEXTURE_DIR "/floor_normal.png",
	DUNGEON_TEXTURE_DIR "/wall_normal.png",
	DUNGEON_TEXTURE_DIR "/exit_normal.png",
	[DUNGEON_MESH_DOOR] = DUNGEON_TEXTURE_DIR "/exit_normal.png",
	[DUNGEON_MESH_LOCK_BODY] = DUNGEON_TEXTURE_DIR "/lock_normal.png",
	[DUNGEON_MESH_LOCK_HOUSING] = DUNGEON_TEXTURE_DIR "/lock_normal.png",
	[DUNGEON_MESH_LOCK_CHANNEL] = DUNGEON_TEXTURE_DIR "/lock_normal.png",
	[DUNGEON_MESH_LOCK_DIAL] = DUNGEON_TEXTURE_DIR "/lock_normal.png",
	[DUNGEON_MESH_LOCK_PIN] = DUNGEON_TEXTURE_DIR "/lock_normal.png",
	[DUNGEON_MESH_LOCK_SPRING] = DUNGEON_TEXTURE_DIR "/lock_normal.png",
	[DUNGEON_MESH_LOCK_PICK] = DUNGEON_TEXTURE_DIR "/lock_normal.png",
};

static DrawPushConstants dungeon_push(const Mesh *mesh, WorldPosition camera_position)
{
	return (DrawPushConstants){
		.local_to_camera_relative =
			coordinate_local_to_camera_relative(&mesh->local_to_world, camera_position),
		.geometry = {{1.0f, 1.0f, 1.0f, 1.0f}},
		.elevation_uv = {{1.0f, 1.0f, 1.0f, 0.0f}},
		.material = {{1.0f, 1.0f, 1.0f, 1.0f}},
		.debug = {{0.0f, 0.0f, 1.0f, 1.0f}},
	};
}

static DrawPushConstants player_push(const Mesh *mesh, WorldPosition camera_position)
{
	DrawPushConstants push = dungeon_push(mesh, camera_position);
	push.geometry = (vec4s){{0.12f, 0.42f, 0.95f, 1.0f}};
	push.material = (vec4s){{0.0f, 1.0f, 1.0f, 1.0f}};
	push.debug.y = 1.0f; /* dynamic object: reject stale temporal history */
	push.debug.w = 0.0f;
	return push;
}

/* Untextured near-black water; real reflections are Phase 5 (fake, walls
 * only) -- for now this just keeps puddles visually distinct from the floor
 * they sit on. */
static DrawPushConstants puddle_push(const Mesh *mesh, WorldPosition camera_position)
{
	DrawPushConstants push = dungeon_push(mesh, camera_position);
	push.geometry = (vec4s){{0.02f, 0.03f, 0.045f, 1.0f}};
	push.material = (vec4s){{0.0f, 1.0f, 0.0f, 0.0f}};
	return push;
}

static DrawPushConstants torch_push(const LocalToWorldTransform *transform,
									const GltfMaterial *material, WorldPosition camera_position)
{
	return (DrawPushConstants){
		.local_to_camera_relative =
			coordinate_local_to_camera_relative(transform, camera_position),
		.geometry = {{material->base_color_factor[0], material->base_color_factor[1],
					  material->base_color_factor[2], material->base_color_factor[3]}},
		.elevation_uv = {{material->roughness_factor, material->normal_scale,
						  material->occlusion_strength, 0.0f}},
		.material = {{material->metallic_factor, 1.0f, 1.0f, 1.0f}},
		.debug = {{0.0f, 0.0f, 1.0f, 1.0f}},
	};
}

/* The source torch's handle starts at local Y=1.14289. Seat that
 * point on the cube top; its head and light share this transform. */
static LocalToWorldTransform player_torch_transform(const DungeonScene *scene)
{
	const double scale = 1.35;
	LocalToWorldTransform transform = coordinate_identity_transform((WorldPosition){
		scene->player.position.x, scene->level.floor_y + 0.73 - 1.14289 * scale,
		scene->player.position.z + 0.06251 * scale});
	for (int i = 0; i < 3; ++i) transform.rotation[i][i] = scale;
	return transform;
}

static DungeonPoint segment_closest_point(DungeonPoint point, DungeonSegment segment)
{
	float dx = segment.b.x - segment.a.x, dz = segment.b.z - segment.a.z;
	float length2 = dx * dx + dz * dz;
	float t = length2 > 1e-12f
				 ? ((point.x - segment.a.x) * dx + (point.z - segment.a.z) * dz) / length2
				 : 0.0f;
	t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
	return (DungeonPoint){segment.a.x + dx * t, segment.a.z + dz * t};
}

static bool mount_torch(const DungeonLevel *level, const DungeonShadow *shadow, DungeonLight *light,
						LocalToWorldTransform *transform)
{
	float best_distance2 = FLT_MAX;
	DungeonPoint mount = {0}, inward = {0};
	for (uint32_t i = 0; i < level->collider_count; ++i)
	{
		if (level->colliders[i].type != DUNGEON_COLLIDER_SEGMENT)
			continue;
		DungeonPoint point = segment_closest_point(light->position, level->colliders[i].segment);
		float dx = light->position.x - point.x, dz = light->position.z - point.z;
		float distance2 = dx * dx + dz * dz;
		if (distance2 > 1e-6f && distance2 < best_distance2)
		{
			float inverse_distance = 1.0f / sqrtf(distance2);
			best_distance2 = distance2;
			mount = point;
			inward = (DungeonPoint){dx * inverse_distance, dz * inverse_distance};
		}
	}
	if (best_distance2 == FLT_MAX)
		return false;
	/* Collision contours only choose a direction. Seat the flame against
	 * the rendered wall at flame height, including the wall's bulge. */
	float origin[3]={light->position.x,level->floor_y+1.58f,light->position.z};
	float direction[3]={-inward.x,0,-inward.z}, hit_normal[3];
	float distance=sqrtf(best_distance2)+2.0f;
	if (!dungeon_shadow_trace(shadow,origin,direction,&distance,hit_normal)) return false;
	mount=(DungeonPoint){origin[0]+direction[0]*distance,origin[2]+direction[2]*distance};
	/* Moving back along the verified clear ray keeps the head outside. */
	/* The source torch projects along local -Z. Keep its back plate just clear
	   of the wall and put the point light at the head of the mesh. */
	mount.x += inward.x * 0.01f;
	mount.z += inward.z * 0.01f;
	light->position =
		(DungeonPoint){mount.x + inward.x * 0.18f, mount.z + inward.z * 0.18f};
	light->height = level->floor_y + 1.58f;
	double yaw = atan2(-(double)inward.x, -(double)inward.z);
	*transform = coordinate_rotation_y(
		yaw, (WorldPosition){mount.x, level->floor_y + 0.05f, mount.z});
	return true;
}


/* Local +X maps to world (cos t, 0, -sin t) under coordinate_rotation_y, so
 * aiming local +X along a world XZ direction needs atan2(-dz, dx). Every door
 * and lock piece is placed through one of these two helpers; getting the sign
 * wrong here silently mirrors the whole assembly. */
static LocalToWorldTransform aim_x(DungeonPoint direction, WorldPosition translation)
{
	return coordinate_rotation_y(atan2((double)-direction.z, (double)direction.x), translation);
}

/* Slot layout matches torch_push, and getting it wrong is not subtle: roughness
 * is the ORM roughness FACTOR in elevation_uv.x, metallic is
 * material_factors.x, and material_factors.y is the default-lit flag. An
 * earlier version passed roughness as material_factors.y and left
 * elevation_uv.x at 1.0, which pinned every lock piece to fully-rough
 * metallic-zero -- flat plastic under any light. */
static DrawPushConstants tinted_push(const LocalToWorldTransform *transform, WorldPosition camera,
									 float r, float g, float b, float roughness, float metallic)
{
	return (DrawPushConstants){
		.local_to_camera_relative = coordinate_local_to_camera_relative(transform, camera),
		.geometry = {{r, g, b, 1.0f}},
		.elevation_uv = {{roughness, 1.0f, 1.0f, 0.0f}},
		.material = {{metallic, 1.0f, 1.0f, 1.0f}},
		.debug = {{0.0f, 1.0f, 1.0f, 1.0f}}, /* .y: dynamic, so TAA keeps no stale trail */
	};
}

/* Appends one instance draw of a unit mesh. The mesh's own local_to_world is
 * overwritten per instance, which is safe because every draw is recorded into
 * the command buffer before the next instance is written. */
static void push_instance(DungeonScene *scene, DungeonMeshBatchKind kind,
						  LocalToWorldTransform transform, WorldPosition camera, float r, float g,
						  float b, float roughness, float metallic, RendererDraw *out,
						  uint32_t *count, uint32_t capacity)
{
	if (!scene->uploaded[kind] || *count >= capacity)
		return;
	scene->meshes[kind].local_to_world = transform;
	out[(*count)++] =
		(RendererDraw){.mesh = &scene->meshes[kind],
					   .material_set = scene->meshes[kind].material_set,
					   .push = tinted_push(&transform, camera, r, g, b, roughness, metallic),
					   .static_mesh = true};
}

/* Named so the call sites read as materials rather than as pairs of floats. */
#define LOCK_BRASS_ROUGHNESS 0.30f
#define LOCK_STEEL_ROUGHNESS 0.24f
#define LOCK_IRON_ROUGHNESS 0.52f
#define LOCK_METAL 1.0f

/* Layout constants all come from dungeon_lock_layout.h, which is also what the
 * clearance tests reason about -- keeping a private copy here is how the drawn
 * pick and the tested pick would quietly diverge. */
#define LOCK_BODY_Y DUNGEON_LOCK_BODY_Y
#define LOCK_HOUSING_Y DUNGEON_LOCK_HOUSING_Y
#define LOCK_HOUSING_HEIGHT_M (DUNGEON_LOCK_HOUSING_TOP_Y - DUNGEON_LOCK_HOUSING_Y)
#define LOCK_CHANNEL_Y DUNGEON_LOCK_CHANNEL_Y
#define LOCK_PIN_BASE_Y DUNGEON_LOCK_PIN_BASE_Y
#define LOCK_PIN_RISE_M DUNGEON_LOCK_PIN_RISE_M
#define LOCK_PIN_LENGTH_M DUNGEON_LOCK_PIN_LENGTH_M
#define LOCK_BORE_PITCH_M DUNGEON_LOCK_BORE_PITCH_M
#define LOCK_PROUD_BACK_M DUNGEON_LOCK_PROUD_BACK_M
#define LOCK_PROUD_PIN_M DUNGEON_LOCK_PROUD_PIN_M
#define LOCK_PIN_SET_SINK_M DUNGEON_LOCK_PIN_SET_SINK_M
#define LOCK_SPRING_COILS DUNGEON_LOCK_SPRING_COIL_COUNT

/* The lock's frame on the door face: `across` runs along the door, `normal`
 * points out of it toward the player. */
static void lock_frame(const DungeonLevel *level, const DungeonSession *session, uint32_t index,
					   DungeonPoint *out_origin, DungeonPoint *out_across, DungeonPoint *out_normal)
{
	const DungeonDoorway *door = &level->doors[index];
	float side = session->doors[index].hardware_side;
	DungeonPoint normal = {cosf(door->yaw) * side, sinf(door->yaw) * side};
	*out_origin = dungeon_session_lock_origin(session, index);
	/* The focus camera looks along -normal, and for a forward of (cos y, sin y)
	 * this renderer's right vector is (-forward.z, forward.x) -- which works out
	 * to (normal.z, -normal.x). So the camera's RIGHT is the negated
	 * perpendicular, and laying the bores out along the un-negated one put pin 0
	 * on the right of screen: pressing "right" then walked the selection
	 * leftwards. */
	*out_across = (DungeonPoint){normal.z, -normal.x};
	*out_normal = normal;
}

/* A point on the lock face: `lateral` along the door, `proud` out of it. */
static WorldPosition lock_point(DungeonPoint origin, DungeonPoint across, DungeonPoint normal,
								float lateral, float proud, float height)
{
	return (WorldPosition){origin.x + across.x * lateral + normal.x * proud, height,
						   origin.z + across.z * lateral + normal.z * proud};
}

/* The casing, shared by both lock kinds -- it is what makes a door read as
 * locked from across the room, before any of the mechanism is legible. */
static void append_lock_body(DungeonScene *scene, DungeonPoint origin, DungeonPoint across,
							 DungeonPoint normal, float tint, WorldPosition camera,
							 RendererDraw *out, uint32_t *count, uint32_t capacity)
{
	push_instance(scene, DUNGEON_MESH_LOCK_BODY,
				  aim_x(across, lock_point(origin, across, normal, 0.0f, 0.0f, LOCK_BODY_Y)),
				  camera, 0.62f * tint, 0.58f * tint, 0.54f * tint, LOCK_BRASS_ROUGHNESS, LOCK_METAL,
				  out, count, capacity);
}

static void append_pin_tumbler_draws(DungeonScene *scene, uint32_t index, bool focused,
									 WorldPosition camera, RendererDraw *out, uint32_t *count,
									 uint32_t capacity)
{
	const DungeonSession *session = &scene->session;
	const DungeonPinTumbler *pins = &session->doors[index].pins;
	DungeonPoint origin, across, normal;
	lock_frame(&scene->level, session, index, &origin, &across, &normal);
	float tint = focused ? 1.0f : 0.5f;

	append_lock_body(scene, origin, across, normal, tint, camera, out, count, capacity);
	push_instance(scene, DUNGEON_MESH_LOCK_HOUSING,
				  aim_x(across, lock_point(origin, across, normal, 0.0f, 0.0f, LOCK_HOUSING_Y)),
				  camera, 0.34f * tint, 0.33f * tint, 0.32f * tint, 0.66f, LOCK_METAL, out, count,
				  capacity);

	float span = (float)(pins->pin_count - 1u) * LOCK_BORE_PITCH_M * 0.5f;
	for (uint32_t pin = 0; pin < pins->pin_count; ++pin)
	{
		float lateral = (float)pin * LOCK_BORE_PITCH_M - span;
		bool seated = pins->heights[pin] == pins->target[pin];

		push_instance(scene, DUNGEON_MESH_LOCK_CHANNEL,
					  aim_x(across, lock_point(origin, across, normal, lateral, LOCK_PROUD_BACK_M,
											   LOCK_CHANNEL_Y)),
					  camera, 0.10f * tint, 0.10f * tint, 0.105f * tint, 0.85f, LOCK_METAL, out,
					  count, capacity);

		/* Every pin is the same worn metal. Which one the keys are driving is
		 * shown by the pick resting against it, and whether it has set is shown
		 * by it sinking into its bore -- neither needs a colour, and the
		 * coloured blocks and amber target marks that used to carry both read
		 * as a puzzle diagram bolted onto a door rather than as a lock. */
		float pin_y = LOCK_PIN_BASE_Y + (float)pins->heights[pin] * LOCK_PIN_RISE_M;
		float pin_proud = LOCK_PROUD_PIN_M - (seated ? LOCK_PIN_SET_SINK_M : 0.0f);
		push_instance(scene, DUNGEON_MESH_LOCK_PIN,
					  aim_x(across, lock_point(origin, across, normal, lateral, pin_proud, pin_y)),
					  camera, 0.80f * tint, 0.79f * tint, 0.76f * tint, 0.34f, LOCK_METAL, out,
					  count, capacity);

		/* The spring above the pin, drawn as a stack of coils spread over
		 * whatever headroom is left: raising a pin bunches them up, which is
		 * the motion that makes a pin tumbler read as a mechanism rather than
		 * as a slider. Only for the lock being picked -- at three doors' worth
		 * it is a lot of draws for something nobody is looking at. */
		if (!focused)
			continue;
		float coil_low = pin_y + LOCK_PIN_LENGTH_M + 0.010f;
		float coil_high = LOCK_HOUSING_Y + LOCK_HOUSING_HEIGHT_M - 0.030f;
		float pitch = (coil_high - coil_low) / (float)LOCK_SPRING_COILS;
		for (uint32_t coil = 0; coil < LOCK_SPRING_COILS; ++coil)
			push_instance(scene, DUNGEON_MESH_LOCK_SPRING,
						  aim_x(across, lock_point(origin, across, normal, lateral,
												   LOCK_PROUD_PIN_M, coil_low +
																		 (float)coil * pitch)),
						  camera, 0.74f * tint, 0.76f * tint, 0.78f * tint, 0.30f, LOCK_METAL, out,
						  count, capacity);
	}

	/* The pick, slid so its tip stub rests under whichever pin the keys are
	 * driving. It stays upright and rides in front of the whole mechanism --
	 * an earlier version was posed by rolling it about the door normal to
	 * point at the pin, which swung the shaft diagonally through the housing
	 * on the way between bores. dungeon_lock_layout_pick_clear is what pins
	 * that down now. */
	if (focused && scene->pick_active)
		push_instance(scene, DUNGEON_MESH_LOCK_PICK,
					  aim_x(across, lock_point(origin, across, normal, scene->pick_lateral,
											   DUNGEON_LOCK_PROUD_PICK_M, scene->pick_height)),
					  camera, 0.86f * tint, 0.84f * tint, 0.78f * tint, 0.30f, LOCK_METAL, out,
					  count, capacity);
}

bool dungeon_scene_pin_world(const DungeonScene *scene, uint32_t door_index, uint32_t pin,
							 WorldPosition *out)
{
	if (!scene || !out || door_index >= scene->session.door_count)
		return false;
	const DungeonPinTumbler *pins = &scene->session.doors[door_index].pins;
	if (pin >= pins->pin_count)
		return false;
	DungeonPoint origin, across, normal;
	lock_frame(&scene->level, &scene->session, door_index, &origin, &across, &normal);
	*out = lock_point(origin, across, normal, dungeon_lock_layout_bore_lateral(pins, pin),
					  LOCK_PROUD_PIN_M,
					  LOCK_PIN_BASE_Y + (float)pins->heights[pin] * LOCK_PIN_RISE_M);
	return true;
}

static void append_vault_dial_draws(DungeonScene *scene, uint32_t index, bool focused,
									WorldPosition camera, RendererDraw *out, uint32_t *count,
									uint32_t capacity)
{
	const DungeonSession *session = &scene->session;
	const DungeonDoorState *state = &session->doors[index];
	DungeonPoint origin, across, normal;
	lock_frame(&scene->level, session, index, &origin, &across, &normal);
	float tint = focused ? 1.0f : 0.5f;

	append_lock_body(scene, origin, across, normal, tint, camera, out, count, capacity);
	push_instance(scene, DUNGEON_MESH_LOCK_DIAL,
				  aim_x(across, lock_point(origin, across, normal, 0.0f, 0.0f, LOCK_HOUSING_Y)),
				  camera, 0.13f * tint, 0.125f * tint, 0.12f * tint, 0.66f, LOCK_METAL, out, count,
				  capacity);

	/* Four marks around the dial, the last one entered lit. The dial has no
	 * absolute position of its own, so this is what makes a turn visible. */
	float centre_y = LOCK_HOUSING_Y + 0.175f;
	const float mark_lateral[4] = {-0.115f, 0.115f, 0.0f, 0.0f};
	const float mark_height[4] = {centre_y, centre_y, centre_y + 0.115f, centre_y - 0.115f};
	for (uint32_t direction = 0; direction < 4u; ++direction)
	{
		bool lit = state->has_last_input && (uint32_t)state->last_input == direction;
		push_instance(scene, DUNGEON_MESH_LOCK_NOTCH,
					  aim_x(across, lock_point(origin, across, normal, mark_lateral[direction],
											   0.042f, mark_height[direction])),
					  camera, (lit ? 1.00f : 0.30f) * tint, (lit ? 0.72f : 0.31f) * tint,
					  (lit ? 0.10f : 0.34f) * tint, 0.42f, 0.0f, out, count, capacity);
	}

	/* Progress pips below the dial: how many steps are banked, never which.
	 * Spaced wider than a pip is, or they merge into one bar and stop counting
	 * anything. `across` is the camera's right, so a plain ascending offset
	 * fills them left-to-right on screen. */
	float pip_span = (float)(state->dial.step_count - 1u) * 0.145f * 0.5f;
	for (uint32_t step = 0; step < state->dial.step_count; ++step)
	{
		bool banked = step < state->dial.progress;
		push_instance(scene, DUNGEON_MESH_LOCK_NOTCH,
					  aim_x(across, lock_point(origin, across, normal,
											   (float)step * 0.145f - pip_span, LOCK_PROUD_BACK_M,
											   0.45f)),
					  camera, (banked ? 1.00f : 0.15f) * tint, (banked ? 0.74f : 0.15f) * tint,
					  (banked ? 0.12f : 0.17f) * tint, 0.42f, 0.0f, out, count, capacity);
	}
}

/* Doors and their lock hardware. A door leaf sinks into the floor as it opens
 * rather than swinging: it is exactly as wide as the hallway, so a swung leaf
 * would pass through the corridor wall. Its collider is already gone by then
 * -- the session drops it the instant the lock turns -- so the player never
 * waits on the animation. */
static void append_door_draws(DungeonScene *scene, WorldPosition camera, RendererDraw *out,
							  uint32_t *count, uint32_t capacity)
{
	const DungeonSession *session = &scene->session;
	for (uint32_t i = 0; i < session->door_count; ++i)
	{
		const DungeonDoorway *door = &scene->level.doors[i];
		const DungeonDoorState *state = &session->doors[i];
		if (state->swing > 0.999f)
			continue; /* fully sunk; nothing left above the floor to draw */
		DungeonPoint along = {door->blocker.b.x - door->blocker.a.x,
							  door->blocker.b.z - door->blocker.a.z};
		float drop = state->swing * (scene->level.wall_height + 0.12f);
		LocalToWorldTransform transform =
			aim_x(along, (WorldPosition){door->blocker.a.x, scene->level.floor_y - drop,
										 door->blocker.a.z});
		/* Textured with the checked-in medieval-wood set, so a locked door reads
		 * as a door rather than as a flat coloured slab. */
		push_instance(scene, DUNGEON_MESH_DOOR, transform, camera, 0.90f, 0.86f, 0.82f, 1.0f, 1.0f,
					  out, count, capacity);
		if (state->open)
			continue;
		bool focused = session->phase != DUNGEON_PHASE_EXPLORING && session->focused_door == i;
		if (door->lock == DUNGEON_LOCK_VAULT_DIAL)
			append_vault_dial_draws(scene, i, focused, camera, out, count, capacity);
		else
			append_pin_tumbler_draws(scene, i, focused, camera, out, count, capacity);
	}
}

bool dungeon_scene_create(Renderer *renderer, DungeonScene *out, DungeonLevelError *error)
{
	if (!renderer || !out)
		return false;
	*out = (DungeonScene){0};
	const char *map_override = getenv("DUNGEON_MAP");
	bool compiled;
	if (map_override)
		compiled = dungeon_grid_compile_file(map_override, 2.0f, &out->level, error);
	else
	{
		uint32_t seed = 1u;
		const char *seed_env = getenv("DUNGEON_SEED");
		if (seed_env)
			seed = (uint32_t)strtoul(seed_env, NULL, 10);
		DungeonCaveParams params = dungeon_cave_default_params(seed);
		compiled = dungeon_cave_compile(&params, &out->level, error);
	}
	if (!compiled || !dungeon_mesh_build(&out->level, &out->geometry, error))
	{
		dungeon_scene_destroy(renderer, out);
		return false;
	}
	texture_load(renderer->device, renderer->allocator, renderer->upload, &out->moss_albedo,
		DUNGEON_TEXTURE_DIR "/moss.jpeg", renderer->max_anisotropy);
	for (uint32_t i = 0; i < DUNGEON_MESH_BATCH_COUNT; ++i)
	{
		DungeonMeshBatch *batch = &out->geometry.batches[i];
		if (!batch->vertex_count)
			continue; /* e.g. no puddles were placed; leave uploaded[i] false */
		out->meshes[i] = (Mesh){
			.local_to_world = coordinate_identity_transform((WorldPosition){0}),
			.vertices = batch->vertices,
			.vertex_count = batch->vertex_count,
			.indices = batch->indices,
			.index_count = batch->index_count,
			.texture_path = ALBEDO_PATHS[i],
			.orm_path = ORM_PATHS[i],
			.normal_path = NORMAL_PATHS[i],
		};
		mesh_upload(renderer, &out->meshes[i]);
		out->uploaded[i] = true;
		if (i == DUNGEON_MESH_FLOOR || i == DUNGEON_MESH_WALL)
		{
			/* The surface permutation uses binding 3 for the user's moss JPEG. */
			VkDescriptorImageInfo image = {.sampler = out->moss_albedo.sampler,
				.imageView = out->moss_albedo.view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
			VkWriteDescriptorSet write = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = out->meshes[i].material_set, .dstBinding = 3,
				.descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &image};
			vkUpdateDescriptorSets(renderer->device, 1, &write, 0, NULL);
		}
	}
	fprintf(stdout, "Moss: %u tufts, %u triangles\n",
		out->geometry.batches[DUNGEON_MESH_MOSS].vertex_count / 180u,
		out->geometry.batches[DUNGEON_MESH_MOSS].index_count / 3u);
	dungeon_player_init(&out->player, out->level.spawn);
	if (!dungeon_session_create(&out->session, &out->level))
	{
		if (error)
			snprintf(error->message, sizeof(error->message), "out of memory building lock session");
		dungeon_scene_destroy(renderer, out);
		return false;
	}
	/* Two slots are held back from the torches: slot zero is the light the
	 * player carries, and one more is the light over the lock being picked.
	 * With the camera zoomed onto a door the wall torches behind the player
	 * contribute almost nothing, and there is no HUD to read pin heights off
	 * if they fall dark. */
	if (!dungeon_scene_prepare_shadows(out, renderer)) {
		if (error) snprintf(error->message,sizeof(error->message),"could not build wall shadows");
		dungeon_scene_destroy(renderer,out); return false;
	}
	out->light_count = dungeon_lighting_build(&out->level, out->lights, DUNGEON_MAX_LIGHTS - 2u);
	GltfLoadError torch_error = {0};
	if (gltf_scene_create(renderer, DUNGEON_TORCH_PATH,
					  &(GltfLoadOptions){.placement =
									 coordinate_identity_transform((WorldPosition){0})},
					  &out->torch, &torch_error) != GLTF_LOAD_OK)
	{
		if (error)
			snprintf(error->message, sizeof(error->message), "could not load dungeon torch: %.220s",
					 torch_error.message);
		dungeon_scene_destroy(renderer, out);
		return false;
	}
	for (uint32_t i = 0; i < out->light_count; ++i)
		if (mount_torch(&out->level, &out->wall_shadow, &out->lights[i],
						&out->torch_transforms[out->torch_count]))
			++out->torch_count;
		else {
			/* A fixture without a verified wall mount must not leave an
			 * invisible point light at its original candidate position. */
			out->lights[i].intensity = 0.0f;
			fprintf(stderr, "Skipped torch %u: no visible wall mount\n", i);
		}
	uint32_t uploaded_batches = 0;
	for (uint32_t i = 0; i < DUNGEON_MESH_BATCH_COUNT; ++i)
		uploaded_batches += out->uploaded[i] ? 1u : 0u;
	fprintf(stdout,
		   "Dungeon: %s: %u collider segments, %u puddles, %u locked doors, %u draw batches\n",
		   map_override ? map_override : "procedural rooms and hallways",
		   out->level.collider_count, out->level.puddle_count, out->level.door_count,
		   uploaded_batches + out->torch_count * out->torch.primitive_count);
	return true;
}

uint32_t dungeon_scene_draws(DungeonScene *scene, WorldPosition camera_position,
							 RendererDraw *out, uint32_t capacity, uint32_t *out_shadow_draw_count)
{
	if (!scene || !out)
		return 0;
	scene->meshes[DUNGEON_MESH_PLAYER].local_to_world.translation =
		(WorldPosition){scene->player.position.x, scene->level.floor_y + 0.03f,
						 scene->player.position.z};
	uint32_t draw_count = 0;
	for (uint32_t i = 0; i < DUNGEON_MESH_BATCH_COUNT && draw_count < capacity; ++i)
	{
		if (i == DUNGEON_MESH_PUDDLE || !scene->uploaded[i])
			continue; /* puddles are appended last, below, and excluded from shadows */
		if (i >= DUNGEON_MESH_DOOR)
			continue; /* unit meshes: drawn per instance by append_door_draws */
		out[draw_count++] = (RendererDraw){.mesh = &scene->meshes[i],
											.material_set = scene->meshes[i].material_set,
											.push = i == DUNGEON_MESH_PLAYER
												? player_push(&scene->meshes[i], camera_position)
												: dungeon_push(&scene->meshes[i], camera_position),
											.static_mesh = true,
											.pipeline = (i == DUNGEON_MESH_WALL || i == DUNGEON_MESH_FLOOR)
												? RENDERER_PIPELINE_DUNGEON_SURFACE
												: i == DUNGEON_MESH_MOSS ? RENDERER_PIPELINE_DUNGEON_MOSS
												: RENDERER_PIPELINE_AUTO};
		/* Opaque surfaces use alpha to carry floor height in mesh coordinates. */
		out[draw_count - 1].push.geometry.w = scene->level.floor_y;
	}
	/* Reserve the carried torch before optional wall fixtures. */
	LocalToWorldTransform carried = player_torch_transform(scene);
	for (uint32_t i = 0; i < scene->torch.primitive_count && draw_count < capacity; ++i)
	{
		GltfPrimitive *primitive = &scene->torch.primitives[i];
		GltfMaterial *material = &scene->torch.materials[primitive->material_index];
		DrawPushConstants push = torch_push(&carried, material, camera_position);
		push.debug.y = 1.0f; /* carried object: reject stale temporal history */
		out[draw_count++] = (RendererDraw){.mesh = &primitive->mesh,
			.material_set = material->descriptor_set, .push = push, .static_mesh = true};
	}
	for (uint32_t instance = 0; instance < scene->torch_count && draw_count < capacity; ++instance)
		for (uint32_t primitive_index = 0; primitive_index < scene->torch.primitive_count;
			 ++primitive_index)
		{
			if (draw_count >= capacity)
				break;
			GltfPrimitive *primitive = &scene->torch.primitives[primitive_index];
			GltfMaterial *material = &scene->torch.materials[primitive->material_index];
			out[draw_count++] = (RendererDraw){
				.mesh = &primitive->mesh,
				.material_set = material->descriptor_set,
				.push = torch_push(&scene->torch_transforms[instance], material, camera_position),
				.static_mesh = true,
			};
		}
	append_door_draws(scene, camera_position, out, &draw_count, capacity);
	if (out_shadow_draw_count)
		*out_shadow_draw_count = draw_count; /* a flat coplanar disc casts nothing worth shadowing */
	if (scene->uploaded[DUNGEON_MESH_PUDDLE] && draw_count < capacity)
		out[draw_count++] = (RendererDraw){
			.mesh = &scene->meshes[DUNGEON_MESH_PUDDLE],
			.material_set = scene->meshes[DUNGEON_MESH_PUDDLE].material_set,
			.push = puddle_push(&scene->meshes[DUNGEON_MESH_PUDDLE], camera_position),
			.static_mesh = true,
			.pipeline = RENDERER_PIPELINE_DUNGEON_PUDDLE};
	return draw_count;
}

/* Eases the pick toward the selected pin. Snaps into place the first frame a
 * lock is entered -- sweeping in from wherever the last lock left it would have
 * the pick fly across the door out of nowhere. */
static void update_pick(DungeonScene *scene, float dt)
{
	const DungeonSession *session = &scene->session;
	if (session->phase != DUNGEON_PHASE_PIN_TUMBLER)
	{
		scene->pick_active = false;
		return;
	}
	const DungeonPinTumbler *pins = &session->doors[session->focused_door].pins;
	float target_lateral = 0.0f, target_height = 0.0f;
	dungeon_lock_layout_pick_target(pins, pins->selected, &target_lateral, &target_height);
	if (!scene->pick_active)
	{
		scene->pick_active = true;
		scene->pick_lateral = target_lateral;
		scene->pick_height = target_height;
		return;
	}
	float blend = 1.0f - expf(-14.0f * dt);
	scene->pick_lateral += (target_lateral - scene->pick_lateral) * blend;
	scene->pick_height += (target_height - scene->pick_height) * blend;
}

bool dungeon_scene_update(DungeonScene *scene, float move_forward, float move_right,
						  float camera_yaw_degrees, float dt)
{
	if (!scene)
		return false;
	dungeon_session_update(&scene->session, dt);
	update_pick(scene, dt);
	if (scene->session.phase != DUNGEON_PHASE_EXPLORING)
	{
		/* Picking: ease into the stance beside the lock instead of taking
		 * movement input. Routed through the same swept-circle solver as
		 * walking, so the stance can never push the player into rock even if a
		 * doorway sits tight against a corner. */
		DungeonPoint stance = dungeon_session_pick_stance(&scene->session);
		DungeonPoint position = scene->player.position;
		float blend = 1.0f - expf(-8.0f * dt);
		DungeonPoint step = {(stance.x - position.x) * blend, (stance.z - position.z) * blend};
		scene->player.position =
			dungeon_collision_move(scene->session.colliders, scene->session.collider_count,
								   position, step, scene->player.radius);
		return false;
	}
	/* Collide against the session's array, not the level's: it carries a
	 * segment for every door still shut, and loses it the moment one opens. */
	return dungeon_player_update(&scene->player, &scene->level, scene->session.colliders,
								 scene->session.collider_count, move_forward, move_right,
								 camera_yaw_degrees, dt);
}

uint32_t dungeon_scene_write_lights(const DungeonScene *scene, WorldPosition camera_position,
								   vec4s *positions, vec4s *colors, uint32_t capacity)
{
	if (!scene || !positions || !colors || capacity == 0)
		return 0;
	/* Slot zero is always the player light, even if the light budget is full. */
	LocalToWorldTransform carried = player_torch_transform(scene);
	WorldPosition head = coordinate_local_to_world(&carried, (TileLocalPosition){0, 1.56f, -0.13f});
	CameraRelativePosition relative = coordinate_camera_relative(head, camera_position);
	positions[0] = (vec4s){{relative.x, relative.y, relative.z, 7.5f}};
	colors[0] = (vec4s){{1.0f, 0.76f, 0.48f, 32.0f}};
	uint32_t count = scene->light_count < capacity - 1u ? scene->light_count : capacity - 1u;
	for (uint32_t i = 0; i < count; ++i)
	{
		WorldPosition world = {scene->lights[i].position.x, scene->lights[i].height,
							   scene->lights[i].position.z};
		CameraRelativePosition relative = coordinate_camera_relative(world, camera_position);
		positions[i + 1u] = (vec4s){{relative.x, relative.y, relative.z, scene->lights[i].radius}};
		colors[i + 1u] = (vec4s){{scene->lights[i].color[0], scene->lights[i].color[1],
							 scene->lights[i].color[2], scene->lights[i].intensity}};
	}
	uint32_t written = count + 1u;
	/* The other reserved slot: a small warm light over the lock being picked,
	 * so the pin bars are lit by something the player is not standing behind. */
	if (scene->session.phase != DUNGEON_PHASE_EXPLORING && written < capacity &&
		scene->session.focused_door < scene->session.door_count)
	{
		DungeonPoint focus = dungeon_session_focus_point(&scene->session);
		WorldPosition world = {focus.x, scene->level.floor_y + 1.15f, focus.z};
		CameraRelativePosition to_lock = coordinate_camera_relative(world, camera_position);
		/* Kept deliberately weak: the focus camera sits about a metre from the
		 * door, so anything brighter blows the leaf out to flat white and takes
		 * the lock's own shading with it. */
		positions[written] = (vec4s){{to_lock.x, to_lock.y, to_lock.z, 2.4f}};
		colors[written] = (vec4s){{1.0f, 0.89f, 0.72f, 2.0f}};
		++written;
	}
	return written;
}

/* Static walls are built once. Closed doors are added on state changes, never
 * selected by camera distance. Open doors stop blocking with their collider. */
bool dungeon_scene_prepare_shadows(DungeonScene *scene, Renderer *renderer)
{
    uint32_t mask=0;
    for(uint32_t i=0;i<scene->session.door_count;i++)
        if(!scene->session.doors[i].open) mask|=1u<<i;
    if(scene->shadow_uploaded && mask==scene->shadow_door_mask) return true;
    const DungeonMeshBatch *wall=&scene->geometry.batches[DUNGEON_MESH_WALL];
    const DungeonMeshBatch *door=&scene->geometry.batches[DUNGEON_MESH_DOOR];
    uint32_t capacity=wall->index_count/3+scene->session.door_count*(door->index_count/3);
    DungeonShadowTriangle *triangles=malloc((size_t)capacity*sizeof(*triangles));
    if(!triangles) return false;
    uint32_t count=0;
    for(uint32_t i=0;i<wall->index_count;i+=3) {
        DungeonShadowTriangle *t=&triangles[count++];
        memcpy(t->a,wall->vertices[wall->indices[i]].position,12);
        memcpy(t->b,wall->vertices[wall->indices[i+1]].position,12);
        memcpy(t->c,wall->vertices[wall->indices[i+2]].position,12);
    }
    if(!scene->wall_shadow.count && !dungeon_shadow_build(triangles,count,&scene->wall_shadow)) {
        free(triangles); return false;
    }
    for(uint32_t j=0;j<scene->session.door_count;j++) if(mask&(1u<<j)) {
        DungeonSegment s=scene->level.doors[j].blocker;
        LocalToWorldTransform transform=aim_x((DungeonPoint){s.b.x-s.a.x,s.b.z-s.a.z},
            (WorldPosition){s.a.x,scene->level.floor_y,s.a.z});
        for(uint32_t i=0;i<door->index_count;i+=3) {
            DungeonShadowTriangle *t=&triangles[count++];
            float *dest[3]={t->a,t->b,t->c};
            for(int k=0;k<3;k++) {
                const float *p=door->vertices[door->indices[i+k]].position;
                WorldPosition w=coordinate_local_to_world(&transform,(TileLocalPosition){p[0],p[1],p[2]});
                dest[k][0]=(float)w.x; dest[k][1]=(float)w.y; dest[k][2]=(float)w.z;
            }
        }
    }
    DungeonShadow next={0};
    bool ok=dungeon_shadow_build(triangles,count,&next); free(triangles);
    if(!ok) return false;
    renderer_upload_point_shadows(renderer,next.nodes,(size_t)next.count*sizeof(*next.nodes));
    dungeon_shadow_destroy(&scene->shadow); scene->shadow=next;
    scene->shadow_door_mask=mask; scene->shadow_uploaded=true;
    return true;
}

void dungeon_scene_destroy(Renderer *renderer, DungeonScene *scene)
{
	if (!scene)
		return;
	if (renderer)
	{
		gltf_scene_destroy(renderer, &scene->torch);
		for (uint32_t i = 0; i < DUNGEON_MESH_BATCH_COUNT; ++i)
			if (scene->uploaded[i])
				mesh_destroy(renderer, &scene->meshes[i]);
		texture_destroy(renderer->device, renderer->allocator, &scene->moss_albedo);
	}
	else
		gltf_scene_destroy(NULL, &scene->torch);
	dungeon_session_destroy(&scene->session);
	dungeon_shadow_destroy(&scene->wall_shadow);
	dungeon_shadow_destroy(&scene->shadow);
	dungeon_mesh_destroy(&scene->geometry);
	dungeon_level_destroy(&scene->level);
	*scene = (DungeonScene){0};
}
