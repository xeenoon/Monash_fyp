/* Does moving the selection actually MOVE the lockpick, and by enough to see?
 *
 * Against the real exported asset, on the CPU: gltf_scene_parse needs no
 * device, and "the pick tells you which pin you are on" is a question about
 * millimetres, not pixels. Skips when the asset has not been baked -- the same
 * condition under which the game falls back to the generated mechanism.
 *
 * The renderer symbols the importer would need for an upload are stubbed: no
 * upload happens on this path. */
#include "dungeon_padlock_pose.h"
#include "renderer.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void texture_load(VkDevice d, GpuAllocator *a, UploadContext *u, Texture *t, const char *p, float f)
{ (void)d; (void)a; (void)u; (void)t; (void)p; (void)f; assert(!"upload from a CPU-only test"); }
void texture_load_linear(VkDevice d, GpuAllocator *a, UploadContext *u, Texture *t, const char *p, float f)
{ (void)d; (void)a; (void)u; (void)t; (void)p; (void)f; assert(!"upload from a CPU-only test"); }
void texture_destroy(VkDevice d, GpuAllocator *a, Texture *t) { (void)d; (void)a; (void)t; }
VkDescriptorSet renderer_allocate_pbr5_set(Renderer *r, const Texture *a, const Texture *b,
										   const Texture *c, const Texture *d, const Texture *e)
{ (void)r; (void)a; (void)b; (void)c; (void)d; (void)e; return VK_NULL_HANDLE; }
void renderer_free_material_set(Renderer *r, VkDescriptorSet s) { (void)r; (void)s; }
void mesh_upload(Renderer *r, Mesh *m) { (void)r; (void)m; }
void mesh_destroy(Renderer *r, Mesh *m) { (void)r; (void)m; }

/* A door running along world Z with its hardware facing +X, which is the frame
 * lock_frame builds: `across` is the camera's right, (normal.z, -normal.x). */
static const DungeonPoint ORIGIN = {3.0f, -2.0f};
static const DungeonPoint NORMAL = {1.0f, 0.0f};
static const DungeonPoint ACROSS = {0.0f, -1.0f};

static GltfScene scene;
static DungeonPadlockModel model;
static GltfTransform *pose;
static mat4s *world;

static WorldPosition node_world(uint32_t node)
{
	LocalToWorldTransform placement =
		dungeon_padlock_model_placement(&model, ORIGIN, ACROSS, NORMAL);
	return coordinate_compose(&placement, world[node]).translation;
}

static double distance(WorldPosition a, WorldPosition b)
{
	double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
	return sqrt(dx * dx + dy * dy + dz * dz);
}

/* How far along the camera's right a point lands: the screen-x the player
 * reads the row off. */
static double across_of(WorldPosition p)
{
	return p.x * (double)ACROSS.x + p.z * (double)ACROSS.z;
}

static DungeonPinTumbler working_lock(uint32_t selected)
{
	DungeonPinTumbler pins = {.pin_count = DUNGEON_PADLOCK_PINS, .selected = selected};
	for (uint32_t i = 0; i < DUNGEON_PADLOCK_PINS; ++i)
		pins.target[i] = 1u + i % (DUNGEON_LOCK_PIN_STATES - 1u);
	return pins;
}

/* Eases the drawn heights until they have caught up with the puzzle. */
static void settle(DungeonPadlockAnim *anim, const DungeonPinTumbler *pins)
{
	for (int i = 0; i < 240; ++i)
		dungeon_padlock_show_heights(anim, pins->heights, pins->pin_count, 1.0f / 60.0f);
}

/* A lock mid-pick: the insert has run, the jiggle is looping. */
static DungeonPadlockAnim picking(void)
{
	DungeonPadlockAnim anim;
	dungeon_padlock_reset(&anim);
	dungeon_padlock_engage(&anim, true);
	for (int i = 0; i < 600; ++i)
		dungeon_padlock_update(&anim, model.duration, 1.0f / 60.0f);
	assert(anim.layers[DUNGEON_PADLOCK_CLIP_JIGGLE].playing);
	return anim;
}

static void the_pick_moves_to_the_selected_pin(void)
{
	DungeonPadlockAnim anim = picking();
	WorldPosition at[DUNGEON_PADLOCK_PINS];
	double offsets[DUNGEON_PADLOCK_PINS];
	for (uint32_t selected = 0; selected < DUNGEON_PADLOCK_PINS; ++selected)
	{
		DungeonPinTumbler pins = working_lock(selected);
		/* The jiggle is a LOOP, so hold its clock still across the sweep: what
		 * is being measured is the selection's effect, not the loop's. */
		settle(&anim, &pins);
		dungeon_padlock_model_pose(&model, &anim, &pins, true, pose, world);
		at[selected] = node_world(model.pick_node);
		float resolved = 0.0f;
		assert(dungeon_padlock_model_pick_offset(&model, &anim, &pins, pose, world, &resolved));
		offsets[selected] = (double)resolved;
		/* The offset the engine applied resolves back to the pin the keys are
		 * on. An inverted axis, a wrong scale, or a step taken in the wrong
		 * frame all land here. */
		assert(fabs(offsets[selected] - (double)selected) < 0.05);
	}

	/* The readout has to be VISIBLE, not merely present. One pin of selection
	 * is one pin of keyway travel, which at the drawn scale is over a
	 * centimetre -- on a 0.38 m lock, a movement of a few per cent of its
	 * height. Anything under 5 mm would be a readout only a debugger can see. */
	double smallest = 1e9, total = distance(at[0], at[DUNGEON_PADLOCK_PINS - 1u]);
	for (uint32_t i = 1; i < DUNGEON_PADLOCK_PINS; ++i)
	{
		double step = distance(at[i - 1u], at[i]);
		smallest = step < smallest ? step : smallest;
	}
	printf("  pick travel: %.1f mm per pin, %.1f mm along the keyway\n", smallest * 1000.0,
		   total * 1000.0);
	assert(smallest > 0.005);
	assert(total > 0.03);

	/* Face-on, keyway depth deliberately projects to one screen position. A
	 * small mouse orbit reveals that depth; the distance checks above ensure it
	 * is really there and the resolved offsets ensure it reaches each pin. */
	for (uint32_t i = 1; i < DUNGEON_PADLOCK_PINS; ++i)
		assert(fabs(across_of(at[i]) - across_of(at[0])) < 1e-5);
}

static void the_pick_only_moves_while_the_lock_is_being_worked(void)
{
	/* Disengaged, the selection must not drag the pick around: the player is
	 * not holding it, and the clips own it entirely. */
	DungeonPadlockAnim anim;
	dungeon_padlock_reset(&anim);
	DungeonPinTumbler first = working_lock(0), last = working_lock(3);
	dungeon_padlock_model_pose(&model, &anim, &first, true, pose, world);
	WorldPosition a = node_world(model.pick_node);
	dungeon_padlock_model_pose(&model, &anim, &last, true, pose, world);
	assert(distance(a, node_world(model.pick_node)) < 1e-6);
	assert(!dungeon_padlock_pick_visible(&anim));
}

static void the_lock_is_upright(void)
{
	DungeonPadlockAnim anim = picking();
	DungeonPinTumbler pins = working_lock(0);
	dungeon_padlock_model_pose(&model, &anim, &pins, true, pose, world);
	/* The lock stands upright rather than lying on its side: its geometry
	 * reaches above the point the focus camera aims at, and the shackle is what
	 * is up there. */
	LocalToWorldTransform placement =
		dungeon_padlock_model_placement(&model, ORIGIN, ACROSS, NORMAL);
	double top = -1e9, shackle_top = -1e9;
	uint32_t shackle = gltf_scene_find_node(&scene, "SHACKLE");
	for (uint32_t i = 0; i < scene.primitive_count; ++i)
	{
		const GltfPrimitive *primitive = &scene.primitives[i];
		if (primitive->node == GLTF_NO_NODE)
			continue;
		LocalToWorldTransform node = coordinate_compose(&placement, world[primitive->node]);
		for (uint32_t v = 0; v < primitive->mesh.vertex_count; ++v)
		{
			const float *local = primitive->vertices[v].position;
			double y = node.translation.y + node.rotation[0][1] * (double)local[0] +
					   node.rotation[1][1] * (double)local[1] +
					   node.rotation[2][1] * (double)local[2];
			top = y > top ? y : top;
			if (primitive->node == shackle || scene.nodes[primitive->node].parent == shackle)
				shackle_top = y > shackle_top ? y : shackle_top;
		}
	}
	printf("  lock reaches %.3f m, camera aims at %.3f m, shackle tops out at %.3f m\n", top,
		   (double)DUNGEON_LOCK_CENTRE_Y_M, shackle_top);
	assert(top > (double)DUNGEON_LOCK_CENTRE_Y_M);
	assert(shackle_top > top - 1e-6); /* the shackle IS the top of the lock */
}

static void a_pin_rises_a_slot_at_a_time(void)
{
	DungeonPadlockAnim anim = picking();
	DungeonPinTumbler pins = working_lock(0);
	double height[DUNGEON_LOCK_PIN_STATES];
	for (uint32_t slot = 0; slot < DUNGEON_LOCK_PIN_STATES; ++slot)
	{
		pins.heights[0] = (uint8_t)slot;
		settle(&anim, &pins);
		dungeon_padlock_model_pose(&model, &anim, &pins, true, pose, world);
		height[slot] = node_world(model.pin_node[0]).y;
	}
	/* Every slot has to be tellable from its neighbour, and by more than a
	 * pixel: this is the number the puzzle is solved from. */
	double smallest = 1e9;
	for (uint32_t slot = 1; slot < DUNGEON_LOCK_PIN_STATES; ++slot)
	{
		double step = fabs(height[slot] - height[slot - 1u]);
		smallest = step < smallest ? step : smallest;
	}
	printf("  pin height readout: %.1f mm per slot, %.1f mm from slot 0 to slot %u\n",
		   smallest * 1000.0,
		   fabs(height[DUNGEON_LOCK_PIN_STATES - 1u] - height[0]) * 1000.0,
		   DUNGEON_LOCK_PIN_STATES - 1u);
	assert(smallest > 0.004);

	/* A key press has to READ as the pin rising, so the drawn height lags the
	 * puzzle's for a few frames rather than teleporting to it. */
	pins.heights[0] = 0u;
	settle(&anim, &pins);
	pins.heights[0] = 1u;
	dungeon_padlock_show_heights(&anim, pins.heights, pins.pin_count, 1.0f / 60.0f);
	dungeon_padlock_model_pose(&model, &anim, &pins, true, pose, world);
	double first_frame = node_world(model.pin_node[0]).y;
	assert(fabs(first_frame - height[0]) < fabs(height[1] - height[0]) * 0.75);
	settle(&anim, &pins);
	dungeon_padlock_model_pose(&model, &anim, &pins, true, pose, world);
	assert(fabs(node_world(model.pin_node[0]).y - height[1]) < 1e-4);
}

/* The bug this replaced: the pop clip lifts every pin by the same authored
 * travel, so four pins with four different targets all came to rest at the same
 * height and a solved lock read as "everything pushed to the top". Where a pin
 * sits is the puzzle's to say. */
static void a_solved_lock_shows_its_combination(void)
{
	DungeonPadlockAnim anim = picking();
	/* Use the real generator rather than hand-authoring the answer: this catches
	 * a generator that promises varied levels while the renderer test quietly
	 * continues to prove only its own synthetic fixture. */
	DungeonPinTumbler pins;
	dungeon_pin_tumbler_init(&pins, 91u, DUNGEON_PADLOCK_PINS);
	for (uint32_t pin = 0; pin < DUNGEON_PADLOCK_PINS; ++pin)
	{
		pins.heights[pin] = pins.target[pin];
		assert(dungeon_pin_tumbler_pin_set(&pins, pin));
		dungeon_padlock_pop_pin(&anim, pin);
	}
	for (int i = 0; i < 600; ++i)
		dungeon_padlock_update(&anim, model.duration, 1.0f / 60.0f);
	settle(&anim, &pins);
	dungeon_padlock_model_pose(&model, &anim, &pins, true, pose, world);

	double at[DUNGEON_PADLOCK_PINS];
	for (uint32_t pin = 0; pin < DUNGEON_PADLOCK_PINS; ++pin)
		at[pin] = node_world(model.pin_node[pin]).y;
	/* Pins on the same target stand together; pins on different targets do
	 * not. Anything else means the lock is not showing its combination. */
	for (uint32_t a = 0; a < DUNGEON_PADLOCK_PINS; ++a)
		for (uint32_t b = a + 1u; b < DUNGEON_PADLOCK_PINS; ++b)
		{
			double apart = fabs(at[a] - at[b]);
			if (pins.target[a] == pins.target[b])
				assert(apart < 1e-4);
			else
				assert(apart > 0.004);
		}
	printf("  solved lock: targets %u%u%u%u at %.1f, %.1f, %.1f, %.1f mm\n", pins.target[0],
		   pins.target[1], pins.target[2], pins.target[3], at[0] * 1000.0, at[1] * 1000.0,
		   at[2] * 1000.0, at[3] * 1000.0);
}

/* A set pin is locked by the puzzle, so nothing may move it afterwards -- not a
 * looping clip, and not the pop it just played. */
static void a_set_pin_stops_moving(void)
{
	DungeonPadlockAnim anim = picking();
	DungeonPinTumbler pins = working_lock(0);
	pins.target[0] = 2u;
	pins.heights[0] = 2u;
	dungeon_padlock_pop_pin(&anim, 0u);
	settle(&anim, &pins);
	dungeon_padlock_model_pose(&model, &anim, &pins, true, pose, world);
	double resting = node_world(model.pin_node[0]).y;
	/* Run the jiggle and the pop right through, several loops of it. */
	for (int i = 0; i < 900; ++i)
	{
		dungeon_padlock_update(&anim, model.duration, 1.0f / 60.0f);
		dungeon_padlock_show_heights(&anim, pins.heights, pins.pin_count, 1.0f / 60.0f);
		dungeon_padlock_model_pose(&model, &anim, &pins, true, pose, world);
		assert(fabs(node_world(model.pin_node[0]).y - resting) < 1e-5);
	}
	/* The keys cannot move it either: dungeon_lock locks a set pin. */
	dungeon_pin_tumbler_adjust(&pins, 1);
	assert(pins.heights[0] == 2u);
}

int main(int argc, char **argv)
{
	const char *path = argc > 1 ? argv[1] : "assets/dungeons/padlock/runtime/padlock.gltf";
	GltfLoadError error = {0};
	GltfLoadResult result = gltf_scene_parse(path, NULL, &scene, &error);
	if (result == GLTF_LOAD_IO_ERROR)
	{
		printf("padlock pose tests skipped: no %s (run tools/export_padlock_runtime.py)\n", path);
		return 0;
	}
	if (result != GLTF_LOAD_OK)
	{
		fprintf(stderr, "padlock pose tests: %s\n", error.message);
		return 1;
	}
	char bind_error[256] = {0};
	if (!dungeon_padlock_model_bind(&model, &scene, bind_error, sizeof(bind_error)))
	{
		fprintf(stderr, "padlock pose tests: %s\n", bind_error);
		return 1;
	}
	pose = calloc(scene.node_count, sizeof(*pose));
	world = calloc(scene.node_count, sizeof(*world));
	assert(pose && world);
	the_pick_moves_to_the_selected_pin();
	the_pick_only_moves_while_the_lock_is_being_worked();
	the_lock_is_upright();
	a_pin_rises_a_slot_at_a_time();
	a_solved_lock_shows_its_combination();
	a_set_pin_stops_moving();
	free(pose);
	free(world);
	gltf_scene_destroy(NULL, &scene);
	printf("padlock pose tests passed\n");
	return 0;
}
