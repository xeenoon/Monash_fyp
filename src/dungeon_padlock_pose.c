#include "dungeon_padlock_pose.h"

#include <float.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

/* The local translation a pin gains over its own pop clip. */
static bool pin_travel(const GltfScene *scene, uint32_t clip_index, uint32_t node, vec3s *out)
{
	if (clip_index >= scene->clip_count)
		return false;
	const GltfClip *clip = &scene->clips[clip_index];
	for (uint32_t i = 0; i < clip->channel_count; ++i)
	{
		const GltfChannel *channel = &clip->channels[i];
		if (channel->node != node || channel->path != GLTF_PATH_TRANSLATION || !channel->key_count)
			continue;
		const float *first = channel->values;
		const float *last = &channel->values[(size_t)(channel->key_count - 1u) * 3u];
		*out = (vec3s){{last[0] - first[0], last[1] - first[1], last[2] - first[2]}};
		return true;
	}
	return false;
}

/* Rest-pose bounds, for the placement. `toward_player` is the door normal in
 * the model's own axes, which depends on the turn the placement applies. */
static void measure(DungeonPadlockModel *out, GltfTransform *pose, mat4s *world)
{
	const GltfScene *scene = out->model;
	gltf_scene_rest_pose(scene, pose);
	gltf_scene_world_matrices(scene, pose, world);
	float turn = DUNGEON_PADLOCK_FACE_TURN_DEGREES * 3.14159265358979f / 180.0f;
	float toward_player[3] = {cosf(turn), 0.0f, sinf(turn)};
	float low = FLT_MAX, high = -FLT_MAX, deepest = FLT_MAX;
	for (uint32_t i = 0; i < scene->primitive_count; ++i)
	{
		const GltfPrimitive *primitive = &scene->primitives[i];
		if (primitive->node == GLTF_NO_NODE)
			continue;
		mat4s to_model = world[primitive->node];
		for (uint32_t v = 0; v < primitive->mesh.vertex_count; ++v)
		{
			const float *p = primitive->vertices[v].position;
			vec4s point = glms_mat4_mulv(to_model, (vec4s){{p[0], p[1], p[2], 1.0f}});
			low = point.y < low ? point.y : low;
			high = point.y > high ? point.y : high;
			float along = point.x * toward_player[0] + point.z * toward_player[2];
			deepest = along < deepest ? along : deepest;
		}
	}
	out->reach = -deepest;
	out->centre_height = 0.5f * (low + high);
}

bool dungeon_padlock_model_bind(DungeonPadlockModel *out, const GltfScene *model, char *error,
								size_t error_size)
{
	if (!out || !model)
		return false;
	*out = (DungeonPadlockModel){.model = model, .pick_node = GLTF_NO_NODE};
	for (uint32_t clip = 0; clip < DUNGEON_PADLOCK_CLIP_COUNT; ++clip)
	{
		out->clip[clip] = gltf_scene_find_clip(model, DUNGEON_PADLOCK_CLIP_NAMES[clip]);
		if (out->clip[clip] == GLTF_NO_NODE)
		{
			if (error)
				snprintf(error, error_size, "the padlock has no '%s' clip",
						 DUNGEON_PADLOCK_CLIP_NAMES[clip]);
			return false;
		}
		out->duration[clip] = model->clips[out->clip[clip]].duration;
	}
	out->pick_node = gltf_scene_find_node(model, "PICK");
	for (uint32_t pin = 0; pin < DUNGEON_PADLOCK_PINS; ++pin)
	{
		char name[8];
		snprintf(name, sizeof(name), "PIN_0%u", pin + 1u);
		out->pin_node[pin] = gltf_scene_find_node(model, name);
		if (out->pin_node[pin] == GLTF_NO_NODE ||
			!pin_travel(model, out->clip[DUNGEON_PADLOCK_CLIP_PIN_01_POP + pin], out->pin_node[pin],
						&out->pin_travel[pin]))
		{
			if (error)
				snprintf(error, error_size, "the padlock's %s does not move in its own pop clip",
						 name);
			return false;
		}
	}
	GltfTransform *pose = calloc(model->node_count, sizeof(*pose));
	mat4s *world = calloc(model->node_count, sizeof(*world));
	if (!pose || !world)
	{
		free(pose);
		free(world);
		if (error)
			snprintf(error, error_size, "out of memory measuring the padlock");
		return false;
	}
	measure(out, pose, world);
	/* One pin's worth of keyway depth, in the PICK node's own parent frame: the
	 * pin row's spacing carried through the inverse of the pick's parent, so
	 * the pick reaches exactly as far as the pins are actually apart. */
	if (out->pick_node != GLTF_NO_NODE)
	{
		mat4s first = world[out->pin_node[0]];
		mat4s last = world[out->pin_node[DUNGEON_PADLOCK_PINS - 1u]];
		vec3s step = glms_vec3_divs(glms_vec3_sub(glms_vec3(last.col[3]), glms_vec3(first.col[3])),
									(float)(DUNGEON_PADLOCK_PINS - 1u));
		uint32_t parent = model->nodes[out->pick_node].parent;
		mat4s to_parent =
			parent == GLTF_NO_NODE ? glms_mat4_identity() : glms_mat4_inv(world[parent]);
		out->pick_step = glms_mat4_mulv3(to_parent, step, 0.0f);
	}
	free(pose);
	free(world);
	return true;
}

LocalToWorldTransform dungeon_padlock_model_placement(const DungeonPadlockModel *model,
													  DungeonPoint origin, DungeonPoint across,
													  DungeonPoint normal)
{
	double scale = (double)DUNGEON_PADLOCK_SCALE;
	/* The lock's BACK sits on the standoff plane the rest of the door hardware
	 * hangs off -- DUNGEON_LOCK_STANDOFF_M is already clear of the leaf -- and
	 * the model's own footprint along the door normal decides how far proud of
	 * that it stands. Measured from the TURNED box, since turning the lock
	 * swings its keyway depth into the normal as well, so a re-export that
	 * changes the casing cannot bury it in the door. */
	double proud = (double)model->reach * scale;
	LocalToWorldTransform transform = {
		.translation = {origin.x + (double)normal.x * proud,
						(double)DUNGEON_LOCK_CENTRE_Y_M - (double)model->centre_height * scale,
						origin.z + (double)normal.z * proud}};
	/* Turned about world up by DUNGEON_PADLOCK_FACE_TURN_DEGREES: at zero the
	 * cut face looks straight down the door normal and the pin row runs flat
	 * across the screen; at ninety the padlock's front faces the player and the
	 * row recedes into it. See the constant for why it sits between them.
	 *
	 *   model +X = cos t * normal + sin t * across   (the lock's width)
	 *   model +Y = up                                 (shackle above the body)
	 *   model +Z = sin t * normal - cos t * across    (keyway, and the pin row)
	 *
	 * +Z keeps a negative `across` component for any turn short of ninety, so
	 * PIN_01 -- the puzzle's pin 0 -- stays on the left of the screen. */
	double turn = (double)DUNGEON_PADLOCK_FACE_TURN_DEGREES * 3.14159265358979 / 180.0;
	double c = cos(turn), s = sin(turn);
	transform.rotation[0][0] = (c * (double)normal.x + s * (double)across.x) * scale;
	transform.rotation[0][1] = 0.0;
	transform.rotation[0][2] = (c * (double)normal.z + s * (double)across.z) * scale;
	transform.rotation[1][0] = 0.0;
	transform.rotation[1][1] = scale;
	transform.rotation[1][2] = 0.0;
	transform.rotation[2][0] = (s * (double)normal.x - c * (double)across.x) * scale;
	transform.rotation[2][1] = 0.0;
	transform.rotation[2][2] = (s * (double)normal.z - c * (double)across.z) * scale;
	return transform;
}

void dungeon_padlock_model_pose(const DungeonPadlockModel *model, const DungeonPadlockAnim *anim,
								const DungeonPinTumbler *pins, bool apply_state,
								GltfTransform *pose, mat4s *world)
{
	if (!model || !model->model || !anim || !pins || !pose || !world)
		return;
	gltf_scene_rest_pose(model->model, pose);
	for (uint32_t clip = 0; clip < DUNGEON_PADLOCK_CLIP_COUNT; ++clip)
		if (anim->layers[clip].playing && model->clip[clip] != GLTF_NO_NODE)
			gltf_clip_sample(model->model, model->clip[clip], anim->layers[clip].time, pose);
	if (!apply_state)
	{
		gltf_scene_world_matrices(model->model, pose, world);
		return;
	}
	/* Where a pin sits is the PUZZLE's to say, not a clip's, so this overwrites
	 * the node rather than adding to it.
	 *
	 * Letting the pop clip place a set pin looked right for one pin and wrong
	 * for a lock: the clip lifts every pin by the same authored travel, so four
	 * pins with four different targets all came to rest at the same height and
	 * a solved lock read as "everything pushed to the top". A set pin stands at
	 * ITS target now, which is the combination, and which is the only thing on
	 * the lock that says what the answer was.
	 *
	 * Overwriting also takes the pins out of the looping jiggle: the clip's own
	 * idle wobble on the pin stacks is noise over a readout measured in
	 * millimetres. The drivers and springs above them still move with their
	 * clips, which is what announces a pin dropping in. */
	for (uint32_t pin = 0; pin < DUNGEON_PADLOCK_PINS && pin < pins->pin_count; ++pin)
	{
		uint32_t node = model->pin_node[pin];
		if (node == GLTF_NO_NODE)
			continue;
		float fraction = anim->shown[pin] / (float)(DUNGEON_LOCK_PIN_STATES - 1u) *
						 DUNGEON_PADLOCK_PIN_HEIGHT_SPAN;
		const GltfTransform *rest = &model->model->nodes[node].rest;
		for (uint32_t axis = 0; axis < 3; ++axis)
			pose[node].translation[axis] =
				rest->translation[axis] + model->pin_travel[pin].raw[axis] * fraction;
	}
	/* Which pin the keys are driving is the pick's DEPTH in the keyway, not a
	 * sideways slide: the four pin stacks are spaced along the keyway axis, so
	 * reaching pin 3 means reaching further in -- which is what picking a real
	 * lock looks like. The authored insertion sits at the middle of the row. */
	if (model->pick_node != GLTF_NO_NODE && anim->engaged && pins->pin_count)
	{
		float offset = (float)pins->selected - 0.5f * (float)(pins->pin_count - 1u);
		for (uint32_t axis = 0; axis < 3; ++axis)
			pose[model->pick_node].translation[axis] += model->pick_step.raw[axis] * offset;
	}
	gltf_scene_world_matrices(model->model, pose, world);
}

bool dungeon_padlock_model_node_is_pick(const DungeonPadlockModel *model, uint32_t node)
{
	if (!model || model->pick_node == GLTF_NO_NODE || node == GLTF_NO_NODE)
		return false;
	return node == model->pick_node || model->model->nodes[node].parent == model->pick_node;
}

bool dungeon_padlock_model_pick_offset(const DungeonPadlockModel *model,
									   const DungeonPadlockAnim *anim,
									   const DungeonPinTumbler *pins, GltfTransform *pose,
									   mat4s *world, float *out_pins)
{
	if (!model || model->pick_node == GLTF_NO_NODE || !out_pins || !anim || !pins)
		return false;
	float pitch = glms_vec3_norm(model->pick_step);
	if (!(pitch > 1e-9f) || !pins->pin_count)
		return false;
	/* Twice, because the clips drive the pick along the keyway too -- inserting
	 * it is most of a centimetre of the same axis -- so measuring against the
	 * rest pose would read the animator's travel and the selection offset added
	 * together. */
	dungeon_padlock_model_pose(model, anim, pins, false, pose, world);
	GltfTransform animated = pose[model->pick_node];
	dungeon_padlock_model_pose(model, anim, pins, true, pose, world);
	const GltfTransform *posed = &pose[model->pick_node];
	vec3s offset = {{posed->translation[0] - animated.translation[0],
					 posed->translation[1] - animated.translation[1],
					 posed->translation[2] - animated.translation[2]}};
	if (!isfinite(offset.x) || !isfinite(offset.y) || !isfinite(offset.z))
		return false;
	*out_pins = glms_vec3_dot(offset, model->pick_step) / (pitch * pitch) +
				0.5f * (float)(pins->pin_count - 1u);
	return true;
}
