#include "dungeon_scene.h"

#include "dungeon_cave.h"
#include "dungeon_collision.h"
#include "dungeon_grid.h"
#include "dungeon_lab.h"
#include "dungeon_lock_layout.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef DUNGEON_TEXTURE_DIR
#define DUNGEON_TEXTURE_DIR "textures/dungeon/runtime"
#endif

/* Vertices and indices one moss tuft contributes, matching append_moss_tuft in
 * dungeon_mesh.c. Only the torch lab needs them, to truncate the batch. */
#define DUNGEON_MOSS_TUFT_VERTICES 180u
#define DUNGEON_MOSS_TUFT_INDICES 270u

/* Physical width the hessian material covers, from textures/dungeon/manifest.json.
 * The wrap's UVs are in metres divided by this, so the weave is life-sized. */
#define DUNGEON_CLOTH_WIDTH_M 0.35f

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
	[DUNGEON_MESH_PRISM_BOARD] = DUNGEON_TEXTURE_DIR "/exit_albedo.jpg",
	[DUNGEON_MESH_PRISM] = DUNGEON_TEXTURE_DIR "/exit_albedo.jpg",
	/* Aged metal on the STRUCTURE only -- casing, housing, bores, safe face. The
	 * READOUTS (pins, springs, shear marks) stay untextured on purpose:
	 * mesh.frag multiplies the albedo sample by draw.geometry.rgb, so a tint
	 * can only ever darken what the texture already has, and on a saturated
	 * rust sample a blue "selected" pin came out muddy orange because there is
	 * no blue in it to keep. Their colour is the information; the structure is
	 * what carries the material. */
	[DUNGEON_MESH_LOCK_BODY] = DUNGEON_TEXTURE_DIR "/lock_albedo.jpg",
	[DUNGEON_MESH_LOCK_HOUSING] = DUNGEON_TEXTURE_DIR "/lock_albedo.jpg",
	[DUNGEON_MESH_LOCK_CHANNEL] = DUNGEON_TEXTURE_DIR "/lock_albedo.jpg",
	[DUNGEON_MESH_LOCK_FACE] = DUNGEON_TEXTURE_DIR "/lock_albedo.jpg",
	/* The pins, their springs and the pick all carry the same worn metal now
	 * that none of them encode state through colour -- the safe's pins included:
	 * what they say, they say by how far they stand out of the face. */
	[DUNGEON_MESH_LOCK_PIN] = DUNGEON_TEXTURE_DIR "/lock_albedo.jpg",
	[DUNGEON_MESH_LOCK_SAFE_PIN] = DUNGEON_TEXTURE_DIR "/lock_albedo.jpg",
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
	[DUNGEON_MESH_LOCK_FACE] = DUNGEON_TEXTURE_DIR "/lock_orm.png",
	[DUNGEON_MESH_LOCK_PIN] = DUNGEON_TEXTURE_DIR "/lock_orm.png",
	[DUNGEON_MESH_LOCK_SAFE_PIN] = DUNGEON_TEXTURE_DIR "/lock_orm.png",
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
	[DUNGEON_MESH_LOCK_FACE] = DUNGEON_TEXTURE_DIR "/lock_normal.png",
	[DUNGEON_MESH_LOCK_PIN] = DUNGEON_TEXTURE_DIR "/lock_normal.png",
	[DUNGEON_MESH_LOCK_SAFE_PIN] = DUNGEON_TEXTURE_DIR "/lock_normal.png",
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

/* The burning end of the carried torch, in world space. The carried point
 * light and the carried flame billboard both hang off this, so they can never
 * drift apart. */
static WorldPosition player_torch_head(const DungeonScene *scene)
{
	LocalToWorldTransform carried = player_torch_transform(scene);
	/* Local Y 1.5223 is the top of the source mesh's bowl, not a round number
	 * near it: seating the anchor five centimetres proud of the bowl, as an
	 * earlier value did, leaves a visible gap under the carried flame -- the
	 * one torch the camera ever gets close to. */
	return coordinate_local_to_world(&carried, (TileLocalPosition){0.0f, 1.5223f, -0.13f});
}

/* The carried torch is the one fixture with no entry in scene->lights, so it
 * needs a flicker phase of its own. Any value the golden-ratio sequence in
 * dungeon_lighting_build does not produce will do. */
#define PLAYER_TORCH_PHASE 4.13f

/* Flame billboard size, and how far its bottom edge hangs BELOW the torch
 * head, in metres. The drop is not a fudge to hide a seam: the flame wraps
 * down around the fuel, so the quad has to extend past it. It must equal
 * FLAME_BASE * FLAME_HEIGHT_M from dungeon_flame.frag (0.26 * 0.44), or the
 * shader's fuel line and the actual top of the head stop coinciding. */
#define FLAME_WIDTH_M 0.22f
#define FLAME_HEIGHT_M 0.44f
#define FLAME_BASE_DROP_M 0.114f
/* How far above the fuel the point light sits. A flame's luminous centroid is
 * partway up the plume, not at the wick, and it matters for more than realism:
 * an emitter level with the torch head lights it at grazing incidence, so the
 * head renders black under its own fire. The flame billboard stays anchored to
 * the fuel -- only the light rises. */
#define FLAME_LIGHT_RISE_M 0.10f

/* Radiance of the white-hot core: far above anything a lit surface reaches,
 * so it blows out under tone mapping the way a real flame does on a camera,
 * and so it clears the bloom threshold and gets its halo from the real bloom
 * pyramid rather than from anything painted on the billboard. */
#define FLAME_CORE_RADIANCE 90.0f

/* How far the quad's up axis is allowed to tip back toward a camera looking
 * down on it. 0 is a strictly Y-locked billboard, 1 is fully camera-facing.
 *
 * A flame is vertical, so Y-locked is the honest choice -- and at eye level it
 * is what the formula below produces, exactly, because a level view direction
 * is already perpendicular to world up. But this game's camera sits up to 82
 * degrees above the floor, and a strictly Y-locked quad seen from there is
 * eight degrees off edge-on: the first build of this rendered every torch as
 * a three-pixel smudge. Tipping most of the way splits the difference. */
#define FLAME_CAMERA_TILT 0.75

/* Billboard the flame quad: local +Y is the (tipped) up axis, +X spans the
 * width across the view, +Z is the quad normal. */
static LocalToWorldTransform flame_billboard(WorldPosition anchor, WorldPosition camera)
{
	double to_camera[3] = {camera.x - anchor.x, camera.y - anchor.y, camera.z - anchor.z};
	double length = sqrt(to_camera[0] * to_camera[0] + to_camera[1] * to_camera[1] +
						 to_camera[2] * to_camera[2]);
	if (length < 1e-6)
	{
		to_camera[0] = 0.0;
		to_camera[1] = 0.0;
		to_camera[2] = 1.0;
		length = 1.0;
	}
	for (int axis = 0; axis < 3; ++axis)
		to_camera[axis] /= length;
	/* Remove part of the view direction from world up. The removed amount is
	 * proportional to how far the camera is above the flame, so this is a
	 * no-op for a level view and tips furthest for a top-down one. */
	double elevation = to_camera[1] * FLAME_CAMERA_TILT;
	double up[3] = {-to_camera[0] * elevation, 1.0 - to_camera[1] * elevation,
					-to_camera[2] * elevation};
	double up_length = sqrt(up[0] * up[0] + up[1] * up[1] + up[2] * up[2]);
	if (up_length < 1e-6)
	{
		/* Only reachable looking straight down the flame's own axis, where no
		 * orientation is better than any other. */
		up[0] = 1.0;
		up[1] = 0.0;
		up[2] = 0.0;
		up_length = 1.0;
	}
	for (int axis = 0; axis < 3; ++axis)
		up[axis] /= up_length;
	/* right = up x to_camera, and then normal = right x up, so the three come
	 * out right-handed with +Z pointing back at the camera. Reversing either
	 * cross mirrors the quad and the flame leans the wrong way. */
	double right[3] = {up[1] * to_camera[2] - up[2] * to_camera[1],
					   up[2] * to_camera[0] - up[0] * to_camera[2],
					   up[0] * to_camera[1] - up[1] * to_camera[0]};
	double right_length =
		sqrt(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]);
	if (right_length < 1e-6)
	{
		right[0] = 1.0;
		right[1] = 0.0;
		right[2] = 0.0;
		right_length = 1.0;
	}
	for (int axis = 0; axis < 3; ++axis)
		right[axis] /= right_length;
	double normal[3] = {right[1] * up[2] - right[2] * up[1], right[2] * up[0] - right[0] * up[2],
						right[0] * up[1] - right[1] * up[0]};
	LocalToWorldTransform transform = {0};
	for (int axis = 0; axis < 3; ++axis)
	{
		transform.rotation[0][axis] = right[axis] * FLAME_WIDTH_M;
		transform.rotation[1][axis] = up[axis] * FLAME_HEIGHT_M;
		transform.rotation[2][axis] = normal[axis];
	}
	/* The quad is centred, so lift it half its height along its own up axis to
	 * seat the bottom edge just inside the torch bowl. */
	transform.translation = (WorldPosition){
		anchor.x + up[0] * (FLAME_HEIGHT_M * 0.5 - FLAME_BASE_DROP_M),
		anchor.y + up[1] * (FLAME_HEIGHT_M * 0.5 - FLAME_BASE_DROP_M),
		anchor.z + up[2] * (FLAME_HEIGHT_M * 0.5 - FLAME_BASE_DROP_M)};
	return transform;
}

/* The standing torch (lab scene 2) is two pieces of generated geometry: a
 * tapered shaft, and the rag binding at its head.
 *
 * The binding is a stack of flat tori -- rings whose cross-section is an
 * ellipse flattened against the shaft, which is what a strip of cloth wound
 * round a stick actually looks like. Each band is tilted and offset slightly
 * so the wrap reads as hand-wound rather than machined. It is opaque geometry
 * that writes depth, so the flame (depth-tested, depth-write off) is occluded
 * where it passes behind the cloth instead of glowing through it.
 *
 * Generated rather than authored for the same reason flame_quad is: it is
 * scaffolding for looking at a flame, not level geometry. */
#define LAB_POLE_SIDES 8u
#define LAB_POLE_RINGS 4u
#define LAB_WRAP_MAJOR 20u
#define LAB_WRAP_MINOR 8u
static Vertex lab_pole_vertices[LAB_POLE_SIDES * LAB_POLE_RINGS];
static uint32_t lab_pole_indices[LAB_POLE_SIDES * (LAB_POLE_RINGS - 1u) * 6u];
static Vertex lab_wrap_vertices[DUNGEON_LAB_WRAP_BANDS * LAB_WRAP_MAJOR * LAB_WRAP_MINOR];
static uint32_t lab_wrap_indices[DUNGEON_LAB_WRAP_BANDS * LAB_WRAP_MAJOR * LAB_WRAP_MINOR * 6u];

static void build_lab_pole(float floor_y)
{
	/* Radius and height at each ring: a shaft that narrows with height. The
	 * socket flare that used to sit on top is gone -- the rag binding is what
	 * holds the fuel now, and the flare only ever read as a jagged collar. */
	const float ring_height[LAB_POLE_RINGS] = {0.0f, 0.45f, 0.86f, 1.0f};
	const float ring_radius[LAB_POLE_RINGS] = {1.30f, 1.05f, 0.92f, 0.88f};
	for (uint32_t ring = 0; ring < LAB_POLE_RINGS; ++ring)
		for (uint32_t side = 0; side < LAB_POLE_SIDES; ++side)
		{
			float angle = (float)side / (float)LAB_POLE_SIDES * 6.2831853f;
			/* Jitter the radius: a perfect prism gives a perfect vertical
			 * specular strip down the shaft, the "studio strip light" look
			 * that makes CG geometry obvious. Rough-hewn wood has none. */
			float jitter = sinf((float)(ring * 7u + side * 13u) * 1.7f) * 0.5f + 0.5f;
			float radius = DUNGEON_LAB_POLE_RADIUS_M * ring_radius[ring] * (0.90f + 0.20f * jitter);
			Vertex *v = &lab_pole_vertices[ring * LAB_POLE_SIDES + side];
			*v = (Vertex){0};
			v->position[0] = cosf(angle) * radius;
			v->position[1] = floor_y + DUNGEON_LAB_POLE_HEIGHT_M * ring_height[ring];
			v->position[2] = sinf(angle) * radius;
			v->normal[0] = cosf(angle);
			v->normal[2] = sinf(angle);
			/* Two turns of grain around the shaft and a repeat every 30 cm.
			 * Mapping the 2.4 m wood material at its true physical scale onto
			 * a seven-centimetre pole samples a postage stamp of it, which is
			 * why the shaft came out looking like leopard print. */
			v->texcoord[0] = (float)side / (float)LAB_POLE_SIDES * 2.0f;
			v->texcoord[1] = ring_height[ring] * DUNGEON_LAB_POLE_HEIGHT_M / 0.30f;
			v->tangent[0] = -sinf(angle);
			v->tangent[2] = cosf(angle);
			v->tangent[3] = 1.0f;
		}
	uint32_t index = 0;
	for (uint32_t ring = 0; ring + 1u < LAB_POLE_RINGS; ++ring)
		for (uint32_t side = 0; side < LAB_POLE_SIDES; ++side)
		{
			uint32_t next = (side + 1u) % LAB_POLE_SIDES;
			uint32_t a = ring * LAB_POLE_SIDES + side, b = ring * LAB_POLE_SIDES + next;
			uint32_t c = (ring + 1u) * LAB_POLE_SIDES + side;
			uint32_t d = (ring + 1u) * LAB_POLE_SIDES + next;
			lab_pole_indices[index++] = a;
			lab_pole_indices[index++] = c;
			lab_pole_indices[index++] = d;
			lab_pole_indices[index++] = a;
			lab_pole_indices[index++] = d;
			lab_pole_indices[index++] = b;
		}
}

/* One flat torus per band, wound up the head. `cloth_width_m` is the physical
 * width the material covers, so the weave comes out life-sized. */
static void build_lab_wrap(float floor_y, float cloth_width_m)
{
	const float top = floor_y + DUNGEON_LAB_POLE_HEIGHT_M;
	const float shaft_radius = DUNGEON_LAB_POLE_RADIUS_M * 0.90f;
	uint32_t vertex = 0, index = 0;
	for (uint32_t band = 0; band < DUNGEON_LAB_WRAP_BANDS; ++band)
	{
		float along = (float)band / (float)(DUNGEON_LAB_WRAP_BANDS - 1u);
		float centre_y = top - DUNGEON_LAB_WRAP_HEIGHT_M * along;
		/* Hand-wound: each turn sits a little differently. */
		float wobble = sinf((float)band * 2.3f) * 0.5f + 0.5f;
		/* Bands must OVERLAP. At a half-height under the band spacing they
		 * separate into a stack of rings with gaps between them, which reads
		 * as tyres on an axle rather than as cloth wound round a stick. */
		float spacing = DUNGEON_LAB_WRAP_HEIGHT_M / (float)(DUNGEON_LAB_WRAP_BANDS - 1u);
		/* Just over half the spacing: the turns touch and read as continuous
		 * wound cloth, but each one's rolled upper edge still shows. Push it
		 * to a full overlap and the visible surface becomes the outer equator
		 * alone, whose normals face horizontally -- so under a light sitting
		 * directly above the head the whole wrap goes black. */
		float half_height = spacing * (0.55f + 0.13f * wobble);
		float thickness = 0.0035f + 0.0018f * wobble;   /* out from the shaft */
		float tilt = (sinf((float)band * 1.7f)) * 0.055f; /* radians, off level */
		/* The bundle is fattest at the head and tapers back to the shaft, so
		 * the wrap is a torch head rather than a sleeve. */
		float bulge = 0.008f * (1.0f - along) * (1.0f - along);
		float major_radius = shaft_radius + thickness * 0.6f + bulge;
		uint32_t base = vertex;
		for (uint32_t major = 0; major < LAB_WRAP_MAJOR; ++major)
		{
			float phi = (float)major / (float)LAB_WRAP_MAJOR * 6.2831853f;
			float cos_phi = cosf(phi), sin_phi = sinf(phi);
			for (uint32_t minor = 0; minor < LAB_WRAP_MINOR; ++minor)
			{
				float theta = (float)minor / (float)LAB_WRAP_MINOR * 6.2831853f;
				/* Flattened cross-section: tall along the shaft, shallow out
				 * of it. A circular tube would read as rope, not cloth. */
				float out = thickness * cosf(theta);
				float up = half_height * sinf(theta);
				float radius = major_radius + out;
				Vertex *v = &lab_wrap_vertices[vertex++];
				*v = (Vertex){0};
				v->position[0] = cos_phi * radius;
				v->position[1] = centre_y + up + cos_phi * tilt * major_radius;
				v->position[2] = sin_phi * radius;
				/* Normal of the flattened ellipse, carried back out to world. */
				float normal_out = cosf(theta) * half_height;
				float normal_up = sinf(theta) * thickness;
				float length = sqrtf(normal_out * normal_out + normal_up * normal_up);
				if (length < 1e-6f)
					length = 1.0f;
				v->normal[0] = cos_phi * normal_out / length;
				v->normal[1] = normal_up / length;
				v->normal[2] = sin_phi * normal_out / length;
				v->texcoord[0] = phi * major_radius / cloth_width_m;
				v->texcoord[1] = (centre_y + up) / cloth_width_m;
				v->tangent[0] = -sin_phi;
				v->tangent[2] = cos_phi;
				v->tangent[3] = 1.0f;
			}
		}
		for (uint32_t major = 0; major < LAB_WRAP_MAJOR; ++major)
			for (uint32_t minor = 0; minor < LAB_WRAP_MINOR; ++minor)
			{
				uint32_t next_major = (major + 1u) % LAB_WRAP_MAJOR;
				uint32_t next_minor = (minor + 1u) % LAB_WRAP_MINOR;
				uint32_t a = base + major * LAB_WRAP_MINOR + minor;
				uint32_t b = base + next_major * LAB_WRAP_MINOR + minor;
				uint32_t c = base + major * LAB_WRAP_MINOR + next_minor;
				uint32_t d = base + next_major * LAB_WRAP_MINOR + next_minor;
				lab_wrap_indices[index++] = a;
				lab_wrap_indices[index++] = c;
				lab_wrap_indices[index++] = d;
				lab_wrap_indices[index++] = a;
				lab_wrap_indices[index++] = d;
				lab_wrap_indices[index++] = b;
			}
	}
}

/* The flame billboard's geometry: a unit quad in local XY, centred on the
 * origin, with uv (0,0) at the bottom-left -- dungeon_flame.frag reads uv.y as
 * height up the plume, so the winding and the uv origin both matter. */
static const Vertex FLAME_QUAD_VERTICES[4] = {
	{{-0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}, 0.0f, {1.0f, 0.0f, 0.0f, 1.0f}},
	{{0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}, 0.0f, {1.0f, 0.0f, 0.0f, 1.0f}},
	{{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}, 0.0f, {1.0f, 0.0f, 0.0f, 1.0f}},
	{{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}, 0.0f, {1.0f, 0.0f, 0.0f, 1.0f}},
};
static const uint32_t FLAME_QUAD_INDICES[6] = {0, 1, 2, 0, 2, 3};

/* One fire. `tint` multiplies the shader's own temperature ramp, so a warm
 * torch passes something near neutral and a coloured fixture (the exit's blue)
 * burns in its own colour without the ramp being rewritten. */
static void append_flame_draw(DungeonScene *scene, WorldPosition anchor, const float tint[3],
							  float phase, DungeonFlicker flicker, WorldPosition camera,
							  RendererDraw *out, uint32_t *count, uint32_t capacity)
{
	if (!scene->flame_quad_uploaded || *count >= capacity)
		return;
	/* The fire's vertical sway moves the billboard itself, for the reason
	 * dungeon_flame.frag gives at its `float y` -- shifting it inside the quad
	 * clips the root flat. Lateral sway stays in the shader, where it leans
	 * the plume about a root that does not move. */
	anchor.y += flicker.sway_y;
	LocalToWorldTransform transform = flame_billboard(anchor, camera);
	scene->flame_quad.local_to_world = transform;
	out[(*count)++] = (RendererDraw){
		.mesh = &scene->flame_quad,
		.material_set = scene->flame_quad.material_set,
		.push =
			{
				.local_to_camera_relative =
					coordinate_local_to_camera_relative(&transform, camera),
				.geometry = {{tint[0], tint[1], tint[2], flicker.intensity_scale}},
				/* The same phase dungeon_light_flicker() was given: the shader
				 * folds it into its own clock and its ember seeds, so the drawn
				 * flame and the light it casts move as one fire. */
				.elevation_uv = {{scene->time, phase, FLAME_WIDTH_M, FLAME_HEIGHT_M}},
				.material = {{flicker.sway_x / FLAME_HEIGHT_M, 0.0f, FLAME_CORE_RADIANCE, 0.0f}},
				.debug = {{0.0f, 1.0f, 0.0f, 0.0f}},
			},
		.static_mesh = true,
		.pipeline = RENDERER_PIPELINE_DUNGEON_FLAME};
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
	/* Collision contours only choose a direction. Seat the fixture against the
	 * RENDERED wall at flame height, bulge, masonry and all.
	 *
	 * Several rays, not one. The wall's visible surface is individual bevelled
	 * stones with mortar joints between them, and a single thin ray that
	 * happens to thread a joint reports the backing eight centimetres further
	 * back -- which seats the whole torch inside the stones, with only the tip
	 * of its head poking out. A flat back plate rests on the PROUDEST stone
	 * under it, so sample the plate's own footprint and keep the nearest hit.
	 * The offsets are in the wall plane: lateral (perpendicular to the mount
	 * direction, in XZ) and vertical. */
	const float plate_half_width = 0.05f; /* the source mesh spans X +-0.048 */
	const float plate_half_height = 0.05f;
	DungeonPoint lateral = {-inward.z, inward.x};
	float direction[3] = {-inward.x, 0.0f, -inward.z};
	float reach = sqrtf(best_distance2) + 2.0f;
	float nearest = reach;
	bool any_hit = false;
	for (int sample = 0; sample < 5; ++sample)
	{
		/* Centre, then the four edge midpoints of the plate's footprint. */
		static const float offsets[5][2] = {
			{0.0f, 0.0f}, {-1.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, -1.0f}, {0.0f, 1.0f}};
		float side = offsets[sample][0] * plate_half_width;
		float rise = offsets[sample][1] * plate_half_height;
		float origin[3] = {light->position.x + lateral.x * side, level->floor_y + 1.58f + rise,
						   light->position.z + lateral.z * side};
		float distance = reach, hit_normal[3];
		if (!dungeon_shadow_trace(shadow, origin, direction, &distance, hit_normal))
			continue;
		any_hit = true;
		if (distance < nearest)
			nearest = distance;
	}
	if (!any_hit)
		return false;
	/* Rebuild the contact point on the mount axis itself, so a hit found by an
	 * off-axis sample still seats the fixture square to the wall. */
	mount = (DungeonPoint){light->position.x + direction[0] * nearest,
						   light->position.z + direction[2] * nearest};
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

/* A transform for a piece whose axis is the door NORMAL rather than world up:
 * local +Y maps to the normal, and local +X/+Z spin in the plane of the door
 * face by `spin` radians. coordinate_rotation_y cannot express either half of
 * that. Only the safe needs it -- its face plate is a disc lying against the
 * door, and its pins drive straight out of that face toward the player, which
 * is the door normal and not any axis coordinate_rotation_y can name. */
static LocalToWorldTransform lock_face_spin(WorldPosition translation, DungeonPoint across,
											DungeonPoint normal, float spin)
{
	LocalToWorldTransform transform = {.translation = translation};
	double c = cos((double)spin), s = sin((double)spin);
	/* [column][row]: local X, then Y, then Z = X cross Y. */
	transform.rotation[0][0] = c * (double)across.x;
	transform.rotation[0][1] = s;
	transform.rotation[0][2] = c * (double)across.z;
	transform.rotation[1][0] = (double)normal.x;
	transform.rotation[1][1] = 0.0;
	transform.rotation[1][2] = (double)normal.z;
	transform.rotation[2][0] = s * (double)normal.z;
	transform.rotation[2][1] = -c;
	transform.rotation[2][2] = -s * (double)normal.x;
	return transform;
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

static void append_safe_pins_draws(DungeonScene *scene, uint32_t index, bool focused,
								   WorldPosition camera, RendererDraw *out, uint32_t *count,
								   uint32_t capacity)
{
	const DungeonSession *session = &scene->session;
	const DungeonDoorState *state = &session->doors[index];
	DungeonPoint origin, across, normal;
	lock_frame(&scene->level, session, index, &origin, &across, &normal);
	float tint = focused ? 1.0f : 0.5f;

	append_lock_body(scene, origin, across, normal, tint, camera, out, count, capacity);

	/* The jolt a press lands with, along the door normal so the whole face
	 * kicks toward the player and settles. It decays in the session, and sin()
	 * of the decaying value rings down on its own without a second piece of
	 * state to carry a phase. Given to a driven pin and to a reset alike --
	 * confirm always lands, whichever way it went. */
	float jolt = sinf(state->safe_shake * 46.0f) * state->safe_shake * 0.006f;
	push_instance(scene, DUNGEON_MESH_LOCK_FACE,
				  lock_face_spin(lock_point(origin, across, normal, 0.0f, jolt,
											DUNGEON_LOCK_FACE_Y),
								 across, normal, 0.0f),
				  camera, 0.46f * tint, 0.44f * tint, 0.42f * tint, 0.52f, LOCK_METAL, out, count,
				  capacity);

	/* The four pins, in a row across the face. How far out of the face a pin
	 * stands is the whole of this puzzle's readout: flush is untouched, a
	 * fraction out is the one the keys are on, and all the way out is driven
	 * and banked. The session eases `safe_push`, so a press animates and a
	 * wrong press springs every pin back without the renderer tracking any of
	 * it. */
	float span = (float)(state->safe.pin_count - 1u) * DUNGEON_SAFE_PIN_PITCH_M * 0.5f;
	for (uint32_t pin = 0; pin < state->safe.pin_count; ++pin)
	{
		float lateral = (float)pin * DUNGEON_SAFE_PIN_PITCH_M - span;
		float proud = DUNGEON_SAFE_PIN_BASE_PROUD_M + jolt +
					  state->safe_push[pin] * DUNGEON_SAFE_PIN_DRIVE_M;
		push_instance(scene, DUNGEON_MESH_LOCK_SAFE_PIN,
					  lock_face_spin(lock_point(origin, across, normal, lateral, proud,
												DUNGEON_LOCK_FACE_Y + DUNGEON_SAFE_PIN_LIFT_M),
									 across, normal, 0.0f),
					  camera, 0.80f * tint, 0.79f * tint, 0.76f * tint, 0.34f, LOCK_METAL, out,
					  count, capacity);
	}
}

#define PRISM_BOARD_WIDTH .78f

/* Local XY is screen-right/up, +Z out of the door. Scaling is used only for
 * flat quads; crystal transforms remain rigid for analytic refraction. */
static LocalToWorldTransform prism_transform(WorldPosition position, DungeonPoint across,
											 DungeonPoint normal, float angle, float sx, float sy)
{
	LocalToWorldTransform t = {.translation = position};
	float c = cosf(angle), s = sinf(angle);
	t.rotation[0][0] = across.x * c * sx;
	t.rotation[0][1] = s * sx;
	t.rotation[0][2] = across.z * c * sx;
	t.rotation[1][0] = -across.x * s * sy;
	t.rotation[1][1] = c * sy;
	t.rotation[1][2] = -across.z * s * sy;
	t.rotation[2][0] = normal.x;
	t.rotation[2][2] = normal.z;
	return t;
}
static void prism_draw(DungeonScene *scene, DungeonMeshBatchKind mesh,
					   LocalToWorldTransform transform, WorldPosition camera, int mode, float r,
					   float g, float b, float bx, float by, float angle, bool lit, bool rotating,
					   RendererDraw *out, uint32_t *count, uint32_t capacity)
{
	uint32_t before = *count;
	push_instance(scene, mesh, transform, camera, r, g, b, .2f, 0, out, count, capacity);
	if (*count == before)
		return;
	RendererDraw *draw = &out[*count - 1];
	draw->pipeline = mode == 1	 ? RENDERER_PIPELINE_DUNGEON_GLASS
					 : mode == 2 ? RENDERER_PIPELINE_DUNGEON_LIGHT
					 : mode == 0 ? RENDERER_PIPELINE_DUNGEON_PRISM
								 : RENDERER_PIPELINE_DUNGEON_BEAM;
	draw->push.elevation_uv = (vec4s){{bx, by, angle, PRISM_BOARD_WIDTH}};
	draw->push.material = (vec4s){{(float)mode, lit ? 1.f : 0.f, rotating ? 1.f : 0.f, 0}};
}
static void append_prism_draws(DungeonScene *scene, uint32_t index, bool overlays,
							   WorldPosition camera, RendererDraw *out, uint32_t *count,
							   uint32_t capacity)
{
	const DungeonSession *session = &scene->session;
	const DungeonPrismPuzzle *p = &session->doors[index].prism;
	DungeonPoint origin, across, normal;
	lock_frame(&scene->level, session, index, &origin, &across, &normal);
	bool focused = session->phase == DUNGEON_PHASE_PRISM && session->focused_door == index;
	DungeonPrismTrace trace = dungeon_prism_trace(p);
	float center = DUNGEON_LOCK_CENTRE_Y_M;
	if (!overlays)
	{
		WorldPosition position = lock_point(origin, across, normal, 0, 0, center);
		prism_draw(
			scene, DUNGEON_MESH_PRISM_BOARD,
			prism_transform(position, across, normal, 0, PRISM_BOARD_WIDTH, PRISM_BOARD_WIDTH),
			camera, 0, 1, 1, 1, 0, 0, 0, false, false, out, count, capacity);
		return;
	}
	for (uint32_t i = 0; i < p->count; ++i)
	{
		float x = p->prisms[i].position.x;
		float y = p->prisms[i].position.y;
		float turn = (float)p->prisms[i].orientation;
		if (i == p->selected && p->turn_remaining > 0)
		{
			float a = 1.f - p->turn_remaining / DUNGEON_PRISM_TURN_SECONDS;
			a = a * a * (3.f - 2.f * a);
			turn += p->turn_direction * a;
		}
		float angle = -turn * .01745329252f;
		WorldPosition position = lock_point(origin, across, normal, x, .015f, center + y);
		DungeonMeshBatchKind mesh =
			p->prisms[i].kind == DUNGEON_OPTIC_CONVEX	 ? DUNGEON_MESH_CONVEX_LENS
			: p->prisms[i].kind == DUNGEON_OPTIC_CONCAVE ? DUNGEON_MESH_CONCAVE_LENS
														 : DUNGEON_MESH_PRISM;
		uint32_t before = *count;
		prism_draw(scene, mesh, prism_transform(position, across, normal, angle, 1, 1), camera, 1,
				   1, 1, 1, x, y, angle, (trace.lit_mask & (1u << i)) != 0, false, out, count,
				   capacity);
		if (*count > before)
			out[*count - 1].push.material.w = (float)p->prisms[i].kind;
	}
	/* Soft fans share exact solver endpoints and continuous widths. They
	 * enter the scene snapshot before glass refracts them in the final pass. */
	for (uint32_t i = 0; i < trace.count; ++i)
	{
		DungeonPrismSegment segment = trace.segments[i];
		float ax = segment.a.x, ay = segment.a.y;
		float bx = segment.b.x, by = segment.b.y;
		WorldPosition position =
			lock_point(origin, across, normal, (ax + bx) * .5f, .022f, center + (ay + by) * .5f);
		prism_draw(scene, DUNGEON_MESH_PRISM_QUAD,
				   prism_transform(position, across, normal, atan2f(by - ay, bx - ax),
								   hypotf(bx - ax, by - ay), segment.width_b),
				   camera, 2, segment.channel == 0 ? 24.f * segment.intensity : 0,
				   segment.channel == 1 ? 19.f * segment.intensity : 0,
				   segment.channel == 2 ? 12.f * segment.intensity : 0,
				   segment.width_a / segment.width_b, 1, 0, false, false, out, count, capacity);
	}
	for (uint32_t i = 0; i < p->count; ++i)
	{
		float x = p->prisms[i].position.x;
		float y = p->prisms[i].position.y;
		bool selected = focused && p->selected == i;
		WorldPosition position = lock_point(origin, across, normal, x, .063f, center + y);
		prism_draw(scene, DUNGEON_MESH_PRISM_QUAD,
				   prism_transform(position, across, normal, 0, .12f, .12f), camera, 3,
				   selected ? (p->rotating ? .85f : .65f) : .13f,
				   selected ? (p->rotating ? .58f : .60f) : .11f,
				   selected ? (p->rotating ? .25f : .46f) : .08f, 0, 0, 0, false,
				   selected && p->rotating, out, count, capacity);
	}
	DungeonPrismPoint glyphs[2] = {p->source, p->key};
	for (unsigned i = 0; i < 2; ++i)
	{
		WorldPosition position =
			lock_point(origin, across, normal, glyphs[i].x, .027f, center + glyphs[i].y);
		float angle = i ? p->key_angle : p->source_angle;
		bool lit = !i || trace.hit_key;
		prism_draw(scene, DUNGEON_MESH_PRISM_QUAD,
				   prism_transform(position, across, normal, angle, .085f, .085f), camera,
				   i ? 4 : 5, lit ? 1.0f : .35f, lit ? .72f : .25f, lit ? .34f : .12f, 0, 0, 0,
				   false, false, out, count, capacity);
	} // A live power gauge beside the keyhole. Its fill is current collected
	// light / required light, and falls immediately when a beam moves away.
	float gauge_x = fmaxf(-.29f, fminf(.29f, p->key.x));
	float gauge_y = p->key.y > .27f ? p->key.y - .052f : p->key.y + .052f;
	WorldPosition gauge = lock_point(origin, across, normal, gauge_x, .067f, center + gauge_y);
	uint32_t before = *count;
	prism_draw(scene, DUNGEON_MESH_PRISM_QUAD,
			   prism_transform(gauge, across, normal, 0, .12f, .016f), camera, 6, 1, .72f, .35f, 0,
			   0, 0, false, false, out, count, capacity);
	if (*count > before)
		out[*count - 1].push.material.z = fminf(trace.key_power / DUNGEON_OPTIC_KEY_POWER, 1.f);
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
		if (door->lock == DUNGEON_LOCK_PRISM)
			append_prism_draws(scene, i, false, camera, out, count, capacity);
		else if (door->lock == DUNGEON_LOCK_SAFE_PINS)
			append_safe_pins_draws(scene, i, focused, camera, out, count, capacity);
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
	out->lab = getenv("DUNGEON_LAB") ? (DungeonLabScene)atoi(getenv("DUNGEON_LAB"))
									 : DUNGEON_LAB_NONE;
	if (out->lab > DUNGEON_LAB_CARRIED)
		out->lab = DUNGEON_LAB_WALL;
	bool compiled;
	if (out->lab)
		compiled = dungeon_lab_compile(&out->level, error);
	else if (map_override)
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
	if (out->lab)
	{
		/* One clump, not a carpet. The scatter's reservoir fills its budget in
		 * triangle order before it starts replacing, so the first tufts it
		 * emitted are a contiguous patch of floor rather than a thinned-out
		 * spread -- truncating the batch keeps that patch and drops the rest.
		 * Done here rather than by teaching dungeon_mesh.c a budget: the
		 * shipping scatter is one of the things this scene exists to look at,
		 * and it should not gain a debug-only parameter. */
		const uint32_t lab_tufts = 40u;
		DungeonMeshBatch *moss = &out->geometry.batches[DUNGEON_MESH_MOSS];
		if (moss->vertex_count > lab_tufts * DUNGEON_MOSS_TUFT_VERTICES)
		{
			moss->vertex_count = lab_tufts * DUNGEON_MOSS_TUFT_VERTICES;
			moss->index_count = lab_tufts * DUNGEON_MOSS_TUFT_INDICES;
		}
	}
	if (out->lab == DUNGEON_LAB_STANDING)
	{
		build_lab_pole(out->level.floor_y);
		out->lab_pole = (Mesh){
			.local_to_world = coordinate_identity_transform(
				(WorldPosition){dungeon_lab_torch_position(out->lab).x, 0.0,
								dungeon_lab_torch_position(out->lab).z}),
			.vertices = lab_pole_vertices,
			.vertex_count = (uint32_t)(sizeof(lab_pole_vertices) / sizeof(*lab_pole_vertices)),
			.indices = lab_pole_indices,
			.index_count = (uint32_t)(sizeof(lab_pole_indices) / sizeof(*lab_pole_indices)),
			/* The exit's medieval-wood set is the only wood checked in, and a
			 * torch pole is exactly what it suits. */
			.texture_path = DUNGEON_TEXTURE_DIR "/exit_albedo.jpg",
			.orm_path = DUNGEON_TEXTURE_DIR "/exit_orm.png",
			.normal_path = DUNGEON_TEXTURE_DIR "/exit_normal.png",
		};
		mesh_upload(renderer, &out->lab_pole);
		out->lab_pole_uploaded = true;

		build_lab_wrap(out->level.floor_y, DUNGEON_CLOTH_WIDTH_M);
		out->lab_wrap = (Mesh){
			.local_to_world = out->lab_pole.local_to_world,
			.vertices = lab_wrap_vertices,
			.vertex_count = (uint32_t)(sizeof(lab_wrap_vertices) / sizeof(*lab_wrap_vertices)),
			.indices = lab_wrap_indices,
			.index_count = (uint32_t)(sizeof(lab_wrap_indices) / sizeof(*lab_wrap_indices)),
			.texture_path = DUNGEON_TEXTURE_DIR "/cloth_albedo.jpg",
			.orm_path = DUNGEON_TEXTURE_DIR "/cloth_orm.png",
			.normal_path = DUNGEON_TEXTURE_DIR "/cloth_normal.png",
		};
		mesh_upload(renderer, &out->lab_wrap);
		out->lab_wrap_uploaded = true;
	}
	/* dungeon_flame.frag samples no textures, so the quad needs no material
	 * of its own; mesh_upload's fallback set satisfies the shared layout. */
	out->flame_quad = (Mesh){
		.local_to_world = coordinate_identity_transform((WorldPosition){0}),
		.vertices = FLAME_QUAD_VERTICES,
		.vertex_count = 4u,
		.indices = FLAME_QUAD_INDICES,
		.index_count = 6u,
	};
	mesh_upload(renderer, &out->flame_quad);
	out->flame_quad_uploaded = true;
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
	if (out->lab)
	{
		/* One fire, and nothing else lit: the exit beacon dungeon_lighting_build
		 * always adds is a second light source, and two of them make it
		 * impossible to attribute anything on screen to the torch. Scene 3's
		 * fire is the one the player carries, so it places no fixture at all
		 * and leaves the light to dungeon_scene_write_lights. */
		if (out->lab == DUNGEON_LAB_CARRIED)
			out->light_count = 0u;
		else
		{
			out->lights[0] = (DungeonLight){
				.position = dungeon_lab_torch_position(out->lab),
				.height = out->lab == DUNGEON_LAB_STANDING
							  ? out->level.floor_y + DUNGEON_LAB_POLE_HEIGHT_M + 0.01f
							  : 1.58f,
				.radius = 6.5f,
				.color = {1.0f, 0.38f, 0.12f},
				.intensity = 12.0f,
				.flame_tint = {1.0f, 1.0f, 1.0f}};
			out->light_count = 1u;
		}
	}
	else
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
		if (out->lab == DUNGEON_LAB_STANDING)
		{
			/* Nothing to mount against: the fixture stands on its own pole in
			 * the middle of the room, so the flame keeps the position the lab
			 * chose and there is no wall-bracket mesh. */
			fprintf(stdout, "Torch %u: standing flame at (%.2f, %.2f, %.2f)\n", i,
					(double)out->lights[i].position.x, (double)out->lights[i].height,
					(double)out->lights[i].position.z);
		}
		else if (mount_torch(&out->level, &out->wall_shadow, &out->lights[i],
							 &out->torch_transforms[out->torch_count]))
		{
			/* Where each fire actually ended up, so a harness script can aim a
			 * camera at one without anybody reading it off a picture. */
			fprintf(stdout, "Torch %u: flame at (%.2f, %.2f, %.2f)\n", out->torch_count,
					(double)out->lights[i].position.x, (double)out->lights[i].height,
					(double)out->lights[i].position.z);
			++out->torch_count;
		}
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

/* Every fire in the level, carried torch included. Appended after the shadow
 * prefix on purpose: a billboard of hot gas has no silhouette worth casting,
 * and putting one in the depth-only pass would stamp a black rectangle on the
 * wall behind every torch. */
static void append_flame_draws(DungeonScene *scene, WorldPosition camera_position,
							   RendererDraw *out, uint32_t *count, uint32_t capacity)
{
	if (!scene->lab || scene->lab == DUNGEON_LAB_CARRIED)
		append_flame_draw(scene, player_torch_head(scene), (const float[3]){1.0f, 1.0f, 1.0f},
						  PLAYER_TORCH_PHASE,
						  dungeon_light_flicker(PLAYER_TORCH_PHASE, scene->time), camera_position,
						  out, count, capacity);
	for (uint32_t i = 0; i < scene->light_count; ++i)
	{
		/* Zero intensity is how a fixture that found no wall to mount on is
		 * recorded (see dungeon_scene_create). It has no torch mesh, so it
		 * must not have a flame hanging in mid-air either. */
		if (scene->lights[i].intensity <= 0.0f)
			continue;
		WorldPosition anchor = {scene->lights[i].position.x, scene->lights[i].height,
								scene->lights[i].position.z};
		append_flame_draw(scene, anchor, scene->lights[i].flame_tint, scene->lights[i].phase,
						  dungeon_light_flicker(scene->lights[i].phase, scene->time),
						  camera_position, out, count, capacity);
	}
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
		if (scene->lab && scene->lab != DUNGEON_LAB_CARRIED && i == DUNGEON_MESH_PLAYER)
			continue; /* scenes 1 and 2 are about the fixture, not who holds one */
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
	if (scene->lab_pole_uploaded && draw_count < capacity)
	{
		out[draw_count++] = (RendererDraw){
			.mesh = &scene->lab_pole,
			.material_set = scene->lab_pole.material_set,
			.push = dungeon_push(&scene->lab_pole, camera_position),
			.static_mesh = true};
		out[draw_count - 1].push.geometry.w = scene->level.floor_y;
	}
	if (scene->lab_wrap_uploaded && draw_count < capacity)
	{
		out[draw_count++] = (RendererDraw){
			.mesh = &scene->lab_wrap,
			.material_set = scene->lab_wrap.material_set,
			.push = dungeon_push(&scene->lab_wrap, camera_position),
			.static_mesh = true};
		/* Sooty, and not shiny: hessian that has been burning is matte. */
		out[draw_count - 1].push.geometry = (vec4s){{0.52f, 0.42f, 0.33f, scene->level.floor_y}};
		out[draw_count - 1].push.elevation_uv.x = 0.92f; /* roughness factor */
	}
	/* Reserve the carried torch before optional wall fixtures. */
	LocalToWorldTransform carried = player_torch_transform(scene);
	bool carried_torch = !scene->lab || scene->lab == DUNGEON_LAB_CARRIED;
	for (uint32_t i = 0; i < scene->torch.primitive_count && draw_count < capacity && carried_torch;
		 ++i)
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
	append_flame_draws(scene, camera_position, out, &draw_count, capacity);
	if (scene->uploaded[DUNGEON_MESH_PUDDLE] && draw_count < capacity)
		out[draw_count++] = (RendererDraw){
			.mesh = &scene->meshes[DUNGEON_MESH_PUDDLE],
			.material_set = scene->meshes[DUNGEON_MESH_PUDDLE].material_set,
			.push = puddle_push(&scene->meshes[DUNGEON_MESH_PUDDLE], camera_position),
			.static_mesh = true,
			.pipeline = RENDERER_PIPELINE_DUNGEON_PUDDLE};
	/* Alpha ribbons and glyphs are excluded from the shadow prefix. */
	for (uint32_t i = 0; i < scene->session.door_count; ++i)
	{
		if (!scene->session.doors[i].open && scene->level.doors[i].lock == DUNGEON_LOCK_PRISM)
			append_prism_draws(scene, i, true, camera_position, out, &draw_count, capacity);
	}
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
	/* Advanced before the early return below, so torches keep burning while a
	 * lock is being picked. */
	scene->time += dt;
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
	/* Every torch here is flickered by the same dungeon_light_flicker() call
	 * the flame billboard in append_flame_draws() used, at the same
	 * scene->time. That is the whole point of computing it on the CPU: the
	 * room brightens on exactly the frame the fire the player is looking at
	 * flares, and the sway walks the lit patch on the wall with the plume. */
	/* Slot zero is always the player light, even if the light budget is full --
	 * except in the lab, where the one wall torch has to be the only thing
	 * lighting anything, and it takes slot zero instead. */
	if (scene->lab && scene->lab != DUNGEON_LAB_CARRIED)
	{
		uint32_t lab_count = scene->light_count < capacity ? scene->light_count : capacity;
		for (uint32_t i = 0; i < lab_count; ++i)
		{
			DungeonFlicker flicker = dungeon_light_flicker(scene->lights[i].phase, scene->time);
			WorldPosition world = {scene->lights[i].position.x + flicker.sway_x,
								   scene->lights[i].height + flicker.sway_y,
								   scene->lights[i].position.z};
			world.y += FLAME_LIGHT_RISE_M;
			CameraRelativePosition to_light = coordinate_camera_relative(world, camera_position);
			positions[i] =
				(vec4s){{to_light.x, to_light.y, to_light.z, scene->lights[i].radius}};
			float rgb[3];
			dungeon_light_warmth_color(scene->lights[i].color, flicker.warmth, rgb);
			colors[i] = (vec4s){
				{rgb[0], rgb[1], rgb[2], scene->lights[i].intensity * flicker.intensity_scale}};
		}
		return lab_count;
	}
	WorldPosition head = player_torch_head(scene);
	DungeonFlicker carried_flicker = dungeon_light_flicker(PLAYER_TORCH_PHASE, scene->time);
	head.x += carried_flicker.sway_x;
	head.y += carried_flicker.sway_y + FLAME_LIGHT_RISE_M;
	CameraRelativePosition relative = coordinate_camera_relative(head, camera_position);
	positions[0] = (vec4s){{relative.x, relative.y, relative.z, 7.5f}};
	static const float carried_color[3] = {1.0f, 0.76f, 0.48f};
	float carried_rgb[3];
	dungeon_light_warmth_color(carried_color, carried_flicker.warmth, carried_rgb);
	colors[0] = (vec4s){{carried_rgb[0], carried_rgb[1], carried_rgb[2],
						 16.0f * carried_flicker.intensity_scale}};
	uint32_t count = scene->light_count < capacity - 1u ? scene->light_count : capacity - 1u;
	for (uint32_t i = 0; i < count; ++i)
	{
		DungeonFlicker flicker = dungeon_light_flicker(scene->lights[i].phase, scene->time);
		WorldPosition world = {scene->lights[i].position.x + flicker.sway_x,
							   scene->lights[i].height + flicker.sway_y + FLAME_LIGHT_RISE_M,
							   scene->lights[i].position.z};
		CameraRelativePosition relative = coordinate_camera_relative(world, camera_position);
		positions[i + 1u] = (vec4s){{relative.x, relative.y, relative.z, scene->lights[i].radius}};
		float rgb[3];
		dungeon_light_warmth_color(scene->lights[i].color, flicker.warmth, rgb);
		colors[i + 1u] = (vec4s){
			{rgb[0], rgb[1], rgb[2], scene->lights[i].intensity * flicker.intensity_scale}};
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
		if (scene->flame_quad_uploaded)
			mesh_destroy(renderer, &scene->flame_quad);
		if (scene->lab_pole_uploaded)
			mesh_destroy(renderer, &scene->lab_pole);
		if (scene->lab_wrap_uploaded)
			mesh_destroy(renderer, &scene->lab_wrap);
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
