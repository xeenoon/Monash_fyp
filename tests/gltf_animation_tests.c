/* The glTF importer's node/clip half, on the CPU.
 *
 * Everything here is written out as a small glTF at run time rather than
 * checked in as a fixture: the cases that matter are structural -- a child
 * listed before its parent, a clip that touches one node and not another, a
 * sampler that runs out before the caller stops asking -- and they are far
 * easier to read as the code that builds them than as a blob.
 *
 * gltf_scene_parse is the documented CPU-only entry point, so the renderer
 * symbols it would need for an upload are stubbed below rather than linked. */
#include "gltf_scene.h"
#include "renderer.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- renderer stubs: gltf_scene_parse never reaches any of these --- */
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

/* --- the fixture --- */

/* One triangle, then two animation samplers, in one buffer. */
#define POSITION_OFFSET 0
#define INDEX_OFFSET 36
#define TIME_OFFSET 44
#define TRANSLATION_OFFSET 52
#define STEP_TIME_OFFSET 76
#define ROTATION_OFFSET 84
/* Six VEC3s: a CUBICSPLINE sampler stores an in-tangent, a value and an
 * out-tangent per key, and a file that does not is rejected by cgltf's own
 * validation before the importer ever sees the interpolation mode. */
#define CUBIC_OFFSET 116
#define BUFFER_BYTES 188

static const char *DIRECTORY = NULL;

static void write_buffer(const char *path)
{
	unsigned char bytes[BUFFER_BYTES];
	memset(bytes, 0, sizeof(bytes));
	const float positions[9] = {0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
	const unsigned short indices[3] = {0, 1, 2};
	const float times[2] = {0.f, 2.f};
	const float translations[6] = {1.f, 0.f, 0.f, 3.f, 0.f, 0.f};
	const float step_times[2] = {0.f, 1.f};
	/* identity, then a quarter turn about +Y */
	const float rotations[8] = {0.f, 0.f, 0.f, 1.f, 0.f, 0.70710678f, 0.f, 0.70710678f};
	memcpy(bytes + POSITION_OFFSET, positions, sizeof(positions));
	memcpy(bytes + INDEX_OFFSET, indices, sizeof(indices));
	memcpy(bytes + TIME_OFFSET, times, sizeof(times));
	memcpy(bytes + TRANSLATION_OFFSET, translations, sizeof(translations));
	memcpy(bytes + STEP_TIME_OFFSET, step_times, sizeof(step_times));
	memcpy(bytes + ROTATION_OFFSET, rotations, sizeof(rotations));
	FILE *file = fopen(path, "wb");
	assert(file);
	assert(fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes));
	fclose(file);
}

/* `extra` is spliced in before the closing brace, which is how the rejection
 * cases add a skin, a morph target or a cubic sampler to an otherwise valid
 * file. `animations` may be empty, for the static case. */
static char *compose(const char *animations, const char *extra, const char *primitive_extra,
					 const char *child_extra)
{
	static char json[8192];
	snprintf(json, sizeof(json),
		"{\"asset\":{\"version\":\"2.0\"},"
		"\"scene\":0,\"scenes\":[{\"nodes\":[1]}],"
		/* The child is deliberately listed FIRST: glTF does not require
		 * parents to precede children, and posing must not assume it. */
		"\"nodes\":["
		  "{%s\"name\":\"child\",\"mesh\":0,\"translation\":[1,0,0]},"
		  "{\"name\":\"parent\",\"children\":[0],\"translation\":[0,2,0],\"scale\":[2,2,2]}],"
		"\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0},\"indices\":1%s}]}],"
		"\"buffers\":[{\"byteLength\":%d,\"uri\":\"fixture.bin\"}],"
		"\"bufferViews\":["
		  "{\"buffer\":0,\"byteOffset\":%d,\"byteLength\":36},"
		  "{\"buffer\":0,\"byteOffset\":%d,\"byteLength\":6},"
		  "{\"buffer\":0,\"byteOffset\":%d,\"byteLength\":8},"
		  "{\"buffer\":0,\"byteOffset\":%d,\"byteLength\":24},"
		  "{\"buffer\":0,\"byteOffset\":%d,\"byteLength\":8},"
		  "{\"buffer\":0,\"byteOffset\":%d,\"byteLength\":32},"
		  "{\"buffer\":0,\"byteOffset\":%d,\"byteLength\":72}],"
		"\"accessors\":["
		  "{\"bufferView\":0,\"componentType\":5126,\"count\":3,\"type\":\"VEC3\","
		   "\"min\":[0,0,0],\"max\":[1,1,0]},"
		  "{\"bufferView\":1,\"componentType\":5123,\"count\":3,\"type\":\"SCALAR\"},"
		  "{\"bufferView\":2,\"componentType\":5126,\"count\":2,\"type\":\"SCALAR\","
		   "\"min\":[0],\"max\":[2]},"
		  "{\"bufferView\":3,\"componentType\":5126,\"count\":2,\"type\":\"VEC3\"},"
		  "{\"bufferView\":4,\"componentType\":5126,\"count\":2,\"type\":\"SCALAR\","
		   "\"min\":[0],\"max\":[1]},"
		  "{\"bufferView\":5,\"componentType\":5126,\"count\":2,\"type\":\"VEC4\"},"
		  "{\"bufferView\":6,\"componentType\":5126,\"count\":6,\"type\":\"VEC3\"}]"
		"%s%s}",
		child_extra, primitive_extra, BUFFER_BYTES, POSITION_OFFSET, INDEX_OFFSET, TIME_OFFSET,
		TRANSLATION_OFFSET, STEP_TIME_OFFSET, ROTATION_OFFSET, CUBIC_OFFSET, animations, extra);
	return json;
}

static const char *ANIMATIONS =
	",\"animations\":["
	  "{\"name\":\"slide\",\"samplers\":[{\"input\":2,\"output\":3,\"interpolation\":\"LINEAR\"}],"
	   "\"channels\":[{\"sampler\":0,\"target\":{\"node\":0,\"path\":\"translation\"}}]},"
	  "{\"name\":\"turn\",\"samplers\":[{\"input\":4,\"output\":5,\"interpolation\":\"STEP\"}],"
	   "\"channels\":[{\"sampler\":0,\"target\":{\"node\":1,\"path\":\"rotation\"}}]}]";

static char path_buffer[512];
static const char *write_fixture(const char *json)
{
	snprintf(path_buffer, sizeof(path_buffer), "%s/fixture.gltf", DIRECTORY);
	FILE *file = fopen(path_buffer, "wb");
	assert(file);
	fputs(json, file);
	fclose(file);
	return path_buffer;
}

static bool close_enough(float a, float b) { return fabsf(a - b) < 1e-4f; }

/* --- the cases --- */

static void a_static_file_is_still_flattened(void)
{
	GltfScene scene;
	GltfLoadError error = {0};
	assert(gltf_scene_parse(write_fixture(compose("", "", "", "")), NULL, &scene, &error) ==
		   GLTF_LOAD_OK);
	/* No animation means no hierarchy to keep: the node transforms are baked
	 * into the vertices, exactly as the torch and the quarry have always been
	 * loaded. The child sits at (1,0,0) under a parent at (0,2,0) scaled 2. */
	assert(scene.node_count == 0);
	assert(scene.clip_count == 0);
	assert(scene.primitive_count == 1);
	assert(scene.primitives[0].node == GLTF_NO_NODE);
	assert(close_enough(scene.primitives[0].vertices[0].position[0], 2.0f));
	assert(close_enough(scene.primitives[0].vertices[0].position[1], 2.0f));
	assert(close_enough(scene.primitives[0].vertices[1].position[0], 4.0f));
	gltf_scene_destroy(NULL, &scene);
}

static void an_animated_file_keeps_its_hierarchy(void)
{
	GltfScene scene;
	GltfLoadError error = {0};
	assert(gltf_scene_parse(write_fixture(compose(ANIMATIONS, "", "", "")), NULL, &scene, &error) ==
		   GLTF_LOAD_OK);
	assert(scene.node_count == 2);
	assert(scene.clip_count == 2);
	/* Vertices stay in the node's own space -- baking the rest pose in would
	 * fight the pose that replaces it every frame. */
	assert(scene.primitives[0].node == gltf_scene_find_node(&scene, "child"));
	assert(close_enough(scene.primitives[0].vertices[0].position[0], 0.0f));
	assert(close_enough(scene.primitives[0].vertices[1].position[0], 1.0f));

	uint32_t child = gltf_scene_find_node(&scene, "child");
	uint32_t parent = gltf_scene_find_node(&scene, "parent");
	assert(child != GLTF_NO_NODE && parent != GLTF_NO_NODE);
	assert(scene.nodes[child].parent == parent);
	assert(scene.nodes[parent].parent == GLTF_NO_NODE);
	assert(gltf_scene_find_node(&scene, "nobody") == GLTF_NO_NODE);
	/* The file lists the child first; posing needs the parent first. */
	assert(scene.node_order[0] == parent && scene.node_order[1] == child);

	assert(gltf_scene_find_clip(&scene, "slide") != GLTF_NO_NODE);
	assert(gltf_scene_find_clip(&scene, "turn") != GLTF_NO_NODE);
	assert(gltf_scene_find_clip(&scene, "nothing") == GLTF_NO_NODE);
	assert(close_enough(scene.clips[gltf_scene_find_clip(&scene, "slide")].duration, 2.0f));
	assert(close_enough(scene.clips[gltf_scene_find_clip(&scene, "turn")].duration, 1.0f));
	gltf_scene_destroy(NULL, &scene);
}

static void sampling_interpolates_clamps_and_layers(void)
{
	GltfScene scene;
	assert(gltf_scene_parse(write_fixture(compose(ANIMATIONS, "", "", "")), NULL, &scene, NULL) ==
		   GLTF_LOAD_OK);
	uint32_t child = gltf_scene_find_node(&scene, "child");
	uint32_t parent = gltf_scene_find_node(&scene, "parent");
	uint32_t slide = gltf_scene_find_clip(&scene, "slide");
	uint32_t turn = gltf_scene_find_clip(&scene, "turn");
	GltfTransform pose[2];
	mat4s world[2];

	gltf_scene_rest_pose(&scene, pose);
	assert(close_enough(pose[child].translation[0], 1.0f));
	assert(close_enough(pose[parent].scale[1], 2.0f));

	/* Halfway along a LINEAR channel. */
	gltf_clip_sample(&scene, slide, 1.0f, pose);
	assert(close_enough(pose[child].translation[0], 2.0f));
	/* Before the first key and after the last one, sampling CLAMPS -- which is
	 * what lets a caller hold a finished clip on its last pose by parking its
	 * time past the end, instead of it wrapping or running off. */
	gltf_clip_sample(&scene, slide, -5.0f, pose);
	assert(close_enough(pose[child].translation[0], 1.0f));
	gltf_clip_sample(&scene, slide, 900.0f, pose);
	assert(close_enough(pose[child].translation[0], 3.0f));

	/* STEP holds the key it is on rather than blending toward the next. */
	gltf_scene_rest_pose(&scene, pose);
	gltf_clip_sample(&scene, turn, 0.99f, pose);
	assert(close_enough(pose[parent].rotation[1], 0.0f));
	gltf_clip_sample(&scene, turn, 1.0f, pose);
	assert(close_enough(pose[parent].rotation[1], 0.70710678f));

	/* Layering: `turn` writes the parent and leaves the child's translation
	 * exactly as `slide` left it. This is the whole reason sampling writes
	 * into a pose instead of returning one. */
	gltf_scene_rest_pose(&scene, pose);
	gltf_clip_sample(&scene, slide, 2.0f, pose);
	gltf_clip_sample(&scene, turn, 1.0f, pose);
	assert(close_enough(pose[child].translation[0], 3.0f));
	assert(close_enough(pose[parent].rotation[1], 0.70710678f));

	/* World matrices chain: the child's (3,0,0) is scaled by 2 and turned a
	 * quarter turn about +Y by the parent, which sends +X to -Z, then offset
	 * by the parent's own (0,2,0). */
	gltf_scene_world_matrices(&scene, pose, world);
	assert(close_enough(world[child].col[3].x, 0.0f));
	assert(close_enough(world[child].col[3].y, 2.0f));
	assert(close_enough(world[child].col[3].z, -6.0f));
	assert(close_enough(world[parent].col[3].y, 2.0f));

	/* Out-of-range indices are ignored rather than read. */
	gltf_clip_sample(&scene, 99u, 0.0f, pose);
	gltf_scene_destroy(NULL, &scene);
}

/* Linear-blend skinning, on a scene built in memory: two joints, one vertex
 * on each and one split evenly between them. */
static void skinning_blends_joint_matrices(void)
{
	GltfScene scene = {0};
	mat4s world[2] = {GLMS_MAT4_IDENTITY, glms_translate_make((vec3s){{0.0f, 4.0f, 0.0f}})};
	mat4s inverse_bind[2] = {GLMS_MAT4_IDENTITY, glms_translate_make((vec3s){{-1.0f, 0.0f, 0.0f}})};
	uint32_t joints[2] = {0, 1};
	GltfSkin skin = {.joint_count = 2, .joints = joints, .inverse_bind = inverse_bind};
	Vertex bind[3] = {{.position = {0, 0, 0}, .normal = {0, 1, 0}, .tangent = {1, 0, 0, 1}},
					  {.position = {1, 0, 0}, .normal = {0, 1, 0}, .tangent = {1, 0, 0, 1}},
					  {.position = {2, 0, 0}, .normal = {0, 1, 0}, .tangent = {1, 0, 0, 1}}};
	uint16_t vertex_joints[3][4] = {{0}, {0, 1}, {1}};
	float weights[3][4] = {{1}, {0.5f, 0.5f}, {1}};
	GltfPrimitive primitive = {.vertices = bind, .joints = vertex_joints, .weights = weights,
							   .skin = 0, .mesh = {.vertex_count = 3}};
	scene.primitives = &primitive;
	scene.primitive_count = 1;
	scene.skins = &skin;
	scene.skin_count = 1;
	Vertex out[3];
	gltf_scene_skin(&scene, 0, world, out);
	/* Joint 0 is identity; joint 1 moves bind (2,0,0) to (1,4,0). */
	assert(close_enough(out[0].position[1], 0.0f));
	assert(close_enough(out[2].position[0], 1.0f) && close_enough(out[2].position[1], 4.0f));
	/* Half and half: halfway between (1,0,0) and (0,4,0). */
	assert(close_enough(out[1].position[0], 0.5f) && close_enough(out[1].position[1], 2.0f));
	assert(close_enough(out[1].normal[1], 1.0f));
	/* The joint matrix a held prop rides on. */
	mat4s hand = gltf_scene_joint_matrix(&scene, 0, 1, world);
	assert(close_enough(hand.col[3].x, -1.0f) && close_enough(hand.col[3].y, 4.0f));
	mat4s missing = gltf_scene_joint_matrix(&scene, 0, 7, world);
	assert(close_enough(missing.col[3].y, 0.0f));
}

static void unsupported_files_are_refused(void)
{
	GltfScene scene;
	GltfLoadError error = {0};
	/* A skin whose primitive carries no JOINTS_0/WEIGHTS_0 cannot be skinned:
	 * it is refused as malformed rather than drawn undeformed. */
	const char *skinned =
		",\"skins\":[{\"joints\":[1]}],\"animations\":[{\"name\":\"a\",\"samplers\":"
		"[{\"input\":2,\"output\":3}],\"channels\":[{\"sampler\":0,\"target\":"
		"{\"node\":0,\"path\":\"translation\"}}]}]";
	assert(gltf_scene_parse(write_fixture(compose("", skinned, "", "\"skin\":0,")), NULL, &scene,
						   &error) == GLTF_LOAD_INVALID);
	assert(strstr(error.message, "JOINTS_0"));

	/* Morph targets: phase 2 for the padlock's tear-off, unsupported today. */
	assert(gltf_scene_parse(write_fixture(compose("", "", ",\"targets\":[{\"POSITION\":0}]", "")), NULL,
							&scene, &error) == GLTF_LOAD_UNSUPPORTED);
	assert(strstr(error.message, "morph"));

	/* Morph WEIGHT channels, which arrive separately from the targets. A file
	 * carrying one without targets to weight is malformed, so cgltf's own
	 * validation refuses it before the importer's weights branch is reached --
	 * what matters is that it never loads. */
	const char *weights =
		",\"animations\":[{\"name\":\"a\",\"samplers\":[{\"input\":2,\"output\":2}],"
		"\"channels\":[{\"sampler\":0,\"target\":{\"node\":0,\"path\":\"weights\"}}]}]";
	assert(gltf_scene_parse(write_fixture(compose(weights, "", "", "")), NULL, &scene, &error) !=
		   GLTF_LOAD_OK);

	/* CUBICSPLINE, which stores tangents either side of every value. */
	const char *cubic =
		",\"animations\":[{\"name\":\"a\",\"samplers\":[{\"input\":2,\"output\":6,"
		"\"interpolation\":\"CUBICSPLINE\"}],\"channels\":[{\"sampler\":0,\"target\":"
		"{\"node\":0,\"path\":\"translation\"}}]}]";
	assert(gltf_scene_parse(write_fixture(compose(cubic, "", "", "")), NULL, &scene, &error) ==
		   GLTF_LOAD_UNSUPPORTED);
	assert(strstr(error.message, "CUBICSPLINE"));

	assert(gltf_scene_parse("/nonexistent/nothing.gltf", NULL, &scene, &error) ==
		   GLTF_LOAD_IO_ERROR);
}

int main(int argc, char **argv)
{
	DIRECTORY = argc > 1 ? argv[1] : ".";
	char buffer_path[512];
	snprintf(buffer_path, sizeof(buffer_path), "%s/fixture.bin", DIRECTORY);
	write_buffer(buffer_path);
	a_static_file_is_still_flattened();
	an_animated_file_keeps_its_hierarchy();
	sampling_interpolates_clamps_and_layers();
	skinning_blends_joint_matrices();
	unsupported_files_are_refused();
	printf("glTF animation tests passed\n");
	return 0;
}
