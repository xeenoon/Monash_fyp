#pragma once

/* Strict, static subset of glTF 2.0 used by the static mesh renderer.  The
 * public type intentionally owns both the CPU conversion and its GPU upload;
 * callers never need to retain cgltf objects or source buffers. */
#include <stdbool.h>
#include <stdint.h>
#include "coordinate.h"
#include "mesh.h"

typedef enum {
    GLTF_LOAD_OK = 0,
    GLTF_LOAD_IO_ERROR,
    GLTF_LOAD_INVALID,
    GLTF_LOAD_UNSUPPORTED,
    GLTF_LOAD_OUT_OF_MEMORY
} GltfLoadResult;

typedef struct {
    char message[512];
    const char *source_path;
    uint32_t node_index;
    uint32_t primitive_index;
} GltfLoadError;

typedef struct {
    bool use_metallic_roughness_red_as_occlusion;
    LocalToWorldTransform placement;
} GltfLoadOptions;

/* Sentinel for "no node": a primitive from a flattened file, or a root's
 * parent. */
#define GLTF_NO_NODE UINT32_MAX

/* A node's local transform, kept as TRS rather than as a matrix because that
 * is what glTF animates -- a clip drives translation, rotation and scale as
 * three independent channels, and composing them back into a matrix before
 * sampling would make layering two clips over one node impossible. */
typedef struct {
    float translation[3];
    float rotation[4]; /* xyzw, glTF's quaternion order */
    float scale[3];
} GltfTransform;

typedef enum {
    GLTF_PATH_TRANSLATION,
    GLTF_PATH_ROTATION,
    GLTF_PATH_SCALE
} GltfPath;

/* One animated node property. `times` is strictly increasing seconds; sampling
 * outside the range clamps, which is what lets a caller hold a finished clip on
 * its last pose by parking its time past the end. */
typedef struct {
    uint32_t node;
    GltfPath path;
    bool step; /* STEP interpolation; LINEAR otherwise (slerp for rotation) */
    uint32_t key_count;
    float *times;
    float *values; /* key_count * (3, or 4 for rotation) */
} GltfChannel;

typedef struct {
    char *name;
    float duration;
    GltfChannel *channels;
    uint32_t channel_count;
} GltfClip;

typedef struct {
    char *name;
    uint32_t parent; /* GLTF_NO_NODE at a root */
    GltfTransform rest;
} GltfNode;

typedef struct {
    Mesh mesh;
    Vertex *vertices;
    uint32_t *indices;
    uint32_t material_index;
    /* Which node the geometry hangs off, or GLTF_NO_NODE when the file had no
     * animation and the hierarchy was flattened into the vertices at load. */
    uint32_t node;
    /* Skinned primitives: the skin index (GLTF_NO_NODE when rigid), four
     * joint slots (indices into that skin's joint list) and weights per
     * vertex, parallel to `vertices`, which stay in bind pose. See
     * gltf_scene_skin. */
    uint32_t skin;
    uint16_t (*joints)[4];
    float (*weights)[4];
} GltfPrimitive;

typedef struct {
    uint32_t joint_count;
    uint32_t *joints;     /* node index of each joint */
    mat4s *inverse_bind;  /* per joint */
} GltfSkin;

typedef struct {
    float base_color_factor[4];
    float metallic_factor, roughness_factor, normal_scale, occlusion_strength;
    bool use_metallic_roughness_red_as_occlusion;
    VkDescriptorSet descriptor_set;
    char *base_color_path, *metallic_roughness_path, *normal_path, *occlusion_path;
    Texture base_color, metallic_roughness, normal, occlusion;
	bool material_cached;
} GltfMaterial;

/* An imported scene is in ONE of two shapes, decided by whether the file has
 * animation:
 *
 *   static  -- every node transform is baked into the vertices at load, nodes
 *              and clips are empty, and a primitive's node is GLTF_NO_NODE.
 *              This is what the wall torch and the quarry have always been,
 *              and it stays byte-for-byte what it was.
 *   posed   -- vertices stay in their own node's space, the hierarchy is kept,
 *              and the caller poses it every frame (see gltf_scene_rest_pose).
 *
 * Flattening an animated file would be losing exactly the information the
 * animation addresses, and keeping a hierarchy for a static one would make
 * every existing caller pay a matrix chain for an identity. */
typedef struct GltfScene {
    const char *source_path;
    GltfPrimitive *primitives;
    uint32_t primitive_count;
    GltfMaterial *materials;
    uint32_t material_count;
    LocalToWorldTransform placement;
    GltfNode *nodes;
    uint32_t node_count;
    /* Parents before children, so a world-matrix pass is one forward sweep.
     * glTF does not require the file to be ordered that way. */
    uint32_t *node_order;
    GltfClip *clips;
    uint32_t clip_count;
    GltfSkin *skins;
    uint32_t skin_count;
} GltfScene;

struct Renderer;
GltfLoadResult gltf_scene_create(struct Renderer *renderer, const char *path,
                                 const GltfLoadOptions *options, GltfScene *out,
                                 GltfLoadError *error);
/* CPU-only entry point: suitable for importer tests and tools. */
GltfLoadResult gltf_scene_parse(const char *path, const GltfLoadOptions *options,
                                GltfScene *out, GltfLoadError *error);
GltfLoadResult gltf_scene_upload(struct Renderer *renderer, GltfScene *scene,
                                 GltfLoadError *error);
void gltf_scene_destroy(struct Renderer *renderer, GltfScene *scene);

/* Lookups by name. Names are what an asset pipeline can promise across a
 * re-export; node and clip INDICES are not. Both return GLTF_NO_NODE when
 * there is no such name. */
uint32_t gltf_scene_find_node(const GltfScene *scene, const char *name);
uint32_t gltf_scene_find_clip(const GltfScene *scene, const char *name);

/* Posing is deliberately three separate steps rather than one "evaluate"
 * call:
 *
 *   1. gltf_scene_rest_pose fills `pose` with every node's authored transform.
 *   2. gltf_clip_sample writes ONLY the nodes a clip actually animates, so
 *      clips over disjoint node sets LAYER by being applied in turn -- a pin
 *      popping while the pick keeps jiggling is two clips, not a third clip
 *      authored for the combination. (The exporter strips channels that never
 *      leave rest, which is what makes the node sets disjoint.)
 *   3. gltf_scene_world_matrices chains `pose` down the hierarchy.
 *
 * Between 2 and 3 the caller owns `pose` and may edit it: that is where game
 * state that the animation does not cover goes -- which pin the pick is
 * riding, how far an unset pin has been raised -- without the importer needing
 * to know that any of those things exist.
 *
 * `pose` and `out` are both node_count long. */
void gltf_scene_rest_pose(const GltfScene *scene, GltfTransform *pose);
void gltf_clip_sample(const GltfScene *scene, uint32_t clip, float time, GltfTransform *pose);
void gltf_scene_world_matrices(const GltfScene *scene, const GltfTransform *pose, mat4s *out);

/* Linear-blend skinning on the CPU: deforms primitive `primitive`'s bind-pose
 * vertices by the joints' `world` matrices (from gltf_scene_world_matrices)
 * into `out` (vertex_count long), positions, normals and tangents. The result
 * is in the scene's root space -- a skinned mesh's own node transform is
 * ignored, as glTF specifies. Rigid primitives are copied through. */
void gltf_scene_skin(const GltfScene *scene, uint32_t primitive, const mat4s *world, Vertex *out);
/* One joint's skinning matrix (world * inverse bind): where a point given in
 * bind pose and attached to that joint ends up. GLMS identity if absent. */
mat4s gltf_scene_joint_matrix(const GltfScene *scene, uint32_t skin, uint32_t node,
                              const mat4s *world);
