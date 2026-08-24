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

typedef struct {
    Mesh mesh;
    Vertex *vertices;
    uint32_t *indices;
    uint32_t material_index;
} GltfPrimitive;

typedef struct {
    float base_color_factor[4];
    float metallic_factor, roughness_factor, normal_scale, occlusion_strength;
    bool use_metallic_roughness_red_as_occlusion;
    VkDescriptorSet descriptor_set;
    char *base_color_path, *metallic_roughness_path, *normal_path, *occlusion_path;
    Texture base_color, metallic_roughness, normal, occlusion;
} GltfMaterial;

typedef struct GltfScene {
    const char *source_path;
    GltfPrimitive *primitives;
    uint32_t primitive_count;
    GltfMaterial *materials;
    uint32_t material_count;
    LocalToWorldTransform placement;
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
