#pragma once

#include <vulkan/vulkan.h>
#include <stdint.h>

#include "texture.h"

typedef struct { float position[3]; float normal[3]; float texcoord[2]; } Vertex;

/* Base "class". Every primitive embeds this as its FIRST member, so a
   Cube* casts cleanly to Mesh* — the well-defined version of the
   sockaddr / sockaddr_in trick (C11 6.7.2.1p15). */
typedef struct {
    const Vertex   *vertices;     /* CPU geometry (may point at static data) */
    uint32_t        vertex_count;
    VkBuffer        buffer;       /* GPU copy, filled by mesh_upload         */
    VkDeviceMemory  memory;

    /* Optional index buffer. Leave indices=NULL / index_count=0 for a plain
       non-indexed draw; set both to have mesh_upload build a GPU index buffer
       and mesh_draw switch to vkCmdDrawIndexed. */
    const uint32_t *indices;      /* CPU indices (may point at static data)  */
    uint32_t        index_count;
    VkBuffer        index_buffer;
    VkDeviceMemory  index_memory;

    /* Optional albedo texture. A mesh implementation sets texture_path to its
       own image file; mesh_upload then loads it and allocates descriptor_set.
       Leave texture_path NULL to draw untextured (renderer binds a fallback). */
    const char     *texture_path;
    Texture         texture;
    VkDescriptorSet descriptor_set;
} Mesh;

struct Renderer;

/* Uploads mesh->vertices to a GPU buffer, filling buffer/memory. */
void mesh_upload (struct Renderer *r, Mesh *mesh);
void mesh_destroy(struct Renderer *r, Mesh *mesh);
void mesh_draw   (VkCommandBuffer cmd, const Mesh *mesh);

/* Vertex layout, consumed by the pipeline in renderer.c */
VkVertexInputBindingDescription          mesh_binding_description(void);
const VkVertexInputAttributeDescription *mesh_attribute_descriptions(uint32_t *count);
