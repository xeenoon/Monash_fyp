#include "mesh.h"
#include "renderer.h"
#include "vk_common.h"

#include <stddef.h>
#include <string.h>

/* Allocate a host-visible buffer and copy `src` into it. */
static void upload_buffer(struct Renderer *r, VkDeviceSize size, VkBufferUsageFlags usage,
                          const void *src, VkBuffer *buffer, VkDeviceMemory *memory) {
    renderer_create_buffer(r, size, usage,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        buffer, memory);
    void *mapped;
    VK_CHECK(vkMapMemory(r->device, *memory, 0, size, 0, &mapped));
    memcpy(mapped, src, (size_t)size);
    vkUnmapMemory(r->device, *memory);
}

void mesh_upload(struct Renderer *r, Mesh *mesh) {
    upload_buffer(r, sizeof(Vertex) * mesh->vertex_count, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                  mesh->vertices, &mesh->buffer, &mesh->memory);
    if (mesh->index_count)
        upload_buffer(r, sizeof(uint32_t) * mesh->index_count, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                      mesh->indices, &mesh->index_buffer, &mesh->index_memory);
}

void mesh_destroy(struct Renderer *r, Mesh *mesh) {
    vkDestroyBuffer(r->device, mesh->buffer, NULL);
    vkFreeMemory(r->device, mesh->memory, NULL);
    if (mesh->index_count) {
        vkDestroyBuffer(r->device, mesh->index_buffer, NULL);
        vkFreeMemory(r->device, mesh->index_memory, NULL);
    }
}

void mesh_draw(VkCommandBuffer cmd, const Mesh *mesh) {
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &mesh->buffer, &offset);
    if (mesh->index_count) {
        vkCmdBindIndexBuffer(cmd, mesh->index_buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, mesh->index_count, 1, 0, 0, 0);
    } else {
        vkCmdDraw(cmd, mesh->vertex_count, 1, 0, 0);
    }
}

VkVertexInputBindingDescription mesh_binding_description(void) {
    return (VkVertexInputBindingDescription){0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
}

const VkVertexInputAttributeDescription *mesh_attribute_descriptions(uint32_t *count) {
    static const VkVertexInputAttributeDescription attributes[2] = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)}
    };
    *count = 2;
    return attributes;
}
