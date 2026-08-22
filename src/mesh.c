#include "mesh.h"
#include "renderer.h"
#include "vk_common.h"

#include <stddef.h>
#include <string.h>

void mesh_upload(struct Renderer *r, Mesh *mesh) {
    VkDeviceSize size = sizeof(Vertex) * mesh->vertex_count;
    renderer_create_buffer(r, size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &mesh->buffer, &mesh->memory);
    void *mapped;
    VK_CHECK(vkMapMemory(r->device, mesh->memory, 0, size, 0, &mapped));
    memcpy(mapped, mesh->vertices, (size_t)size);
    vkUnmapMemory(r->device, mesh->memory);
}

void mesh_destroy(struct Renderer *r, Mesh *mesh) {
    vkDestroyBuffer(r->device, mesh->buffer, NULL);
    vkFreeMemory(r->device, mesh->memory, NULL);
}

void mesh_draw(VkCommandBuffer cmd, const Mesh *mesh) {
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &mesh->buffer, &offset);
    vkCmdDraw(cmd, mesh->vertex_count, 1, 0, 0);
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
