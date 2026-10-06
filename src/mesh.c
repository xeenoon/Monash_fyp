#include "mesh.h"
#include "renderer.h"
#include "upload.h"
#include "vk_common.h"

#include <stddef.h>

/* Create a device-local buffer and stage `src` into it through the renderer's
   reusable upload context. The batch is opened/submitted by the caller. */
static GpuBuffer device_buffer(struct Renderer *r, VkDeviceSize size, VkBufferUsageFlags usage,
							   const void *src)
{
	GpuBuffer buffer =
		gpu_buffer_create(r->device, r->allocator, size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
						  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	upload_buffer(r->upload, &buffer, 0, src, size);
	return buffer;
}

void mesh_upload(struct Renderer *r, Mesh *mesh)
{
	upload_begin(r->upload);
	mesh->vertex_buffer = device_buffer(r, sizeof(Vertex) * mesh->vertex_count,
										VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh->vertices);
	if (mesh->index_count)
		mesh->index_buffer = device_buffer(r, sizeof(uint32_t) * mesh->index_count,
										   VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mesh->indices);
	upload_submit(r->upload);

	if (mesh->texture_path)
	{
		bool pbr = mesh->orm_path && mesh->normal_path;
		mesh->material_set = renderer_acquire_file_material(
			r, mesh->texture_path, pbr ? mesh->orm_path : NULL,
			pbr ? mesh->normal_path : NULL, NULL, pbr, &mesh->texture, &mesh->orm,
			&mesh->normal_map, NULL);
		mesh->material_cached = true;
	}
}

void mesh_destroy(struct Renderer *r, Mesh *mesh)
{
	if (mesh->material_set && !mesh->material_cached)
		renderer_free_material_set(r, mesh->material_set);
	gpu_buffer_destroy(r->device, r->allocator, &mesh->vertex_buffer);
	if (mesh->index_count)
		gpu_buffer_destroy(r->device, r->allocator, &mesh->index_buffer);
	if (mesh->texture_path && !mesh->material_cached)
		texture_destroy(r->device, r->allocator, &mesh->texture);
	if (mesh->orm_path && !mesh->material_cached)
		texture_destroy(r->device, r->allocator, &mesh->orm);
	if (mesh->normal_path && !mesh->material_cached)
		texture_destroy(r->device, r->allocator, &mesh->normal_map);
}

void mesh_draw(VkCommandBuffer cmd, const Mesh *mesh)
{
	VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &mesh->vertex_buffer.buffer, &offset);
	if (mesh->index_count)
	{
		vkCmdBindIndexBuffer(cmd, mesh->index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);
		vkCmdDrawIndexed(cmd, mesh->index_count, 1, 0, 0, 0);
	}
	else
	{
		vkCmdDraw(cmd, mesh->vertex_count, 1, 0, 0);
	}
}

VkVertexInputBindingDescription mesh_binding_description(void)
{
	return (VkVertexInputBindingDescription){0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
}

const VkVertexInputAttributeDescription *mesh_attribute_descriptions(uint32_t *count)
{
	static const VkVertexInputAttributeDescription attributes[5] = {
		{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
		{1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
		{2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, texcoord)},
		{3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, tangent)},
		{4, 0, VK_FORMAT_R32_SFLOAT, offsetof(Vertex, untextured)}};
	*count = 5;
	return attributes;
}
