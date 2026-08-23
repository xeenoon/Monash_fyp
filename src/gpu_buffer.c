#include "gpu_buffer.h"
#include "vk_common.h"

GpuBuffer gpu_buffer_create(VkDevice device, GpuAllocator *allocator, VkDeviceSize size,
							VkBufferUsageFlags usage, VkMemoryPropertyFlags properties)
{
	GpuBuffer buffer = {.size = size};
	VkBufferCreateInfo info = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
							   .size = size,
							   .usage = usage,
							   .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
	VK_CHECK(vkCreateBuffer(device, &info, NULL, &buffer.buffer));

	VkMemoryRequirements requirements;
	vkGetBufferMemoryRequirements(device, buffer.buffer, &requirements);
	/* Buffers are always linearly tiled, so they share the allocator's linear
	   block class (kept apart from optimally-tiled images). */
	buffer.allocation = gpu_alloc(allocator, requirements, properties, true);
	VK_CHECK(vkBindBufferMemory(device, buffer.buffer, buffer.allocation.memory,
								buffer.allocation.offset));
	return buffer;
}

void gpu_buffer_destroy(VkDevice device, GpuAllocator *allocator, GpuBuffer *buffer)
{
	if (buffer->buffer == VK_NULL_HANDLE)
		return;
	vkDestroyBuffer(device, buffer->buffer, NULL);
	gpu_free(allocator, buffer->allocation);
	*buffer = (GpuBuffer){0};
}
