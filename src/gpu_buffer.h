#pragma once

#include <vulkan/vulkan.h>

#include "gpu_memory.h"

/* A VkBuffer plus the suballocated memory backing it. Every buffer in the
   renderer is created and destroyed through this wrapper so there is exactly
   one ownership path: gpu_buffer_create pairs with gpu_buffer_destroy, and the
   memory range always returns to the allocator. */
typedef struct {
    VkBuffer      buffer;
    GpuAllocation allocation;
    VkDeviceSize  size;
} GpuBuffer;

/* Create a buffer of `size` bytes with `usage`, backed by memory satisfying
   `properties`. Host-visible buffers expose their mapping via
   allocation.mapped; device-local buffers do not. */
GpuBuffer gpu_buffer_create(VkDevice device, GpuAllocator *allocator, VkDeviceSize size,
                            VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
void      gpu_buffer_destroy(VkDevice device, GpuAllocator *allocator, GpuBuffer *buffer);
