// Minimal headless Vulkan compute harness for the offline terrain baker.
//
// This is deliberately independent of renderer.c: it needs no surface,
// swapchain, or graphics pipeline — just an instance, a device with one
// compute queue, and helpers to move buffers in/out and dispatch a compute
// shader.  It is the foundation the terrain-synthesis compute stages plug into,
// each validated against the Python CPU pipeline as the reference oracle.
#ifndef BAKE_GPU_H
#define BAKE_GPU_H

#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

typedef struct {
    VkInstance instance;
    VkPhysicalDevice physical;
    VkDevice device;
    uint32_t queue_family;
    VkQueue queue;
    VkCommandPool command_pool;
    VkPhysicalDeviceMemoryProperties memory_properties;
    char device_name[256];
} BakeGpu;

typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize size;
    void *mapped;  // non-NULL: host-visible + coherent, safe to read/write
} BakeBuffer;

typedef struct {
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;
    VkShaderModule module;
    uint32_t binding_count;
    uint32_t push_size;
} BakePipeline;

// Bring up instance + compute device.  Returns false (with a message on stderr)
// if no compute-capable Vulkan device is available.
bool bake_gpu_init(BakeGpu *gpu);
void bake_gpu_destroy(BakeGpu *gpu);

// A host-visible, host-coherent storage buffer: map once, read/write via .mapped.
BakeBuffer bake_buffer_host(BakeGpu *gpu, VkDeviceSize size);
void bake_buffer_destroy(BakeGpu *gpu, BakeBuffer *buffer);

// Compile a compute pipeline from a .spv file with `binding_count` std430
// storage buffers (bindings 0..n-1) and `push_size` bytes of push constants.
BakePipeline bake_pipeline_create(BakeGpu *gpu, const char *spv_path,
                                  uint32_t binding_count, uint32_t push_size);
void bake_pipeline_destroy(BakeGpu *gpu, BakePipeline *pipeline);

// Bind buffers[0..count-1] to bindings 0..count-1, push `push_size` bytes, and
// dispatch (groups_x, groups_y, groups_z), blocking until the GPU is done.
void bake_dispatch(BakeGpu *gpu, BakePipeline *pipeline,
                   const BakeBuffer *buffers, uint32_t count,
                   const void *push, uint32_t push_size,
                   uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);

#endif  // BAKE_GPU_H
