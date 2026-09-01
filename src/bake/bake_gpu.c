#include "bake_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "file_utils.h"

#define BAKE_CHECK(expr)                                                       \
    do {                                                                       \
        VkResult _r = (expr);                                                  \
        if (_r != VK_SUCCESS) {                                                \
            fprintf(stderr, "bake_gpu: %s failed (VkResult %d) at %s:%d\n",    \
                    #expr, (int)_r, __FILE__, __LINE__);                       \
            abort();                                                           \
        }                                                                      \
    } while (0)

bool bake_gpu_init(BakeGpu *gpu) {
    memset(gpu, 0, sizeof(*gpu));

    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .pApplicationName = "terrain_bake",
                             .apiVersion = VK_API_VERSION_1_2};
    VkInstanceCreateInfo instance_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    if (vkCreateInstance(&instance_info, NULL, &gpu->instance) != VK_SUCCESS) {
        fprintf(stderr, "bake_gpu: vkCreateInstance failed\n");
        return false;
    }

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(gpu->instance, &device_count, NULL);
    if (device_count == 0) {
        fprintf(stderr, "bake_gpu: no Vulkan physical devices\n");
        return false;
    }
    VkPhysicalDevice *devices = calloc(device_count, sizeof(*devices));
    vkEnumeratePhysicalDevices(gpu->instance, &device_count, devices);

    // Pick the first device exposing a compute-capable queue family.
    bool found = false;
    for (uint32_t i = 0; i < device_count && !found; ++i) {
        uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &family_count, NULL);
        VkQueueFamilyProperties *families = calloc(family_count, sizeof(*families));
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &family_count, families);
        for (uint32_t f = 0; f < family_count; ++f) {
            if (families[f].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                gpu->physical = devices[i];
                gpu->queue_family = f;
                found = true;
                break;
            }
        }
        free(families);
    }
    free(devices);
    if (!found) {
        fprintf(stderr, "bake_gpu: no compute queue family found\n");
        return false;
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(gpu->physical, &props);
    snprintf(gpu->device_name, sizeof(gpu->device_name), "%s", props.deviceName);
    vkGetPhysicalDeviceMemoryProperties(gpu->physical, &gpu->memory_properties);

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = gpu->queue_family,
        .queueCount = 1,
        .pQueuePriorities = &priority};
    VkDeviceCreateInfo device_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info};
    BAKE_CHECK(vkCreateDevice(gpu->physical, &device_info, NULL, &gpu->device));
    vkGetDeviceQueue(gpu->device, gpu->queue_family, 0, &gpu->queue);

    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = gpu->queue_family};
    BAKE_CHECK(vkCreateCommandPool(gpu->device, &pool_info, NULL, &gpu->command_pool));
    return true;
}

void bake_gpu_destroy(BakeGpu *gpu) {
    if (gpu->command_pool) vkDestroyCommandPool(gpu->device, gpu->command_pool, NULL);
    if (gpu->device) vkDestroyDevice(gpu->device, NULL);
    if (gpu->instance) vkDestroyInstance(gpu->instance, NULL);
    memset(gpu, 0, sizeof(*gpu));
}

static uint32_t find_memory_type(const BakeGpu *gpu, uint32_t type_bits,
                                 VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < gpu->memory_properties.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) &&
            (gpu->memory_properties.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    fprintf(stderr, "bake_gpu: no memory type with flags 0x%x\n", want);
    abort();
}

BakeBuffer bake_buffer_host(BakeGpu *gpu, VkDeviceSize size) {
    BakeBuffer buffer = {.size = size};
    VkBufferCreateInfo info = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                               .size = size,
                               .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    BAKE_CHECK(vkCreateBuffer(gpu->device, &info, NULL, &buffer.buffer));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(gpu->device, buffer.buffer, &req);
    VkMemoryAllocateInfo alloc = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = find_memory_type(
            gpu, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    BAKE_CHECK(vkAllocateMemory(gpu->device, &alloc, NULL, &buffer.memory));
    BAKE_CHECK(vkBindBufferMemory(gpu->device, buffer.buffer, buffer.memory, 0));
    BAKE_CHECK(vkMapMemory(gpu->device, buffer.memory, 0, size, 0, &buffer.mapped));
    return buffer;
}

void bake_buffer_destroy(BakeGpu *gpu, BakeBuffer *buffer) {
    if (buffer->memory) vkUnmapMemory(gpu->device, buffer->memory);
    if (buffer->buffer) vkDestroyBuffer(gpu->device, buffer->buffer, NULL);
    if (buffer->memory) vkFreeMemory(gpu->device, buffer->memory, NULL);
    memset(buffer, 0, sizeof(*buffer));
}

BakePipeline bake_pipeline_create(BakeGpu *gpu, const char *spv_path,
                                  uint32_t binding_count, uint32_t push_size) {
    BakePipeline pipeline = {.binding_count = binding_count, .push_size = push_size};

    uint8_t *code = NULL;
    size_t code_size = 0;
    FileReadResult read = file_read_all(spv_path, &code, &code_size);
    if (read != FILE_READ_OK) {
        fprintf(stderr, "bake_gpu: cannot read SPIR-V '%s': %s\n", spv_path,
                file_read_result_string(read));
        abort();
    }
    VkShaderModuleCreateInfo module_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code_size,
        .pCode = (const uint32_t *)code};
    BAKE_CHECK(vkCreateShaderModule(gpu->device, &module_info, NULL, &pipeline.module));
    free(code);

    VkDescriptorSetLayoutBinding *bindings =
        calloc(binding_count, sizeof(*bindings));
    for (uint32_t i = 0; i < binding_count; ++i) {
        bindings[i] = (VkDescriptorSetLayoutBinding){
            .binding = i,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    }
    VkDescriptorSetLayoutCreateInfo set_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = binding_count,
        .pBindings = bindings};
    BAKE_CHECK(vkCreateDescriptorSetLayout(gpu->device, &set_info, NULL,
                                           &pipeline.set_layout));
    free(bindings);

    VkPushConstantRange push_range = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                      .offset = 0,
                                      .size = push_size};
    VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &pipeline.set_layout,
        .pushConstantRangeCount = push_size ? 1u : 0u,
        .pPushConstantRanges = push_size ? &push_range : NULL};
    BAKE_CHECK(vkCreatePipelineLayout(gpu->device, &layout_info, NULL,
                                      &pipeline.pipeline_layout));

    VkComputePipelineCreateInfo compute_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = pipeline.module,
                  .pName = "main"},
        .layout = pipeline.pipeline_layout};
    BAKE_CHECK(vkCreateComputePipelines(gpu->device, VK_NULL_HANDLE, 1,
                                        &compute_info, NULL, &pipeline.pipeline));
    return pipeline;
}

void bake_pipeline_destroy(BakeGpu *gpu, BakePipeline *pipeline) {
    if (pipeline->pipeline) vkDestroyPipeline(gpu->device, pipeline->pipeline, NULL);
    if (pipeline->pipeline_layout)
        vkDestroyPipelineLayout(gpu->device, pipeline->pipeline_layout, NULL);
    if (pipeline->set_layout)
        vkDestroyDescriptorSetLayout(gpu->device, pipeline->set_layout, NULL);
    if (pipeline->module) vkDestroyShaderModule(gpu->device, pipeline->module, NULL);
    memset(pipeline, 0, sizeof(*pipeline));
}

void bake_dispatch(BakeGpu *gpu, BakePipeline *pipeline,
                   const BakeBuffer *buffers, uint32_t count,
                   const void *push, uint32_t push_size,
                   uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    VkDescriptorPoolSize pool_size = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      .descriptorCount = count};
    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size};
    VkDescriptorPool pool;
    BAKE_CHECK(vkCreateDescriptorPool(gpu->device, &pool_info, NULL, &pool));

    VkDescriptorSetAllocateInfo set_alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &pipeline->set_layout};
    VkDescriptorSet set;
    BAKE_CHECK(vkAllocateDescriptorSets(gpu->device, &set_alloc, &set));

    VkDescriptorBufferInfo *infos = calloc(count, sizeof(*infos));
    VkWriteDescriptorSet *writes = calloc(count, sizeof(*writes));
    for (uint32_t i = 0; i < count; ++i) {
        infos[i] = (VkDescriptorBufferInfo){
            .buffer = buffers[i].buffer, .offset = 0, .range = VK_WHOLE_SIZE};
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[i]};
    }
    vkUpdateDescriptorSets(gpu->device, count, writes, 0, NULL);
    free(writes);
    free(infos);

    VkCommandBufferAllocateInfo cmd_alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = gpu->command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1};
    VkCommandBuffer cmd;
    BAKE_CHECK(vkAllocateCommandBuffers(gpu->device, &cmd_alloc, &cmd));
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    BAKE_CHECK(vkBeginCommandBuffer(cmd, &begin));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline->pipeline_layout, 0, 1, &set, 0, NULL);
    if (push_size) {
        vkCmdPushConstants(cmd, pipeline->pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push);
    }
    vkCmdDispatch(cmd, groups_x, groups_y, groups_z);
    BAKE_CHECK(vkEndCommandBuffer(cmd));

    VkFenceCreateInfo fence_info = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    BAKE_CHECK(vkCreateFence(gpu->device, &fence_info, NULL, &fence));
    VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                           .commandBufferCount = 1,
                           .pCommandBuffers = &cmd};
    BAKE_CHECK(vkQueueSubmit(gpu->queue, 1, &submit, fence));
    BAKE_CHECK(vkWaitForFences(gpu->device, 1, &fence, VK_TRUE, UINT64_MAX));

    vkDestroyFence(gpu->device, fence, NULL);
    vkFreeCommandBuffers(gpu->device, gpu->command_pool, 1, &cmd);
    vkDestroyDescriptorPool(gpu->device, pool, NULL);
}
