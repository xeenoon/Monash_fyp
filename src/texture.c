#include "texture.h"
#include "renderer.h"
#include "vk_common.h"
#include "stb_image.h"

#include <stdint.h>
#include <string.h>

/* Upload width*height RGBA8 pixels into a sampled sRGB texture, filling `t`. */
static void create_from_pixels(struct Renderer *r, Texture *t,
                               const void *pixels, uint32_t width, uint32_t height) {
    VkDeviceSize size = (VkDeviceSize)width * height * 4;

    VkBuffer staging; VkDeviceMemory staging_memory;
    renderer_create_buffer(r, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &staging, &staging_memory);
    void *mapped;
    VK_CHECK(vkMapMemory(r->device, staging_memory, 0, size, 0, &mapped));
    memcpy(mapped, pixels, (size_t)size);
    vkUnmapMemory(r->device, staging_memory);

    VkImageCreateInfo image_info = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_SRGB,
        .extent = {width, height, 1}, .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
    VK_CHECK(vkCreateImage(r->device, &image_info, NULL, &t->image));
    VkMemoryRequirements requirements;
    vkGetImageMemoryRequirements(r->device, t->image, &requirements);
    VkMemoryAllocateInfo allocation = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = requirements.size, .memoryTypeIndex = renderer_find_memory_type(r,
        requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) };
    VK_CHECK(vkAllocateMemory(r->device, &allocation, NULL, &t->memory));
    VK_CHECK(vkBindImageMemory(r->device, t->image, t->memory, 0));

    /* UNDEFINED -> TRANSFER_DST, copy, TRANSFER_DST -> SHADER_READ_ONLY. */
    VkCommandBuffer command = renderer_begin_single_time(r);
    VkImageMemoryBarrier barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = t->image, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        .srcAccessMask = 0, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT };
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, NULL, 0, NULL, 1, &barrier);
    VkBufferImageCopy copy = { .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageExtent = {width, height, 1} };
    vkCmdCopyBufferToImage(command, staging, t->image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, NULL, 0, NULL, 1, &barrier);
    renderer_end_single_time(r, command);
    vkDestroyBuffer(r->device, staging, NULL);
    vkFreeMemory(r->device, staging_memory, NULL);

    VkImageViewCreateInfo view = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = t->image, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_SRGB,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1} };
    VK_CHECK(vkCreateImageView(r->device, &view, NULL, &t->view));
    VkSamplerCreateInfo sampler = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR };
    VK_CHECK(vkCreateSampler(r->device, &sampler, NULL, &t->sampler));
}

void texture_load(struct Renderer *r, Texture *t, const char *path) {
    int width, height, channels;
    stbi_uc *pixels = stbi_load(path, &width, &height, &channels, STBI_rgb_alpha);
    if (!pixels) {
        fprintf(stderr, "Could not load texture %s: %s\n", path, stbi_failure_reason());
        exit(EXIT_FAILURE);
    }
    create_from_pixels(r, t, pixels, (uint32_t)width, (uint32_t)height);
    stbi_image_free(pixels);
}

void texture_create_white(struct Renderer *r, Texture *t) {
    const uint8_t white[4] = {255, 255, 255, 255};
    create_from_pixels(r, t, white, 1, 1);
}

void texture_destroy(struct Renderer *r, Texture *t) {
    vkDestroySampler(r->device, t->sampler, NULL);
    vkDestroyImageView(r->device, t->view, NULL);
    vkDestroyImage(r->device, t->image, NULL);
    vkFreeMemory(r->device, t->memory, NULL);
}
