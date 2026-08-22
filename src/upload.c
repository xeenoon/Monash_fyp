#include "upload.h"
#include "vk_common.h"

#include <string.h>

#define UPLOAD_SLOTS 3

typedef struct {
    VkCommandBuffer cmd;
    VkFence         fence;
    GpuBuffer       staging;
    VkDeviceSize    used;
    bool            open;       /* between upload_begin and upload_submit  */
    bool            submitted;  /* fence armed, awaiting completion         */
} UploadSlot;

struct UploadContext {
    VkDevice      device;
    GpuAllocator *allocator;
    VkQueue       queue;
    VkCommandPool pool;
    VkDeviceSize  staging_capacity;

    UploadSlot slots[UPLOAD_SLOTS];
    uint32_t   current;
};

static void slot_reset_fence(UploadContext *ctx, UploadSlot *slot) {
    if (slot->submitted) {
        VK_CHECK(vkWaitForFences(ctx->device, 1, &slot->fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(ctx->device, 1, &slot->fence));
        slot->submitted = false;
    }
}

static void slot_ensure_staging(UploadContext *ctx, UploadSlot *slot, VkDeviceSize needed) {
    if (slot->staging.size >= needed) return;
    /* Only grows when nothing is staged yet this batch, so no recorded copy can
       still reference the old buffer. */
    gpu_buffer_destroy(ctx->device, ctx->allocator, &slot->staging);
    slot->staging = gpu_buffer_create(ctx->device, ctx->allocator, needed,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

UploadContext *upload_context_create(VkDevice device, GpuAllocator *allocator,
                                     VkQueue queue, uint32_t queue_family,
                                     VkDeviceSize staging_capacity) {
    UploadContext *ctx = calloc(1, sizeof(*ctx));
    ctx->device = device;
    ctx->allocator = allocator;
    ctx->queue = queue;
    ctx->staging_capacity = staging_capacity;

    VkCommandPoolCreateInfo pool = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
               | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = queue_family };
    VK_CHECK(vkCreateCommandPool(device, &pool, NULL, &ctx->pool));

    for (uint32_t i = 0; i < UPLOAD_SLOTS; ++i) {
        VkCommandBufferAllocateInfo alloc = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = ctx->pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
        VK_CHECK(vkAllocateCommandBuffers(device, &alloc, &ctx->slots[i].cmd));
        VkFenceCreateInfo fence = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        VK_CHECK(vkCreateFence(device, &fence, NULL, &ctx->slots[i].fence));
        ctx->slots[i].staging = gpu_buffer_create(device, allocator, staging_capacity,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    return ctx;
}

void upload_context_destroy(UploadContext *ctx) {
    if (!ctx) return;
    upload_wait_idle(ctx);
    for (uint32_t i = 0; i < UPLOAD_SLOTS; ++i) {
        gpu_buffer_destroy(ctx->device, ctx->allocator, &ctx->slots[i].staging);
        vkDestroyFence(ctx->device, ctx->slots[i].fence, NULL);
    }
    vkDestroyCommandPool(ctx->device, ctx->pool, NULL);
    free(ctx);
}

void upload_begin(UploadContext *ctx) {
    UploadSlot *slot = &ctx->slots[ctx->current];
    slot_reset_fence(ctx, slot);
    slot->used = 0;
    slot->open = true;
    VK_CHECK(vkResetCommandBuffer(slot->cmd, 0));
    VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    VK_CHECK(vkBeginCommandBuffer(slot->cmd, &begin));
}

/* Reserve `size` bytes in the current slot's staging buffer and copy `data`
   into it. Auto-flushes to a fresh slot if the running batch would overflow. */
static VkDeviceSize stage(UploadContext *ctx, const void *data, VkDeviceSize size) {
    UploadSlot *slot = &ctx->slots[ctx->current];
    const VkDeviceSize alignment = 16;
    VkDeviceSize offset = (slot->used + alignment - 1) & ~(alignment - 1);

    if (offset + size > slot->staging.size) {
        if (slot->used == 0) {
            slot_ensure_staging(ctx, slot, size);
            offset = 0;
        } else {
            upload_submit(ctx);
            upload_begin(ctx);
            slot = &ctx->slots[ctx->current];
            slot_ensure_staging(ctx, slot, size);
            offset = 0;
        }
    }
    memcpy((char *)slot->staging.allocation.mapped + offset, data, (size_t)size);
    slot->used = offset + size;
    return offset;
}

void upload_buffer(UploadContext *ctx, GpuBuffer *dst, VkDeviceSize dst_offset,
                   const void *data, VkDeviceSize size) {
    VkDeviceSize src_offset = stage(ctx, data, size);
    UploadSlot *slot = &ctx->slots[ctx->current];
    VkBufferCopy copy = { .srcOffset = src_offset, .dstOffset = dst_offset, .size = size };
    vkCmdCopyBuffer(slot->cmd, slot->staging.buffer, dst->buffer, 1, &copy);
}

static void image_barrier(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
                          uint32_t base_mip, uint32_t mip_count,
                          VkImageLayout old_layout, VkImageLayout new_layout,
                          VkAccessFlags src_access, VkAccessFlags dst_access,
                          VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
    VkImageMemoryBarrier barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .oldLayout = old_layout, .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image, .subresourceRange = {aspect, base_mip, mip_count, 0, 1},
        .srcAccessMask = src_access, .dstAccessMask = dst_access };
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1, &barrier);
}

void upload_image(UploadContext *ctx, VkImage image, VkFormat format,
                  uint32_t width, uint32_t height, uint32_t mip_levels,
                  VkImageAspectFlags aspect, VkImageLayout final_layout,
                  const void *data, VkDeviceSize size) {
    (void)format;
    VkDeviceSize src_offset = stage(ctx, data, size);
    UploadSlot *slot = &ctx->slots[ctx->current];
    VkCommandBuffer cmd = slot->cmd;

    /* All levels -> TRANSFER_DST, then fill level 0 from staging. */
    image_barrier(cmd, image, aspect, 0, mip_levels,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        0, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy copy = { .bufferOffset = src_offset,
        .imageSubresource = {aspect, 0, 0, 1}, .imageExtent = {width, height, 1} };
    vkCmdCopyBufferToImage(cmd, slot->staging.buffer, image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    /* Generate the remaining chain by successive half-size blits. */
    int32_t mip_w = (int32_t)width, mip_h = (int32_t)height;
    for (uint32_t level = 1; level < mip_levels; ++level) {
        image_barrier(cmd, image, aspect, level - 1, 1,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        int32_t next_w = mip_w > 1 ? mip_w / 2 : 1;
        int32_t next_h = mip_h > 1 ? mip_h / 2 : 1;
        VkImageBlit blit = {
            .srcSubresource = {aspect, level - 1, 0, 1}, .srcOffsets = {{0,0,0}, {mip_w, mip_h, 1}},
            .dstSubresource = {aspect, level,     0, 1}, .dstOffsets = {{0,0,0}, {next_w, next_h, 1}} };
        vkCmdBlitImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
        mip_w = next_w; mip_h = next_h;
    }

    /* Transition everything to the final layout. Levels 0..n-2 are TRANSFER_SRC
       (or just level 0 when there is no chain); the last level is TRANSFER_DST. */
    if (mip_levels > 1)
        image_barrier(cmd, image, aspect, 0, mip_levels - 1,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, final_layout,
            VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    image_barrier(cmd, image, aspect, mip_levels - 1, 1,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, final_layout,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
}

void upload_submit(UploadContext *ctx) {
    UploadSlot *slot = &ctx->slots[ctx->current];
    if (!slot->open) return;

    /* Make transfer writes visible to later work on this (graphics) queue.
       Image copies already carry their own layout/availability barriers; this
       covers buffer copies, whose reads happen at vertex-input/shader stages.
       The barrier's second scope spans submission order, so a graphics submit
       issued after this one sees the uploads without any fence/idle wait. */
    VkMemoryBarrier barrier = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT
                       | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT };
    vkCmdPipelineBarrier(slot->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
        | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 1, &barrier, 0, NULL, 0, NULL);

    VK_CHECK(vkEndCommandBuffer(slot->cmd));
    VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &slot->cmd };
    VK_CHECK(vkQueueSubmit(ctx->queue, 1, &submit, slot->fence));
    slot->open = false;
    slot->submitted = true;
    ctx->current = (ctx->current + 1) % UPLOAD_SLOTS;
}

void upload_wait_idle(UploadContext *ctx) {
    for (uint32_t i = 0; i < UPLOAD_SLOTS; ++i)
        slot_reset_fence(ctx, &ctx->slots[i]);
}
