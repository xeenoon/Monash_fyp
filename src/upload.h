#pragma once

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdint.h>

#include "gpu_buffer.h"

/* Reusable staging/upload path.

   The old code allocated a throwaway command buffer and blocked the whole queue
   (vkQueueWaitIdle) for every single resource. This context instead keeps a
   small ring of transfer slots, each owning a persistent staging buffer, a
   command buffer, and a fence. Many resources batch into one recording and one
   submit; the fence is waited on only when a slot is reused or when the caller
   explicitly drains, never as a device-wide idle.

   Usage:
       upload_begin(ctx);
       upload_buffer(ctx, &dst, offset, data, size);
       upload_image(ctx, image, ..., pixels, size);   // records transitions too
       upload_submit(ctx);                             // returns without waiting
       ...
       upload_wait_idle(ctx);   // block on outstanding transfers when required */

typedef struct UploadContext UploadContext;

typedef struct {
    uint32_t slot;
    uint64_t serial;
} UploadTicket;

UploadContext *upload_context_create(VkDevice device, GpuAllocator *allocator,
                                     VkQueue queue, uint32_t queue_family,
                                     VkDeviceSize staging_capacity);
void           upload_context_destroy(UploadContext *ctx);

/* Open a batch on the next ring slot, waiting only on that slot's own fence if
   it is still in flight. Safe to call again after upload_submit. */
void upload_begin(UploadContext *ctx);

/* Stage `data` and record a copy into an existing device-local buffer. The
   destination buffer must have been created with VK_BUFFER_USAGE_TRANSFER_DST_BIT. */
void upload_buffer(UploadContext *ctx, GpuBuffer *dst, VkDeviceSize dst_offset,
                   const void *data, VkDeviceSize size);

/* Stage `data`, copy it into mip level 0 of `image`, optionally generate the
   remaining mip chain by blitting, and leave every subresource in `final_layout`.
   The image must have been created with TRANSFER_DST (and TRANSFER_SRC when
   mip_levels > 1) usage. `pixel_size` is bytes-per-texel of the base level. */
void upload_image(UploadContext *ctx, VkImage image, VkFormat format,
                  uint32_t width, uint32_t height, uint32_t mip_levels,
                  VkImageAspectFlags aspect, VkImageLayout final_layout,
                  const void *data, VkDeviceSize size);

/* Close and submit the current batch. Does not wait for completion. */
UploadTicket upload_submit(UploadContext *ctx);

/* A ticket names the fence submission that contains an upload. If its ring
   slot has since been reused, reuse already waited for that submission. */
UploadTicket upload_last_ticket(const UploadContext *ctx);
bool         upload_complete(const UploadContext *ctx, UploadTicket ticket);

/* Block until every submitted transfer has completed (waits on slot fences,
   not the device). Normally graphics-queue ordering avoids this; use it for
   teardown or when CPU code must observe completion immediately. */
void upload_wait_idle(UploadContext *ctx);
