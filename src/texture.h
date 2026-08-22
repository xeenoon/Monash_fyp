#pragma once

#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdint.h>

#include "gpu_memory.h"

struct UploadContext;

/* General GPU texture: image, its suballocation, view, optional sampler, and
   the descriptive state a renderer needs to reason about it (format, extent,
   mip count, layers, usage, current layout, aspect). One create call pairs with
   one texture_destroy. Sampling imagery, single-channel elevation, HDR colour
   targets, and depth targets are all built from the same struct via the helpers
   below. */
typedef struct {
    VkImage            image;
    GpuAllocation      allocation;
    VkImageView        view;
    VkSampler          sampler;      /* VK_NULL_HANDLE for non-sampled targets */
    VkFormat           format;
    VkExtent2D         extent;
    uint32_t           mip_levels;
    uint32_t           array_layers;
    VkImageUsageFlags  usage;
    VkImageAspectFlags aspect;
    VkImageLayout      layout;       /* current layout, updated by uploads     */
} Texture;

/* Sampler + image creation policy. Zeroed fields take sensible defaults:
   mip_levels 0 means "full chain from extent", array_layers 0 means 1,
   filter 0 is NEAREST so callers usually set LINEAR, max_anisotropy 0 disables
   anisotropy, and compare_enable creates a depth-comparison sampler using
   compare_op. Set create_sampler=false for render-only attachments. */
typedef struct {
    VkFormat             format;
    uint32_t             width, height;
    uint32_t             mip_levels;
    uint32_t             array_layers;
    VkImageUsageFlags    usage;
    VkImageAspectFlags   aspect;
    VkFilter             filter;
    VkSamplerAddressMode address_mode;
    float                max_anisotropy;
    bool                 create_sampler;
    bool                 compare_enable;
    VkCompareOp          compare_op;
} TextureDesc;

/* Create image + view (+ optional sampler). No pixel data; layout is UNDEFINED. */
Texture texture_create(VkDevice device, GpuAllocator *allocator, const TextureDesc *desc);

/* Number of mip levels for a full chain covering width x height. */
uint32_t texture_mip_levels(uint32_t width, uint32_t height);

/* Decode an image file into an sRGB RGBA texture with a full mip chain and
   upload it through `upload`. `max_anisotropy` is clamped by the caller to the
   device limit (0 disables). */
void texture_load(VkDevice device, GpuAllocator *allocator, struct UploadContext *upload,
                  Texture *t, const char *path, float max_anisotropy);

/* 1x1 opaque-white sRGB texture used as the fallback for untextured meshes. */
void texture_create_white(VkDevice device, GpuAllocator *allocator,
                          struct UploadContext *upload, Texture *t);

/* Sampled single-channel R16_UNORM elevation texture, uploaded from `heights`
   (width*height uint16 texels). Address mode clamps to edge; no mips. */
void texture_create_elevation(VkDevice device, GpuAllocator *allocator, struct UploadContext *upload,
                              Texture *t, const uint16_t *heights, uint32_t width, uint32_t height);

/* Shared linear RGBA detail map: tangent normal in RG and macro noise in BA. */
void texture_create_terrain_detail(VkDevice device, GpuAllocator *allocator,
                                   struct UploadContext *upload, Texture *t,
                                   float max_anisotropy);

/* HDR (RGBA16_SFLOAT) colour target, usable as colour attachment and sampled. */
Texture texture_create_hdr_target(VkDevice device, GpuAllocator *allocator,
                                  uint32_t width, uint32_t height);

/* Depth target with the given format, usable as depth attachment. */
Texture texture_create_depth_target(VkDevice device, GpuAllocator *allocator,
                                    VkFormat format, uint32_t width, uint32_t height);

/* Sampleable depth-array target used by cascaded directional shadows. */
Texture texture_create_shadow_array(VkDevice device, GpuAllocator *allocator,
                                    VkFormat format, uint32_t resolution,
                                    uint32_t layers);

void texture_destroy(VkDevice device, GpuAllocator *allocator, Texture *t);
