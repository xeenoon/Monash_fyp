#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

#include "gpu_memory.h"

struct UploadContext;

/* General GPU texture: image, its suballocation, view, optional sampler, and
   the descriptive state a renderer needs to reason about it (format, extent,
   mip count, layers, usage, current layout, aspect). One create call pairs with
   one texture_destroy. Sampling imagery, single-channel elevation, HDR colour
   targets, and depth targets are all built from the same struct via the helpers
   below. */
typedef struct
{
	VkImage image;
	GpuAllocation allocation;
	VkImageView view;
	VkSampler sampler; /* VK_NULL_HANDLE for non-sampled targets */
	VkFormat format;
	VkExtent2D extent;
	uint32_t mip_levels;
	uint32_t array_layers;
	VkImageUsageFlags usage;
	VkImageAspectFlags aspect;
	VkImageLayout layout; /* current layout, updated by uploads     */
} Texture;

/* Sampler + image creation policy. Zeroed fields take sensible defaults:
   mip_levels 0 means "full chain from extent", array_layers 0 means 1,
   filter 0 is NEAREST so callers usually set LINEAR, max_anisotropy 0 disables
   anisotropy, and compare_enable creates a depth-comparison sampler using
   compare_op. Set create_sampler=false for render-only attachments. */
typedef struct
{
	VkFormat format;
	uint32_t width, height;
	uint32_t mip_levels;
	uint32_t array_layers;
	VkImageUsageFlags usage;
	VkImageAspectFlags aspect;
	VkFilter filter;
	VkSamplerAddressMode address_mode;
	float max_anisotropy;
	bool create_sampler;
	bool compare_enable;
	VkCompareOp compare_op;
	/* Cube-compatible image with a VK_IMAGE_VIEW_TYPE_CUBE sampled view.
	   Forces array_layers to 6 regardless of the field above. */
	bool cube;
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
/* Identical decoding/upload path but a linear UNORM image for data textures
   such as metallic/roughness/AO and tangent-space normals. */
void texture_load_linear(VkDevice device, GpuAllocator *allocator, struct UploadContext *upload,
						 Texture *t, const char *path, float max_anisotropy);
/* Linear image loading with a capped mip chain.  Use this for atlases whose
   per-cell gutters are valid only through a fixed mip level; generating the
   remaining global mips would blend neighbouring atlas cells. */
void texture_load_linear_mip_limited(VkDevice device, GpuAllocator *allocator,
								 struct UploadContext *upload, Texture *t, const char *path,
								 float max_anisotropy, uint32_t max_mip_levels);

/* 1x1 opaque-white sRGB texture used as the fallback for untextured meshes. */
void texture_create_white(VkDevice device, GpuAllocator *allocator, struct UploadContext *upload,
						  Texture *t);
/* Programmatic 1x1 RGBA material texel; sRGB is selected explicitly. */
void texture_create_solid_rgba8(VkDevice device, GpuAllocator *allocator, struct UploadContext *upload,
								Texture *t, const uint8_t rgba[4], bool srgb);

/* Sampled single-channel R16_UNORM elevation texture, uploaded from `heights`
   (width*height uint16 texels). Address mode clamps to edge; no mips. */
void texture_create_elevation(VkDevice device, GpuAllocator *allocator,
							  struct UploadContext *upload, Texture *t, const uint16_t *heights,
							  uint32_t width, uint32_t height);

/* Shared linear RGBA detail map: tangent normal in RG and macro noise in BA. */
void texture_create_terrain_detail(VkDevice device, GpuAllocator *allocator,
								   struct UploadContext *upload, Texture *t, float max_anisotropy);

/* HDR (RGBA16_SFLOAT) colour target, usable as colour attachment and sampled. */
Texture texture_create_hdr_target(VkDevice device, GpuAllocator *allocator, uint32_t width,
								  uint32_t height);

/* Phase-8 full-resolution temporal attachments. Colour/history use RGBA16F;
   motion uses RG16F and history depth uses a colour-sampleable R32F target. */
Texture texture_create_motion_target(VkDevice device, GpuAllocator *allocator, uint32_t width,
									 uint32_t height);
Texture texture_create_history_depth_target(VkDevice device, GpuAllocator *allocator,
											uint32_t width, uint32_t height);

/* Depth target with the given format, usable as depth attachment. */
Texture texture_create_depth_target(VkDevice device, GpuAllocator *allocator, VkFormat format,
									uint32_t width, uint32_t height);

/* Sampleable scene depth for the depth-clamped atmosphere composite. */
Texture texture_create_sampled_depth_target(VkDevice device, GpuAllocator *allocator,
											VkFormat format, uint32_t width, uint32_t height);

/* Linear RGBA16F storage/sampled LUT. Layers > 1 form a 2D-array volume so the
   same general texture abstraction can hold the aerial-perspective froxels. */
Texture texture_create_atmosphere_lut(VkDevice device, GpuAllocator *allocator, uint32_t width,
									  uint32_t height, uint32_t layers);

/* Sampleable depth-array target used by cascaded directional shadows. */
Texture texture_create_shadow_array(VkDevice device, GpuAllocator *allocator, VkFormat format,
									uint32_t resolution, uint32_t layers);

/* RGBA16F cube with a full mip chain, storage + sampled usage, seamless
   (clamp-to-edge) filtering. Compute fills mip 0 (equirect->cube) and prefilters
   mips 1..N as increasing roughness (see environment.c / renderer.c B2). */
Texture texture_create_environment_cube(VkDevice device, GpuAllocator *allocator,
										uint32_t face_size);

/* A cube's sampled view cannot be bound as a storage image. This is a
   VK_IMAGE_VIEW_TYPE_2D_ARRAY (6 layers) view of exactly one mip level, usable
   as a compute storage-image target when writing that mip's six faces. Caller
   owns and destroys the returned view (not tracked by the Texture). */
VkImageView texture_create_storage_mip_view(VkDevice device, const Texture *t, uint32_t mip);

/* A single mip level of a 2D texture as a VK_IMAGE_VIEW_TYPE_2D view, usable
   as a compute storage-image target. The Texture's own view spans the whole
   chain, which is what a sampler wants and what a storage binding cannot use.
   Caller owns and destroys the returned view (not tracked by the Texture). */
VkImageView texture_create_2d_mip_view(VkDevice device, const Texture *t, uint32_t mip);

/* A VK_IMAGE_VIEW_TYPE_CUBE view restricted to [base_mip, base_mip+mip_count).
   Used to sample a specific mip range (e.g. only the freshly-written mip 0)
   while other mips of the same image are still being written. Caller owns and
   destroys the returned view. */
VkImageView texture_create_cube_view(VkDevice device, const Texture *t, uint32_t base_mip,
									 uint32_t mip_count);

void texture_destroy(VkDevice device, GpuAllocator *allocator, Texture *t);
