#include "texture.h"
#include "stb_image.h"
#include "surface_detail.h"
#include "upload.h"
#include "vk_common.h"

#include <stdlib.h>
#include <string.h>

uint32_t texture_mip_levels(uint32_t width, uint32_t height)
{
	uint32_t size = width > height ? width : height;
	uint32_t levels = 1;
	while (size > 1)
	{
		size >>= 1;
		++levels;
	}
	return levels;
}

Texture texture_create(VkDevice device, GpuAllocator *allocator, const TextureDesc *desc)
{
	uint32_t layers = desc->cube ? 6u : (desc->array_layers ? desc->array_layers : 1);
	uint32_t mips =
		desc->mip_levels ? desc->mip_levels : texture_mip_levels(desc->width, desc->height);
	VkImageAspectFlags aspect = desc->aspect ? desc->aspect : VK_IMAGE_ASPECT_COLOR_BIT;

	Texture t = {.format = desc->format,
				 .extent = {desc->width, desc->height},
				 .mip_levels = mips,
				 .array_layers = layers,
				 .usage = desc->usage,
				 .aspect = aspect,
				 .layout = VK_IMAGE_LAYOUT_UNDEFINED};

	VkImageCreateInfo image_info = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
									.flags = desc->cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0,
									.imageType = VK_IMAGE_TYPE_2D,
									.format = desc->format,
									.extent = {desc->width, desc->height, 1},
									.mipLevels = mips,
									.arrayLayers = layers,
									.samples = VK_SAMPLE_COUNT_1_BIT,
									.tiling = VK_IMAGE_TILING_OPTIMAL,
									.usage = desc->usage,
									.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
									.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
	VK_CHECK(vkCreateImage(device, &image_info, NULL, &t.image));

	VkMemoryRequirements requirements;
	vkGetImageMemoryRequirements(device, t.image, &requirements);
	/* Optimally-tiled image: non-linear allocator class. */
	t.allocation = gpu_alloc(allocator, requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
	VK_CHECK(vkBindImageMemory(device, t.image, t.allocation.memory, t.allocation.offset));

	VkImageViewCreateInfo view = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
								  .image = t.image,
								  .viewType = desc->cube ? VK_IMAGE_VIEW_TYPE_CUBE
										  : layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
													   : VK_IMAGE_VIEW_TYPE_2D,
								  .format = desc->format,
								  .subresourceRange = {aspect, 0, mips, 0, layers}};
	VK_CHECK(vkCreateImageView(device, &view, NULL, &t.view));

	if (desc->create_sampler)
	{
		VkSamplerCreateInfo sampler = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
									   .magFilter = desc->filter,
									   .minFilter = desc->filter,
									   .addressModeU = desc->address_mode,
									   .addressModeV = desc->address_mode,
									   .addressModeW = desc->address_mode,
									   .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
									   .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
									   .maxLod = (float)mips,
									   .anisotropyEnable =
										   desc->max_anisotropy > 1.0f ? VK_TRUE : VK_FALSE,
									   .maxAnisotropy = desc->max_anisotropy,
									   .compareEnable = desc->compare_enable,
									   .compareOp = desc->compare_op};
		VK_CHECK(vkCreateSampler(device, &sampler, NULL, &t.sampler));
	}
	return t;
}

/* Upload width*height texels of `bytes_per_texel`, generating mips if the
   texture was created with more than one level. Leaves it SHADER_READ_ONLY. */
static void upload_pixels(struct UploadContext *upload, Texture *t, const void *pixels,
						  uint32_t bytes_per_texel)
{
	VkDeviceSize size = (VkDeviceSize)t->extent.width * t->extent.height * bytes_per_texel;
	upload_begin(upload);
	upload_image(upload, t->image, t->format, t->extent.width, t->extent.height, t->mip_levels,
				 t->aspect, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, pixels, size);
	upload_submit(upload);
	t->layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

static Texture create_sampled(VkDevice device, GpuAllocator *allocator, VkFormat format,
							  uint32_t width, uint32_t height, uint32_t mips,
							  VkSamplerAddressMode address, float max_anisotropy)
{
	VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	if (mips > 1)
		usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT; /* blit source for mip-gen */
	TextureDesc desc = {.format = format,
						.width = width,
						.height = height,
						.mip_levels = mips,
						.usage = usage,
						.aspect = VK_IMAGE_ASPECT_COLOR_BIT,
						.filter = VK_FILTER_LINEAR,
						.address_mode = address,
						.max_anisotropy = max_anisotropy,
						.create_sampler = true};
	return texture_create(device, allocator, &desc);
}

static void texture_load_format(VkDevice device, GpuAllocator *allocator,
								struct UploadContext *upload, Texture *t, const char *path,
								float max_anisotropy, VkFormat format)
{
	int width, height, channels;
	stbi_uc *pixels = stbi_load(path, &width, &height, &channels, STBI_rgb_alpha);
	if (!pixels)
	{
		fprintf(stderr, "Could not load texture %s: %s\n", path, stbi_failure_reason());
		exit(EXIT_FAILURE);
	}
	uint32_t mips = texture_mip_levels((uint32_t)width, (uint32_t)height);
	*t = create_sampled(device, allocator, format, (uint32_t)width, (uint32_t)height, mips,
						VK_SAMPLER_ADDRESS_MODE_REPEAT, max_anisotropy);
	upload_pixels(upload, t, pixels, 4);
	stbi_image_free(pixels);
}

void texture_load(VkDevice device, GpuAllocator *allocator, struct UploadContext *upload,
				  Texture *t, const char *path, float max_anisotropy)
{
	texture_load_format(device, allocator, upload, t, path, max_anisotropy,
						VK_FORMAT_R8G8B8A8_SRGB);
}

void texture_load_linear(VkDevice device, GpuAllocator *allocator, struct UploadContext *upload,
						 Texture *t, const char *path, float max_anisotropy)
{
	texture_load_format(device, allocator, upload, t, path, max_anisotropy,
						VK_FORMAT_R8G8B8A8_UNORM);
}

void texture_create_white(VkDevice device, GpuAllocator *allocator, struct UploadContext *upload,
						  Texture *t)
{
	const uint8_t white[4] = {255, 255, 255, 255};
	*t = create_sampled(device, allocator, VK_FORMAT_R8G8B8A8_SRGB, 1, 1, 1,
						VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.0f);
	upload_pixels(upload, t, white, 4);
}

void texture_create_solid_rgba8(VkDevice device, GpuAllocator *allocator, struct UploadContext *upload,
								Texture *t, const uint8_t rgba[4], bool srgb)
{
	*t = create_sampled(device, allocator, srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM,
						1, 1, 1, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.0f);
	upload_pixels(upload, t, rgba, 4);
}

void texture_create_elevation(VkDevice device, GpuAllocator *allocator,
							  struct UploadContext *upload, Texture *t, const uint16_t *heights,
							  uint32_t width, uint32_t height)
{
	*t = create_sampled(device, allocator, VK_FORMAT_R16_UNORM, width, height, 1,
						VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0.0f);
	upload_pixels(upload, t, heights, 2);
}

void texture_create_terrain_detail(VkDevice device, GpuAllocator *allocator,
								   struct UploadContext *upload, Texture *t, float max_anisotropy)
{
	enum
	{
		DETAIL_SIZE = 128
	};
	uint8_t *pixels = malloc(DETAIL_SIZE * DETAIL_SIZE * 4u);
	if (!pixels || !surface_detail_generate_rgba8(pixels, DETAIL_SIZE, DETAIL_SIZE))
	{
		free(pixels);
		fprintf(stderr, "Could not generate terrain detail texture\n");
		exit(EXIT_FAILURE);
	}
	*t = create_sampled(device, allocator, VK_FORMAT_R8G8B8A8_UNORM, DETAIL_SIZE, DETAIL_SIZE,
						texture_mip_levels(DETAIL_SIZE, DETAIL_SIZE),
						VK_SAMPLER_ADDRESS_MODE_REPEAT, max_anisotropy);
	upload_pixels(upload, t, pixels, 4);
	free(pixels);
}

Texture texture_create_hdr_target(VkDevice device, GpuAllocator *allocator, uint32_t width,
								  uint32_t height)
{
	TextureDesc desc = {.format = VK_FORMAT_R16G16B16A16_SFLOAT,
						.width = width,
						.height = height,
						.mip_levels = 1,
						.aspect = VK_IMAGE_ASPECT_COLOR_BIT,
						.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
						.filter = VK_FILTER_LINEAR,
						.address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
						.create_sampler = true};
	return texture_create(device, allocator, &desc);
}

Texture texture_create_motion_target(VkDevice device, GpuAllocator *allocator, uint32_t width,
									 uint32_t height)
{
	TextureDesc desc = {.format = VK_FORMAT_R16G16_SFLOAT,
						.width = width,
						.height = height,
						.mip_levels = 1,
						.aspect = VK_IMAGE_ASPECT_COLOR_BIT,
						.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
						.filter = VK_FILTER_NEAREST,
						.address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
						.create_sampler = true};
	return texture_create(device, allocator, &desc);
}

Texture texture_create_history_depth_target(VkDevice device, GpuAllocator *allocator,
											uint32_t width, uint32_t height)
{
	TextureDesc desc = {.format = VK_FORMAT_R32_SFLOAT,
						.width = width,
						.height = height,
						.mip_levels = 1,
						.aspect = VK_IMAGE_ASPECT_COLOR_BIT,
						.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
						.filter = VK_FILTER_NEAREST,
						.address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
						.create_sampler = true};
	return texture_create(device, allocator, &desc);
}

Texture texture_create_depth_target(VkDevice device, GpuAllocator *allocator, VkFormat format,
									uint32_t width, uint32_t height)
{
	TextureDesc desc = {.format = format,
						.width = width,
						.height = height,
						.mip_levels = 1,
						.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
						.aspect = VK_IMAGE_ASPECT_DEPTH_BIT,
						.create_sampler = false};
	return texture_create(device, allocator, &desc);
}

Texture texture_create_sampled_depth_target(VkDevice device, GpuAllocator *allocator,
											VkFormat format, uint32_t width, uint32_t height)
{
	TextureDesc desc = {.format = format,
						.width = width,
						.height = height,
						.mip_levels = 1,
						.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
								 VK_IMAGE_USAGE_SAMPLED_BIT,
						.aspect = VK_IMAGE_ASPECT_DEPTH_BIT,
						.filter = VK_FILTER_NEAREST,
						.address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
						.create_sampler = true};
	return texture_create(device, allocator, &desc);
}

Texture texture_create_atmosphere_lut(VkDevice device, GpuAllocator *allocator, uint32_t width,
									  uint32_t height, uint32_t layers)
{
	TextureDesc desc = {.format = VK_FORMAT_R16G16B16A16_SFLOAT,
						.width = width,
						.height = height,
						.mip_levels = 1,
						.array_layers = layers,
						.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
						.aspect = VK_IMAGE_ASPECT_COLOR_BIT,
						.filter = VK_FILTER_LINEAR,
						.address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
						.create_sampler = true};
	return texture_create(device, allocator, &desc);
}

Texture texture_create_shadow_array(VkDevice device, GpuAllocator *allocator, VkFormat format,
									uint32_t resolution, uint32_t layers)
{
	TextureDesc desc = {.format = format,
						.width = resolution,
						.height = resolution,
						.mip_levels = 1,
						.array_layers = layers,
						.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
								 VK_IMAGE_USAGE_SAMPLED_BIT,
						.aspect = VK_IMAGE_ASPECT_DEPTH_BIT,
						.filter = VK_FILTER_LINEAR,
						.address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
						.create_sampler = true,
						.compare_enable = true,
						.compare_op = VK_COMPARE_OP_LESS_OR_EQUAL};
	return texture_create(device, allocator, &desc);
}

Texture texture_create_environment_cube(VkDevice device, GpuAllocator *allocator,
										uint32_t face_size)
{
	TextureDesc desc = {.format = VK_FORMAT_R16G16B16A16_SFLOAT,
						.width = face_size,
						.height = face_size,
						.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
						.aspect = VK_IMAGE_ASPECT_COLOR_BIT,
						.filter = VK_FILTER_LINEAR,
						.address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
						.create_sampler = true,
						.cube = true};
	return texture_create(device, allocator, &desc);
}

VkImageView texture_create_storage_mip_view(VkDevice device, const Texture *t, uint32_t mip)
{
	VkImageViewCreateInfo view = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
								  .image = t->image,
								  .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
								  .format = t->format,
								  .subresourceRange = {t->aspect, mip, 1, 0, t->array_layers}};
	VkImageView view_handle;
	VK_CHECK(vkCreateImageView(device, &view, NULL, &view_handle));
	return view_handle;
}

VkImageView texture_create_cube_view(VkDevice device, const Texture *t, uint32_t base_mip,
									 uint32_t mip_count)
{
	VkImageViewCreateInfo view = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = t->image,
		.viewType = VK_IMAGE_VIEW_TYPE_CUBE,
		.format = t->format,
		.subresourceRange = {t->aspect, base_mip, mip_count, 0, t->array_layers}};
	VkImageView view_handle;
	VK_CHECK(vkCreateImageView(device, &view, NULL, &view_handle));
	return view_handle;
}

void texture_destroy(VkDevice device, GpuAllocator *allocator, Texture *t)
{
	if (t->sampler)
		vkDestroySampler(device, t->sampler, NULL);
	vkDestroyImageView(device, t->view, NULL);
	vkDestroyImage(device, t->image, NULL);
	gpu_free(allocator, t->allocation);
	*t = (Texture){0};
}
