#include "renderer.h"
#include "environment.h"
#include "file_utils.h"
#include "vk_common.h"

#include <SDL3/SDL_vulkan.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vulkan/vulkan_core.h>

#ifdef DEBUG_SHADER_DUMP
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

/* Layout of the set-0 binding-3 append buffer every instrumented fragment
   shader writes to. A 16-byte header (count, capacity, 2x pad) precedes an
   array of 96-byte records. Kept in sync with struct DumpRecord in
   shaders/shader_dump.glsl. */
#define SHADER_DUMP_HEADER_WORDS 4u
#define SHADER_DUMP_RECORD_FLOATS 24u /* six vec4s */
typedef struct
{
	float meta[4];							 /* shader_id, frag_x, frag_y, reserved */
	float v0[4], v1[4], v2[4], v3[4], v4[4]; /* 20 generic floats, meaning per shader_id */
} DumpRecord;
_Static_assert(sizeof(DumpRecord) == SHADER_DUMP_RECORD_FLOATS * 4u,
			   "DumpRecord must match std430 DumpRecord in shader_dump.glsl");

/* Kept in sync with the DUMP_SHADER_* constants in shaders/shader_dump.glsl. */
#define DUMP_SHADER_TERRAIN 0u
#define DUMP_SHADER_MESH 1u
#define DUMP_SHADER_CUBE 2u
#define DUMP_SHADER_ATMOSPHERE_COMPOSITE 3u
#define DUMP_SHADER_TONEMAP 4u
#define DUMP_SHADER_TEMPORAL_RESOLVE 5u
#define DUMP_SHADER_ENVIRONMENT_IBL 6u
#define DUMP_SHADER_MATERIAL_DETAIL 7u

static const char *shader_dump_name(uint32_t shader_id)
{
	switch (shader_id)
	{
	case DUMP_SHADER_TERRAIN:
		return "terrain";
	case DUMP_SHADER_MESH:
		return "mesh";
	case DUMP_SHADER_CUBE:
		return "cube";
	case DUMP_SHADER_ATMOSPHERE_COMPOSITE:
		return "atmosphere_composite";
	case DUMP_SHADER_TONEMAP:
		return "tonemap";
	case DUMP_SHADER_TEMPORAL_RESOLVE:
		return "temporal_resolve";
	case DUMP_SHADER_ENVIRONMENT_IBL:
		return "environment_ibl";
	case DUMP_SHADER_MATERIAL_DETAIL:
		return "material_detail";
	default:
		return "unknown";
	}
}

/* Records per screen pixel the dump buffer can hold before truncating (i.e.
   how many overdrawn/overlapping shader invocations per pixel across every
   instrumented shader in a frame). Terrain/mesh now add environment_ibl as a
   second record but the existing eight-layer default remains sufficient.
   8 at 1280x720 is ~7.4M records, ~700 MB.
   Override with TERRAIN_DUMP_LAYERS. */
#define DUMP_LAYER_BUDGET_DEFAULT 8u
#endif

/* Keep the C frame UBO byte-for-byte compatible with shaders/common.glsl std140. */
_Static_assert(offsetof(FrameUniforms, projection) == 0, "FrameUniforms projection offset");
_Static_assert(offsetof(FrameUniforms, view) == 64, "FrameUniforms view offset");
_Static_assert(offsetof(FrameUniforms, view_projection) == 128, "FrameUniforms VP offset");
_Static_assert(offsetof(FrameUniforms, inverse_view_projection) == 192,
			   "FrameUniforms inverse VP offset");
_Static_assert(offsetof(FrameUniforms, previous_projection) == 256,
			   "FrameUniforms previous projection offset");
_Static_assert(offsetof(FrameUniforms, previous_view) == 320, "FrameUniforms previous view offset");
_Static_assert(offsetof(FrameUniforms, previous_view_projection) == 384,
			   "FrameUniforms previous VP offset");
_Static_assert(offsetof(FrameUniforms, local_to_camera_relative) == 448,
			   "FrameUniforms local transform offset");
_Static_assert(offsetof(FrameUniforms, previous_local_to_camera_relative) == 512,
			   "FrameUniforms previous local transform offset");
_Static_assert(offsetof(FrameUniforms, sun_direction) == 576, "FrameUniforms sun offset");
_Static_assert(offsetof(FrameUniforms, time) == 592, "FrameUniforms time offset");
_Static_assert(offsetof(FrameUniforms, near_plane) == 596, "FrameUniforms near offset");
_Static_assert(offsetof(FrameUniforms, debug_view) == 600, "FrameUniforms debug view offset");
_Static_assert(offsetof(FrameUniforms, relight_strength) == 604, "FrameUniforms relight offset");
_Static_assert(offsetof(FrameUniforms, shadow_view_projection) == 608,
			   "FrameUniforms shadow matrices offset");
_Static_assert(offsetof(FrameUniforms, shadow_splits) == 864, "FrameUniforms shadow splits offset");
_Static_assert(offsetof(FrameUniforms, shadow_parameters) == 880,
			   "FrameUniforms shadow parameters offset");
_Static_assert(offsetof(FrameUniforms, shadow_radii) == 896, "FrameUniforms shadow radii offset");
_Static_assert(offsetof(FrameUniforms, shadow_quality) == 912, "FrameUniforms shadow quality offset");
_Static_assert(offsetof(FrameUniforms, shadow_pcss) == 928, "FrameUniforms PCSS offset");
_Static_assert(offsetof(FrameUniforms, sun_radiance) == 944, "FrameUniforms radiance offset");
_Static_assert(offsetof(FrameUniforms, atmosphere_radii) == 960,
			   "FrameUniforms atmosphere radii offset");
_Static_assert(offsetof(FrameUniforms, atmosphere_rayleigh) == 976,
			   "FrameUniforms Rayleigh offset");
_Static_assert(offsetof(FrameUniforms, atmosphere_mie_scatter) == 992,
			   "FrameUniforms Mie scatter offset");
_Static_assert(offsetof(FrameUniforms, atmosphere_mie_extinct) == 1008,
			   "FrameUniforms Mie extinction offset");
_Static_assert(offsetof(FrameUniforms, atmosphere_absorption) == 1024, "FrameUniforms ozone offset");
_Static_assert(offsetof(FrameUniforms, atmosphere_ground) == 1040,
			   "FrameUniforms atmosphere ground offset");
_Static_assert(offsetof(FrameUniforms, atmosphere_options) == 1056,
			   "FrameUniforms atmosphere options offset");
_Static_assert(offsetof(FrameUniforms, temporal_parameters) == 1072,
			   "FrameUniforms temporal parameters offset");
_Static_assert(offsetof(FrameUniforms, temporal_jitter) == 1088,
			   "FrameUniforms temporal jitter offset");
_Static_assert(offsetof(FrameUniforms, shader_dump) == 1104, "FrameUniforms shader dump offset");
_Static_assert(offsetof(FrameUniforms, material_curvature) == 1120, "FrameUniforms curvature offset");
_Static_assert(offsetof(FrameUniforms, material_normal_filter) == 1136, "FrameUniforms normal filter offset");
_Static_assert(sizeof(FrameUniforms) == 1152, "FrameUniforms std140 size");
_Static_assert(sizeof(DrawPushConstants) == 128, "terrain push constant size");
_Static_assert(offsetof(TemporalExposure, histogram) == 16,
			   "TemporalExposure std430 histogram offset");
/* Keep the environment UBO byte-for-byte compatible with
   shaders/environment_lighting.glsl std140. */
_Static_assert(offsetof(EnvironmentUniforms, sh) == 0, "EnvironmentUniforms sh offset");
_Static_assert(offsetof(EnvironmentUniforms, env_params) == 144,
			   "EnvironmentUniforms env_params offset");
_Static_assert(sizeof(EnvironmentUniforms) == 160, "EnvironmentUniforms std140 size");

/* Staging capacity for the upload ring: large enough for the 2048x2048 albedo
   (16 MiB) plus the terrain mesh in a single batch. */
#define UPLOAD_STAGING_CAPACITY (80u * 1024u * 1024u)
#define MAX_TEXTURE_SETS 1024
/* B2 specular IBL cube face size; ENV_CUBE_MIPS (renderer.h) must match
   texture_mip_levels(ENV_FACE_SIZE, ENV_FACE_SIZE). */
#define ENV_FACE_SIZE 128u

VkDescriptorSet renderer_allocate_terrain_set(Renderer *r, VkImageView albedo_view,
												  VkSampler albedo_sampler, VkImageView elevation_view,
												  VkSampler elevation_sampler, VkImageView parent_albedo_view,
												  VkSampler parent_albedo_sampler)
{
	VkDescriptorSetAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
										 .descriptorPool = r->descriptor_pool,
										 .descriptorSetCount = 1,
										 .pSetLayouts = &r->material_set_layout};
	VkDescriptorSet set;
	VK_CHECK(vkAllocateDescriptorSets(r->device, &alloc, &set));
	VkDescriptorImageInfo images[6] = {
		{.sampler = albedo_sampler,
		 .imageView = albedo_view,
		 .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
		{.sampler = elevation_sampler,
		 .imageView = elevation_view,
		 .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
		{.sampler = r->terrain_micro_albedo.sampler,
		 .imageView = r->terrain_micro_albedo.view,
		 .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
		{.sampler = r->terrain_micro_normal.sampler,
		 .imageView = r->terrain_micro_normal.view,
		 .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
		{.sampler = r->terrain_micro_ormh.sampler,
		 .imageView = r->terrain_micro_ormh.view,
		 .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
		{.sampler = parent_albedo_sampler,
		 .imageView = parent_albedo_view,
		 .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
	};
	VkWriteDescriptorSet writes[6] = {
		{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		 .dstSet = set,
		 .dstBinding = 0,
		 .descriptorCount = 1,
		 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 .pImageInfo = &images[0]},
		{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		 .dstSet = set,
		 .dstBinding = 1,
		 .descriptorCount = 1,
		 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 .pImageInfo = &images[1]},
		{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		 .dstSet = set,
		 .dstBinding = 2,
		 .descriptorCount = 1,
		 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 .pImageInfo = &images[2]},
		{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		 .dstSet = set,
		 .dstBinding = 3,
		 .descriptorCount = 1,
		 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 .pImageInfo = &images[3]},
		{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		 .dstSet = set,
		 .dstBinding = 4,
		 .descriptorCount = 1,
		 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 .pImageInfo = &images[4]},
		{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		 .dstSet = set,
		 .dstBinding = 5,
		 .descriptorCount = 1,
		 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 .pImageInfo = &images[5]},
	};
	vkUpdateDescriptorSets(r->device, 6, writes, 0, NULL);
	return set;
}

VkDescriptorSet renderer_allocate_material_set(Renderer *r, VkImageView view, VkSampler sampler)
{
	/* Texture-only static meshes still need typed data fallbacks, rather than
	 * interpreting sRGB white as an ORM or tangent normal. */
	Texture albedo = {.view = view, .sampler = sampler};
	return renderer_allocate_pbr5_set(r, &albedo, &r->fallback_linear_texture,
									  &r->fallback_normal_texture, &r->fallback_linear_texture,
									  &r->fallback_linear_texture);
}

VkDescriptorSet renderer_allocate_pbr_set(Renderer *r, const Texture *albedo, const Texture *orm,
									  const Texture *normal_map)
{
	return renderer_allocate_pbr5_set(r, albedo, orm, normal_map, orm, &r->fallback_linear_texture);
}

VkDescriptorSet renderer_allocate_pbr4_set(Renderer *r, const Texture *albedo, const Texture *orm,
										   const Texture *normal_map, const Texture *occlusion)
{
	return renderer_allocate_pbr5_set(r, albedo, orm, normal_map, occlusion,
									  &r->fallback_linear_texture);
}

VkDescriptorSet renderer_allocate_pbr5_set(Renderer *r, const Texture *albedo, const Texture *orm,
									   const Texture *normal_map, const Texture *occlusion,
									   const Texture *cavity)
{
	VkDescriptorSetAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
										 .descriptorPool = r->descriptor_pool,
										 .descriptorSetCount = 1,
										 .pSetLayouts = &r->material_set_layout};
	VkDescriptorSet set;
	VK_CHECK(vkAllocateDescriptorSets(r->device, &alloc, &set));
	const Texture *textures[5] = {albedo, orm, normal_map, occlusion, cavity};
	VkDescriptorImageInfo images[5];
	VkWriteDescriptorSet writes[5];
	for (uint32_t i = 0; i < 5; ++i)
	{
		images[i] =
			(VkDescriptorImageInfo){.sampler = textures[i]->sampler,
									.imageView = textures[i]->view,
									.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
		writes[i] =
			(VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
								   .dstSet = set,
								   .dstBinding = i,
								   .descriptorCount = 1,
								   .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
								   .pImageInfo = &images[i]};
	}
	vkUpdateDescriptorSets(r->device, 5, writes, 0, NULL);
	return set;
}

void renderer_free_material_set(Renderer *r, VkDescriptorSet set)
{
	if (r && set)
		VK_CHECK(vkFreeDescriptorSets(r->device, r->descriptor_pool, 1, &set));
}

/* Load + project the diffuse-IBL environment once at startup and upload the
   result to a small static UBO. Must run before create_descriptors, which
   binds r->environment_ubo at set 0 binding 4. Absent-able: if ENV_HDR_PATH
   doesn't decode, env_params.x stays 0 and the shader falls back to the
   original hemispheric-ambient constant -- draw submission is unchanged
   either way (Phase B architecture invariant #5, unrealplan.md).

   Also stages the B2 specular prefilter's inputs: the equirect HDR uploaded
   as r->environment_equirect (a 1x1 black texel when absent), and the empty
   r->environment_cube it will be baked into. Both must exist before
   create_descriptors binds environment_cube at set 0 binding 5 -- the actual
   prefilter compute runs later, in environment_prefilter(), once the compute
   pipelines it needs exist. */
static void create_environment(Renderer *r)
{
	EnvironmentUniforms uniforms = {0};
	float *pixels = NULL;
	int width = 0, height = 0;
	const char *environment_path = r->environment_path[0] ? r->environment_path : ENV_HDR_PATH;
	bool loaded = environment_load_hdr(environment_path, &pixels, &width, &height);
	if (loaded)
	{
		EnvironmentSH sh;
		environment_project_sh9(pixels, width, height, &sh);
		for (uint32_t i = 0; i < 9; ++i)
			uniforms.sh[i] = (vec4s){{sh.coeffs[i][0], sh.coeffs[i][1], sh.coeffs[i][2], 0.0f}};
		printf("Environment: loaded IBL from %dx%d %s\n", width, height, environment_path);
	}
	else
		printf("Environment: no HDR at %s, using hemispheric ambient fallback\n", environment_path);
	uniforms.env_params =
		(vec4s){{loaded ? 1.0f : 0.0f, 0.30f, 0.10f, (float)(ENV_CUBE_MIPS - 1u)}};

	r->environment_ubo = gpu_buffer_create(
		r->device, r->allocator, sizeof(EnvironmentUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	memcpy(r->environment_ubo.allocation.mapped, &uniforms, sizeof(uniforms));

	const float black_pixel[4] = {0.0f, 0.0f, 0.0f, 1.0f};
	uint32_t equirect_width = loaded ? (uint32_t)width : 1u;
	uint32_t equirect_height = loaded ? (uint32_t)height : 1u;
	const float *equirect_pixels = loaded ? pixels : black_pixel;
	TextureDesc equirect_desc = {.format = VK_FORMAT_R32G32B32A32_SFLOAT,
								 .width = equirect_width,
								 .height = equirect_height,
								 .mip_levels = 1,
								 .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
								 .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
								 .filter = VK_FILTER_LINEAR,
								 .address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
								 .create_sampler = true};
	r->environment_equirect = texture_create(r->device, r->allocator, &equirect_desc);
	upload_begin(r->upload);
	upload_image(r->upload, r->environment_equirect.image, r->environment_equirect.format,
				equirect_width, equirect_height, 1, VK_IMAGE_ASPECT_COLOR_BIT,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, equirect_pixels,
				(VkDeviceSize)equirect_width * equirect_height * 4u * sizeof(float));
	upload_submit(r->upload);
	r->environment_equirect.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	if (loaded)
		environment_free_hdr(pixels);

	r->environment_cube = texture_create_environment_cube(r->device, r->allocator, ENV_FACE_SIZE);
}

/* Descriptor roles: set 0 = per-frame UBO; set 1 = pass-local material/HDR;
   set 2 = atmosphere LUTs for graphics. Compute sees that same atmosphere set
   as set 1, avoiding duplicate descriptors. */
static void create_descriptors(Renderer *r)
{
	/* Bindings 4 (environment UBO) and 5 (specular IBL cube, B2) are
	   unconditional; binding 3 (debug dump SSBO) stays conditional and must
	   stay last in this array so the non-debug build's frame_binding_count ==
	   5 slice excludes it. Non-contiguous binding numbers are legal in Vulkan. */
	VkDescriptorSetLayoutBinding frame_bindings[6] = {
		{.binding = 0,
		 .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		 .descriptorCount = 1,
		 .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT |
					   VK_SHADER_STAGE_COMPUTE_BIT},
		{.binding = 1,
		 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 .descriptorCount = 1,
		 .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
		{.binding = 2,
		 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 .descriptorCount = 1,
		 .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
		{.binding = 4,
		 .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		 .descriptorCount = 1,
		 .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
		{.binding = 5,
		 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 .descriptorCount = 1,
		 .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
		{.binding = 3,
		 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		 .descriptorCount = 1,
		 .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
	};
	uint32_t frame_binding_count = 5;
#ifdef DEBUG_SHADER_DUMP
	frame_binding_count = 6;
#endif
	VkDescriptorSetLayoutCreateInfo frame_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = frame_binding_count,
		.pBindings = frame_bindings};
	VK_CHECK(vkCreateDescriptorSetLayout(r->device, &frame_layout, NULL, &r->frame_set_layout));

	VkDescriptorSetLayoutBinding material_bindings[6] = {
		{.binding = 0,
		 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 .descriptorCount = 1,
		 .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
		{.binding = 1,
		 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 .descriptorCount = 1,
		 .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
		{.binding = 2,
		 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 .descriptorCount = 1,
		 .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
		{.binding = 3,
		 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 .descriptorCount = 1,
		 .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
		{.binding = 4,
		 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 .descriptorCount = 1,
		 .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
		{.binding = 5,
		 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 .descriptorCount = 1,
		 .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
	};
	VkDescriptorSetLayoutCreateInfo material_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 6,
		.pBindings = material_bindings};
	VK_CHECK(
		vkCreateDescriptorSetLayout(r->device, &material_layout, NULL, &r->material_set_layout));

	VkDescriptorSetLayoutBinding display_bindings[2] = {
		{.binding = 0,
		 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 .descriptorCount = 1,
		 .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
		{.binding = 1,
		 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 .descriptorCount = 1,
		 .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
	};
	VkDescriptorSetLayoutCreateInfo display_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 2,
		.pBindings = display_bindings};
	VK_CHECK(vkCreateDescriptorSetLayout(r->device, &display_layout, NULL, &r->display_set_layout));

	VkDescriptorSetLayoutBinding temporal_bindings[7];
	for (uint32_t i = 0; i < 7; ++i)
		temporal_bindings[i] = (VkDescriptorSetLayoutBinding){
			.binding = i,
			.descriptorCount = 1,
			.descriptorType = i < 6 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
									: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT};
	VkDescriptorSetLayoutCreateInfo temporal_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 7,
		.pBindings = temporal_bindings};
	VK_CHECK(
		vkCreateDescriptorSetLayout(r->device, &temporal_layout, NULL, &r->temporal_set_layout));

	VkDescriptorSetLayoutBinding atmosphere_bindings[10];
	for (uint32_t i = 0; i < 10; ++i)
		atmosphere_bindings[i] = (VkDescriptorSetLayoutBinding){
			.binding = i,
			.descriptorCount = 1,
			.descriptorType = i < 5 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
									: VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | (i < 5 ? VK_SHADER_STAGE_FRAGMENT_BIT : 0)};
	VkDescriptorSetLayoutCreateInfo atmosphere_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 10,
		.pBindings = atmosphere_bindings};
	VK_CHECK(vkCreateDescriptorSetLayout(r->device, &atmosphere_layout, NULL,
										 &r->atmosphere_set_layout));

	/* +ENV_CUBE_MIPS combined-image-sampler/storage-image/set slots below are
	   the one-shot B2 prefilter descriptor sets (one per output cube mip: b0
	   samples the equirect or cube mip 0, b1 is that mip's storage view). */
	VkDescriptorPoolSize pool_sizes[4] = {
		{.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		 .descriptorCount = MAX_FRAMES_IN_FLIGHT * 2u /* frame UBO + environment UBO */},
		{.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 .descriptorCount =
			 MAX_TEXTURE_SETS * 6u + MAX_FRAMES_IN_FLIGHT * 2u + 20u + ENV_CUBE_MIPS},
		{.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 5u + ENV_CUBE_MIPS},
		{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = MAX_FRAMES_IN_FLIGHT + 2u}};
	uint32_t pool_size_count = 4;
	VkDescriptorPoolCreateInfo pool = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
		.poolSizeCount = pool_size_count,
		.pPoolSizes = pool_sizes,
		.maxSets = MAX_FRAMES_IN_FLIGHT + MAX_TEXTURE_SETS + 4u + ENV_CUBE_MIPS};
	VK_CHECK(vkCreateDescriptorPool(r->device, &pool, NULL, &r->descriptor_pool));

	/* One persistently-mapped UBO + set per frame in flight. */
	for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
	{
		r->frame_ubo[i] = gpu_buffer_create(
			r->device, r->allocator, sizeof(FrameUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		VkDescriptorSetAllocateInfo alloc = {.sType =
												 VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
											 .descriptorPool = r->descriptor_pool,
											 .descriptorSetCount = 1,
											 .pSetLayouts = &r->frame_set_layout};
		VK_CHECK(vkAllocateDescriptorSets(r->device, &alloc, &r->frame_set[i]));
		VkDescriptorBufferInfo info = {
			.buffer = r->frame_ubo[i].buffer, .offset = 0, .range = sizeof(FrameUniforms)};
		VkDescriptorBufferInfo env_info = {
			.buffer = r->environment_ubo.buffer, .offset = 0, .range = sizeof(EnvironmentUniforms)};
		VkDescriptorImageInfo env_cube_info = {
			.sampler = r->environment_cube.sampler,
			.imageView = r->environment_cube.view,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
		VkDescriptorImageInfo shadow_images[2] = {
			{.sampler = r->shadow_map.sampler,
			 .imageView = r->shadow_map.view,
			 .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL},
			{.sampler = r->shadow_raw_sampler,
			 .imageView = r->shadow_map.view,
			 .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL},
		};
		VkWriteDescriptorSet writes[5] = {
			{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			 .dstSet = r->frame_set[i],
			 .dstBinding = 0,
			 .descriptorCount = 1,
			 .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			 .pBufferInfo = &info},
			{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			 .dstSet = r->frame_set[i],
			 .dstBinding = 1,
			 .descriptorCount = 1,
			 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			 .pImageInfo = &shadow_images[0]},
			{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			 .dstSet = r->frame_set[i],
			 .dstBinding = 2,
			 .descriptorCount = 1,
			 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			 .pImageInfo = &shadow_images[1]},
			{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			 .dstSet = r->frame_set[i],
			 .dstBinding = 4,
			 .descriptorCount = 1,
			 .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			 .pBufferInfo = &env_info},
			{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			 .dstSet = r->frame_set[i],
			 .dstBinding = 5,
			 .descriptorCount = 1,
			 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			 .pImageInfo = &env_cube_info},
		};
		vkUpdateDescriptorSets(r->device, 5, writes, 0, NULL);
	}

	texture_create_white(r->device, r->allocator, r->upload, &r->fallback_texture);
	const uint8_t linear_white[4] = {255, 255, 255, 255};
	const uint8_t flat_normal[4] = {128, 128, 255, 255};
	texture_create_solid_rgba8(r->device, r->allocator, r->upload, &r->fallback_linear_texture,
							  linear_white, false);
	texture_create_solid_rgba8(r->device, r->allocator, r->upload, &r->fallback_normal_texture,
							  flat_normal, false);
	/* Atlas cells have an 8-pixel wrap gutter.  Four levels (0..3) preserve at
	   least one gutter texel; lower global mips blend the 4x4 atlas cells and
	   the intentionally empty black sixteenth cell into terrain samples. */
	const uint32_t micro_atlas_mip_levels = 4u;
	texture_load_linear_mip_limited(r->device, r->allocator, r->upload,
									 &r->terrain_micro_albedo,
									 TEXTURE_DIR "/runtime/terrain_micro_albedo.png",
									 r->max_anisotropy, micro_atlas_mip_levels);
	texture_load_linear_mip_limited(r->device, r->allocator, r->upload,
									 &r->terrain_micro_normal,
									 TEXTURE_DIR "/runtime/terrain_micro_normal.png",
									 r->max_anisotropy, micro_atlas_mip_levels);
	texture_load_linear_mip_limited(r->device, r->allocator, r->upload,
									 &r->terrain_micro_ormh,
									 TEXTURE_DIR "/runtime/terrain_micro_ormh.png",
									 r->max_anisotropy, micro_atlas_mip_levels);
	r->fallback_material_set = renderer_allocate_pbr5_set(r, &r->fallback_texture,
		&r->fallback_linear_texture, &r->fallback_normal_texture, &r->fallback_linear_texture,
		&r->fallback_linear_texture);

	VkDescriptorSetAllocateInfo atmosphere_alloc = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = r->descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &r->atmosphere_set_layout};
	VK_CHECK(vkAllocateDescriptorSets(r->device, &atmosphere_alloc, &r->atmosphere_set));
	Texture *textures[5] = {&r->atmosphere_transmittance, &r->atmosphere_multiscattering,
							&r->atmosphere_skyview, &r->atmosphere_aerial_scattering,
							&r->atmosphere_aerial_transmittance};
	VkDescriptorImageInfo sampled[5], storage[5];
	VkWriteDescriptorSet atmosphere_writes[10];
	for (uint32_t i = 0; i < 5; ++i)
	{
		sampled[i] = (VkDescriptorImageInfo){.sampler = textures[i]->sampler,
											 .imageView = textures[i]->view,
											 .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
		storage[i] = (VkDescriptorImageInfo){.imageView = textures[i]->view,
											 .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
		atmosphere_writes[i] =
			(VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
								   .dstSet = r->atmosphere_set,
								   .dstBinding = i,
								   .descriptorCount = 1,
								   .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
								   .pImageInfo = &sampled[i]};
		atmosphere_writes[i + 5] =
			(VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
								   .dstSet = r->atmosphere_set,
								   .dstBinding = i + 5,
								   .descriptorCount = 1,
								   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
								   .pImageInfo = &storage[i]};
	}
	vkUpdateDescriptorSets(r->device, 10, atmosphere_writes, 0, NULL);

	r->exposure_buffer = gpu_buffer_create(
		r->device, r->allocator, sizeof(TemporalExposure),
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	TemporalExposure *exposure = r->exposure_buffer.allocation.mapped;
	*exposure = (TemporalExposure){.exposure = 1.0f, .average_luminance = 0.18f};
}

static VkFormat find_depth_format(VkPhysicalDevice physical_device)
{
	const VkFormat candidates[] = {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
								   VK_FORMAT_D24_UNORM_S8_UINT};
	for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i)
	{
		VkFormatProperties properties;
		vkGetPhysicalDeviceFormatProperties(physical_device, candidates[i], &properties);
		const VkFormatFeatureFlags required =
			VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
		if ((properties.optimalTilingFeatures & required) == required)
			return candidates[i];
	}
	fprintf(stderr, "No sampleable depth-attachment format found\n");
	exit(EXIT_FAILURE);
}

static VkFormat find_shadow_format(VkPhysicalDevice physical_device)
{
	const VkFormat candidates[] = {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM,
								   VK_FORMAT_D24_UNORM_S8_UINT};
	const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
										  VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
										  VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
	for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i)
	{
		VkFormatProperties properties;
		vkGetPhysicalDeviceFormatProperties(physical_device, candidates[i], &properties);
		if ((properties.optimalTilingFeatures & required) == required)
			return candidates[i];
	}
	fprintf(stderr, "No filterable sampled depth format found for shadows\n");
	exit(EXIT_FAILURE);
}

static void create_shadow_texture(Renderer *r)
{
	VkFormat format = find_shadow_format(r->physical_device);
	r->shadow_map = texture_create_shadow_array(r->device, r->allocator, format,
												r->shadow_resolution, SHADOW_CASCADE_COUNT);

	VkSamplerCreateInfo raw_sampler = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
									   .magFilter = VK_FILTER_NEAREST,
									   .minFilter = VK_FILTER_NEAREST,
									   .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
									   .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
									   .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
									   .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
									   .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
									   .maxLod = 0.0f};
	VK_CHECK(vkCreateSampler(r->device, &raw_sampler, NULL, &r->shadow_raw_sampler));

	for (uint32_t layer = 0; layer < SHADOW_CASCADE_COUNT; ++layer)
	{
		VkImageViewCreateInfo view = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = r->shadow_map.image,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = format,
			.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, layer, 1}};
		VK_CHECK(vkCreateImageView(r->device, &view, NULL, &r->shadow_layer_views[layer]));
	}
}

static void create_atmosphere_textures(Renderer *r)
{
	/* Initial sizes are the proven Unreal demo defaults. The aerial volume is
	   represented as a 32-layer 2D array because Vulkan samples array layers
	   portably and our final pass performs the small z interpolation itself. */
	r->atmosphere_transmittance =
		texture_create_atmosphere_lut(r->device, r->allocator, 256, 64, 1);
	r->atmosphere_multiscattering =
		texture_create_atmosphere_lut(r->device, r->allocator, 32, 32, 1);
	r->atmosphere_skyview = texture_create_atmosphere_lut(r->device, r->allocator, 192, 108, 1);
	r->atmosphere_aerial_scattering =
		texture_create_atmosphere_lut(r->device, r->allocator, 32, 32, 32);
	r->atmosphere_aerial_transmittance =
		texture_create_atmosphere_lut(r->device, r->allocator, 32, 32, 32);
}

static VkShaderModule create_shader_module(Renderer *r, const char *path)
{
	size_t size;
	uint8_t *code;
	FileReadResult result = file_read_all(path, &code, &size);
	if (result != FILE_READ_OK)
	{
		fprintf(stderr, "Could not read %s: %s\n", path, file_read_result_string(result));
		exit(EXIT_FAILURE);
	}
	VkShaderModuleCreateInfo info = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
									 .codeSize = size,
									 .pCode = (const uint32_t *)code};
	VkShaderModule module;
	VK_CHECK(vkCreateShaderModule(r->device, &info, NULL, &module));
	free(code);
	return module;
}

static bool device_has_swapchain(VkPhysicalDevice device)
{
	uint32_t count = 0;
	vkEnumerateDeviceExtensionProperties(device, NULL, &count, NULL);
	VkExtensionProperties *extensions = malloc(sizeof(*extensions) * count);
	vkEnumerateDeviceExtensionProperties(device, NULL, &count, extensions);
	bool found = false;
	for (uint32_t i = 0; i < count; ++i)
		if (strcmp(extensions[i].extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0)
			found = true;
	free(extensions);
	return found;
}

static bool find_queue_families(Renderer *r, VkPhysicalDevice device, uint32_t *graphics,
								uint32_t *present)
{
	uint32_t count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &count, NULL);
	VkQueueFamilyProperties *families = malloc(sizeof(*families) * count);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families);
	bool has_graphics = false, has_present = false;
	for (uint32_t i = 0; i < count; ++i)
	{
		VkBool32 supported = VK_FALSE;
		vkGetPhysicalDeviceSurfaceSupportKHR(device, i, r->surface, &supported);
		if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
		{
			*graphics = i;
			has_graphics = true;
		}
		if (supported)
		{
			*present = i;
			has_present = true;
		}
		if (has_graphics && has_present)
			break;
	}
	free(families);
	return has_graphics && has_present;
}

static void create_instance_and_device(Renderer *r)
{
	uint32_t extension_count = 0;
	const char *const *extensions = SDL_Vulkan_GetInstanceExtensions(&extension_count);
	if (!extensions)
	{
		fprintf(stderr, "SDL Vulkan extensions: %s\n", SDL_GetError());
		exit(EXIT_FAILURE);
	}

	VkApplicationInfo application = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
									 .pApplicationName = "Terrain Renderer",
									 .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
									 .pEngineName = "none",
									 .engineVersion = VK_MAKE_VERSION(1, 0, 0),
									 .apiVersion = VK_API_VERSION_1_0};
	VkInstanceCreateInfo instance_info = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
										  .pApplicationInfo = &application,
										  .enabledExtensionCount = extension_count,
										  .ppEnabledExtensionNames = extensions};
	VK_CHECK(vkCreateInstance(&instance_info, NULL, &r->instance));
	if (!SDL_Vulkan_CreateSurface(r->window, r->instance, NULL, &r->surface))
	{
		fprintf(stderr, "SDL_Vulkan_CreateSurface: %s\n", SDL_GetError());
		exit(EXIT_FAILURE);
	}

	uint32_t device_count = 0;
	vkEnumeratePhysicalDevices(r->instance, &device_count, NULL);
	if (!device_count)
	{
		fprintf(stderr, "No Vulkan device found\n");
		exit(EXIT_FAILURE);
	}
	VkPhysicalDevice *devices = malloc(sizeof(*devices) * device_count);
	vkEnumeratePhysicalDevices(r->instance, &device_count, devices);
	for (uint32_t i = 0; i < device_count; ++i)
	{
		uint32_t graphics, present;
		if (device_has_swapchain(devices[i]) &&
			find_queue_families(r, devices[i], &graphics, &present))
		{
			r->physical_device = devices[i];
			r->graphics_family = graphics;
			r->present_family = present;
			break;
		}
	}
	free(devices);
	if (r->physical_device == VK_NULL_HANDLE)
	{
		fprintf(stderr, "No suitable Vulkan device found\n");
		exit(EXIT_FAILURE);
	}

	/* Enable anisotropic sampling within the device limit when supported. */
	VkPhysicalDeviceFeatures supported;
	vkGetPhysicalDeviceFeatures(r->physical_device, &supported);
	VkPhysicalDeviceProperties properties;
	vkGetPhysicalDeviceProperties(r->physical_device, &properties);
	r->max_anisotropy = supported.samplerAnisotropy ? properties.limits.maxSamplerAnisotropy : 0.0f;

	float priority = 1.0f;
	uint32_t families[2] = {r->graphics_family, r->present_family};
	VkDeviceQueueCreateInfo queues[2] = {0};
	uint32_t queue_count = r->graphics_family == r->present_family ? 1 : 2;
	for (uint32_t i = 0; i < queue_count; ++i)
		queues[i] = (VkDeviceQueueCreateInfo){.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
											  .queueFamilyIndex = families[i],
											  .queueCount = 1,
											  .pQueuePriorities = &priority};
	const char *device_extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
	if (!supported.shaderStorageImageExtendedFormats)
	{
		fprintf(stderr, "RGBA16F storage images are required for atmosphere LUTs\n");
		exit(EXIT_FAILURE);
	}
	VkPhysicalDeviceFeatures features = {.samplerAnisotropy = supported.samplerAnisotropy,
										 .shaderStorageImageExtendedFormats = VK_TRUE,
										 .fragmentStoresAndAtomics = VK_TRUE};
	VkDeviceCreateInfo device_info = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
									  .queueCreateInfoCount = queue_count,
									  .pQueueCreateInfos = queues,
									  .enabledExtensionCount = 1,
									  .ppEnabledExtensionNames = device_extensions,
									  .pEnabledFeatures = &features};
	VK_CHECK(vkCreateDevice(r->physical_device, &device_info, NULL, &r->device));
	vkGetDeviceQueue(r->device, r->graphics_family, 0, &r->graphics_queue);
	vkGetDeviceQueue(r->device, r->present_family, 0, &r->present_queue);
}

static void create_scene_render_pass(Renderer *r, VkFormat depth_format)
{
	VkAttachmentDescription attachments[3] = {
		{.format = VK_FORMAT_R16G16B16A16_SFLOAT,
		 .samples = VK_SAMPLE_COUNT_1_BIT,
		 .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		 .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		 .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
		 .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
		 .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		 .finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
		{.format = VK_FORMAT_R16G16_SFLOAT,
		 .samples = VK_SAMPLE_COUNT_1_BIT,
		 .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		 .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		 .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
		 .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
		 .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		 .finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
		{.format = depth_format,
		 .samples = VK_SAMPLE_COUNT_1_BIT,
		 .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		 .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		 .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
		 .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
		 .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		 .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL}};
	VkAttachmentReference color_refs[2] = {{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
										   {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}};
	VkAttachmentReference depth_ref = {2, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
	VkSubpassDescription subpass = {.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
									.colorAttachmentCount = 2,
									.pColorAttachments = color_refs,
									.pDepthStencilAttachment = &depth_ref};
	VkSubpassDependency dependencies[2] = {
		{.srcSubpass = VK_SUBPASS_EXTERNAL,
		 .dstSubpass = 0,
		 .srcStageMask =
			 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
		 .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
						 VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
		 .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
		 .dstAccessMask =
			 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT},
		{.srcSubpass = 0,
		 .dstSubpass = VK_SUBPASS_EXTERNAL,
		 .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
						 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
		 .dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		 .srcAccessMask =
			 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		 .dstAccessMask = VK_ACCESS_SHADER_READ_BIT},
	};
	VkRenderPassCreateInfo info = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
								   .attachmentCount = 3,
								   .pAttachments = attachments,
								   .subpassCount = 1,
								   .pSubpasses = &subpass,
								   .dependencyCount = 2,
								   .pDependencies = dependencies};
	VK_CHECK(vkCreateRenderPass(r->device, &info, NULL, &r->scene_render_pass));
}

static VkRenderPass create_color_post_render_pass(Renderer *r, uint32_t attachment_count,
												  const VkFormat *formats)
{
	VkAttachmentDescription attachments[2] = {0};
	VkAttachmentReference references[2] = {0};
	for (uint32_t i = 0; i < attachment_count; ++i)
	{
		attachments[i] =
			(VkAttachmentDescription){.format = formats[i],
									  .samples = VK_SAMPLE_COUNT_1_BIT,
									  .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
									  .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
									  .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
									  .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
									  .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
									  .finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
		references[i] = (VkAttachmentReference){i, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
	}
	VkSubpassDescription subpass = {.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
									.colorAttachmentCount = attachment_count,
									.pColorAttachments = references};
	VkSubpassDependency dependencies[2] = {
		{.srcSubpass = VK_SUBPASS_EXTERNAL,
		 .dstSubpass = 0,
		 .srcStageMask =
			 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		 .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		 .srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		 .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT},
		{.srcSubpass = 0,
		 .dstSubpass = VK_SUBPASS_EXTERNAL,
		 .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		 .dstStageMask =
			 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		 .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		 .dstAccessMask = VK_ACCESS_SHADER_READ_BIT}};
	VkRenderPassCreateInfo info = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
								   .attachmentCount = attachment_count,
								   .pAttachments = attachments,
								   .subpassCount = 1,
								   .pSubpasses = &subpass,
								   .dependencyCount = 2,
								   .pDependencies = dependencies};
	VkRenderPass render_pass;
	VK_CHECK(vkCreateRenderPass(r->device, &info, NULL, &render_pass));
	return render_pass;
}

static void create_display_render_pass(Renderer *r)
{
	VkAttachmentDescription attachment = {.format = r->swapchain_format,
										  .samples = VK_SAMPLE_COUNT_1_BIT,
										  .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
										  .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
										  .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
										  .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
										  .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
										  .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR};
	VkAttachmentReference color = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
	VkSubpassDescription subpass = {.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
									.colorAttachmentCount = 1,
									.pColorAttachments = &color};
	VkSubpassDependency dependency = {.srcSubpass = VK_SUBPASS_EXTERNAL,
									  .dstSubpass = 0,
									  .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
									  .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
									  .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT};
	VkRenderPassCreateInfo info = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
								   .attachmentCount = 1,
								   .pAttachments = &attachment,
								   .subpassCount = 1,
								   .pSubpasses = &subpass,
								   .dependencyCount = 1,
								   .pDependencies = &dependency};
	VK_CHECK(vkCreateRenderPass(r->device, &info, NULL, &r->display_render_pass));
}

static void create_shadow_render_pass(Renderer *r, VkFormat depth_format)
{
	VkAttachmentDescription attachment = {.format = depth_format,
										  .samples = VK_SAMPLE_COUNT_1_BIT,
										  .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
										  .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
										  .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
										  .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
										  .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
										  .finalLayout =
											  VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
	VkAttachmentReference depth = {0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
	VkSubpassDescription subpass = {.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
									.pDepthStencilAttachment = &depth};
	VkSubpassDependency dependencies[2] = {
		{.srcSubpass = VK_SUBPASS_EXTERNAL,
		 .dstSubpass = 0,
		 .srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		 .dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
		 .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
		 .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT},
		{.srcSubpass = 0,
		 .dstSubpass = VK_SUBPASS_EXTERNAL,
		 .srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
		 .dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		 .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		 .dstAccessMask = VK_ACCESS_SHADER_READ_BIT},
	};
	VkRenderPassCreateInfo info = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
								   .attachmentCount = 1,
								   .pAttachments = &attachment,
								   .subpassCount = 1,
								   .pSubpasses = &subpass,
								   .dependencyCount = 2,
								   .pDependencies = dependencies};
	VK_CHECK(vkCreateRenderPass(r->device, &info, NULL, &r->shadow_render_pass));
}

static void create_shadow_framebuffers(Renderer *r)
{
	for (uint32_t layer = 0; layer < SHADOW_CASCADE_COUNT; ++layer)
	{
		VkFramebufferCreateInfo info = {.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
										.renderPass = r->shadow_render_pass,
										.attachmentCount = 1,
										.pAttachments = &r->shadow_layer_views[layer],
										.width = r->shadow_resolution,
										.height = r->shadow_resolution,
										.layers = 1};
		VK_CHECK(vkCreateFramebuffer(r->device, &info, NULL, &r->shadow_framebuffers[layer]));
	}
}

/* Graphics share set 2 atmosphere resources. Compute uses frame set 0 plus the
   same atmosphere layout at set 1, matching the compact compute shaders. */
static void create_pipeline_layout(Renderer *r)
{
	VkDescriptorSetLayout layouts[3] = {r->frame_set_layout, r->material_set_layout,
										r->atmosphere_set_layout};
	VkPushConstantRange push = {.stageFlags =
									VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
								.offset = 0,
								.size = sizeof(DrawPushConstants)};
	VkPipelineLayoutCreateInfo layout_info = {.sType =
												  VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
											  .setLayoutCount = 3,
											  .pSetLayouts = layouts,
											  .pushConstantRangeCount = 1,
											  .pPushConstantRanges = &push};
	VK_CHECK(vkCreatePipelineLayout(r->device, &layout_info, NULL, &r->pipeline_layout));

	VkDescriptorSetLayout display_layouts[3] = {r->frame_set_layout, r->display_set_layout,
												r->atmosphere_set_layout};
	VkPipelineLayoutCreateInfo display_info = {.sType =
												   VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
											   .setLayoutCount = 3,
											   .pSetLayouts = display_layouts};
	VK_CHECK(vkCreatePipelineLayout(r->device, &display_info, NULL, &r->display_pipeline_layout));

	VkDescriptorSetLayout temporal_layouts[2] = {r->frame_set_layout, r->temporal_set_layout};
	VkPipelineLayoutCreateInfo temporal_info = {.sType =
													VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
												.setLayoutCount = 2,
												.pSetLayouts = temporal_layouts};
	VK_CHECK(vkCreatePipelineLayout(r->device, &temporal_info, NULL, &r->temporal_pipeline_layout));

	VkDescriptorSetLayout compute_layouts[2] = {r->frame_set_layout, r->atmosphere_set_layout};
	VkPipelineLayoutCreateInfo compute_info = {.sType =
												   VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
											   .setLayoutCount = 2,
											   .pSetLayouts = compute_layouts};
	VK_CHECK(
		vkCreatePipelineLayout(r->device, &compute_info, NULL, &r->atmosphere_pipeline_layout));
}

static VkPipeline create_compute_pipeline(Renderer *r, const char *shader_name)
{
	char path[1024];
	snprintf(path, sizeof(path), "%s/%s.spv", SHADER_DIR, shader_name);
	VkShaderModule module = create_shader_module(r, path);
	VkPipelineShaderStageCreateInfo stage = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.stage = VK_SHADER_STAGE_COMPUTE_BIT,
		.module = module,
		.pName = "main"};
	VkComputePipelineCreateInfo info = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
										.stage = stage,
										.layout = r->atmosphere_pipeline_layout};
	VkPipeline pipeline;
	VK_CHECK(vkCreateComputePipelines(r->device, VK_NULL_HANDLE, 1, &info, NULL, &pipeline));
	vkDestroyShaderModule(r->device, module, NULL);
	return pipeline;
}

static void create_atmosphere_pipelines(Renderer *r)
{
	r->atmosphere_transmittance_pipeline =
		create_compute_pipeline(r, "atmosphere_transmittance.comp");
	r->atmosphere_multiscattering_pipeline =
		create_compute_pipeline(r, "atmosphere_multiscattering.comp");
	r->atmosphere_skyview_pipeline = create_compute_pipeline(r, "atmosphere_skyview.comp");
	r->atmosphere_aerial_pipeline = create_compute_pipeline(r, "atmosphere_aerial.comp");
}

static VkPipeline create_temporal_compute_pipeline(Renderer *r, const char *shader_name)
{
	char path[1024];
	snprintf(path, sizeof(path), "%s/%s.spv", SHADER_DIR, shader_name);
	VkShaderModule module = create_shader_module(r, path);
	VkPipelineShaderStageCreateInfo stage = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.stage = VK_SHADER_STAGE_COMPUTE_BIT,
		.module = module,
		.pName = "main"};
	VkComputePipelineCreateInfo info = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
										.stage = stage,
										.layout = r->temporal_pipeline_layout};
	VkPipeline pipeline;
	VK_CHECK(vkCreateComputePipelines(r->device, VK_NULL_HANDLE, 1, &info, NULL, &pipeline));
	vkDestroyShaderModule(r->device, module, NULL);
	return pipeline;
}

static void create_temporal_compute_pipelines(Renderer *r)
{
	r->luminance_histogram_pipeline =
		create_temporal_compute_pipeline(r, "luminance_histogram.comp");
	r->exposure_pipeline = create_temporal_compute_pipeline(r, "exposure.comp");
}

/* Push constants shared by both B2 prefilter compute shaders (see
   shaders/environment_equirect_to_cube.comp / environment_prefilter.comp).
   roughness is unused by the equirect-to-cube pass. */
typedef struct
{
	float roughness;
	float face_size;
} EnvironmentPrefilterPush;

/* Like atmosphere_image_barrier, but scoped to [base_mip, base_mip+mip_count)
   of a cube (6 layers) instead of the whole image, and with an explicit
   `new_layout` instead of always GENERAL -- the B2 prefilter has mip 0 and
   mips 1..N-1 in different layouts at the same point in the command buffer
   (mip 0 finishes and becomes readable while the rest are still being
   written), which a single whole-image barrier can't express. Does not touch
   texture->layout; the caller sets it once every mip has converged. */
static void environment_mip_barrier(VkCommandBuffer command, Texture *texture, uint32_t base_mip,
									uint32_t mip_count, VkImageLayout old_layout,
									VkImageLayout new_layout, VkPipelineStageFlags source_stage,
									VkAccessFlags source_access,
									VkPipelineStageFlags destination_stage,
									VkAccessFlags destination_access)
{
	VkImageMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = source_access,
		.dstAccessMask = destination_access,
		.oldLayout = old_layout,
		.newLayout = new_layout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = texture->image,
		.subresourceRange = {texture->aspect, base_mip, mip_count, 0, texture->array_layers}};
	vkCmdPipelineBarrier(command, source_stage, destination_stage, 0, 0, NULL, 0, NULL, 1,
						 &barrier);
}

/* Standalone descriptor/pipeline layout for the B2 specular prefilter: it
   needs neither the per-frame UBO nor the atmosphere LUTs, just one sampled
   input and one storage-image output per dispatch (see environment_prefilter
   below). Built once at init and torn down with the renderer, unlike the
   compute shaders' resources which are freed right after use. */
static void create_environment_prefilter_pipelines(Renderer *r)
{
	VkDescriptorSetLayoutBinding bindings[2] = {
		{.binding = 0,
		 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 .descriptorCount = 1,
		 .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
		{.binding = 1,
		 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		 .descriptorCount = 1,
		 .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	};
	VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 2,
		.pBindings = bindings};
	VK_CHECK(vkCreateDescriptorSetLayout(r->device, &layout_info, NULL,
										 &r->environment_prefilter_set_layout));

	VkPushConstantRange push = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
								.offset = 0,
								.size = sizeof(EnvironmentPrefilterPush)};
	VkPipelineLayoutCreateInfo pipeline_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &r->environment_prefilter_set_layout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push};
	VK_CHECK(vkCreatePipelineLayout(r->device, &pipeline_layout_info, NULL,
									&r->environment_pipeline_layout));

	VkShaderModule to_cube_module =
		create_shader_module(r, SHADER_DIR "/environment_equirect_to_cube.comp.spv");
	VkShaderModule prefilter_module =
		create_shader_module(r, SHADER_DIR "/environment_prefilter.comp.spv");
	VkComputePipelineCreateInfo infos[2] = {
		{.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		 .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				   .stage = VK_SHADER_STAGE_COMPUTE_BIT,
				   .module = to_cube_module,
				   .pName = "main"},
		 .layout = r->environment_pipeline_layout},
		{.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		 .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				   .stage = VK_SHADER_STAGE_COMPUTE_BIT,
				   .module = prefilter_module,
				   .pName = "main"},
		 .layout = r->environment_pipeline_layout},
	};
	VkPipeline pipelines[2];
	VK_CHECK(vkCreateComputePipelines(r->device, VK_NULL_HANDLE, 2, infos, NULL, pipelines));
	r->environment_to_cube_pipeline = pipelines[0];
	r->environment_prefilter_pipeline = pipelines[1];
	vkDestroyShaderModule(r->device, to_cube_module, NULL);
	vkDestroyShaderModule(r->device, prefilter_module, NULL);
}

/* One-shot init-time compute submission (its own command buffer + fence, like
   upload.c) that bakes r->environment_equirect into r->environment_cube: mip 0
   is a direct equirect->cube resample, mips 1..ENV_CUBE_MIPS-1 are the GGX
   prefilter at increasing roughness. Not part of the per-frame render graph --
   runs once between create_environment_prefilter_pipelines and the first
   frame. Destroys the transient equirect texture and mip views when done;
   r->environment_cube and its pipelines/layout persist for the renderer's
   lifetime. */
static void environment_prefilter(Renderer *r)
{
	/* Guarantee the equirect upload (create_environment) is fully visible on
	   the host before this queue submission touches it -- simpler than
	   reasoning about cross-submission barrier stage scopes for a one-shot
	   init step that is not on any hot path. */
	upload_wait_idle(r->upload);

	VkImageView mip_views[ENV_CUBE_MIPS];
	VkDescriptorSet sets[ENV_CUBE_MIPS];
	for (uint32_t mip = 0; mip < ENV_CUBE_MIPS; ++mip)
		mip_views[mip] = texture_create_storage_mip_view(r->device, &r->environment_cube, mip);
	VkImageView cube_mip0_view = texture_create_cube_view(r->device, &r->environment_cube, 0, 1);

	VkDescriptorSetLayout set_layouts[ENV_CUBE_MIPS];
	for (uint32_t mip = 0; mip < ENV_CUBE_MIPS; ++mip)
		set_layouts[mip] = r->environment_prefilter_set_layout;
	VkDescriptorSetAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
										 .descriptorPool = r->descriptor_pool,
										 .descriptorSetCount = ENV_CUBE_MIPS,
										 .pSetLayouts = set_layouts};
	VK_CHECK(vkAllocateDescriptorSets(r->device, &alloc, sets));

	VkDescriptorImageInfo sampled[ENV_CUBE_MIPS], storage[ENV_CUBE_MIPS];
	VkWriteDescriptorSet writes[ENV_CUBE_MIPS * 2];
	for (uint32_t mip = 0; mip < ENV_CUBE_MIPS; ++mip)
	{
		sampled[mip] = (VkDescriptorImageInfo){
			.sampler = mip == 0 ? r->environment_equirect.sampler : r->environment_cube.sampler,
			.imageView = mip == 0 ? r->environment_equirect.view : cube_mip0_view,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
		storage[mip] = (VkDescriptorImageInfo){.imageView = mip_views[mip],
											   .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
		writes[mip * 2] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
												 .dstSet = sets[mip],
												 .dstBinding = 0,
												 .descriptorCount = 1,
												 .descriptorType =
													 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
												 .pImageInfo = &sampled[mip]};
		writes[mip * 2 + 1] = (VkWriteDescriptorSet){.sType =
														 VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
													 .dstSet = sets[mip],
													 .dstBinding = 1,
													 .descriptorCount = 1,
													 .descriptorType =
														 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
													 .pImageInfo = &storage[mip]};
	}
	vkUpdateDescriptorSets(r->device, ENV_CUBE_MIPS * 2, writes, 0, NULL);

	VkCommandBufferAllocateInfo cmd_alloc = {.sType =
												 VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
											 .commandPool = r->command_pool,
											 .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
											 .commandBufferCount = 1};
	VkCommandBuffer command;
	VK_CHECK(vkAllocateCommandBuffers(r->device, &cmd_alloc, &command));
	VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
									  .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
	VK_CHECK(vkBeginCommandBuffer(command, &begin));

	environment_mip_barrier(command, &r->environment_cube, 0, ENV_CUBE_MIPS,
							VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
							VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);

	EnvironmentPrefilterPush push = {.roughness = 0.0f, .face_size = (float)ENV_FACE_SIZE};
	vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, r->environment_to_cube_pipeline);
	vkCmdPushConstants(command, r->environment_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
					   sizeof(push), &push);
	vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, r->environment_pipeline_layout,
							0, 1, &sets[0], 0, NULL);
	uint32_t groups0 = (ENV_FACE_SIZE + 7u) / 8u;
	vkCmdDispatch(command, groups0, groups0, 6);

	environment_mip_barrier(command, &r->environment_cube, 0, 1, VK_IMAGE_LAYOUT_GENERAL,
							VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

	vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, r->environment_prefilter_pipeline);
	for (uint32_t mip = 1; mip < ENV_CUBE_MIPS; ++mip)
	{
		uint32_t face_size = ENV_FACE_SIZE >> mip;
		EnvironmentPrefilterPush mip_push = {.roughness = (float)mip / (float)(ENV_CUBE_MIPS - 1u),
											 .face_size = (float)face_size};
		vkCmdPushConstants(command, r->environment_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
						   sizeof(mip_push), &mip_push);
		vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
								r->environment_pipeline_layout, 0, 1, &sets[mip], 0, NULL);
		uint32_t groups = (face_size + 7u) / 8u;
		vkCmdDispatch(command, groups, groups, 6);
	}

	environment_mip_barrier(command, &r->environment_cube, 0, 1,
							VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
							VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
							VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
	environment_mip_barrier(command, &r->environment_cube, 1, ENV_CUBE_MIPS - 1,
							VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
							VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
							VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
	r->environment_cube.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	VK_CHECK(vkEndCommandBuffer(command));
	VkFenceCreateInfo fence_info = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	VkFence fence;
	VK_CHECK(vkCreateFence(r->device, &fence_info, NULL, &fence));
	VkSubmitInfo submit = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &command};
	VK_CHECK(vkQueueSubmit(r->graphics_queue, 1, &submit, fence));
	VK_CHECK(vkWaitForFences(r->device, 1, &fence, VK_TRUE, UINT64_MAX));

	vkDestroyFence(r->device, fence, NULL);
	vkFreeCommandBuffers(r->device, r->command_pool, 1, &command);
	VK_CHECK(vkFreeDescriptorSets(r->device, r->descriptor_pool, ENV_CUBE_MIPS, sets));
	vkDestroyImageView(r->device, cube_mip0_view, NULL);
	for (uint32_t mip = 0; mip < ENV_CUBE_MIPS; ++mip)
		vkDestroyImageView(r->device, mip_views[mip], NULL);
	texture_destroy(r->device, r->allocator, &r->environment_equirect);
}

/* Terrain and imported static meshes share every graphics-pipeline state except
   their shaders, so both are built from this one description. */
static void create_scene_pipeline(Renderer *r, const char *vert_name, const char *frag_name,
								  VkPipeline *out_pipeline)
{
	char vert_path[1024], frag_path[1024];
	snprintf(vert_path, sizeof(vert_path), "%s/%s.spv", SHADER_DIR, vert_name);
	snprintf(frag_path, sizeof(frag_path), "%s/%s.spv", SHADER_DIR, frag_name);
	VkShaderModule vert = create_shader_module(r, vert_path);
	VkShaderModule frag = create_shader_module(r, frag_path);
	VkPipelineShaderStageCreateInfo stages[2] = {
		{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		 .stage = VK_SHADER_STAGE_VERTEX_BIT,
		 .module = vert,
		 .pName = "main"},
		{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		 .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
		 .module = frag,
		 .pName = "main"}};
	VkVertexInputBindingDescription binding = mesh_binding_description();
	uint32_t attribute_count;
	const VkVertexInputAttributeDescription *attributes =
		mesh_attribute_descriptions(&attribute_count);
	VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding,
		.vertexAttributeDescriptionCount = attribute_count,
		.pVertexAttributeDescriptions = attributes};
	VkPipelineInputAssemblyStateCreateInfo assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
	VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1};
	VkPipelineRasterizationStateCreateInfo rasterizer = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f};
	VkPipelineMultisampleStateCreateInfo multisampling = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
	VkPipelineDepthStencilStateCreateInfo depth = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL};
	VkPipelineColorBlendAttachmentState blend_attachments[2] = {
		{.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
						   VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT}};
	blend_attachments[1] = blend_attachments[0];
	VkPipelineColorBlendStateCreateInfo blending = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 2,
		.pAttachments = blend_attachments};
	VkDynamicState dynamics[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dynamic = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamics};
	VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &assembly,
		.pViewportState = &viewport,
		.pRasterizationState = &rasterizer,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = &depth,
		.pColorBlendState = &blending,
		.pDynamicState = &dynamic,
		.layout = r->pipeline_layout,
		.renderPass = r->scene_render_pass,
		.subpass = 0};
	VK_CHECK(vkCreateGraphicsPipelines(r->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL,
									   out_pipeline));
	vkDestroyShaderModule(r->device, frag, NULL);
	vkDestroyShaderModule(r->device, vert, NULL);
}

static void create_terrain_pipeline(Renderer *r)
{
	create_scene_pipeline(r, "terrain.vert", "terrain.frag", &r->terrain_pipeline);
	create_scene_pipeline(r, "mesh.vert", "mesh.frag", &r->mesh_pipeline);
}

static void create_shadow_pipeline(Renderer *r)
{
	char path[1024];
	snprintf(path, sizeof(path), "%s/shadow.vert.spv", SHADER_DIR);
	VkShaderModule vert = create_shader_module(r, path);
	VkPipelineShaderStageCreateInfo stage = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.stage = VK_SHADER_STAGE_VERTEX_BIT,
		.module = vert,
		.pName = "main"};
	VkVertexInputBindingDescription binding = mesh_binding_description();
	uint32_t attribute_count;
	const VkVertexInputAttributeDescription *attributes =
		mesh_attribute_descriptions(&attribute_count);
	VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding,
		.vertexAttributeDescriptionCount = attribute_count,
		.pVertexAttributeDescriptions = attributes};
	VkPipelineInputAssemblyStateCreateInfo assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
	VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1};
	VkPipelineRasterizationStateCreateInfo rasterizer = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
		.depthBiasEnable = VK_TRUE};
	VkPipelineMultisampleStateCreateInfo multisampling = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
	VkPipelineDepthStencilStateCreateInfo depth = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL};
	VkPipelineColorBlendStateCreateInfo blending = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
	VkDynamicState dynamics[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
								 VK_DYNAMIC_STATE_DEPTH_BIAS};
	VkPipelineDynamicStateCreateInfo dynamic = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 3,
		.pDynamicStates = dynamics};
	VkGraphicsPipelineCreateInfo info = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
										 .stageCount = 1,
										 .pStages = &stage,
										 .pVertexInputState = &vertex_input,
										 .pInputAssemblyState = &assembly,
										 .pViewportState = &viewport,
										 .pRasterizationState = &rasterizer,
										 .pMultisampleState = &multisampling,
										 .pDepthStencilState = &depth,
										 .pColorBlendState = &blending,
										 .pDynamicState = &dynamic,
										 .layout = r->pipeline_layout,
										 .renderPass = r->shadow_render_pass,
										 .subpass = 0};
	VK_CHECK(
		vkCreateGraphicsPipelines(r->device, VK_NULL_HANDLE, 1, &info, NULL, &r->shadow_pipeline));
	vkDestroyShaderModule(r->device, vert, NULL);
}

static VkPipeline create_fullscreen_pipeline(Renderer *r, const char *fragment_shader,
											 VkRenderPass render_pass, VkPipelineLayout layout,
											 uint32_t attachment_count)
{
	char vert_path[1024], frag_path[1024];
	snprintf(vert_path, sizeof(vert_path), "%s/fullscreen.vert.spv", SHADER_DIR);
	snprintf(frag_path, sizeof(frag_path), "%s/%s.spv", SHADER_DIR, fragment_shader);
	VkShaderModule vert = create_shader_module(r, vert_path);
	VkShaderModule frag = create_shader_module(r, frag_path);
	VkPipelineShaderStageCreateInfo stages[2] = {
		{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		 .stage = VK_SHADER_STAGE_VERTEX_BIT,
		 .module = vert,
		 .pName = "main"},
		{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		 .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
		 .module = frag,
		 .pName = "main"},
	};
	VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
	VkPipelineInputAssemblyStateCreateInfo assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
	VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1};
	VkPipelineRasterizationStateCreateInfo rasterizer = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f};
	VkPipelineMultisampleStateCreateInfo multisampling = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
	VkPipelineColorBlendAttachmentState blend_attachments[2] = {0};
	for (uint32_t i = 0; i < attachment_count; ++i)
		blend_attachments[i].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
											  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	VkPipelineColorBlendStateCreateInfo blending = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = attachment_count,
		.pAttachments = blend_attachments};
	VkDynamicState dynamics[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dynamic = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamics};
	VkGraphicsPipelineCreateInfo info = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
										 .stageCount = 2,
										 .pStages = stages,
										 .pVertexInputState = &vertex_input,
										 .pInputAssemblyState = &assembly,
										 .pViewportState = &viewport,
										 .pRasterizationState = &rasterizer,
										 .pMultisampleState = &multisampling,
										 .pColorBlendState = &blending,
										 .pDynamicState = &dynamic,
										 .layout = layout,
										 .renderPass = render_pass,
										 .subpass = 0};
	VkPipeline pipeline;
	VK_CHECK(vkCreateGraphicsPipelines(r->device, VK_NULL_HANDLE, 1, &info, NULL, &pipeline));
	vkDestroyShaderModule(r->device, frag, NULL);
	vkDestroyShaderModule(r->device, vert, NULL);
	return pipeline;
}

static void create_post_pipelines(Renderer *r)
{
	r->atmosphere_composite_pipeline = create_fullscreen_pipeline(
		r, "atmosphere_composite.frag", r->composite_render_pass, r->display_pipeline_layout, 1);
	r->taa_pipeline = create_fullscreen_pipeline(r, "temporal_resolve.frag", r->taa_render_pass,
												 r->temporal_pipeline_layout, 2);
	r->tone_map_pipeline = create_fullscreen_pipeline(r, "tonemap.frag", r->display_render_pass,
													  r->temporal_pipeline_layout, 1);
}

static void create_swapchain(Renderer *r)
{
	VkSurfaceCapabilitiesKHR capabilities;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(r->physical_device, r->surface, &capabilities);
	uint32_t format_count = 0, present_count = 0;
	vkGetPhysicalDeviceSurfaceFormatsKHR(r->physical_device, r->surface, &format_count, NULL);
	VkSurfaceFormatKHR *formats = malloc(sizeof(*formats) * format_count);
	vkGetPhysicalDeviceSurfaceFormatsKHR(r->physical_device, r->surface, &format_count, formats);
	vkGetPhysicalDeviceSurfacePresentModesKHR(r->physical_device, r->surface, &present_count, NULL);
	VkPresentModeKHR *present_modes = malloc(sizeof(*present_modes) * present_count);
	vkGetPhysicalDeviceSurfacePresentModesKHR(r->physical_device, r->surface, &present_count,
											  present_modes);
	VkSurfaceFormatKHR chosen = formats[0];
	for (uint32_t i = 0; i < format_count; ++i)
		if (formats[i].format == VK_FORMAT_B8G8R8A8_SRGB &&
			formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			chosen = formats[i];
	VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
	for (uint32_t i = 0; i < present_count; ++i)
		if (present_modes[i] == VK_PRESENT_MODE_MAILBOX_KHR)
			present_mode = present_modes[i];
	free(formats);
	free(present_modes);

	if (capabilities.currentExtent.width != UINT32_MAX)
		r->swapchain_extent = capabilities.currentExtent;
	else
	{
		int width, height;
		SDL_GetWindowSizeInPixels(r->window, &width, &height);
		r->swapchain_extent.width = (uint32_t)width;
		r->swapchain_extent.height = (uint32_t)height;
		if (r->swapchain_extent.width < capabilities.minImageExtent.width)
			r->swapchain_extent.width = capabilities.minImageExtent.width;
		if (r->swapchain_extent.width > capabilities.maxImageExtent.width)
			r->swapchain_extent.width = capabilities.maxImageExtent.width;
		if (r->swapchain_extent.height < capabilities.minImageExtent.height)
			r->swapchain_extent.height = capabilities.minImageExtent.height;
		if (r->swapchain_extent.height > capabilities.maxImageExtent.height)
			r->swapchain_extent.height = capabilities.maxImageExtent.height;
	}
	uint32_t desired_count = capabilities.minImageCount + 1;
	if (capabilities.maxImageCount && desired_count > capabilities.maxImageCount)
		desired_count = capabilities.maxImageCount;
	uint32_t indices[] = {r->graphics_family, r->present_family};
	VkSwapchainCreateInfoKHR info = {.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
									 .surface = r->surface,
									 .minImageCount = desired_count,
									 .imageFormat = chosen.format,
									 .imageColorSpace = chosen.colorSpace,
									 .imageExtent = r->swapchain_extent,
									 .imageArrayLayers = 1,
									 .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
									 .preTransform = capabilities.currentTransform,
									 .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
									 .presentMode = present_mode,
									 .clipped = VK_TRUE};
	if (r->graphics_family != r->present_family)
	{
		info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		info.queueFamilyIndexCount = 2;
		info.pQueueFamilyIndices = indices;
	}
	else
		info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VK_CHECK(vkCreateSwapchainKHR(r->device, &info, NULL, &r->swapchain));
	r->swapchain_format = chosen.format;
	vkGetSwapchainImagesKHR(r->device, r->swapchain, &r->image_count, NULL);
	r->images = malloc(sizeof(*r->images) * r->image_count);
	r->image_views = malloc(sizeof(*r->image_views) * r->image_count);
	r->display_framebuffers = malloc(sizeof(*r->display_framebuffers) * r->image_count);
	vkGetSwapchainImagesKHR(r->device, r->swapchain, &r->image_count, r->images);
	for (uint32_t i = 0; i < r->image_count; ++i)
	{
		VkImageViewCreateInfo view = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
									  .image = r->images[i],
									  .viewType = VK_IMAGE_VIEW_TYPE_2D,
									  .format = r->swapchain_format,
									  .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
		VK_CHECK(vkCreateImageView(r->device, &view, NULL, &r->image_views[i]));
	}
	VkFormat depth_format = find_depth_format(r->physical_device);
	create_scene_render_pass(r, depth_format);
	create_display_render_pass(r);
	const VkFormat composite_formats[1] = {VK_FORMAT_R16G16B16A16_SFLOAT};
	const VkFormat taa_formats[2] = {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32_SFLOAT};
	r->composite_render_pass = create_color_post_render_pass(r, 1, composite_formats);
	r->taa_render_pass = create_color_post_render_pass(r, 2, taa_formats);
	create_terrain_pipeline(r);
	create_post_pipelines(r);
	r->hdr_color = texture_create_hdr_target(r->device, r->allocator, r->swapchain_extent.width,
											 r->swapchain_extent.height);
	r->motion = texture_create_motion_target(r->device, r->allocator, r->swapchain_extent.width,
											 r->swapchain_extent.height);
	r->depth =
		texture_create_sampled_depth_target(r->device, r->allocator, depth_format,
											r->swapchain_extent.width, r->swapchain_extent.height);
	r->composite_color = texture_create_hdr_target(
		r->device, r->allocator, r->swapchain_extent.width, r->swapchain_extent.height);
	for (uint32_t i = 0; i < 2; ++i)
	{
		r->taa_history[i] = texture_create_hdr_target(
			r->device, r->allocator, r->swapchain_extent.width, r->swapchain_extent.height);
		r->taa_history_depth[i] = texture_create_history_depth_target(
			r->device, r->allocator, r->swapchain_extent.width, r->swapchain_extent.height);
	}

	VkDescriptorSetAllocateInfo set_alloc = {.sType =
												 VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
											 .descriptorPool = r->descriptor_pool,
											 .descriptorSetCount = 1,
											 .pSetLayouts = &r->display_set_layout};
	VK_CHECK(vkAllocateDescriptorSets(r->device, &set_alloc, &r->display_set));
	VkDescriptorImageInfo display_images[2] = {
		{.sampler = r->hdr_color.sampler,
		 .imageView = r->hdr_color.view,
		 .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
		{.sampler = r->depth.sampler,
		 .imageView = r->depth.view,
		 .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL},
	};
	VkWriteDescriptorSet display_writes[2] = {
		{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		 .dstSet = r->display_set,
		 .dstBinding = 0,
		 .descriptorCount = 1,
		 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 .pImageInfo = &display_images[0]},
		{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		 .dstSet = r->display_set,
		 .dstBinding = 1,
		 .descriptorCount = 1,
		 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 .pImageInfo = &display_images[1]},
	};
	vkUpdateDescriptorSets(r->device, 2, display_writes, 0, NULL);

	VkDescriptorSetAllocateInfo temporal_alloc = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = r->descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &r->temporal_set_layout};
	VkDescriptorBufferInfo exposure_info = {
		.buffer = r->exposure_buffer.buffer, .offset = 0, .range = sizeof(TemporalExposure)};
	for (uint32_t destination = 0; destination < 2; ++destination)
	{
		VK_CHECK(
			vkAllocateDescriptorSets(r->device, &temporal_alloc, &r->temporal_set[destination]));
		uint32_t previous = destination ^ 1u;
		Texture *sampled_textures[6] = {&r->composite_color,
										&r->depth,
										&r->motion,
										&r->taa_history[previous],
										&r->taa_history_depth[previous],
										&r->taa_history[destination]};
		VkDescriptorImageInfo temporal_images[6];
		VkWriteDescriptorSet temporal_writes[7];
		for (uint32_t binding = 0; binding < 6; ++binding)
		{
			temporal_images[binding] = (VkDescriptorImageInfo){
				.sampler = sampled_textures[binding]->sampler,
				.imageView = sampled_textures[binding]->view,
				.imageLayout = binding == 1 ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
											: VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
			temporal_writes[binding] =
				(VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
									   .dstSet = r->temporal_set[destination],
									   .dstBinding = binding,
									   .descriptorCount = 1,
									   .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
									   .pImageInfo = &temporal_images[binding]};
		}
		temporal_writes[6] =
			(VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
								   .dstSet = r->temporal_set[destination],
								   .dstBinding = 6,
								   .descriptorCount = 1,
								   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
								   .pBufferInfo = &exposure_info};
		vkUpdateDescriptorSets(r->device, 7, temporal_writes, 0, NULL);
	}

	VkImageView scene_attachments[] = {r->hdr_color.view, r->motion.view, r->depth.view};
	VkFramebufferCreateInfo scene_fb = {.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
										.renderPass = r->scene_render_pass,
										.attachmentCount = 3,
										.pAttachments = scene_attachments,
										.width = r->swapchain_extent.width,
										.height = r->swapchain_extent.height,
										.layers = 1};
	VK_CHECK(vkCreateFramebuffer(r->device, &scene_fb, NULL, &r->scene_framebuffer));
	VkFramebufferCreateInfo composite_fb = {.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
											.renderPass = r->composite_render_pass,
											.attachmentCount = 1,
											.pAttachments = &r->composite_color.view,
											.width = r->swapchain_extent.width,
											.height = r->swapchain_extent.height,
											.layers = 1};
	VK_CHECK(vkCreateFramebuffer(r->device, &composite_fb, NULL, &r->composite_framebuffer));
	for (uint32_t i = 0; i < 2; ++i)
	{
		VkImageView attachments[2] = {r->taa_history[i].view, r->taa_history_depth[i].view};
		VkFramebufferCreateInfo fb = {.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
									  .renderPass = r->taa_render_pass,
									  .attachmentCount = 2,
									  .pAttachments = attachments,
									  .width = r->swapchain_extent.width,
									  .height = r->swapchain_extent.height,
									  .layers = 1};
		VK_CHECK(vkCreateFramebuffer(r->device, &fb, NULL, &r->taa_framebuffers[i]));
	}
	for (uint32_t i = 0; i < r->image_count; ++i)
	{
		VkImageView attachments[] = {r->image_views[i]};
		VkFramebufferCreateInfo fb = {.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
									  .renderPass = r->display_render_pass,
									  .attachmentCount = 1,
									  .pAttachments = attachments,
									  .width = r->swapchain_extent.width,
									  .height = r->swapchain_extent.height,
									  .layers = 1};
		VK_CHECK(vkCreateFramebuffer(r->device, &fb, NULL, &r->display_framebuffers[i]));
	}

#ifdef DEBUG_SHADER_DUMP
	/* Capacity covers every instrumented shader's worst-case overdraw for a
	   whole frame, not just one pass's worth of pixels. Recreated with the
	   swapchain on resize. */
	r->shader_dump_layer_budget = DUMP_LAYER_BUDGET_DEFAULT;
	const char *layer_budget_env = getenv("TERRAIN_DUMP_LAYERS");
	if (layer_budget_env)
	{
		int parsed = atoi(layer_budget_env);
		if (parsed > 0)
			r->shader_dump_layer_budget = (uint32_t)parsed;
	}
	r->shader_dump_capacity =
		r->swapchain_extent.width * r->swapchain_extent.height * r->shader_dump_layer_budget;
	VkDeviceSize dump_size = SHADER_DUMP_HEADER_WORDS * sizeof(uint32_t) +
							 (VkDeviceSize)r->shader_dump_capacity * sizeof(DumpRecord);
	r->shader_dump_buffer = gpu_buffer_create(
		r->device, r->allocator, dump_size,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	uint32_t *dump_header = r->shader_dump_buffer.allocation.mapped;
	dump_header[0] = 0;						  /* count  */
	dump_header[1] = r->shader_dump_capacity; /* capacity */
	dump_header[2] = 0;
	dump_header[3] = 0;
	VkDescriptorBufferInfo dump_info = {
		.buffer = r->shader_dump_buffer.buffer, .offset = 0, .range = dump_size};
	for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
	{
		VkWriteDescriptorSet dump_write = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
										   .dstSet = r->frame_set[i],
										   .dstBinding = 3,
										   .descriptorCount = 1,
										   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
										   .pBufferInfo = &dump_info};
		vkUpdateDescriptorSets(r->device, 1, &dump_write, 0, NULL);
	}
#endif
	r->history_index = 0;
	r->temporal_history_valid = false;
}

static void destroy_swapchain(Renderer *r)
{
#ifdef DEBUG_SHADER_DUMP
	gpu_buffer_destroy(r->device, r->allocator, &r->shader_dump_buffer);
#endif
	renderer_free_material_set(r, r->display_set);
	r->display_set = VK_NULL_HANDLE;
	for (uint32_t i = 0; i < 2; ++i)
	{
		renderer_free_material_set(r, r->temporal_set[i]);
		r->temporal_set[i] = VK_NULL_HANDLE;
		vkDestroyFramebuffer(r->device, r->taa_framebuffers[i], NULL);
	}
	for (uint32_t i = 0; i < r->image_count; ++i)
		vkDestroyFramebuffer(r->device, r->display_framebuffers[i], NULL);
	vkDestroyFramebuffer(r->device, r->composite_framebuffer, NULL);
	vkDestroyFramebuffer(r->device, r->scene_framebuffer, NULL);
	for (uint32_t i = 0; i < 2; ++i)
	{
		texture_destroy(r->device, r->allocator, &r->taa_history_depth[i]);
		texture_destroy(r->device, r->allocator, &r->taa_history[i]);
	}
	texture_destroy(r->device, r->allocator, &r->composite_color);
	texture_destroy(r->device, r->allocator, &r->depth);
	texture_destroy(r->device, r->allocator, &r->motion);
	texture_destroy(r->device, r->allocator, &r->hdr_color);
	vkDestroyPipeline(r->device, r->tone_map_pipeline, NULL);
	vkDestroyPipeline(r->device, r->taa_pipeline, NULL);
	vkDestroyPipeline(r->device, r->atmosphere_composite_pipeline, NULL);
	vkDestroyPipeline(r->device, r->mesh_pipeline, NULL);
	vkDestroyPipeline(r->device, r->terrain_pipeline, NULL);
	vkDestroyRenderPass(r->device, r->display_render_pass, NULL);
	vkDestroyRenderPass(r->device, r->taa_render_pass, NULL);
	vkDestroyRenderPass(r->device, r->composite_render_pass, NULL);
	vkDestroyRenderPass(r->device, r->scene_render_pass, NULL);
	for (uint32_t i = 0; i < r->image_count; ++i)
		vkDestroyImageView(r->device, r->image_views[i], NULL);
	vkDestroySwapchainKHR(r->device, r->swapchain, NULL);
	free(r->display_framebuffers);
	free(r->image_views);
	free(r->images);
}

static void recreate_swapchain(Renderer *r)
{
	int width = 0, height = 0;
	SDL_GetWindowSizeInPixels(r->window, &width, &height);
	while (width == 0 || height == 0)
	{
		SDL_Event event;
		SDL_WaitEvent(&event);
		SDL_GetWindowSizeInPixels(r->window, &width, &height);
	}
	vkDeviceWaitIdle(r->device);
	destroy_swapchain(r);
	create_swapchain(r);
}

void renderer_reload_pipeline(Renderer *r)
{
	vkDeviceWaitIdle(r->device);
	vkDestroyPipeline(r->device, r->atmosphere_aerial_pipeline, NULL);
	vkDestroyPipeline(r->device, r->atmosphere_skyview_pipeline, NULL);
	vkDestroyPipeline(r->device, r->atmosphere_multiscattering_pipeline, NULL);
	vkDestroyPipeline(r->device, r->atmosphere_transmittance_pipeline, NULL);
	vkDestroyPipeline(r->device, r->shadow_pipeline, NULL);
	vkDestroyPipeline(r->device, r->exposure_pipeline, NULL);
	vkDestroyPipeline(r->device, r->luminance_histogram_pipeline, NULL);
	vkDestroyPipeline(r->device, r->tone_map_pipeline, NULL);
	vkDestroyPipeline(r->device, r->taa_pipeline, NULL);
	vkDestroyPipeline(r->device, r->atmosphere_composite_pipeline, NULL);
	vkDestroyPipeline(r->device, r->mesh_pipeline, NULL);
	vkDestroyPipeline(r->device, r->terrain_pipeline, NULL);
	create_shadow_pipeline(r);
	create_terrain_pipeline(r);
	create_post_pipelines(r);
	create_atmosphere_pipelines(r);
	create_temporal_compute_pipelines(r);
	r->atmosphere_static_ready = false;
	r->temporal_history_valid = false;
	printf("Shaders reloaded\n");
}

#ifdef DEBUG_SHADER_DUMP
/* Create every parent directory in `path` (mkdir -p on the dirname). */
static void ensure_parent_directory(const char *path)
{
	char *copy = strdup(path);
	if (!copy)
		return;
	char *dir = dirname(copy);
	/* Walk the components so nested paths (debug_dumps/foo) all get created. */
	char build[1024];
	size_t len = strlen(dir);
	if (len == 0 || len >= sizeof(build))
	{
		free(copy);
		return;
	}
	for (size_t i = 0; i <= len; ++i)
	{
		if (dir[i] == '/' || dir[i] == '\0')
		{
			if (i == 0)
			{
				build[0] = '/';
				build[1] = '\0';
				continue;
			}
			memcpy(build, dir, i);
			build[i] = '\0';
			mkdir(build, 0755);
		}
	}
	free(copy);
}

void renderer_clear_shader_dump(const char *path)
{
	ensure_parent_directory(path);
	FILE *file = fopen(path, "w");
	if (file)
	{
		fclose(file);
		printf("Cleared shader dump: %s\n", path);
	}
	else
		fprintf(stderr, "Could not clear shader dump %s\n", path);
}

/* Per-shader documentation for the 20 generic f0..f19 columns. Keep in sync
   with the shader_dump() call in each instrumented .frag file. */
static const char *const SHADER_DUMP_LEGEND[] = {
	"# legend: terrain f0-2=macro_tint f3=micro_luminance f4-6=surface_N f7=detail_weight "
	"f8-10=micro_tangent_N f11=grass_weight f12-14=base_color f15=roughness "
	"f16-18=final_HDR f19=NoL; mesh retains legacy terrain/mesh layout\n",
	"# legend: cube f0-2=normal f3=diffuse f4=lighting f5-7=out_color f8-19=_\n",
	"# legend: atmosphere_composite f0-1=texcoord f2=depth f3=branch(1=lut,2=pass,3=sky,4=aerial) "
	"f4-6=out_color f7=_ f8-10=view_dir f11=distance_km f12-14=scattering f15=near_weight "
	"f16-18=transmittance f19=w\n",
	"# legend: tonemap f0-1=texcoord f2=exposure f3=dither f4-6=resolved f7=avg_luminance "
	"f8-10=display_linear f11=debug_view f12-14=encoded f15=_ f16-18=out_color f19=_\n",
	"# legend: temporal_resolve f0-1=texcoord f2=depth f3=valid f4-5=velocity f6=motion_px "
	"f7=current_weight f8-10=current f11=_ f12-14=history f15=_ f16-18=resolved f19=_\n",
	"# legend: environment_ibl f0-2=reflection_direction f3=mip f4-6=raw_cube f7=NoV "
	"f8-10=ggx_specular_energy f11=reflection_visibility f12-14=unoccluded_specular f15=AO "
	"f16-18=final_specular f19=roughness\n",
	"# legend: material_detail f0=authored_roughness f1=effective_roughness f2=geometric_floor f3=geometric_variance "
	"f4=filtered_normal_length f5=mip_variance f6=mip_kernel f7=mip_roughness f8=AO_visibility f9=cavity_visibility "
	"f10=phase_D_active f11=normal_strength f12-14=final_pre_TAA_HDR f15=normal_map_LOD f16-19=_\n",
};

/* Optional CPU-side output filters, applied only at write time (the GPU always
   dumps everything). DUMP_SHADER=<name> keeps only that shader's records;
   DUMP_RECT=x0,y0,x1,y1 keeps only records inside that (inclusive) frag-coord
   rectangle. Both are re-read on every call so they can change between dumps. */
typedef struct
{
	bool has_shader;
	uint32_t shader_id;
	bool has_rect;
	float rect[4];
} ShaderDumpFilter;

static ShaderDumpFilter shader_dump_read_filters(void)
{
	ShaderDumpFilter filter = {0};
	const char *shader_env = getenv("DUMP_SHADER");
	if (shader_env)
	{
		for (uint32_t id = 0; id <= DUMP_SHADER_MATERIAL_DETAIL; ++id)
		{
			if (strcasecmp(shader_env, shader_dump_name(id)) == 0)
			{
				filter.has_shader = true;
				filter.shader_id = id;
				break;
			}
		}
		if (!filter.has_shader)
			fprintf(stderr, "DUMP_SHADER=%s does not match a known shader; ignoring\n", shader_env);
	}
	const char *rect_env = getenv("DUMP_RECT");
	if (rect_env && sscanf(rect_env, "%f,%f,%f,%f", &filter.rect[0], &filter.rect[1],
						   &filter.rect[2], &filter.rect[3]) == 4)
		filter.has_rect = true;
	return filter;
}

static bool shader_dump_record_passes(const DumpRecord *rec, const ShaderDumpFilter *filter)
{
	if (filter->has_shader && (uint32_t)rec->meta[0] != filter->shader_id)
		return false;
	if (filter->has_rect && (rec->meta[1] < filter->rect[0] || rec->meta[1] > filter->rect[2] ||
							 rec->meta[2] < filter->rect[1] || rec->meta[2] > filter->rect[3]))
		return false;
	return true;
}

uint32_t renderer_dump_shader_data(Renderer *r, const char *path, WorldPosition camera_position,
								   float camera_yaw, float camera_pitch, const char *metadata)
{
	/* The dump buffer holds the most recently submitted frame; wait for all GPU
	   work so the host read below sees complete, coherent records. */
	vkDeviceWaitIdle(r->device);
	const uint32_t *header = r->shader_dump_buffer.allocation.mapped;
	uint32_t count = header[0];
	uint32_t capacity = header[1];
	bool truncated = count > capacity;
	if (truncated)
		count = capacity;
	const DumpRecord *records = (const DumpRecord *)(header + SHADER_DUMP_HEADER_WORDS);
	ShaderDumpFilter filter = shader_dump_read_filters();

	ensure_parent_directory(path);
	FILE *file = fopen(path, "a");
	if (!file)
	{
		fprintf(stderr, "Could not open shader dump %s\n", path);
		return 0;
	}

	time_t now = time(NULL);
	char stamp[64];
	strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
	fprintf(file, "# dump %s  records=%u  view=%ux%u  layers=%u%s\n", stamp, count,
			r->swapchain_extent.width, r->swapchain_extent.height, r->shader_dump_layer_budget,
			truncated ? "  (TRUNCATED: capacity exceeded)" : "");
	fprintf(file,
			"# camera pos=%.6f,%.6f,%.6f  yaw=%.3f  pitch=%.3f  "
			"(replay: TERRAIN_DUMP_POS=\"%.6f %.6f %.6f\" TERRAIN_DUMP_YAW=%.3f "
			"TERRAIN_DUMP_PITCH=%.3f)\n",
			camera_position.x, camera_position.y, camera_position.z, camera_yaw, camera_pitch,
			camera_position.x, camera_position.y, camera_position.z, camera_yaw, camera_pitch);
	if (metadata && *metadata)
		fprintf(file, "# %s\n", metadata);
	for (size_t i = 0; i < sizeof(SHADER_DUMP_LEGEND) / sizeof(SHADER_DUMP_LEGEND[0]); ++i)
		fputs(SHADER_DUMP_LEGEND[i], file);
	fputs("shader,frag_x,frag_y,f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,"
		  "f19\n",
		  file);
	uint32_t written = 0;
	for (uint32_t i = 0; i < count; ++i)
	{
		const DumpRecord *rec = &records[i];
		if (!shader_dump_record_passes(rec, &filter))
			continue;
		fprintf(file,
				"%s,%.1f,%.1f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
				"%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
				shader_dump_name((uint32_t)rec->meta[0]), rec->meta[1], rec->meta[2], rec->v0[0],
				rec->v0[1], rec->v0[2], rec->v0[3], rec->v1[0], rec->v1[1], rec->v1[2], rec->v1[3],
				rec->v2[0], rec->v2[1], rec->v2[2], rec->v2[3], rec->v3[0], rec->v3[1], rec->v3[2],
				rec->v3[3], rec->v4[0], rec->v4[1], rec->v4[2], rec->v4[3]);
		++written;
	}
	fclose(file);
	printf("Wrote %u of %u shader records to %s%s\n", written, count, path,
		   truncated ? " (truncated)" : "");
	return written;
}
#endif

static void create_command_and_sync(Renderer *r)
{
	VkCommandPoolCreateInfo pool = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
									.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
									.queueFamilyIndex = r->graphics_family};
	VK_CHECK(vkCreateCommandPool(r->device, &pool, NULL, &r->command_pool));
	VkCommandBufferAllocateInfo allocation = {.sType =
												  VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
											  .commandPool = r->command_pool,
											  .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
											  .commandBufferCount = MAX_FRAMES_IN_FLIGHT};
	VK_CHECK(vkAllocateCommandBuffers(r->device, &allocation, r->command_buffers));
	for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
	{
		VkSemaphoreCreateInfo semaphore = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
		VkFenceCreateInfo fence = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
								   .flags = VK_FENCE_CREATE_SIGNALED_BIT};
		VK_CHECK(vkCreateSemaphore(r->device, &semaphore, NULL, &r->image_available[i]));
		VK_CHECK(vkCreateSemaphore(r->device, &semaphore, NULL, &r->render_finished[i]));
		VK_CHECK(vkCreateFence(r->device, &fence, NULL, &r->in_flight[i]));
	}
}

static void set_viewport_scissor(VkCommandBuffer command, VkExtent2D extent)
{
	VkViewport viewport = {0, 0, (float)extent.width, (float)extent.height, 0, 1};
	VkRect2D scissor = {{0, 0}, extent};
	vkCmdSetViewport(command, 0, 1, &viewport);
	vkCmdSetScissor(command, 0, 1, &scissor);
}

static void bind_draw(Renderer *r, VkCommandBuffer command, const RendererDraw *draw,
					  const DrawPushConstants *push)
{
	const Mesh *mesh = draw->mesh;
	VkDescriptorSet material_set = draw->material_set ? draw->material_set :
		(mesh->material_set ? mesh->material_set : r->fallback_material_set);
	vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, r->pipeline_layout, 1, 1,
							&material_set, 0, NULL);
	vkCmdPushConstants(command, r->pipeline_layout,
					   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(*push),
					   push);
	mesh_draw(command, mesh);
}

static void atmosphere_image_barrier(VkCommandBuffer command, Texture *texture,
									 VkImageLayout old_layout, VkPipelineStageFlags source_stage,
									 VkAccessFlags source_access,
									 VkPipelineStageFlags destination_stage,
									 VkAccessFlags destination_access)
{
	VkImageMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = source_access,
		.dstAccessMask = destination_access,
		.oldLayout = old_layout,
		.newLayout = VK_IMAGE_LAYOUT_GENERAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = texture->image,
		.subresourceRange = {texture->aspect, 0, texture->mip_levels, 0, texture->array_layers}};
	vkCmdPipelineBarrier(command, source_stage, destination_stage, 0, 0, NULL, 0, NULL, 1,
						 &barrier);
	texture->layout = VK_IMAGE_LAYOUT_GENERAL;
}

static void atmosphere_compute_barrier(VkCommandBuffer command, Texture *texture)
{
	atmosphere_image_barrier(command, texture, VK_IMAGE_LAYOUT_GENERAL,
							 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
							 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
}

static void record_atmosphere(Renderer *r, VkCommandBuffer command)
{
	bool build_static = !r->atmosphere_static_ready;
	Texture *all_luts[5] = {&r->atmosphere_transmittance, &r->atmosphere_multiscattering,
							&r->atmosphere_skyview, &r->atmosphere_aerial_scattering,
							&r->atmosphere_aerial_transmittance};
	if (build_static)
	{
		for (uint32_t i = 0; i < 5; ++i)
		{
			bool first_use = all_luts[i]->layout == VK_IMAGE_LAYOUT_UNDEFINED;
			atmosphere_image_barrier(command, all_luts[i], all_luts[i]->layout,
									 first_use ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
											   : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
													 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
									 first_use ? 0 : VK_ACCESS_SHADER_READ_BIT,
									 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
									 VK_ACCESS_SHADER_WRITE_BIT);
		}
	}
	else
	{
		/* Sky view and camera volume are camera/sun dependent. Synchronise last
		   frame's display sampling before overwriting them in-place. */
		for (uint32_t i = 2; i < 5; ++i)
			atmosphere_image_barrier(
				command, all_luts[i], VK_IMAGE_LAYOUT_GENERAL,
				VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
	}

	VkDescriptorSet compute_sets[2] = {r->frame_set[r->frame], r->atmosphere_set};
	vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, r->atmosphere_pipeline_layout,
							0, 2, compute_sets, 0, NULL);

	if (build_static)
	{
		vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
						  r->atmosphere_transmittance_pipeline);
		vkCmdDispatch(command, (256u + 7u) / 8u, (64u + 7u) / 8u, 1);
		atmosphere_compute_barrier(command, &r->atmosphere_transmittance);

		vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
						  r->atmosphere_multiscattering_pipeline);
		vkCmdDispatch(command, (32u + 7u) / 8u, (32u + 7u) / 8u, 1);
		atmosphere_compute_barrier(command, &r->atmosphere_multiscattering);
		r->atmosphere_static_ready = true;
	}

	vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, r->atmosphere_skyview_pipeline);
	vkCmdDispatch(command, (192u + 7u) / 8u, (108u + 7u) / 8u, 1);
	atmosphere_compute_barrier(command, &r->atmosphere_skyview);

	vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, r->atmosphere_aerial_pipeline);
	vkCmdDispatch(command, (32u + 3u) / 4u, (32u + 3u) / 4u, (32u + 3u) / 4u);

	/* Make all camera-dependent writes visible to terrain/display fragments.
	   The two volume targets share one barrier call each for explicitness. */
	atmosphere_image_barrier(command, &r->atmosphere_skyview, VK_IMAGE_LAYOUT_GENERAL,
							 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
							 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
	atmosphere_image_barrier(command, &r->atmosphere_aerial_scattering, VK_IMAGE_LAYOUT_GENERAL,
							 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
							 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
	atmosphere_image_barrier(command, &r->atmosphere_aerial_transmittance, VK_IMAGE_LAYOUT_GENERAL,
							 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
							 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
	if (build_static)
	{
		atmosphere_image_barrier(command, &r->atmosphere_transmittance, VK_IMAGE_LAYOUT_GENERAL,
								 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
								 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
		atmosphere_image_barrier(command, &r->atmosphere_multiscattering, VK_IMAGE_LAYOUT_GENERAL,
								 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
								 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
	}
}

static void record_commands(Renderer *r, uint32_t image_index, const FrameUniforms *frame,
							const RendererDraw *draws, uint32_t draw_count,
							const RendererDraw *shadow_draws, uint32_t shadow_draw_count)
{
	VkCommandBuffer command = r->command_buffers[r->frame];
	VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	VK_CHECK(vkBeginCommandBuffer(command, &begin));

	record_atmosphere(r, command);

	VkClearValue shadow_clear = {.depthStencil = {1.0f, 0}};
	for (uint32_t cascade = 0; cascade < SHADOW_CASCADE_COUNT; ++cascade)
	{
		VkRenderPassBeginInfo shadow = {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = r->shadow_render_pass,
			.framebuffer = r->shadow_framebuffers[cascade],
			.renderArea = {{0, 0}, {r->shadow_resolution, r->shadow_resolution}},
			.clearValueCount = 1,
			.pClearValues = &shadow_clear};
		vkCmdBeginRenderPass(command, &shadow, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, r->shadow_pipeline);
		set_viewport_scissor(command, (VkExtent2D){r->shadow_resolution, r->shadow_resolution});
		vkCmdSetDepthBias(command, frame->shadow_parameters.w, 0.0f, 1.75f);
		vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, r->pipeline_layout, 0, 1,
								&r->frame_set[r->frame], 0, NULL);
		for (uint32_t i = 0; i < shadow_draw_count; ++i)
		{
			DrawPushConstants push = shadow_draws[i].push;
			push.debug.x = (float)cascade;
			bind_draw(r, command, &shadow_draws[i], &push);
		}
		vkCmdEndRenderPass(command);
	}

#ifdef DEBUG_SHADER_DUMP
	/* Zero the record counter before the scene pass so the buffer holds exactly
	   this frame's fragments. Queue submits are serialised, so the previous
	   frame's writes are complete before this fill runs. */
	vkCmdFillBuffer(command, r->shader_dump_buffer.buffer, 0, sizeof(uint32_t), 0);
	VkBufferMemoryBarrier dump_reset_barrier = {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
												.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
												.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
																 VK_ACCESS_SHADER_WRITE_BIT,
												.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
												.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
												.buffer = r->shader_dump_buffer.buffer,
												.offset = 0,
												.size = VK_WHOLE_SIZE};
	vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
						 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 1, &dump_reset_barrier,
						 0, NULL);
#endif

	VkClearValue scene_clear[3] = {{.color = {{0.018f, 0.024f, 0.035f, 1.0f}}},
								   {.color = {{2.0f, 2.0f, 0.0f, 0.0f}}},
								   {.depthStencil = {0.0f, 0}}};
	VkRenderPassBeginInfo scene = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
								   .renderPass = r->scene_render_pass,
								   .framebuffer = r->scene_framebuffer,
								   .renderArea = {{0, 0}, r->swapchain_extent},
								   .clearValueCount = 3,
								   .pClearValues = scene_clear};
	vkCmdBeginRenderPass(command, &scene, VK_SUBPASS_CONTENTS_INLINE);
	set_viewport_scissor(command, r->swapchain_extent);
	/* Set 0: per-frame data. Set 1 + push constants: per terrain tile / mesh.
	   Both pipelines share this layout, so the frame and atmosphere sets bind
	   once regardless of which pipeline each draw selects. */
	vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, r->pipeline_layout, 0, 1,
							&r->frame_set[r->frame], 0, NULL);
	vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, r->pipeline_layout, 2, 1,
							&r->atmosphere_set, 0, NULL);
	VkPipeline bound_pipeline = VK_NULL_HANDLE;
	for (uint32_t i = 0; i < draw_count; ++i)
	{
		VkPipeline wanted = draws[i].static_mesh ? r->mesh_pipeline : r->terrain_pipeline;
		if (wanted != bound_pipeline)
		{
			vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, wanted);
			bound_pipeline = wanted;
		}
		bind_draw(r, command, &draws[i], &draws[i].push);
	}
	vkCmdEndRenderPass(command);

#ifdef DEBUG_SHADER_DUMP
	/* Make the fragment-shader writes visible to a host read after the fence. */
	VkBufferMemoryBarrier dump_host_barrier = {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
											   .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
											   .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
											   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
											   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
											   .buffer = r->shader_dump_buffer.buffer,
											   .offset = 0,
											   .size = VK_WHOLE_SIZE};
	vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
						 0, 0, NULL, 1, &dump_host_barrier, 0, NULL);
#endif

	VkClearValue hdr_clear = {.color = {{0.0f, 0.0f, 0.0f, 1.0f}}};
	VkRenderPassBeginInfo composite = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
									   .renderPass = r->composite_render_pass,
									   .framebuffer = r->composite_framebuffer,
									   .renderArea = {{0, 0}, r->swapchain_extent},
									   .clearValueCount = 1,
									   .pClearValues = &hdr_clear};
	vkCmdBeginRenderPass(command, &composite, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, r->atmosphere_composite_pipeline);
	set_viewport_scissor(command, r->swapchain_extent);
	VkDescriptorSet composite_sets[3] = {r->frame_set[r->frame], r->display_set, r->atmosphere_set};
	vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, r->display_pipeline_layout, 0,
							3, composite_sets, 0, NULL);
	vkCmdDraw(command, 3, 1, 0, 0);
	vkCmdEndRenderPass(command);

	uint32_t destination_history = r->history_index ^ 1u;
	VkClearValue taa_clear[2] = {{.color = {{0.0f, 0.0f, 0.0f, 1.0f}}},
								 {.color = {{0.0f, 0.0f, 0.0f, 0.0f}}}};
	VkRenderPassBeginInfo taa = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
								 .renderPass = r->taa_render_pass,
								 .framebuffer = r->taa_framebuffers[destination_history],
								 .renderArea = {{0, 0}, r->swapchain_extent},
								 .clearValueCount = 2,
								 .pClearValues = taa_clear};
	vkCmdBeginRenderPass(command, &taa, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, r->taa_pipeline);
	set_viewport_scissor(command, r->swapchain_extent);
	VkDescriptorSet temporal_sets[2] = {r->frame_set[r->frame],
										r->temporal_set[destination_history]};
	vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, r->temporal_pipeline_layout,
							0, 2, temporal_sets, 0, NULL);
	vkCmdDraw(command, 3, 1, 0, 0);
	vkCmdEndRenderPass(command);

	/* Preserve the adapted exposure while clearing this frame's count/bins. */
	vkCmdFillBuffer(command, r->exposure_buffer.buffer, offsetof(TemporalExposure, sample_count),
					sizeof(TemporalExposure) - offsetof(TemporalExposure, sample_count), 0);
	VkBufferMemoryBarrier histogram_clear = {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
											 .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
											 .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
															  VK_ACCESS_SHADER_WRITE_BIT,
											 .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
											 .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
											 .buffer = r->exposure_buffer.buffer,
											 .offset = 0,
											 .size = VK_WHOLE_SIZE};
	vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
						 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 1, &histogram_clear, 0,
						 NULL);
	vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, r->temporal_pipeline_layout, 0,
							2, temporal_sets, 0, NULL);
	vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, r->luminance_histogram_pipeline);
	vkCmdDispatch(command, (r->swapchain_extent.width + 15u) / 16u,
				  (r->swapchain_extent.height + 15u) / 16u, 1);
	VkBufferMemoryBarrier histogram_ready = histogram_clear;
	histogram_ready.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
						 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 1, &histogram_ready, 0,
						 NULL);
	vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, r->exposure_pipeline);
	vkCmdDispatch(command, 1, 1, 1);
	VkBufferMemoryBarrier exposure_ready = histogram_clear;
	exposure_ready.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	exposure_ready.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
						 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 1, &exposure_ready, 0,
						 NULL);

	VkClearValue display_clear = {.color = {{0.0f, 0.0f, 0.0f, 1.0f}}};
	VkRenderPassBeginInfo display = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
									 .renderPass = r->display_render_pass,
									 .framebuffer = r->display_framebuffers[image_index],
									 .renderArea = {{0, 0}, r->swapchain_extent},
									 .clearValueCount = 1,
									 .pClearValues = &display_clear};
	vkCmdBeginRenderPass(command, &display, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, r->tone_map_pipeline);
	set_viewport_scissor(command, r->swapchain_extent);
	vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, r->temporal_pipeline_layout,
							0, 2, temporal_sets, 0, NULL);
	vkCmdDraw(command, 3, 1, 0, 0);
	vkCmdEndRenderPass(command);
	r->history_index = destination_history;
	VK_CHECK(vkEndCommandBuffer(command));
}

void renderer_draw_frame(Renderer *r, const FrameUniforms *frame, const RendererDraw *draws,
						 uint32_t draw_count, const RendererDraw *shadow_draws,
						 uint32_t shadow_draw_count, bool resized)
{
	VkFence fence = r->in_flight[r->frame];
	VK_CHECK(vkWaitForFences(r->device, 1, &fence, VK_TRUE, UINT64_MAX));
	uint32_t image_index;
	VkResult acquired =
		vkAcquireNextImageKHR(r->device, r->swapchain, UINT64_MAX, r->image_available[r->frame],
							  VK_NULL_HANDLE, &image_index);
	if (acquired == VK_ERROR_OUT_OF_DATE_KHR)
	{
		recreate_swapchain(r);
		return;
	}
	if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR)
		VK_CHECK(acquired);
	VK_CHECK(vkResetFences(r->device, 1, &fence));

	/* Resize/out-of-date recreation owns the GPU history lifetime. Even if the
	   caller's camera history is continuous, never sample newly-created images. */
	FrameUniforms effective_frame = *frame;
	// effective_frame.temporal_parameters.x =
	//	frame->temporal_parameters.x > 0.5f && r->temporal_history_valid ? 1.0f : 0.0f;
	effective_frame.temporal_parameters.x = 0; // TEMP DEBUG

	effective_frame.temporal_parameters.z = r->swapchain_format == VK_FORMAT_B8G8R8A8_SRGB ||
													r->swapchain_format == VK_FORMAT_R8G8B8A8_SRGB
												? 1.0f
												: 0.0f;
	/* The frame's fence guarded this UBO's previous use, so it is safe to write. */
	memcpy(r->frame_ubo[r->frame].allocation.mapped, &effective_frame, sizeof(effective_frame));

	VK_CHECK(vkResetCommandBuffer(r->command_buffers[r->frame], 0));
	record_commands(r, image_index, &effective_frame, draws, draw_count, shadow_draws,
					shadow_draw_count);
	VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
						   .waitSemaphoreCount = 1,
						   .pWaitSemaphores = &r->image_available[r->frame],
						   .pWaitDstStageMask = &wait_stage,
						   .commandBufferCount = 1,
						   .pCommandBuffers = &r->command_buffers[r->frame],
						   .signalSemaphoreCount = 1,
						   .pSignalSemaphores = &r->render_finished[r->frame]};
	VK_CHECK(vkQueueSubmit(r->graphics_queue, 1, &submit, fence));
	VkPresentInfoKHR present = {.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
								.waitSemaphoreCount = 1,
								.pWaitSemaphores = &r->render_finished[r->frame],
								.swapchainCount = 1,
								.pSwapchains = &r->swapchain,
								.pImageIndices = &image_index};
	VkResult presented = vkQueuePresentKHR(r->present_queue, &present);
	if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR || resized)
		recreate_swapchain(r);
	else if (presented != VK_SUCCESS)
		VK_CHECK(presented);
	else
		r->temporal_history_valid = true;
	r->frame = (r->frame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void renderer_init(Renderer *r, SDL_Window *window, const RendererConfig *config)
{
	*r = (Renderer){0};
	r->window = window;
	r->shadow_quality = config ? config->shadow_quality : (ShadowQualitySettings){0};
	if (!r->shadow_quality.resolution)
		r->shadow_quality.resolution = SHADOW_MAP_RESOLUTION_DEFAULT;
	if (r->shadow_quality.filter_mode > SHADOW_FILTER_PCSS ||
		(r->shadow_quality.resolution != 2048u && r->shadow_quality.resolution != 4096u))
	{
		fprintf(stderr, "Invalid shadow settings: resolution must be 2048 or 4096\n");
		exit(EXIT_FAILURE);
	}
	r->shadow_resolution = r->shadow_quality.resolution;
	if (config && config->environment_path)
		snprintf(r->environment_path, sizeof(r->environment_path), "%s", config->environment_path);
	create_instance_and_device(r);
	VkPhysicalDeviceProperties shadow_properties;
	vkGetPhysicalDeviceProperties(r->physical_device, &shadow_properties);
	if (r->shadow_resolution > shadow_properties.limits.maxImageDimension2D)
	{
		fprintf(stderr, "Requested shadow resolution %u exceeds Vulkan maximum %u\n",
			r->shadow_resolution, shadow_properties.limits.maxImageDimension2D);
		exit(EXIT_FAILURE);
	}
	r->allocator = gpu_allocator_create(r->physical_device, r->device);
	r->upload = upload_context_create(r->device, r->allocator, r->graphics_queue,
									  r->graphics_family, UPLOAD_STAGING_CAPACITY);
	create_shadow_texture(r);
	create_atmosphere_textures(r);
	create_environment(r);
	create_descriptors(r);
	create_command_and_sync(r);
	create_pipeline_layout(r);
	create_atmosphere_pipelines(r);
	create_temporal_compute_pipelines(r);
	create_environment_prefilter_pipelines(r);
	environment_prefilter(r);
	create_shadow_render_pass(r, r->shadow_map.format);
	create_shadow_framebuffers(r);
	create_shadow_pipeline(r);
	create_swapchain(r);
}

void renderer_wait_idle(Renderer *r)
{
	vkDeviceWaitIdle(r->device);
}

float renderer_aspect(const Renderer *r)
{
	return (float)r->swapchain_extent.width / (float)r->swapchain_extent.height;
}

void renderer_shutdown(Renderer *r)
{
	vkDeviceWaitIdle(r->device);
	destroy_swapchain(r);
	vkDestroyPipeline(r->device, r->environment_prefilter_pipeline, NULL);
	vkDestroyPipeline(r->device, r->environment_to_cube_pipeline, NULL);
	vkDestroyPipelineLayout(r->device, r->environment_pipeline_layout, NULL);
	vkDestroyDescriptorSetLayout(r->device, r->environment_prefilter_set_layout, NULL);
	vkDestroyPipeline(r->device, r->exposure_pipeline, NULL);
	vkDestroyPipeline(r->device, r->luminance_histogram_pipeline, NULL);
	vkDestroyPipeline(r->device, r->atmosphere_aerial_pipeline, NULL);
	vkDestroyPipeline(r->device, r->atmosphere_skyview_pipeline, NULL);
	vkDestroyPipeline(r->device, r->atmosphere_multiscattering_pipeline, NULL);
	vkDestroyPipeline(r->device, r->atmosphere_transmittance_pipeline, NULL);
	vkDestroyPipeline(r->device, r->shadow_pipeline, NULL);
	for (uint32_t i = 0; i < SHADOW_CASCADE_COUNT; ++i)
		vkDestroyFramebuffer(r->device, r->shadow_framebuffers[i], NULL);
	vkDestroyRenderPass(r->device, r->shadow_render_pass, NULL);
	vkDestroyPipelineLayout(r->device, r->atmosphere_pipeline_layout, NULL);
	vkDestroyPipelineLayout(r->device, r->temporal_pipeline_layout, NULL);
	vkDestroyPipelineLayout(r->device, r->display_pipeline_layout, NULL);
	vkDestroyPipelineLayout(r->device, r->pipeline_layout, NULL);
	for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
	{
		vkDestroySemaphore(r->device, r->render_finished[i], NULL);
		vkDestroySemaphore(r->device, r->image_available[i], NULL);
		vkDestroyFence(r->device, r->in_flight[i], NULL);
		gpu_buffer_destroy(r->device, r->allocator, &r->frame_ubo[i]);
	}
	gpu_buffer_destroy(r->device, r->allocator, &r->exposure_buffer);
	gpu_buffer_destroy(r->device, r->allocator, &r->environment_ubo);
	vkDestroyCommandPool(r->device, r->command_pool, NULL);
	vkDestroyDescriptorPool(r->device, r->descriptor_pool, NULL);
	vkDestroyDescriptorSetLayout(r->device, r->atmosphere_set_layout, NULL);
	vkDestroyDescriptorSetLayout(r->device, r->temporal_set_layout, NULL);
	vkDestroyDescriptorSetLayout(r->device, r->display_set_layout, NULL);
	vkDestroyDescriptorSetLayout(r->device, r->material_set_layout, NULL);
	vkDestroyDescriptorSetLayout(r->device, r->frame_set_layout, NULL);
	texture_destroy(r->device, r->allocator, &r->fallback_texture);
	texture_destroy(r->device, r->allocator, &r->fallback_linear_texture);
	texture_destroy(r->device, r->allocator, &r->fallback_normal_texture);
	texture_destroy(r->device, r->allocator, &r->terrain_micro_ormh);
	texture_destroy(r->device, r->allocator, &r->terrain_micro_normal);
	texture_destroy(r->device, r->allocator, &r->terrain_micro_albedo);
	texture_destroy(r->device, r->allocator, &r->environment_cube);
	texture_destroy(r->device, r->allocator, &r->atmosphere_aerial_transmittance);
	texture_destroy(r->device, r->allocator, &r->atmosphere_aerial_scattering);
	texture_destroy(r->device, r->allocator, &r->atmosphere_skyview);
	texture_destroy(r->device, r->allocator, &r->atmosphere_multiscattering);
	texture_destroy(r->device, r->allocator, &r->atmosphere_transmittance);
	for (uint32_t i = 0; i < SHADOW_CASCADE_COUNT; ++i)
		vkDestroyImageView(r->device, r->shadow_layer_views[i], NULL);
	vkDestroySampler(r->device, r->shadow_raw_sampler, NULL);
	texture_destroy(r->device, r->allocator, &r->shadow_map);
	upload_context_destroy(r->upload);
	gpu_allocator_destroy(r->allocator);
	vkDestroyDevice(r->device, NULL);
	vkDestroySurfaceKHR(r->instance, r->surface, NULL);
	vkDestroyInstance(r->instance, NULL);
}
