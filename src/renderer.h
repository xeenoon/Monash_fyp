#pragma once

#include <SDL3/SDL.h>
#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

#include "gpu_memory.h"
#include "mesh.h"
#include "shadow_cascade.h"
#include "texture.h"
#include "upload.h"
#include <cglm/struct.h>

#define MAX_FRAMES_IN_FLIGHT 1

/* Per-frame shader data (descriptor set 0). All matrices operate on small,
   camera-relative floats. Absolute world positions never cross the CPU/GPU
   boundary. Current and previous state is present now so motion vectors can be
   added without changing the coordinate contract later. std140 layout. */
typedef struct
{
	mat4s projection;
	mat4s view;
	mat4s view_projection;
	mat4s inverse_view_projection;
	mat4s previous_projection;
	mat4s previous_view;
	mat4s previous_view_projection;
	mat4s local_to_camera_relative;
	mat4s previous_local_to_camera_relative;
	vec4s sun_direction;
	float time;
	float near_plane;
	float debug_view;
	float relight_strength;
	mat4s shadow_view_projection[SHADOW_CASCADE_COUNT];
	vec4s shadow_splits;
	vec4s shadow_parameters; /* normal bias m, PCF radius px, exposure, depth bias */
	vec4s sun_radiance;
	vec4s atmosphere_radii;
	vec4s atmosphere_rayleigh;
	vec4s atmosphere_mie_scatter;
	vec4s atmosphere_mie_extinct;
	vec4s atmosphere_absorption;
	vec4s atmosphere_ground;
	vec4s atmosphere_options;
	vec4s temporal_parameters; /* history valid, dt, sRGB swapchain, reserved */
	vec4s temporal_jitter;	   /* current NDC xy, previous NDC xy */
	vec4s shader_dump;		   /* dump enabled (x), reserved */
} FrameUniforms;

typedef struct
{
	float exposure;
	float average_luminance;
	uint32_t sample_count;
	uint32_t padding;
	uint32_t histogram[256];
} TemporalExposure;

/* Exactly 128 bytes, the Vulkan minimum guaranteed push-constant capacity.
   This is tile/draw data; camera and lighting remain in the frame UBO. */
typedef struct
{
	mat4s local_to_camera_relative;
	vec4s geometry;		/* tile span X/Z, elevation range, skirt depth */
	vec4s elevation_uv; /* scale U/V, bias U/V into the guttered raster */
	union
	{
		vec4s imagery_uv; /* terrain: scale U/V, bias U/V into imagery */
		vec4s material;	  /* static mesh: metallic factor, shading mode */
	};
	vec4s debug; /* LOD and XYZ surface-texture world phase */
} DrawPushConstants;

typedef struct
{
	const Mesh *mesh;
	DrawPushConstants push;
	/* Route through the triplanar static-mesh pipeline instead of the terrain
	   pipeline. Terrain tiles leave this false; imported meshes set it. */
	bool static_mesh;
} RendererDraw;

typedef struct Renderer
{
	SDL_Window *window;
	VkInstance instance;
	VkSurfaceKHR surface;
	VkPhysicalDevice physical_device;
	VkDevice device;
	uint32_t graphics_family;
	uint32_t present_family;
	VkQueue graphics_queue;
	VkQueue present_queue;
	float max_anisotropy;

	/* Reusable GPU resource infrastructure, shared by every subsystem. */
	GpuAllocator *allocator;
	UploadContext *upload;

	VkSwapchainKHR swapchain;
	VkFormat swapchain_format;
	VkExtent2D swapchain_extent;
	uint32_t image_count;
	VkImage *images;
	VkImageView *image_views;
	VkFramebuffer *display_framebuffers;
	Texture depth;
	Texture hdr_color;
	Texture motion;
	Texture composite_color;
	Texture taa_history[2];
	Texture taa_history_depth[2];
	VkFramebuffer scene_framebuffer;
	VkFramebuffer composite_framebuffer;
	VkFramebuffer taa_framebuffers[2];

	/* Descriptor roles. Set 0 is frame data plus the shared shadow array; set
	   1 is terrain material, scene composite input, or temporal history; set 2
	   supplies shared atmosphere LUTs. Compute uses the matching set-1 layout. */
	VkDescriptorSetLayout frame_set_layout;		 /* set 0 */
	VkDescriptorSetLayout material_set_layout;	 /* set 1 */
	VkDescriptorSetLayout display_set_layout;	 /* tone-map set 1 */
	VkDescriptorSetLayout temporal_set_layout;	 /* TAA/exposure set 1 */
	VkDescriptorSetLayout atmosphere_set_layout; /* graphics set 2 / compute set 1 */
	VkDescriptorPool descriptor_pool;
	GpuBuffer frame_ubo[MAX_FRAMES_IN_FLIGHT];
	GpuBuffer exposure_buffer;
	VkDescriptorSet frame_set[MAX_FRAMES_IN_FLIGHT];
	Texture fallback_texture;
	Texture terrain_detail_texture;
	VkDescriptorSet fallback_material_set;
	VkDescriptorSet display_set;
	VkDescriptorSet temporal_set[2];
	VkDescriptorSet atmosphere_set;

	VkRenderPass scene_render_pass;
	VkRenderPass display_render_pass;
	VkRenderPass composite_render_pass;
	VkRenderPass taa_render_pass;
	VkRenderPass shadow_render_pass;
	VkPipelineLayout pipeline_layout;
	VkPipelineLayout display_pipeline_layout;
	VkPipelineLayout temporal_pipeline_layout;
	VkPipelineLayout atmosphere_pipeline_layout;
	VkPipeline terrain_pipeline;
	VkPipeline mesh_pipeline; /* triplanar static-mesh pipeline (imported assets) */
	VkPipeline tone_map_pipeline;
	VkPipeline atmosphere_composite_pipeline;
	VkPipeline taa_pipeline;
	VkPipeline luminance_histogram_pipeline;
	VkPipeline exposure_pipeline;
	VkPipeline shadow_pipeline;
	VkPipeline atmosphere_transmittance_pipeline;
	VkPipeline atmosphere_multiscattering_pipeline;
	VkPipeline atmosphere_skyview_pipeline;
	VkPipeline atmosphere_aerial_pipeline;
	Texture atmosphere_transmittance;
	Texture atmosphere_multiscattering;
	Texture atmosphere_skyview;
	Texture atmosphere_aerial_scattering;
	Texture atmosphere_aerial_transmittance;
	bool atmosphere_static_ready;
	Texture shadow_map;
	VkSampler shadow_raw_sampler;
	VkImageView shadow_layer_views[SHADOW_CASCADE_COUNT];
	VkFramebuffer shadow_framebuffers[SHADOW_CASCADE_COUNT];
	VkCommandPool command_pool;
	VkCommandBuffer command_buffers[MAX_FRAMES_IN_FLIGHT];
	VkSemaphore image_available[MAX_FRAMES_IN_FLIGHT];
	VkSemaphore render_finished[MAX_FRAMES_IN_FLIGHT];
	VkFence in_flight[MAX_FRAMES_IN_FLIGHT];
	uint32_t frame;
	uint32_t history_index;
	bool temporal_history_valid;
#ifdef DEBUG_SHADER_DUMP
	/* Debug-only append buffer (set 0, binding 3): every instrumented fragment
	   shader appends one labelled record per invocation when frame.shader_dump.x
	   is set. Sized to swapchain_extent * shader_dump_layer_budget so overdraw
	   across all shaders in a frame fits; recreated with the swapchain. See
	   renderer_dump_shader_data. */
	GpuBuffer shader_dump_buffer;
	uint32_t shader_dump_capacity;
	uint32_t shader_dump_layer_budget;
#endif
} Renderer;

void renderer_init(Renderer *r, SDL_Window *window);
void renderer_shutdown(Renderer *r);
void renderer_wait_idle(Renderer *r);
float renderer_aspect(const Renderer *r);

/* Acquire, update the frame UBO, record (bind sets + draw mesh), submit, present.
   Recreates the swapchain on OUT_OF_DATE/SUBOPTIMAL or when `resized`. The shadow
   pass draws `shadow_draws` (camera-orientation-independent casters); the scene
   pass draws `draws` (the camera-visible set). */
void renderer_draw_frame(Renderer *r, const FrameUniforms *frame, const RendererDraw *draws,
						 uint32_t draw_count, const RendererDraw *shadow_draws,
						 uint32_t shadow_draw_count, bool resized);

/* Allocate a set-1 combined-image-sampler descriptor set bound to view+sampler.
   Meshes call this in mesh_upload to get a material set they can bind. */
VkDescriptorSet renderer_allocate_material_set(Renderer *r, VkImageView view, VkSampler sampler);
VkDescriptorSet renderer_allocate_pbr_set(Renderer *r, const Texture *albedo, const Texture *orm,
										  const Texture *normal_map);
VkDescriptorSet renderer_allocate_terrain_set(Renderer *r, VkImageView albedo_view,
											  VkSampler albedo_sampler, VkImageView elevation_view,
											  VkSampler elevation_sampler);
void renderer_free_material_set(Renderer *r, VkDescriptorSet set);

/* Manual shader reload: rebuild the graphics pipeline from the current .spv on
   disk at a frame boundary. Safe to call from the main loop (e.g. on a keypress). */
void renderer_reload_pipeline(Renderer *r);

#ifdef DEBUG_SHADER_DUMP
/* Wait for the GPU, read back the per-fragment records every instrumented
   shader wrote this frame (only populated when frame.shader_dump.x was set),
   and append them as CSV (with a timestamped header and per-shader legend) to
   `path`, creating parent directories as needed. Honours the optional
   DUMP_SHADER and DUMP_RECT env filters. Returns the number of records
   written after filtering. */
uint32_t renderer_dump_shader_data(Renderer *r, const char *path);
/* Truncate the dump file at `path` so subsequent dumps start fresh. */
void renderer_clear_shader_dump(const char *path);
#endif
