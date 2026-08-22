#pragma once

#include <SDL3/SDL.h>
#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdint.h>

#include <cglm/struct.h>
#include "gpu_memory.h"
#include "upload.h"
#include "mesh.h"
#include "texture.h"

#define MAX_FRAMES_IN_FLIGHT 1

/* Per-frame shader data (descriptor set 0). All matrices operate on small,
   camera-relative floats. Absolute world positions never cross the CPU/GPU
   boundary. Current and previous state is present now so motion vectors can be
   added without changing the coordinate contract later. std140 layout. */
typedef struct {
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
} FrameUniforms;

/* Exactly 128 bytes, the Vulkan minimum guaranteed push-constant capacity.
   This is tile/draw data; camera and lighting remain in the frame UBO. */
typedef struct {
    mat4s local_to_camera_relative;
    vec4s geometry;      /* tile span X/Z, elevation range, skirt depth */
    vec4s elevation_uv;  /* scale U/V, bias U/V into the guttered raster */
    vec4s imagery_uv;    /* scale U/V, bias U/V into the guttered image  */
    vec4s debug;         /* LOD and XYZ surface-texture world phase */
} DrawPushConstants;

typedef struct {
    const Mesh *mesh;
    DrawPushConstants push;
} RendererDraw;

typedef struct Renderer {
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
    GpuAllocator  *allocator;
    UploadContext *upload;

    VkSwapchainKHR swapchain;
    VkFormat swapchain_format;
    VkExtent2D swapchain_extent;
    uint32_t image_count;
    VkImage *images;
    VkImageView *image_views;
    VkFramebuffer *framebuffers;
    Texture depth;

    /* Descriptor roles. Set 0 is per-frame data (camera/sun/time), set 1 is
       per-material/tile data (imagery, elevation, shared surface detail). Sets
       are stable so later passes can add shadow maps and atmosphere LUTs. */
    VkDescriptorSetLayout frame_set_layout;     /* set 0 */
    VkDescriptorSetLayout material_set_layout;  /* set 1 */
    VkDescriptorPool descriptor_pool;
    GpuBuffer       frame_ubo[MAX_FRAMES_IN_FLIGHT];
    VkDescriptorSet frame_set[MAX_FRAMES_IN_FLIGHT];
    Texture         fallback_texture;
    Texture         terrain_detail_texture;
    VkDescriptorSet fallback_material_set;

    VkRenderPass render_pass;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffers[MAX_FRAMES_IN_FLIGHT];
    VkSemaphore image_available[MAX_FRAMES_IN_FLIGHT];
    VkSemaphore render_finished[MAX_FRAMES_IN_FLIGHT];
    VkFence in_flight[MAX_FRAMES_IN_FLIGHT];
    uint32_t frame;
} Renderer;

void  renderer_init(Renderer *r, SDL_Window *window);
void  renderer_shutdown(Renderer *r);
void  renderer_wait_idle(Renderer *r);
float renderer_aspect(const Renderer *r);

/* Acquire, update the frame UBO, record (bind sets + draw mesh), submit, present.
   Recreates the swapchain on OUT_OF_DATE/SUBOPTIMAL or when `resized`. */
void renderer_draw_frame(Renderer *r, const FrameUniforms *frame,
                         const RendererDraw *draws, uint32_t draw_count,
                         bool resized);

/* Allocate a set-1 combined-image-sampler descriptor set bound to view+sampler.
   Meshes call this in mesh_upload to get a material set they can bind. */
VkDescriptorSet renderer_allocate_material_set(Renderer *r, VkImageView view, VkSampler sampler);
VkDescriptorSet renderer_allocate_terrain_set(Renderer *r,
    VkImageView albedo_view, VkSampler albedo_sampler,
    VkImageView elevation_view, VkSampler elevation_sampler);
void renderer_free_material_set(Renderer *r, VkDescriptorSet set);

/* Manual shader reload: rebuild the graphics pipeline from the current .spv on
   disk at a frame boundary. Safe to call from the main loop (e.g. on a keypress). */
void renderer_reload_pipeline(Renderer *r);
