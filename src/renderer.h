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

/* Per-frame shader data (descriptor set 0). Kept in a UBO rather than push
   constants so it scales to sun/atmosphere/history state later. std140 layout:
   vec4-align every member and pad the tail. */
typedef struct {
    mat4s view_projection;
    vec4s camera_position;   /* xyz + pad */
    vec4s sun_direction;     /* xyz + pad */
    float time;
    float _pad[3];
} FrameUniforms;

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
       per-material/tile data (currently the albedo sampler). Sets are stable so
       later passes can slot shadow maps, atmosphere LUTs, and tile metadata into
       the same scheme. */
    VkDescriptorSetLayout frame_set_layout;     /* set 0 */
    VkDescriptorSetLayout material_set_layout;  /* set 1 */
    VkDescriptorPool descriptor_pool;
    GpuBuffer       frame_ubo[MAX_FRAMES_IN_FLIGHT];
    VkDescriptorSet frame_set[MAX_FRAMES_IN_FLIGHT];
    Texture         fallback_texture;
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
                         const Mesh *mesh, bool resized);

/* Allocate a set-1 combined-image-sampler descriptor set bound to view+sampler.
   Meshes call this in mesh_upload to get a material set they can bind. */
VkDescriptorSet renderer_allocate_material_set(Renderer *r, VkImageView view, VkSampler sampler);

/* Manual shader reload: rebuild the graphics pipeline from the current .spv on
   disk at a frame boundary. Safe to call from the main loop (e.g. on a keypress). */
void renderer_reload_pipeline(Renderer *r);
