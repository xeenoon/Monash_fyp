#pragma once

#include <SDL3/SDL.h>
#include <vulkan/vulkan.h>
#include <stdbool.h>
#include <stdint.h>

#include <cglm/struct.h>
#include "mesh.h"

#define MAX_FRAMES_IN_FLIGHT 1

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

    VkSwapchainKHR swapchain;
    VkFormat swapchain_format;
    VkExtent2D swapchain_extent;
    uint32_t image_count;
    VkImage *images;
    VkImageView *image_views;
    VkFramebuffer *framebuffers;
    VkImage depth_image;
    VkDeviceMemory depth_memory;
    VkImageView depth_view;
    VkFormat depth_format;

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

/* Acquire, record (view_projection push constant + draw mesh), submit, present.
   Recreates the swapchain on OUT_OF_DATE/SUBOPTIMAL or when `resized`. */
void renderer_draw_frame(Renderer *r, const mat4s *view_projection,
                         const Mesh *mesh, bool resized);

/* GPU allocation helpers, exported for mesh.c */
uint32_t renderer_find_memory_type(Renderer *r, uint32_t type_bits,
                                   VkMemoryPropertyFlags properties);
void     renderer_create_buffer(Renderer *r, VkDeviceSize size,
                                VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                                VkBuffer *buffer, VkDeviceMemory *memory);
