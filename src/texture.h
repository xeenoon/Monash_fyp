#pragma once

#include <vulkan/vulkan.h>

struct Renderer;

/* A GPU texture: a sampled image with its view and sampler. Loading (via
   stb_image) and all Vulkan image plumbing live in the texture module; the
   renderer only ever sees the finished view + sampler handles. */
typedef struct {
    VkImage        image;
    VkDeviceMemory memory;
    VkImageView    view;
    VkSampler      sampler;
} Texture;

/* Decode an image file (PNG/JPEG/...) into a sampled sRGB texture. */
void texture_load(struct Renderer *r, Texture *t, const char *path);

/* Build a 1x1 opaque-white texture, used as the fallback for untextured meshes
   so a single sampler-bound pipeline stays valid regardless of the mesh. */
void texture_create_white(struct Renderer *r, Texture *t);

void texture_destroy(struct Renderer *r, Texture *t);
