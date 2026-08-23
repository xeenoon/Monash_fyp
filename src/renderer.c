#include "renderer.h"
#include "file_utils.h"
#include "vk_common.h"

#include <SDL3/SDL_vulkan.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef DEBUG_SHADER_DUMP
#include <stdio.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <libgen.h>
#include <stdlib.h>

/* Layout of the set-0 binding-3 append buffer written by terrain.frag. A 16-byte
   header (count, capacity, 2x pad) precedes an array of 96-byte records. Kept in
   sync with struct DumpRecord in shaders/terrain.frag. */
#define SHADER_DUMP_HEADER_WORDS 4u
#define SHADER_DUMP_RECORD_FLOATS 24u   /* six vec4s */
typedef struct {
    float frag[4];    /* frag_coord.xy, distance_m, cascade */
    float cam_rel[4]; /* camera_relative_position.xyz, NoL */
    float world[4];   /* surface_position.xyz, visibility */
    float normal[4];  /* geometric_normal.xyz, receiver_bias */
    float shadow[4];  /* shadow.coordinate.xyz, lod level */
    float extra[4];   /* to_sun.xyz, grazing */
} ShaderDumpRecord;
_Static_assert(sizeof(ShaderDumpRecord) == SHADER_DUMP_RECORD_FLOATS * 4u,
               "ShaderDumpRecord must match std430 DumpRecord");
#endif

/* Keep the C UBO byte-for-byte compatible with shaders/common.glsl std140. */
_Static_assert(offsetof(FrameUniforms, projection) == 0, "FrameUniforms projection offset");
_Static_assert(offsetof(FrameUniforms, view) == 64, "FrameUniforms view offset");
_Static_assert(offsetof(FrameUniforms, view_projection) == 128, "FrameUniforms VP offset");
_Static_assert(offsetof(FrameUniforms, inverse_view_projection) == 192, "FrameUniforms inverse VP offset");
_Static_assert(offsetof(FrameUniforms, previous_projection) == 256, "FrameUniforms previous projection offset");
_Static_assert(offsetof(FrameUniforms, previous_view) == 320, "FrameUniforms previous view offset");
_Static_assert(offsetof(FrameUniforms, previous_view_projection) == 384, "FrameUniforms previous VP offset");
_Static_assert(offsetof(FrameUniforms, local_to_camera_relative) == 448, "FrameUniforms local transform offset");
_Static_assert(offsetof(FrameUniforms, previous_local_to_camera_relative) == 512, "FrameUniforms previous local transform offset");
_Static_assert(offsetof(FrameUniforms, sun_direction) == 576, "FrameUniforms sun offset");
_Static_assert(offsetof(FrameUniforms, time) == 592, "FrameUniforms time offset");
_Static_assert(offsetof(FrameUniforms, near_plane) == 596, "FrameUniforms near offset");
_Static_assert(offsetof(FrameUniforms, debug_view) == 600, "FrameUniforms debug view offset");
_Static_assert(offsetof(FrameUniforms, relight_strength) == 604, "FrameUniforms relight offset");
_Static_assert(offsetof(FrameUniforms, shadow_view_projection) == 608, "FrameUniforms shadow matrices offset");
_Static_assert(offsetof(FrameUniforms, shadow_splits) == 864, "FrameUniforms shadow splits offset");
_Static_assert(offsetof(FrameUniforms, shadow_parameters) == 880, "FrameUniforms shadow parameters offset");
_Static_assert(offsetof(FrameUniforms, sun_radiance) == 896, "FrameUniforms radiance offset");
_Static_assert(offsetof(FrameUniforms, atmosphere_radii) == 912, "FrameUniforms atmosphere radii offset");
_Static_assert(offsetof(FrameUniforms, atmosphere_rayleigh) == 928, "FrameUniforms Rayleigh offset");
_Static_assert(offsetof(FrameUniforms, atmosphere_mie_scatter) == 944, "FrameUniforms Mie scatter offset");
_Static_assert(offsetof(FrameUniforms, atmosphere_mie_extinct) == 960, "FrameUniforms Mie extinction offset");
_Static_assert(offsetof(FrameUniforms, atmosphere_absorption) == 976, "FrameUniforms ozone offset");
_Static_assert(offsetof(FrameUniforms, atmosphere_ground) == 992, "FrameUniforms atmosphere ground offset");
_Static_assert(offsetof(FrameUniforms, atmosphere_options) == 1008, "FrameUniforms atmosphere options offset");
_Static_assert(sizeof(FrameUniforms) == 1024, "FrameUniforms std140 size");
_Static_assert(sizeof(DrawPushConstants) == 128, "terrain push constant size");

/* Staging capacity for the upload ring: large enough for the 2048x2048 albedo
   (16 MiB) plus the terrain mesh in a single batch. */
#define UPLOAD_STAGING_CAPACITY (32u * 1024u * 1024u)
#define MAX_TEXTURE_SETS 256

VkDescriptorSet renderer_allocate_terrain_set(Renderer *r,
    VkImageView albedo_view, VkSampler albedo_sampler,
    VkImageView elevation_view, VkSampler elevation_sampler) {
    VkDescriptorSetAllocateInfo alloc = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = r->descriptor_pool, .descriptorSetCount = 1,
        .pSetLayouts = &r->material_set_layout };
    VkDescriptorSet set;
    VK_CHECK(vkAllocateDescriptorSets(r->device, &alloc, &set));
    VkDescriptorImageInfo images[3] = {
        { .sampler = albedo_sampler, .imageView = albedo_view,
          .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
        { .sampler = elevation_sampler, .imageView = elevation_view,
          .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
        { .sampler = r->terrain_detail_texture.sampler,
          .imageView = r->terrain_detail_texture.view,
          .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
    };
    VkWriteDescriptorSet writes[3] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = set, .dstBinding = 0, .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .pImageInfo = &images[0] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = set, .dstBinding = 1, .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .pImageInfo = &images[1] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = set, .dstBinding = 2, .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .pImageInfo = &images[2] },
    };
    vkUpdateDescriptorSets(r->device, 3, writes, 0, NULL);
    return set;
}

VkDescriptorSet renderer_allocate_material_set(Renderer *r, VkImageView view, VkSampler sampler) {
    return renderer_allocate_terrain_set(r, view, sampler, view, sampler);
}

void renderer_free_material_set(Renderer *r, VkDescriptorSet set) {
    if (r && set) VK_CHECK(vkFreeDescriptorSets(r->device, r->descriptor_pool, 1, &set));
}

/* Descriptor roles: set 0 = per-frame UBO; set 1 = pass-local material/HDR;
   set 2 = atmosphere LUTs for graphics. Compute sees that same atmosphere set
   as set 1, avoiding duplicate descriptors. */
static void create_descriptors(Renderer *r) {
    VkDescriptorSetLayoutBinding frame_bindings[4] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT |
                        VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
        { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
        { .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
    };
    uint32_t frame_binding_count = 3;
#ifdef DEBUG_SHADER_DUMP
    frame_binding_count = 4;
#endif
    VkDescriptorSetLayoutCreateInfo frame_layout = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = frame_binding_count, .pBindings = frame_bindings };
    VK_CHECK(vkCreateDescriptorSetLayout(r->device, &frame_layout, NULL, &r->frame_set_layout));

    VkDescriptorSetLayoutBinding material_bindings[3] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
        { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
    };
    VkDescriptorSetLayoutCreateInfo material_layout = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3, .pBindings = material_bindings };
    VK_CHECK(vkCreateDescriptorSetLayout(r->device, &material_layout, NULL, &r->material_set_layout));

    VkDescriptorSetLayoutBinding display_bindings[2] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
    };
    VkDescriptorSetLayoutCreateInfo display_layout = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2, .pBindings = display_bindings };
    VK_CHECK(vkCreateDescriptorSetLayout(r->device, &display_layout, NULL,
                                         &r->display_set_layout));

    VkDescriptorSetLayoutBinding atmosphere_bindings[10];
    for (uint32_t i = 0; i < 10; ++i) atmosphere_bindings[i] =
        (VkDescriptorSetLayoutBinding){ .binding = i, .descriptorCount = 1,
            .descriptorType = i < 5 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                    : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT |
                          (i < 5 ? VK_SHADER_STAGE_FRAGMENT_BIT : 0) };
    VkDescriptorSetLayoutCreateInfo atmosphere_layout = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 10, .pBindings = atmosphere_bindings };
    VK_CHECK(vkCreateDescriptorSetLayout(r->device, &atmosphere_layout, NULL,
                                         &r->atmosphere_set_layout));

    VkDescriptorPoolSize pool_sizes[4] = {
        { .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
        { .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = MAX_TEXTURE_SETS * 3u +
                             MAX_FRAMES_IN_FLIGHT * 2u + 8u },
        { .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 5u },
        { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = MAX_FRAMES_IN_FLIGHT } };
    uint32_t pool_size_count = 3;
#ifdef DEBUG_SHADER_DUMP
    pool_size_count = 4;
#endif
    VkDescriptorPoolCreateInfo pool = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .poolSizeCount = pool_size_count, .pPoolSizes = pool_sizes,
        .maxSets = MAX_FRAMES_IN_FLIGHT + MAX_TEXTURE_SETS + 2u };
    VK_CHECK(vkCreateDescriptorPool(r->device, &pool, NULL, &r->descriptor_pool));

    /* One persistently-mapped UBO + set per frame in flight. */
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        r->frame_ubo[i] = gpu_buffer_create(r->device, r->allocator, sizeof(FrameUniforms),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDescriptorSetAllocateInfo alloc = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = r->descriptor_pool, .descriptorSetCount = 1,
            .pSetLayouts = &r->frame_set_layout };
        VK_CHECK(vkAllocateDescriptorSets(r->device, &alloc, &r->frame_set[i]));
        VkDescriptorBufferInfo info = { .buffer = r->frame_ubo[i].buffer, .offset = 0,
            .range = sizeof(FrameUniforms) };
        VkDescriptorImageInfo shadow_images[2] = {
            { .sampler = r->shadow_map.sampler, .imageView = r->shadow_map.view,
              .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL },
            { .sampler = r->shadow_raw_sampler, .imageView = r->shadow_map.view,
              .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL },
        };
        VkWriteDescriptorSet writes[3] = {
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
              .dstSet = r->frame_set[i], .dstBinding = 0, .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              .pBufferInfo = &info },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
              .dstSet = r->frame_set[i], .dstBinding = 1, .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              .pImageInfo = &shadow_images[0] },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
              .dstSet = r->frame_set[i], .dstBinding = 2, .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              .pImageInfo = &shadow_images[1] },
        };
        vkUpdateDescriptorSets(r->device, 3, writes, 0, NULL);
    }

    texture_create_white(r->device, r->allocator, r->upload, &r->fallback_texture);
    texture_create_terrain_detail(r->device, r->allocator, r->upload,
                                  &r->terrain_detail_texture, r->max_anisotropy);
    r->fallback_material_set = renderer_allocate_material_set(r, r->fallback_texture.view,
                                                              r->fallback_texture.sampler);

    VkDescriptorSetAllocateInfo atmosphere_alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = r->descriptor_pool, .descriptorSetCount = 1,
        .pSetLayouts = &r->atmosphere_set_layout };
    VK_CHECK(vkAllocateDescriptorSets(r->device, &atmosphere_alloc,
                                      &r->atmosphere_set));
    Texture *textures[5] = {&r->atmosphere_transmittance,
        &r->atmosphere_multiscattering, &r->atmosphere_skyview,
        &r->atmosphere_aerial_scattering, &r->atmosphere_aerial_transmittance};
    VkDescriptorImageInfo sampled[5], storage[5];
    VkWriteDescriptorSet atmosphere_writes[10];
    for (uint32_t i = 0; i < 5; ++i) {
        sampled[i] = (VkDescriptorImageInfo){ .sampler = textures[i]->sampler,
            .imageView = textures[i]->view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
        storage[i] = (VkDescriptorImageInfo){ .imageView = textures[i]->view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
        atmosphere_writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = r->atmosphere_set, .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &sampled[i] };
        atmosphere_writes[i + 5] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = r->atmosphere_set, .dstBinding = i + 5,
            .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &storage[i] };
    }
    vkUpdateDescriptorSets(r->device, 10, atmosphere_writes, 0, NULL);
}

static VkFormat find_depth_format(VkPhysicalDevice physical_device) {
    const VkFormat candidates[] = {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
                                   VK_FORMAT_D24_UNORM_S8_UINT};
    for (size_t i = 0; i < sizeof(candidates)/sizeof(candidates[0]); ++i) {
        VkFormatProperties properties;
        vkGetPhysicalDeviceFormatProperties(physical_device, candidates[i], &properties);
        const VkFormatFeatureFlags required =
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if ((properties.optimalTilingFeatures & required) == required)
            return candidates[i];
    }
    fprintf(stderr, "No sampleable depth-attachment format found\n");
    exit(EXIT_FAILURE);
}

static VkFormat find_shadow_format(VkPhysicalDevice physical_device) {
    const VkFormat candidates[] = {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM,
                                   VK_FORMAT_D24_UNORM_S8_UINT};
    const VkFormatFeatureFlags required =
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        VkFormatProperties properties;
        vkGetPhysicalDeviceFormatProperties(physical_device, candidates[i], &properties);
        if ((properties.optimalTilingFeatures & required) == required)
            return candidates[i];
    }
    fprintf(stderr, "No filterable sampled depth format found for shadows\n");
    exit(EXIT_FAILURE);
}

static void create_shadow_texture(Renderer *r) {
    VkFormat format = find_shadow_format(r->physical_device);
    r->shadow_map = texture_create_shadow_array(r->device, r->allocator, format,
        SHADOW_MAP_RESOLUTION, SHADOW_CASCADE_COUNT);

    VkSamplerCreateInfo raw_sampler = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
        .maxLod = 0.0f };
    VK_CHECK(vkCreateSampler(r->device, &raw_sampler, NULL, &r->shadow_raw_sampler));

    for (uint32_t layer = 0; layer < SHADOW_CASCADE_COUNT; ++layer) {
        VkImageViewCreateInfo view = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = r->shadow_map.image, .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, layer, 1} };
        VK_CHECK(vkCreateImageView(r->device, &view, NULL,
                                   &r->shadow_layer_views[layer]));
    }
}

static void create_atmosphere_textures(Renderer *r) {
    /* Initial sizes are the proven Unreal demo defaults. The aerial volume is
       represented as a 32-layer 2D array because Vulkan samples array layers
       portably and our final pass performs the small z interpolation itself. */
    r->atmosphere_transmittance = texture_create_atmosphere_lut(
        r->device, r->allocator, 256, 64, 1);
    r->atmosphere_multiscattering = texture_create_atmosphere_lut(
        r->device, r->allocator, 32, 32, 1);
    r->atmosphere_skyview = texture_create_atmosphere_lut(
        r->device, r->allocator, 192, 108, 1);
    r->atmosphere_aerial_scattering = texture_create_atmosphere_lut(
        r->device, r->allocator, 32, 32, 32);
    r->atmosphere_aerial_transmittance = texture_create_atmosphere_lut(
        r->device, r->allocator, 32, 32, 32);
}

static VkShaderModule create_shader_module(Renderer *r, const char *path) {
    size_t size;
    uint8_t *code;
    FileReadResult result = file_read_all(path, &code, &size);
    if (result != FILE_READ_OK) {
        fprintf(stderr, "Could not read %s: %s\n", path,
                file_read_result_string(result));
        exit(EXIT_FAILURE);
    }
    VkShaderModuleCreateInfo info = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size, .pCode = (const uint32_t *)code };
    VkShaderModule module;
    VK_CHECK(vkCreateShaderModule(r->device, &info, NULL, &module));
    free(code);
    return module;
}

static bool device_has_swapchain(VkPhysicalDevice device) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(device, NULL, &count, NULL);
    VkExtensionProperties *extensions = malloc(sizeof(*extensions) * count);
    vkEnumerateDeviceExtensionProperties(device, NULL, &count, extensions);
    bool found = false;
    for (uint32_t i = 0; i < count; ++i)
        if (strcmp(extensions[i].extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) found = true;
    free(extensions);
    return found;
}

static bool find_queue_families(Renderer *r, VkPhysicalDevice device, uint32_t *graphics, uint32_t *present) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, NULL);
    VkQueueFamilyProperties *families = malloc(sizeof(*families) * count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families);
    bool has_graphics = false, has_present = false;
    for (uint32_t i = 0; i < count; ++i) {
        VkBool32 supported = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, r->surface, &supported);
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { *graphics = i; has_graphics = true; }
        if (supported) { *present = i; has_present = true; }
        if (has_graphics && has_present) break;
    }
    free(families);
    return has_graphics && has_present;
}

static void create_instance_and_device(Renderer *r) {
    uint32_t extension_count = 0;
    const char *const *extensions = SDL_Vulkan_GetInstanceExtensions(&extension_count);
    if (!extensions) { fprintf(stderr, "SDL Vulkan extensions: %s\n", SDL_GetError()); exit(EXIT_FAILURE); }

    VkApplicationInfo application = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Terrain Renderer", .applicationVersion = VK_MAKE_VERSION(1,0,0),
        .pEngineName = "none", .engineVersion = VK_MAKE_VERSION(1,0,0),
        .apiVersion = VK_API_VERSION_1_0 };
    VkInstanceCreateInfo instance_info = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &application, .enabledExtensionCount = extension_count,
        .ppEnabledExtensionNames = extensions };
    VK_CHECK(vkCreateInstance(&instance_info, NULL, &r->instance));
    if (!SDL_Vulkan_CreateSurface(r->window, r->instance, NULL, &r->surface)) {
        fprintf(stderr, "SDL_Vulkan_CreateSurface: %s\n", SDL_GetError()); exit(EXIT_FAILURE);
    }

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(r->instance, &device_count, NULL);
    if (!device_count) { fprintf(stderr, "No Vulkan device found\n"); exit(EXIT_FAILURE); }
    VkPhysicalDevice *devices = malloc(sizeof(*devices) * device_count);
    vkEnumeratePhysicalDevices(r->instance, &device_count, devices);
    for (uint32_t i = 0; i < device_count; ++i) {
        uint32_t graphics, present;
        if (device_has_swapchain(devices[i]) && find_queue_families(r, devices[i], &graphics, &present)) {
            r->physical_device = devices[i]; r->graphics_family = graphics; r->present_family = present;
            break;
        }
    }
    free(devices);
    if (r->physical_device == VK_NULL_HANDLE) { fprintf(stderr, "No suitable Vulkan device found\n"); exit(EXIT_FAILURE); }

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
    for (uint32_t i = 0; i < queue_count; ++i) queues[i] = (VkDeviceQueueCreateInfo){
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = families[i],
        .queueCount = 1, .pQueuePriorities = &priority };
    const char *device_extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    if (!supported.shaderStorageImageExtendedFormats) {
        fprintf(stderr, "RGBA16F storage images are required for atmosphere LUTs\n");
        exit(EXIT_FAILURE);
    }
    VkPhysicalDeviceFeatures features = {
        .samplerAnisotropy = supported.samplerAnisotropy,
        .shaderStorageImageExtendedFormats = VK_TRUE };
    VkDeviceCreateInfo device_info = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = queue_count, .pQueueCreateInfos = queues,
        .enabledExtensionCount = 1, .ppEnabledExtensionNames = device_extensions,
        .pEnabledFeatures = &features };
    VK_CHECK(vkCreateDevice(r->physical_device, &device_info, NULL, &r->device));
    vkGetDeviceQueue(r->device, r->graphics_family, 0, &r->graphics_queue);
    vkGetDeviceQueue(r->device, r->present_family, 0, &r->present_queue);
}

static void create_scene_render_pass(Renderer *r, VkFormat depth_format) {
    VkAttachmentDescription attachments[2] = {
        { .format = VK_FORMAT_R16G16B16A16_SFLOAT, .samples = VK_SAMPLE_COUNT_1_BIT,
          .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
          .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
          .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
          .finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
        { .format = depth_format, .samples = VK_SAMPLE_COUNT_1_BIT,
          .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
          .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
          .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
          .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL }
    };
    VkAttachmentReference color_ref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depth_ref = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass = { .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1, .pColorAttachments = &color_ref, .pDepthStencilAttachment = &depth_ref };
    VkSubpassDependency dependencies[2] = {
        { .srcSubpass = VK_SUBPASS_EXTERNAL, .dstSubpass = 0,
          .srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                          VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
          .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                          VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
          .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
          .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                           VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT },
        { .srcSubpass = 0, .dstSubpass = VK_SUBPASS_EXTERNAL,
          .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                          VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
          .dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
          .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                           VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_SHADER_READ_BIT },
    };
    VkRenderPassCreateInfo info = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 2, .pAttachments = attachments, .subpassCount = 1,
        .pSubpasses = &subpass, .dependencyCount = 2, .pDependencies = dependencies };
    VK_CHECK(vkCreateRenderPass(r->device, &info, NULL, &r->scene_render_pass));
}

static void create_display_render_pass(Renderer *r) {
    VkAttachmentDescription attachment = {
        .format = r->swapchain_format, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR };
    VkAttachmentReference color = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass = { .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1, .pColorAttachments = &color };
    VkSubpassDependency dependency = { .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT };
    VkRenderPassCreateInfo info = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &attachment, .subpassCount = 1,
        .pSubpasses = &subpass, .dependencyCount = 1,
        .pDependencies = &dependency };
    VK_CHECK(vkCreateRenderPass(r->device, &info, NULL, &r->display_render_pass));
}

static void create_shadow_render_pass(Renderer *r, VkFormat depth_format) {
    VkAttachmentDescription attachment = {
        .format = depth_format, .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL };
    VkAttachmentReference depth = {0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass = { .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .pDepthStencilAttachment = &depth };
    VkSubpassDependency dependencies[2] = {
        { .srcSubpass = VK_SUBPASS_EXTERNAL, .dstSubpass = 0,
          .srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
          .dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
          .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
          .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT },
        { .srcSubpass = 0, .dstSubpass = VK_SUBPASS_EXTERNAL,
          .srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
          .dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
          .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_SHADER_READ_BIT },
    };
    VkRenderPassCreateInfo info = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &attachment, .subpassCount = 1,
        .pSubpasses = &subpass, .dependencyCount = 2,
        .pDependencies = dependencies };
    VK_CHECK(vkCreateRenderPass(r->device, &info, NULL, &r->shadow_render_pass));
}

static void create_shadow_framebuffers(Renderer *r) {
    for (uint32_t layer = 0; layer < SHADOW_CASCADE_COUNT; ++layer) {
        VkFramebufferCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = r->shadow_render_pass, .attachmentCount = 1,
            .pAttachments = &r->shadow_layer_views[layer],
            .width = SHADOW_MAP_RESOLUTION, .height = SHADOW_MAP_RESOLUTION,
            .layers = 1 };
        VK_CHECK(vkCreateFramebuffer(r->device, &info, NULL,
                                     &r->shadow_framebuffers[layer]));
    }
}

/* Graphics share set 2 atmosphere resources. Compute uses frame set 0 plus the
   same atmosphere layout at set 1, matching the compact compute shaders. */
static void create_pipeline_layout(Renderer *r) {
    VkDescriptorSetLayout layouts[3] = {r->frame_set_layout,
        r->material_set_layout, r->atmosphere_set_layout};
    VkPushConstantRange push = { .stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
                                               VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0, .size = sizeof(DrawPushConstants) };
    VkPipelineLayoutCreateInfo layout_info = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 3, .pSetLayouts = layouts,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &push };
    VK_CHECK(vkCreatePipelineLayout(r->device, &layout_info, NULL, &r->pipeline_layout));

    VkDescriptorSetLayout display_layouts[3] = {
        r->frame_set_layout, r->display_set_layout, r->atmosphere_set_layout };
    VkPipelineLayoutCreateInfo display_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 3, .pSetLayouts = display_layouts };
    VK_CHECK(vkCreatePipelineLayout(r->device, &display_info, NULL,
                                    &r->display_pipeline_layout));

    VkDescriptorSetLayout compute_layouts[2] = {
        r->frame_set_layout, r->atmosphere_set_layout };
    VkPipelineLayoutCreateInfo compute_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2, .pSetLayouts = compute_layouts };
    VK_CHECK(vkCreatePipelineLayout(r->device, &compute_info, NULL,
                                    &r->atmosphere_pipeline_layout));
}

static VkPipeline create_compute_pipeline(Renderer *r, const char *shader_name) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.spv", SHADER_DIR, shader_name);
    VkShaderModule module = create_shader_module(r, path);
    VkPipelineShaderStageCreateInfo stage = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main" };
    VkComputePipelineCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stage, .layout = r->atmosphere_pipeline_layout };
    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(r->device, VK_NULL_HANDLE, 1, &info,
                                      NULL, &pipeline));
    vkDestroyShaderModule(r->device, module, NULL);
    return pipeline;
}

static void create_atmosphere_pipelines(Renderer *r) {
    r->atmosphere_transmittance_pipeline = create_compute_pipeline(
        r, "atmosphere_transmittance.comp");
    r->atmosphere_multiscattering_pipeline = create_compute_pipeline(
        r, "atmosphere_multiscattering.comp");
    r->atmosphere_skyview_pipeline = create_compute_pipeline(
        r, "atmosphere_skyview.comp");
    r->atmosphere_aerial_pipeline = create_compute_pipeline(
        r, "atmosphere_aerial.comp");
}

static void create_terrain_pipeline(Renderer *r) {
    char vert_path[1024], frag_path[1024];
    snprintf(vert_path, sizeof(vert_path), "%s/terrain.vert.spv", SHADER_DIR);
    snprintf(frag_path, sizeof(frag_path), "%s/terrain.frag.spv", SHADER_DIR);
    VkShaderModule vert = create_shader_module(r, vert_path);
    VkShaderModule frag = create_shader_module(r, frag_path);
    VkPipelineShaderStageCreateInfo stages[2] = {
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vert, .pName = "main" },
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = frag, .pName = "main" }
    };
    VkVertexInputBindingDescription binding = mesh_binding_description();
    uint32_t attribute_count;
    const VkVertexInputAttributeDescription *attributes = mesh_attribute_descriptions(&attribute_count);
    VkPipelineVertexInputStateCreateInfo vertex_input = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = attribute_count, .pVertexAttributeDescriptions = attributes };
    VkPipelineInputAssemblyStateCreateInfo assembly = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
    VkPipelineViewportStateCreateInfo viewport = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .scissorCount = 1 };
    VkPipelineRasterizationStateCreateInfo rasterizer = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f };
    VkPipelineMultisampleStateCreateInfo multisampling = { .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
    VkPipelineDepthStencilStateCreateInfo depth = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL };
    VkPipelineColorBlendAttachmentState blend_attachment = { .colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT };
    VkPipelineColorBlendStateCreateInfo blending = { .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &blend_attachment };
    VkDynamicState dynamics[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2, .pDynamicStates = dynamics };
    VkGraphicsPipelineCreateInfo pipeline_info = { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2, .pStages = stages, .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &assembly, .pViewportState = &viewport,
        .pRasterizationState = &rasterizer, .pMultisampleState = &multisampling,
        .pDepthStencilState = &depth, .pColorBlendState = &blending, .pDynamicState = &dynamic,
        .layout = r->pipeline_layout, .renderPass = r->scene_render_pass, .subpass = 0 };
    VK_CHECK(vkCreateGraphicsPipelines(r->device, VK_NULL_HANDLE, 1, &pipeline_info,
                                       NULL, &r->terrain_pipeline));
    vkDestroyShaderModule(r->device, frag, NULL);
    vkDestroyShaderModule(r->device, vert, NULL);
}

static void create_shadow_pipeline(Renderer *r) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/shadow.vert.spv", SHADER_DIR);
    VkShaderModule vert = create_shader_module(r, path);
    VkPipelineShaderStageCreateInfo stage = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vert, .pName = "main" };
    VkVertexInputBindingDescription binding = mesh_binding_description();
    uint32_t attribute_count;
    const VkVertexInputAttributeDescription *attributes =
        mesh_attribute_descriptions(&attribute_count);
    VkPipelineVertexInputStateCreateInfo vertex_input = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = attribute_count,
        .pVertexAttributeDescriptions = attributes };
    VkPipelineInputAssemblyStateCreateInfo assembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
    VkPipelineViewportStateCreateInfo viewport = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .scissorCount = 1 };
    VkPipelineRasterizationStateCreateInfo rasterizer = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f,
        .depthBiasEnable = VK_TRUE };
    VkPipelineMultisampleStateCreateInfo multisampling = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
    VkPipelineDepthStencilStateCreateInfo depth = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL };
    VkPipelineColorBlendStateCreateInfo blending = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    VkDynamicState dynamics[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                 VK_DYNAMIC_STATE_SCISSOR,
                                 VK_DYNAMIC_STATE_DEPTH_BIAS};
    VkPipelineDynamicStateCreateInfo dynamic = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 3, .pDynamicStates = dynamics };
    VkGraphicsPipelineCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 1, .pStages = &stage, .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &assembly, .pViewportState = &viewport,
        .pRasterizationState = &rasterizer, .pMultisampleState = &multisampling,
        .pDepthStencilState = &depth, .pColorBlendState = &blending,
        .pDynamicState = &dynamic, .layout = r->pipeline_layout,
        .renderPass = r->shadow_render_pass, .subpass = 0 };
    VK_CHECK(vkCreateGraphicsPipelines(r->device, VK_NULL_HANDLE, 1, &info,
                                       NULL, &r->shadow_pipeline));
    vkDestroyShaderModule(r->device, vert, NULL);
}

static void create_tone_map_pipeline(Renderer *r) {
    char vert_path[1024], frag_path[1024];
    snprintf(vert_path, sizeof(vert_path), "%s/fullscreen.vert.spv", SHADER_DIR);
    snprintf(frag_path, sizeof(frag_path), "%s/tonemap.frag.spv", SHADER_DIR);
    VkShaderModule vert = create_shader_module(r, vert_path);
    VkShaderModule frag = create_shader_module(r, frag_path);
    VkPipelineShaderStageCreateInfo stages[2] = {
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vert, .pName = "main" },
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = frag, .pName = "main" },
    };
    VkPipelineVertexInputStateCreateInfo vertex_input = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo assembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
    VkPipelineViewportStateCreateInfo viewport = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .scissorCount = 1 };
    VkPipelineRasterizationStateCreateInfo rasterizer = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f };
    VkPipelineMultisampleStateCreateInfo multisampling = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
    VkPipelineColorBlendAttachmentState blend_attachment = { .colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT };
    VkPipelineColorBlendStateCreateInfo blending = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &blend_attachment };
    VkDynamicState dynamics[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2, .pDynamicStates = dynamics };
    VkGraphicsPipelineCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2, .pStages = stages, .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &assembly, .pViewportState = &viewport,
        .pRasterizationState = &rasterizer, .pMultisampleState = &multisampling,
        .pColorBlendState = &blending, .pDynamicState = &dynamic,
        .layout = r->display_pipeline_layout,
        .renderPass = r->display_render_pass, .subpass = 0 };
    VK_CHECK(vkCreateGraphicsPipelines(r->device, VK_NULL_HANDLE, 1, &info,
                                       NULL, &r->tone_map_pipeline));
    vkDestroyShaderModule(r->device, frag, NULL);
    vkDestroyShaderModule(r->device, vert, NULL);
}

static void create_swapchain(Renderer *r) {
    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(r->physical_device, r->surface, &capabilities);
    uint32_t format_count = 0, present_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(r->physical_device, r->surface, &format_count, NULL);
    VkSurfaceFormatKHR *formats = malloc(sizeof(*formats) * format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(r->physical_device, r->surface, &format_count, formats);
    vkGetPhysicalDeviceSurfacePresentModesKHR(r->physical_device, r->surface, &present_count, NULL);
    VkPresentModeKHR *present_modes = malloc(sizeof(*present_modes) * present_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(r->physical_device, r->surface, &present_count, present_modes);
    VkSurfaceFormatKHR chosen = formats[0];
    for (uint32_t i = 0; i < format_count; ++i)
        if (formats[i].format == VK_FORMAT_B8G8R8A8_SRGB && formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            chosen = formats[i];
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    for (uint32_t i = 0; i < present_count; ++i)
        if (present_modes[i] == VK_PRESENT_MODE_MAILBOX_KHR) present_mode = present_modes[i];
    free(formats); free(present_modes);

    if (capabilities.currentExtent.width != UINT32_MAX) r->swapchain_extent = capabilities.currentExtent;
    else {
        int width, height;
        SDL_GetWindowSizeInPixels(r->window, &width, &height);
        r->swapchain_extent.width = (uint32_t)width;
        r->swapchain_extent.height = (uint32_t)height;
        if (r->swapchain_extent.width < capabilities.minImageExtent.width) r->swapchain_extent.width = capabilities.minImageExtent.width;
        if (r->swapchain_extent.width > capabilities.maxImageExtent.width) r->swapchain_extent.width = capabilities.maxImageExtent.width;
        if (r->swapchain_extent.height < capabilities.minImageExtent.height) r->swapchain_extent.height = capabilities.minImageExtent.height;
        if (r->swapchain_extent.height > capabilities.maxImageExtent.height) r->swapchain_extent.height = capabilities.maxImageExtent.height;
    }
    uint32_t desired_count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount && desired_count > capabilities.maxImageCount) desired_count = capabilities.maxImageCount;
    uint32_t indices[] = {r->graphics_family, r->present_family};
    VkSwapchainCreateInfoKHR info = { .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = r->surface, .minImageCount = desired_count, .imageFormat = chosen.format,
        .imageColorSpace = chosen.colorSpace, .imageExtent = r->swapchain_extent, .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, .preTransform = capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, .presentMode = present_mode, .clipped = VK_TRUE };
    if (r->graphics_family != r->present_family) {
        info.imageSharingMode = VK_SHARING_MODE_CONCURRENT; info.queueFamilyIndexCount = 2; info.pQueueFamilyIndices = indices;
    } else info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateSwapchainKHR(r->device, &info, NULL, &r->swapchain));
    r->swapchain_format = chosen.format;
    vkGetSwapchainImagesKHR(r->device, r->swapchain, &r->image_count, NULL);
    r->images = malloc(sizeof(*r->images) * r->image_count);
    r->image_views = malloc(sizeof(*r->image_views) * r->image_count);
    r->display_framebuffers = malloc(sizeof(*r->display_framebuffers) * r->image_count);
    vkGetSwapchainImagesKHR(r->device, r->swapchain, &r->image_count, r->images);
    for (uint32_t i = 0; i < r->image_count; ++i) {
        VkImageViewCreateInfo view = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = r->images[i], .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = r->swapchain_format,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1} };
        VK_CHECK(vkCreateImageView(r->device, &view, NULL, &r->image_views[i]));
    }
    VkFormat depth_format = find_depth_format(r->physical_device);
    create_scene_render_pass(r, depth_format);
    create_display_render_pass(r);
    create_terrain_pipeline(r);
    create_tone_map_pipeline(r);
    r->hdr_color = texture_create_hdr_target(r->device, r->allocator,
        r->swapchain_extent.width, r->swapchain_extent.height);
    r->depth = texture_create_sampled_depth_target(r->device, r->allocator,
        depth_format, r->swapchain_extent.width, r->swapchain_extent.height);

    VkDescriptorSetAllocateInfo set_alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = r->descriptor_pool, .descriptorSetCount = 1,
        .pSetLayouts = &r->display_set_layout };
    VK_CHECK(vkAllocateDescriptorSets(r->device, &set_alloc, &r->display_set));
    VkDescriptorImageInfo display_images[2] = {
        { .sampler = r->hdr_color.sampler, .imageView = r->hdr_color.view,
          .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
        { .sampler = r->depth.sampler, .imageView = r->depth.view,
          .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL },
    };
    VkWriteDescriptorSet display_writes[2] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = r->display_set, .dstBinding = 0, .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .pImageInfo = &display_images[0] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = r->display_set, .dstBinding = 1, .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .pImageInfo = &display_images[1] },
    };
    vkUpdateDescriptorSets(r->device, 2, display_writes, 0, NULL);

    VkImageView scene_attachments[] = {r->hdr_color.view, r->depth.view};
    VkFramebufferCreateInfo scene_fb = { .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass = r->scene_render_pass, .attachmentCount = 2,
        .pAttachments = scene_attachments, .width = r->swapchain_extent.width,
        .height = r->swapchain_extent.height, .layers = 1 };
    VK_CHECK(vkCreateFramebuffer(r->device, &scene_fb, NULL, &r->scene_framebuffer));
    for (uint32_t i = 0; i < r->image_count; ++i) {
        VkImageView attachments[] = {r->image_views[i]};
        VkFramebufferCreateInfo fb = { .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = r->display_render_pass, .attachmentCount = 1,
            .pAttachments = attachments,
            .width = r->swapchain_extent.width, .height = r->swapchain_extent.height, .layers = 1 };
        VK_CHECK(vkCreateFramebuffer(r->device, &fb, NULL, &r->display_framebuffers[i]));
    }

#ifdef DEBUG_SHADER_DUMP
    /* One record per screen pixel is the worst case (a fully terrain-covered
       view). Sized to the extent and recreated with the swapchain on resize. */
    r->shader_dump_capacity = r->swapchain_extent.width * r->swapchain_extent.height;
    VkDeviceSize dump_size = SHADER_DUMP_HEADER_WORDS * sizeof(uint32_t) +
                             (VkDeviceSize)r->shader_dump_capacity * sizeof(ShaderDumpRecord);
    r->shader_dump_buffer = gpu_buffer_create(r->device, r->allocator, dump_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    uint32_t *dump_header = r->shader_dump_buffer.allocation.mapped;
    dump_header[0] = 0;                        /* count  */
    dump_header[1] = r->shader_dump_capacity;  /* capacity */
    dump_header[2] = 0;
    dump_header[3] = 0;
    VkDescriptorBufferInfo dump_info = { .buffer = r->shader_dump_buffer.buffer,
        .offset = 0, .range = dump_size };
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        VkWriteDescriptorSet dump_write = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = r->frame_set[i], .dstBinding = 3, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dump_info };
        vkUpdateDescriptorSets(r->device, 1, &dump_write, 0, NULL);
    }
#endif
}

static void destroy_swapchain(Renderer *r) {
#ifdef DEBUG_SHADER_DUMP
    gpu_buffer_destroy(r->device, r->allocator, &r->shader_dump_buffer);
#endif
    renderer_free_material_set(r, r->display_set);
    r->display_set = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < r->image_count; ++i)
        vkDestroyFramebuffer(r->device, r->display_framebuffers[i], NULL);
    vkDestroyFramebuffer(r->device, r->scene_framebuffer, NULL);
    texture_destroy(r->device, r->allocator, &r->depth);
    texture_destroy(r->device, r->allocator, &r->hdr_color);
    vkDestroyPipeline(r->device, r->tone_map_pipeline, NULL);
    vkDestroyPipeline(r->device, r->terrain_pipeline, NULL);
    vkDestroyRenderPass(r->device, r->display_render_pass, NULL);
    vkDestroyRenderPass(r->device, r->scene_render_pass, NULL);
    for (uint32_t i = 0; i < r->image_count; ++i) vkDestroyImageView(r->device, r->image_views[i], NULL);
    vkDestroySwapchainKHR(r->device, r->swapchain, NULL);
    free(r->display_framebuffers); free(r->image_views); free(r->images);
}

static void recreate_swapchain(Renderer *r) {
    int width = 0, height = 0;
    SDL_GetWindowSizeInPixels(r->window, &width, &height);
    while (width == 0 || height == 0) {
        SDL_Event event;
        SDL_WaitEvent(&event);
        SDL_GetWindowSizeInPixels(r->window, &width, &height);
    }
    vkDeviceWaitIdle(r->device);
    destroy_swapchain(r);
    create_swapchain(r);
}

void renderer_reload_pipeline(Renderer *r) {
    vkDeviceWaitIdle(r->device);
    vkDestroyPipeline(r->device, r->atmosphere_aerial_pipeline, NULL);
    vkDestroyPipeline(r->device, r->atmosphere_skyview_pipeline, NULL);
    vkDestroyPipeline(r->device, r->atmosphere_multiscattering_pipeline, NULL);
    vkDestroyPipeline(r->device, r->atmosphere_transmittance_pipeline, NULL);
    vkDestroyPipeline(r->device, r->shadow_pipeline, NULL);
    vkDestroyPipeline(r->device, r->tone_map_pipeline, NULL);
    vkDestroyPipeline(r->device, r->terrain_pipeline, NULL);
    create_shadow_pipeline(r);
    create_terrain_pipeline(r);
    create_tone_map_pipeline(r);
    create_atmosphere_pipelines(r);
    r->atmosphere_static_ready = false;
    printf("Shaders reloaded\n");
}

#ifdef DEBUG_SHADER_DUMP
/* Create every parent directory in `path` (mkdir -p on the dirname). */
static void ensure_parent_directory(const char *path) {
    char *copy = strdup(path);
    if (!copy) return;
    char *dir = dirname(copy);
    /* Walk the components so nested paths (debug_dumps/foo) all get created. */
    char build[1024];
    size_t len = strlen(dir);
    if (len == 0 || len >= sizeof(build)) { free(copy); return; }
    for (size_t i = 0; i <= len; ++i) {
        if (dir[i] == '/' || dir[i] == '\0') {
            if (i == 0) { build[0] = '/'; build[1] = '\0'; continue; }
            memcpy(build, dir, i);
            build[i] = '\0';
            mkdir(build, 0755);
        }
    }
    free(copy);
}

void renderer_clear_shader_dump(const char *path) {
    ensure_parent_directory(path);
    FILE *file = fopen(path, "w");
    if (file) { fclose(file); printf("Cleared shader dump: %s\n", path); }
    else fprintf(stderr, "Could not clear shader dump %s\n", path);
}

uint32_t renderer_dump_shader_data(Renderer *r, const char *path) {
    /* The dump buffer holds the most recently submitted frame; wait for all GPU
       work so the host read below sees complete, coherent records. */
    vkDeviceWaitIdle(r->device);
    const uint32_t *header = r->shader_dump_buffer.allocation.mapped;
    uint32_t count = header[0];
    uint32_t capacity = header[1];
    bool truncated = count > capacity;
    if (truncated) count = capacity;
    const ShaderDumpRecord *records = (const ShaderDumpRecord *)(header + SHADER_DUMP_HEADER_WORDS);

    ensure_parent_directory(path);
    FILE *file = fopen(path, "a");
    if (!file) { fprintf(stderr, "Could not open shader dump %s\n", path); return 0; }

    time_t now = time(NULL);
    char stamp[64];
    strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
    fprintf(file, "# dump %s  records=%u  view=%ux%u%s\n", stamp, count,
            r->swapchain_extent.width, r->swapchain_extent.height,
            truncated ? "  (TRUNCATED: capacity exceeded)" : "");
    fprintf(file, "frag_x,frag_y,distance_m,cascade,"
                  "cam_x,cam_y,cam_z,NoL,"
                  "world_x,world_y,world_z,visibility,"
                  "nrm_x,nrm_y,nrm_z,receiver_bias,"
                  "shcoord_x,shcoord_y,shcoord_z,lod,"
                  "tosun_x,tosun_y,stored_depth,grazing\n");
    for (uint32_t i = 0; i < count; ++i) {
        const ShaderDumpRecord *rec = &records[i];
        fprintf(file,
            "%.1f,%.1f,%.3f,%d,%.3f,%.3f,%.3f,%.4f,"
            "%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,"
            "%.5f,%.5f,%.5f,%d,%.4f,%.4f,%.4f,%.4f\n",
            rec->frag[0], rec->frag[1], rec->frag[2], (int)rec->frag[3],
            rec->cam_rel[0], rec->cam_rel[1], rec->cam_rel[2], rec->cam_rel[3],
            rec->world[0], rec->world[1], rec->world[2], rec->world[3],
            rec->normal[0], rec->normal[1], rec->normal[2], rec->normal[3],
            rec->shadow[0], rec->shadow[1], rec->shadow[2], (int)rec->shadow[3],
            rec->extra[0], rec->extra[1], rec->extra[2], rec->extra[3]);
    }
    fclose(file);
    printf("Wrote %u shader records to %s%s\n", count, path,
           truncated ? " (truncated)" : "");
    return count;
}
#endif

static void create_command_and_sync(Renderer *r) {
    VkCommandPoolCreateInfo pool = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = r->graphics_family };
    VK_CHECK(vkCreateCommandPool(r->device, &pool, NULL, &r->command_pool));
    VkCommandBufferAllocateInfo allocation = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = r->command_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = MAX_FRAMES_IN_FLIGHT };
    VK_CHECK(vkAllocateCommandBuffers(r->device, &allocation, r->command_buffers));
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        VkSemaphoreCreateInfo semaphore = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        VkFenceCreateInfo fence = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT };
        VK_CHECK(vkCreateSemaphore(r->device, &semaphore, NULL, &r->image_available[i]));
        VK_CHECK(vkCreateSemaphore(r->device, &semaphore, NULL, &r->render_finished[i]));
        VK_CHECK(vkCreateFence(r->device, &fence, NULL, &r->in_flight[i]));
    }
}

static void set_viewport_scissor(VkCommandBuffer command, VkExtent2D extent) {
    VkViewport viewport = {0, 0, (float)extent.width, (float)extent.height, 0, 1};
    VkRect2D scissor = {{0,0}, extent};
    vkCmdSetViewport(command, 0, 1, &viewport);
    vkCmdSetScissor(command, 0, 1, &scissor);
}

static void bind_draw(Renderer *r, VkCommandBuffer command,
                      const RendererDraw *draw, const DrawPushConstants *push) {
    const Mesh *mesh = draw->mesh;
    VkDescriptorSet material_set = mesh->material_set ? mesh->material_set
                                                      : r->fallback_material_set;
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
        r->pipeline_layout, 1, 1, &material_set, 0, NULL);
    vkCmdPushConstants(command, r->pipeline_layout,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0, sizeof(*push), push);
    mesh_draw(command, mesh);
}

static void atmosphere_image_barrier(VkCommandBuffer command, Texture *texture,
                                     VkImageLayout old_layout,
                                     VkPipelineStageFlags source_stage,
                                     VkAccessFlags source_access,
                                     VkPipelineStageFlags destination_stage,
                                     VkAccessFlags destination_access) {
    VkImageMemoryBarrier barrier = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = source_access, .dstAccessMask = destination_access,
        .oldLayout = old_layout, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = texture->image,
        .subresourceRange = {texture->aspect, 0, texture->mip_levels,
                             0, texture->array_layers} };
    vkCmdPipelineBarrier(command, source_stage, destination_stage, 0,
                         0, NULL, 0, NULL, 1, &barrier);
    texture->layout = VK_IMAGE_LAYOUT_GENERAL;
}

static void atmosphere_compute_barrier(VkCommandBuffer command, Texture *texture) {
    atmosphere_image_barrier(command, texture, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
}

static void record_atmosphere(Renderer *r, VkCommandBuffer command) {
    bool build_static = !r->atmosphere_static_ready;
    Texture *all_luts[5] = {&r->atmosphere_transmittance,
        &r->atmosphere_multiscattering, &r->atmosphere_skyview,
        &r->atmosphere_aerial_scattering, &r->atmosphere_aerial_transmittance};
    if (build_static) {
        for (uint32_t i = 0; i < 5; ++i) {
            bool first_use = all_luts[i]->layout == VK_IMAGE_LAYOUT_UNDEFINED;
            atmosphere_image_barrier(command, all_luts[i], all_luts[i]->layout,
                first_use ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                          : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                first_use ? 0 : VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
        }
    } else {
        /* Sky view and camera volume are camera/sun dependent. Synchronise last
           frame's display sampling before overwriting them in-place. */
        for (uint32_t i = 2; i < 5; ++i)
            atmosphere_image_barrier(command, all_luts[i], VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    }

    VkDescriptorSet compute_sets[2] = {r->frame_set[r->frame], r->atmosphere_set};
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
        r->atmosphere_pipeline_layout, 0, 2, compute_sets, 0, NULL);

    if (build_static) {
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

    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                      r->atmosphere_skyview_pipeline);
    vkCmdDispatch(command, (192u + 7u) / 8u, (108u + 7u) / 8u, 1);
    atmosphere_compute_barrier(command, &r->atmosphere_skyview);

    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                      r->atmosphere_aerial_pipeline);
    vkCmdDispatch(command, (32u + 3u) / 4u, (32u + 3u) / 4u,
                  (32u + 3u) / 4u);

    /* Make all camera-dependent writes visible to terrain/display fragments.
       The two volume targets share one barrier call each for explicitness. */
    atmosphere_image_barrier(command, &r->atmosphere_skyview,
        VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT);
    atmosphere_image_barrier(command, &r->atmosphere_aerial_scattering,
        VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT);
    atmosphere_image_barrier(command, &r->atmosphere_aerial_transmittance,
        VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT);
    if (build_static) {
        atmosphere_image_barrier(command, &r->atmosphere_transmittance,
            VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT);
        atmosphere_image_barrier(command, &r->atmosphere_multiscattering,
            VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT);
    }
}

static void record_commands(Renderer *r, uint32_t image_index,
                            const FrameUniforms *frame,
                            const RendererDraw *draws, uint32_t draw_count) {
    VkCommandBuffer command = r->command_buffers[r->frame];
    VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    VK_CHECK(vkBeginCommandBuffer(command, &begin));

    record_atmosphere(r, command);

    VkClearValue shadow_clear = {.depthStencil = {1.0f, 0}};
    for (uint32_t cascade = 0; cascade < SHADOW_CASCADE_COUNT; ++cascade) {
        VkRenderPassBeginInfo shadow = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = r->shadow_render_pass,
            .framebuffer = r->shadow_framebuffers[cascade],
            .renderArea = {{0,0}, {SHADOW_MAP_RESOLUTION, SHADOW_MAP_RESOLUTION}},
            .clearValueCount = 1, .pClearValues = &shadow_clear };
        vkCmdBeginRenderPass(command, &shadow, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          r->shadow_pipeline);
        set_viewport_scissor(command,
            (VkExtent2D){SHADOW_MAP_RESOLUTION, SHADOW_MAP_RESOLUTION});
        vkCmdSetDepthBias(command, frame->shadow_parameters.w, 0.0f, 1.75f);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
            r->pipeline_layout, 0, 1, &r->frame_set[r->frame], 0, NULL);
        for (uint32_t i = 0; i < draw_count; ++i) {
            DrawPushConstants push = draws[i].push;
            push.debug.x = (float)cascade;
            bind_draw(r, command, &draws[i], &push);
        }
        vkCmdEndRenderPass(command);
    }

#ifdef DEBUG_SHADER_DUMP
    /* Zero the record counter before the scene pass so the buffer holds exactly
       this frame's fragments. Queue submits are serialised, so the previous
       frame's writes are complete before this fill runs. */
    vkCmdFillBuffer(command, r->shader_dump_buffer.buffer, 0, sizeof(uint32_t), 0);
    VkBufferMemoryBarrier dump_reset_barrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = r->shader_dump_buffer.buffer, .offset = 0, .size = VK_WHOLE_SIZE };
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 1, &dump_reset_barrier, 0, NULL);
#endif

    VkClearValue scene_clear[2] = {
        {.color = {{0.018f, 0.024f, 0.035f, 1.0f}}},
        {.depthStencil = {0.0f, 0}}
    };
    VkRenderPassBeginInfo scene = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = r->scene_render_pass, .framebuffer = r->scene_framebuffer,
        .renderArea = {{0,0}, r->swapchain_extent}, .clearValueCount = 2,
        .pClearValues = scene_clear };
    vkCmdBeginRenderPass(command, &scene, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      r->terrain_pipeline);
    set_viewport_scissor(command, r->swapchain_extent);
    /* Set 0: per-frame data. Set 1 + push constants: per terrain tile. */
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, r->pipeline_layout, 0, 1,
        &r->frame_set[r->frame], 0, NULL);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
        r->pipeline_layout, 2, 1, &r->atmosphere_set, 0, NULL);
    for (uint32_t i = 0; i < draw_count; ++i)
        bind_draw(r, command, &draws[i], &draws[i].push);
    vkCmdEndRenderPass(command);

#ifdef DEBUG_SHADER_DUMP
    /* Make the fragment-shader writes visible to a host read after the fence. */
    VkBufferMemoryBarrier dump_host_barrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = r->shader_dump_buffer.buffer, .offset = 0, .size = VK_WHOLE_SIZE };
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1, &dump_host_barrier, 0, NULL);
#endif

    VkClearValue display_clear = {.color = {{0.0f, 0.0f, 0.0f, 1.0f}}};
    VkRenderPassBeginInfo display = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = r->display_render_pass,
        .framebuffer = r->display_framebuffers[image_index],
        .renderArea = {{0,0}, r->swapchain_extent}, .clearValueCount = 1,
        .pClearValues = &display_clear };
    vkCmdBeginRenderPass(command, &display, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      r->tone_map_pipeline);
    set_viewport_scissor(command, r->swapchain_extent);
    VkDescriptorSet display_sets[3] = {r->frame_set[r->frame], r->display_set,
                                       r->atmosphere_set};
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
        r->display_pipeline_layout, 0, 3, display_sets, 0, NULL);
    vkCmdDraw(command, 3, 1, 0, 0);
    vkCmdEndRenderPass(command);
    VK_CHECK(vkEndCommandBuffer(command));
}

void renderer_draw_frame(Renderer *r, const FrameUniforms *frame,
                         const RendererDraw *draws, uint32_t draw_count,
                         bool resized) {
    VkFence fence = r->in_flight[r->frame];
    VK_CHECK(vkWaitForFences(r->device, 1, &fence, VK_TRUE, UINT64_MAX));
    uint32_t image_index;
    VkResult acquired = vkAcquireNextImageKHR(r->device, r->swapchain, UINT64_MAX,
        r->image_available[r->frame], VK_NULL_HANDLE, &image_index);
    if (acquired == VK_ERROR_OUT_OF_DATE_KHR) { recreate_swapchain(r); return; }
    if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) VK_CHECK(acquired);
    VK_CHECK(vkResetFences(r->device, 1, &fence));

    /* The frame's fence guarded this UBO's previous use, so it is safe to write. */
    memcpy(r->frame_ubo[r->frame].allocation.mapped, frame, sizeof(*frame));

    VK_CHECK(vkResetCommandBuffer(r->command_buffers[r->frame], 0));
    record_commands(r, image_index, frame, draws, draw_count);
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1, .pWaitSemaphores = &r->image_available[r->frame],
        .pWaitDstStageMask = &wait_stage, .commandBufferCount = 1,
        .pCommandBuffers = &r->command_buffers[r->frame], .signalSemaphoreCount = 1,
        .pSignalSemaphores = &r->render_finished[r->frame] };
    VK_CHECK(vkQueueSubmit(r->graphics_queue, 1, &submit, fence));
    VkPresentInfoKHR present = { .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1, .pWaitSemaphores = &r->render_finished[r->frame],
        .swapchainCount = 1, .pSwapchains = &r->swapchain, .pImageIndices = &image_index };
    VkResult presented = vkQueuePresentKHR(r->present_queue, &present);
    if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR || resized)
        recreate_swapchain(r);
    else if (presented != VK_SUCCESS) VK_CHECK(presented);
    r->frame = (r->frame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void renderer_init(Renderer *r, SDL_Window *window) {
    *r = (Renderer){0};
    r->window = window;
    create_instance_and_device(r);
    r->allocator = gpu_allocator_create(r->physical_device, r->device);
    r->upload = upload_context_create(r->device, r->allocator, r->graphics_queue,
                                      r->graphics_family, UPLOAD_STAGING_CAPACITY);
    create_shadow_texture(r);
    create_atmosphere_textures(r);
    create_descriptors(r);
    create_command_and_sync(r);
    create_pipeline_layout(r);
    create_atmosphere_pipelines(r);
    create_shadow_render_pass(r, r->shadow_map.format);
    create_shadow_framebuffers(r);
    create_shadow_pipeline(r);
    create_swapchain(r);
}

void renderer_wait_idle(Renderer *r) {
    vkDeviceWaitIdle(r->device);
}

float renderer_aspect(const Renderer *r) {
    return (float)r->swapchain_extent.width / (float)r->swapchain_extent.height;
}

void renderer_shutdown(Renderer *r) {
    vkDeviceWaitIdle(r->device);
    destroy_swapchain(r);
    vkDestroyPipeline(r->device, r->atmosphere_aerial_pipeline, NULL);
    vkDestroyPipeline(r->device, r->atmosphere_skyview_pipeline, NULL);
    vkDestroyPipeline(r->device, r->atmosphere_multiscattering_pipeline, NULL);
    vkDestroyPipeline(r->device, r->atmosphere_transmittance_pipeline, NULL);
    vkDestroyPipeline(r->device, r->shadow_pipeline, NULL);
    for (uint32_t i = 0; i < SHADOW_CASCADE_COUNT; ++i)
        vkDestroyFramebuffer(r->device, r->shadow_framebuffers[i], NULL);
    vkDestroyRenderPass(r->device, r->shadow_render_pass, NULL);
    vkDestroyPipelineLayout(r->device, r->atmosphere_pipeline_layout, NULL);
    vkDestroyPipelineLayout(r->device, r->display_pipeline_layout, NULL);
    vkDestroyPipelineLayout(r->device, r->pipeline_layout, NULL);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vkDestroySemaphore(r->device, r->render_finished[i], NULL);
        vkDestroySemaphore(r->device, r->image_available[i], NULL);
        vkDestroyFence(r->device, r->in_flight[i], NULL);
        gpu_buffer_destroy(r->device, r->allocator, &r->frame_ubo[i]);
    }
    vkDestroyCommandPool(r->device, r->command_pool, NULL);
    vkDestroyDescriptorPool(r->device, r->descriptor_pool, NULL);
    vkDestroyDescriptorSetLayout(r->device, r->atmosphere_set_layout, NULL);
    vkDestroyDescriptorSetLayout(r->device, r->display_set_layout, NULL);
    vkDestroyDescriptorSetLayout(r->device, r->material_set_layout, NULL);
    vkDestroyDescriptorSetLayout(r->device, r->frame_set_layout, NULL);
    texture_destroy(r->device, r->allocator, &r->fallback_texture);
    texture_destroy(r->device, r->allocator, &r->terrain_detail_texture);
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
