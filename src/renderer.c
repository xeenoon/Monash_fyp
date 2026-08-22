#include "renderer.h"
#include "vk_common.h"

#include <SDL3/SDL_vulkan.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

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
_Static_assert(offsetof(FrameUniforms, depth_debug) == 600, "FrameUniforms depth debug offset");
_Static_assert(sizeof(FrameUniforms) == 608, "FrameUniforms std140 size");

/* Staging capacity for the upload ring: large enough for the 2048x2048 albedo
   (16 MiB) plus the terrain mesh in a single batch. */
#define UPLOAD_STAGING_CAPACITY (32u * 1024u * 1024u)
#define MAX_TEXTURE_SETS 16

static uint8_t *read_file(const char *path, size_t *size) {
    FILE *file = fopen(path, "rb");
    if (!file) { fprintf(stderr, "Could not open %s\n", path); exit(EXIT_FAILURE); }
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    rewind(file);
    uint8_t *data = malloc((size_t)length);
    if (!data || fread(data, 1, (size_t)length, file) != (size_t)length) {
        fprintf(stderr, "Could not read %s\n", path); exit(EXIT_FAILURE);
    }
    fclose(file);
    *size = (size_t)length;
    return data;
}

VkDescriptorSet renderer_allocate_material_set(Renderer *r, VkImageView view, VkSampler sampler) {
    VkDescriptorSetAllocateInfo alloc = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = r->descriptor_pool, .descriptorSetCount = 1,
        .pSetLayouts = &r->material_set_layout };
    VkDescriptorSet set;
    VK_CHECK(vkAllocateDescriptorSets(r->device, &alloc, &set));
    VkDescriptorImageInfo image = { .sampler = sampler, .imageView = view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet write = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set, .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &image };
    vkUpdateDescriptorSets(r->device, 1, &write, 0, NULL);
    return set;
}

/* Descriptor roles: set 0 = per-frame UBO (vertex+fragment), set 1 = per-material
   combined image sampler (fragment). A shared pool serves the frame sets, all
   mesh material sets, and the untextured fallback set. */
static void create_descriptors(Renderer *r) {
    VkDescriptorSetLayoutBinding frame_binding = { .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT };
    VkDescriptorSetLayoutCreateInfo frame_layout = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1, .pBindings = &frame_binding };
    VK_CHECK(vkCreateDescriptorSetLayout(r->device, &frame_layout, NULL, &r->frame_set_layout));

    VkDescriptorSetLayoutBinding material_binding = { .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
    VkDescriptorSetLayoutCreateInfo material_layout = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1, .pBindings = &material_binding };
    VK_CHECK(vkCreateDescriptorSetLayout(r->device, &material_layout, NULL, &r->material_set_layout));

    VkDescriptorPoolSize pool_sizes[2] = {
        { .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = MAX_FRAMES_IN_FLIGHT },
        { .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = MAX_TEXTURE_SETS } };
    VkDescriptorPoolCreateInfo pool = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .poolSizeCount = 2, .pPoolSizes = pool_sizes, .maxSets = MAX_FRAMES_IN_FLIGHT + MAX_TEXTURE_SETS };
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
        VkWriteDescriptorSet write = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = r->frame_set[i], .dstBinding = 0, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &info };
        vkUpdateDescriptorSets(r->device, 1, &write, 0, NULL);
    }

    texture_create_white(r->device, r->allocator, r->upload, &r->fallback_texture);
    r->fallback_material_set = renderer_allocate_material_set(r, r->fallback_texture.view,
                                                              r->fallback_texture.sampler);
}

static VkFormat find_depth_format(VkPhysicalDevice physical_device) {
    const VkFormat candidates[] = {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
                                   VK_FORMAT_D24_UNORM_S8_UINT};
    for (size_t i = 0; i < sizeof(candidates)/sizeof(candidates[0]); ++i) {
        VkFormatProperties properties;
        vkGetPhysicalDeviceFormatProperties(physical_device, candidates[i], &properties);
        if (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
            return candidates[i];
    }
    fprintf(stderr, "No supported depth format found\n");
    exit(EXIT_FAILURE);
}

static VkShaderModule create_shader_module(Renderer *r, const char *path) {
    size_t size;
    uint8_t *code = read_file(path, &size);
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
    VkPhysicalDeviceFeatures features = { .samplerAnisotropy = supported.samplerAnisotropy };
    VkDeviceCreateInfo device_info = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = queue_count, .pQueueCreateInfos = queues,
        .enabledExtensionCount = 1, .ppEnabledExtensionNames = device_extensions,
        .pEnabledFeatures = &features };
    VK_CHECK(vkCreateDevice(r->physical_device, &device_info, NULL, &r->device));
    vkGetDeviceQueue(r->device, r->graphics_family, 0, &r->graphics_queue);
    vkGetDeviceQueue(r->device, r->present_family, 0, &r->present_queue);
}

static void create_render_pass(Renderer *r, VkFormat depth_format) {
    VkAttachmentDescription attachments[2] = {
        { .format = r->swapchain_format, .samples = VK_SAMPLE_COUNT_1_BIT,
          .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
          .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
          .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR },
        { .format = depth_format, .samples = VK_SAMPLE_COUNT_1_BIT,
          .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
          .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
          .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL }
    };
    VkAttachmentReference color_ref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depth_ref = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass = { .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1, .pColorAttachments = &color_ref, .pDepthStencilAttachment = &depth_ref };
    VkSubpassDependency dependency = { .srcSubpass = VK_SUBPASS_EXTERNAL, .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT };
    VkRenderPassCreateInfo info = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 2, .pAttachments = attachments, .subpassCount = 1,
        .pSubpasses = &subpass, .dependencyCount = 1, .pDependencies = &dependency };
    VK_CHECK(vkCreateRenderPass(r->device, &info, NULL, &r->render_pass));
}

/* Pipeline layout is stable: set 0 frame data, set 1 material data, no push
   constants (per-frame data lives in the set-0 UBO). Built once at init. */
static void create_pipeline_layout(Renderer *r) {
    VkDescriptorSetLayout layouts[2] = {r->frame_set_layout, r->material_set_layout};
    VkPipelineLayoutCreateInfo layout_info = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2, .pSetLayouts = layouts };
    VK_CHECK(vkCreatePipelineLayout(r->device, &layout_info, NULL, &r->pipeline_layout));
}

static void create_pipeline(Renderer *r) {
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
        .layout = r->pipeline_layout, .renderPass = r->render_pass, .subpass = 0 };
    VK_CHECK(vkCreateGraphicsPipelines(r->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &r->pipeline));
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
    r->framebuffers = malloc(sizeof(*r->framebuffers) * r->image_count);
    vkGetSwapchainImagesKHR(r->device, r->swapchain, &r->image_count, r->images);
    for (uint32_t i = 0; i < r->image_count; ++i) {
        VkImageViewCreateInfo view = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = r->images[i], .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = r->swapchain_format,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1} };
        VK_CHECK(vkCreateImageView(r->device, &view, NULL, &r->image_views[i]));
    }
    VkFormat depth_format = find_depth_format(r->physical_device);
    create_render_pass(r, depth_format);
    create_pipeline(r);
    r->depth = texture_create_depth_target(r->device, r->allocator, depth_format,
                                           r->swapchain_extent.width, r->swapchain_extent.height);
    for (uint32_t i = 0; i < r->image_count; ++i) {
        VkImageView attachments[] = {r->image_views[i], r->depth.view};
        VkFramebufferCreateInfo fb = { .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = r->render_pass, .attachmentCount = 2, .pAttachments = attachments,
            .width = r->swapchain_extent.width, .height = r->swapchain_extent.height, .layers = 1 };
        VK_CHECK(vkCreateFramebuffer(r->device, &fb, NULL, &r->framebuffers[i]));
    }
}

static void destroy_swapchain(Renderer *r) {
    for (uint32_t i = 0; i < r->image_count; ++i) vkDestroyFramebuffer(r->device, r->framebuffers[i], NULL);
    texture_destroy(r->device, r->allocator, &r->depth);
    vkDestroyPipeline(r->device, r->pipeline, NULL);
    vkDestroyRenderPass(r->device, r->render_pass, NULL);
    for (uint32_t i = 0; i < r->image_count; ++i) vkDestroyImageView(r->device, r->image_views[i], NULL);
    vkDestroySwapchainKHR(r->device, r->swapchain, NULL);
    free(r->framebuffers); free(r->image_views); free(r->images);
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
    vkDestroyPipeline(r->device, r->pipeline, NULL);
    create_pipeline(r);
    printf("Shaders reloaded\n");
}

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

static void record_commands(Renderer *r, uint32_t image_index, const Mesh *mesh) {
    VkCommandBuffer command = r->command_buffers[r->frame];
    VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    VK_CHECK(vkBeginCommandBuffer(command, &begin));
    VkClearValue clear[2] = {
        {.color = {{0.055f, 0.065f, 0.08f, 1.0f}}},
        {.depthStencil = {0.0f, 0}}
    };
    VkRenderPassBeginInfo render = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = r->render_pass, .framebuffer = r->framebuffers[image_index],
        .renderArea = {{0,0}, r->swapchain_extent}, .clearValueCount = 2, .pClearValues = clear };
    vkCmdBeginRenderPass(command, &render, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, r->pipeline);
    VkViewport viewport = {0, 0, (float)r->swapchain_extent.width, (float)r->swapchain_extent.height, 0, 1};
    VkRect2D scissor = {{0,0}, r->swapchain_extent};
    vkCmdSetViewport(command, 0, 1, &viewport);
    vkCmdSetScissor(command, 0, 1, &scissor);
    /* Set 0: per-frame data. Set 1: the mesh's material, or the fallback. */
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, r->pipeline_layout, 0, 1,
        &r->frame_set[r->frame], 0, NULL);
    VkDescriptorSet material_set = mesh->material_set ? mesh->material_set : r->fallback_material_set;
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, r->pipeline_layout, 1, 1,
        &material_set, 0, NULL);
    mesh_draw(command, mesh);
    vkCmdEndRenderPass(command);
    VK_CHECK(vkEndCommandBuffer(command));
}

void renderer_draw_frame(Renderer *r, const FrameUniforms *frame, const Mesh *mesh, bool resized) {
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
    record_commands(r, image_index, mesh);
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
    create_descriptors(r);
    create_command_and_sync(r);
    create_pipeline_layout(r);
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
    vkDestroyPipelineLayout(r->device, r->pipeline_layout, NULL);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vkDestroySemaphore(r->device, r->render_finished[i], NULL);
        vkDestroySemaphore(r->device, r->image_available[i], NULL);
        vkDestroyFence(r->device, r->in_flight[i], NULL);
        gpu_buffer_destroy(r->device, r->allocator, &r->frame_ubo[i]);
    }
    vkDestroyCommandPool(r->device, r->command_pool, NULL);
    vkDestroyDescriptorPool(r->device, r->descriptor_pool, NULL);
    vkDestroyDescriptorSetLayout(r->device, r->material_set_layout, NULL);
    vkDestroyDescriptorSetLayout(r->device, r->frame_set_layout, NULL);
    texture_destroy(r->device, r->allocator, &r->fallback_texture);
    upload_context_destroy(r->upload);
    gpu_allocator_destroy(r->allocator);
    vkDestroyDevice(r->device, NULL);
    vkDestroySurfaceKHR(r->instance, r->surface, NULL);
    vkDestroyInstance(r->instance, NULL);
}
