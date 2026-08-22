#pragma once

/* fprintf (stdio.h) and exit/EXIT_FAILURE (stdlib.h) are used only inside the
   VK_CHECK macro below. Include-analysis tools (clangd/IWYU) expand macros at
   the call site, not here, so they can't see the usage and flag these headers
   as unused. The pragmas mark them as deliberately kept. */
#include <stdio.h>  /* IWYU pragma: keep */
#include <stdlib.h> /* IWYU pragma: keep */
#include <vulkan/vulkan.h>

#define VK_CHECK(call) do { \
    VkResult vk_check_result_ = (call); \
    if (vk_check_result_ != VK_SUCCESS) { \
        fprintf(stderr, "%s failed with VkResult %d at %s:%d\n", #call, vk_check_result_, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)
