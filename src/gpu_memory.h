#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

/* Block-based device-memory suballocator.

   Tile streaming will create and destroy many small buffers and images per
   second. One VkDeviceMemory per resource would exhaust the platform allocation
   limit and thrash the driver, so every resource instead carves a range out of
   a large shared block. Blocks are grouped by concrete memory-type index and by
   tiling class: "linear" blocks hold buffers (and any host-visible staging),
   "non-linear" blocks hold optimally-tiled images. Keeping the two classes in
   separate VkDeviceMemory objects sidesteps bufferImageGranularity entirely, so
   a plain per-range alignment is always sufficient.

   Host-visible blocks are persistently mapped once at creation; an allocation
   out of such a block exposes a `mapped` pointer straight to its range. */

typedef struct GpuAllocator GpuAllocator;

/* A suballocated range. Treat every field as read-only; free it with the exact
   struct returned by gpu_alloc. A zeroed GpuAllocation is a valid "nothing
   allocated" value and is safe to pass to gpu_free. */
typedef struct
{
	VkDeviceMemory memory; /* the owning block's VkDeviceMemory              */
	VkDeviceSize offset;   /* byte offset of this range within `memory`      */
	VkDeviceSize size;	   /* usable size (>= requested; padding folded in)  */
	void *mapped;		   /* CPU pointer to offset, or NULL if not mappable */
	uint32_t block;		   /* owning block index, for gpu_free bookkeeping   */
	bool linear;		   /* tiling class this range was carved from        */
} GpuAllocation;

GpuAllocator *gpu_allocator_create(VkPhysicalDevice physical_device, VkDevice device);
void gpu_allocator_destroy(GpuAllocator *allocator);

/* Suballocate memory satisfying `requirements` (size/alignment/type mask) with
   the given property flags. `linear` selects the tiling class: true for buffers
   and host-visible staging, false for optimally-tiled images. Aborts on OOM. */
GpuAllocation gpu_alloc(GpuAllocator *allocator, VkMemoryRequirements requirements,
						VkMemoryPropertyFlags properties, bool linear);
void gpu_free(GpuAllocator *allocator, GpuAllocation allocation);

/* Live counters for debug overlays / leak checks. */
typedef struct
{
	uint32_t block_count;  /* VkDeviceMemory objects currently held      */
	VkDeviceSize reserved; /* total bytes across all blocks              */
	VkDeviceSize used;	   /* bytes currently handed out via gpu_alloc   */
	uint32_t live_allocs;  /* outstanding allocations not yet freed      */
} GpuMemoryStats;

GpuMemoryStats gpu_allocator_stats(const GpuAllocator *allocator);
