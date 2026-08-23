#include "gpu_memory.h"
#include "vk_common.h"

#include <stdlib.h>
#include <string.h>

/* Default block size. Blocks are created lazily and grown to at least the
   requested allocation, so an unusually large resource still gets a dedicated
   block instead of failing. 64 MiB keeps the block count small for the current
   single-terrain workload while remaining friendly to modest memory budgets. */
#define GPU_BLOCK_SIZE (64u * 1024u * 1024u)

typedef struct
{
	VkDeviceSize offset;
	VkDeviceSize size;
} FreeRange;

typedef struct
{
	VkDeviceMemory memory;
	VkDeviceSize size;
	uint32_t memory_type;
	bool linear;
	void *mapped;	   /* whole-block map, or NULL if not host-visible */
	FreeRange *ranges; /* free list, kept sorted by offset             */
	uint32_t range_count;
	uint32_t range_cap;
} GpuBlock;

struct GpuAllocator
{
	VkPhysicalDevice physical_device;
	VkDevice device;
	VkPhysicalDeviceMemoryProperties memory_properties;

	GpuBlock *blocks;
	uint32_t block_count;
	uint32_t block_cap;

	VkDeviceSize used;
	uint32_t live_allocs;
};

static VkDeviceSize align_up(VkDeviceSize value, VkDeviceSize alignment)
{
	if (alignment == 0)
		return value;
	return (value + alignment - 1) & ~(alignment - 1);
}

static uint32_t find_memory_type(GpuAllocator *a, uint32_t type_bits,
								 VkMemoryPropertyFlags properties)
{
	for (uint32_t i = 0; i < a->memory_properties.memoryTypeCount; ++i)
		if ((type_bits & (1u << i)) &&
			(a->memory_properties.memoryTypes[i].propertyFlags & properties) == properties)
			return i;
	fprintf(stderr, "gpu_memory: no memory type for bits 0x%x properties 0x%x\n", type_bits,
			properties);
	exit(EXIT_FAILURE);
}

/* --- free-list maintenance ------------------------------------------------ */

static void block_insert_free(GpuBlock *block, VkDeviceSize offset, VkDeviceSize size)
{
	if (size == 0)
		return;
	/* Find the sorted insertion point. */
	uint32_t i = 0;
	while (i < block->range_count && block->ranges[i].offset < offset)
		++i;

	/* Coalesce with the previous range if adjacent. */
	if (i > 0 && block->ranges[i - 1].offset + block->ranges[i - 1].size == offset)
	{
		block->ranges[i - 1].size += size;
		/* The merged previous range may now touch the next range. */
		if (i < block->range_count &&
			block->ranges[i - 1].offset + block->ranges[i - 1].size == block->ranges[i].offset)
		{
			block->ranges[i - 1].size += block->ranges[i].size;
			memmove(&block->ranges[i], &block->ranges[i + 1],
					sizeof(FreeRange) * (block->range_count - i - 1));
			--block->range_count;
		}
		return;
	}
	/* Coalesce with the next range if adjacent. */
	if (i < block->range_count && offset + size == block->ranges[i].offset)
	{
		block->ranges[i].offset = offset;
		block->ranges[i].size += size;
		return;
	}
	/* No coalescing: insert a fresh range, growing the array if needed. */
	if (block->range_count == block->range_cap)
	{
		block->range_cap = block->range_cap ? block->range_cap * 2 : 8;
		block->ranges = realloc(block->ranges, sizeof(FreeRange) * block->range_cap);
	}
	memmove(&block->ranges[i + 1], &block->ranges[i], sizeof(FreeRange) * (block->range_count - i));
	block->ranges[i].offset = offset;
	block->ranges[i].size = size;
	++block->range_count;
}

/* Carve [aligned, aligned+size) out of free range `index`, returning the
   aligned offset. The gap before the aligned offset and the tail after the
   allocation stay on the free list. */
static VkDeviceSize block_carve(GpuBlock *block, uint32_t index, VkDeviceSize size,
								VkDeviceSize alignment)
{
	FreeRange range = block->ranges[index];
	VkDeviceSize aligned = align_up(range.offset, alignment);
	VkDeviceSize head_gap = aligned - range.offset;
	VkDeviceSize tail = (range.offset + range.size) - (aligned + size);

	/* Remove the range, then re-insert whatever is left over. */
	memmove(&block->ranges[index], &block->ranges[index + 1],
			sizeof(FreeRange) * (block->range_count - index - 1));
	--block->range_count;
	if (head_gap)
		block_insert_free(block, range.offset, head_gap);
	if (tail)
		block_insert_free(block, aligned + size, tail);
	return aligned;
}

static bool block_fits(const GpuBlock *block, uint32_t index, VkDeviceSize size,
					   VkDeviceSize alignment)
{
	FreeRange range = block->ranges[index];
	VkDeviceSize aligned = align_up(range.offset, alignment);
	return aligned + size <= range.offset + range.size;
}

/* --- block creation ------------------------------------------------------- */

static GpuBlock *create_block(GpuAllocator *a, uint32_t memory_type, bool linear, VkDeviceSize size)
{
	if (a->block_count == a->block_cap)
	{
		a->block_cap = a->block_cap ? a->block_cap * 2 : 8;
		a->blocks = realloc(a->blocks, sizeof(GpuBlock) * a->block_cap);
	}
	GpuBlock *block = &a->blocks[a->block_count];
	*block = (GpuBlock){.memory_type = memory_type, .linear = linear, .size = size};

	VkMemoryAllocateInfo info = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
								 .allocationSize = size,
								 .memoryTypeIndex = memory_type};
	VK_CHECK(vkAllocateMemory(a->device, &info, NULL, &block->memory));

	if (a->memory_properties.memoryTypes[memory_type].propertyFlags &
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
		VK_CHECK(vkMapMemory(a->device, block->memory, 0, VK_WHOLE_SIZE, 0, &block->mapped));

	block_insert_free(block, 0, size);
	++a->block_count;
	return block;
}

GpuAllocator *gpu_allocator_create(VkPhysicalDevice physical_device, VkDevice device)
{
	GpuAllocator *a = calloc(1, sizeof(*a));
	a->physical_device = physical_device;
	a->device = device;
	vkGetPhysicalDeviceMemoryProperties(physical_device, &a->memory_properties);
	return a;
}

void gpu_allocator_destroy(GpuAllocator *a)
{
	if (!a)
		return;
	if (a->live_allocs)
		fprintf(stderr, "gpu_memory: %u allocations leaked at shutdown\n", a->live_allocs);
	for (uint32_t i = 0; i < a->block_count; ++i)
	{
		if (a->blocks[i].mapped)
			vkUnmapMemory(a->device, a->blocks[i].memory);
		vkFreeMemory(a->device, a->blocks[i].memory, NULL);
		free(a->blocks[i].ranges);
	}
	free(a->blocks);
	free(a);
}

GpuAllocation gpu_alloc(GpuAllocator *a, VkMemoryRequirements requirements,
						VkMemoryPropertyFlags properties, bool linear)
{
	uint32_t memory_type = find_memory_type(a, requirements.memoryTypeBits, properties);
	VkDeviceSize size = requirements.size;
	VkDeviceSize alignment = requirements.alignment;

	/* First fit across existing blocks of the matching type and tiling class. */
	for (uint32_t b = 0; b < a->block_count; ++b)
	{
		GpuBlock *block = &a->blocks[b];
		if (block->memory_type != memory_type || block->linear != linear)
			continue;
		for (uint32_t r = 0; r < block->range_count; ++r)
		{
			if (!block_fits(block, r, size, alignment))
				continue;
			VkDeviceSize offset = block_carve(block, r, size, alignment);
			a->used += size;
			++a->live_allocs;
			return (GpuAllocation){.memory = block->memory,
								   .offset = offset,
								   .size = size,
								   .mapped = block->mapped ? (char *)block->mapped + offset : NULL,
								   .block = b,
								   .linear = linear};
		}
	}

	/* Nothing fit: create a new block large enough to hold this allocation. */
	VkDeviceSize block_size = size + alignment > GPU_BLOCK_SIZE ? size + alignment : GPU_BLOCK_SIZE;
	uint32_t new_index = a->block_count;
	GpuBlock *block = create_block(a, memory_type, linear, block_size);
	VkDeviceSize offset = block_carve(block, 0, size, alignment);
	a->used += size;
	++a->live_allocs;
	return (GpuAllocation){.memory = block->memory,
						   .offset = offset,
						   .size = size,
						   .mapped = block->mapped ? (char *)block->mapped + offset : NULL,
						   .block = new_index,
						   .linear = linear};
}

void gpu_free(GpuAllocator *a, GpuAllocation allocation)
{
	if (allocation.memory == VK_NULL_HANDLE || allocation.size == 0)
		return;
	GpuBlock *block = &a->blocks[allocation.block];
	block_insert_free(block, allocation.offset, allocation.size);
	a->used -= allocation.size;
	--a->live_allocs;
}

GpuMemoryStats gpu_allocator_stats(const GpuAllocator *a)
{
	GpuMemoryStats stats = {
		.block_count = a->block_count, .used = a->used, .live_allocs = a->live_allocs};
	for (uint32_t i = 0; i < a->block_count; ++i)
		stats.reserved += a->blocks[i].size;
	return stats;
}
