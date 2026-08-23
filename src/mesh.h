#pragma once

#include <stdint.h>
#include <vulkan/vulkan.h>

#include "coordinate.h"
#include "gpu_buffer.h"
#include "texture.h"

typedef struct
{
	float position[3];
	float normal[3];
	float texcoord[2];
	/* Retained for the terrain tile builder's skirt markers. */
	float untextured;
	float tangent[4]; /* xyz tangent, w handedness for tangent-space normals */
} Vertex;

/* Base "class". Every primitive embeds this as its FIRST member, so a
   Cube* casts cleanly to Mesh* — the well-defined version of the
   sockaddr / sockaddr_in trick (C11 6.7.2.1p15). */
typedef struct
{
	/* Geometry is always tile-local float data. This transform is retained in
	   double precision and made camera-relative immediately before drawing. */
	LocalToWorldTransform local_to_world;

	const Vertex *vertices; /* CPU geometry (may point at static data) */
	uint32_t vertex_count;
	GpuBuffer vertex_buffer; /* device-local GPU copy, via mesh_upload  */

	/* Optional index buffer. Leave indices=NULL / index_count=0 for a plain
	   non-indexed draw; set both to have mesh_upload build a GPU index buffer
	   and mesh_draw switch to vkCmdDrawIndexed. */
	const uint32_t *indices; /* CPU indices (may point at static data)  */
	uint32_t index_count;
	GpuBuffer index_buffer;

	/* Optional albedo texture. A mesh implementation sets texture_path to its
	   own image file; mesh_upload then loads it and allocates material_set.
	   Leave texture_path NULL to draw untextured (renderer binds a fallback). */
	const char *texture_path;
	Texture texture;
	const char *orm_path;
	Texture orm;
	const char *normal_path;
	Texture normal_map;
	VkDescriptorSet material_set; /* set 1: combined image sampler           */
} Mesh;

struct Renderer;

/* Uploads mesh->vertices to a GPU buffer, filling buffer/memory. */
void mesh_upload(struct Renderer *r, Mesh *mesh);
void mesh_destroy(struct Renderer *r, Mesh *mesh);
void mesh_draw(VkCommandBuffer cmd, const Mesh *mesh);

/* Vertex layout, consumed by the pipeline in renderer.c */
VkVertexInputBindingDescription mesh_binding_description(void);
const VkVertexInputAttributeDescription *mesh_attribute_descriptions(uint32_t *count);
