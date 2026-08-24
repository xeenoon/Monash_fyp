#include "benchmark_ground.h"
#include "renderer.h"
#include <float.h>

bool benchmark_ground_create(struct Renderer *r, BenchmarkGround *g, const Mesh *quarry)
{
	if (!r || !g || !quarry || !quarry->vertices || !quarry->vertex_count) return false;
	*g = (BenchmarkGround){0}; float xmin=FLT_MAX,xmax=-FLT_MAX,zmin=FLT_MAX,zmax=-FLT_MAX,ymin=FLT_MAX;
	for(uint32_t i=0;i<quarry->vertex_count;++i) { const Vertex *v=&quarry->vertices[i];
		xmin=fminf(xmin,v->position[0]); xmax=fmaxf(xmax,v->position[0]); zmin=fminf(zmin,v->position[2]); zmax=fmaxf(zmax,v->position[2]); ymin=fminf(ymin,v->position[1]); }
	float pad=fmaxf(10.f, .25f*fmaxf(xmax-xmin,zmax-zmin)), y=ymin-.02f;
	float x0=xmin-pad,x1=xmax+pad,z0=zmin-pad,z1=zmax+pad;
	g->vertices[0]=(Vertex){.position={x0,y,z0},.normal={0,1,0},.texcoord={0,0},.tangent={1,0,0,1}};
	g->vertices[1]=(Vertex){.position={x1,y,z0},.normal={0,1,0},.texcoord={1,0},.tangent={1,0,0,1}};
	g->vertices[2]=(Vertex){.position={x1,y,z1},.normal={0,1,0},.texcoord={1,1},.tangent={1,0,0,1}};
	g->vertices[3]=(Vertex){.position={x0,y,z1},.normal={0,1,0},.texcoord={0,1},.tangent={1,0,0,1}};
	uint32_t ix[6]={0,1,2,0,2,3}; for(uint32_t i=0;i<6;++i)g->indices[i]=ix[i];
	g->mesh=(Mesh){.local_to_world=coordinate_identity_transform((WorldPosition){0}),.vertices=g->vertices,.vertex_count=4,.indices=g->indices,.index_count=6};
	mesh_upload(r,&g->mesh); const uint8_t albedo[4]={128,128,128,255}, orm[4]={255,230,0,255}, normal[4]={128,128,255,255};
	texture_create_solid_rgba8(r->device,r->allocator,r->upload,&g->mesh.texture,albedo,true);
	texture_create_solid_rgba8(r->device,r->allocator,r->upload,&g->mesh.orm,orm,false);
	texture_create_solid_rgba8(r->device,r->allocator,r->upload,&g->mesh.normal_map,normal,false);
	g->mesh.material_set=renderer_allocate_pbr_set(r,&g->mesh.texture,&g->mesh.orm,&g->mesh.normal_map); return true;
}
void benchmark_ground_destroy(struct Renderer *r, BenchmarkGround *g) { if(!g)return; if(g->mesh.material_set)renderer_free_material_set(r,g->mesh.material_set); gpu_buffer_destroy(r->device,r->allocator,&g->mesh.vertex_buffer); gpu_buffer_destroy(r->device,r->allocator,&g->mesh.index_buffer); texture_destroy(r->device,r->allocator,&g->mesh.texture); texture_destroy(r->device,r->allocator,&g->mesh.orm); texture_destroy(r->device,r->allocator,&g->mesh.normal_map); *g=(BenchmarkGround){0}; }
