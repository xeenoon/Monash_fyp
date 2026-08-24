#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

#include "gltf_scene.h"
#include "renderer.h"
#include "mikktspace.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { GltfScene *scene; cgltf_data *data; GltfLoadError *error; GltfLoadResult result; } Import;
typedef struct { Vertex *vertices; uint32_t count; } MikkGeometry;
static int mikk_faces(const SMikkTSpaceContext *c) { return (int)(((MikkGeometry *)c->m_pUserData)->count / 3u); }
static int mikk_verts(const SMikkTSpaceContext *c, const int face) { (void)c; (void)face; return 3; }
static Vertex *mikk_vertex(const SMikkTSpaceContext *c, int face, int vertex) { MikkGeometry *g=c->m_pUserData; return &g->vertices[(uint32_t)face*3u+(uint32_t)vertex]; }
static void mikk_position(const SMikkTSpaceContext *c,float x[],int f,int v){memcpy(x,mikk_vertex(c,f,v)->position,3*sizeof(float));}
static void mikk_normal(const SMikkTSpaceContext *c,float x[],int f,int v){memcpy(x,mikk_vertex(c,f,v)->normal,3*sizeof(float));}
static void mikk_uv(const SMikkTSpaceContext *c,float x[],int f,int v){memcpy(x,mikk_vertex(c,f,v)->texcoord,2*sizeof(float));}
static void mikk_set(const SMikkTSpaceContext *c,const float x[],float sign,int f,int v){Vertex *p=mikk_vertex(c,f,v);memcpy(p->tangent,x,3*sizeof(float));p->tangent[3]=sign<0?-1.f:1.f;}
static bool generate_mikk(Vertex *vertices,uint32_t count) { MikkGeometry g={vertices,count}; SMikkTSpaceInterface i={mikk_faces,mikk_verts,mikk_position,mikk_normal,mikk_uv,mikk_set,NULL}; SMikkTSpaceContext c={&i,&g}; return genTangSpaceDefault(&c) != 0; }

static void fail(Import *in, GltfLoadResult result, uint32_t node, uint32_t primitive,
                 const char *format, ...)
{
    if (in->result != GLTF_LOAD_OK) return;
    in->result = result;
    if (in->error) {
        va_list args; va_start(args, format);
        vsnprintf(in->error->message, sizeof(in->error->message), format, args);
        va_end(args);
        in->error->node_index = node; in->error->primitive_index = primitive;
    }
}
static bool finite_value(float x) { return isfinite(x); }
static uint32_t node_number(const cgltf_data *d, const cgltf_node *n) { return (uint32_t)(n - d->nodes); }
static uint32_t material_number(const cgltf_data *d, const cgltf_material *m) { return m ? (uint32_t)(m - d->materials) + 1u : 0; }
static char *image_path(const char *source, const cgltf_texture_view *view)
{
    if (!view || !view->texture || !view->texture->image || !view->texture->image->uri) return NULL;
    const char *uri = view->texture->image->uri;
    if (strstr(uri, "://") || !strncmp(uri, "data:", 5)) return NULL;
    const char *slash = strrchr(source, '/'); size_t base = slash ? (size_t)(slash - source + 1) : 0;
    char *result = malloc(base + strlen(uri) + 1u); if (!result) return NULL;
    memcpy(result, source, base); strcpy(result + base, uri); return result;
}

static void fallback_tangent(Vertex *v)
{
    float ax = fabsf(v->normal[1]) < .999f ? 0.f : 1.f;
    float ay = fabsf(v->normal[1]) < .999f ? 1.f : 0.f;
    v->tangent[0] = ay*v->normal[2]; v->tangent[1] = 0.f;
    v->tangent[2] = -ay*v->normal[0] + ax*v->normal[1];
    float l = sqrtf(v->tangent[0]*v->tangent[0]+v->tangent[1]*v->tangent[1]+v->tangent[2]*v->tangent[2]);
    if (l > 0.f) { v->tangent[0]/=l; v->tangent[1]/=l; v->tangent[2]/=l; }
    v->tangent[3]=1.f;
}
static void normalize3(float *v) { float l=sqrtf(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); if(l>1e-12f){v[0]/=l;v[1]/=l;v[2]/=l;} else {v[0]=0;v[1]=1;v[2]=0;} }
static void transform_position(const float m[16], const float p[3], float out[3]) {
    out[0]=m[0]*p[0]+m[4]*p[1]+m[8]*p[2]+m[12]; out[1]=m[1]*p[0]+m[5]*p[1]+m[9]*p[2]+m[13]; out[2]=m[2]*p[0]+m[6]*p[1]+m[10]*p[2]+m[14];
}
static float determinant(const float m[16]) { return m[0]*(m[5]*m[10]-m[9]*m[6])-m[4]*(m[1]*m[10]-m[9]*m[2])+m[8]*(m[1]*m[6]-m[5]*m[2]); }
static void transform_direction(const float m[16], float v[3]) { float o[3]={m[0]*v[0]+m[4]*v[1]+m[8]*v[2],m[1]*v[0]+m[5]*v[1]+m[9]*v[2],m[2]*v[0]+m[6]*v[1]+m[10]*v[2]}; memcpy(v,o,sizeof(o)); normalize3(v); }
static bool transform_normal(const float m[16], float v[3])
{
    float d = determinant(m);
    if (!isfinite(d) || fabsf(d) < 1e-10f) return false;
    /* inverse-transpose(L), written for glTF's column-major matrices */
    float o[3] = {(m[5]*m[10]-m[9]*m[6])*v[0] + (m[9]*m[2]-m[1]*m[10])*v[1] + (m[1]*m[6]-m[5]*m[2])*v[2],
                  (m[8]*m[6]-m[4]*m[10])*v[0] + (m[0]*m[10]-m[8]*m[2])*v[1] + (m[4]*m[2]-m[0]*m[6])*v[2],
                  (m[4]*m[9]-m[8]*m[5])*v[0] + (m[8]*m[1]-m[0]*m[9])*v[1] + (m[0]*m[5]-m[4]*m[1])*v[2]};
    memcpy(v, o, sizeof(o)); normalize3(v); return true;
}
static void generate_normals(Vertex *v, uint32_t n, const uint32_t *idx, uint32_t count) {
    for(uint32_t i=0;i<n;i++) memset(v[i].normal,0,3*sizeof(float));
    for(uint32_t i=0;i<count;i+=3) { Vertex *a=&v[idx?idx[i]:i],*b=&v[idx?idx[i+1]:i+1],*c=&v[idx?idx[i+2]:i+2]; float u[3]={b->position[0]-a->position[0],b->position[1]-a->position[1],b->position[2]-a->position[2]}, w[3]={c->position[0]-a->position[0],c->position[1]-a->position[1],c->position[2]-a->position[2]}, q[3]={u[1]*w[2]-u[2]*w[1],u[2]*w[0]-u[0]*w[2],u[0]*w[1]-u[1]*w[0]}; for(int j=0;j<3;j++){a->normal[j]+=q[j];b->normal[j]+=q[j];c->normal[j]+=q[j];} }
    for(uint32_t i=0;i<n;i++) normalize3(v[i].normal);
}

static bool has_attribute(const cgltf_primitive *p, cgltf_attribute_type type, cgltf_accessor **out) {
    bool found = false;
    for (cgltf_size i=0;i<p->attributes_count;i++) {
        if (p->attributes[i].type == type) { if (found) return false; *out=p->attributes[i].data; found = true; }
    }
    return found;
}
static void import_primitive(Import *in, const cgltf_node *node, const cgltf_primitive *p, uint32_t pi)
{
    uint32_t ni=node_number(in->data,node); cgltf_accessor *pos=NULL,*norm=NULL,*uv=NULL,*tan=NULL;
    if (p->type != cgltf_primitive_type_triangles) { fail(in,GLTF_LOAD_UNSUPPORTED,ni,pi,"only TRIANGLES primitives are supported"); return; }
    if (!has_attribute(p,cgltf_attribute_type_position,&pos) || pos->type != cgltf_type_vec3 || !pos->count) { fail(in,GLTF_LOAD_INVALID,ni,pi,"POSITION vec3 is required"); return; }
    has_attribute(p,cgltf_attribute_type_normal,&norm); has_attribute(p,cgltf_attribute_type_texcoord,&uv); has_attribute(p,cgltf_attribute_type_tangent,&tan);
    if ((norm && (norm->type != cgltf_type_vec3 || norm->count != pos->count)) || (uv && (uv->type != cgltf_type_vec2 || uv->count != pos->count)) || (tan && (tan->type != cgltf_type_vec4 || tan->count != pos->count))) { fail(in,GLTF_LOAD_INVALID,ni,pi,"attribute counts/types do not match POSITION"); return; }
    if (p->indices && p->indices->count % 3) { fail(in,GLTF_LOAD_INVALID,ni,pi,"index count is not divisible by three"); return; }
    if (pos->count > UINT32_MAX || (p->indices && p->indices->count > UINT32_MAX)) { fail(in,GLTF_LOAD_INVALID,ni,pi,"accessor count exceeds uint32_t"); return; }
    uint32_t vc=(uint32_t)pos->count, ic=p->indices?(uint32_t)p->indices->count:vc;
    if ((!p->indices && vc%3)) { fail(in,GLTF_LOAD_INVALID,ni,pi,"incomplete non-indexed triangle data"); return; }
    GltfPrimitive *np=realloc(in->scene->primitives,(in->scene->primitive_count+1u)*sizeof(*np));
    if(!np){fail(in,GLTF_LOAD_OUT_OF_MEMORY,ni,pi,"out of memory for primitive");return;} in->scene->primitives=np;
    GltfPrimitive *o=&np[in->scene->primitive_count]; *o=(GltfPrimitive){0}; o->vertices=calloc(vc,sizeof(Vertex)); o->indices=p->indices?malloc((size_t)ic*sizeof(uint32_t)):NULL;
    if(!o->vertices || (p->indices&&!o->indices)){free(o->vertices);free(o->indices);*o=(GltfPrimitive){0};fail(in,GLTF_LOAD_OUT_OF_MEMORY,ni,pi,"out of memory for vertices");return;}
    float m[16]; cgltf_node_transform_world(node,m); float det=determinant(m); if(!isfinite(det)||fabsf(det)<1e-10f){fail(in,GLTF_LOAD_INVALID,ni,pi,"node has a degenerate transform");free(o->vertices);free(o->indices);*o=(GltfPrimitive){0};return;}
    for(uint32_t i=0;i<vc;i++){float x[4]={0};cgltf_accessor_read_float(pos,i,x,3);if(!finite_value(x[0])||!finite_value(x[1])||!finite_value(x[2])){fail(in,GLTF_LOAD_INVALID,ni,pi,"POSITION contains non-finite value");return;}transform_position(m,x,o->vertices[i].position);if(norm){cgltf_accessor_read_float(norm,i,o->vertices[i].normal,3);if(!finite_value(o->vertices[i].normal[0])||!finite_value(o->vertices[i].normal[1])||!finite_value(o->vertices[i].normal[2])||!transform_normal(m,o->vertices[i].normal)){fail(in,GLTF_LOAD_INVALID,ni,pi,"NORMAL contains invalid value");return;}}if(uv){cgltf_accessor_read_float(uv,i,o->vertices[i].texcoord,2);if(!finite_value(o->vertices[i].texcoord[0])||!finite_value(o->vertices[i].texcoord[1])){fail(in,GLTF_LOAD_INVALID,ni,pi,"TEXCOORD contains non-finite value");return;}}if(tan){cgltf_accessor_read_float(tan,i,o->vertices[i].tangent,4);if(!finite_value(o->vertices[i].tangent[0])||!finite_value(o->vertices[i].tangent[1])||!finite_value(o->vertices[i].tangent[2])||!finite_value(o->vertices[i].tangent[3])){fail(in,GLTF_LOAD_INVALID,ni,pi,"TANGENT contains non-finite value");return;}transform_direction(m,o->vertices[i].tangent);if(det<0)o->vertices[i].tangent[3]=-o->vertices[i].tangent[3];}}
    if (p->indices) for(uint32_t i=0;i<ic;i++){size_t x=cgltf_accessor_read_index(p->indices,i);if(x>=vc){fail(in,GLTF_LOAD_INVALID,ni,pi,"index outside POSITION range");return;}o->indices[i]=(uint32_t)x;}
    if(det<0) { if (p->indices) for(uint32_t i=0;i<ic;i+=3){uint32_t x=o->indices[i+1];o->indices[i+1]=o->indices[i+2];o->indices[i+2]=x;} else for(uint32_t i=0;i<vc;i+=3){Vertex x=o->vertices[i+1];o->vertices[i+1]=o->vertices[i+2];o->vertices[i+2]=x;} }
    if(!norm) generate_normals(o->vertices,vc,o->indices,ic);
    if (!tan && p->material && p->material->normal_texture.texture) {
        Vertex *corners = malloc((size_t)ic * sizeof(*corners));
        if (!corners) { fail(in, GLTF_LOAD_OUT_OF_MEMORY, ni, pi, "out of memory expanding tangent corners"); return; }
        for (uint32_t i = 0; i < ic; ++i) corners[i] = o->vertices[o->indices ? o->indices[i] : i];
        free(o->vertices); free(o->indices); o->vertices = corners; o->indices = NULL; vc = ic;
        if (!generate_mikk(o->vertices, vc)) { fail(in, GLTF_LOAD_INVALID, ni, pi, "MikkTSpace tangent generation failed"); return; }
    } else {
        for(uint32_t i=0;i<vc;i++) if(!tan) fallback_tangent(&o->vertices[i]);
    }
    o->mesh=(Mesh){.local_to_world=in->scene->placement,.vertices=o->vertices,.vertex_count=vc,.indices=o->indices,.index_count=o->indices?ic:0}; o->material_index=material_number(in->data,p->material); in->scene->primitive_count++;
}
static void visit(Import *in,const cgltf_node *n){ if(in->result!=GLTF_LOAD_OK)return; if(n->skin){fail(in,GLTF_LOAD_UNSUPPORTED,node_number(in->data,n),0,"skins are unsupported");return;} if(n->mesh)for(cgltf_size i=0;i<n->mesh->primitives_count;i++)import_primitive(in,n,&n->mesh->primitives[i],(uint32_t)i);for(cgltf_size i=0;i<n->children_count;i++)visit(in,n->children[i]); }

GltfLoadResult gltf_scene_parse(const char *path,const GltfLoadOptions *options,GltfScene *out,GltfLoadError *error)
{
    if (!out || !path) return GLTF_LOAD_INVALID;
    *out = (GltfScene){0};
    if (error) *error = (GltfLoadError){.source_path=path,.node_index=UINT32_MAX,.primitive_index=UINT32_MAX};
    char *source_copy = malloc(strlen(path) + 1u);
    if (!source_copy) return GLTF_LOAD_OUT_OF_MEMORY;
    strcpy(source_copy, path);
    out->source_path = source_copy;
    out->placement = options ? options->placement : coordinate_identity_transform((WorldPosition){0});
    cgltf_options co={0};cgltf_data *d=NULL;cgltf_result cr=cgltf_parse_file(&co,path,&d);if(cr!=cgltf_result_success){if(error)snprintf(error->message,sizeof(error->message),"could not parse glTF: %d",cr);return cr==cgltf_result_file_not_found?GLTF_LOAD_IO_ERROR:GLTF_LOAD_INVALID;} cr=cgltf_load_buffers(&co,d,path);if(cr==cgltf_result_success)cr=cgltf_validate(d);if(cr!=cgltf_result_success){if(error)snprintf(error->message,sizeof(error->message),"glTF buffer/validation failure: %d",cr);cgltf_free(d);return GLTF_LOAD_INVALID;}
    if(d->skins_count||d->animations_count){if(error)snprintf(error->message,sizeof(error->message),"skins and animations are unsupported");cgltf_free(d);return GLTF_LOAD_UNSUPPORTED;}
    out->material_count=(uint32_t)d->materials_count+1u;out->materials=calloc(out->material_count,sizeof(*out->materials));if(!out->materials){cgltf_free(d);return GLTF_LOAD_OUT_OF_MEMORY;}
    out->materials[0] = (GltfMaterial){.base_color_factor={1,1,1,1},.metallic_factor=1,.roughness_factor=1,.normal_scale=1,.occlusion_strength=1};
    for(cgltf_size i=0;i<d->materials_count;i++){cgltf_material *m=&d->materials[i];if(m->alpha_mode!=cgltf_alpha_mode_opaque){if(error)snprintf(error->message,sizeof(error->message),"material %zu is not opaque",i);cgltf_free(d);gltf_scene_destroy(NULL,out);return GLTF_LOAD_UNSUPPORTED;}GltfMaterial *x=&out->materials[i+1];memcpy(x->base_color_factor,m->pbr_metallic_roughness.base_color_factor,sizeof(x->base_color_factor));x->metallic_factor=m->pbr_metallic_roughness.metallic_factor;x->roughness_factor=m->pbr_metallic_roughness.roughness_factor;x->normal_scale=m->normal_texture.texture ? m->normal_texture.scale : 1.0f;x->occlusion_strength=m->occlusion_texture.texture ? m->occlusion_texture.scale : 1.0f;x->base_color_path=image_path(path,&m->pbr_metallic_roughness.base_color_texture);x->metallic_roughness_path=image_path(path,&m->pbr_metallic_roughness.metallic_roughness_texture);x->normal_path=image_path(path,&m->normal_texture);x->occlusion_path=image_path(path,&m->occlusion_texture);x->use_metallic_roughness_red_as_occlusion=options && options->use_metallic_roughness_red_as_occlusion && !x->occlusion_path;}
    Import in={.scene=out,.data=d,.error=error,.result=GLTF_LOAD_OK}; cgltf_scene *scene=d->scene; if(scene)for(cgltf_size i=0;i<scene->nodes_count;i++)visit(&in,scene->nodes[i]);else for(cgltf_size i=0;i<d->nodes_count;i++)if(!d->nodes[i].parent)visit(&in,&d->nodes[i]); cgltf_free(d);
    if(in.result!=GLTF_LOAD_OK){gltf_scene_destroy(NULL,out);return in.result;} fprintf(stdout,"glTF: %s: %u primitives, %u materials\n",path,out->primitive_count,out->material_count-1u);return GLTF_LOAD_OK;
}
GltfLoadResult gltf_scene_upload(struct Renderer *renderer, GltfScene *scene, GltfLoadError *error)
{
    if (!renderer || !scene) return GLTF_LOAD_INVALID;
    /* Until a material slot is populated by an image, bind typed fallbacks. */
    for (uint32_t i = 0; i < scene->material_count; ++i) {
        GltfMaterial *m = &scene->materials[i];
        if (m->base_color_path) texture_load(renderer->device, renderer->allocator, renderer->upload, &m->base_color, m->base_color_path, renderer->max_anisotropy);
        if (m->metallic_roughness_path) texture_load_linear(renderer->device, renderer->allocator, renderer->upload, &m->metallic_roughness, m->metallic_roughness_path, renderer->max_anisotropy);
        if (m->normal_path) texture_load_linear(renderer->device, renderer->allocator, renderer->upload, &m->normal, m->normal_path, renderer->max_anisotropy);
        if (m->occlusion_path) texture_load_linear(renderer->device, renderer->allocator, renderer->upload, &m->occlusion, m->occlusion_path, renderer->max_anisotropy);
        m->descriptor_set = renderer_allocate_pbr5_set(renderer,
            m->base_color.image ? &m->base_color : &renderer->fallback_texture,
            m->metallic_roughness.image ? &m->metallic_roughness : &renderer->fallback_linear_texture,
            m->normal.image ? &m->normal : &renderer->fallback_normal_texture,
            m->occlusion.image ? &m->occlusion : (m->use_metallic_roughness_red_as_occlusion && m->metallic_roughness.image ? &m->metallic_roughness : &renderer->fallback_linear_texture),
            &renderer->fallback_linear_texture);
    }
    for (uint32_t i = 0; i < scene->primitive_count; ++i)
        mesh_upload(renderer, &scene->primitives[i].mesh);
    (void)error;
    return GLTF_LOAD_OK;
}
GltfLoadResult gltf_scene_create(struct Renderer *renderer,const char *path,const GltfLoadOptions *options,GltfScene *out,GltfLoadError *error)
{
    GltfLoadResult result = gltf_scene_parse(path, options, out, error);
    if (result != GLTF_LOAD_OK || !renderer) return result;
    result = gltf_scene_upload(renderer, out, error);
    if (result != GLTF_LOAD_OK) gltf_scene_destroy(renderer, out);
    return result;
}
void gltf_scene_destroy(struct Renderer *renderer,GltfScene *scene){if(!scene)return;for(uint32_t i=0;i<scene->primitive_count;i++){if(renderer)mesh_destroy(renderer,&scene->primitives[i].mesh);free(scene->primitives[i].vertices);free(scene->primitives[i].indices);}for(uint32_t i=0;i<scene->material_count;i++){GltfMaterial *m=&scene->materials[i];if(renderer){if(m->descriptor_set)renderer_free_material_set(renderer,m->descriptor_set);if(m->base_color.image)texture_destroy(renderer->device,renderer->allocator,&m->base_color);if(m->metallic_roughness.image)texture_destroy(renderer->device,renderer->allocator,&m->metallic_roughness);if(m->normal.image)texture_destroy(renderer->device,renderer->allocator,&m->normal);if(m->occlusion.image)texture_destroy(renderer->device,renderer->allocator,&m->occlusion);}free(m->base_color_path);free(m->metallic_roughness_path);free(m->normal_path);free(m->occlusion_path);}free((char *)scene->source_path);free(scene->primitives);free(scene->materials);*scene=(GltfScene){0};}
