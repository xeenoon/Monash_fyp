#include "material_stability_demo.h"
#include "renderer.h"
#include <math.h>
#include <stdlib.h>

#define DEMO_DIFF BENCHMARK_DIR "/blue_metal_plate/blue_metal_plate_diff_4k.jpg"
#define DEMO_ORM BENCHMARK_DIR "/blue_metal_plate/blue_metal_plate_arm_4k.png"
#define DEMO_NORMAL BENCHMARK_DIR "/blue_metal_plate/blue_metal_plate_nor_gl_4k.png"

static bool make_panel(Renderer *r, Mesh *mesh, float center_x, float center_y,
                       float width, float height, float curve)
{
    const uint32_t nx = 48, ny = 32;
    const uint32_t count = (nx + 1) * (ny + 1);
    Vertex *v = calloc(count, sizeof(*v));
    uint32_t *ix = calloc(nx * ny * 6, sizeof(*ix));
    if (!v || !ix) { free(v); free(ix); return false; }
    for (uint32_t y = 0; y <= ny; ++y) for (uint32_t x = 0; x <= nx; ++x) {
        float u = (float)x / nx, q = (u - .5f) * width, t = (float)y / ny;
        float dzdx = 2.f * curve * q;
        vec3s n = glms_vec3_normalize((vec3s){{-dzdx, 0.f, 1.f}});
        v[y * (nx + 1) + x] = (Vertex){.position={center_x + q, center_y + (t-.5f)*height, curve*q*q},
            /* Blue Metal Plate is authored as one 2.5m square: preserve that
             * scale so its seam lines are evidence, not an accidental grid. */
            .normal={n.x,n.y,n.z}, .texcoord={u,t}, .tangent={1,0,dzdx,1}};
    }
    uint32_t k=0; for (uint32_t y=0;y<ny;++y) for(uint32_t x=0;x<nx;++x) {
        uint32_t a=y*(nx+1)+x, b=a+1, c=a+nx+1, d=c+1;
        ix[k++]=a; ix[k++]=b; ix[k++]=d; ix[k++]=a; ix[k++]=d; ix[k++]=c;
    }
    *mesh=(Mesh){.local_to_world=coordinate_identity_transform((WorldPosition){0}), .vertices=v,
        .vertex_count=count,.indices=ix,.index_count=k};
    mesh_upload(r,mesh); return true;
}

bool material_stability_demo_create(Renderer *r, MaterialStabilityDemo *d)
{
    if (!r || !d) return false;
    *d=(MaterialStabilityDemo){0};
    texture_load(r->device,r->allocator,r->upload,&d->albedo,DEMO_DIFF,r->max_anisotropy);
    texture_load_linear(r->device,r->allocator,r->upload,&d->orm,DEMO_ORM,r->max_anisotropy);
    texture_load_linear(r->device,r->allocator,r->upload,&d->normal,DEMO_NORMAL,r->max_anisotropy);
    const uint8_t occluded[4]={0,0,0,255};
    texture_create_solid_rgba8(r->device,r->allocator,r->upload,&d->cavity,occluded,false);
    d->neutral_set=renderer_allocate_pbr5_set(r,&d->albedo,&d->orm,&d->normal,&d->orm,&r->fallback_linear_texture);
    d->cavity_set=renderer_allocate_pbr5_set(r,&d->albedo,&d->orm,&d->normal,&d->orm,&d->cavity);
    /* panel 0 is the flyby hero.  1/2 are the optional stationary C/D split;
     * the small cavity fixture is deliberately kept below the beauty framing. */
    return make_panel(r,&d->panels[0], 0.f,1.0f,2.5f,2.5f,.16f) &&
           make_panel(r,&d->panels[1],-1.4f,1.0f,2.5f,2.5f,.16f) &&
           make_panel(r,&d->panels[2], 1.4f,1.0f,2.5f,2.5f,.16f);
}

void material_stability_demo_destroy(Renderer *r, MaterialStabilityDemo *d)
{
    if (!r || !d) return;
    for (uint32_t i=0;i<3;++i) { mesh_destroy(r,&d->panels[i]); free((void *)d->panels[i].vertices); free((void *)d->panels[i].indices); }
    if (d->neutral_set) renderer_free_material_set(r, d->neutral_set);
    if (d->cavity_set) renderer_free_material_set(r, d->cavity_set);
    texture_destroy(r->device,r->allocator,&d->albedo); texture_destroy(r->device,r->allocator,&d->orm);
    texture_destroy(r->device,r->allocator,&d->normal); texture_destroy(r->device,r->allocator,&d->cavity); *d=(MaterialStabilityDemo){0};
}
