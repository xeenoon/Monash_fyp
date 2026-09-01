#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "bake_ordered_quilt.h"

int main(void){
    BakeGpu gpu;if(!bake_gpu_init(&gpu))return 77;
    enum{W=8,H=8,D=2,P=4};size_t pixels=W*H;
    BakeBuffer atlas=bake_buffer_host(&gpu,D*pixels*sizeof(uint32_t));
    for(int d=0;d<D;++d)for(int y=0;y<H;++y)for(int x=0;x<W;++x){uint32_t r=(x*17+y*9+d*51)&255,g=(x*7+y*13+d*23)&255,b=(x*3+y*19+d*31)&255;
        ((uint32_t*)atlas.mapped)[d*pixels+y*W+x]=r|(g<<8)|(b<<16)|0xff000000u;}
    BakeOrderedState s={bake_buffer_host(&gpu,pixels*3*sizeof(float)),bake_buffer_host(&gpu,pixels*sizeof(uint32_t)),
        bake_buffer_host(&gpu,pixels*sizeof(int32_t)),bake_buffer_host(&gpu,pixels*sizeof(int32_t)),
        bake_buffer_host(&gpu,pixels*sizeof(int32_t)),bake_buffer_host(&gpu,D*pixels*sizeof(int32_t))};
    memset(s.level_band.mapped,0,pixels*3*sizeof(float));memset(s.level_known.mapped,0,pixels*sizeof(uint32_t));
    memset(s.donor_map.mapped,0xff,pixels*sizeof(int32_t));memset(s.source_x.mapped,0xff,pixels*sizeof(int32_t));
    memset(s.source_y.mapped,0xff,pixels*sizeof(int32_t));memset(s.source_usage.mapped,0,D*pixels*sizeof(int32_t));
    BakeOrderedPlacement placements[2]={{0,0,0,3},{3,0,3,3}};
    BakeResidualCandidate candidates[6]={{0,0,0,0},{1,1,1,1},{0,2,2,2},{1,0,2,3},{0,3,1,4},{1,4,3,5}};
    float base[6]={0.4f,0.2f,0.8f,0.3f,0.7f,0.1f};int selected[2]={-1,-1};BakePcg64 rng;
    if(!bake_pcg64_pass17_seed(&rng,151500)||!bake_ordered_residual_level_gpu(&gpu,&atlas,W,H,placements,2,candidates,base,P,3,2,&rng,&s,W,H,selected))return 1;
    int known_count=0,usage=0,failed=selected[0]<0||selected[0]>=3||selected[1]<0||selected[1]>=3;
    for(size_t i=0;i<pixels;++i)known_count+=((uint32_t*)s.level_known.mapped)[i]!=0;
    for(size_t i=0;i<D*pixels;++i)usage+=((int32_t*)s.source_usage.mapped)[i];
    if(known_count!=28||usage!=known_count)failed=1;
    bake_buffer_destroy(&gpu,&s.source_usage);bake_buffer_destroy(&gpu,&s.source_y);bake_buffer_destroy(&gpu,&s.source_x);
    bake_buffer_destroy(&gpu,&s.donor_map);bake_buffer_destroy(&gpu,&s.level_known);bake_buffer_destroy(&gpu,&s.level_band);
    bake_buffer_destroy(&gpu,&atlas);bake_gpu_destroy(&gpu);
    if(failed){fprintf(stderr,"ordered scheduler failed: selected=%d,%d known=%d usage=%d\n",selected[0],selected[1],known_count,usage);return 1;}
    printf("bake_ordered_quilt_tests: OK (winners %d,%d committed sequentially)\n",selected[0],selected[1]);return 0;
}
