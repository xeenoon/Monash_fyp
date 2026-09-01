#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "bake_gpu.h"
#include "bake_transition.h"
int main(void){BakeGpu gpu;if(!bake_gpu_init(&gpu))return 77;const int w=96,h=78,n=w*h;size_t ub=(size_t)n*sizeof(uint32_t),ab=ub*3;float *heightmap=malloc((size_t)n*sizeof(float));uint32_t *atlas=malloc(ab),*cl=malloc(ub),*co=malloc(ub),*cq=malloc(ub);int32_t *cs=malloc(ub);
for(int y=0;y<h;++y)for(int x=0;x<w;++x){heightmap[y*w+x]=(float)y+9.0f*sinf((float)x*0.13f)+3.0f*cosf((float)y*0.21f);for(int m=0;m<3;++m){uint32_t r=(uint32_t)((x*3+y*5+m*71)&255),g=(uint32_t)((x*7+y*2+m*43)&255),b=(uint32_t)((x+y*11+m*89)&255);atlas[m*n+y*w+x]=r|(g<<8)|(b<<16)|0xff000000u;}}
bake_transition_cpu(heightmap,atlas,cl,cs,co,cq,w,h,38.5f,1,0,919u);BakeBuffer bh=bake_buffer_host(&gpu,(size_t)n*sizeof(float)),ba=bake_buffer_host(&gpu,ab),gl=bake_buffer_host(&gpu,ub),gs=bake_buffer_host(&gpu,ub),go=bake_buffer_host(&gpu,ub),gq=bake_buffer_host(&gpu,ub);for(int i=0;i<n;++i)((float*)bh.mapped)[i]=heightmap[i];for(int i=0;i<n*3;++i)((uint32_t*)ba.mapped)[i]=atlas[i];bake_transition_gpu(&gpu,&bh,&ba,&gl,&gs,&go,&gq,w,h,38.5f,1,0,919u);
int lb=0,sb=0,ob=0,qb=0;for(int i=0;i<n;++i){lb+=((uint32_t*)gl.mapped)[i]!=cl[i];sb+=((int32_t*)gs.mapped)[i]!=cs[i];ob+=((uint32_t*)go.mapped)[i]!=co[i];qb+=((uint32_t*)gq.mapped)[i]!=cq[i];}bake_buffer_destroy(&gpu,&gq);bake_buffer_destroy(&gpu,&go);bake_buffer_destroy(&gpu,&gs);bake_buffer_destroy(&gpu,&gl);bake_buffer_destroy(&gpu,&ba);bake_buffer_destroy(&gpu,&bh);bake_gpu_destroy(&gpu);free(heightmap);free(atlas);free(cl);free(cs);free(co);free(cq);if(lb||sb||ob||qb){fprintf(stderr,"bake_transition_tests: FAIL label=%d sdf=%d owner=%d rgb=%d\n",lb,sb,ob,qb);return 1;}printf("bake_transition_tests: OK (%dx%d JFA-SDF+PatchMatch bit-exact)\n",w,h);return 0;}
