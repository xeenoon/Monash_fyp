#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "bake_gpu.h"
#include "bake_pipeline.h"
int main(void){BakeGpu gpu;if(!bake_gpu_init(&gpu))return 77;const int w=64,h=64,n=w*h;size_t ub=(size_t)n*sizeof(uint32_t);uint32_t *atlas=malloc(ub*3),*cpu=malloc(ub*BAKE_PRESET_COUNT);float *heightmap=malloc((size_t)n*sizeof(float));
for(int y=0;y<h;++y)for(int x=0;x<w;++x){heightmap[y*w+x]=(float)y+7.0f*sinf((float)x*0.2f);for(int m=0;m<3;++m){uint32_t r=(uint32_t)((x*5+y*3+m*61)&255),g=(uint32_t)((x*2+y*7+m*47)&255),b=(uint32_t)((x*11+y+m*83)&255);atlas[m*n+y*w+x]=r|(g<<8)|(b<<16)|0xff000000u;}}
bake_pipeline_cpu(atlas,heightmap,cpu,w,h,31.5f,151500u);BakeBuffer ba=bake_buffer_host(&gpu,ub*3),bh=bake_buffer_host(&gpu,(size_t)n*sizeof(float)),out[BAKE_PRESET_COUNT];for(int i=0;i<n*3;++i)((uint32_t*)ba.mapped)[i]=atlas[i];for(int i=0;i<n;++i)((float*)bh.mapped)[i]=heightmap[i];for(int p=0;p<BAKE_PRESET_COUNT;++p)out[p]=bake_buffer_host(&gpu,ub);bake_pipeline_gpu(&gpu,&ba,&bh,out,w,h,31.5f,151500u);int bad[BAKE_PRESET_COUNT]={0};for(int p=0;p<BAKE_PRESET_COUNT;++p)for(int i=0;i<n;++i)bad[p]+=((uint32_t*)out[p].mapped)[i]!=cpu[p*n+i];for(int p=0;p<BAKE_PRESET_COUNT;++p)bake_buffer_destroy(&gpu,&out[p]);bake_buffer_destroy(&gpu,&bh);bake_buffer_destroy(&gpu,&ba);bake_gpu_destroy(&gpu);free(atlas);free(cpu);free(heightmap);int total=0;for(int p=0;p<BAKE_PRESET_COUNT;++p)total+=bad[p];if(total){fprintf(stderr,"bake_pipeline_tests: FAIL");for(int p=0;p<BAKE_PRESET_COUNT;++p)fprintf(stderr," %s=%d",bake_preset_names[p],bad[p]);fputc('\n',stderr);return 1;}printf("bake_pipeline_tests: OK (five presets end-to-end bit-exact)\n");return 0;}
