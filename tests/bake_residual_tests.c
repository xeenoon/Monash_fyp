#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bake_filter.h"
#include "bake_gpu.h"
#include "bake_residual.h"

static int reflect_index(int value, int extent) {
    while (value < 0 || value >= extent)
        value = value < 0 ? -value - 1 : 2 * extent - value - 1;
    return value;
}

static void reference(const uint32_t *atlas, const BakeResidualCandidate *candidate,
                      int donor_width, int donor_height, int patch, int level,
                      int phase_seed, float *out) {
    size_t values=(size_t)patch*patch*3;
    float *raw=malloc(values*sizeof(float)),*small=malloc(values*sizeof(float));
    float *large=calloc(values,sizeof(float)),*scratch=malloc(values*sizeof(float));
    int px=((candidate->database_index*17+level*11+phase_seed*5)%13)-6;
    int py=((candidate->database_index*29+level*7+phase_seed*3)%13)-6;
    for(int y=0;y<patch;++y)for(int x=0;x<patch;++x){
        int lx=reflect_index(x-px,patch),ly=reflect_index(y-py,patch);
        uint32_t p=atlas[(size_t)candidate->donor*donor_width*donor_height+
                         (size_t)(candidate->source_y+ly)*donor_width+
                         candidate->source_x+lx];
        size_t i=((size_t)y*patch+x)*3;
        raw[i]=(float)(p&255);raw[i+1]=(float)((p>>8)&255);raw[i+2]=(float)((p>>16)&255);
    }
    int radius=bake_gaussian_radius(level==1?1.2f:2.0f,4.0f);
    float *weights=malloc((size_t)(2*radius+1)*sizeof(float));
    bake_gaussian_weights(level==1?1.2f:2.0f,radius,weights);
    bake_gaussian_cpu_reflect(raw,small,scratch,patch,patch,3,weights,radius);
    if(level<2){
        float sigma=level==0?16.0f:6.0f; radius=bake_gaussian_radius(sigma,4.0f);
        weights=realloc(weights,(size_t)(2*radius+1)*sizeof(float));
        bake_gaussian_weights(sigma,radius,weights);
        bake_gaussian_cpu_reflect(raw,large,scratch,patch,patch,3,weights,radius);
    }
    float weight=level==0?0.55f:(level==1?0.45f:0.95f),mean[3]={0};
    for(size_t p=0;p<(size_t)patch*patch;++p)for(int c=0;c<3;++c){
        float v=level==2?raw[p*3+c]-small[p*3+c]:small[p*3+c]-large[p*3+c];
        out[p*3+c]=weight*v;mean[c]+=out[p*3+c];
    }
    for(int c=0;c<3;++c)mean[c]/=(float)(patch*patch);
    float sum=0;
    for(size_t i=0;i<values;++i){out[i]-=mean[i%3];sum+=out[i]*out[i];}
    float rms=sqrtf(sum/(float)values),scale=fminf(1.0f,28.0f/fmaxf(rms,0.0001f));
    for(size_t i=0;i<values;++i)out[i]*=scale;
    free(weights);free(scratch);free(large);free(small);free(raw);
}

int main(void){
    BakeGpu gpu;if(!bake_gpu_init(&gpu))return 77;
    const int w=40,h=36,patch=24,n=3;size_t plane=(size_t)w*h;
    uint32_t *atlas=malloc(plane*2*sizeof(uint32_t));
    for(int d=0;d<2;++d)for(int y=0;y<h;++y)for(int x=0;x<w;++x){
        uint32_t r=(uint32_t)((x*9+y*5+d*41)&255),g=(uint32_t)((x*3+y*11+d*17)&255);
        uint32_t b=(uint32_t)((x*13+y*7+d*29)&255);atlas[d*plane+y*w+x]=r|(g<<8)|(b<<16)|0xff000000u;
    }
    BakeResidualCandidate records[3]={{0,2,3,11},{1,7,5,99},{0,12,8,203}};
    size_t values=(size_t)n*patch*patch*3;
    BakeBuffer a=bake_buffer_host(&gpu,plane*2*sizeof(uint32_t));
    BakeBuffer c=bake_buffer_host(&gpu,sizeof(records));BakeBuffer out=bake_buffer_host(&gpu,values*sizeof(float));
    memcpy(a.mapped,atlas,plane*2*sizeof(uint32_t));memcpy(c.mapped,records,sizeof(records));
    bake_residual_batch_gpu(&gpu,&a,&c,&out,w,h,patch,n,2,407);
    int bad=0;float max_error=0;float *expected=malloc((size_t)patch*patch*3*sizeof(float));
    for(int q=0;q<n;++q){reference(atlas,&records[q],w,h,patch,2,407,expected);
        for(size_t i=0;i<(size_t)patch*patch*3;++i){float e=fabsf(((float*)out.mapped)[(size_t)q*patch*patch*3+i]-expected[i]);if(e>max_error)max_error=e;if(e>0.001f)++bad;}}
    free(expected);bake_buffer_destroy(&gpu,&out);bake_buffer_destroy(&gpu,&c);bake_buffer_destroy(&gpu,&a);bake_gpu_destroy(&gpu);free(atlas);
    if(bad){fprintf(stderr,"residual mismatch: %d values, max %.6g\n",bad,max_error);return 1;}
    printf("bake_residual_tests: OK (%d candidates batched, max error %.3g)\n",n,max_error);return 0;
}
