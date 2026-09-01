#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "bake_gpu.h"
#include "bake_residual.h"

int main(void){
    BakeGpu gpu;if(!bake_gpu_init(&gpu))return 77;
    enum{W=12,H=10,DW=16,DH=16,P=4,N=2};
    BakeBuffer bands=bake_buffer_host(&gpu,N*P*P*3*sizeof(float));
    BakeBuffer take=bake_buffer_host(&gpu,P*P*sizeof(uint32_t));
    BakeBuffer candidates=bake_buffer_host(&gpu,N*sizeof(BakeResidualCandidate));
    BakeBuffer existing=bake_buffer_host(&gpu,W*H*3*sizeof(float));
    BakeBuffer known=bake_buffer_host(&gpu,W*H*sizeof(uint32_t));
    BakeBuffer donor=bake_buffer_host(&gpu,W*H*sizeof(int32_t));
    BakeBuffer sx=bake_buffer_host(&gpu,W*H*sizeof(int32_t));
    BakeBuffer sy=bake_buffer_host(&gpu,W*H*sizeof(int32_t));
    BakeBuffer usage=bake_buffer_host(&gpu,2*DW*DH*sizeof(int32_t));
    memset(existing.mapped,0,W*H*3*sizeof(float));memset(known.mapped,0,W*H*sizeof(uint32_t));
    memset(donor.mapped,0xff,W*H*sizeof(int32_t));memset(sx.mapped,0xff,W*H*sizeof(int32_t));
    memset(sy.mapped,0xff,W*H*sizeof(int32_t));memset(usage.mapped,0,2*DW*DH*sizeof(int32_t));
    BakeResidualCandidate record[2]={{0,1,2,8},{1,5,6,12}};memcpy(candidates.mapped,record,sizeof(record));
    for(int q=0;q<N;++q)for(int p=0;p<P*P;++p)for(int c=0;c<3;++c)
        ((float*)bands.mapped)[(q*P*P+p)*3+c]=(float)(q*100+p*3+c);
    int expected_taken=0;for(int p=0;p<P*P;++p){uint32_t v=(uint32_t)(p%3!=0);((uint32_t*)take.mapped)[p]=v;expected_taken+=(int)v;}
    bake_residual_commit_gpu(&gpu,&bands,&take,&candidates,&existing,&known,&donor,&sx,&sy,&usage,W,H,DW,DH,3,2,P,1);
    int failed=0,taken=0,usage_total=0;
    for(int y=0;y<P;++y)for(int x=0;x<P;++x){int p=y*P+x,t=(2+y)*W+3+x;
        if(((uint32_t*)take.mapped)[p]){++taken;if(!((uint32_t*)known.mapped)[t]||((int32_t*)donor.mapped)[t]!=1||
            ((int32_t*)sx.mapped)[t]!=5+x||((int32_t*)sy.mapped)[t]!=6+y||
            fabsf(((float*)existing.mapped)[t*3]-(float)(100+p*3))>1e-6f)failed=1;}
        else if(((uint32_t*)known.mapped)[t])failed=1;}
    for(int i=0;i<2*DW*DH;++i)usage_total+=((int32_t*)usage.mapped)[i];
    if(taken!=expected_taken||usage_total!=expected_taken)failed=1;
    bake_buffer_destroy(&gpu,&usage);bake_buffer_destroy(&gpu,&sy);bake_buffer_destroy(&gpu,&sx);
    bake_buffer_destroy(&gpu,&donor);bake_buffer_destroy(&gpu,&known);bake_buffer_destroy(&gpu,&existing);
    bake_buffer_destroy(&gpu,&candidates);bake_buffer_destroy(&gpu,&take);bake_buffer_destroy(&gpu,&bands);bake_gpu_destroy(&gpu);
    if(failed){fprintf(stderr,"ordered commit failed (taken %d/%d, usage %d)\n",taken,expected_taken,usage_total);return 1;}
    puts("bake_residual_commit_tests: OK (ordered GPU ownership/usage commit)");return 0;
}
