#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "bake_gpu.h"
#include "bake_residual.h"

int main(void) {
    BakeGpu gpu;
    if (!bake_gpu_init(&gpu)) return 77;
    enum { W=32,H=32,P=8,N=3 };
    BakeBuffer bands=bake_buffer_host(&gpu,N*P*P*3*sizeof(float));
    BakeBuffer existing=bake_buffer_host(&gpu,W*H*3*sizeof(float));
    BakeBuffer known=bake_buffer_host(&gpu,W*H*sizeof(uint32_t));
    BakeBuffer base=bake_buffer_host(&gpu,N*sizeof(float));
    BakeBuffer scores=bake_buffer_host(&gpu,N*sizeof(float));
    for(int y=0;y<H;++y)for(int x=0;x<W;++x){
        ((uint32_t*)known.mapped)[y*W+x]=(uint32_t)((x+y)%5!=0);
        for(int c=0;c<3;++c)((float*)existing.mapped)[(y*W+x)*3+c]=(float)((x*3+y*7+c*19)%41-20);
    }
    for(int q=0;q<N;++q)for(int y=0;y<P;++y)for(int x=0;x<P;++x)
        for(int c=0;c<3;++c)((float*)bands.mapped)[((q*P*P+y*P+x)*3)+c]=(float)((x*11+y*5+c*13+q*17)%53-26);
    ((float*)base.mapped)[0]=0.2f;((float*)base.mapped)[1]=1.1f;((float*)base.mapped)[2]=2.3f;
    bake_residual_score_gpu(&gpu,&bands,&existing,&known,&base,&scores,W,H,4,5,P,3,N);
    const float expected[N]={1.1048520803f,2.1148748398f,2.8862767220f};
    int failed=0;
    for(int i=0;i<N;++i){float error=fabsf(((float*)scores.mapped)[i]-expected[i]);
        if(error>2e-5f){fprintf(stderr,"score %d %.9g expected %.9g (error %.3g)\n",i,((float*)scores.mapped)[i],expected[i],error);failed=1;}}
    bake_buffer_destroy(&gpu,&scores);bake_buffer_destroy(&gpu,&base);
    bake_buffer_destroy(&gpu,&known);bake_buffer_destroy(&gpu,&existing);
    bake_buffer_destroy(&gpu,&bands);bake_gpu_destroy(&gpu);
    if(!failed)puts("bake_residual_score_tests: OK (GPU shortlist scores match pass17)");
    return failed;
}
