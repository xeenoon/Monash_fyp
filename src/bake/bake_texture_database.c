#include "bake_texture_database.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int axis_origins(int extent,int patch,int stride,int *out){
    int count=0;
    for(int value=0;value<=extent-patch;value+=stride)out[count++]=value;
    if(count==0||out[count-1]!=extent-patch)out[count++]=extent-patch;
    return count;
}

static float srgb_linear(float value){
    return value<=0.04045f?value/12.92f:powf((value+0.055f)/1.055f,2.4f);
}
static float lab_f(float value){
    const float delta=6.0f/29.0f;
    return value>delta*delta*delta?cbrtf(value):value/(3.0f*delta*delta)+4.0f/29.0f;
}
static void pixel_lab(uint32_t p,float out[3]){
    float r=srgb_linear((float)(p&255u)/255.0f);
    float g=srgb_linear((float)((p>>8)&255u)/255.0f);
    float b=srgb_linear((float)((p>>16)&255u)/255.0f);
    float x=(r*.4124564f+g*.3575761f+b*.1804375f)/.95047f;
    float y=r*.2126729f+g*.7151522f+b*.0721750f;
    float z=(r*.0193339f+g*.1191920f+b*.9503041f)/1.08883f;
    float fx=lab_f(x),fy=lab_f(y),fz=lab_f(z);
    out[0]=116.0f*fy-16.0f;out[1]=500.0f*(fx-fy);out[2]=200.0f*(fy-fz);
}

static int allocate_database(BakeTextureDatabase *d,int count){
    d->records=malloc((size_t)count*sizeof(*d->records));
    d->thumbnail=malloc((size_t)count*64);
    d->density=malloc((size_t)count*3*sizeof(float));
    d->terrain=malloc((size_t)count*3*sizeof(float));
    d->terrain_angle=malloc((size_t)count*2*sizeof(float));
    d->mean_lab=malloc((size_t)count*3*sizeof(float));
    d->source_position=malloc((size_t)count*2*sizeof(float));
    d->north_context=malloc((size_t)count*4*8*3);
    d->east_context=malloc((size_t)count*8*4*3);
    d->north_valid=malloc((size_t)count);d->east_valid=malloc((size_t)count);
    d->top_edge=malloc((size_t)count*8*3);d->right_edge=malloc((size_t)count*8*3);
    d->top_gradient=malloc((size_t)count*8*3*sizeof(int16_t));
    d->right_gradient=malloc((size_t)count*8*3*sizeof(int16_t));
    return d->records&&d->thumbnail&&d->density&&d->terrain&&d->terrain_angle&&
           d->mean_lab&&d->source_position&&d->north_context&&d->east_context&&
           d->north_valid&&d->east_valid&&d->top_edge&&d->right_edge&&
           d->top_gradient&&d->right_gradient;
}

bool bake_texture_database_build(const BakeTerrainSample *donors,int donor_count,
                                 int patch,int source_stride,
                                 BakeTextureDatabase *d){
    if(!donors||!d||donor_count<=0||patch<=1||source_stride<=0)return false;
    memset(d,0,sizeof(*d));d->patch=patch;d->source_stride=source_stride;d->donor_count=donor_count;
    int origins[257],count=0;
    for(int donor=0;donor<donor_count;++donor){
        int ny=axis_origins(donors[donor].height,patch,source_stride,origins);
        int nx=axis_origins(donors[donor].width,patch,source_stride,origins);
        count+=nx*ny;
    }
    d->count=count;if(!allocate_database(d,count)){bake_texture_database_free(d);return false;}
    int index=0,thumb[8],context[4];
    for(int i=0;i<8;++i)thumb[i]=(int)lrintf((float)i*(float)(patch-1)/7.0f);
    int band=patch/3<16?patch/3:16;
    for(int i=0;i<4;++i)context[i]=(int)lrintf((float)i*(float)(band-1)/3.0f);
    for(int donor_index=0;donor_index<donor_count;++donor_index){
        const BakeTerrainSample *s=&donors[donor_index];int ys[257],xs[257];
        int ny=axis_origins(s->height,patch,source_stride,ys),nx=axis_origins(s->width,patch,source_stride,xs);
        for(int yi=0;yi<ny;++yi)for(int xi=0;xi<nx;++xi,++index){
            int sy=ys[yi],sx=xs[xi];d->records[index]=(BakeResidualCandidate){donor_index,sx,sy,index};
            double class_count[3]={0},height_sum=0,height_squared=0,slope_sum=0;
            double jxx=0,jyy=0,jxy=0,lab_sum[3]={0};
            for(int y=0;y<patch;++y)for(int x=0;x<patch;++x){size_t p=(size_t)(sy+y)*s->width+sx+x;
                uint32_t kind=s->labels[p]<3?s->labels[p]:0;++class_count[kind];
                double h=s->elevation[p];height_sum+=h;height_squared+=h*h;slope_sum+=s->slope[p];
                double gx=s->gradient_x[p],gy=s->gradient_y[p];jxx+=gx*gx;jyy+=gy*gy;jxy+=gx*gy;
                float lab[3];pixel_lab(s->rgb[p],lab);for(int c=0;c<3;++c)lab_sum[c]+=lab[c];}
            double area=(double)patch*patch,mean=height_sum/area;
            for(int k=0;k<3;++k)d->density[index*3+k]=(float)(class_count[k]/area);
            d->terrain[index*3]=(float)mean;d->terrain[index*3+1]=(float)sqrt(fmax(height_squared/area-mean*mean,0));
            d->terrain[index*3+2]=(float)(slope_sum/area);
            jxx/=area;jyy/=area;jxy/=area;double trace=jxx+jyy,delta=hypot(jxx-jyy,2*jxy);
            d->terrain_angle[index*2]=.5f*(float)atan2(2*jxy,jxx-jyy);
            d->terrain_angle[index*2+1]=(float)(delta/fmax(trace,1e-6));
            for(int c=0;c<3;++c)d->mean_lab[index*3+c]=(float)(lab_sum[c]/area);
            d->source_position[index*2]=(float)s->tile_x+(float)(sx+patch*.5f)/256.0f;
            d->source_position[index*2+1]=(float)s->tile_y+(float)(sy+patch*.5f)/256.0f;
            for(int y=0;y<8;++y)for(int x=0;x<8;++x)
                d->thumbnail[(size_t)index*64+y*8+x]=(uint8_t)s->labels[(size_t)(sy+thumb[y])*s->width+sx+thumb[x]];
            d->north_valid[index]=(uint8_t)(sy>=band);d->east_valid[index]=(uint8_t)(sx+patch+band<=s->width);
            for(int a=0;a<8;++a){uint32_t top=s->rgb[(size_t)sy*s->width+sx+thumb[a]];
                uint32_t top2=s->rgb[(size_t)(sy+1)*s->width+sx+thumb[a]];
                uint32_t right=s->rgb[(size_t)(sy+thumb[a])*s->width+sx+patch-1];
                uint32_t right2=s->rgb[(size_t)(sy+thumb[a])*s->width+sx+patch-2];
                for(int c=0;c<3;++c){int shift=c*8;size_t q=((size_t)index*8+a)*3+c;
                    d->top_edge[q]=(uint8_t)(top>>shift);d->top_gradient[q]=(int16_t)((int)((top2>>shift)&255)-(int)((top>>shift)&255));
                    d->right_edge[q]=(uint8_t)(right>>shift);d->right_gradient[q]=(int16_t)((int)((right>>shift)&255)-(int)((right2>>shift)&255));}}
            memset(d->north_context+(size_t)index*4*8*3,0,4*8*3);memset(d->east_context+(size_t)index*8*4*3,0,8*4*3);
            if(d->north_valid[index])for(int cy=0;cy<4;++cy)for(int ax=0;ax<8;++ax){uint32_t p=s->rgb[(size_t)(sy-band+context[cy])*s->width+sx+thumb[ax]];
                for(int c=0;c<3;++c)d->north_context[((size_t)index*4*8+cy*8+ax)*3+c]=(uint8_t)(p>>(c*8));}
            if(d->east_valid[index])for(int ay=0;ay<8;++ay)for(int cx=0;cx<4;++cx){uint32_t p=s->rgb[(size_t)(sy+thumb[ay])*s->width+sx+patch+context[cx]];
                for(int c=0;c<3;++c)d->east_context[((size_t)index*8*4+ay*4+cx)*3+c]=(uint8_t)(p>>(c*8));}
        }
    }
    return index==count;
}

void bake_texture_database_free(BakeTextureDatabase *d){
    if (!d) return;
    free(d->right_gradient);
    free(d->right_edge);
    free(d->top_gradient);
    free(d->top_edge);
    free(d->east_valid);
    free(d->north_valid);
    free(d->east_context);
    free(d->north_context);
    free(d->source_position);
    free(d->mean_lab);
    free(d->terrain_angle);
    free(d->terrain);
    free(d->density);
    free(d->thumbnail);
    free(d->records);
    memset(d, 0, sizeof(*d));
}
