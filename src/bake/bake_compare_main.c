// Decoded-pixel acceptance comparator for the five frozen pass17 PNGs.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "bake_image.h"

static const char *names[] = {
    "full_snow", "snow_rock", "full_rock", "rock_grass", "full_grass"};

enum { RGB_CHANNELS = 3 };
static const double minimum_similarity_percent = 99.0;

static int path(char *out,size_t size,const char *directory,const char *name){
    int n=snprintf(out,size,"%s/%s.png",directory,name);
    return n>=0&&(size_t)n<size;
}

int main(int argc,char **argv){
    if(argc!=3){fprintf(stderr,"usage: %s GENERATED_DIR PASS17_DIR\n",argv[0]);return 2;}
    int failed=0;
    for(size_t preset=0;preset<sizeof(names)/sizeof(names[0]);++preset){
        char generated_path[1024],golden_path[1024];BakeImage generated={0},golden={0};
        if(!path(generated_path,sizeof(generated_path),argv[1],names[preset])||
           !path(golden_path,sizeof(golden_path),argv[2],names[preset])||
           !bake_image_load(generated_path,&generated)||!bake_image_load(golden_path,&golden)){
            fprintf(stderr,"%s: could not load comparison pair\n",names[preset]);failed=1;goto next;
        }
        if(generated.width!=golden.width||generated.height!=golden.height){
            fprintf(stderr,"%s: dimensions %dx%d != %dx%d\n",names[preset],generated.width,
                    generated.height,golden.width,golden.height);failed=1;goto next;
        }
        size_t pixels=(size_t)generated.width*generated.height,exact=0;
        size_t channels=pixels*RGB_CHANNELS;
        uint64_t absolute=0,squared=0;unsigned maximum=0;
        const uint8_t *a=(const uint8_t*)generated.pixels,*b=(const uint8_t*)golden.pixels;
        for(size_t p=0;p<pixels;++p){int pixel_exact=1;for(int c=0;c<RGB_CHANNELS;++c){
            int difference=(int)a[p*4+c]-(int)b[p*4+c];if(difference<0)difference=-difference;
            pixel_exact&=difference==0;absolute+=(unsigned)difference;squared+=(uint64_t)difference*difference;
            if((unsigned)difference>maximum)maximum=(unsigned)difference;}exact+=(size_t)pixel_exact;}
        double exact_percent=100.0*(double)exact/(double)pixels;
        double mae=(double)absolute/(double)channels,rmse=sqrt((double)squared/(double)channels);
        double similarity=100.0*(1.0-mae/255.0);
        printf("%-11s similarity=%8.4f%% exact=%8.4f%% RGB_MAE=%7.3f "
               "RGB_RMSE=%7.3f max=%3u %s\n",names[preset],similarity,
               exact_percent,mae,rmse,maximum,
               similarity>=minimum_similarity_percent?"PASS":"FAIL");
        if(similarity<minimum_similarity_percent)failed=1;
next:  bake_image_free(&golden);bake_image_free(&generated);
    }
    if(failed){fprintf(stderr,"pass17 comparison: FAIL (requires 99.0%% RGB similarity per image)\n");return 1;}
    puts("pass17 comparison: OK (all five images >=99.0% RGB similarity)");return 0;
}
