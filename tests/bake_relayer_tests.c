#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "bake_gpu.h"
#include "bake_relayer.h"

int main(void) {
    BakeGpu gpu;
    if (!bake_gpu_init(&gpu)) return 77;
    const int width = 67, height = 59, count = width * height;
    size_t ubytes = (size_t)count * sizeof(uint32_t);
    size_t fbytes = (size_t)count * 3 * 3 * sizeof(float);
    uint32_t *base = malloc(ubytes), *labels = malloc(ubytes), *owners = malloc(ubytes);
    uint32_t *cpu = malloc(ubytes);
    float *low = malloc(fbytes), *mid = malloc(fbytes), *meso = malloc(fbytes), *fine = malloc(fbytes);
    for (int i = 0; i < count; ++i) {
        labels[i] = (uint32_t)((i / width / 17 + i / 13) % 3);
        owners[i] = labels[i] * (uint32_t)count + (uint32_t)((i * 37 + i / width) % count);
        base[i] = 0xff808080u;
    }
    for (int i = 0; i < count * 9; ++i) {
        low[i] = 70.0f + (float)(i % 113) * 0.7f;
        mid[i] = (float)(i % 23) - 11.0f;
        meso[i] = ((float)(i % 17) - 8.0f) * 0.7f;
        fine[i] = ((float)(i % 9) - 4.0f) * 0.35f;
    }
    bake_relayer_cpu(base, labels, owners, low, mid, meso, fine, cpu, width, height);
    BakeBuffer bbase = bake_buffer_host(&gpu, ubytes), blabel = bake_buffer_host(&gpu, ubytes);
    BakeBuffer bowner = bake_buffer_host(&gpu, ubytes), bout = bake_buffer_host(&gpu, ubytes);
    BakeBuffer blow = bake_buffer_host(&gpu, fbytes), bmid = bake_buffer_host(&gpu, fbytes);
    BakeBuffer bmeso = bake_buffer_host(&gpu, fbytes), bfine = bake_buffer_host(&gpu, fbytes);
    for (int i = 0; i < count; ++i) {
        ((uint32_t *)bbase.mapped)[i] = base[i]; ((uint32_t *)blabel.mapped)[i] = labels[i];
        ((uint32_t *)bowner.mapped)[i] = owners[i];
    }
    for (int i = 0; i < count * 9; ++i) {
        ((float *)blow.mapped)[i] = low[i]; ((float *)bmid.mapped)[i] = mid[i];
        ((float *)bmeso.mapped)[i] = meso[i]; ((float *)bfine.mapped)[i] = fine[i];
    }
    bake_relayer_gpu(&gpu, &bbase, &blabel, &bowner, &blow, &bmid, &bmeso,
                     &bfine, &bout, width, height);
    int bad = 0;
    for (int i = 0; i < count; ++i) bad += ((uint32_t *)bout.mapped)[i] != cpu[i];
    bake_buffer_destroy(&gpu, &bfine); bake_buffer_destroy(&gpu, &bmeso);
    bake_buffer_destroy(&gpu, &bmid); bake_buffer_destroy(&gpu, &blow);
    bake_buffer_destroy(&gpu, &bout); bake_buffer_destroy(&gpu, &bowner);
    bake_buffer_destroy(&gpu, &blabel); bake_buffer_destroy(&gpu, &bbase);
    bake_gpu_destroy(&gpu); free(base); free(labels); free(owners); free(cpu);
    free(low); free(mid); free(meso); free(fine);
    if (bad) { fprintf(stderr, "bake_relayer_tests: FAIL (%d pixels)\n", bad); return 1; }
    printf("bake_relayer_tests: OK (%dx%d output bit-exact)\n", width, height);
    return 0;
}
