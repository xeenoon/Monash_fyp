#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "bake_gpu.h"
#include "bake_quilt.h"

int main(void) {
    BakeGpu gpu;
    if (!bake_gpu_init(&gpu)) return 77;
    const int width = 101, height = 83, count = width * height;
    size_t bytes = (size_t)count * sizeof(uint32_t);
    uint32_t *atlas = malloc(bytes * 3), *labels = malloc(bytes);
    uint32_t *cpu = malloc(bytes), *cpu_owner = malloc(bytes);
    for (int m = 0; m < 3; ++m) for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            uint32_t r = (uint32_t)((x * (m + 3) + y * 7 + m * 70) & 255);
            uint32_t g = (uint32_t)((x * 11 + y * (m + 5) + m * 31) & 255);
            uint32_t b = (uint32_t)((x * 3 + y * 13 + m * 97) & 255);
            atlas[m * count + y * width + x] = r | (g << 8) | (b << 16) | 0xff000000u;
        }
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x)
        labels[y * width + x] = (uint32_t)((x / 19 + y / 23) % 3);
    bake_quilt_cpu(atlas, labels, cpu, cpu_owner, width, height, 151500u);
    BakeBuffer a = bake_buffer_host(&gpu, bytes * 3), l = bake_buffer_host(&gpu, bytes);
    BakeBuffer o = bake_buffer_host(&gpu, bytes), own = bake_buffer_host(&gpu, bytes);
    for (int i = 0; i < count * 3; ++i) ((uint32_t *)a.mapped)[i] = atlas[i];
    for (int i = 0; i < count; ++i) ((uint32_t *)l.mapped)[i] = labels[i];
    bake_quilt_gpu(&gpu, &a, &l, &o, &own, width, height, 151500u);
    int image_bad = 0, owner_bad = 0;
    for (int i = 0; i < count; ++i) {
        image_bad += ((uint32_t *)o.mapped)[i] != cpu[i];
        owner_bad += ((uint32_t *)own.mapped)[i] != cpu_owner[i];
    }
    bake_buffer_destroy(&gpu, &own); bake_buffer_destroy(&gpu, &o);
    bake_buffer_destroy(&gpu, &l); bake_buffer_destroy(&gpu, &a);
    bake_gpu_destroy(&gpu); free(atlas); free(labels); free(cpu); free(cpu_owner);
    if (image_bad || owner_bad) {
        fprintf(stderr, "bake_quilt_tests: FAIL image=%d owner=%d\n", image_bad, owner_bad);
        return 1;
    }
    printf("bake_quilt_tests: OK (%dx%d parallel patch owners+RGB bit-exact)\n",
           width, height);
    return 0;
}
