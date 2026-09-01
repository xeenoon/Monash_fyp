#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "bake_bands.h"
#include "bake_gpu.h"

static int differences(const float *a, const float *b, size_t n) {
    int bad = 0;
    for (size_t i = 0; i < n; ++i) bad += a[i] != b[i];
    return bad;
}

int main(void) {
    BakeGpu gpu;
    if (!bake_gpu_init(&gpu)) return 77;
    const int width = 93, height = 71, count = width * height;
    const size_t samples = (size_t)count * 3, bytes = samples * sizeof(float);
    uint32_t *pixels = malloc((size_t)count * sizeof(uint32_t));
    float *cpu_store = malloc(bytes * 5);
    BakeBandsCpu cpu = {cpu_store, cpu_store + samples, cpu_store + samples * 2,
                        cpu_store + samples * 3, cpu_store + samples * 4};
    for (int i = 0; i < count; ++i) {
        uint32_t r = (uint32_t)((i * 31 + i / width * 17) & 255);
        uint32_t g = (uint32_t)((i * 13 + 91) & 255);
        uint32_t b = (uint32_t)((i * 7 + i / 11) & 255);
        pixels[i] = r | (g << 8) | (b << 16) | 0xff000000u;
    }
    bake_bands_cpu(pixels, cpu, width, height);
    BakeBuffer in = bake_buffer_host(&gpu, (size_t)count * sizeof(uint32_t));
    BakeBuffer rgb = bake_buffer_host(&gpu, bytes), low = bake_buffer_host(&gpu, bytes);
    BakeBuffer mid = bake_buffer_host(&gpu, bytes), meso = bake_buffer_host(&gpu, bytes);
    BakeBuffer fine = bake_buffer_host(&gpu, bytes);
    for (int i = 0; i < count; ++i) ((uint32_t *)in.mapped)[i] = pixels[i];
    bake_bands_gpu(&gpu, &in, &rgb, &low, &mid, &meso, &fine, width, height);
    int bad[5] = {differences(cpu.rgb, rgb.mapped, samples),
                  differences(cpu.low32, low.mapped, samples),
                  differences(cpu.mid, mid.mapped, samples),
                  differences(cpu.meso, meso.mapped, samples),
                  differences(cpu.fine, fine.mapped, samples)};
    bake_buffer_destroy(&gpu, &fine); bake_buffer_destroy(&gpu, &meso);
    bake_buffer_destroy(&gpu, &mid); bake_buffer_destroy(&gpu, &low);
    bake_buffer_destroy(&gpu, &rgb); bake_buffer_destroy(&gpu, &in);
    bake_gpu_destroy(&gpu); free(pixels); free(cpu_store);
    int total = bad[0] + bad[1] + bad[2] + bad[3] + bad[4];
    if (total) {
        fprintf(stderr, "bake_bands_tests: FAIL rgb=%d low=%d mid=%d meso=%d fine=%d\n",
                bad[0], bad[1], bad[2], bad[3], bad[4]);
        return 1;
    }
    printf("bake_bands_tests: OK (%dx%dx3, all five fields bit-exact)\n",
           width, height);
    return 0;
}
