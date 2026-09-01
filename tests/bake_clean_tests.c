#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "bake_clean.h"
#include "bake_gpu.h"

int main(void) {
    BakeGpu gpu;
    if (!bake_gpu_init(&gpu)) return 77;
    const int width = 197, height = 143, count = width * height;
    uint32_t *input = malloc((size_t)count * sizeof(uint32_t));
    uint32_t *cpu = malloc((size_t)count * sizeof(uint32_t));
    uint32_t *cpu_conf = malloc((size_t)count * sizeof(uint32_t));
    uint32_t *cpu_hard = malloc((size_t)count * sizeof(uint32_t));
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        uint32_t r = (uint32_t)((x * 5 + y * 3) & 255);
        uint32_t g = (uint32_t)((x * 2 + y * 7 + 40) & 255);
        uint32_t b = (uint32_t)((x * 11 + y) & 255);
        if ((x - 40) * (x - 40) + (y - 70) * (y - 70) < 64) r = b = 245, g = 20;
        if (x > 125 && x < 133 && y > 20 && y < 118) r = g = b = 180;
        input[y * width + x] = r | (g << 8) | (b << 16) | 0xff000000u;
    }
    bake_clean_cpu(input, cpu, cpu_conf, cpu_hard, width, height);

    size_t bytes = (size_t)count * sizeof(uint32_t);
    BakeBuffer in = bake_buffer_host(&gpu, bytes), out = bake_buffer_host(&gpu, bytes);
    BakeBuffer conf = bake_buffer_host(&gpu, bytes), hard = bake_buffer_host(&gpu, bytes);
    for (int i = 0; i < count; ++i) ((uint32_t *)in.mapped)[i] = input[i];
    bake_clean_gpu(&gpu, &in, &out, &conf, &hard, width, height);

    int image_bad = 0, conf_bad = 0, hard_bad = 0;
    for (int i = 0; i < count; ++i) {
        image_bad += ((uint32_t *)out.mapped)[i] != cpu[i];
        conf_bad += ((uint32_t *)conf.mapped)[i] != cpu_conf[i];
        hard_bad += ((uint32_t *)hard.mapped)[i] != cpu_hard[i];
    }
    bake_buffer_destroy(&gpu, &hard); bake_buffer_destroy(&gpu, &conf);
    bake_buffer_destroy(&gpu, &out); bake_buffer_destroy(&gpu, &in);
    bake_gpu_destroy(&gpu);
    free(input); free(cpu); free(cpu_conf); free(cpu_hard);
    if (image_bad || conf_bad || hard_bad) {
        fprintf(stderr, "bake_clean_tests: FAIL image=%d confidence=%d hard=%d\n",
                image_bad, conf_bad, hard_bad);
        return 1;
    }
    printf("bake_clean_tests: OK (%dx%d, image+confidence+hard bit-exact)\n",
           width, height);
    return 0;
}
