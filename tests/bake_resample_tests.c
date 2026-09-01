#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "bake_gpu.h"
#include "bake_resample.h"

int main(void) {
    BakeGpu gpu;
    if (!bake_gpu_init(&gpu)) return 77;
    const int sw = 129, sh = 129, ow = 256, oh = 256;
    size_t source_count = (size_t)sw * sh, output_count = (size_t)ow * oh;
    float *source = malloc(source_count * sizeof(float));
    float *cpu = malloc(output_count * sizeof(float));
    for (int y = 0; y < sh; ++y) for (int x = 0; x < sw; ++x)
        source[y * sw + x] = 1040.25f + (float)(x * x + y * 17) * 0.03125f +
                             (float)((x * 13 + y * 7) & 31) * 0.00390625f;
    bake_resample_bilinear_cpu(source, cpu, sw, sh, ow, oh);
    BakeBuffer in = bake_buffer_host(&gpu, source_count * sizeof(float));
    BakeBuffer out = bake_buffer_host(&gpu, output_count * sizeof(float));
    for (size_t i = 0; i < source_count; ++i) ((float *)in.mapped)[i] = source[i];
    bake_resample_bilinear_gpu(&gpu, &in, &out, sw, sh, ow, oh);
    int bad = 0;
    for (size_t i = 0; i < output_count; ++i) bad += ((float *)out.mapped)[i] != cpu[i];
    bake_buffer_destroy(&gpu, &out); bake_buffer_destroy(&gpu, &in);
    bake_gpu_destroy(&gpu); free(source); free(cpu);
    if (bad) { fprintf(stderr, "bake_resample_tests: FAIL (%d values differ)\n", bad); return 1; }
    printf("bake_resample_tests: OK (129x129 -> 256x256 bit-exact)\n");
    return 0;
}
