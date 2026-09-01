// terrain_bake: headless Vulkan compute entry point for the offline baker.
//
// Step 1 only brings up the device and exercises the harness end to end; the
// terrain-synthesis compute stages (preprocess -> exemplar -> quilt -> relayer)
// land on top of this same harness, each behind its own unit test.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "bake_gpu.h"

#ifndef BAKE_SHADER_DIR
#error "BAKE_SHADER_DIR must be defined"
#endif

int main(void) {
    BakeGpu gpu;
    if (!bake_gpu_init(&gpu)) {
        fprintf(stderr, "terrain_bake: no compute-capable Vulkan device\n");
        return 1;
    }
    printf("terrain_bake: device = %s (compute queue family %u)\n",
           gpu.device_name, gpu.queue_family);

    // Harness smoke check: scale a small buffer and confirm the round trip.
    const uint32_t count = 256;
    BakeBuffer in = bake_buffer_host(&gpu, count * sizeof(uint32_t));
    BakeBuffer out = bake_buffer_host(&gpu, count * sizeof(uint32_t));
    uint32_t *in_data = in.mapped, *out_data = out.mapped;
    for (uint32_t i = 0; i < count; ++i) in_data[i] = i;
    BakePipeline pipeline = bake_pipeline_create(
        &gpu, BAKE_SHADER_DIR "/bake_scale.comp.spv", 2, 2 * sizeof(uint32_t));
    struct { uint32_t count; uint32_t factor; } push = {count, 2};
    BakeBuffer buffers[2] = {in, out};
    bake_dispatch(&gpu, &pipeline, buffers, 2, &push, sizeof(push),
                  (count + 63) / 64, 1, 1);
    int ok = (out_data[10] == 20) && (out_data[count - 1] == (count - 1) * 2);
    printf("terrain_bake: harness round trip %s\n", ok ? "OK" : "FAILED");

    bake_pipeline_destroy(&gpu, &pipeline);
    bake_buffer_destroy(&gpu, &out);
    bake_buffer_destroy(&gpu, &in);
    bake_gpu_destroy(&gpu);
    return ok ? 0 : 1;
}
