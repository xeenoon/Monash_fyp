// Step-1 unit test for the headless Vulkan compute harness.
//
// Proves the full round trip on a real device: fill a host buffer, dispatch the
// bake_scale kernel, read the result back, and assert an exact integer match.
// If no compute device is available the test skips (exit 77, the CTest SKIP
// convention) rather than failing, so the suite stays green on GPU-less CI.
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
        fprintf(stderr, "bake_gpu_tests: no compute device; skipping\n");
        return 77;  // CTest SKIP
    }
    printf("bake device: %s\n", gpu.device_name);

    const uint32_t count = 10007;   // deliberately not a multiple of 64
    const uint32_t factor = 3;
    BakeBuffer in = bake_buffer_host(&gpu, (VkDeviceSize)count * sizeof(uint32_t));
    BakeBuffer out = bake_buffer_host(&gpu, (VkDeviceSize)count * sizeof(uint32_t));

    uint32_t *in_data = in.mapped;
    uint32_t *out_data = out.mapped;
    for (uint32_t i = 0; i < count; ++i) {
        in_data[i] = i;
        out_data[i] = 0xDEADBEEFu;  // poison, so we know the kernel wrote every slot
    }

    BakePipeline pipeline = bake_pipeline_create(
        &gpu, BAKE_SHADER_DIR "/bake_scale.comp.spv", 2, 2 * sizeof(uint32_t));

    struct { uint32_t count; uint32_t factor; } push = {count, factor};
    BakeBuffer buffers[2] = {in, out};
    bake_dispatch(&gpu, &pipeline, buffers, 2, &push, sizeof(push),
                  (count + 63) / 64, 1, 1);

    int failures = 0;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t expected = i * factor;
        if (out_data[i] != expected) {
            if (failures < 5) {
                fprintf(stderr, "  mismatch at %u: got %u expected %u\n", i,
                        out_data[i], expected);
            }
            ++failures;
        }
    }

    bake_pipeline_destroy(&gpu, &pipeline);
    bake_buffer_destroy(&gpu, &out);
    bake_buffer_destroy(&gpu, &in);
    bake_gpu_destroy(&gpu);

    if (failures) {
        fprintf(stderr, "bake_gpu_tests: FAIL (%d mismatches)\n", failures);
        return 1;
    }
    printf("bake_gpu_tests: OK (%u elements scaled x%u)\n", count, factor);
    return 0;
}
