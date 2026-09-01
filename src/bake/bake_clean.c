#include "bake_clean.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "bake_filter.h"
#include "bake_jfa.h"

static float channel(uint32_t p, unsigned shift) {
    return (float)((p >> shift) & 0xffu) / 255.0f;
}

static int inside(int x, int y, int width, int height) {
    return x >= 0 && x < width && y >= 0 && y < height;
}

static int any_within(const uint8_t *mask, int x, int y, int radius,
                      int width, int height) {
    for (int dy = -radius; dy <= radius; ++dy) {
        int room = radius - (dy < 0 ? -dy : dy);
        for (int dx = -room; dx <= room; ++dx) {
            int xx = x + dx, yy = y + dy;
            if (inside(xx, yy, width, height) && mask[yy * width + xx]) return 1;
        }
    }
    return 0;
}

void bake_clean_cpu(const uint32_t *input, uint32_t *cleaned,
                    uint32_t *confidence, uint32_t *hard_mask,
                    int width, int height) {
    int count = width * height;
    float *green = malloc((size_t)count * sizeof(float));
    float *green_blur = malloc((size_t)count * sizeof(float));
    float *blur_scratch = malloc((size_t)count * 3 * sizeof(float));
    float *weights10 = malloc((size_t)(2 * bake_gaussian_radius(10.0f, 4.0f) + 1) *
                              sizeof(float));
    uint8_t *tree = calloc((size_t)count, 1);
    uint8_t *base_hard = calloc((size_t)count, 1);
    uint8_t *repair = calloc((size_t)count, 1);
    uint8_t *mask8 = malloc((size_t)count);
    int32_t *owner = malloc((size_t)count * sizeof(int32_t));
    int32_t *jfa_scratch = malloc((size_t)count * sizeof(int32_t));
    float *filled = malloc((size_t)count * 3 * sizeof(float));
    float *soft = malloc((size_t)count * 3 * sizeof(float));
    float *weights2 = malloc((size_t)(2 * bake_gaussian_radius(2.0f, 4.0f) + 1) *
                             sizeof(float));

    for (int i = 0; i < count; ++i) green[i] = channel(input[i], 8) - channel(input[i], 0);
    int r10 = bake_gaussian_radius(10.0f, 4.0f);
    bake_gaussian_weights(10.0f, r10, weights10);
    bake_gaussian_cpu(green, green_blur, blur_scratch, width, height, 1,
                      weights10, r10);

    for (int i = 0; i < count; ++i) {
        float r = channel(input[i], 0), g = channel(input[i], 8);
        float b = channel(input[i], 16);
        float mx = fmaxf(r, fmaxf(g, b)), mn = fminf(r, fminf(g, b));
        float sat = (mx - mn) / fmaxf(mx, 1.0e-5f);
        float lum = r * 0.2126f + g * 0.7152f + b * 0.0722f;
        int green_ctx = green_blur[i] > 0.015f;
        tree[i] = (uint8_t)((g >= b - 0.02f) && (g > r - 0.07f) &&
                            (lum < 0.31f) && (g > 0.05f) && green_ctx);
        int road = (lum > 0.60f) && (sat < 0.13f) && green_ctx;
        int water = (b > r + 0.035f) && (b > g + 0.015f) && (lum < 0.52f);
        int magenta = (r > g + 0.11f) && (b > g + 0.07f) && (sat > 0.22f);
        base_hard[i] = (uint8_t)(road || water || magenta);
    }
    int any_repair = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int i = y * width + x;
            int hard = any_within(base_hard, x, y, 5, width, height);
            int fix = any_within(tree, x, y, 4, width, height) ||
                      any_within(base_hard, x, y, 9, width, height);
            hard_mask[i] = (uint32_t)hard;
            repair[i] = mask8[i] = (uint8_t)fix;
            any_repair |= fix;
        }
    }
    if (!any_repair) {
        memcpy(cleaned, input, (size_t)count * sizeof(uint32_t));
        for (int i = 0; i < count; ++i) confidence[i] = 255;
        goto done;
    }

    bake_jfa_cpu(mask8, owner, jfa_scratch, width, height);
    for (int i = 0; i < count; ++i) {
        uint32_t p = input[owner[i] >= 0 ? owner[i] : i];
        filled[i * 3 + 0] = (float)(p & 0xffu);
        filled[i * 3 + 1] = (float)((p >> 8) & 0xffu);
        filled[i * 3 + 2] = (float)((p >> 16) & 0xffu);
    }
    int r2 = bake_gaussian_radius(2.0f, 4.0f);
    bake_gaussian_weights(2.0f, r2, weights2);
    bake_gaussian_cpu(filled, soft, blur_scratch, width, height, 3, weights2, r2);
    for (int i = 0; i < count; ++i) {
        if (repair[i]) {
            uint32_t r = (uint32_t)fminf(fmaxf(soft[i * 3 + 0], 0.0f), 255.0f);
            uint32_t g = (uint32_t)fminf(fmaxf(soft[i * 3 + 1], 0.0f), 255.0f);
            uint32_t b = (uint32_t)fminf(fmaxf(soft[i * 3 + 2], 0.0f), 255.0f);
            cleaned[i] = r | (g << 8) | (b << 16) | (input[i] & 0xff000000u);
        } else {
            cleaned[i] = input[i];
        }
        confidence[i] = hard_mask[i] ? 32u : (repair[i] ? 128u : 255u);
    }

done:
    free(green); free(green_blur); free(blur_scratch); free(weights10);
    free(tree); free(base_hard); free(repair); free(mask8);
    free(owner); free(jfa_scratch); free(filled); free(soft); free(weights2);
}

void bake_clean_gpu(BakeGpu *gpu, const BakeBuffer *input,
                    const BakeBuffer *cleaned, const BakeBuffer *confidence,
                    const BakeBuffer *hard_mask, int width, int height) {
    size_t count = (size_t)width * height;
    BakeBuffer green = bake_buffer_host(gpu, count * sizeof(float));
    BakeBuffer green_blur = bake_buffer_host(gpu, count * sizeof(float));
    BakeBuffer scalar_scratch = bake_buffer_host(gpu, count * sizeof(float));
    BakeBuffer repair = bake_buffer_host(gpu, count * sizeof(uint32_t));
    BakeBuffer owner_a = bake_buffer_host(gpu, count * sizeof(int32_t));
    BakeBuffer owner_b = bake_buffer_host(gpu, count * sizeof(int32_t));
    BakeBuffer filled = bake_buffer_host(gpu, count * 3 * sizeof(float));
    BakeBuffer soft = bake_buffer_host(gpu, count * 3 * sizeof(float));
    BakeBuffer rgb_scratch = bake_buffer_host(gpu, count * 3 * sizeof(float));
    struct { uint32_t width, height; } size = {(uint32_t)width, (uint32_t)height};
    uint32_t gx = (uint32_t)(width + 7) / 8, gy = (uint32_t)(height + 7) / 8;

    BakePipeline green_pipe = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_green.comp.spv", 2, sizeof(size));
    BakeBuffer green_bind[2] = {*input, green};
    bake_dispatch(gpu, &green_pipe, green_bind, 2, &size, sizeof(size), gx, gy, 1);
    bake_pipeline_destroy(gpu, &green_pipe);
    bake_gaussian_gpu(gpu, &green, &green_blur, &scalar_scratch,
                      width, height, 1, 10.0f, 4.0f);

    BakePipeline masks_pipe = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_clean_masks.comp.spv", 4, sizeof(size));
    BakeBuffer mask_bind[4] = {*input, green_blur, *hard_mask, repair};
    bake_dispatch(gpu, &masks_pipe, mask_bind, 4, &size, sizeof(size), gx, gy, 1);
    bake_pipeline_destroy(gpu, &masks_pipe);

    BakePipeline init_pipe = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_jfa_init.comp.spv", 2, sizeof(size));
    BakeBuffer init_bind[2] = {repair, owner_a};
    bake_dispatch(gpu, &init_pipe, init_bind, 2, &size, sizeof(size), gx, gy, 1);
    bake_pipeline_destroy(gpu, &init_pipe);

    BakePipeline jfa = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_jfa.comp.spv", 2, 3 * sizeof(uint32_t));
    int longest = width > height ? width : height, start = 1;
    while (start < longest) start <<= 1;
    start >>= 1;
    BakeBuffer cur = owner_a, other = owner_b;
    for (int step = start; step >= 1; step >>= 1) {
        struct { uint32_t width, height; int32_t step; } push = {
            (uint32_t)width, (uint32_t)height, step};
        BakeBuffer pass[2] = {cur, other};
        bake_dispatch(gpu, &jfa, pass, 2, &push, sizeof(push), gx, gy, 1);
        BakeBuffer tmp = cur; cur = other; other = tmp;
    }
    bake_pipeline_destroy(gpu, &jfa);

    BakePipeline inpaint = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_inpaint.comp.spv", 3, sizeof(size));
    BakeBuffer inpaint_bind[3] = {*input, cur, filled};
    bake_dispatch(gpu, &inpaint, inpaint_bind, 3, &size, sizeof(size), gx, gy, 1);
    bake_pipeline_destroy(gpu, &inpaint);
    bake_gaussian_gpu(gpu, &filled, &soft, &rgb_scratch,
                      width, height, 3, 2.0f, 4.0f);

    BakePipeline finish = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_clean_finish.comp.spv", 6, sizeof(size));
    BakeBuffer finish_bind[6] = {*input, soft, repair, *hard_mask,
                                 *cleaned, *confidence};
    bake_dispatch(gpu, &finish, finish_bind, 6, &size, sizeof(size), gx, gy, 1);
    bake_pipeline_destroy(gpu, &finish);

    bake_buffer_destroy(gpu, &rgb_scratch); bake_buffer_destroy(gpu, &soft);
    bake_buffer_destroy(gpu, &filled); bake_buffer_destroy(gpu, &owner_b);
    bake_buffer_destroy(gpu, &owner_a); bake_buffer_destroy(gpu, &repair);
    bake_buffer_destroy(gpu, &scalar_scratch); bake_buffer_destroy(gpu, &green_blur);
    bake_buffer_destroy(gpu, &green);
}
