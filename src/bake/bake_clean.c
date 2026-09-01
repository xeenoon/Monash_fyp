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

// scipy.ndimage generates its Gaussian coefficients in float64 and performs
// each 1-D correlation in float64 before casting the result back to the
// float32 output array.  Cleaning is a one-time source preparation step, so
// matching that behavior here avoids changing material labels on repaired
// boundary pixels without taking any synthesis work off the GPU.
static void scipy_gaussian_reflect(const float *input, float *output,
                                   float *scratch, int width, int height,
                                   int channels, double sigma) {
    const int radius = (int)(4.0 * sigma + 0.5);
    const int taps = 2 * radius + 1;
    double *weights = malloc((size_t)taps * sizeof(*weights));
    double sum = 0.0;
    for (int k = -radius; k <= radius; ++k) {
        const double value = exp(-0.5 * (double)(k * k) / (sigma * sigma));
        weights[k + radius] = value;
        sum += value;
    }
    for (int k = 0; k < taps; ++k) weights[k] /= sum;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                double value = 0.0;
                for (int k = -radius; k <= radius; ++k) {
                    int xx = x + k;
                    while (xx < 0 || xx >= width)
                        xx = xx < 0 ? -xx - 1 : 2 * width - xx - 1;
                    value += weights[k + radius] *
                             input[((size_t)y * width + xx) * channels + c];
                }
                scratch[((size_t)y * width + x) * channels + c] = (float)value;
            }
        }
    }
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                double value = 0.0;
                for (int k = -radius; k <= radius; ++k) {
                    int yy = y + k;
                    while (yy < 0 || yy >= height)
                        yy = yy < 0 ? -yy - 1 : 2 * height - yy - 1;
                    value += weights[k + radius] *
                             scratch[((size_t)yy * width + x) * channels + c];
                }
                output[((size_t)y * width + x) * channels + c] = (float)value;
            }
        }
    }
    free(weights);
}

void bake_clean_cpu(const uint32_t *input, uint32_t *cleaned,
                    uint32_t *confidence, uint32_t *hard_mask,
                    int width, int height) {
    int count = width * height;
    float *green = malloc((size_t)count * sizeof(float));
    float *green_blur = malloc((size_t)count * sizeof(float));
    float *blur_scratch = malloc((size_t)count * 3 * sizeof(float));
    uint8_t *tree = calloc((size_t)count, 1);
    uint8_t *base_hard = calloc((size_t)count, 1);
    uint8_t *repair = calloc((size_t)count, 1);
    uint8_t *mask8 = malloc((size_t)count);
    int32_t *owner = malloc((size_t)count * sizeof(int32_t));
    float *filled = malloc((size_t)count * 3 * sizeof(float));
    float *soft = malloc((size_t)count * 3 * sizeof(float));

    for (int i = 0; i < count; ++i) green[i] = channel(input[i], 8) - channel(input[i], 0);
    scipy_gaussian_reflect(green, green_blur, blur_scratch, width, height, 1,
                           10.0);

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

    bake_edt_cpu(mask8, owner, width, height);
    for (int i = 0; i < count; ++i) {
        uint32_t p = input[owner[i] >= 0 ? owner[i] : i];
        filled[i * 3 + 0] = (float)(p & 0xffu);
        filled[i * 3 + 1] = (float)((p >> 8) & 0xffu);
        filled[i * 3 + 2] = (float)((p >> 16) & 0xffu);
    }
    scipy_gaussian_reflect(filled, soft, blur_scratch, width, height, 3, 2.0);
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
    free(green); free(green_blur); free(blur_scratch);
    free(tree); free(base_hard); free(repair); free(mask8);
    free(owner); free(filled); free(soft);
}

void bake_clean_gpu(BakeGpu *gpu, const BakeBuffer *input,
                    const BakeBuffer *cleaned, const BakeBuffer *confidence,
                    const BakeBuffer *hard_mask, int width, int height) {
    size_t count = (size_t)width * height;
    BakeBuffer green = bake_buffer_host(gpu, count * sizeof(float));
    BakeBuffer green_blur = bake_buffer_host(gpu, count * sizeof(float));
    BakeBuffer scalar_scratch = bake_buffer_host(gpu, count * sizeof(float));
    BakeBuffer repair = bake_buffer_host(gpu, count * sizeof(uint32_t));
    BakeBuffer owner = bake_buffer_host(gpu, count * sizeof(int32_t));
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
    bake_gaussian_gpu_reflect(gpu, &green, &green_blur, &scalar_scratch,
                              width, height, 1, 10.0f, 4.0f);

    BakePipeline masks_pipe = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_clean_masks.comp.spv", 4, sizeof(size));
    BakeBuffer mask_bind[4] = {*input, green_blur, *hard_mask, repair};
    bake_dispatch(gpu, &masks_pipe, mask_bind, 4, &size, sizeof(size), gx, gy, 1);
    bake_pipeline_destroy(gpu, &masks_pipe);

    bake_edt_gpu(gpu, &repair, &owner, width, height);

    BakePipeline inpaint = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_inpaint.comp.spv", 3, sizeof(size));
    BakeBuffer inpaint_bind[3] = {*input, owner, filled};
    bake_dispatch(gpu, &inpaint, inpaint_bind, 3, &size, sizeof(size), gx, gy, 1);
    bake_pipeline_destroy(gpu, &inpaint);
    bake_gaussian_gpu_reflect(gpu, &filled, &soft, &rgb_scratch,
                              width, height, 3, 2.0f, 4.0f);

    BakePipeline finish = bake_pipeline_create(
        gpu, BAKE_SHADER_DIR "/bake_clean_finish.comp.spv", 6, sizeof(size));
    BakeBuffer finish_bind[6] = {*input, soft, repair, *hard_mask,
                                 *cleaned, *confidence};
    bake_dispatch(gpu, &finish, finish_bind, 6, &size, sizeof(size), gx, gy, 1);
    bake_pipeline_destroy(gpu, &finish);

    bake_buffer_destroy(gpu, &rgb_scratch); bake_buffer_destroy(gpu, &soft);
    bake_buffer_destroy(gpu, &filled); bake_buffer_destroy(gpu, &owner);
    bake_buffer_destroy(gpu, &repair);
    bake_buffer_destroy(gpu, &scalar_scratch); bake_buffer_destroy(gpu, &green_blur);
    bake_buffer_destroy(gpu, &green);
}
