#include "bake_pass17_relayer.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "bake_filter.h"
#include "bake_random.h"
#include "bake_seam.h"

enum { MATERIALS = 3, CHANNELS = 3, EX_PATCH = 80, EX_BATCH = 5 };

static uint8_t byte_channel(uint32_t pixel, int channel) {
    return (uint8_t)(pixel >> (channel * 8));
}

static float pixel_luma(const float *rgb) {
    return (rgb[0] * .2126f + rgb[1] * .7152f + rgb[2] * .0722f) / 255.0f;
}

static bool filter_layers(BakeGpu *gpu, const float *input, float *output,
                          int width, int height, int channels, int layers,
                          float sigma) {
    const size_t values = (size_t)width * height * channels * layers;
    BakeBuffer source = bake_buffer_host(gpu, values * sizeof(float));
    BakeBuffer result = bake_buffer_host(gpu, values * sizeof(float));
    BakeBuffer scratch = bake_buffer_host(gpu, values * sizeof(float));
    if (!source.mapped || !result.mapped || !scratch.mapped) return false;
    memcpy(source.mapped, input, values * sizeof(float));
    bake_gaussian_batch_gpu(gpu, &source, &result, &scratch, width, height,
                            channels, layers, sigma, 4.0f, 0);
    memcpy(output, result.mapped, values * sizeof(float));
    bake_buffer_destroy(gpu, &scratch);
    bake_buffer_destroy(gpu, &result);
    bake_buffer_destroy(gpu, &source);
    return true;
}

static void rgb_seam_cost(const float *existing, const float *candidate,
                          int patch, float *cost) {
    for (int y = 0; y < patch; ++y) {
        const int ym = y > 0 ? y - 1 : y;
        const int yp = y + 1 < patch ? y + 1 : y;
        const float yscale = y > 0 && y + 1 < patch ? .5f : 1.0f;
        for (int x = 0; x < patch; ++x) {
            const int xm = x > 0 ? x - 1 : x;
            const int xp = x + 1 < patch ? x + 1 : x;
            const float xscale = x > 0 && x + 1 < patch ? .5f : 1.0f;
            const size_t p = ((size_t)y * patch + x) * 3;
            float colour = 0.0f;
            for (int c = 0; c < 3; ++c) {
                const float delta = (existing[p + c] - candidate[p + c]) / 255.0f;
                colour += delta * delta;
            }
            colour /= 3.0f;
            const float egx = (pixel_luma(existing + ((size_t)y * patch + xp) * 3) -
                               pixel_luma(existing + ((size_t)y * patch + xm) * 3)) *
                              xscale;
            const float egy = (pixel_luma(existing + ((size_t)yp * patch + x) * 3) -
                               pixel_luma(existing + ((size_t)ym * patch + x) * 3)) *
                              yscale;
            const float cgx = (pixel_luma(candidate + ((size_t)y * patch + xp) * 3) -
                               pixel_luma(candidate + ((size_t)y * patch + xm) * 3)) *
                              xscale;
            const float cgy = (pixel_luma(candidate + ((size_t)yp * patch + x) * 3) -
                               pixel_luma(candidate + ((size_t)ym * patch + x) * 3)) *
                              yscale;
            const float dx = egx - cgx, dy = egy - cgy;
            cost[(size_t)y * patch + x] = colour + .5f * (dx * dx + dy * dy);
        }
    }
}

static bool build_material_exemplar(
    BakeGpu *gpu, const BakeTerrainSample *sources, int source_count, int kind,
    BakePcg64 *rng, int width, int height, float *exemplar) {
    int usable[16], usable_count = 0;
    for (int source = 0; source < source_count; ++source) {
        size_t matching = 0;
        const size_t pixels = (size_t)sources[source].width * sources[source].height;
        for (size_t p = 0; p < pixels; ++p)
            matching += sources[source].labels[p] == (uint32_t)kind;
        if ((double)matching / pixels >= .5) usable[usable_count++] = source;
    }
    if (!usable_count) return false;
    const size_t output_pixels = (size_t)width * height;
    memset(exemplar, 0, output_pixels * 3 * sizeof(float));
    uint8_t *filled = calloc(output_pixels, 1);
    const size_t patch_values = (size_t)EX_PATCH * EX_PATCH * 3;
    float *crops = malloc(EX_BATCH * patch_values * sizeof(float));
    float *smooth_2 = malloc(EX_BATCH * patch_values * sizeof(float));
    float *smooth_12 = malloc(EX_BATCH * patch_values * sizeof(float));
    float *cost = malloc((size_t)EX_PATCH * EX_PATCH * sizeof(float));
    uint8_t *known = malloc((size_t)EX_PATCH * EX_PATCH);
    uint8_t *take = malloc((size_t)EX_PATCH * EX_PATCH);
    if (!filled || !crops || !smooth_2 || !smooth_12 || !cost || !known || !take)
        return false;
    int origins[8], origin_count = 0;
    for (int value = 0; value <= width - EX_PATCH; value += EX_PATCH - 20)
        origins[origin_count++] = value;
    if (!origin_count || origins[origin_count - 1] != width - EX_PATCH)
        origins[origin_count++] = width - EX_PATCH;
    for (int yi = 0; yi < origin_count; ++yi) {
        for (int xi = 0; xi < origin_count; ++xi) {
            int sampled_source[EX_BATCH], sampled_x[EX_BATCH], sampled_y[EX_BATCH];
            float purity[EX_BATCH];
            for (int candidate = 0; candidate < EX_BATCH; ++candidate) {
                const int source = usable[bake_pcg64_bounded(
                    rng, 0, (uint32_t)usable_count)];
                const int sy = (int)bake_pcg64_bounded(
                    rng, 0, (uint32_t)(sources[source].height - EX_PATCH + 1));
                const int sx = (int)bake_pcg64_bounded(
                    rng, 0, (uint32_t)(sources[source].width - EX_PATCH + 1));
                sampled_source[candidate] = source;
                sampled_x[candidate] = sx;
                sampled_y[candidate] = sy;
                size_t matching = 0;
                float *crop = crops + (size_t)candidate * patch_values;
                for (int y = 0; y < EX_PATCH; ++y) {
                    for (int x = 0; x < EX_PATCH; ++x) {
                        const size_t source_pixel =
                            (size_t)(sy + y) * sources[source].width + sx + x;
                        matching += sources[source].labels[source_pixel] ==
                                    (uint32_t)kind;
                        const uint32_t pixel = sources[source].rgb[source_pixel];
                        const size_t target = ((size_t)y * EX_PATCH + x) * 3;
                        for (int c = 0; c < 3; ++c)
                            crop[target + c] = byte_channel(pixel, c);
                    }
                }
                purity[candidate] = (float)((double)matching /
                                            (EX_PATCH * EX_PATCH));
            }
            if (!filter_layers(gpu, crops, smooth_2, EX_PATCH, EX_PATCH, 3,
                               EX_BATCH, 2.0f) ||
                !filter_layers(gpu, crops, smooth_12, EX_PATCH, EX_PATCH, 3,
                               EX_BATCH, 12.0f))
                return false;
            int fallback = 0, best = -1;
            float fallback_purity = -1.0f, best_energy = -1.0f;
            for (int candidate = 0; candidate < EX_BATCH; ++candidate) {
                if (purity[candidate] > fallback_purity) {
                    fallback_purity = purity[candidate];
                    fallback = candidate;
                }
                if (purity[candidate] < .7f) continue;
                float sum = 0.0f;
                const size_t base = (size_t)candidate * patch_values;
                for (size_t i = 0; i < patch_values; ++i) {
                    const float delta = smooth_2[base + i] - smooth_12[base + i];
                    sum += delta * delta;
                }
                const float energy = sqrtf(sum / (float)patch_values);
                if (energy > best_energy) {
                    best_energy = energy;
                    best = candidate;
                }
            }
            if (best < 0) best = fallback;
            (void)sampled_source;
            (void)sampled_x;
            (void)sampled_y;
            float existing[EX_PATCH * EX_PATCH * 3];
            for (int y = 0; y < EX_PATCH; ++y) {
                for (int x = 0; x < EX_PATCH; ++x) {
                    const size_t global = (size_t)(origins[yi] + y) * width +
                                          origins[xi] + x;
                    const size_t local = (size_t)y * EX_PATCH + x;
                    memcpy(existing + local * 3, exemplar + global * 3,
                           3 * sizeof(float));
                    known[local] = filled[global];
                }
            }
            const float *candidate = crops + (size_t)best * patch_values;
            rgb_seam_cost(existing, candidate, EX_PATCH, cost);
            if (!bake_quilt_take_mask_from_cost(
                    cost, known, EX_PATCH, 20, origins[xi] > 0,
                    origins[yi] > 0, take))
                return false;
            for (int y = 0; y < EX_PATCH; ++y) {
                for (int x = 0; x < EX_PATCH; ++x) {
                    const size_t global = (size_t)(origins[yi] + y) * width +
                                          origins[xi] + x;
                    const size_t local = (size_t)y * EX_PATCH + x;
                    if (take[local])
                        memcpy(exemplar + global * 3, candidate + local * 3,
                               3 * sizeof(float));
                    filled[global] = 1;
                }
            }
        }
    }
    free(take);
    free(known);
    free(cost);
    free(smooth_12);
    free(smooth_2);
    free(crops);
    free(filled);
    return true;
}

static int float_compare(const void *left, const void *right) {
    const float a = *(const float *)left, b = *(const float *)right;
    return (a > b) - (a < b);
}

static float median(float *values, int count) {
    qsort(values, (size_t)count, sizeof(*values), float_compare);
    return count & 1 ? values[count / 2]
                     : .5f * (values[count / 2 - 1] + values[count / 2]);
}

static float rms_difference(const float *first, const float *second,
                            size_t count) {
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        const float delta = first[i] - second[i];
        sum += delta * delta;
    }
    return sqrtf(sum / (float)count);
}

bool bake_pass17_material_relayer_gpu(
    BakeGpu *gpu, const uint32_t *base, const uint32_t *labels,
    const BakeTerrainSample *texture_donors, int texture_donor_count,
    const BakeTerrainSample *material_sources[3],
    const int material_source_counts[3], uint32_t *output,
    int width, int height) {
    if (!gpu || !base || !labels || !texture_donors || texture_donor_count <= 0 ||
        !output || width != 256 || height != 256)
        return false;
    const size_t pixels = (size_t)width * height;
    const size_t values = pixels * 3;
    const size_t donor_values = (size_t)texture_donor_count * values;
    float *donor_rgb = malloc(donor_values * sizeof(float));
    float *donor_2 = malloc(donor_values * sizeof(float));
    float *donor_12 = malloc(donor_values * sizeof(float));
    float *donor_32 = malloc(donor_values * sizeof(float));
    if (!donor_rgb || !donor_2 || !donor_12 || !donor_32) return false;
    for (int donor = 0; donor < texture_donor_count; ++donor) {
        for (size_t p = 0; p < pixels; ++p) {
            for (int c = 0; c < 3; ++c)
                donor_rgb[((size_t)donor * pixels + p) * 3 + c] =
                    byte_channel(texture_donors[donor].rgb[p], c);
        }
    }
    if (!filter_layers(gpu, donor_rgb, donor_2, width, height, 3,
                       texture_donor_count, 2.0f) ||
        !filter_layers(gpu, donor_rgb, donor_12, width, height, 3,
                       texture_donor_count, 12.0f) ||
        !filter_layers(gpu, donor_rgb, donor_32, width, height, 3,
                       texture_donor_count, 32.0f))
        return false;
    float mid_energy[64], meso_energy[64], fine_energy[64];
    for (int donor = 0; donor < texture_donor_count; ++donor) {
        const size_t offset = (size_t)donor * values;
        mid_energy[donor] = rms_difference(donor_2 + offset,
                                           donor_12 + offset, values);
        meso_energy[donor] = rms_difference(donor_12 + offset,
                                            donor_32 + offset, values);
        fine_energy[donor] = rms_difference(donor_rgb + offset,
                                            donor_2 + offset, values);
    }
    float *mid = calloc(values, sizeof(float));
    float *meso = calloc(values, sizeof(float));
    float *fine = calloc(values, sizeof(float));
    float *exemplars = malloc(MATERIALS * values * sizeof(float));
    float *exemplar_2 = malloc(MATERIALS * values * sizeof(float));
    float *exemplar_12 = malloc(MATERIALS * values * sizeof(float));
    float *exemplar_32 = malloc(MATERIALS * values * sizeof(float));
    float *masks = malloc(MATERIALS * pixels * sizeof(float));
    float *mask_2 = malloc(MATERIALS * pixels * sizeof(float));
    float *mask_3 = malloc(MATERIALS * pixels * sizeof(float));
    if (!mid || !meso || !fine || !exemplars || !exemplar_2 || !exemplar_12 ||
        !exemplar_32 || !masks || !mask_2 || !mask_3)
        return false;
    BakePcg64 rng;
    if (!bake_pcg64_pass17_seed(&rng, 1515)) return false;
    for (int kind = 0; kind < MATERIALS; ++kind) {
        if (!build_material_exemplar(
                gpu, material_sources[kind], material_source_counts[kind],
                kind, &rng, width, height, exemplars + (size_t)kind * values))
            return false;
        for (size_t p = 0; p < pixels; ++p)
            masks[(size_t)kind * pixels + p] = labels[p] == (uint32_t)kind;
    }
    if (!filter_layers(gpu, exemplars, exemplar_2, width, height, 3,
                       MATERIALS, 2.0f) ||
        !filter_layers(gpu, exemplars, exemplar_12, width, height, 3,
                       MATERIALS, 12.0f) ||
        !filter_layers(gpu, exemplars, exemplar_32, width, height, 3,
                       MATERIALS, 32.0f) ||
        !filter_layers(gpu, masks, mask_2, width, height, 1,
                       MATERIALS, 2.0f) ||
        !filter_layers(gpu, masks, mask_3, width, height, 1,
                       MATERIALS, 3.0f))
        return false;
    float material_colours[3][3];
    for (int kind = 0; kind < MATERIALS; ++kind) {
        const float *exemplar = exemplars + (size_t)kind * values;
        double colour_sum[3] = {0};
        for (size_t p = 0; p < pixels; ++p)
            for (int c = 0; c < 3; ++c) colour_sum[c] += exemplar[p * 3 + c];
        const float maximum = fmaxf((float)colour_sum[0],
                                    fmaxf((float)colour_sum[1],
                                          (float)colour_sum[2]));
        const float minimum = fminf((float)colour_sum[0],
                                    fminf((float)colour_sum[1],
                                          (float)colour_sum[2]));
        const float keep = fminf(fmaxf(((maximum - minimum) /
                                        fmaxf(maximum, 1.0e-5f) - .05f) * 3.5f,
                                       .10f),
                                 1.0f);
        for (size_t p = 0; p < pixels; ++p) {
            float bands[3][3];
            for (int c = 0; c < 3; ++c) {
                const size_t q = ((size_t)kind * pixels + p) * 3 + c;
                bands[0][c] = exemplar_2[q] - exemplar_12[q];
                bands[1][c] = exemplar_12[q] - exemplar_32[q];
                bands[2][c] = exemplars[q] - exemplar_2[q];
            }
            for (int band = 0; band < 3; ++band) {
                const float mean = (bands[band][0] + bands[band][1] +
                                    bands[band][2]) /
                                   3.0f;
                for (int c = 0; c < 3; ++c)
                    bands[band][c] = mean + keep * (bands[band][c] - mean);
            }
            const float weight = mask_2[(size_t)kind * pixels + p];
            for (int c = 0; c < 3; ++c) {
                mid[p * 3 + c] += weight * bands[0][c];
                meso[p * 3 + c] += weight * bands[1][c];
                fine[p * 3 + c] += weight * bands[2][c];
            }
        }
        for (int c = 0; c < 3; ++c) {
            float means[64];
            int count = 0;
            for (int donor = 0; donor < texture_donor_count; ++donor) {
                double sum = 0.0;
                int matching = 0;
                for (size_t p = 0; p < pixels; ++p) {
                    if (texture_donors[donor].labels[p] != (uint32_t)kind) continue;
                    sum += byte_channel(texture_donors[donor].rgb[p], c);
                    ++matching;
                }
                if (matching >= 256) means[count++] = (float)(sum / matching);
            }
            if (!count) {
                for (int donor = 0; donor < texture_donor_count; ++donor) {
                    double sum = 0.0;
                    for (size_t p = 0; p < pixels; ++p)
                        sum += byte_channel(texture_donors[donor].rgb[p], c);
                    means[count++] = (float)(sum / pixels);
                }
            }
            material_colours[kind][c] = median(means, count);
        }
    }
    float *base_f = malloc(values * sizeof(float));
    float *base_2 = malloc(values * sizeof(float));
    float *base_12 = malloc(values * sizeof(float));
    float *base_32 = malloc(values * sizeof(float));
    float *colour_grade = calloc(values, sizeof(float));
    if (!base_f || !base_2 || !base_12 || !base_32 || !colour_grade) return false;
    for (size_t p = 0; p < pixels; ++p)
        for (int c = 0; c < 3; ++c)
            base_f[p * 3 + c] = byte_channel(base[p], c);
    if (!filter_layers(gpu, base_f, base_2, width, height, 3, 1, 2.0f) ||
        !filter_layers(gpu, base_f, base_12, width, height, 3, 1, 12.0f) ||
        !filter_layers(gpu, base_f, base_32, width, height, 3, 1, 32.0f))
        return false;
    for (int kind = 0; kind < MATERIALS; ++kind) {
        double current[3] = {0};
        int count = 0;
        for (size_t p = 0; p < pixels; ++p) {
            if (labels[p] != (uint32_t)kind) continue;
            for (int c = 0; c < 3; ++c) current[c] += base_12[p * 3 + c];
            ++count;
        }
        for (int c = 0; c < 3; ++c) {
            const float mean = count ? (float)(current[c] / count)
                                     : material_colours[kind][c];
            const float shift = fminf(fmaxf(material_colours[kind][c] - mean,
                                            -48.0f),
                                      48.0f);
            for (size_t p = 0; p < pixels; ++p)
                colour_grade[p * 3 + c] +=
                    mask_3[(size_t)kind * pixels + p] * shift;
        }
    }
    float donor_mid[64], donor_meso[64], donor_fine[64];
    memcpy(donor_mid, mid_energy, (size_t)texture_donor_count * sizeof(float));
    memcpy(donor_meso, meso_energy, (size_t)texture_donor_count * sizeof(float));
    memcpy(donor_fine, fine_energy, (size_t)texture_donor_count * sizeof(float));
    const float base_mid = rms_difference(base_2, base_12, values);
    const float base_meso = rms_difference(base_12, base_32, values);
    const float base_fine = rms_difference(base_f, base_2, values);
    const float mid_topup = fminf(fmaxf(median(donor_mid, texture_donor_count) /
                                            fmaxf(base_mid, 1.0e-5f) -
                                        1.0f,
                                    0.0f),
                              1.5f);
    const float meso_topup = fminf(fmaxf(median(donor_meso, texture_donor_count) /
                                             fmaxf(base_meso, 1.0e-5f) -
                                         1.0f,
                                     0.0f),
                               1.5f);
    const float fine_topup = fminf(fmaxf(median(donor_fine, texture_donor_count) /
                                             fmaxf(base_fine, 1.0e-5f) -
                                         1.0f,
                                     0.0f),
                               1.5f);
    for (size_t p = 0; p < pixels; ++p) {
        uint32_t packed = 0xff000000u;
        for (int c = 0; c < 3; ++c) {
            float value = base_f[p * 3 + c] + .80f * colour_grade[p * 3 + c] +
                          mid_topup * mid[p * 3 + c] +
                          meso_topup * meso[p * 3 + c] +
                          fine_topup * fine[p * 3 + c];
            if (value < 0.0f) value = 0.0f;
            if (value > 255.0f) value = 255.0f;
            packed |= (uint32_t)(uint8_t)value << (c * 8);
        }
        output[p] = packed;
    }
    free(colour_grade);
    free(base_32);
    free(base_12);
    free(base_2);
    free(base_f);
    free(mask_3);
    free(mask_2);
    free(masks);
    free(exemplar_32);
    free(exemplar_12);
    free(exemplar_2);
    free(exemplars);
    free(fine);
    free(meso);
    free(mid);
    free(donor_32);
    free(donor_12);
    free(donor_2);
    free(donor_rgb);
    return true;
}
